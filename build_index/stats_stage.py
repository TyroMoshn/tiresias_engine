# build_index/stats_stage.py
from __future__ import annotations
import io
import math
from collections import Counter, defaultdict
from typing import Dict, Tuple, List
import struct
import numpy as np
import polars as pl
from tqdm import tqdm

from .config import Config
from .utils import newer_than, ensure_dir, log, encode_varint_delta

import scipy.sparse as sp

def step_tag_stats(cfg: Config) -> None:
    dict_src = cfg.root / "tags_dict.parquet"
    out = cfg.tags_parquet
    if out.exists() and not cfg.force and newer_than(out, dict_src, cfg.posts_parquet / "_SUCCESS"):
        log("[stats] already fresh - skip")
        return

    log("[stats] calculate idf(pop_count-smooth.) and popularity...")
    posts_lf = pl.scan_parquet(f"{cfg.posts_parquet.as_posix()}/**/*.parquet")
    if cfg.reliable_only:
        posts_lf = posts_lf.filter(~pl.col("is_deleted") & ~pl.col("is_pending"))
    posts = posts_lf.select(["id", "score", "fav_count"]).collect(engine="streaming")
    n_posts = posts.height

    tag_dict = pl.read_parquet(dict_src)

    long = pl.scan_parquet(f"{cfg.post_tags_parquet.as_posix()}/**/*.parquet")
    joined = long.join(posts.lazy(), left_on="post_id", right_on="id", how="inner")
    pop = joined.group_by("tag_id").agg([
        pl.col("score").mean().alias("avg_score"),
        pl.col("fav_count").mean().alias("avg_fav"),
        pl.len().alias("df_local"),
    ]).collect()

    tag_stats = tag_dict.join(pop, on="tag_id", how="left").with_columns(
        pl.col("df_local").fill_null(0)
    )

    alpha = float(cfg.idf_alpha)

    def _compute_corr_for_auto(tbl: pl.DataFrame) -> float:
        top = tbl.sort("post_count", descending=True).head(max(1, int(tbl.height * 0.01)))
        a = top.get_column("post_count").fill_null(0).to_numpy()
        b = top.get_column("df_local").fill_null(0).to_numpy()
        if a.size > 1 and b.size > 1 and a.sum() > 0 and b.sum() > 0:
            return float(np.corrcoef(np.log1p(a), np.log1p(b))[0, 1])
        return 1.0

    use_local = (cfg.idf_source == "local")
    if cfg.idf_source == "auto":
        corr = _compute_corr_for_auto(tag_stats)
        log(f"[stats] corr(tags.post_count vs df_local)={corr:.3f} ; threshold={cfg.idf_auto_switch_threshold:.2f}")
        use_local = (corr < cfg.idf_auto_switch_threshold)
        log(f"[stats] IDF source => {'LOCAL (df_local)' if use_local else 'TAGS_CSV (post_count)'}")

    denom = (pl.col("df_local") if use_local else pl.col("post_count")).cast(pl.Float64)
    tag_stats = tag_stats.with_columns(
        (((pl.lit(n_posts, dtype=pl.Float64) + alpha) / (denom + alpha)).log()).alias("idf")
    )

    tag_stats.write_parquet(out, compression="zstd")

    try:
        top = tag_stats.sort("post_count", descending=True).head(max(1, int(tag_stats.height * 0.01)))
        a = top.get_column("post_count").fill_null(0).to_numpy()
        b = top.get_column("df_local").fill_null(0).to_numpy()
        if a.size > 1 and b.size > 1 and a.sum() > 0 and b.sum() > 0:
            corr = float(np.corrcoef(np.log1p(a), np.log1p(b))[0, 1])
            log(f"[stats] corr(log post_count, log df_local) ≈ {corr:.4f}")
    except Exception as e:
        log(f"[stats] validation failed: {e}")

    log("[stats] done")

def step_pmi(cfg: Config) -> None:
    out = cfg.root / "tag_pmi.parquet"
    if out.exists() and not cfg.force and newer_than(out, cfg.post_tags_parquet / "_SUCCESS"):
        log("[pmi] already fresh - skip")
        return

    log("[pmi] calculate co-occurrence")
    long = pl.scan_parquet(f"{cfg.post_tags_parquet.as_posix()}/**/*.parquet")
    tags_meta = pl.read_parquet(cfg.root / "tags_dict.parquet").select([pl.col("tag_id"), pl.col("post_count")])

    ln = long.join(tags_meta.lazy(), on="tag_id", how="left")
    top_m = (ln.group_by("post_id")
               .agg(pl.col("tag_id").sort_by(pl.col("post_count"), descending=False).head(cfg.pmi_top_m_per_post))
               .collect(engine="streaming"))

    if top_m.is_empty():
        pl.DataFrame({"a": [], "b": [], "support": [], "pmi": []}).write_parquet(out, compression="zstd")
        log("[pmi] empty - saved empty parquet")
        return

    uniq_tags = (top_m
        .select(pl.col("tag_id").explode().alias("tag_id"))
        .unique()
        .with_row_count("tid"))  # tid: 0..T-1

    long_rows = (top_m
        .with_row_count("row_id")
        .select(["row_id", "tag_id"])
        .explode("tag_id")
        .drop_nulls()
        .join(uniq_tags, on="tag_id", how="inner")   # -> row_id, tag_id, tid
        .select(["row_id", "tid"]))

    N = int(top_m.height)
    T = int(uniq_tags.height)
    r = long_rows.get_column("row_id").to_numpy().astype(np.int64, copy=False)
    c = long_rows.get_column("tid").to_numpy().astype(np.int32, copy=False)
    data = np.ones_like(r, dtype=np.uint8)
    X = sp.csr_matrix((data, (r, c)), shape=(N, T), dtype=np.uint8)

    tag_cnt = np.asarray(X.sum(axis=0)).ravel().astype(np.int64)  # shape (T,)

    C = (X.T @ X).tocsr()
    C.setdiag(0)
    C.eliminate_zeros()

    coo = C.tocoo()
    mask = coo.row < coo.col
    if int(cfg.pmi_support) > 1:
        mask &= (coo.data >= int(cfg.pmi_support))

    ai = coo.row[mask]
    bi = coo.col[mask]
    supp = coo.data[mask].astype(np.int32, copy=False)

    if ai.size == 0:
        pl.DataFrame({"a": [], "b": [], "support": [], "pmi": []}).write_parquet(out, compression="zstd")
        log("[pmi] There are no pairs after the support threshold - the empty parquet was saved")
        return

    Nf = float(N)
    pa = tag_cnt[ai] / Nf
    pb = tag_cnt[bi] / Nf
    pab = supp.astype(np.float64) / Nf
    pmi = np.log((pab + 1e-12) / (pa * pb + 1e-12)).astype(np.float32, copy=False)

    tid2tag = uniq_tags.sort("tid").get_column("tag_id").to_numpy()
    a_ids = tid2tag[ai].astype(np.int32, copy=False)
    b_ids = tid2tag[bi].astype(np.int32, copy=False)

    pl.DataFrame(
        {"a": a_ids, "b": b_ids, "support": supp, "pmi": pmi}
    ).write_parquet(out, compression="zstd")

    log(f"[pmi] ready: {ai.size:,} pairs (N={N:,}, T={T:,}, nnz={C.nnz:,})")

def step_topk(cfg: Config) -> None:
    from collections import defaultdict
    import numpy as np
    ensure_dir(cfg.topk_dir)
    sentinel = cfg.topk_dir / "_SUCCESS"
    if sentinel.exists() and not cfg.force and newer_than(sentinel, cfg.posts_parquet / "_SUCCESS", cfg.post_tags_parquet / "_SUCCESS"):
        log("[topk] already fresh - skip")
        return

    log("[topk] Calculate the top-K posts by tags...")
    posts_lf = pl.scan_parquet(f"{cfg.posts_parquet.as_posix()}/**/*.parquet")
    if cfg.reliable_only:
        posts_lf = posts_lf.filter(~pl.col("is_deleted") & ~pl.col("is_pending"))
    posts_lf = posts_lf.select(["id", "fav_count", "score"]).rename({"id": "post_id"})
    long = pl.scan_parquet(f"{cfg.post_tags_parquet.as_posix()}/**/*.parquet")
    joined = long.join(posts_lf, on="post_id", how="inner")

    sorted_joined = joined.sort(["tag_id", "fav_count", "score", "post_id"],
                                descending=[False, True, True, True])

    if cfg.topk_mode == "static":
        topk_rows = sorted_joined.group_by("tag_id", maintain_order=True).head(cfg.topk_k).collect(engine="streaming")
    else:
        df_tbl = (long.group_by("tag_id")
                       .agg(pl.len().alias("df_local"))
                       .with_columns(pl.col("df_local").cast(pl.Float64)))
        k_tbl = df_tbl.with_columns(
            (pl.col("df_local").sqrt() * pl.lit(cfg.topk_beta)).ceil()
              .clip(cfg.topk_k_min, cfg.topk_k_max)
              .cast(pl.Int32)
              .alias("k")
        )
        with_k = (sorted_joined
                    .join(k_tbl.lazy(), on="tag_id", how="left")
                    .with_columns(pl.col("k").fill_null(cfg.topk_k).alias("k"))
                    .with_columns(pl.row_number().over("tag_id").alias("rn")))
        topk_rows = with_k.filter(pl.col("rn") < pl.col("k")).collect(engine="streaming")

    shard_dirs = sorted(p for p in cfg.post_tags_parquet.glob("tag_shard=*") if p.is_dir())
    tag_shards = len(shard_dirs) if shard_dirs else 1
    
    packs: Dict[int, "io.BufferedWriter"] = {}
    idxs: Dict[int, "io.BufferedWriter"] = {}
    offsets: Dict[int, int] = {}
    
    def _get_writers(shard: int):
        if shard not in packs:
            outp = cfg.topk_dir / f"shard_{shard:04d}.topkpack"
            outi = cfg.topk_dir / f"index_{shard:04d}.bin"
            packs[shard] = open(outp, "wb")
            idxs[shard]  = open(outi, "wb")
            offsets[shard] = 0
        return packs[shard], idxs[shard]
    
    rows = topk_rows.select(["tag_id", "post_id"]).sort("tag_id")
    
    cur_tag: int | None = None
    acc: List[int] = []
    
    def _flush_current():
        if cur_tag is None or not acc:
            return
        shard = cur_tag % tag_shards
        pack, ix = _get_writers(shard)
        arr = np.array(acc, dtype=np.int64); arr.sort()
        encoded = encode_varint_delta(arr.tolist())
        off = offsets[shard]
        pack.write(encoded)
        ix.write(struct.pack("<iqi", int(cur_tag), off, len(encoded)))
        offsets[shard] = off + len(encoded)
    
    for tag_id, post_id in rows.iter_rows():
        tag_id = int(tag_id); post_id = int(post_id)
        if cur_tag is None:
            cur_tag = tag_id
        if tag_id != cur_tag:
            _flush_current()
            cur_tag, acc = tag_id, [post_id]
        else:
            acc.append(post_id)
    
    _flush_current()
    
    for f in packs.values(): f.close()
    for f in idxs.values(): f.close()
    
    sentinel.write_text("ok")
    log("[topk] done (pack)")