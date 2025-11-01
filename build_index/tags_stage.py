# build_index/tags_stage.py
from __future__ import annotations
import math, shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
import polars as pl
from tqdm import tqdm
import duckdb

from .config import Config
from .utils import ensure_dir, newer_than, log
from .graphs import tarjan_scc

import pyarrow as pa
import pyarrow.parquet as pq

# ---------- loaders ----------
def _read_tags_csv(path: Path) -> pl.DataFrame:
    df = pl.read_csv(path, infer_schema_length=0, ignore_errors=True)
    for col in ("id", "category", "post_count"):
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Int64, strict=False))
        else:
            df = df.with_columns(pl.lit(None).cast(pl.Int64).alias(col))
    if "name" not in df.columns:
        df = df.with_columns(pl.lit("").alias("name"))
    return (df
            .select(["id","name","category","post_count"])
            .with_columns([
                pl.col("id").cast(pl.Int64),
                pl.col("category").cast(pl.Int32),
                pl.col("post_count").cast(pl.Int64).fill_null(0),
            ]))

def _read_aliases_csv(path: Path) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame({"antecedent_name": [], "consequent_name": [], "status": []})
    df = pl.read_csv(path, infer_schema_length=0, ignore_errors=True)
    cols = {c.lower(): c for c in df.columns}
    def pick(name: str) -> str: return cols.get(name, name)
    return (df
            .rename({pick("antecedent_name"): "antecedent_name",
                     pick("consequent_name"): "consequent_name",
                     pick("status"): "status"})
            .select(["antecedent_name", "consequent_name", "status"]))

def _read_implications_csv(path: Path) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame({"antecedent_name": [], "consequent_name": [], "status": []})
    df = pl.read_csv(path, infer_schema_length=0, ignore_errors=True)
    cols = {c.lower(): c for c in df.columns}
    def pick(name: str) -> str: return cols.get(name, name)
    return (df
            .rename({pick("antecedent_name"): "antecedent_name",
                     pick("consequent_name"): "consequent_name",
                     pick("status"): "status"})
            .select(["antecedent_name", "consequent_name", "status"]))

# ---------- alias map ----------
def build_alias_map(tags_df: pl.DataFrame, aliases_df: pl.DataFrame) -> dict[str, str]:
    if aliases_df.is_empty():
        return {}
    act = aliases_df.filter(pl.col("status") == "active")\
                    .select(["antecedent_name", "consequent_name"])\
                    .to_dict(as_series=False)
    ants = act.get("antecedent_name", [])
    cons = act.get("consequent_name", [])
    next_of: Dict[str, str] = {}
    for a, b in zip(ants, cons):
        if a and b and a != b:
            next_of[a] = b

    nodes = sorted(set(list(next_of.keys()) + list(next_of.values())))
    edges: Dict[str, List[str]] = defaultdict(list)
    for a, b in next_of.items():
        edges[a].append(b)

    name2pc = {r[0]: int(r[1]) for r in tags_df.select(["name","post_count"]).iter_rows()}

    sccs = tarjan_scc(nodes, edges)
    in_cycle = set()
    canon_for: Dict[str, str] = {}
    for comp in sccs:
        if len(comp) > 1:
            in_cycle.update(comp)
        best, best_pc = None, -1
        for name in comp:
            pc = name2pc.get(name, -1)
            if pc > best_pc or (pc == best_pc and (best is None or name < best)):
                best, best_pc = name, pc
        best = best or sorted(comp)[0]
        for name in comp:
            canon_for[name] = best

    def compress(name: str) -> str:
        seen = set()
        cur = name
        while cur in next_of and cur not in seen:
            seen.add(cur); cur = next_of[cur]
        if cur in seen or cur in in_cycle:
            return canon_for.get(cur, canon_for.get(name, cur))
        return cur

    alias_map: Dict[str, str] = {}
    for a, b in next_of.items():
        alias_map[a] = compress(b)
    return alias_map

# ---------- step: tags + post_tags ----------
def _tags_arrow_for_duckdb(tags_df: pl.DataFrame):
    # tag->tag_id lookup as Arrow table
    return tags_df.rename({"name": "tag", "id": "tag_id"})\
                 .select(["tag", "tag_id", "category", "post_count"])\
                 .to_arrow()

def _alias_map_arrow(alias_map: dict[str, str]):
    import pyarrow as pa
    if not alias_map:
        return pa.table({"tag": pa.array([], type=pa.string()),
                         "canon": pa.array([], type=pa.string())})
    tags = list(alias_map.keys())
    canons = [alias_map[t] for t in tags]
    return pa.table({"tag": tags, "canon": canons})

def step_tags_and_post_tags(cfg: Config) -> None:
    ensure_dir(cfg.post_tags_parquet)
    tags_out = cfg.root / "tags_dict.parquet"
    sentinel = cfg.post_tags_parquet / "_SUCCESS"
    if sentinel.exists() and tags_out.exists() and not cfg.force and \
       newer_than(sentinel, cfg.posts_parquet / "_SUCCESS", cfg.tags_csv, cfg.tag_aliases_csv):
        log("[tags] already fresh - skip")
        return

    log("[tags] read tags.csv and aliases...")
    tags_df = _read_tags_csv(cfg.tags_csv)  # id, name, category, post_count
    aliases_df = _read_aliases_csv(cfg.tag_aliases_csv)
    alias_map = build_alias_map(tags_df, aliases_df)
    log(f"[tags] active aliases: {len(alias_map):,}")

    if cfg.post_tags_parquet.exists():
        shutil.rmtree(cfg.post_tags_parquet, ignore_errors=True)
    ensure_dir(cfg.post_tags_parquet)

    tags_lookup = tags_df.rename({"name": "tag"}).select([
        "tag", pl.col("id").alias("tag_id"), "category", "post_count"
    ])

    # ParquetWriter factory
    writers: Dict[int, pq.ParquetWriter] = {}
    writer_schema = pa.schema([("post_id", pa.int64()), ("tag_id", pa.int32())])

    def get_writer_for_shard(shard: int) -> pq.ParquetWriter:
        if shard in writers:
            return writers[shard]
        shard_dir = cfg.post_tags_parquet / f"tag_shard={shard}"
        ensure_dir(shard_dir)
        out_path = shard_dir / "data.parquet"
        w = pq.ParquetWriter(
            out_path.as_posix(),
            writer_schema,
            compression="zstd",
            version="2.6"
        )
        writers[shard] = w
        return w

    used_tag_ids: set[int] = set()

    files = sorted(Path(cfg.posts_parquet).rglob("*.parquet"))
    log(f"[tags] Processing parquet files: {len(files):,}")

    for i, f in enumerate(files, 1):
        df = pl.read_parquet(f, columns=["id", "tag_string", "is_deleted", "is_pending"])
        if cfg.reliable_only:
            df = df.filter(~pl.col("is_deleted") & ~pl.col("is_pending"))
        if df.is_empty():
            if i % 100 == 0: log(f"[tags] progress: {i}/{len(files)} (empty)")
            continue

        # explode tag_string -> tag
        long_names = (
            df.select([pl.col("id").alias("post_id"), pl.col("tag_string")])
              .with_columns(pl.col("tag_string").str.split(" ").alias("tags"))
              .drop("tag_string")
              .explode("tags")
              .rename({"tags": "tag"})
              .drop_nulls()
        )
        # aliases
        if alias_map:
            long_names = long_names.with_columns(pl.col("tag").replace(alias_map).alias("tag"))

        # Inside the chunk, remove duplicate post_id and tag entries.
        long_names = long_names.unique(subset=["post_id", "tag"])

        # join with a small dictionary -> tag_id
        joined = (long_names
                  .join(tags_lookup, on="tag", how="left")
                  .drop_nulls(subset=["tag_id"])
                  .select([pl.col("post_id").cast(pl.Int64),
                           pl.col("tag_id").cast(pl.Int32)]))
        if joined.is_empty():
            if i % 100 == 0: log(f"[tags] progress: {i}/{len(files)} (after join empty)")
            continue

        used_tag_ids.update(int(x) for x in joined.get_column("tag_id").to_list())

        # consider the shard as a module (uniform balance and fixed number of files)
        joined = joined.with_columns((pl.col("tag_id") % cfg.tag_shards).alias("tag_shard"))

        # break it down by shard in this chunk and add it to the corresponding file
        for shard_val, subdf in joined.group_by("tag_shard", maintain_order=True):
            shard = int(shard_val[0] if isinstance(shard_val, tuple) else shard_val)
            # post_id, tag_id (without shard)
            tbl = subdf.select(["post_id", "tag_id"]).to_arrow()
            if cfg.parquet_row_group_rows and len(tbl) > cfg.parquet_row_group_rows:
                start = 0
                rgs = cfg.parquet_row_group_rows
                w = get_writer_for_shard(shard)
                while start < len(tbl):
                    w.write_table(tbl.slice(start, rgs))
                    start += rgs
            else:
                w = get_writer_for_shard(shard)
                w.write_table(tbl)

        if i % 100 == 0 or i == len(files):
            log(f"[tags] progress: {i}/{len(files)}")

    for w in writers.values():
        w.close()

    if used_tag_ids:
        dict_df = (tags_df
                   .filter(pl.col("id").is_in(list(used_tag_ids)))
                   .rename({"id": "tag_id", "name": "tag"}))
        dict_df.write_parquet(tags_out, compression="zstd")
    else:
        pl.DataFrame({"tag_id": [], "tag": [], "category": [], "post_count": []})\
          .write_parquet(tags_out, compression="zstd")

    (cfg.post_tags_parquet / "_SUCCESS").write_text("ok")
    log(f"[tags] done: in index {len(used_tag_ids):,} tags; partitions={cfg.tag_shards}, per file per partition")

# ---------- step: implications ----------
def step_implications(cfg: Config) -> None:
    out_edges = cfg.root / "tag_implications.parquet"
    out_cache = cfg.root / "tag_ancestors_cache.parquet"
    if out_edges.exists() and out_cache.exists() and not cfg.force and \
       newer_than(out_edges, cfg.tag_implications_csv, cfg.tag_aliases_csv, cfg.tags_csv):
        log("[impl] already fresh - skip")
        return

    tags_df = _read_tags_csv(cfg.tags_csv)
    alias_map = build_alias_map(tags_df, _read_aliases_csv(cfg.tag_aliases_csv))

    impl = _read_implications_csv(cfg.tag_implications_csv)
    impl = impl.filter(pl.col("status") == "active")\
               .select(["antecedent_name","consequent_name"]).drop_nulls()

    if alias_map:
        impl = (impl.with_columns([
            pl.when(pl.col("antecedent_name").is_in(list(alias_map.keys())))
              .then(pl.col("antecedent_name").replace(alias_map))
              .otherwise(pl.col("antecedent_name")).alias("antecedent_name"),
            pl.when(pl.col("consequent_name").is_in(list(alias_map.keys())))
              .then(pl.col("consequent_name").replace(alias_map))
              .otherwise(pl.col("consequent_name")).alias("consequent_name"),
        ]))

    impl = impl.filter(pl.col("antecedent_name") != pl.col("consequent_name")).unique()

    names_set = set(tags_df.select("name").to_series().to_list())
    impl = impl.filter(pl.col("antecedent_name").is_in(list(names_set)) &
                       pl.col("consequent_name").is_in(list(names_set)))

    nodes = sorted(set(impl.select(["antecedent_name"]).to_series().to_list()
                       + impl.select(["consequent_name"]).to_series().to_list()))
    edges = defaultdict(list)
    for a, b in impl.iter_rows():
        edges[a].append(b)

    sccs = tarjan_scc(nodes, edges)
    comp_id: Dict[str,int] = {}
    for i, comp in enumerate(sccs):
        for v in comp: comp_id[v] = i

    dag_edges = []
    for a, b in impl.iter_rows():
        if comp_id[a] != comp_id[b]:
            dag_edges.append((a, b))
    dag = pl.DataFrame(dag_edges, schema=["a","b"]).unique()

    name2id = {r[0]: int(r[1]) for r in tags_df.select(["name","id"]).iter_rows()}
    dag = (dag.with_columns([
        pl.col("a").replace(name2id).alias("a_id"),
        pl.col("b").replace(name2id).alias("b_id"),
    ]).drop(["a","b"]).drop_nulls())
    dag.write_parquet(out_edges, compression="zstd")
    log(f"[impl] ribs in DAG: {dag.height:,}")

    # ancestor cache for top tags by post_count
    depth = max(1, int(cfg.anc_cache_depth))
    top_percent = min(1.0, max(0.0, cfg.anc_cache_top_percent))
    used = set(dag.select(["a_id","b_id"]).to_series(0).to_list()
               + dag.select(["a_id","b_id"]).to_series(1).to_list())
    stats = _read_tags_csv(cfg.tags_csv).filter(pl.col("id").is_in(list(used)))\
                                        .select(["id","post_count"]).sort("post_count", descending=True)
    cutoff = max(1, int(stats.height * top_percent))
    top_ids = set(stats.head(cutoff).select("id").to_series().to_list())

    adj = defaultdict(list)
    for a_id, b_id in dag.iter_rows():
        adj[a_id].append(b_id)

    from collections import deque
    pairs: list[tuple[int,int,int]] = []
    for t in tqdm(top_ids, desc="anc_cache"):
        seen = {t}
        q = deque([(t, 0)])
        while q:
            cur, d = q.popleft()
            if d == depth: continue
            for up in adj.get(cur, []):
                if up not in seen:
                    seen.add(up)
                    pairs.append((t, up, d + 1))
                    q.append((up, d + 1))

    if pairs:
        pl.DataFrame(pairs, schema={"tag_id": pl.Int32, "ancestor_id": pl.Int32, "depth": pl.Int8})\
          .write_parquet(out_cache, compression="zstd")
        log(f"[impl] ancestral cache written: {len(pairs):,} pairs")
    else:
        ensure_dir(out_cache.parent)
        pl.DataFrame({"tag_id": [], "ancestor_id": [], "depth": []})\
          .write_parquet(out_cache, compression="zstd")
        log("[impl] ancestral cache empty")