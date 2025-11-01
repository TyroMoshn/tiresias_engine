# autotest_index.py
# Location: I:/TIRESIAS_ENGINE/build_index/autotest_index.py
# Purpose: Validate integrity & consistency of pre-indexed data produced by build_index pipeline.
# Outputs: human-readable console report (+ optional txt report & checksum manifest).
# Bonus: can emit a Windows .bat launcher (see --emit-bat).

from __future__ import annotations
import argparse
import os
import sys
import io
import json
import math
import time
import struct
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

# 3rd-party
try:
    import polars as pl  # type: ignore
except Exception as e:
    print("[FATAL] polars is required: pip install polars")
    raise

try:
    import duckdb  # type: ignore
except Exception:
    duckdb = None

try:
    import numpy as np  # type: ignore
except Exception as e:
    print("[FATAL] numpy is required: pip install numpy")
    raise

try:
    from pyroaring import BitMap  # type: ignore
except Exception:
    BitMap = None  # ROAR checks will be skipped

# --- optional: prefer xxhash for fast file hashing ---
try:
    import xxhash  # type: ignore
except Exception:
    xxhash = None

# --- local project import (so this file can import build_index/* when placed next to them) ---
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Minimal-safe import: Config only (no step_* executed)
try:
    from build_index.config import Config  # type: ignore
except Exception:
    # Allow running even if package import fails; we'll reconstruct paths manually.
    Config = None  # type: ignore

# ---------------------- helpers ----------------------

@dataclass
class CheckResult:
    name: str
    status: str  # PASS | FAIL | WARN | SKIP
    details: List[str] = field(default_factory=list)
    duration_s: float = 0.0

    def fmt(self) -> str:
        s = f"[{self.status}] {self.name} ({self.duration_s:.2f}s)\n"
        for line in self.details:
            s += f"  - {line}\n"
        return s


class Suite:
    def __init__(self) -> None:
        self.results: List[CheckResult] = []

    def add(self, res: CheckResult) -> None:
        self.results.append(res)

    def summary(self) -> Tuple[int, int, int, int]:
        p = sum(1 for r in self.results if r.status == "PASS")
        f = sum(1 for r in self.results if r.status == "FAIL")
        w = sum(1 for r in self.results if r.status == "WARN")
        s = sum(1 for r in self.results if r.status == "SKIP")
        return p, f, w, s

    def render(self) -> str:
        out = io.StringIO()
        print("=" * 78, file=out)
        print("TIRESIAS_ENGINE – Autotest for Indexed Data", file=out)
        print("=" * 78, file=out)
        for r in self.results:
            print(r.fmt(), end="", file=out)
        p, f, w, s = self.summary()
        print("-" * 78, file=out)
        print(f"Summary: PASS={p}  FAIL={f}  WARN={w}  SKIP={s}", file=out)
        return out.getvalue()


def human_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    units = ["KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        x /= 1024.0
        if x < 1024.0:
            return f"{x:.2f} {u}"
    return f"{x:.2f} PB"


def file_hash(path: Path) -> str:
    if xxhash is not None:
        h = xxhash.xxh3_128()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    # fallback
    h = hashlib.blake2b(digest_size=16)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def varint_iter(buf: bytes) -> Iterable[int]:
    """Decode unsigned little-endian varints from bytes, yielding ints."""
    n = 0
    shift = 0
    for b in buf:
        n |= (b & 0x7F) << shift
        if b & 0x80:
            shift += 7
        else:
            yield n
            n = 0
            shift = 0
    if shift != 0:
        raise ValueError("truncated varint stream")


def decode_varint_delta(buf: bytes) -> List[int]:
    prev = 0
    out: List[int] = []
    for d in varint_iter(buf):
        prev += d
        out.append(prev)
    return out


# ---------------------- checks ----------------------

@dataclass
class Env:
    root: Path
    build_dir: Path
    quick: bool = False

    @property
    def paths(self) -> Dict[str, Path]:
        # Prefer Config if import succeeded (ensures same structure), else replicate default layout
        if Config is not None:
            cfg = Config(
                root=self.root,
                csv=self.root / "db.csv",
                posts_parquet=self.root / "posts_parquet",
                post_tags_parquet=self.root / "post_tags_parquet",
                bitmaps_dir=self.root / "bitmaps",
                topk_dir=self.root / "topk",
                features_dir=self.root / "features",
                tags_parquet=self.root / "tags.parquet",
                tags_csv=self.root / "tags.csv",
                tag_aliases_csv=self.root / "tag_aliases.csv",
                tag_implications_csv=self.root / "tag_implications.csv",
                pools_csv=self.root / "pools.csv",
                pools_parquet=self.root / "pools_parquet",
                pools_meta_parquet=self.root / "pools_meta.parquet",
                pools_entropy_parquet=self.root / "pool_entropy.parquet",
                pools_edges_parquet=self.root / "pool_edges.parquet",
                tag_co_from_pools_parquet=self.root / "tag_co_from_pools.parquet",
                mmaps_dir=self.root / "mmaps",
            )
            return {
                "posts_parquet": cfg.posts_parquet,
                "post_tags_parquet": cfg.post_tags_parquet,
                "tags_dict_parquet": self.root / "tags_dict.parquet",
                "tags_parquet": cfg.tags_parquet,
                "mmaps": cfg.mmaps_dir,
                "bitmaps": cfg.bitmaps_dir,
                "topk": cfg.topk_dir,
                "pools_parquet": cfg.pools_parquet,
                "pools_meta": cfg.pools_meta_parquet,
                "pool_entropy": cfg.pools_entropy_parquet,
                "pool_edges": cfg.pools_edges_parquet,
                "tag_co_from_pools": cfg.tag_co_from_pools_parquet,
                "csv_db": cfg.csv,
                "csv_tags": cfg.tags_csv,
                "csv_pools": cfg.pools_csv,
            }
        # Fallback
        return {
            "posts_parquet": self.root / "posts_parquet",
            "post_tags_parquet": self.root / "post_tags_parquet",
            "tags_dict_parquet": self.root / "tags_dict.parquet",
            "tags_parquet": self.root / "tags.parquet",
            "mmaps": self.root / "mmaps",
            "bitmaps": self.root / "bitmaps",
            "topk": self.root / "topk",
            "pools_parquet": self.root / "pools_parquet",
            "pools_meta": self.root / "pools_meta.parquet",
            "pool_entropy": self.root / "pool_entropy.parquet",
            "pool_edges": self.root / "pool_edges.parquet",
            "tag_co_from_pools": self.root / "tag_co_from_pools.parquet",
            "csv_db": self.root / "db.csv",
            "csv_tags": self.root / "tags.csv",
            "csv_pools": self.root / "pools.csv",
        }


# 1) Schema & basic integrity -------------------------------------------------

def check_posts_parquet_schema(env: Env) -> CheckResult:
    t0 = time.perf_counter()
    pp = env.paths["posts_parquet"]
    if not pp.exists():
        return CheckResult("posts_parquet schema", "FAIL", [f"Missing: {pp}"])
    # Pick a small sample of files (partitioned by rating/year/month)
    files = sorted(pp.rglob("*.parquet"))
    if not files:
        return CheckResult("posts_parquet schema", "FAIL", ["No parquet files found inside posts_parquet/"])
    sample = files[:max(1, 6 if not env.quick else 2)]

    # Expected columns (based on io_stage)
    expected = {
        "id", "uploader_id", "created_at", "md5", "source", "rating",
        "image_width", "image_height", "tag_string", "locked_tags", "fav_count",
        "file_ext", "parent_id", "change_seq", "approver_id", "file_size",
        "comment_count", "description", "duration", "updated_at", "is_deleted",
        "is_pending", "is_flagged", "score", "up_score", "down_score",
        "is_rating_locked", "is_status_locked", "is_note_locked", "year", "month",
    }

    problems: List[str] = []
    ok_details: List[str] = []
    for f in sample:
        df = pl.read_parquet(f)
        cols = set(df.columns)

        # --- NEW: detect hive-style partition keys from file path relative to posts_parquet ---
        partition_cols: Dict[str,str] = {}
        try:
            rel_parts = f.relative_to(pp).parts  # parts inside posts_parquet dir
        except Exception:
            rel_parts = f.parts  # fallback, but ideally relative_to(pp) should work
        # ignore last part (filename); iterate directory parts and parse key=value
        for part in rel_parts[:-1]:
            if '=' in part:
                k, v = part.split('=', 1)
                partition_cols[k] = v

        if partition_cols:
            ok_details.append(f"{f.name}: detected partition keys {sorted(partition_cols.keys())}")

        # missing columns that are neither in file columns nor provided by partition keys
        missing = sorted([c for c in (expected - cols) if c not in partition_cols])
        extra = sorted(list(cols - expected))

        if missing:
            problems.append(f"{f.name}: missing columns: {missing[:8]}{'…' if len(missing)>8 else ''}")        # Type spot-checks
        dtypes = df.schema
        def chk(col: str, allowed: Tuple[pl.DataType, ...]) -> None:
            if col in dtypes and dtypes[col] not in allowed:
                problems.append(f"{f.name}: column '{col}' has dtype {dtypes[col]}, expected one of {allowed}")
        chk("id", (pl.Int64,))
        chk("image_width", (pl.Int32, pl.Int64))
        chk("image_height", (pl.Int32, pl.Int64))
        chk("fav_count", (pl.Int32, pl.Int64))
        chk("score", (pl.Int32, pl.Int64))
        chk("is_deleted", (pl.Boolean, pl.UInt8, pl.Int8))
        chk("is_pending", (pl.Boolean, pl.UInt8, pl.Int8))
        if "created_at" in dtypes and dtypes["created_at"] != pl.Datetime:
            problems.append(f"{f.name}: created_at dtype is {dtypes['created_at']} (expected Datetime)")
        if not missing:
            ok_details.append(f"{f.name}: schema OK ({len(cols)} cols)")
        if env.quick:
            break

    dur = time.perf_counter() - t0
    if problems:
        return CheckResult("posts_parquet schema", "FAIL", problems, dur)
    return CheckResult("posts_parquet schema", "PASS", ok_details[:4], dur)


def check_post_tags_schema(env: Env) -> CheckResult:
    t0 = time.perf_counter()
    pt = env.paths["post_tags_parquet"]
    if not pt.exists():
        return CheckResult("post_tags_parquet schema", "FAIL", [f"Missing: {pt}"])

    shards = sorted([p for p in pt.glob("tag_shard=*") if p.is_dir()])
    if not shards:
        return CheckResult("post_tags_parquet schema", "FAIL", ["No tag_shard=* partitions found."])

    problems: List[str] = []
    ok: List[str] = []
    # Sample up to 5 shards
    for shard in shards[: (2 if env.quick else 5) ]:
        files = sorted(shard.glob("*.parquet"))
        if not files:
            problems.append(f"{shard.name}: empty shard")
            continue
        df = pl.read_parquet(files[0])
        cols = set(df.columns)
        if cols != {"post_id", "tag_id"}:
            problems.append(f"{shard.name}: unexpected columns {sorted(cols)} (expected ['post_id','tag_id'])")
            continue
        # dtype check
        sch = df.schema
        if sch.get("post_id") not in (pl.Int64, pl.Int32):
            problems.append(f"{shard.name}: post_id dtype {sch.get('post_id')} (expected Int64/Int32)")
        if sch.get("tag_id") not in (pl.Int32, pl.Int64):
            problems.append(f"{shard.name}: tag_id dtype {sch.get('tag_id')} (expected Int32/Int64)")
        # no duplicates within file
        if not env.quick:
            total = df.height
            unique_pairs = df.unique(subset=["post_id", "tag_id"]).height
            dup = total - unique_pairs
            if dup != 0:
                problems.append(f"{shard.name}: found {dup:,} duplicate (post_id,tag_id) pairs")
            if int(dup) != 0:
                problems.append(f"{shard.name}: found {int(dup):,} duplicate (post_id,tag_id) pairs")
        ok.append(f"{shard.name}: schema OK, rows≈{df.height:,}")

    dur = time.perf_counter() - t0
    if problems:
        return CheckResult("post_tags_parquet schema", "FAIL", problems, dur)
    return CheckResult("post_tags_parquet schema", "PASS", ok[:4], dur)


def check_tags_dict(env: Env) -> CheckResult:
    t0 = time.perf_counter()
    tp = env.paths["tags_dict_parquet"]
    if not tp.exists():
        return CheckResult("tags_dict.parquet", "FAIL", [f"Missing: {tp}"])
    df = pl.read_parquet(tp)
    cols = set(df.columns)
    need = {"tag_id", "tag", "category", "post_count"}
    missing = sorted(list(need - cols))
    details = [f"rows={df.height:,}"]
    if missing:
        return CheckResult("tags_dict.parquet", "FAIL", [f"Missing columns: {missing}"] + details, time.perf_counter()-t0)
    sch = df.schema
    probs = []
    if sch.get("tag_id") not in (pl.Int32, pl.Int64):
        probs.append(f"tag_id dtype {sch.get('tag_id')} (expected Int32/Int64)")
    if sch.get("post_count") not in (pl.Int64, pl.Int32):
        probs.append(f"post_count dtype {sch.get('post_count')} (expected Int64/Int32)")
    if probs:
        return CheckResult("tags_dict.parquet", "WARN", probs + details, time.perf_counter()-t0)
    return CheckResult("tags_dict.parquet", "PASS", details, time.perf_counter()-t0)


# 2) Cross-format consistency (hashes, dimensions) ----------------------------

def check_mmaps(env: Env) -> CheckResult:
    t0 = time.perf_counter()
    mm = env.paths["mmaps"]
    if not mm.exists():
        return CheckResult("mmaps/*.bin", "FAIL", [f"Missing dir: {mm}"])

    # Required arrays
    req = [
        "post_ids.bin", "epoch_day.bin", "w.bin", "h.bin",
        "score.bin", "fav_count.bin", "is_deleted.bin", "is_pending.bin",
    ]
    missing = [r for r in req if not (mm / r).exists()]
    if missing:
        return CheckResult("mmaps/*.bin", "FAIL", [f"Missing files: {missing}"])

    # Load and validate alignment
    post_ids = np.memmap(mm / "post_ids.bin", mode="r", dtype=np.int64)
    n = post_ids.shape[0]
    probs = []
    details = [f"rows={n:,}"]

    def chk(name: str, dtype: Any) -> None:
        arr = np.memmap(mm / f"{name}.bin", mode="r", dtype=dtype)
        if arr.shape[0] != n:
            probs.append(f"{name}.bin length {arr.shape[0]:,} != post_ids {n:,}")
    chk("epoch_day", np.int32)
    chk("w", np.int32)
    chk("h", np.int32)
    chk("score", np.int32)
    chk("fav_count", np.int32)
    chk("is_deleted", np.uint8)
    chk("is_pending", np.uint8)

    # Sorted post_ids check
    if n > 1:
        dif = np.diff(post_ids)
        if not np.all(dif > 0):
            bad = int(np.sum(dif <= 0))
            probs.append(f"post_ids not strictly increasing; {bad} non-positive diffs")
        else:
            details.append("post_ids strictly increasing")

    # Optional: post_in_pools_count
    pip = mm / "post_in_pools_count.bin"
    if pip.exists():
        arr = np.memmap(pip, mode="r", dtype=np.int32)
        if arr.shape[0] != n:
            probs.append(f"post_in_pools_count.bin length {arr.shape[0]:,} != post_ids {n:,}")
        else:
            details.append(f"post_in_pools_count present (max={int(arr.max())})")

    dur = time.perf_counter() - t0
    if probs:
        return CheckResult("mmaps/*.bin", "FAIL", probs + details, dur)
    return CheckResult("mmaps/*.bin", "PASS", details, dur)


def check_rating_roars(env: Env) -> CheckResult:
    t0 = time.perf_counter()
    if BitMap is None:
        return CheckResult("rating_*.roar", "SKIP", ["pyroaring not installed"])
    mm = env.paths["mmaps"]
    roars = [mm / "rating_s.roar", mm / "rating_q.roar", mm / "rating_e.roar"]
    if not all(p.exists() for p in roars):
        return CheckResult("rating_*.roar", "SKIP", ["rating roar files not found – stage may be optional or failed"])

    # Read counts from bitmaps
    sets: Dict[str, BitMap] = {}
    sizes: Dict[str, int] = {}
    for name, p in zip(["s","q","e"], roars):
        blob = p.read_bytes()
        bm = BitMap.deserialize(blob)
        sets[name] = bm
        sizes[name] = len(bm)

    details = [f"sizes: s={sizes['s']:,} q={sizes['q']:,} e={sizes['e']:,}"]

    # Pairwise disjointness & union size vs posts_parquet count
    inter = {('s','q'): len(sets['s'] & sets['q']),
             ('s','e'): len(sets['s'] & sets['e']),
             ('q','e'): len(sets['q'] & sets['e'])}
    problems = []
    for (a,b), v in inter.items():
        if v != 0:
            problems.append(f"rating_{a} ∩ rating_{b} = {v:,} (expected 0)")

    total_roar = len(sets['s'] | sets['q'] | sets['e'])

    # Count rows via DuckDB or directory enumeration
    if duckdb is not None:
        con = duckdb.connect()
        con.execute("PRAGMA threads=%d" % os.cpu_count())
        # Only COUNT(*) per rating by scanning the partition
        posts_root = env.paths["posts_parquet"].as_posix().replace("\\", "/")
        cnt = 0
        for r in ("s","q","e"):
            pat = f"{posts_root}/rating={r}/**/*.parquet"
            cur = con.execute(f"SELECT COUNT(*) FROM parquet_scan('{pat}')").fetchone()[0]
            cnt += int(cur)
        cnt_posts = int(cnt)
    else:
        cnt_posts = sum(1 for _ in env.paths["posts_parquet"].rglob("*.parquet"))  # very rough fallback

    if total_roar != cnt_posts:
        problems.append(f"|s∪q∪e| = {total_roar:,} vs posts_parquet rows = {cnt_posts:,}")
    else:
        details.append("rating masks cover all posts (union matches count)")

    dur = time.perf_counter() - t0
    if problems:
        return CheckResult("rating_*.roar", "FAIL", problems + details, dur)
    return CheckResult("rating_*.roar", "PASS", details, dur)


# 3) Bitmaps & ROAR indexing --------------------------------------------------

def check_tag_bitmaps_against_post_tags(env: Env, sample_tags: int = 5) -> CheckResult:
    t0 = time.perf_counter()
    bb = env.paths["bitmaps"]
    pt_dir = env.paths["post_tags_parquet"]
    if not bb.exists():
        return CheckResult("bitmaps", "SKIP", ["bitmaps dir not found – run bitmaps stage to enable"], time.perf_counter()-t0)
    if BitMap is None:
        return CheckResult("bitmaps", "SKIP", ["pyroaring not installed"], time.perf_counter()-t0)

    # detect old vs new layout
    roar_files = sorted(bb.glob("tag_*.roar"))
    packs = sorted(bb.glob("shard_*.roarpack"))
    idxs  = sorted(bb.glob("index_*.bin"))

    # ---------------- legacy mode ----------------
    if roar_files:
        sample = roar_files[: (2 if env.quick else min(sample_tags, len(roar_files)))]
        problems: list[str] = []
        details: list[str] = []
        for rf in sample:
            tag_id = int(rf.stem.split("_")[1])
            bm = BitMap.deserialize(rf.read_bytes())
            lf = pl.scan_parquet(f"{pt_dir.as_posix()}/**/*.parquet").filter(pl.col("tag_id") == tag_id).select("post_id")
            ids = set(lf.collect(streaming=True).get_column("post_id").to_list())
            roar_ids = set(map(int, list(bm)))
            if ids != roar_ids:
                miss = len(ids - roar_ids)
                extra = len(roar_ids - ids)
                problems.append(f"tag {tag_id:06d}: mismatch (missing {miss:,}, extra {extra:,})")
            else:
                details.append(f"tag {tag_id:06d}: OK (|ids|={len(ids):,})")
            if env.quick:
                break

        dur = time.perf_counter() - t0
        if problems:
            return CheckResult("bitmaps/tag_*.roar", "FAIL", problems + details, dur)
        return CheckResult("bitmaps/tag_*.roar", "PASS", details, dur)

    # ---------------- roarpack mode ----------------
    if not packs or not idxs:
        return CheckResult("bitmaps/*", "SKIP", ["No tag_*.roar or shard_*.roarpack found"], time.perf_counter()-t0)

    # sanity check index ↔ pack
    for pack, idx in zip(packs, idxs):
        data = pack.read_bytes()
        b = idx.read_bytes()
        rec_size = struct.calcsize("<iqi")
        n = len(b) // rec_size
        ok = 0
        for i in range(n):
            tag_id, ofs, ln = struct.unpack_from("<iqi", b, i * rec_size)
            if ofs < 0 or ln <= 0 or (ofs + ln) > len(data):
                return CheckResult("roarpack/index", "FAIL",
                                   [f"Corrupted index in {idx.name} at entry {i} (tag {tag_id})"],
                                   time.perf_counter() - t0)
            try:
                bm = BitMap.deserialize(data[ofs:ofs+ln])
                _ = len(bm)
                ok += 1
            except Exception as e:
                return CheckResult("roarpack/index", "FAIL",
                                   [f"Deserialize failure {idx.name}:{i}: {e}"],
                                   time.perf_counter() - t0)

    # cross-check one random tag
    if not env.quick:
        import random
        shard_idx = random.choice(range(len(idxs)))
        b = idxs[shard_idx].read_bytes()
        data = packs[shard_idx].read_bytes()
        rec_size = struct.calcsize("<iqi")
        recs = len(b) // rec_size
        if recs:
            entry = random.randrange(recs)
            tag_id, ofs, ln = struct.unpack_from("<iqi", b, entry * rec_size)
            bm = BitMap.deserialize(data[ofs:ofs+ln])
            lf = pl.scan_parquet(f"{pt_dir.as_posix()}/**/*.parquet").filter(pl.col("tag_id") == tag_id).select("post_id")
            ids = set(lf.collect(streaming=True).get_column("post_id").to_list())
            roar_ids = set(map(int, list(bm)))
            if ids != roar_ids:
                miss = len(ids - roar_ids)
                extra = len(roar_ids - ids)
                return CheckResult("roarpack/index", "FAIL",
                                   [f"sample tag {tag_id}: mismatch (missing {miss}, extra {extra})"],
                                   time.perf_counter() - t0)

    return CheckResult("roarpack/index", "PASS",
                       [f"Validated {len(packs)} shard(s), structure OK"], time.perf_counter()-t0)


# 4) TOPK structures ----------------------------------------------------------

def check_topk(env: Env, sample_tags: int = 5, k_cap: int = 2000) -> CheckResult:
    t0 = time.perf_counter()
    topk_dir = env.paths["topk"]
    if not topk_dir.exists():
        return CheckResult("topk", "SKIP", ["topk dir not found – run topk stage to enable"], time.perf_counter()-t0)

    legacy_files = sorted(topk_dir.glob("tag_*.topk.bin"))
    packs_all = sorted(topk_dir.glob("shard_*.topkpack"))
    idxs_all  = sorted(topk_dir.glob("index_*.bin"))

    # ---------------- legacy mode ----------------
    if legacy_files:
        files = legacy_files[: (2 if env.quick else min(sample_tags, len(legacy_files)))]
        return _check_topk_files(env, files, k_cap, t0)

    # ---------------- new pack mode ----------------
    if not packs_all or not idxs_all:
        return CheckResult("topk", "SKIP", ["No tag_*.topk.bin or shard_*.topkpack found"], time.perf_counter()-t0)

    # If requested, sample a limited number of shards (deterministic, evenly spaced)
    packs = packs_all
    idxs = idxs_all
    total_shards = len(packs_all)
    if sample_tags and sample_tags > 0 and total_shards > sample_tags:
        # choose sample_tags indices evenly spaced across [0, total_shards-1], include endpoints
        if sample_tags == 1:
            sel = [0]
        else:
            sel = [int(round(i * (total_shards - 1) / (sample_tags - 1))) for i in range(sample_tags)]
        packs = [packs_all[i] for i in sel]
        idxs  = [idxs_all[i]  for i in sel]

    rec_size = struct.calcsize("<iqi")
    problems: list[str] = []
    details: list[str] = []
    total_entries = 0

    for pack, idx in zip(packs, idxs):
        data = pack.read_bytes()
        b = idx.read_bytes()
        n = len(b) // rec_size
        total_entries += n
        # iterate entries but keep checks reasonably fast; heavy membership check only for first entry per shard (as before)
        for i in range(n):
            tag_id, ofs, ln = struct.unpack_from("<iqi", b, i * rec_size)
            if ofs < 0 or ln <= 0 or (ofs + ln) > len(data):
                problems.append(f"{idx.name}: corrupted entry {i} (tag {tag_id})")
                continue
            buf = data[ofs:ofs+ln]
            try:
                ids = decode_varint_delta(buf)
            except Exception as e:
                problems.append(f"{idx.name}:{i} decode error: {e}")
                continue
            if any(ids[i] >= ids[i+1] for i in range(len(ids)-1)):
                problems.append(f"{idx.name}:{i} ids not strictly increasing (tag {tag_id})")
            if not env.quick and len(ids) > 0:
                # validate top-K membership for one representative tag entry per shard (keeps cost bounded)
                if i == 0:
                    K = min(len(ids), k_cap)
                    top_set = _compute_top_set(env, tag_id, K)
                    if not set(ids[:K]).issubset(top_set):
                        problems.append(f"{idx.name}: tag {tag_id} topK membership mismatch")

        details.append(f"{idx.name}: {n} entries checked")

    # report whether we sampled or scanned all
    if len(packs) != total_shards:
        details.insert(0, f"Sampled {len(packs)}/{total_shards} shard(s) (sample_tags={sample_tags})")

    dur = time.perf_counter() - t0
    if problems:
        return CheckResult("topk/pack", "WARN", problems + details, dur)
    return CheckResult("topk/pack", "PASS", [f"{total_entries:,} entries OK"] + details, dur)



def _compute_top_set(env: Env, tag_id: int, K: int) -> set[int]:
    posts_lf = pl.scan_parquet(f"{env.paths['posts_parquet'].as_posix()}/**/*.parquet")
    posts_lf = posts_lf.select(["id","fav_count","score"]).rename({"id": "post_id"})
    long = pl.scan_parquet(f"{env.paths['post_tags_parquet'].as_posix()}/**/*.parquet")
    joined = long.filter(pl.col("tag_id") == tag_id).join(posts_lf, on="post_id", how="inner")
    tbl = joined.sort(["fav_count","score","post_id"], descending=[True,True,True]).select("post_id").unique().head(K)
    return set(tbl.collect(streaming=True).get_column("post_id").to_list())


def _check_topk_files(env: Env, files: list[Path], k_cap: int, t0: float) -> CheckResult:
    problems: list[str] = []
    details: list[str] = []

    for f in files:
        tag_id = int(f.stem.split("_")[1])
        buf = f.read_bytes()
        try:
            ids = decode_varint_delta(buf)
        except Exception as e:
            problems.append(f"{f.name}: decode error: {e}")
            continue
        if any(ids[i] >= ids[i+1] for i in range(len(ids)-1)):
            problems.append(f"{f.name}: ids not strictly increasing after delta decode")
            continue
        details.append(f"{f.name}: decoded {len(ids):,} ids")
        if not env.quick:
            K = min(len(ids), k_cap)
            top_set = _compute_top_set(env, tag_id, K)
            if not set(ids[:K]).issubset(top_set) or len(set(ids)) < len(top_set) * 0.8:
                problems.append(f"{f.name}: membership differs from computed TOP{K}")

    dur = time.perf_counter() - t0
    if problems:
        return CheckResult("topk/*.bin", "WARN", problems + details, dur)
    return CheckResult("topk/*.bin", "PASS", details, dur)


# 5) Pools-derived structures --------------------------------------------------

def check_pools(env: Env) -> CheckResult:
    t0 = time.perf_counter()
    meta = env.paths["pools_meta"]
    long_dir = env.paths["pools_parquet"]
    H_path = env.paths["pool_entropy"]

    if not (meta.exists() and long_dir.exists() and H_path.exists()):
        return CheckResult("pools structures", "SKIP", ["pools_meta/pools_parquet/pool_entropy are not all present"] , time.perf_counter()-t0)

    # Meta schema
    m = pl.read_parquet(meta)
    need_meta = {"pool_id","size","category","creator_id"}
    miss = sorted(list(need_meta - set(m.columns)))
    if miss:
        return CheckResult("pools structures", "FAIL", [f"pools_meta missing {miss}"] , time.perf_counter()-t0)

    # Long parquet sample
    shards = sorted(long_dir.glob("pool_shard=*"))
    if not shards:
        return CheckResult("pools structures", "FAIL", ["no pool_shard=* partitions"] , time.perf_counter()-t0)
    df = pl.read_parquet(next(iter(shards)).glob("*.parquet").__iter__().__next__())
    if set(df.columns) != {"pool_id","post_id"}:
        return CheckResult("pools structures", "FAIL", ["pools long parquet must have ['pool_id','post_id']"] , time.perf_counter()-t0)

    # Entropy plausible & joinable
    H = pl.read_parquet(H_path)
    cols = set(H.columns)
    for col in ("pool_id","H_tags"):
        if col not in cols:
            return CheckResult("pools structures", "FAIL", [f"pool_entropy missing {col}"] , time.perf_counter()-t0)
    if H.height == 0 or float(H.select(pl.col("H_tags").max())[0,0]) <= 0:
        return CheckResult("pools structures", "FAIL", ["pool_entropy is empty or non-positive"] , time.perf_counter()-t0)

    # Optional edges/tag-co quick sanity
    probs: List[str] = []
    details: List[str] = []
    edges = env.paths["pool_edges"]
    if edges.exists():
        E = pl.read_parquet(edges)
        if not set(["a_post_id","b_post_id","weight","pools"]).issubset(E.columns):
            probs.append("pool_edges.parquet: unexpected schema")
        else:
            details.append(f"pool_edges rows={E.height:,}")
            if E.height > 0:
                wmin = float(E.select(pl.col("weight").min())[0,0])
                wmax = float(E.select(pl.col("weight").max())[0,0])
                if not (0.0 < wmin <= wmax):
                    probs.append("pool_edges weights not positive")
    tagco = env.paths["tag_co_from_pools"]
    if tagco.exists():
        T = pl.read_parquet(tagco)
        if not set(["a_tag_id","b_tag_id","weight","pools"]).issubset(T.columns):
            probs.append("tag_co_from_pools: unexpected schema")
        else:
            details.append(f"tag_co_from_pools rows={T.height:,}")
    dur = time.perf_counter() - t0
    if probs:
        return CheckResult("pools structures", "WARN", probs + details, dur)
    return CheckResult("pools structures", "PASS", details, dur)


# 6) CSV ↔ Parquet reconciliation (via DuckDB) --------------------------------

def check_against_csv(env: Env, limit_rows: int = 0) -> CheckResult:
    t0 = time.perf_counter()
    if duckdb is None:
        return CheckResult("CSV vs Parquet", "SKIP", ["duckdb not installed"])

    db_csv = env.paths["csv_db"]
    tags_csv = env.paths["csv_tags"]
    if not db_csv.exists():
        return CheckResult("CSV vs Parquet", "SKIP", [f"db.csv not found at {db_csv}"])

    con = duckdb.connect()
    con.execute("PRAGMA threads=%d" % os.cpu_count())

    posts_root = env.paths["posts_parquet"].as_posix().replace("\\", "/")
    # Parquet stats
    q_pq = f"SELECT COUNT(*) AS n, SUM(id) AS sum_id, MIN(id) AS min_id, MAX(id) AS max_id FROM parquet_scan('{posts_root}/**/*.parquet')"
    n_pq, sum_id_pq, min_id_pq, max_id_pq = con.execute(q_pq).fetchone()

    # CSV stats (typed similarly to io_stage: ALL_VARCHAR=1, the CAST happens in io stage, but here we do typed read)
    # We avoid full materialization by letting duckdb stream.
    opt = "SAMPLE_SIZE=-1"
    if limit_rows > 0:
        q_csv = f"SELECT COUNT(*) AS n, SUM(CAST(id AS BIGINT)) AS sum_id, MIN(CAST(id AS BIGINT)) AS min_id, MAX(CAST(id AS BIGINT)) AS max_id FROM read_csv_auto('{db_csv.as_posix().replace('\\','/')}', ALL_VARCHAR=1, {opt}) LIMIT {limit_rows}"
    else:
        q_csv = f"SELECT COUNT(*) AS n, SUM(CAST(id AS BIGINT)) AS sum_id, MIN(CAST(id AS BIGINT)) AS min_id, MAX(CAST(id AS BIGINT)) AS max_id FROM read_csv_auto('{db_csv.as_posix().replace('\\','/')}', ALL_VARCHAR=1, {opt})"
    n_csv, sum_id_csv, min_id_csv, max_id_csv = con.execute(q_csv).fetchone()

    details = [f"posts_parquet: n={int(n_pq):,}  CSV: n={int(n_csv):,}"]
    probs: List[str] = []
    if int(n_pq) != int(n_csv):
        probs.append("Row count differs between CSV and Parquet")
    if int(min_id_pq) != int(min_id_csv) or int(max_id_pq) != int(max_id_csv):
        probs.append("ID min/max differ between CSV and Parquet")
    if int(sum_id_pq) != int(sum_id_csv):
        probs.append("SUM(id) differs – data conversion inconsistency suspected")

    dur = time.perf_counter() - t0
    if probs:
        return CheckResult("CSV vs Parquet", "FAIL", probs + details, dur)
    return CheckResult("CSV vs Parquet", "PASS", details, dur)


# 7) Reproducibility checks (manifest) ----------------------------------------

def reproducibility(env: Env, manifest: Path) -> CheckResult:
    t0 = time.perf_counter()
    # Hash a representative subset of files
    candidates = []
    for p in [
        env.paths["posts_parquet"], env.paths["post_tags_parquet"], env.paths["tags_dict_parquet"],
        env.paths["mmaps"], env.paths["bitmaps"], env.paths["topk"], env.paths["pool_entropy"],
        env.paths["pool_edges"], env.paths["tag_co_from_pools"],
    ]:
        if not p.exists():
            continue
        if p.is_file():
            candidates.append(p)
        else:
            # Take up to N small-ish files deterministically
            files = sorted([f for f in p.rglob("*") if f.is_file()])
            if not files:
                continue
            # First, middle, last 5
            pick = files[:3] + files[max(0,len(files)//2-2):len(files)//2+3] + files[-3:]
            candidates.extend(pick)

    snapshot = {str(p.relative_to(env.root)): {"size": p.stat().st_size, "hash": file_hash(p)} for p in candidates}

    if manifest.exists():
        old = json.loads(manifest.read_text(encoding="utf-8"))
        added = sorted(set(snapshot.keys()) - set(old.keys()))
        removed = sorted(set(old.keys()) - set(snapshot.keys()))
        changed = [k for k in snapshot.keys() if k in old and (snapshot[k]["hash"] != old[k]["hash"] or snapshot[k]["size"] != old[k]["size"])]
        details = [f"files: {len(snapshot)}  changed={len(changed)} added={len(added)} removed={len(removed)}"]
        if changed or added or removed:
            status = "WARN"
            if not added and not removed and len(changed) == 0:
                status = "PASS"
            return CheckResult("Reproducibility (manifest)", status, details, time.perf_counter()-t0)
        else:
            return CheckResult("Reproducibility (manifest)", "PASS", details, time.perf_counter()-t0)
    else:
        # First run: write manifest
        manifest.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        return CheckResult("Reproducibility (manifest)", "PASS", [f"Created manifest with {len(snapshot)} files: {manifest.name}"], time.perf_counter()-t0)


# 8) Performance & memory ------------------------------------------------------

def perf_profile(env: Env) -> CheckResult:
    t0 = time.perf_counter()
    details: List[str] = []
    probs: List[str] = []

    # mmap open times
    mm = env.paths["mmaps"]
    if mm.exists():
        t = time.perf_counter()
        post_ids = np.memmap(mm / "post_ids.bin", mode="r", dtype=np.int64)
        t_open = time.perf_counter() - t
        details.append(f"mmaps/post_ids.bin: open {t_open*1000:.1f} ms, rows={post_ids.shape[0]:,}")
        # Random slicing cost
        t = time.perf_counter()
        _ = post_ids[::1024][:1024]
        t_slice = time.perf_counter() - t
        details.append(f"  slice [::1024][:1024]: {t_slice*1000:.1f} ms")

    # ROAR decode timing
    if BitMap is not None and (mm / "rating_s.roar").exists():
        blob = (mm / "rating_s.roar").read_bytes()
        t = time.perf_counter()
        bm = BitMap.deserialize(blob)
        t_dec = time.perf_counter() - t
        details.append(f"rating_s.roar: deserialize {t_dec*1000:.1f} ms, |S|={len(bm):,}")

    # post_tags sampling read
    pt = env.paths["post_tags_parquet"]
    if pt.exists():
        t = time.perf_counter()
        cnt = pl.scan_parquet(f"{pt.as_posix()}/**/*.parquet").select(pl.len()).collect(streaming=True)[0,0]
        t_cnt = time.perf_counter() - t
        details.append(f"post_tags_parquet: COUNT(*) in {t_cnt:.2f} s -> {int(cnt):,} rows")

    dur = time.perf_counter() - t0
    return CheckResult("Performance snapshot", "PASS", details, dur)


# ---------------------- batch launcher writer --------------------------------

BAT_TEXT = r"""@echo off
setlocal ENABLEDELAYEDEXPANSION
REM TIRESIAS_ENGINE / Autotest launcher

REM Root folder of the project (adjust if needed)
set ROOT=I:\\TIRESIAS_ENGINE
set DATA=%ROOT%\\data
set BUILD=%ROOT%\\build_index

REM Ensure UTF-8 in console
set PYTHONUTF8=1

REM Add build_index to PYTHONPATH to import Config
set PYTHONPATH=%BUILD%;%PYTHONPATH%

python "%BUILD%\\autotest_index.py" --root "%DATA%" --report "%BUILD%\\autotest_report.txt" --manifest "%BUILD%\\autotest_manifest.json" --sample-tags 10 --k-cap 1000 --emit-bat 0
set ERR=%ERRORLEVEL%
if %ERR% NEQ 0 (
  echo Autotest: FAIL (exit code %ERR%)
  exit /b %ERR%
) else (
  echo Autotest: OK
  exit /b 0
)
"""


def maybe_emit_bat(build_dir: Path, do_emit: bool) -> Optional[Path]:
    if not do_emit:
        return None
    p = build_dir / "run_autotest.bat"
    p.write_text(BAT_TEXT, encoding="utf-8")
    return p


# ---------------------- main --------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="TIRESIAS_ENGINE – Autotest for Indexed Data")
    ap.add_argument("--root", type=Path, default=Path("I:/TIRESIAS_ENGINE/data"), help="Path to data root")
    ap.add_argument("--report", type=Path, default=None, help="Optional path to write a txt report")
    ap.add_argument("--manifest", type=Path, default=None, help="Path to reproducibility manifest JSON")
    ap.add_argument("--quick", action="store_true", help="Speed up by sampling and skipping heavy checks")
    ap.add_argument("--sample-tags", type=int, default=5, help="How many tag samples to cross-check for ROAR/TOPK")
    ap.add_argument("--k-cap", type=int, default=2000, help="Max K used in deep TOPK membership check")
    ap.add_argument("--emit-bat", type=int, default=1, help="Write run_autotest.bat next to this script (1/0)")
    args = ap.parse_args()

    env = Env(root=args.root.resolve(), build_dir=HERE, quick=args.quick)

    suite = Suite()

    suite.add(check_posts_parquet_schema(env))
    suite.add(check_post_tags_schema(env))
    suite.add(check_tags_dict(env))
    suite.add(check_mmaps(env))
    suite.add(check_rating_roars(env))
    suite.add(check_tag_bitmaps_against_post_tags(env, sample_tags=args.sample_tags))
    suite.add(check_topk(env, sample_tags=args.sample_tags, k_cap=args.k_cap))
    suite.add(check_pools(env))
    suite.add(check_against_csv(env))

    # reproducibility
    if args.manifest is not None:
        suite.add(reproducibility(env, args.manifest))

    suite.add(perf_profile(env))

    # Emit .bat if requested
    bat = maybe_emit_bat(HERE, bool(args.emit_bat))

    # Render
    report = suite.render()
    print(report)

    if args.report is not None:
        args.report.write_text(report, encoding="utf-8")
        if bat is not None:
            with args.report.open("a", encoding="utf-8") as f:
                f.write("\n\n--- Created launcher: %s ---\n" % bat)

    # Exit code: 0 unless any FAIL
    p, f, w, s = suite.summary()
    return 1 if f > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
