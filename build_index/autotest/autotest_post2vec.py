#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import polars as pl

try:
    import faiss  # type: ignore
except ImportError:
    faiss = None  # type: ignore


@dataclass
class CheckResult:
    name: str
    status: str
    details: List[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [f"[{self.status}] {self.name}"]
        lines.extend(f"  - {line}" for line in self.details)
        return "\n".join(lines)


class Suite:
    def __init__(self) -> None:
        self.results: List[CheckResult] = []

    def add(self, result: CheckResult) -> None:
        self.results.append(result)

    def has_failures(self) -> bool:
        return any(r.status == "FAIL" for r in self.results)

    def summary(self) -> str:
        total = len(self.results)
        fail = sum(r.status == "FAIL" for r in self.results)
        warn = sum(r.status == "WARN" for r in self.results)
        skip = sum(r.status == "SKIP" for r in self.results)
        return f"Total={total}  FAIL={fail}  WARN={warn}  SKIP={skip}"

    def render(self) -> str:
        return "\n".join(r.render() for r in self.results)


def to_int(value: object, default: int = 0) -> int:
    return int(value) if value is not None else default


class QueryError(RuntimeError):
    pass


def ensure_array(value: Any) -> np.ndarray:
    if isinstance(value, pl.Series):
        return np.asarray(value.to_list(), dtype=np.float32)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=np.float32)
    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False)
    try:
        return np.asarray(value, dtype=np.float32)
    except Exception as exc:
        raise QueryError(f"Unsupported vector payload type: {type(value)!r}") from exc


def scan_stats(path: Path, column: str) -> Tuple[int, int, int]:
    lf = pl.scan_parquet(path.as_posix())
    agg = lf.select(
        pl.len().alias("rows"),
        pl.col(column).n_unique().alias("unique"),
        pl.col(column).hash(0).sum().alias("hash_sum"),
    ).collect(engine="streaming")
    rows = to_int(agg[0, 0])
    unique = to_int(agg[0, 1])
    hash_sum = to_int(agg[0, 2])
    return rows, unique, hash_sum


def inspect_post2vec(vec_path: Path) -> Tuple[CheckResult, Dict[str, int | str]]:
    if not vec_path.exists():
        return CheckResult(
            "post2vec.parquet presence",
            "FAIL",
            [f"Missing file: {vec_path}"],
        ), {}

    schema = pl.read_parquet(vec_path, n_rows=0).schema
    rows, unique, hash_sum = scan_stats(vec_path, "post_id")

    # Probe a single vector to infer dimension and confirm dtype
    dim = 0
    vec_dtype = "unknown"
    if rows > 0:
        sample = pl.read_parquet(vec_path, columns=["vec"], n_rows=1)
        series = sample["vec"][0]
        dim = len(series)
        vec_dtype = str(series.dtype) if hasattr(series, "dtype") else "list"

    details = [
        f"rows={rows:,}",
        f"unique_post_id={unique:,}",
        f"schema={schema}",
        f"vector_dim={dim}",
        f"vector_dtype={vec_dtype}",
    ]
    status = "PASS" if rows == unique and dim > 0 else "WARN"
    return CheckResult("post2vec.parquet integrity", status, details), {
        "rows": rows,
        "unique": unique,
        "hash": hash_sum,
        "dim": dim,
    }


def inspect_ids(ids_path: Path) -> Tuple[CheckResult, Dict[str, int]]:
    if not ids_path.exists():
        return CheckResult(
            "post2vec_faiss_ids.parquet presence",
            "FAIL",
            [f"Missing file: {ids_path}"],
        ), {}

    lf = pl.scan_parquet(ids_path.as_posix())
    agg = lf.select(
        pl.len().alias("rows"),
        pl.col("row").min().alias("min_row"),
        pl.col("row").max().alias("max_row"),
        pl.col("row").n_unique().alias("unique_row"),
        pl.col("post_id").n_unique().alias("unique_post"),
        pl.col("post_id").hash(0).sum().alias("hash_sum"),
    ).collect(engine="streaming")
    rows = to_int(agg[0, 0])
    min_row = to_int(agg[0, 1], default=-1)
    max_row = to_int(agg[0, 2], default=-1)
    unique_row = to_int(agg[0, 3])
    unique_post = to_int(agg[0, 4])
    hash_sum = to_int(agg[0, 5])
    details = [
        f"rows={rows:,}",
        f"row_range=[{min_row}, {max_row}]",
        f"unique_rows={unique_row:,}",
        f"unique_post_id={unique_post:,}",
    ]
    if rows == 0:
        status = "WARN"
    elif min_row == 0 and max_row == rows - 1 and unique_row == rows:
        status = "PASS"
    else:
        status = "FAIL"
        details.append("Row index sequence is not continuous.")
    return CheckResult("post2vec_faiss_ids integrity", status, details), {
        "rows": rows,
        "hash": hash_sum,
        "unique_post": unique_post,
    }


def inspect_index(idx_path: Path, expected_rows: int, expected_dim: int) -> Tuple[CheckResult, Any]:
    if not idx_path.exists():
        return CheckResult(
            "post2vec_faiss.index presence",
            "FAIL",
            [f"Missing file: {idx_path}"],
        ), None
    if faiss is None:
        return CheckResult(
            "FAISS index integrity",
            "SKIP",
            ["faiss module not available; skipping index verification."],
        ), None
    idx = faiss.read_index(idx_path.as_posix())
    ntotal = int(idx.ntotal)
    dim = int(idx.d)
    details = [
        f"ntotal={ntotal:,}",
        f"dim={dim}",
        f"type={idx.__class__.__name__}",
    ]
    if ntotal != expected_rows:
        details.append(f"Expected {expected_rows:,} vectors.")
        return CheckResult("FAISS index size", "FAIL", details), idx
    if expected_dim > 0 and dim != expected_dim:
        details.append(f"Expected dim={expected_dim}.")
        return CheckResult("FAISS index dimension", "FAIL", details), idx
    return CheckResult("FAISS index integrity", "PASS", details), idx


def load_sample_rows(ids_path: Path, rows: Sequence[int]) -> Dict[int, int]:
    if not rows:
        return {}
    lf = pl.scan_parquet(ids_path.as_posix()).filter(pl.col("row").is_in(rows))
    df = lf.collect(engine="streaming")
    return {int(rec["row"]): int(rec["post_id"]) for rec in df.iter_rows(named=True)}


def load_vectors(vec_path: Path, post_ids: Iterable[int]) -> Dict[int, np.ndarray]:
    ids = list(set(int(pid) for pid in post_ids))
    if not ids:
        return {}
    lf = pl.scan_parquet(vec_path.as_posix()).filter(pl.col("post_id").is_in(ids))
    df = lf.collect(engine="streaming")
    vectors: Dict[int, np.ndarray] = {}
    for rec in df.iter_rows(named=True):
        pid = int(rec["post_id"])
        raw = rec["vec"]
        vec = np.asarray(raw, dtype=np.float32)
        vectors[pid] = vec
    return vectors


def fetch_post_vector(vec_path: Path, post_id: int) -> np.ndarray:
    lf = (
        pl.scan_parquet(vec_path.as_posix())
        .filter(pl.col("post_id") == post_id)
        .select(["post_id", "vec"])
        .limit(1)
    )
    df = lf.collect(engine="streaming")
    if df.height == 0:
        raise QueryError(f"post_id={post_id} not found in post2vec.parquet")
    rec = next(df.iter_rows(named=True))
    vec = ensure_array(rec["vec"])
    if vec.size == 0:
        raise QueryError(f"post_id={post_id} vector is empty.")
    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        vec = vec / norm
    return vec.astype(np.float32, copy=False)


def find_post_row(ids_path: Path, post_id: int) -> int:
    lf = (
        pl.scan_parquet(ids_path.as_posix())
        .filter(pl.col("post_id") == post_id)
        .select(pl.col("row"))
        .limit(1)
    )
    df = lf.collect(engine="streaming")
    if df.height == 0:
        raise QueryError(f"post_id={post_id} not present in post2vec_faiss_ids.parquet")
    return to_int(df[0, 0], default=-1)


def search_similar_rows(index: Any, query_vec: np.ndarray, topk: int) -> Tuple[List[int], List[float]]:
    if faiss is None:
        raise QueryError("faiss module not available; cannot search for similar posts.")
    if topk <= 0:
        raise QueryError("topk must be positive.")
    q = query_vec.reshape(1, -1).astype(np.float32, copy=False)
    distances, rows = index.search(q, topk)
    row_list = rows[0].tolist()
    dist_list = distances[0].tolist()
    results: List[int] = []
    sims: List[float] = []
    for r, d in zip(row_list, dist_list):
        if r < 0:
            continue
        results.append(int(r))
        sims.append(float(d))
    return results, sims


def load_post_tags(root: Path, post_ids: Iterable[int]) -> Dict[int, List[int]]:
    ids = list(set(int(pid) for pid in post_ids))
    if not ids:
        return {}
    base = root / "post_tags_parquet"
    lf = (
        pl.scan_parquet(f"{base.as_posix()}/**/*.parquet")
        .filter(pl.col("post_id").is_in(ids))
        .group_by("post_id")
        .agg(pl.col("tag_id").unique().cast(pl.Int32))
    )
    df = lf.collect(engine="streaming")
    tags: Dict[int, List[int]] = {}
    for rec in df.iter_rows(named=True):
        pid = int(rec["post_id"])
        tags[pid] = [int(t) for t in rec["tag_id"]]
    return tags


def load_tag_vectors(features_dir: Path, tag_ids: Iterable[int]) -> Dict[int, np.ndarray]:
    ids = list(set(int(t) for t in tag_ids))
    if not ids:
        return {}
    vec_path = features_dir / "tag2vec.parquet"
    lf = pl.scan_parquet(vec_path.as_posix()).filter(pl.col("tag_id").is_in(ids))
    df = lf.collect(engine="streaming")
    out: Dict[int, np.ndarray] = {}
    for rec in df.iter_rows(named=True):
        tag_id = int(rec["tag_id"])
        vec = np.asarray(rec["vec"], dtype=np.float32)
        out[tag_id] = vec
    return out


def load_idf(root: Path, tag_ids: Iterable[int]) -> Dict[int, float]:
    ids = list(set(int(t) for t in tag_ids))
    if not ids:
        return {}
    tags_parquet = root / "tags.parquet"
    if tags_parquet.exists():
        lf = pl.scan_parquet(tags_parquet.as_posix()).filter(pl.col("tag_id").is_in(ids))
        df = lf.select(
            pl.col("tag_id").cast(pl.Int32).alias("tag_id"),
            pl.coalesce([pl.col("idf"), pl.lit(1.0)]).cast(pl.Float64).alias("idf"),
        ).collect(engine="streaming")
    else:
        alt = root / "tags_dict.parquet"
        lf = pl.scan_parquet(alt.as_posix()).filter(pl.col("tag_id").is_in(ids))
        df = lf.select(pl.col("tag_id").cast(pl.Int32), pl.lit(1.0).alias("idf")).collect(engine="streaming")
    return {int(rec["tag_id"]): float(rec["idf"]) for rec in df.iter_rows(named=True)}


def compute_expected_vectors(
    post_tags: Dict[int, List[int]],
    tag_vecs: Dict[int, np.ndarray],
    idf: Dict[int, float],
    dim: int,
) -> Dict[int, np.ndarray]:
    expected: Dict[int, np.ndarray] = {}
    for pid, tags in post_tags.items():
        vec = np.zeros((dim,), dtype=np.float32)
        for tag_id in tags:
            tv = tag_vecs.get(tag_id)
            if tv is None or tv.size == 0:
                continue
            w = float(idf.get(tag_id, 1.0))
            if w != 0.0:
                vec += w * tv
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        expected[pid] = vec
    return expected


def verify_samples(
    root: Path,
    vec_path: Path,
    ids_path: Path,
    index: Any,
    row_count: int,
    dim: int,
    sample_size: int,
    tolerance: float,
) -> CheckResult:
    if row_count == 0 or sample_size <= 0:
        return CheckResult("Sample vector verification", "WARN", ["Dataset is empty; skip sample checks."])
    if dim <= 0:
        return CheckResult("Sample vector verification", "WARN", ["Unknown vector dimension; skip sample checks."])

    step = max(row_count // sample_size, 1)
    target_rows = {min(row_count - 1, i) for i in range(0, row_count, step)}
    target_rows.add(row_count - 1)
    target_rows = sorted(target_rows)
    target_rows = target_rows[:sample_size]

    row_to_post = load_sample_rows(ids_path, target_rows)
    if len(row_to_post) != len(target_rows):
        missing = set(target_rows) - set(row_to_post)
        return CheckResult(
            "Sample vector verification",
            "FAIL",
            [f"Failed to load mapping for rows: {sorted(missing)}"],
        )

    vectors = load_vectors(vec_path, row_to_post.values())
    if len(vectors) != len(row_to_post):
        missing_posts = sorted(set(row_to_post.values()) - set(vectors))
        return CheckResult(
            "Sample vector verification",
            "FAIL",
            [f"post2vec.parquet missing vectors for post_ids: {missing_posts}"],
        )

    post_tags = load_post_tags(root, row_to_post.values())
    if not post_tags:
        return CheckResult(
            "Sample vector verification",
            "WARN",
            ["Could not load post tags for the sampled posts."],
        )

    sample_tag_ids = {tag for tags in post_tags.values() for tag in tags}
    tag_vecs = load_tag_vectors(root / "features", sample_tag_ids)
    idf = load_idf(root, sample_tag_ids)
    expected_vectors = compute_expected_vectors(post_tags, tag_vecs, idf, dim)

    issues: List[str] = []
    for row_id in target_rows:
        post_id = row_to_post[row_id]
        actual = vectors.get(post_id)
        if actual is None:
            issues.append(f"Missing actual vector for post_id={post_id}")
            continue
        expected = expected_vectors.get(post_id)
        if expected is None:
            issues.append(f"Missing expected vector for post_id={post_id}")
            continue

        diff = float(np.linalg.norm(actual - expected))
        if diff > tolerance:
            issues.append(f"post_id={post_id} diff={diff:.2e} exceeds tolerance {tolerance:g}")

        # Verify FAISS index alignment
        q = actual.reshape(1, -1).astype(np.float32)
        distances, indices = index.search(q, 1)
        top_row = int(indices[0, 0])
        distance = float(distances[0, 0])
        if top_row != row_id:
            issues.append(f"FAISS index mismatch: row={row_id} -> post_id={post_id}, but search returned row={top_row}")
        if not math.isfinite(distance) or distance < 0 or distance > 1.0005:
            issues.append(f"FAISS self-similarity abnormal for post_id={post_id}: {distance:.4f}")

    status = "PASS" if not issues else "FAIL"
    detail = [f"sample_size={len(target_rows)}", f"tolerance={tolerance}"]
    detail.extend(issues)
    return CheckResult("Sample vector verification", status, detail)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate post2vec artifacts.")
    ap.add_argument("--root-data", type=Path, default=Path(r"I:\TIRESIAS_ENGINE\data"), help="Path to data directory.")
    ap.add_argument("--sample", type=int, default=5, help="Number of evenly spaced samples to validate.")
    ap.add_argument("--tolerance", type=float, default=5e-4, help="L2 tolerance for vector comparison.")
    ap.add_argument("--query-post", type=int, default=None, help="Post ID to lookup similar posts.")
    ap.add_argument("--topk", type=int, default=10, help="Number of similar posts to return when querying.")
    ap.add_argument("--include-query", action="store_true", help="Include the query post itself in the neighbor list.")
    args = ap.parse_args(argv)

    root = args.root_data.resolve()
    features = root / "features"
    vec_path = features / "post2vec.parquet"
    ids_path = features / "post2vec_faiss_ids.parquet"
    idx_path = features / "post2vec_faiss.index"

    suite = Suite()

    post2vec_res, vec_meta = inspect_post2vec(vec_path)
    suite.add(post2vec_res)

    ids_res, ids_meta = inspect_ids(ids_path)
    suite.add(ids_res)

    if vec_meta and ids_meta:
        if vec_meta["rows"] != ids_meta["rows"]:
            suite.add(CheckResult(
                "Row count agreement",
                "FAIL",
                [f"post2vec rows={vec_meta['rows']:,} vs ids rows={ids_meta['rows']:,}"],
            ))
        elif vec_meta["hash"] != ids_meta["hash"]:
            suite.add(CheckResult(
                "Post ID hash agreement",
                "WARN",
                ["Hash mismatch between post2vec and faiss ids (possible differing sets)."],
            ))
        else:
            suite.add(CheckResult(
                "Row count agreement",
                "PASS",
                [f"rows={vec_meta['rows']:,}"],
            ))

    expected_rows = int(vec_meta["rows"]) if "rows" in vec_meta else 0
    expected_dim = int(vec_meta["dim"]) if "dim" in vec_meta else 0
    index_res, index_obj = inspect_index(idx_path, expected_rows, expected_dim)
    suite.add(index_res)

    if expected_rows > 0 and index_obj is not None:
        suite.add(
            verify_samples(
                root=root,
                vec_path=vec_path,
                ids_path=ids_path,
                index=index_obj,
                row_count=expected_rows,
                dim=expected_dim,
                sample_size=args.sample,
                tolerance=args.tolerance,
            )
        )
    else:
        if expected_rows > 0 and faiss is None:
            suite.add(CheckResult(
                "Sample vector verification",
                "SKIP",
                ["faiss module not available; skipping sample verification."],
            ))
        else:
            suite.add(CheckResult(
                "Sample vector verification",
                "WARN",
                ["No vectors available; skipped sample verification."],
            ))

    print("=" * 72)
    print("POST2VEC AUTOTEST")
    print("=" * 72)
    print(suite.render())
    print("-" * 72)
    print(suite.summary())

    query_error = False
    if args.query_post is not None:
        print("\n" + "=" * 72)
        print(f"SIMILAR POSTS FOR post_id={args.query_post}")
        print("=" * 72)
        try:
            if index_obj is None:
                raise QueryError("FAISS index unavailable; run with faiss installed to query similar posts.")
            query_vec = fetch_post_vector(vec_path, args.query_post)
            query_row = find_post_row(ids_path, args.query_post)
            search_k = args.topk if args.include_query else args.topk + 1
            rows, sims = search_similar_rows(index_obj, query_vec, search_k)
            filtered: List[Tuple[int, float]] = []
            for row_id, score in zip(rows, sims):
                if not args.include_query and row_id == query_row:
                    continue
                filtered.append((row_id, score))
                if len(filtered) >= args.topk:
                    break
            if not filtered:
                raise QueryError("No neighbors returned from FAISS index.")
            row_map = load_sample_rows(ids_path, [r for r, _ in filtered])
            lines: List[str] = []
            for rank, (row_id, score) in enumerate(filtered, start=1):
                post_id = row_map.get(row_id)
                if post_id is None:
                    raise QueryError(f"Missing mapping for row={row_id} in post2vec_faiss_ids.parquet")
                lines.append(f"{rank:>2}. post_id={post_id}  row={row_id}  sim={score:.4f}")
            print("\n".join(lines))
        except QueryError as exc:
            print(f"[QUERY] {exc}")
            query_error = True
        except Exception as exc:  # pragma: no cover
            print(f"[QUERY] Unexpected error: {exc}")
            query_error = True

    exit_code = 1 if suite.has_failures() else 0
    if query_error:
        exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
