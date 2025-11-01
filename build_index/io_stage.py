# build_index/io_stage.py
from __future__ import annotations
import csv as pycsv
from pathlib import Path
import duckdb
import polars as pl
from .config import Config
from .utils import ensure_dir, newer_than, log

_EXPECTED = [
    ("id", "CAST(id AS BIGINT)", "CAST(NULL AS BIGINT)"),
    ("uploader_id", "CAST(uploader_id AS BIGINT)", "CAST(NULL AS BIGINT)"),
    ("created_at", "CAST(created_at AS TIMESTAMP)", "CAST(NULL AS TIMESTAMP)"),
    ("md5", "md5", "NULL"),
    ("source", "source", "NULL"),
    ("rating", "rating", "NULL"),
    ("image_width", "CAST(image_width AS INTEGER)", "CAST(NULL AS INTEGER)"),
    ("image_height", "CAST(image_height AS INTEGER)", "CAST(NULL AS INTEGER)"),
    ("tag_string", "tag_string", "''"),
    ("locked_tags", "locked_tags", "''"),
    ("fav_count", "CAST(fav_count AS INTEGER)", "CAST(0 AS INTEGER)"),
    ("file_ext", "file_ext", "NULL"),
    ("parent_id", "NULLIF(parent_id,'')::BIGINT", "CAST(NULL AS BIGINT)"),
    ("change_seq", "CAST(change_seq AS BIGINT)", "CAST(NULL AS BIGINT)"),
    ("approver_id", "NULLIF(approver_id,'')::BIGINT", "CAST(NULL AS BIGINT)"),
    ("file_size", "CAST(file_size AS BIGINT)", "CAST(NULL AS BIGINT)"),
    ("comment_count", "CAST(comment_count AS INTEGER)", "CAST(0 AS INTEGER)"),
    ("description", "description", "''"),
    ("duration", "NULLIF(duration,'')", "NULL"),
    ("updated_at", "CAST(updated_at AS TIMESTAMP)", "CAST(NULL AS TIMESTAMP)"),
    ("is_deleted", "(is_deleted='t')", "CAST(FALSE AS BOOLEAN)"),
    ("is_pending", "(is_pending='t')", "CAST(FALSE AS BOOLEAN)"),
    ("is_flagged", "(is_flagged='t')", "CAST(FALSE AS BOOLEAN)"),
    ("score", "CAST(score AS INTEGER)", "CAST(0 AS INTEGER)"),
    ("up_score", "CAST(up_score AS INTEGER)", "CAST(0 AS INTEGER)"),
    ("down_score", "CAST(down_score AS INTEGER)", "CAST(0 AS INTEGER)"),
    ("is_rating_locked", "(is_rating_locked='t')", "CAST(FALSE AS BOOLEAN)"),
    ("is_status_locked", "(is_status_locked='t')", "CAST(FALSE AS BOOLEAN)"),
    ("is_note_locked", "(is_note_locked='t')", "CAST(FALSE AS BOOLEAN)"),
]

def _csv_header(path: Path) -> set[str]:
    with path.open('r', encoding='utf-8', newline='') as f:
        reader = pycsv.reader(f)
        row = next(reader)
        return {c.strip() for c in row}

def step_parquet(cfg: Config) -> None:
    ensure_dir(cfg.posts_parquet)
    sentinel = cfg.posts_parquet / "_SUCCESS"
    if sentinel.exists() and not cfg.force and newer_than(sentinel, cfg.csv):
        log("[parquet] already fresh - skip")
        return

    log("[parquet] Read CSV and write Parquet with partitioning")
    con = duckdb.connect()
    con.execute("PRAGMA threads=%d" % cfg.workers)
    con.execute("SET timezone='UTC'")

    cols = _csv_header(cfg.csv)
    select_exprs = []
    for name, present_expr, missing_expr in _EXPECTED:
        expr = present_expr if name in cols else f"{missing_expr} AS {name}"
        if name in ("is_deleted","is_pending","is_flagged","is_rating_locked","is_status_locked","is_note_locked") and name not in cols:
            pass
        else:
            if not expr.lower().strip().endswith(f" as {name}"):
                expr = f"{expr} AS {name}"
        select_exprs.append(expr)

    select_exprs.extend([
        "strftime(CAST(created_at AS TIMESTAMP), '%Y') AS year",
        "strftime(CAST(created_at AS TIMESTAMP), '%m') AS month",
    ])
    sel = ",\n              ".join(select_exprs)

    con.execute(
        f"""
        COPY (
            SELECT {sel}
            FROM read_csv_auto('{cfg.csv.as_posix()}', SAMPLE_SIZE=-1, ALL_VARCHAR=1)
        ) TO '{cfg.posts_parquet.as_posix()}' (FORMAT PARQUET,
             PARTITION_BY (rating, year, month), COMPRESSION ZSTD);
        """
    )
    sentinel.write_text("ok")
    log("[parquet] done")