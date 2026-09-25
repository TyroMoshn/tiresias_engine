# build_index/generate_media_artifacts.py
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import polars as pl
from pyroaring import BitMap

def main():
    root = Path("data")
    posts_parquet = root / "posts_parquet"
    mmaps_dir = root / "mmaps"
    bitmaps_dir = root / "bitmaps"

    mmaps_dir.mkdir(parents=True, exist_ok=True)
    bitmaps_dir.mkdir(parents=True, exist_ok=True)

    print(f"[media_artifacts] Reading posts from {posts_parquet}...")
    t0 = time.time()
    scan = pl.scan_parquet(f"{posts_parquet.as_posix()}/**/*.parquet", hive_partitioning=True)
    df = scan.select(["id", "file_ext", "is_deleted"]).collect().sort("id")
    n = len(df)
    print(f"[media_artifacts] Loaded and sorted {n:,} posts in {time.time() - t0:.2f}s")

    # Verify canonical post_ids alignment
    post_ids_bin = mmaps_dir / "post_ids.bin"
    if post_ids_bin.exists():
        canon_ids = np.memmap(post_ids_bin, dtype=np.int64, mode="r")
        df_ids = df["id"].to_numpy().astype(np.int64)
        assert np.array_equal(canon_ids, df_ids), "ERROR: Post IDs order does not match canonical post_ids.bin!"
        print("[media_artifacts] Canonical post_ids.bin alignment verified 100%.")

    # Extension codes:
    # 0: png, 1: jpg, 2: webp  (IMAGES)
    # 3: gif, 4: webm, 5: mp4  (VIDEOS & ANIMATIONS)
    # 6: swf                   (FLASH - BANNED)
    # 7: other                 (BANNED)
    ext_code_map = {
        "png": 0,
        "jpg": 1,
        "jpeg": 1,
        "webp": 2,
        "gif": 3,
        "webm": 4,
        "mp4": 5,
        "swf": 6,
    }

    ext_strings = [str(x).lower().strip() if x is not None else "" for x in df["file_ext"].to_list()]
    ext_codes = np.array([ext_code_map.get(s, 7) for s in ext_strings], dtype=np.uint8)

    # is_deleted boolean array
    is_del_col = df["is_deleted"].to_numpy().astype(bool)

    # Save file_ext.bin memmap
    file_ext_bin = mmaps_dir / "file_ext.bin"
    mm = np.memmap(file_ext_bin, mode="w+", dtype=np.uint8, shape=ext_codes.shape)
    mm[:] = ext_codes
    mm.flush()
    print(f"[media_artifacts] Dumped {file_ext_bin} ({file_ext_bin.stat().st_size:,} bytes)")

    # Build roaring bitmaps
    # 1. not_deleted: is_deleted == False AND file_ext != 'swf' AND file_ext != 'other'
    alive_condition = (~is_del_col) & (ext_codes < 6)
    alive_dense_idx = np.where(alive_condition)[0].astype(np.uint32)
    bm_not_deleted = BitMap(alive_dense_idx.tolist())
    print(f"[media_artifacts] Alive posts (not deleted, non-swf): {len(bm_not_deleted):,}")

    # 2. media_image: file_ext in (0: png, 1: jpg, 2: webp) and alive
    image_condition = (~is_del_col) & (ext_codes <= 2)
    image_dense_idx = np.where(image_condition)[0].astype(np.uint32)
    bm_image = BitMap(image_dense_idx.tolist())
    print(f"[media_artifacts] Images (png, jpg, webp): {len(bm_image):,}")

    # 3. media_video: file_ext in (3: gif, 4: webm, 5: mp4) and alive
    video_condition = (~is_del_col) & (ext_codes >= 3) & (ext_codes <= 5)
    video_dense_idx = np.where(video_condition)[0].astype(np.uint32)
    bm_video = BitMap(video_dense_idx.tolist())
    print(f"[media_artifacts] Videos & GIF (gif, webm, mp4): {len(bm_video):,}")

    # Write roaring bitmaps to both mmaps_dir and bitmaps_dir for robust discovery
    def write_bm(name: str, bm: BitMap):
        data = bm.serialize()
        for d in [mmaps_dir, bitmaps_dir]:
            p = d / f"{name}.roar"
            with open(p, "wb") as f:
                f.write(data)
            print(f"[media_artifacts] Wrote {p} ({len(data):,} bytes)")

    write_bm("not_deleted", bm_not_deleted)
    write_bm("media_image", bm_image)
    write_bm("media_video", bm_video)

    print(f"[media_artifacts] Done in {time.time() - t0:.2f}s!")

if __name__ == "__main__":
    main()
