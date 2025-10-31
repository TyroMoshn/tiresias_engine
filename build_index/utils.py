# build_index/utils.py
from __future__ import annotations
from datetime import datetime
from pathlib import Path

def log(msg: str) -> None:
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] {msg}")

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def newer_than(out: Path, *ins: Path) -> bool:
    if not out.exists():
        return False
    out_m = out.stat().st_mtime
    return all(out_m >= i.stat().st_mtime for i in ins if i and i.exists())

# ---- varint / delta (для Top-K) ----
def encode_varint(n: int) -> bytes:
    out = bytearray()
    while True:
        to_write = n & 0x7F
        n >>= 7
        if n:
            out.append(to_write | 0x80)
        else:
            out.append(to_write)
            break
    return bytes(out)

def encode_varint_delta(sorted_ids: list[int]) -> bytes:
    prev = 0
    buf = bytearray()
    for x in sorted_ids:
        d = x - prev
        buf += encode_varint(d)
        prev = x
    return bytes(buf)