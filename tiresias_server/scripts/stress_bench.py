#!/usr/bin/env python3
"""
High-performance asynchronous benchmark and stress-testing suite for TIRESIAS Serving API.
Evaluates latency (p50, p90, p95, p99), throughput (RPS), memory RSS, and leak stability.
Supports testing both local Windows server and remote Linux Lite laptop / VPS.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
import time
from typing import Any, Dict, List, Optional

import httpx
import numpy as np

# Typical seed post IDs for testing
SAMPLE_POST_IDS = [
    5353063,
    5355177,
    5341735,
    5372730,
    5371661,
    5368940,
    5365510,
    5362900,
    5359480,
    5357120,
    4819201,
    4820100,
]


async def fetch_health(client: httpx.AsyncClient, base_url: str) -> Optional[Dict[str, Any]]:
    try:
        r = await client.get(f"{base_url}/api/v1/system/health", timeout=5.0)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


async def run_worker(
    worker_id: int,
    base_url: str,
    queue: asyncio.Queue,
    latencies: List[float],
    status_codes: Dict[int, int],
    test_board_id: Optional[str],
) -> None:
    async with httpx.AsyncClient(timeout=10.0) as client:
        while not queue.empty():
            req_type = await queue.get()
            t0 = time.perf_counter()
            code = 0
            try:
                if req_type == "feed":
                    user = f"bench_user_{random.randint(0, 20)}"
                    r = await client.post(
                        f"{base_url}/api/v1/recommend/feed",
                        json={"user_id": user, "limit": 30, "ratings": ["s", "q"]},
                    )
                    code = r.status_code
                elif req_type == "similar":
                    pid = random.choice(SAMPLE_POST_IDS)
                    r = await client.get(
                        f"{base_url}/api/v1/recommend/similar?post_id={pid}&limit=20"
                    )
                    code = r.status_code
                elif req_type == "board_rec" and test_board_id:
                    r = await client.get(
                        f"{base_url}/api/v1/boards/{test_board_id}/recommend?limit=20"
                    )
                    code = r.status_code
                elif req_type == "feedback":
                    user = f"bench_user_{random.randint(0, 20)}"
                    pid = random.choice(SAMPLE_POST_IDS)
                    sig = random.choice(["like", "seen"])
                    if sig == "like":
                        r = await client.post(
                            f"{base_url}/api/v1/feedback",
                            json={"user_id": user, "post_id": pid, "signal_type": "like"},
                        )
                    else:
                        r = await client.post(
                            f"{base_url}/api/v1/feedback/seen",
                            json={"user_id": user, "post_ids": [pid]},
                        )
                    code = r.status_code
                else:
                    # fallback to health
                    r = await client.get(f"{base_url}/api/v1/system/health")
                    code = r.status_code
            except Exception:
                code = -1
            finally:
                dt_ms = (time.perf_counter() - t0) * 1000.0
                latencies.append(dt_ms)
                status_codes[code] = status_codes.get(code, 0) + 1
                queue.task_done()


async def run_benchmark(
    base_url: str,
    total_requests: int = 500,
    concurrency: int = 10,
    is_leak_test: bool = False,
) -> int:
    base_url = base_url.rstrip("/")
    print("=" * 65)
    print(f"TIRESIAS Serving API Benchmark: {base_url}")
    print(f"Concurrency: {concurrency} | Total Requests: {total_requests}")
    print(f"Mode: {'Memory Leak Probe' if is_leak_test else 'Standard Stress Test'}")
    print("=" * 65)

    async with httpx.AsyncClient(timeout=10.0) as client:
        initial_health = await fetch_health(client, base_url)
        if not initial_health:
            print(f"[FAIL] Could not connect to API at {base_url}. Is the server running?")
            return 1

        print("[STATUS] Target Server Initial State:")
        print(f"  - Status:         {initial_health.get('status')}")
        print(
            f"  - FAISS SQ8:      {'READY' if initial_health.get('faiss_ready') else 'NOT LOADED'} ({initial_health.get('faiss_total_vectors', 0):,} vectors)"
        )
        print(
            f"  - Mmaps:          {initial_health.get('total_posts_indexed', 0):,} posts indexed"
        )
        print(f"  - Memory RSS:     {initial_health.get('memory_rss_mb', 0):.1f} MB")
        print("-" * 65)

        # Create a temporary board for testing board recommendations
        test_board_id = None
        try:
            b_res = await client.post(
                f"{base_url}/api/v1/boards",
                json={"user_id": "bench_tester", "name": "Benchmark Temp Board"},
            )
            if b_res.status_code == 201:
                test_board_id = b_res.json()["board_id"]
                await client.post(
                    f"{base_url}/api/v1/boards/{test_board_id}/posts",
                    json={"post_ids": [5353063, 5355177]},
                )
        except Exception:
            test_board_id = None

    # Populate request queue with realistic mix
    queue: asyncio.Queue = asyncio.Queue()
    for _ in range(total_requests):
        rand = random.random()
        if rand < 0.40:
            queue.put_nowait("feed")
        elif rand < 0.70:
            queue.put_nowait("similar")
        elif rand < 0.90:
            queue.put_nowait("board_rec" if test_board_id else "feed")
        else:
            queue.put_nowait("feedback")

    latencies: List[float] = []
    status_codes: Dict[int, int] = {}

    print("Executing benchmark...")
    t_start = time.perf_counter()

    workers = [
        asyncio.create_task(
            run_worker(
                i,
                base_url,
                queue,
                latencies,
                status_codes,
                test_board_id,
            )
        )
        for i in range(concurrency)
    ]

    await queue.join()
    for w in workers:
        w.cancel()

    t_total = time.perf_counter() - t_start

    # Final health check
    async with httpx.AsyncClient(timeout=10.0) as client:
        final_health = await fetch_health(client, base_url)

    # Print results
    print("-" * 65)
    print("BENCHMARK METRICS SUMMARY")
    print("-" * 65)
    print(f"Total Requests:     {len(latencies)}")
    print(f"Duration:           {t_total:.2f} s")
    rps = len(latencies) / max(0.01, t_total)
    print(f"Throughput:         {rps:.1f} req/s (RPS)")
    print(f"Status Codes:       {status_codes}")

    if latencies:
        mean_lat = float(np.mean(latencies))
        p50 = float(np.percentile(latencies, 50))
        p90 = float(np.percentile(latencies, 90))
        p95 = float(np.percentile(latencies, 95))
        p99 = float(np.percentile(latencies, 99))
        max_lat = float(np.max(latencies))

        print(f"\nLatency Percentiles (ms):")
        print(f"  - Mean:           {mean_lat:6.2f} ms")
        print(f"  - p50 (Median):   {p50:6.2f} ms")
        print(f"  - p90:            {p90:6.2f} ms")
        print(f"  - p95:            {p95:6.2f} ms (Target < 50 ms)")
        print(f"  - p99:            {p99:6.2f} ms")
        print(f"  - Max:            {max_lat:6.2f} ms")

    if initial_health and final_health:
        rss_init = initial_health.get("memory_rss_mb", 0)
        rss_final = final_health.get("memory_rss_mb", 0)
        delta = rss_final - rss_init
        print(f"\nMemory Stability:")
        print(f"  - Initial RSS:    {rss_init:.1f} MB")
        print(f"  - Final RSS:      {rss_final:.1f} MB")
        print(f"  - Delta RSS:      {delta:+.1f} MB")
        if is_leak_test:
            if delta < 35.0:
                print(f"  [PASS] Memory leak check PASSED (Delta < 35 MB over {total_requests} requests)")
            else:
                print(f"  [WARN] Potential memory creep detected: Delta is {delta:.1f} MB")

    print("=" * 65)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="TIRESIAS Serving API Stress Benchmark")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base server URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "-n", "--requests", type=int, default=500, help="Total requests to send"
    )
    parser.add_argument(
        "-c", "--concurrency", type=int, default=10, help="Concurrent workers"
    )
    parser.add_argument(
        "--leak-test",
        action="store_true",
        help="Run longer sequential memory leak probe (2000 reqs, concurrency 2)",
    )

    args = parser.parse_args()
    reqs = 2000 if args.leak_test else args.requests
    conc = 2 if args.leak_test else args.concurrency

    return asyncio.run(
        run_benchmark(
            base_url=args.url,
            total_requests=reqs,
            concurrency=conc,
            is_leak_test=args.leak_test,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
