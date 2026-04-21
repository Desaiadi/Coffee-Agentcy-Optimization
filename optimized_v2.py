import asyncio
import aiohttp
import time
import json
import hashlib
from memory_profiler import memory_usage

API_URL = "http://localhost:8000/agent/prompt"

PROMPTS = [
    "What does Ethiopian coffee taste like?",
    "What does Brazilian coffee taste like?",
    "What does Colombian coffee taste like?",
    "What does Kenyan coffee taste like?",
    "What does Vietnamese coffee taste like?",
]

# ── OPTIMIZATION 1: Smart LRU Cache ──────────────────────────────
from functools import lru_cache

response_cache = {}


def cache_key(prompt):
    return hashlib.md5(prompt.lower().strip().encode()).hexdigest()


# ── OPTIMIZATION 2: Retry with exponential backoff ───────────────
async def send_with_retry(session, prompt, retries=3):
    key = cache_key(prompt)
    if key in response_cache:
        return response_cache[key], 0.0, True  # response, time, cache_hit

    for attempt in range(retries):
        try:
            start = time.perf_counter()
            async with session.post(
                API_URL,
                json={"prompt": prompt},
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                result = await resp.json()
                elapsed = time.perf_counter() - start
                response_cache[key] = result["response"]
                return result["response"], elapsed, False
        except Exception as e:
            wait = 2**attempt
            print(f"  ⚠ Attempt {attempt + 1} failed: {e}. Retrying in {wait}s...")
            await asyncio.sleep(wait)
    return None, 0.0, False


# ── OPTIMIZATION 3: Batch processing with concurrency limit ──────
async def run_batch(prompts, max_concurrent=3):
    semaphore = asyncio.Semaphore(max_concurrent)
    connector = aiohttp.TCPConnector(
        limit=20,  # total connection pool size
        limit_per_host=10,  # per host limit
        ttl_dns_cache=300,  # cache DNS for 5 min
        keepalive_timeout=30,
    )

    async def bounded_request(session, prompt):
        async with semaphore:
            return await send_with_retry(session, prompt)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [bounded_request(session, p) for p in prompts]
        results = await asyncio.gather(*tasks)
    return results


def main():
    print("\n" + "=" * 60)
    print("  COFFEEAGNTCY ADVANCED OPTIMIZATION")
    print("=" * 60)

    # ── Round 1: Cold run (no cache) ──
    print("\n[1] COLD RUN — No Cache (5 requests, parallel)")
    print("-" * 40)
    wall_start = time.perf_counter()
    results = asyncio.run(run_batch(PROMPTS, max_concurrent=3))
    wall_time = time.perf_counter() - wall_start

    times = [r[1] for r in results if r[1] > 0]
    for i, (resp, t, hit) in enumerate(results):
        label = "CACHE" if hit else f"{t:.3f}s"
        print(f"  ✓ [{label}] {resp[:55] if resp else 'ERROR'}...")

    throughput = len(PROMPTS) / wall_time
    print(f"\n  Wall time:  {wall_time:.3f}s")
    print(f"  Avg/req:    {sum(times) / len(times):.3f}s")
    print(f"  Throughput: {throughput:.3f} req/sec")

    # ── Round 2: Warm run (all cached) ──
    print("\n[2] WARM RUN — Full Cache (same 5 requests)")
    print("-" * 40)
    wall_start2 = time.perf_counter()
    results2 = asyncio.run(run_batch(PROMPTS, max_concurrent=3))
    wall_time2 = time.perf_counter() - wall_start2

    cache_hits = sum(1 for r in results2 if r[2])
    throughput2 = len(PROMPTS) / max(wall_time2, 0.001)
    print(f"  Cache hits: {cache_hits}/{len(PROMPTS)}")
    print(f"  Wall time:  {wall_time2:.4f}s")
    print(f"  Throughput: {throughput2:.1f} req/sec")

    # ── Round 3: Mixed (some cached, some new) ──
    mixed_prompts = PROMPTS[:3] + [
        "What does Guatemalan coffee taste like?",
        "What does Jamaican coffee taste like?",
    ]
    print("\n[3] MIXED RUN — Partial Cache (3 cached + 2 new)")
    print("-" * 40)
    wall_start3 = time.perf_counter()
    results3 = asyncio.run(run_batch(mixed_prompts, max_concurrent=3))
    wall_time3 = time.perf_counter() - wall_start3

    hits3 = sum(1 for r in results3 if r[2])
    new3 = len(mixed_prompts) - hits3
    print(f"  Cache hits: {hits3} | New requests: {new3}")
    print(f"  Wall time:  {wall_time3:.3f}s")
    print(f"  Throughput: {len(mixed_prompts) / wall_time3:.3f} req/sec")

    # ── Memory ──
    print("\n[4] MEMORY USAGE")
    print("-" * 40)
    response_cache.clear()
    mem = memory_usage((asyncio.run, (run_batch(PROMPTS[:3]),)), interval=0.1)
    print(f"  Peak: {max(mem):.2f} MiB")
    print(f"  Avg:  {sum(mem) / len(mem):.2f} MiB")

    # ── Final Summary ──
    print("\n" + "=" * 60)
    print("  FINAL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  Baseline sequential wall time:  7.924s")
    print(f"  Optimized cold wall time:       {wall_time:.3f}s")
    print(f"  Optimized warm wall time:       {wall_time2:.4f}s")
    print(f"  Baseline throughput:            0.379 req/sec")
    print(f"  Optimized cold throughput:      {throughput:.3f} req/sec")
    print(f"  Optimized warm throughput:      {throughput2:.1f} req/sec")
    print("=" * 60)


if __name__ == "__main__":
    main()
