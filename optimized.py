import asyncio
import aiohttp
import time
import json
from memory_profiler import memory_usage

API_URL = "http://localhost:8000/agent/prompt"
PROMPTS = [
    "What does Ethiopian coffee taste like?",
    "What does Brazilian coffee taste like?",
    "What does Colombian coffee taste like?",
]


# ─── OPTIMIZATION 1: Async concurrent requests ───────────────────
async def send_request_async(session, prompt):
    start = time.perf_counter()
    async with session.post(
        API_URL, json={"prompt": prompt}, headers={"Content-Type": "application/json"}
    ) as response:
        result = await response.json()
        elapsed = time.perf_counter() - start
        print(f"  ✓ [{elapsed:.3f}s] {result['response'][:60]}...")
        return elapsed


async def run_concurrent():
    """Send all requests at the same time (optimized)"""
    connector = aiohttp.TCPConnector(limit=10)  # Connection pooling
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [send_request_async(session, p) for p in PROMPTS]
        times = await asyncio.gather(*tasks)
    return list(times)


# ─── OPTIMIZATION 2: Response caching ────────────────────────────
cache = {}


async def send_request_cached(session, prompt):
    if prompt in cache:
        print(f"  ✓ [0.000s] CACHE HIT: {cache[prompt][:60]}...")
        return 0.0
    start = time.perf_counter()
    async with session.post(
        API_URL, json={"prompt": prompt}, headers={"Content-Type": "application/json"}
    ) as response:
        result = await response.json()
        cache[prompt] = result["response"]
        elapsed = time.perf_counter() - start
        print(f"  ✓ [{elapsed:.3f}s] {result['response'][:60]}...")
        return elapsed


async def run_with_cache():
    """Run same prompts twice — second run uses cache"""
    connector = aiohttp.TCPConnector(limit=10)
    async with aiohttp.ClientSession(connector=connector) as session:
        print("  First pass (populating cache):")
        tasks = [send_request_cached(session, p) for p in PROMPTS]
        times1 = await asyncio.gather(*tasks)
        print("\n  Second pass (cache hits):")
        tasks2 = [send_request_cached(session, p) for p in PROMPTS]
        times2 = await asyncio.gather(*tasks2)
    return list(times1), list(times2)


def main():
    print("\n" + "=" * 60)
    print("  COFFEEAGNTCY OPTIMIZED PROFILER")
    print("=" * 60)

    # ── Test 1: Concurrent requests ──
    print("\n[1] CONCURRENT REQUESTS (asyncio + aiohttp)")
    print("-" * 40)
    wall_start = time.perf_counter()
    times = asyncio.run(run_concurrent())
    wall_time = time.perf_counter() - wall_start
    avg = sum(times) / len(times)
    throughput = len(PROMPTS) / wall_time

    print(f"\n  Min:        {min(times):.3f}s")
    print(f"  Max:        {max(times):.3f}s")
    print(f"  Avg:        {avg:.3f}s")
    print(f"  Wall time:  {wall_time:.3f}s  ← all 3 ran in parallel!")
    print(f"  Throughput: {throughput:.3f} req/sec")

    # ── Test 2: Caching ──
    print("\n[2] RESPONSE CACHING")
    print("-" * 40)
    times1, times2 = asyncio.run(run_with_cache())
    print(f"\n  First pass avg:   {sum(times1) / len(times1):.3f}s")
    print(f"  Second pass avg:  {sum(times2) / len(times2):.3f}s  ← near zero!")

    # ── Memory comparison ──
    print("\n[3] MEMORY USAGE (optimized)")
    print("-" * 40)
    mem = memory_usage((asyncio.run, (run_concurrent(),)), interval=0.1)
    print(f"  Peak memory: {max(mem):.2f} MiB")
    print(f"  Avg memory:  {sum(mem) / len(mem):.2f} MiB")

    print("\n" + "=" * 60)
    print("  OPTIMIZATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
