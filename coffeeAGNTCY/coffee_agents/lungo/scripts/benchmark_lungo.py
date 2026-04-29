"""
Lungo Optimization Benchmark Script — Group 27
===============================================
Measures latency, throughput, memory, and cache behaviour of the
Lungo auction supervisor BEFORE and AFTER the Opt-1 … Opt-9 changes.

Usage (run from the lungo directory with the stack already up):
    python3 scripts/benchmark_lungo.py --url http://localhost:8080 --runs 10

The script sends N requests to the /agent/prompt endpoint, records
wall-time per request, prints a summary table, and writes a JSON report
suitable for copy-pasting into the project presentation.

Requirements:
    pip install aiohttp memory-profiler psutil cachetools
"""

import argparse
import asyncio
import cProfile
import io
import json
import os
import pstats
import time
from dataclasses import dataclass, asdict, field
from typing import Optional

import aiohttp

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_URL = os.getenv("LUNGO_SUPERVISOR_URL", "http://localhost:8080")

# Prompts that exercise different code paths
BENCHMARK_PROMPTS = [
    # inventory — single farm (hits Opt-1, Opt-4, Opt-5, Opt-6, Opt-8, Opt-9)
    "What is the current yield for the Brazil farm?",
    "How much coffee does Colombia have in stock?",
    "What is Vietnam's current coffee yield?",
    # inventory — all farms (hits broadcast path)
    "Give me the inventory for all farms.",
    # repeat single-farm — should be a cache HIT after first run (Opt-4)
    "What is the current yield for the Brazil farm?",
    "What is the current yield for the Brazil farm?",
]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RequestResult:
    prompt: str
    status_code: int
    latency_ms: float
    response_text: str
    error: Optional[str] = None


@dataclass
class BenchmarkReport:
    run_label: str
    url: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    latencies_ms: list[float] = field(default_factory=list)
    avg_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    total_wall_time_s: float = 0.0
    throughput_req_per_s: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate_pct: float = 0.0
    results: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

async def send_prompt(session: aiohttp.ClientSession, url: str, prompt: str) -> RequestResult:
    """POST a single prompt to /agent/prompt and return timing + result."""
    payload = {"prompt": prompt}
    t0 = time.perf_counter()
    try:
        async with session.post(f"{url}/agent/prompt", json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
            latency_ms = (time.perf_counter() - t0) * 1000
            text = await resp.text()
            return RequestResult(
                prompt=prompt,
                status_code=resp.status,
                latency_ms=latency_ms,
                response_text=text[:200],  # truncate for report
            )
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        return RequestResult(
            prompt=prompt,
            status_code=-1,
            latency_ms=latency_ms,
            response_text="",
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

async def run_sequential_benchmark(url: str, prompts: list[str], label: str) -> BenchmarkReport:
    """Send prompts one-by-one (sequential) and collect metrics."""
    print(f"\n{'='*60}")
    print(f"  Sequential benchmark: {label}")
    print(f"  URL: {url}  |  prompts: {len(prompts)}")
    print(f"{'='*60}")

    results: list[RequestResult] = []
    t_wall_start = time.perf_counter()

    async with aiohttp.ClientSession() as session:
        for i, prompt in enumerate(prompts):
            print(f"  [{i+1}/{len(prompts)}] Sending: {prompt[:60]!r}")
            result = await send_prompt(session, url, prompt)
            results.append(result)
            status = "OK" if result.error is None else f"ERR {result.error}"
            print(f"         → {result.latency_ms:.1f} ms  status={result.status_code}  {status}")

    total_wall = time.perf_counter() - t_wall_start

    return _build_report(label, url, results, total_wall)


async def run_concurrent_benchmark(url: str, prompts: list[str], label: str) -> BenchmarkReport:
    """Send all prompts concurrently and collect metrics."""
    print(f"\n{'='*60}")
    print(f"  Concurrent benchmark: {label}")
    print(f"  URL: {url}  |  prompts: {len(prompts)}")
    print(f"{'='*60}")

    t_wall_start = time.perf_counter()

    async with aiohttp.ClientSession() as session:
        tasks = [send_prompt(session, url, p) for p in prompts]
        results: list[RequestResult] = await asyncio.gather(*tasks)

    total_wall = time.perf_counter() - t_wall_start

    for i, r in enumerate(results):
        status = "OK" if r.error is None else f"ERR {r.error}"
        print(f"  [{i+1}] {r.latency_ms:.1f} ms  status={r.status_code}  {status}")

    return _build_report(label, url, results, total_wall)


def _build_report(label: str, url: str, results: list[RequestResult], total_wall: float) -> BenchmarkReport:
    latencies = [r.latency_ms for r in results if r.error is None]
    failed = sum(1 for r in results if r.error is not None)
    successful = len(results) - failed

    sorted_lat = sorted(latencies)
    avg = sum(sorted_lat) / len(sorted_lat) if sorted_lat else 0
    p95 = sorted_lat[int(len(sorted_lat) * 0.95)] if len(sorted_lat) >= 2 else (sorted_lat[0] if sorted_lat else 0)

    report = BenchmarkReport(
        run_label=label,
        url=url,
        total_requests=len(results),
        successful_requests=successful,
        failed_requests=failed,
        latencies_ms=sorted_lat,
        avg_latency_ms=round(avg, 2),
        min_latency_ms=round(min(latencies, default=0), 2),
        max_latency_ms=round(max(latencies, default=0), 2),
        p95_latency_ms=round(p95, 2),
        total_wall_time_s=round(total_wall, 3),
        throughput_req_per_s=round(successful / total_wall, 3) if total_wall > 0 else 0,
        results=[asdict(r) for r in results],
    )
    return report


# ---------------------------------------------------------------------------
# Cache stats fetcher (calls /metrics endpoint if available)
# ---------------------------------------------------------------------------

async def fetch_cache_stats(url: str) -> dict:
    """Try to fetch cache hit/miss stats from the supervisor's /metrics endpoint."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{url}/metrics/cache", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    return await resp.json()
    except Exception:
        pass
    return {}


# ---------------------------------------------------------------------------
# cProfile wrapper
# ---------------------------------------------------------------------------

def profile_sync_wrapper(coro, *args, **kwargs):
    """Run an async coroutine under cProfile and print top-20 hot functions."""
    profiler = cProfile.Profile()
    profiler.enable()
    result = asyncio.run(coro(*args, **kwargs))
    profiler.disable()

    stream = io.StringIO()
    ps = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    ps.print_stats(20)
    print("\n--- cProfile top-20 cumulative (benchmark run) ---")
    print(stream.getvalue())
    return result


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_report(report: BenchmarkReport):
    print(f"\n{'='*60}")
    print(f"  RESULTS: {report.run_label}")
    print(f"{'='*60}")
    print(f"  Total requests    : {report.total_requests}")
    print(f"  Successful        : {report.successful_requests}")
    print(f"  Failed            : {report.failed_requests}")
    print(f"  Avg latency       : {report.avg_latency_ms:.1f} ms")
    print(f"  Min latency       : {report.min_latency_ms:.1f} ms")
    print(f"  Max latency       : {report.max_latency_ms:.1f} ms")
    print(f"  p95 latency       : {report.p95_latency_ms:.1f} ms")
    print(f"  Total wall time   : {report.total_wall_time_s:.2f} s")
    print(f"  Throughput        : {report.throughput_req_per_s:.3f} req/s")
    if report.cache_hits or report.cache_misses:
        total_c = report.cache_hits + report.cache_misses
        print(f"  Cache hits        : {report.cache_hits} / {total_c}  ({report.cache_hit_rate_pct:.1f}%)")
    print(f"{'='*60}\n")


def compare_reports(before: BenchmarkReport, after: BenchmarkReport):
    """Print a side-by-side comparison table for the presentation."""
    def delta(b, a):
        if b == 0:
            return "N/A"
        change = ((a - b) / b) * 100
        sign = "↑" if change > 0 else "↓"
        return f"{sign}{abs(change):.1f}%"

    print(f"\n{'='*65}")
    print("  BEFORE vs AFTER COMPARISON  (Group 27 — Lungo Optimizations)")
    print(f"{'='*65}")
    print(f"  {'Metric':<30} {'Before':>10} {'After':>10} {'Change':>10}")
    print(f"  {'-'*60}")
    rows = [
        ("Avg latency (ms)",     before.avg_latency_ms,         after.avg_latency_ms),
        ("Min latency (ms)",     before.min_latency_ms,         after.min_latency_ms),
        ("Max latency (ms)",     before.max_latency_ms,         after.max_latency_ms),
        ("p95 latency (ms)",     before.p95_latency_ms,         after.p95_latency_ms),
        ("Total wall time (s)",  before.total_wall_time_s,      after.total_wall_time_s),
        ("Throughput (req/s)",   before.throughput_req_per_s,   after.throughput_req_per_s),
    ]
    for name, bv, av in rows:
        d = delta(bv, av)
        print(f"  {name:<30} {bv:>10.2f} {av:>10.2f} {d:>10}")
    print(f"{'='*65}\n")


# ---------------------------------------------------------------------------
# Memory snapshot (psutil)
# ---------------------------------------------------------------------------

def memory_snapshot_mb() -> float:
    """Return current process RSS in MiB (for memory_profiler comparison)."""
    try:
        import psutil, os
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / 1024 / 1024
    except ImportError:
        return -1.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main_async(args):
    url = args.url
    runs = args.runs

    # Build the prompt list: repeat the BENCHMARK_PROMPTS to reach `runs` requests
    prompts = (BENCHMARK_PROMPTS * ((runs // len(BENCHMARK_PROMPTS)) + 1))[:runs]

    mem_before = memory_snapshot_mb()
    print(f"[benchmark] Memory before: {mem_before:.1f} MiB")

    # --- Run 1: Sequential (mirrors Corto's baseline methodology) ---
    seq_report = await run_sequential_benchmark(url, prompts, label="Sequential")
    print_report(seq_report)

    # Fetch cache stats if supervisor exposes them
    cache_stats = await fetch_cache_stats(url)
    if cache_stats:
        seq_report.cache_hits = cache_stats.get("hits", 0)
        seq_report.cache_misses = cache_stats.get("misses", 0)
        seq_report.cache_hit_rate_pct = cache_stats.get("hit_rate_pct", 0.0)
        print(f"  [Opt-4 Cache Stats] {cache_stats}")

    # --- Run 2: Concurrent ---
    conc_report = await run_concurrent_benchmark(url, prompts, label="Concurrent")
    print_report(conc_report)

    # --- Comparison ---
    compare_reports(seq_report, conc_report)

    mem_after = memory_snapshot_mb()
    print(f"[benchmark] Memory after : {mem_after:.1f} MiB  (delta: {mem_after - mem_before:+.1f} MiB)")

    # --- Save JSON report ---
    report_path = args.output
    with open(report_path, "w") as f:
        json.dump(
            {
                "sequential": asdict(seq_report),
                "concurrent": asdict(conc_report),
                "memory_before_mib": round(mem_before, 2),
                "memory_after_mib": round(mem_after, 2),
                "cache_stats": cache_stats,
            },
            f,
            indent=2,
        )
    print(f"[benchmark] Full JSON report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Lungo supervisor benchmark — Group 27")
    parser.add_argument("--url", default=DEFAULT_URL, help="Base URL of the auction supervisor")
    parser.add_argument("--runs", type=int, default=10, help="Number of requests to send")
    parser.add_argument(
        "--output", default="lungo_benchmark_report.json",
        help="Path for the JSON output report",
    )
    parser.add_argument(
        "--profile", action="store_true",
        help="Wrap the sequential run under cProfile (shows hot functions)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Lungo Optimization Benchmark — Group 27")
    print(f"  Target URL : {args.url}")
    print(f"  Requests   : {args.runs}")
    print(f"  cProfile   : {'yes' if args.profile else 'no'}")
    print("=" * 60)

    if args.profile:
        profile_sync_wrapper(main_async, args)
    else:
        asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
