import cProfile
import pstats
import time
import urllib.request
import urllib.error
import json
from memory_profiler import memory_usage

API_URL = "http://localhost:8000/agent/prompt"
PROMPTS = [
    "What does Ethiopian coffee taste like?",
    "What does Brazilian coffee taste like?",
    "What does Colombian coffee taste like?",
]


def send_request(prompt):
    data = json.dumps({"prompt": prompt}).encode("utf-8")
    req = urllib.request.Request(
        API_URL, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode("utf-8"))


def run_sequential():
    """Run all prompts one by one (baseline)"""
    times = []
    for prompt in PROMPTS:
        start = time.perf_counter()
        result = send_request(prompt)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  ✓ [{elapsed:.3f}s] {result['response'][:60]}...")
    return times


def main():
    print("\n" + "=" * 60)
    print("  COFFEEAGNTCY BASELINE PROFILER")
    print("=" * 60)

    # 1. Latency profiling
    print("\n[1] LATENCY TEST (Sequential Requests)")
    print("-" * 40)
    times = run_sequential()
    avg = sum(times) / len(times)
    print(f"\n  Min:     {min(times):.3f}s")
    print(f"  Max:     {max(times):.3f}s")
    print(f"  Avg:     {avg:.3f}s")
    print(f"  Total:   {sum(times):.3f}s")

    # 2. Throughput
    print("\n[2] THROUGHPUT")
    print("-" * 40)
    throughput = len(PROMPTS) / sum(times)
    print(f"  {throughput:.3f} requests/second")

    # 3. Memory profiling
    print("\n[3] MEMORY USAGE")
    print("-" * 40)
    mem = memory_usage(
        (send_request, ("What does Kenyan coffee taste like?",)), interval=0.1
    )
    print(f"  Peak memory: {max(mem):.2f} MiB")
    print(f"  Avg memory:  {sum(mem) / len(mem):.2f} MiB")

    # 4. cProfile
    print("\n[4] CPROFILE (Top 10 functions by time)")
    print("-" * 40)
    profiler = cProfile.Profile()
    profiler.enable()
    send_request("What does Kenyan coffee taste like?")
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    stats.print_stats(10)

    print("\n" + "=" * 60)
    print("  PROFILING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
