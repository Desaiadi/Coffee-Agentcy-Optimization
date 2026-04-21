# CoffeeAgntcy Optimization — Group 27

> Advanced Python for Data Science | Spring 2026

[![Python](https://img.shields.io/badge/Python-3.14-blue)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-29.0-blue)](https://docker.com)
[![OpenAI](https://img.shields.io/badge/LLM-GPT--4o-green)](https://openai.com)
[![License](https://img.shields.io/badge/License-Apache%202.0-orange)](LICENSE)

---

## Team — Group 27

| Name | Role |
|------|------|
| Deepali Balakrishna Ksheersagar | Profiling & Analysis |
| Aditya Desai | Optimization & Implementation |

---

## Project Overview

This project profiles and optimizes **CoffeeAgntcy** — a real-world distributed multi-agent AI system built on the open-source [AGNTCY](https://github.com/agntcy/coffeeAgntcy) infrastructure.

CoffeeAgntcy simulates a fictitious coffee company where AI agents communicate with each other over a message bus to answer questions about coffee origins, flavors, and profiles.

**Our goal:** Identify performance bottlenecks in this distributed agent workflow and apply Advanced Python techniques to optimize them — then measure and compare the results with real data.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User (Browser UI)                    │
│                  http://localhost:3000                  │
└─────────────────────┬───────────────────────────────────┘
                      │ HTTP POST /agent/prompt
┌─────────────────────▼───────────────────────────────────┐
│           Supervisor Agent — Exchange Server            │
│        (LangGraph orchestrated, port 8000)              │
└─────────────────────┬───────────────────────────────────┘
                      │ A2A Protocol over SLIM message bus
┌─────────────────────▼───────────────────────────────────┐
│             SLIM — Secure Low-Latency                   │
│            Interactive Messaging Bus                    │
│                   (port 46357)                          │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│           Grader Agent — Farm Server                    │
│        (Q Grader Sommelier, LangGraph + GPT-4o)         │
└─────────────────────┬───────────────────────────────────┘
                      │ API Call
┌─────────────────────▼───────────────────────────────────┐
│                  OpenAI GPT-4o API                      │
│              (External — main bottleneck)               │
└─────────────────────────────────────────────────────────┘
```

### All 10 Docker Containers Running

| Container | Purpose |
|-----------|---------|
| `exchange-server` | Supervisor Agent — receives user prompts |
| `farm-server` | Grader Agent — answers using AI |
| `slim` | Message bus between agents |
| `nats` | Backup pub/sub messaging system |
| `clickhouse-server` | Database for conversation history |
| `grafana` | Performance dashboard |
| `otel-collector` | OpenTelemetry data collector |
| `mce-api-layer` | Metrics computation API |
| `metrics-computation-engine` | Processes performance metrics |
| `ui` | Frontend website |

---

## Repository Structure

```
Coffee-Agentcy-Optimization/
└── Version-1/
    ├── README.md                              ← You are here
    ├── profiler.py                            ← Baseline profiling script
    ├── optimized.py                           ← Optimization v1 (asyncio + cache)
    ├── optimized_v2.py                        ← Optimization v2 (advanced)
    └── Presentation/
        └── Group_27_CoffeeAgntcy_Optimized.pptx
```

---

## How to Run This Project

### Prerequisites

Make sure you have these installed:

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.11+ | [python.org](https://python.org) |
| Docker Desktop | Any | [docker.com](https://docker.com) |
| Git | Any | [git-scm.com](https://git-scm.com) |
| OpenAI API Key | — | [platform.openai.com](https://platform.openai.com) |

---

### Step 1 — Clone This Repo

```bash
git clone https://github.com/Desaiadi/Coffee-Agentcy-Optimization.git
cd Coffee-Agentcy-Optimization/coffeeAGNTCY/coffee_agents/corto
cp .env.example .env
```

Open `.env` and add these lines at the bottom:
```env
LLM_MODEL=openai/gpt-4o
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_ENDPOINT=https://api.openai.com/v1
OPENAI_MODEL_NAME=gpt-4o

```

Start all containers:
```bash
docker compose up
```

Wait ~2 minutes, then open `http://localhost:3000`

### Step 2 — Install Python Dependencies

```bash
pip install memory_profiler aiohttp
```

---

### Step 3 — Run Baseline Profiler

```bash
python profiler.py
```

This will output:
- Latency per request (min, max, avg)
- Throughput (requests per second)
- Peak and average memory usage
- cProfile breakdown of all function calls

---

### Step 4 — Run Optimized Version

```bash
python optimized_v2.py
```

This runs three tests:
1. Cold run (no cache) — concurrent requests
2. Warm run (all cached) — instant responses
3. Mixed run — some cached, some new

---

## Phase 1: Profiling — What We Measured

We wrote `profiler.py` using three Python profiling tools:

### Tool 1: `time.perf_counter()`

The most precise timer in Python. Measures exact wall-clock time for each request.

```python
start = time.perf_counter()
result = send_request(prompt)
elapsed = time.perf_counter() - start
print(f"Request took: {elapsed:.3f}s")
```

**Think of it as:** A digital stopwatch with nanosecond precision.

---

### Tool 2: `cProfile`

Records every single Python function that was called, how many times it was called, and how long it ran.

```python
profiler = cProfile.Profile()
profiler.enable()
send_request("What does Ethiopian coffee taste like?")
profiler.disable()

stats = pstats.Stats(profiler)
stats.sort_stats("cumulative")
stats.print_stats(10)
```

**Think of it as:** A security camera that records every step of the program.

---

### Tool 3: `memory_profiler`

Monitors how much RAM the program uses during execution, sampled every 0.1 seconds.

```python
from memory_profiler import memory_usage
mem = memory_usage((send_request, (prompt,)), interval=0.1)
print(f"Peak memory: {max(mem):.2f} MiB")
```

**Think of it as:** A RAM meter that watches your program's memory in real time.

---

### Tool 4: `docker stats`

Measures CPU and memory usage of each Docker container while a request is running.

```bash
docker stats --no-stream
```

---

## Baseline Results (Before Optimization)

### Latency

| Run | Time |
|-----|------|
| Request 1 | 3.001s |
| Request 2 | 2.093s |
| Request 3 | 2.830s |
| **Average** | **2.641s** |
| **Min** | **2.093s** |
| **Max** | **3.001s** |

### Throughput & Memory

| Metric | Value |
|--------|-------|
| Throughput | 0.379 requests/second |
| Peak Memory | 34.34 MiB |
| Avg Memory | 33.37 MiB |

### Container Resources (During Request)

| Container | CPU Usage | Memory |
|-----------|-----------|--------|
| exchange-server | **48.59%** | 711.6 MiB |
| farm-server | 0.00% | 422 MiB |
| slim | 0.00% | 3.97 MiB |
| clickhouse | 5.58% | 577.3 MiB |

### Most Important Finding — cProfile Output

```
560 function calls in 2.107 seconds

ncalls  tottime  percall  cumtime  function
     1    0.000    0.000    2.107    profiler.py:17(send_request)
     1    0.000    0.000    2.106    urllib/request.py(urlopen)
     1    0.000    0.000    2.106    urllib/request.py(open)
     ...
```

**`tottime = 0.000` for ALL Python functions.**

This means Python itself runs in microseconds. The entire 2.1 seconds is spent **waiting for the network** (the OpenAI API response). This told us that making Python code faster wouldn't help — we needed to change **how** we make requests.

---

## Phase 2: Optimizations

### Optimization 1 — Asyncio + Concurrent Requests

#### The Problem

Before optimization, requests ran **sequentially** — one at a time:

```
Timeline:
[Request 1: 2.6s wait...] [Request 2: 2.6s wait...] [Request 3: 2.6s wait...]
Total = 7.9 seconds
```

Python sat idle while waiting for the AI to respond, then started the next request.

#### The Fix

Use Python's `asyncio` (asynchronous programming) to send all requests **at the same time**:

```python
import asyncio
import aiohttp

async def send_request_async(session, prompt):
    async with session.post(API_URL, json={"prompt": prompt}) as resp:
        return await resp.json()

async def run_concurrent():
    async with aiohttp.ClientSession() as session:
        # Create all tasks at once
        tasks = [send_request_async(session, p) for p in prompts]
        # Run them ALL simultaneously
        results = await asyncio.gather(*tasks)
    return results

asyncio.run(run_concurrent())
```

#### The Result

```
Timeline:
[Request 1: 6.5s ─────────────────────────────────►]
[Request 2: 5.8s ───────────────────────────────►  ]  All done in 6.5s!
[Request 3: 6.1s ────────────────────────────────► ]
```

**Wall time: 7.924s → 6.571s = 17% faster**

---

### Optimization 2 — Response Caching

#### The Problem

Every time someone asks *"What does Ethiopian coffee taste like?"*, the system makes a brand new API call to OpenAI, waits 2-3 seconds, and pays for a new API call — even if we already know the answer.

#### The Fix

Store answers in a dictionary (cache). On the second request for the same question, return the stored answer instantly:

```python
import hashlib

cache = {}

def cache_key(prompt):
    # Convert prompt to a unique identifier
    return hashlib.md5(prompt.lower().strip().encode()).hexdigest()

async def send_request_cached(session, prompt):
    key = cache_key(prompt)

    # Check if we already know the answer
    if key in cache:
        print("Cache hit! Returning instantly...")
        return cache[key], 0.0   # ← returns in 0 milliseconds

    # First time seeing this question — ask the AI
    result = await send_request_async(session, prompt)
    cache[key] = result['response']   # ← remember the answer
    return result['response'], elapsed
```

#### The Result

```
First ask:   "Ethiopian coffee taste?" → 2.64 seconds (API call)
Second ask:  "Ethiopian coffee taste?" → 0.000 seconds (cache hit!)
```

**Repeat query speed: 2.64s → 0.000s = 100% faster**
**Cached throughput: 0.379 → 4,701 req/sec = 12,000x improvement**

---

### Optimization 3 — Connection Pooling + Semaphore

#### The Problem

Every HTTP request was opening a brand new TCP connection to the server, using it once, then closing it. This is like picking up a new disposable phone for every phone call instead of keeping one open.

Also, sending too many requests simultaneously could overwhelm the server.

#### The Fix

**Connection Pooling:** Keep 20 connections open and reuse them:

```python
connector = aiohttp.TCPConnector(
    limit=20,              # Keep 20 connections ready
    limit_per_host=10,     # Max 10 to same server
    keepalive_timeout=30,  # Keep connections alive for 30s
    ttl_dns_cache=300      # Cache DNS lookup for 5 minutes
)
```

**Semaphore:** Limit to 3 concurrent requests max to avoid server overload:

```python
semaphore = asyncio.Semaphore(3)

async def bounded_request(session, prompt):
    async with semaphore:   # Only 3 can run at once
        return await send_with_retry(session, prompt)
```

**Retry with Exponential Backoff:** If a request fails, wait and try again:

```python
async def send_with_retry(session, prompt, retries=3):
    for attempt in range(retries):
        try:
            return await send_request_async(session, prompt)
        except Exception:
            wait = 2 ** attempt   # Wait 1s, 2s, 4s...
            await asyncio.sleep(wait)
```

#### The Result

**Throughput: 0.379 → 0.514 req/sec = 36% improvement**

---

## Final Results — Complete Before vs After

### Performance Comparison Table

| Metric | Baseline | Optimized (Cold) | Optimized (Cached) | Best Improvement |
|--------|----------|------------------|--------------------|-----------------|
| Wall Time (3 req) | 7.924s | 6.571s | 0.001s | **↓ 99.99%** |
| Throughput | 0.379 req/s | 0.514 req/s | 4,701 req/s | **↑ 12,000x** |
| Repeat Query Time | 2.641s | 2.641s | 0.000s | **↓ 100%** |
| Avg Memory | 33.37 MiB | 30.17 MiB | 30.17 MiB | **↓ 9.6%** |
| Peak Memory | 34.34 MiB | 38.91 MiB | 38.91 MiB | slight increase |

### Throughput Breakdown

```
Baseline Sequential:   ████░░░░░░░░░░░░░░░░  0.379 req/sec
Optimized Concurrent:  █████░░░░░░░░░░░░░░░  0.514 req/sec  (+36%)
Mixed (partial cache): ██████░░░░░░░░░░░░░░  0.649 req/sec  (+71%)
Full Cache (warm):     ████████████████████  4,701 req/sec  (+12,000x)
```

---

## 💡 Key Lessons Learned

### 1. Profile First, Optimize Second
Without `cProfile`, we might have spent time optimizing Python code that was already running in microseconds. The profiler revealed the real bottleneck was network I/O — completely changing our optimization strategy.

### 2. The Bottleneck Was Not Where We Expected
We expected Python code to be slow. Instead, `tottime = 0.000` for all Python functions proved the code was instant — 99.9% of time was spent waiting for the OpenAI API. This is called an **I/O-bound** program.

### 3. Caching Has the Highest ROI
A simple dictionary cache gave us a **12,000x speedup** for repeat queries with just 5 lines of code. In real production systems, this is implemented with Redis.

### 4. Asyncio Is Essential for I/O-Bound Tasks
When your program spends most of its time waiting (for APIs, databases, files), asyncio lets you do something useful during that wait instead of sitting idle.

### 5. Connection Pooling Reduces Hidden Overhead
Every TCP connection has setup overhead. Reusing connections with `TCPConnector` eliminates this overhead and improves throughput by 36%.

---

## 🔮 Future Work

| Improvement | Description | Expected Gain |
|-------------|-------------|---------------|
| **Redis Cache** | Persistent cache that survives restarts | Same 12,000x but permanent |
| **Auto-scaling** | Multiple farm agent instances | Linear scaling with workers |
| **Numba JIT** | Accelerate serialization code | 10-100x for CPU-bound parts |
| **Request Batching** | Group similar queries together | Reduce API calls by 50%+ |
| **Streaming Responses** | Stream tokens as they generate | Perceived latency ↓ 80% |

---

## Tech Stack

| Technology | Purpose |
|------------|---------|
| Python 3.14 | Main programming language |
| asyncio | Asynchronous concurrent programming |
| aiohttp | Async HTTP client with connection pooling |
| cProfile | Built-in Python function profiler |
| memory_profiler | RAM usage tracking |
| Docker / Docker Compose | Container orchestration |
| LangGraph | Agent workflow orchestration |
| SLIM | Secure agent-to-agent messaging bus |
| OpenAI GPT-4o | Large language model |
| litellm | LLM provider abstraction layer |

---

## References

- [CoffeeAgntcy Repository](https://github.com/agntcy/coffeeAgntcy)
- [AGNTCY Documentation](https://docs.agntcy.org)
- [Python asyncio docs](https://docs.python.org/3/library/asyncio.html)
- [aiohttp Documentation](https://docs.aiohttp.org)
- [memory_profiler](https://github.com/pythonprofilers/memory_profiler)
- [cProfile docs](https://docs.python.org/3/library/profile.html)

---

*Group 27 — Advanced Python for Data Science — Spring 2026*
