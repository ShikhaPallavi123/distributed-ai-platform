"""Bounded multi-replica LLM experiment via authorized private SSH tunnels.

One model replica per GPU, not tensor/model parallelism. Standard-library
only; no processes, tunnels or servers started by this script.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import threading
import time
from urllib.request import Request, build_opener, HTTPRedirectHandler

PROMPTS = (
    "Explain what a computer is in two short sentences.",
    "Describe the purpose of a database in two short sentences.",
    "Explain why testing software is useful in two short sentences.",
    "Explain the difference between RAM and storage in two short sentences.",
    "Explain what a network is in two short sentences.",
    "Explain what an algorithm is in two short sentences.",
    "Explain what a GPU does in two short sentences.",
    "Explain why backups are useful in two short sentences.",
)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Endpoint:
    def __init__(self, label, port, key, model="qwen"):
        if not 1024 <= port <= 65535:
            raise ValueError("invalid local port")
        self.label = label
        self.url = "http://127.0.0.1:" + str(port) + "/v1/chat/completions"
        self.key = key
        self.model = model

    def generate(self, prompt):
        data = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                "temperature": 0, "max_tokens": 64, "stream": False}
        req = Request(self.url, data=json.dumps(data).encode(),
                      headers={"Content-Type": "application/json",
                               "Authorization": "Bearer " + self.key})
        with build_opener(NoRedirect()).open(req, timeout=30) as response:
            raw = response.read(1048577)
        if len(raw) > 1048576:
            raise ValueError("oversized response")
        result = json.loads(raw)
        tokens = result["usage"]["completion_tokens"]
        answer = result["choices"][0]["message"]["content"]
        if type(tokens) is not int or not 1 <= tokens <= 64 or not isinstance(answer, str) or not answer.strip():
            raise ValueError("invalid token count or output")
        return tokens


class Scheduler:
    """Round robin baseline or in-flight and observed-latency scheduling.

    Skip failed replicas for a short cooldown. Retry an unfinished request at
    most once per replica. Retry is safe here because generation has no tools.
    """
    def __init__(self, endpoints, policy, clock=time.monotonic, cooldown=5):
        if not endpoints or policy not in ("round_robin", "adaptive"):
            raise ValueError("invalid endpoints or policy")
        self.endpoints = endpoints
        self.policy = policy
        self.clock = clock
        self.cooldown = cooldown
        self.lock = threading.Lock()
        self.cursor = 0
        self.stats = [{"inflight": 0, "ewma": None, "retry_at": 0,
                       "completed": 0, "failures": 0} for _ in endpoints]

    def choose(self, attempted):
        eligible = [i for i, s in enumerate(self.stats)
                    if i not in attempted and s["retry_at"] <= self.clock()]
        if not eligible:
            return None
        if self.policy == "round_robin":
            i = min(eligible, key=lambda j: (j - self.cursor) % len(self.endpoints))
        else:
            i = min(eligible, key=lambda j: (
                (self.stats[j]["inflight"] + 1) * (self.stats[j]["ewma"] or .15),
                (j - self.cursor) % len(self.endpoints)))
        self.cursor = (i + 1) % len(self.endpoints)
        self.stats[i]["inflight"] += 1
        return i

    def generate(self, prompt):
        attempted = set()
        start = self.clock()
        while len(attempted) < len(self.endpoints):
            with self.lock:
                i = self.choose(attempted)
            if i is None:
                break
            attempted.add(i)
            t0 = self.clock()
            try:
                tokens = self.endpoints[i].generate(prompt)
            except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError):
                with self.lock:
                    s = self.stats[i]
                    s["inflight"] -= 1
                    s["failures"] += 1
                    s["retry_at"] = self.clock() + self.cooldown
                continue
            elapsed = self.clock() - t0
            with self.lock:
                s = self.stats[i]
                s["inflight"] -= 1
                s["completed"] += 1
                s["ewma"] = elapsed if s["ewma"] is None else .3 * elapsed + .7 * s["ewma"]
            return {"status": "ok", "worker": self.endpoints[i].label,
                    "tokens": tokens, "seconds": self.clock() - start,
                    "attempts": len(attempted)}
        return {"status": "failed", "tokens": 0, "seconds": self.clock() - start,
                "attempts": len(attempted)}


def run(endpoints, policy, concurrency, repeats=3):
    results = []
    for _ in range(repeats):
        scheduler = Scheduler(endpoints, policy)
        prompts = PROMPTS * 2
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            records = list(pool.map(scheduler.generate, prompts))
        elapsed = time.perf_counter() - start
        completed = [r for r in records if r["status"] == "ok"]
        results.append({"seconds": elapsed, "completed": len(completed),
                        "failed": len(records) - len(completed),
                        "tokens_per_second": sum(r["tokens"] for r in completed) / elapsed,
                        "mean_success_latency_seconds": statistics.mean(
                            r["seconds"] for r in completed) if completed else None,
                        "worker_stats": {e.label: s for e, s in zip(endpoints, scheduler.stats)},
                        "requests": records})
    return {"workers": len(endpoints), "policy": policy, "concurrency": concurrency,
            "repeats": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ports", type=int, nargs="+", required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--model", default="qwen")
    args = parser.parse_args()
    if not 1 <= len(args.ports) <= 8 or len(set(args.ports)) != len(args.ports):
        parser.error("provide 1–8 distinct loopback ports")
    key = args.key_file.read_text().strip()
    if len(key) < 16:
        parser.error("provide a private API key")
    endpoints = [Endpoint("worker-" + str(i + 1), p, key, args.model)
                 for i, p in enumerate(args.ports)]
    for endpoint in endpoints:
        endpoint.generate("Say hello briefly.")
    print(json.dumps({"scope": "distributed model replicas, not tensor parallelism",
                      "requests_per_repeat": len(PROMPTS) * 2,
                      "results": [run(endpoints[:n], policy, 4)
                                  for n in sorted({1, len(endpoints)})
                                  for policy in ("round_robin", "adaptive")]}, indent=2))


if __name__ == "__main__":
    main()
