"""Keep N requests in flight against vLLM for a while, logging tokens per request.

Stdlib only, so it runs in a bare python:3.11-slim pod inside the cluster (`just load`).
Ray Serve adds a replica once ongoing requests per replica stay above target_ongoing_requests (2),
and a new GPU replica takes ~8 min to serve, so the default duration covers one scale-out.
"""
import argparse
import json
import threading
import time
import urllib.request

QUESTIONS = [
    "Talk soal LLM on EKS jam berapa dan di ruangan mana?",
    "Apa saja sesi setelah makan siang di Main Stage?",
    "Apakah tiketnya bisa di-refund?",
    "Siapa saja pembicara dari AWS?",
    "Explain how Karpenter provisions GPU nodes, in 5 bullet points.",
    "Write a short haiku about Kubernetes autoscaling.",
    "What is LoRA fine-tuning? Answer in 3 sentences.",
    "Compare vLLM and Ray Serve in a short table.",
]

lock = threading.Lock()
totals = {"ok": 0, "err": 0, "prompt": 0, "completion": 0, "latency": 0.0}


def one_request(url: str, question: str, max_tokens: int) -> None:
    body = json.dumps({
        "model": "qwen-lora",
        "messages": [{"role": "user", "content": question}],
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(f"{url}/v1/chat/completions", body, {"Content-Type": "application/json"})
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            usage = json.load(resp)["usage"]
        latency = time.time() - start
        with lock:
            totals["ok"] += 1
            totals["prompt"] += usage["prompt_tokens"]
            totals["completion"] += usage["completion_tokens"]
            totals["latency"] += latency
        print(f"{time.strftime('%H:%M:%S')} 200 {latency:6.1f}s  prompt={usage['prompt_tokens']:4d} "
              f"completion={usage['completion_tokens']:4d} total={usage['total_tokens']:4d}", flush=True)
    except Exception as e:  # keep the load going; errors are part of what we're measuring
        with lock:
            totals["err"] += 1
        print(f"{time.strftime('%H:%M:%S')} ERR {time.time() - start:6.1f}s  {e}", flush=True)


def worker(url: str, deadline: float, offset: int, max_tokens: int) -> None:
    i = offset
    while time.time() < deadline:
        one_request(url, QUESTIONS[i % len(QUESTIONS)], max_tokens)
        i += 1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://vllm-serve-svc.kuberay.svc.cluster.local:8000")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--duration", type=int, default=600, help="seconds")
    p.add_argument("--max-tokens", type=int, default=256)
    a = p.parse_args()

    start = time.time()
    threads = [threading.Thread(target=worker, args=(a.url, start + a.duration, i, a.max_tokens))
               for i in range(a.concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    elapsed = time.time() - start
    ok = totals["ok"] or 1
    print(f"\n{totals['ok']} ok, {totals['err']} errors in {elapsed:.0f}s, concurrency {a.concurrency}")
    print(f"avg latency {totals['latency'] / ok:.1f}s, "
          f"tokens prompt={totals['prompt']} completion={totals['completion']}, "
          f"throughput {totals['completion'] / elapsed:.1f} completion tok/s")


if __name__ == "__main__":
    main()
