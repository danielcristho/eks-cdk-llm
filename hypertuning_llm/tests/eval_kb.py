"""Score the served model on questions generated from src/knowledge_base.md."""
import argparse
import json
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prompt import KNOWLEDGE_BASE, SYSTEM_PROMPT  # noqa: E402

NO_KB_PROMPT = "You are the assistant for AWS Community Day Jakarta 2026. Reply in the user's language."

DONT_KNOW = re.compile(
    r"tidak tahu|tidak ada|tidak tersedia|tidak disebut|tidak tercantum|tidak ditemukan|belum ada|belum tersedia"
    r"|tidak memiliki informasi|tidak dapat menemukan|don't know|do not know|not (?:listed|mentioned|available|specified|provided)"
    r"|no information|not in the|isn't (?:listed|mentioned|available)|cannot find|can't find",
    re.IGNORECASE,
)

OUT_OF_KB = [
    ("Apakah parkir di venue gratis?", "id"),
    ("What is the wifi password at the venue?", "en"),
    ("Apa dress code untuk acara ini?", "id"),
    ("How much does the ticket cost in rupiah?", "en"),
    ("Apakah makan siangnya ada pilihan vegetarian?", "id"),
    ("Will the sessions be recorded and uploaded to YouTube?", "en"),
]

SKIP_TITLES = {"Open Gate", "Opening", "Keynote", "Tech Talk", "Closing", "Prayer, Networking & Lunch Time"}


def parse_agenda(kb: str) -> list[dict]:
    sessions = []
    for line in kb.split("## Agenda", 1)[1].splitlines():
        if not re.match(r"- \d\d:\d\d-", line):
            continue
        parts = line[2:].split(" | ")
        start, end = parts[0].split("-")
        speakers = " | ".join(parts[3:])  # a speaker's company can contain " | "
        names = [s.split(" (")[0].strip() for s in re.split(r"\), ", speakers) if s.strip()]
        sessions.append({"start": start, "end": end, "room": parts[1], "title": parts[2], "speakers": names})
    return sessions


def room_names(room: str) -> list[str]:
    m = re.match(r"(.+?) \((.+)\)$", room)
    if not m:
        return [room]
    return [m.group(1)] + ([m.group(2)] if not m.group(2).isdigit() else [])


def build_questions(sessions: list[dict]) -> list[dict]:
    counts = Counter(s["title"] for s in sessions)
    all_speakers = sorted({n for s in sessions for n in s["speakers"]})
    qs = []
    unique = [s for s in sessions if counts[s["title"]] == 1 and s["title"] not in SKIP_TITLES]
    for i, s in enumerate(unique):
        t = s["title"]
        en = i % 2 == 1
        qs.append({"cat": "time", "q": f"What time does the session '{t}' start?" if en else f"Jam berapa sesi '{t}' dimulai?",
                   "expect_any": [s["start"], s["start"].replace(":", ".")]})
        qs.append({"cat": "room", "q": f"Which room is the session '{t}' in?" if en else f"Di ruangan mana sesi '{t}'?",
                   "expect_any": room_names(s["room"])})
        if s["speakers"]:
            qs.append({"cat": "speaker", "q": f"Who presents '{t}'?" if en else f"Siapa pembicara sesi '{t}'?",
                       "expect_any": [s["speakers"][0].split()[-1]]})  # two speakers share the first name Muhammad
        else:
            qs.append({"cat": "unknown_speaker", "q": f"Who presents '{t}'?" if en else f"Siapa pembicara sesi '{t}'?",
                       "dont_know": True, "forbid": all_speakers})

    by_room = defaultdict(list)
    for s in sessions:
        by_room[s["room"]].append(s)
    for room, items in by_room.items():
        lunch = next((s for s in items if s["title"].startswith("Prayer")), None)
        after = next((s for s in items if lunch and s["start"] >= lunch["end"] and s["title"] not in SKIP_TITLES), None)
        if after:
            qs.append({"cat": "after_lunch", "q": f"Apa sesi pertama setelah makan siang di {room}?",
                       "expect_any": [" ".join(after["title"].split()[:4])]})

    for q, _ in OUT_OF_KB:
        qs.append({"cat": "out_of_kb", "q": q, "dont_know": True, "forbid": all_speakers})
    return qs


def score(q: dict, reply: str) -> bool:
    low = reply.lower()
    if q.get("dont_know"):
        invented = any(name.lower() in low for name in q.get("forbid", []))
        return bool(DONT_KNOW.search(reply)) and not invented
    return any(e.lower() in low for e in q["expect_any"])


def ask(url: str, model: str, system: str, question: str, max_tokens: int) -> tuple[str, dict]:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": question}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(f"{url}/v1/chat/completions", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.load(resp)
    return data["choices"][0]["message"]["content"], data["usage"]


def selftest() -> None:
    sessions = parse_agenda(KNOWLEDGE_BASE)
    assert room_names("Main Stage (Atrium)") == ["Main Stage", "Atrium"]
    assert room_names("Classroom 1 (800)") == ["Classroom 1"]
    assert room_names("Classroom 2") == ["Classroom 2"]
    qs = build_questions(sessions)
    cats = Counter(q["cat"] for q in qs)
    assert cats["time"] > 10 and cats["unknown_speaker"] > 0 and cats["after_lunch"] > 0, cats
    last = [q["expect_any"][0] for q in qs if q["cat"] == "speaker"]
    assert len(last) == len(set(last)), "speaker last names must be unique to score fairly"
    time_q = next(q for q in qs if q["cat"] == "time")
    assert score(time_q, f"Sesi itu mulai jam {time_q['expect_any'][0]} WIB.")
    assert not score(time_q, "Sesi itu mulai jam 23:59.")
    unk = next(q for q in qs if q["cat"] == "unknown_speaker")
    assert score(unk, "Pembicaranya tidak tercantum di informasi acara.")
    assert not score(unk, f"Pembicaranya adalah {unk['forbid'][0]}.")
    assert not score(unk, "Pembicaranya adalah John Doe.")
    print(f"selftest ok: {len(qs)} questions {dict(cats)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--model", default="qwen-lora")
    p.add_argument("--label", default="model-v1+kb", help="name for this run in the results file")
    p.add_argument("--no-kb", action="store_true", help="leave the knowledge base out of the system prompt")
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--out", help="write per-question results as JSON")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        return selftest()

    system = NO_KB_PROMPT if a.no_kb else SYSTEM_PROMPT
    qs = build_questions(parse_agenda(KNOWLEDGE_BASE))
    results, start = [], time.time()
    for q in qs:
        reply, usage = ask(a.url, a.model, system, q["q"], a.max_tokens)
        ok = score(q, reply)
        results.append({**q, "reply": reply, "pass": ok, "completion_tokens": usage["completion_tokens"]})
        print(f"{'PASS' if ok else 'FAIL'}  {q['cat']:<15} {q['q'][:70]:<70}  -> {reply[:90]!r}", flush=True)

    by_cat = defaultdict(lambda: [0, 0])
    for r in results:
        by_cat[r["cat"]][0] += r["pass"]
        by_cat[r["cat"]][1] += 1
    total = sum(r["pass"] for r in results)
    avg_tokens = sum(r["completion_tokens"] for r in results) / len(results)
    print(f"\n{a.label}  ({'no kb' if a.no_kb else 'kb in system prompt'}, {time.time() - start:.0f}s)")
    for cat, (ok, n) in sorted(by_cat.items()):
        print(f"  {cat:<15} {ok}/{n}")
    print(f"  {'total':<15} {total}/{len(results)} = {total / len(results):.0%}, avg {avg_tokens:.0f} completion tokens")

    if a.out:
        summary = {cat: {"pass": ok, "n": n} for cat, (ok, n) in by_cat.items()}
        Path(a.out).write_text(json.dumps(
            {"label": a.label, "kb": not a.no_kb, "summary": summary, "results": results}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
