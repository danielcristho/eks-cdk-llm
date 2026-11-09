"""Regenerate knowledge_base.md from the AWS Community Day Jakarta 2026 API.
"""
import json
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path

API = "https://api.awscommunity.id/api/events/community-day-2026"
OUT = Path(__file__).with_name("knowledge_base.md")

STATIC = """\
# AWS Community Day Jakarta 2026

Source: https://jakarta.awscommunity.id/comday/ (retrieved {retrieved})

## Overview
- Date: Saturday, October 31, 2026, 08:30-17:00 WIB (gate opens 08:00)
- Venue: Atrium, BINUS @Anggrek, Jakarta Barat, Indonesia
- Organizer: AWS User Group Jakarta
- Community-led conference: technical talks, workshops and networking around AWS
- 300+ expected attendees
- Topic areas: Generative AI & ML, Serverless & Modern Ops, Platform Eng & DevOps, Security & Resiliency

## FAQ
- Getting there: reachable by public transit or private vehicle; the event page links to Google Maps.
- Ticket fee: a small commitment fee to prevent no-shows. General admission is fully refundable
  when you check in at the registration desk on the event day.
- Tickets: sold on the event page, powered by Pretix.
- Speaking: Community Day 2026 submissions are closed; monthly meetups still accept speakers.
- Sponsoring: see the Sponsors page on the event site.
"""


def get(path: str) -> dict:
    with urllib.request.urlopen(f"{API}/{path}", timeout=30) as resp:
        return json.load(resp)


def hhmm(iso: str) -> str:
    return datetime.fromisoformat(iso).strftime("%H:%M")


def main() -> None:
    schedule = sorted(get("schedule")["schedule"], key=lambda s: (s["startsAt"], s["room"]))
    speakers = {p["id"]: p for p in get("speakers")["speakers"]}

    lines = [STATIC.format(retrieved=datetime.now().strftime("%Y-%m-%d")), "## Agenda (WIB)"]
    for s in schedule:
        names = ", ".join(
            f"{sp['name']} ({sp['role']}, {sp['company']})"
            for sp in (speakers.get(i) for i in s["speakerIds"])
            if sp
        )
        room = s["room"] if s["room"] == s["track"] else f"{s['track']} ({s['room']})"
        row = f"- {hhmm(s['startsAt'])}-{hhmm(s['endsAt'])} | {room} | {s['title']}"
        lines.append(f"{row} | {names}" if names else row)

    # list questions ("speakers from AWS?") need the model to scan every agenda row, which a 1.5B model
    # gets wrong; grouping once here turns them into a single-line lookup
    by_company = defaultdict(list)
    for sid in dict.fromkeys(i for s in schedule for i in s["speakerIds"]):
        if sid in speakers:
            sp = speakers[sid]
            by_company[sp["company"]].append(f"{sp['name']} ({sp['role']})")
    lines += ["", "## Speakers by company"]
    lines += [f"- {company}: {', '.join(names)}" for company, names in sorted(by_company.items())]

    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT} ({len(schedule)} sessions)")


if __name__ == "__main__":
    main()
