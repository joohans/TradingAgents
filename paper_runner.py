#!/usr/bin/env python3
"""Paper-trading runner for the 1-month PolyAgent trial.

Scans near-term Polymarket events, runs the full multi-agent analysis on the
top N by 24h volume, and appends one JSONL record per analyzed event to
results/paper_trades.jsonl — enough to settle & evaluate later (entry prices,
token ids, end date, decision, simulated bet). NEVER places a real order:
PAPER_ONLY hard-locks execute_bet to dry-run.

Env:
  POLY_PROXY   socks5h://127.0.0.1:1080  (Canada exit; set by the service)
  PAPER_ONLY   1                          (hard-lock, always set)
  PAPER_MAX_EVENTS   default 3            (how many events per run)
  PAPER_MAX_DAYS     default 45           (only events ending within N days)
  TRADINGAGENTS_RESULTS_DIR  default ./results
"""
import os
import sys
import json
import time
import logging
from datetime import datetime, timezone

from tradingagents.agents.utils.polymarket_tools import _api_get
from tradingagents.default_config import DEFAULT_CONFIG
from scanner.analyzer import analyze_single
from scanner.trader import execute_bet, should_execute

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("paper_runner")

GAMMA = "https://gamma-api.polymarket.com"
MAX_EVENTS = int(os.getenv("PAPER_MAX_EVENTS", "3"))
MAX_DAYS = int(os.getenv("PAPER_MAX_DAYS", "45"))
RESULTS_DIR = os.getenv("TRADINGAGENTS_RESULTS_DIR", os.path.join(os.path.dirname(__file__), "results"))
OUT = os.path.join(RESULTS_DIR, "paper_trades.jsonl")


def _within_days(end_date: str, days: int) -> bool:
    if not end_date:
        return False
    try:
        end = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
    except ValueError:
        return False
    delta = (end - datetime.now(timezone.utc)).total_seconds()
    return 0 < delta <= days * 86400


def pick_events(limit: int, max_days: int) -> list[dict]:
    """Top active events by 24h volume, ending within max_days."""
    raw = _api_get(f"{GAMMA}/events", params={
        "limit": limit * 8, "active": "true", "closed": "false",
        "order": "volume24hr", "ascending": "false",
    })
    out = []
    for ev in raw if isinstance(raw, list) else []:
        mkts = ev.get("markets") or []
        if not mkts:
            continue
        if not _within_days(mkts[0].get("endDate", ev.get("endDate", "")), max_days):
            continue
        out.append(ev)
        if len(out) >= limit:
            break
    return out


def snapshot(ev: dict) -> dict:
    m = (ev.get("markets") or [{}])[0]
    return {
        "outcomes": m.get("outcomes"),
        "outcome_prices": m.get("outcomePrices"),
        "clob_token_ids": m.get("clobTokenIds"),
        "end_date": m.get("endDate", ev.get("endDate")),
        "closed": m.get("closed"),
        "volume24hr": m.get("volume24hr"),
    }


def run_once() -> int:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    events = pick_events(MAX_EVENTS, MAX_DAYS)
    log.warning("picked %d events", len(events))
    n = 0
    for ev in events:
        eid = str(ev.get("id"))
        title = ev.get("title", "")
        snap = snapshot(ev)
        log.warning("analyzing %s: %s", eid, title[:60])
        t = time.time()
        try:
            r = analyze_single(eid)
        except Exception as e:  # noqa: BLE001
            log.error("analyze failed %s: %s", eid, e)
            _append({"ts": _now(), "event_id": eid, "title": title,
                     "snapshot": snap, "error": str(e)})
            continue
        decision = r.get("decision", {})
        elapsed = round(time.time() - t)

        # Simulate the bet (dry-run hard-locked by PAPER_ONLY)
        sim = None
        try:
            token_ids = json.loads(snap.get("clob_token_ids") or "[]")
            yes_token = token_ids[0] if token_ids else ""
            ok, reason = should_execute(decision)
            sim = execute_bet(decision, yes_token, dry_run=True)
            sim["would_bet"] = ok
            sim["gate_reason"] = reason
        except Exception as e:  # noqa: BLE001
            sim = {"error": str(e)}

        _append({
            "ts": _now(),
            "event_id": eid,
            "title": title,
            "slug": ev.get("slug"),
            "snapshot": snap,          # entry prices / token ids / end date
            "decision": decision,      # action / confidence / edge / position_size / reasoning
            "sim_bet": sim,            # dry-run order the agent would place
            "elapsed_s": elapsed,
            "llm": DEFAULT_CONFIG.get("deep_think_llm"),
        })
        n += 1
    log.warning("logged %d decisions -> %s", n, OUT)
    return n


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append(rec: dict) -> None:
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    if not os.getenv("PAPER_ONLY"):
        print("refusing to run without PAPER_ONLY=1", file=sys.stderr)
        sys.exit(2)
    sys.exit(0 if run_once() >= 0 else 1)
