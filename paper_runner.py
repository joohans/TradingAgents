#!/usr/bin/env python3
"""Paper-trading runner for the 1-month PolyAgent trial.

Scans near-term Polymarket events, runs the full multi-agent analysis on the
top N by 24h volume (skipping markets already logged), and appends one JSONL
record per analyzed event to results/paper_trades.jsonl. NEVER places a real
order: PAPER_ONLY hard-locks execute_bet to dry-run.

Edge handling: the LLM self-reports an `edge` which is unreliable (observed
84% vs 1% on the same market). We recompute a GROUNDED edge from the real
market price + the model's confidence, use it for the bet gate, and log both
(edge_grounded vs edge_llm) so the trial can measure the gap.

Env:
  POLY_PROXY   socks5h://127.0.0.1:1080  (Canada exit; set by the service)
  PAPER_ONLY   1                          (hard-lock, always set)
  PAPER_MAX_EVENTS   default 3            (how many NEW events per run)
  PAPER_MAX_DAYS     default 45           (only events ending within N days)
  PAPER_MIN_PRICE    default 0.02         (target market must have YES price in [p, 1-p]
                                           and be open; per event the top-volume such
                                           market is analyzed by its own question)
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
from scanner.analyzer import _analyze_event
from scanner.trader import execute_bet, should_execute

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("paper_runner")

GAMMA = "https://gamma-api.polymarket.com"
MAX_EVENTS = int(os.getenv("PAPER_MAX_EVENTS", "3"))
MAX_DAYS = int(os.getenv("PAPER_MAX_DAYS", "45"))
MIN_PRICE = float(os.getenv("PAPER_MIN_PRICE", "0.02"))   # skip YES price outside [p, 1-p]
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


def logged_keys() -> tuple[set[str], set[str]]:
    """(market_ids, event_ids) already recorded — skip to avoid duplicate entries.

    Dedup is per MARKET (an event can hold dozens of markets); event_ids are kept
    for old records that predate market_id logging.
    """
    mids: set[str] = set()
    eids: set[str] = set()
    if not os.path.exists(OUT):
        return mids, eids
    with open(OUT, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            mid = (rec.get("snapshot") or {}).get("market_id")
            if mid:
                mids.add(str(mid))
            else:
                eids.add(str(rec.get("event_id")))
    return mids, eids


def _yes_price(m: dict):
    """YES price of a binary market, or None if not a clean 2-outcome market."""
    try:
        prices = json.loads(m.get("outcomePrices") or "[]")
        outs = json.loads(m.get("outcomes") or "[]")
        if len(prices) != 2 or len(outs) != 2:
            return None
        labels = [str(o).strip().lower() for o in outs]
        yes_idx = labels.index("yes") if "yes" in labels else 0
        return float(prices[yes_idx])
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def market_reason(m: dict, max_days: int, exclude_mids: set[str]) -> str | None:
    """Why a market is NOT a clean paper-trade target (None = OK).

    Day-1 lessons: the event-level closed=false filter let through markets that
    had already resolved (prices 0/1), and 0.9995-priced markets are not really
    tradable on the underdog side -> require open + inside a price band.
    """
    if str(m.get("id")) in exclude_mids:
        return "already logged"
    if m.get("closed") or m.get("active") is False:
        return "closed"
    if not _within_days(m.get("endDate", ""), max_days):
        return "end date out of range"
    p = _yes_price(m)
    if p is None:
        return "not binary"
    if p < MIN_PRICE or p > 1 - MIN_PRICE:
        return f"price {p:.4f} outside band"
    return None


def pick_target(ev: dict, max_days: int, exclude_mids: set[str]) -> dict | None:
    """Choose the ONE market inside an event to analyze: the highest-24h-volume
    market that passes market_reason(). Top-volume Polymarket events are almost
    all multi-market ("Bitcoin above ___?" = 11 strikes, esports = game/match
    markets), and analyzing "the event" while betting markets[0] produced
    meaningless records on day 1."""
    best, best_vol = None, -1.0
    for m in ev.get("markets") or []:
        why = market_reason(m, max_days, exclude_mids)
        if why:
            continue
        vol = float(m.get("volume24hr") or 0)
        if vol > best_vol:
            best, best_vol = m, vol
    return best


def pick_events(limit: int, max_days: int, exclude_mids: set[str], exclude_eids: set[str]) -> list[tuple[dict, dict]]:
    """Top active events by 24h volume -> [(event, target_market)], not already logged."""
    raw = _api_get(f"{GAMMA}/events", params={
        "limit": max(limit * 12, 40), "active": "true", "closed": "false",
        "order": "volume24hr", "ascending": "false",
    })
    out = []
    for ev in raw if isinstance(raw, list) else []:
        if str(ev.get("id")) in exclude_eids:
            continue
        m = pick_target(ev, max_days, exclude_mids)
        if m is None:
            log.info("skip event %s (%s): no tradable market", ev.get("id"), (ev.get("title") or "")[:40])
            continue
        out.append((ev, m))
        if len(out) >= limit:
            break
    return out


def agent_question(m: dict, fallback: str) -> str:
    """Question handed to the agents. Sports/O-U markets label outcomes
    'Over'/'Under' or team names instead of Yes/No; the decision schema is
    YES/NO, so spell out the mapping (YES = first outcome = our 'yes' side)."""
    q = m.get("question") or fallback
    try:
        outs = json.loads(m.get("outcomes") or "[]")
    except (json.JSONDecodeError, TypeError):
        outs = []
    labels = [str(o).strip() for o in outs]
    if len(labels) == 2 and [l.lower() for l in labels] != ["yes", "no"]:
        q = f"{q} (YES = '{labels[0]}', NO = '{labels[1]}')"
    return q


def snapshot(ev: dict, m: dict) -> dict:
    return {
        "market_id": m.get("id"),
        "question": m.get("question"),
        "n_markets": len(ev.get("markets") or []),
        "outcomes": m.get("outcomes"),
        "outcome_prices": m.get("outcomePrices"),
        "clob_token_ids": m.get("clobTokenIds"),
        "end_date": m.get("endDate", ev.get("endDate")),
        "closed": m.get("closed"),
        "volume24hr": m.get("volume24hr"),
    }


def grounded_edge(decision: dict, snap: dict):
    """Recompute edge from real market price + model confidence.

    Returns (edge, yes_price_mkt, yes_prob_est) or (None, None, None) if the
    market is not a clean binary Yes/No. Edge is signed in favor of the chosen
    action: positive = the chosen side looks underpriced by the market.
    """
    try:
        prices = json.loads(snap.get("outcome_prices") or "[]")
        outs = json.loads(snap.get("outcomes") or "[]")
    except (json.JSONDecodeError, TypeError):
        return None, None, None
    if len(prices) != 2 or len(outs) != 2:
        return None, None, None
    labels = [str(o).strip().lower() for o in outs]
    yes_idx = labels.index("yes") if "yes" in labels else 0
    try:
        p_yes_mkt = float(prices[yes_idx])
    except (ValueError, TypeError):
        return None, None, None
    conf = float(decision.get("confidence") or 0)
    action = decision.get("action")
    if action == "YES":
        p_yes_est = conf
        edge = p_yes_est - p_yes_mkt
    elif action == "NO":
        p_yes_est = 1.0 - conf
        edge = p_yes_mkt - p_yes_est          # NO underpriced when market over-weights YES
    else:
        return 0.0, p_yes_mkt, None
    return round(edge, 4), round(p_yes_mkt, 4), round(p_yes_est, 4)


def run_once() -> int:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    seen_mids, seen_eids = logged_keys()
    pairs = pick_events(MAX_EVENTS, MAX_DAYS, seen_mids, seen_eids)
    log.warning("picked %d new markets (skipping %d markets / %d legacy events already logged)",
                len(pairs), len(seen_mids), len(seen_eids))
    n = 0
    for ev, m in pairs:
        eid = str(ev.get("id"))
        title = ev.get("title", "")
        question = agent_question(m, title)        # the agents decide THIS question
        snap = snapshot(ev, m)
        log.warning("analyzing %s/%s: %s", eid, m.get("id"), question[:70])
        t = time.time()
        try:
            decision, _reports = _analyze_event(eid, question, DEFAULT_CONFIG.copy())
        except Exception as e:  # noqa: BLE001
            log.error("analyze failed %s: %s", eid, e)
            _append({"ts": _now(), "event_id": eid, "title": title, "question": question,
                     "snapshot": snap, "error": str(e)})
            continue
        elapsed = round(time.time() - t)
        edge_llm = decision.get("edge")

        # Grounded edge from real price + confidence; use it for the gate.
        g_edge, p_mkt, p_est = grounded_edge(decision, snap)
        gate_decision = dict(decision)
        if g_edge is not None:
            gate_decision["edge"] = g_edge

        sim = None
        try:
            token_ids = json.loads(snap.get("clob_token_ids") or "[]")
            yes_token = token_ids[0] if token_ids else ""
            ok, reason = should_execute(gate_decision)
            # should_execute uses abs(edge); a NEGATIVE grounded edge means the
            # chosen side is already over-priced by the market -> never bet.
            if ok and g_edge is not None and g_edge < 0:
                ok, reason = False, f"Grounded edge {g_edge:.1%} negative (chosen side over-priced)"
            sim = execute_bet(gate_decision, yes_token, dry_run=True)
            sim["would_bet"] = ok
            sim["gate_reason"] = reason
        except Exception as e:  # noqa: BLE001
            sim = {"error": str(e)}

        _append({
            "ts": _now(),
            "event_id": eid,
            "title": title,
            "question": question,
            "slug": ev.get("slug"),
            "snapshot": snap,
            "decision": decision,
            "edge_grounded": g_edge,        # recomputed, trustworthy
            "edge_llm": edge_llm,           # LLM self-reported (for comparison)
            "yes_price_mkt": p_mkt,
            "yes_prob_est": p_est,
            "sim_bet": sim,
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
