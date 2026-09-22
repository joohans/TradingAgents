#!/usr/bin/env python3
"""Settle & evaluate PolyAgent paper trades (results/paper_trades.jsonl).

For every logged decision, re-fetch the Gamma event, find the market that was
analyzed, and if it has resolved compute:

  * outcome          YES won / NO won
  * realistic PnL    for records the gate approved (would_bet=True), assuming a
                     fill at the market price seen at entry (NOT the LLM's
                     self-chosen price): YES -> buy at p_yes, NO -> buy at 1-p_yes.
  * calibration      Brier score of the model's implied P(YES) vs. the market's
                     P(YES) at entry, on the same resolved markets.

Records are tagged valid_entry=False when the entry itself was bad (multi-market
event analyzed as a whole, market already closed at entry, price outside the
tradable band) so the trial can be evaluated on clean entries only. Writes
results/paper_settlement.json and prints a Markdown summary. Read-only against
Polymarket; never trades.

Env: POLY_PROXY (see paper_runner), TRADINGAGENTS_RESULTS_DIR, PAPER_MIN_PRICE.
"""
import os
import sys
import json
import logging
from datetime import datetime, timezone

from tradingagents.agents.utils.polymarket_tools import _api_get

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("paper_settle")

GAMMA = "https://gamma-api.polymarket.com"
RESULTS_DIR = os.getenv("TRADINGAGENTS_RESULTS_DIR", os.path.join(os.path.dirname(__file__), "results"))
TRADES = os.path.join(RESULTS_DIR, "paper_trades.jsonl")
OUT = os.path.join(RESULTS_DIR, "paper_settlement.json")
MIN_PRICE = float(os.getenv("PAPER_MIN_PRICE", "0.02"))


# ---------------------------------------------------------------- helpers
def _load_trades() -> list[dict]:
    if not os.path.exists(TRADES):
        return []
    out = []
    with open(TRADES, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                log.warning("bad line skipped")
    return out


def _binary(m: dict):
    """(p_yes, yes_idx) for a 2-outcome market, else (None, None)."""
    try:
        prices = json.loads(m.get("outcomePrices") or "[]")
        outs = json.loads(m.get("outcomes") or "[]")
    except (json.JSONDecodeError, TypeError):
        return None, None
    if len(prices) != 2 or len(outs) != 2:
        return None, None
    labels = [str(o).strip().lower() for o in outs]
    yes_idx = labels.index("yes") if "yes" in labels else 0
    try:
        return float(prices[yes_idx]), yes_idx
    except (ValueError, TypeError):
        return None, None


def _entry_yes_price(rec: dict):
    if rec.get("yes_price_mkt") is not None:
        return float(rec["yes_price_mkt"])
    snap = rec.get("snapshot") or {}
    p, _ = _binary({"outcomePrices": snap.get("outcome_prices"), "outcomes": snap.get("outcomes")})
    return p


def _model_p_yes(rec: dict):
    """Model-implied P(YES): logged yes_prob_est, else derived from action+confidence."""
    if rec.get("yes_prob_est") is not None:
        return float(rec["yes_prob_est"])
    d = rec.get("decision") or {}
    conf = d.get("confidence")
    if conf is None:
        return None
    conf = float(conf)
    if d.get("action") == "YES":
        return conf
    if d.get("action") == "NO":
        return 1.0 - conf
    return None  # SKIP: no directional estimate


def _resolution(m: dict):
    """('YES'|'NO'|None, detail) from a fetched market."""
    p, _ = _binary(m)
    if p is None:
        return None, "not binary"
    status = (m.get("umaResolutionStatus") or "").lower()
    if m.get("closed") and (status == "resolved" or p in (0.0, 1.0)):
        if p >= 0.99:
            return "YES", status
        if p <= 0.01:
            return "NO", status
        return None, f"closed but ambiguous price {p}"
    return None, "open"


_event_cache: dict[str, dict] = {}


def _fetch_event(eid: str) -> dict | None:
    if eid in _event_cache:
        return _event_cache[eid]
    try:
        ev = _api_get(f"{GAMMA}/events/{eid}")
    except Exception as e:  # noqa: BLE001
        log.error("fetch event %s failed: %s", eid, e)
        ev = None
    _event_cache[eid] = ev
    return ev


def _find_market(ev: dict, rec: dict) -> dict | None:
    mkts = ev.get("markets") or []
    if not mkts:
        return None
    mid = (rec.get("snapshot") or {}).get("market_id")
    if mid:
        for m in mkts:
            if str(m.get("id")) == str(mid):
                return m
    return mkts[0]


# ---------------------------------------------------------------- core
def settle_record(rec: dict) -> dict:
    snap = rec.get("snapshot") or {}
    d = rec.get("decision") or {}
    sim = rec.get("sim_bet") or {}
    row = {
        "ts": rec.get("ts"),
        "event_id": rec.get("event_id"),
        "title": rec.get("title"),
        "action": d.get("action"),
        "confidence": d.get("confidence"),
        "edge_grounded": rec.get("edge_grounded"),
        "edge_llm": rec.get("edge_llm", d.get("edge")),
        "would_bet": bool(sim.get("would_bet")),
        "stake": float(sim.get("amount_usdc") or 0) if sim.get("would_bet") else 0.0,
        "p_yes_entry": _entry_yes_price(rec),
        "p_yes_model": _model_p_yes(rec),
        "end_date": snap.get("end_date"),
        "error": rec.get("error"),
    }

    # --- entry validity (was this a clean thing to trade?)
    invalid = []
    ev = _fetch_event(str(rec.get("event_id")))
    n_markets = snap.get("n_markets")
    if n_markets is None and ev:
        n_markets = len(ev.get("markets") or [])
    # Multi-market event is only a problem for legacy records that analyzed the
    # whole event and bet markets[0]; new records carry the target market_id.
    if n_markets and n_markets != 1 and not snap.get("market_id"):
        invalid.append(f"multi-market event ({n_markets}) analyzed as a whole")
    if snap.get("closed"):
        invalid.append("market closed at entry")
    p0 = row["p_yes_entry"]
    if p0 is not None and (p0 < MIN_PRICE or p0 > 1 - MIN_PRICE):
        invalid.append(f"entry price {p0} outside band")
    if rec.get("error"):
        invalid.append("analysis error")
    row["valid_entry"] = not invalid
    row["invalid_reasons"] = invalid

    # --- resolution
    m = _find_market(ev, rec) if ev else None
    if m is None:
        row.update(status="unknown", outcome=None, detail="event/market not found")
        return row
    row["question"] = m.get("question")
    outcome, detail = _resolution(m)
    row["detail"] = detail
    if outcome is None:
        row.update(status="pending", outcome=None)
        return row
    row.update(status="resolved", outcome=outcome)
    y = 1.0 if outcome == "YES" else 0.0

    # --- calibration
    if row["p_yes_model"] is not None and p0 is not None:
        row["brier_model"] = round((row["p_yes_model"] - y) ** 2, 4)
        row["brier_market"] = round((p0 - y) ** 2, 4)
    if row["action"] in ("YES", "NO"):
        row["direction_correct"] = (row["action"] == outcome)

    # --- realistic PnL for approved bets
    if row["would_bet"] and row["action"] in ("YES", "NO") and p0 is not None:
        fill = p0 if row["action"] == "YES" else 1.0 - p0
        fill = min(max(fill, 0.001), 0.999)
        stake = row["stake"]
        win = row["action"] == outcome
        row["fill_price"] = round(fill, 4)
        row["pnl"] = round(stake * (1.0 / fill - 1.0), 2) if win else round(-stake, 2)
        row["win"] = win
    return row


def summarize(rows: list[dict]) -> dict:
    def _agg(rs: list[dict]) -> dict:
        resolved = [r for r in rs if r["status"] == "resolved"]
        bets = [r for r in resolved if "pnl" in r]
        cal = [r for r in resolved if "brier_model" in r]
        dirs = [r for r in resolved if "direction_correct" in r]
        s = {
            "records": len(rs),
            "resolved": len(resolved),
            "pending": sum(1 for r in rs if r["status"] == "pending"),
            "bets_settled": len(bets),
            "bets_won": sum(1 for r in bets if r["win"]),
            "stake_total": round(sum(r["stake"] for r in bets), 2),
            "pnl_total": round(sum(r["pnl"] for r in bets), 2),
            "direction_hit_rate": round(sum(r["direction_correct"] for r in dirs) / len(dirs), 3) if dirs else None,
            "brier_model": round(sum(r["brier_model"] for r in cal) / len(cal), 4) if cal else None,
            "brier_market": round(sum(r["brier_market"] for r in cal) / len(cal), 4) if cal else None,
        }
        s["roi"] = round(s["pnl_total"] / s["stake_total"], 3) if s["stake_total"] else None
        return s

    return {
        "settled_at": datetime.now(timezone.utc).isoformat(),
        "all": _agg(rows),
        "valid_entries": _agg([r for r in rows if r["valid_entry"]]),
        "open_bets_pending": sum(1 for r in rows if r["status"] == "pending" and r["would_bet"]),
    }


def render_md(rows: list[dict], summary: dict) -> str:
    L = [f"# Paper settlement — {summary['settled_at'][:16]}Z", ""]
    for key, label in (("all", "All records"), ("valid_entries", "Valid entries only")):
        s = summary[key]
        L.append(f"## {label}")
        L.append(f"- records {s['records']} | resolved {s['resolved']} | pending {s['pending']}")
        L.append(f"- bets settled {s['bets_settled']} (won {s['bets_won']}) | stake ${s['stake_total']} | "
                 f"PnL ${s['pnl_total']} | ROI {s['roi']}")
        L.append(f"- direction hit-rate {s['direction_hit_rate']} | Brier model {s['brier_model']} "
                 f"vs market {s['brier_market']} (lower = better)")
        L.append("")
    L.append("## Records")
    L.append("| ts | event | act | conf | p_mkt | p_model | valid | status | outcome | bet | pnl |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        L.append("| {ts} | {t} | {a} | {c} | {pm} | {pe} | {v} | {st} | {o} | {b} | {p} |".format(
            ts=(r["ts"] or "")[5:16], t=(r["title"] or "")[:38], a=r["action"], c=r["confidence"],
            pm=r["p_yes_entry"], pe=r["p_yes_model"],
            v="ok" if r["valid_entry"] else "✗ " + "; ".join(r["invalid_reasons"])[:40],
            st=r["status"], o=r.get("outcome") or "-",
            b=f"${r['stake']:.0f}" if r["would_bet"] else "-",
            p=r.get("pnl", "-")))
    return "\n".join(L)


def main() -> int:
    recs = _load_trades()
    if not recs:
        print("no trades logged yet")
        return 0
    rows = [settle_record(r) for r in recs]
    summary = summarize(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=1)
    md = render_md(rows, summary)
    with open(os.path.join(RESULTS_DIR, "paper_settlement.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
