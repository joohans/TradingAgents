"""Automated betting execution via Polymarket CLOB API."""
import os
import logging
import json

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY, SELL

logger = logging.getLogger(__name__)

# Safety limits
MAX_BET_USDC = 50.0          # Max single bet
MAX_DAILY_USDC = 200.0       # Max daily total
MIN_EDGE = 0.05              # Minimum 5% edge to bet
MIN_CONFIDENCE = 0.5         # Minimum 50% confidence

_daily_spent = 0.0


def create_clob_client() -> ClobClient:
    """Create an authenticated CLOB client from env vars."""
    pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
    if not pk.startswith("0x"):
        pk = "0x" + pk

    creds = ApiCreds(
        api_key=os.getenv("POLYMARKET_API_KEY", ""),
        api_secret=os.getenv("POLYMARKET_API_SECRET", ""),
        api_passphrase=os.getenv("POLYMARKET_API_PASSPHRASE", ""),
    )

    return ClobClient(
        host="https://clob.polymarket.com",
        key=pk,
        chain_id=137,
        creds=creds,
    )


def should_execute(decision: dict) -> tuple[bool, str]:
    """Check if a decision meets safety thresholds for execution.

    Returns (should_bet, reason).
    """
    global _daily_spent

    action = decision.get("action", "SKIP")
    if action == "SKIP":
        return False, "Action is SKIP"

    confidence = decision.get("confidence", 0)
    if confidence < MIN_CONFIDENCE:
        return False, f"Confidence {confidence:.0%} < {MIN_CONFIDENCE:.0%} minimum"

    edge = abs(decision.get("edge", 0))
    if edge < MIN_EDGE:
        return False, f"Edge {edge:.1%} < {MIN_EDGE:.1%} minimum"

    position_size = decision.get("position_size", 0)
    bet_amount = min(position_size * 1000, MAX_BET_USDC)  # Assume $1000 bankroll
    if bet_amount <= 0:
        return False, "Position size is 0"

    if _daily_spent + bet_amount > MAX_DAILY_USDC:
        return False, f"Daily limit reached (${_daily_spent:.0f}/${MAX_DAILY_USDC:.0f})"

    return True, f"Approved: ${bet_amount:.2f} bet"


def execute_bet(
    decision: dict,
    token_id: str,
    neg_risk: bool = False,
    dry_run: bool = True,
) -> dict:
    """Execute a bet on Polymarket based on agent decision.

    Args:
        decision: Agent decision dict with action, confidence, edge, position_size
        token_id: The CLOB token ID for the YES outcome
        neg_risk: Whether the market uses neg_risk (multi-outcome markets)
        dry_run: If True, only simulate (don't actually place order)

    Returns:
        Dict with execution result
    """
    global _daily_spent

    # PAPER-ONLY HARD LOCK: if PAPER_ONLY env is set, force dry_run regardless of caller.
    # Guarantees no real order is ever placed during the paper-trading trial.
    if os.getenv("PAPER_ONLY", "").strip() not in ("", "0", "false", "False"):
        dry_run = True

    action = decision.get("action", "SKIP")
    confidence = decision.get("confidence", 0)
    edge = abs(decision.get("edge", 0))
    position_size = decision.get("position_size", 0)

    # Calculate bet
    bet_amount = min(position_size * 1000, MAX_BET_USDC)
    side = BUY if action == "YES" else SELL
    # Price = our estimated probability (market price + edge for YES, market price - edge for NO)
    price = min(max(confidence, 0.01), 0.99)  # Clamp to valid range
    size = round(bet_amount / price, 2)

    result = {
        "action": action,
        "side": "BUY" if side == BUY else "SELL",
        "token_id": token_id,
        "price": price,
        "size": size,
        "amount_usdc": bet_amount,
        "dry_run": dry_run,
        "order_id": None,
        "status": "pending",
        "error": None,
    }

    if dry_run:
        result["status"] = "simulated"
        logger.info(f"DRY RUN: Would place {result['side']} order: {size} @ {price} (${bet_amount:.2f})")
        return result

    try:
        client = create_clob_client()

        # Get tick size for this market
        tick_size = client.get_tick_size(token_id)
        # Round price to valid tick
        price = round(price / float(tick_size)) * float(tick_size)
        price = round(price, 4)

        order_args = OrderArgs(
            price=price,
            size=size,
            side=side,
            token_id=token_id,
        )

        signed_order = client.create_order(order_args)
        resp = client.post_order(signed_order, neg_risk=neg_risk)

        result["order_id"] = resp.get("orderID", resp.get("id", str(resp)))
        result["status"] = "placed"
        result["price"] = price
        _daily_spent += bet_amount

        logger.info(f"ORDER PLACED: {result['side']} {size} @ {price} (${bet_amount:.2f}) → {result['order_id']}")

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        logger.error(f"ORDER FAILED: {e}")

    return result


def format_bet_message(result: dict, event_title: str) -> str:
    """Format bet execution result for Telegram notification (plain text)."""
    if result["dry_run"]:
        prefix = "🧪 DRY RUN"
    elif result["status"] == "placed":
        prefix = "💰 BET PLACED"
    elif result["status"] == "failed":
        prefix = "❌ BET FAILED"
    else:
        prefix = "📋 BET STATUS"

    lines = [
        prefix,
        f"Event: {event_title}",
        f"Side: {result['side']} @ {result['price']}",
        f"Size: {result['size']} (${result['amount_usdc']:.2f})",
    ]

    if result.get("order_id"):
        lines.append(f"Order: {result['order_id'][:20]}")
    if result.get("error"):
        lines.append(f"Error: {result['error']}")

    return "\n".join(lines)


def reset_daily_limit():
    """Reset daily spending counter. Call at midnight."""
    global _daily_spent
    _daily_spent = 0.0
    logger.info("Daily spending limit reset.")
