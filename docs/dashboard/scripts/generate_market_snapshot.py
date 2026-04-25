"""
generate_market_snapshot.py

Fetches live market data via yfinance and writes data/market_snapshot.json.
Produces: equities, rates, commodities, global_markets, macro_summary, market_status.
"""

import sys
from datetime import datetime, timezone

from common import (
    data_path, write_json, utc_now_iso,
    fetch_price_data, log,
)
from config import SNAPSHOT_SYMBOLS


# ============================================================
# MARKET STATUS
# ============================================================

def infer_market_status() -> str:
    """
    Infer U.S. market status from current UTC time.
    NYSE hours: 14:30–21:00 UTC (Mon–Fri).
    Pre-market: 09:30–14:30 UTC. After-hours: 21:00–01:00 UTC.
    """
    now     = datetime.now(timezone.utc)
    weekday = now.weekday()  # 0=Mon, 6=Sun
    hour    = now.hour
    minute  = now.minute
    hm      = hour * 60 + minute  # minutes since midnight UTC

    if weekday >= 5:
        return "Closed (Weekend)"

    if 570 <= hm < 870:     # 09:30–14:30 UTC
        return "Pre-Market"
    elif 870 <= hm < 1260:  # 14:30–21:00 UTC
        return "Open"
    elif hm >= 1260 or hm < 60:  # 21:00–01:00 UTC
        return "After-Hours"
    else:
        return "Closed"


# ============================================================
# CBOE RATE INDEX TICKERS
# These report in units of 0.1% (e.g. 46.3 = 4.63% yield).
# ============================================================

CBOE_RATE_TICKERS = {"^TNX", "^IRX", "^TYX", "^FVX"}


# ============================================================
# ITEM BUILDERS
# ============================================================

def build_simple_item(cfg: dict) -> dict | None:
    """
    Build a snapshot item for an equity, commodity, or global index.
    cfg keys: label, ticker, (optional) display_ticker.
    """
    data = fetch_price_data(cfg["ticker"])
    if data is None:
        return None

    return {
        "label":          cfg["label"],
        "ticker":         cfg.get("display_ticker", cfg["ticker"]),
        "price":          data["price"],
        "change":         data["change"],
        "percent_change": data["percent_change"],
        "direction":      data["direction"],
    }


def build_rate_item(cfg: dict) -> dict | None:
    """
    Build a snapshot item for a yield/rate.
    Handles CBOE rate index scaling and international yield fallbacks.
    """
    data = fetch_price_data(cfg["ticker"])

    if data is None:
        # Use fallback value if available in config
        if "fallback" in cfg:
            return {
                "label":          cfg["label"],
                "ticker":         cfg.get("display_ticker", cfg["ticker"]),
                "price":          cfg["fallback"],
                "change":         0.0,
                "percent_change": 0.0,
                "direction":      "flat",
            }
        return None

    # CBOE rate indices report in tenths of a percent — divide by 10
    if cfg["ticker"] in CBOE_RATE_TICKERS:
        price      = round(data["price"] / 10, 3)
        prev       = round(data["prev_close"] / 10, 3) if "prev_close" in data else price
        change     = round(price - prev, 3)
        pct_change = round((change / prev) * 100, 2) if prev else 0.0
        direction  = "up" if change > 0 else ("down" if change < 0 else "flat")
    else:
        price      = data["price"]
        change     = data["change"]
        pct_change = data["percent_change"]
        direction  = data["direction"]

    return {
        "label":          cfg["label"],
        "ticker":         cfg.get("display_ticker", cfg["ticker"]),
        "price":          price,
        "change":         change,
        "percent_change": pct_change,
        "direction":      direction,
    }


# ============================================================
# MACRO SUMMARY
# ============================================================

def build_macro_summary(equities: list, rates: list, commodities: list) -> str:
    """
    Generate a one-line rule-based macro summary from fetched data.
    All logic is deterministic — no LLM calls.
    """
    lines = []

    # Equity tone (exclude VIX from average)
    eq_changes = [e["percent_change"] for e in equities if e.get("ticker") != "VIX"]
    vix_items  = [e for e in equities if e.get("ticker") == "VIX"]
    vix_level  = vix_items[0]["price"] if vix_items else None

    if eq_changes:
        avg_eq = sum(eq_changes) / len(eq_changes)
        if avg_eq <= -0.5:
            lines.append("Equity futures under broad pressure")
        elif avg_eq >= 0.5:
            lines.append("Equity futures broadly bid")
        else:
            lines.append("Equity futures mixed")

    if vix_level is not None:
        if vix_level >= 25:
            lines.append(f"VIX elevated at {vix_level:.1f} — stress signals active")
        elif vix_level >= 18:
            lines.append(f"VIX at {vix_level:.1f} — market caution elevated")

    # 10Y rate tone
    us10y_items = [r for r in rates if r.get("ticker") == "US10Y"]
    if us10y_items:
        r10 = us10y_items[0]
        if r10["change"] >= 0.04:
            lines.append(
                f"10Y yields rising ({r10['price']:.2f}%), adding pressure to duration assets"
            )
        elif r10["change"] <= -0.04:
            lines.append(
                f"10Y yields easing ({r10['price']:.2f}%), providing rate relief"
            )
        else:
            lines.append(f"Rates broadly stable; 10Y at {r10['price']:.2f}%")

    # Gold tone
    gold_items = [c for c in commodities if c.get("ticker") == "GC"]
    if gold_items:
        g = gold_items[0]
        if g["percent_change"] >= 0.5:
            lines.append(f"Gold firming ({g['percent_change']:+.1f}%), safe-haven demand evident")
        elif g["percent_change"] <= -0.5:
            lines.append(f"Gold retreating ({g['percent_change']:+.1f}%)")

    # Crude tone
    crude_items = [c for c in commodities if c.get("ticker") == "CL"]
    if crude_items:
        cl = crude_items[0]
        if cl["percent_change"] <= -1.0:
            lines.append(f"Crude sliding ({cl['percent_change']:+.1f}%)")
        elif cl["percent_change"] >= 1.0:
            lines.append(f"Crude firming ({cl['percent_change']:+.1f}%)")

    if not lines:
        return "Market data refreshed. Monitor key levels and upcoming macro data releases."

    return "; ".join(lines) + "."


# ============================================================
# MAIN
# ============================================================

def main():
    log.info("=== generate_market_snapshot.py ===")

    snapshot = {
        "last_updated":   utc_now_iso(),
        "macro_summary":  "",
        "market_status":  infer_market_status(),
        "equities":       [],
        "rates":          [],
        "commodities":    [],
        "global_markets": {"asia": [], "europe": []},
    }

    # Equities
    for cfg in SNAPSHOT_SYMBOLS.get("equities", []):
        item = build_simple_item(cfg)
        if item:
            snapshot["equities"].append(item)
        else:
            log.warning(f"Skipping equity: {cfg['label']}")

    # Rates
    for cfg in SNAPSHOT_SYMBOLS.get("rates", []):
        item = build_rate_item(cfg)
        if item:
            snapshot["rates"].append(item)
        else:
            log.warning(f"Skipping rate: {cfg['label']}")

    # Commodities
    for cfg in SNAPSHOT_SYMBOLS.get("commodities", []):
        item = build_simple_item(cfg)
        if item:
            snapshot["commodities"].append(item)
        else:
            log.warning(f"Skipping commodity: {cfg['label']}")

    # Global markets
    for region in ("asia", "europe"):
        for cfg in SNAPSHOT_SYMBOLS["global_markets"].get(region, []):
            item = build_simple_item(cfg)
            if item:
                snapshot["global_markets"][region].append(item)
            else:
                log.warning(f"Skipping {region} index: {cfg['label']}")

    # Derive macro summary from fetched data
    snapshot["macro_summary"] = build_macro_summary(
        snapshot["equities"],
        snapshot["rates"],
        snapshot["commodities"],
    )

    log.info(f"Market status: {snapshot['market_status']}")
    log.info(f"Equities: {len(snapshot['equities'])} | Rates: {len(snapshot['rates'])} | "
             f"Commodities: {len(snapshot['commodities'])}")

    success = write_json(data_path("market_snapshot.json"), snapshot)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
