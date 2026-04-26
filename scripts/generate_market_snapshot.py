"""
generate_market_snapshot.py

Fetches live market data and writes data/market_snapshot.json.
Includes equities, rates, commodities, FX, and global markets.

RATE INDEX NOTE: Yahoo Finance ^TNX and ^IRX return yield values
directly as percentages (e.g. 4.63 = 4.63%). Do NOT divide by 10.
"""

import sys
from datetime import datetime, timezone

from common import data_path, write_json, utc_now_iso, fetch_price_data, log
from config import SNAPSHOT_SYMBOLS, CBOE_RATE_TICKERS


# ============================================================
# MARKET STATUS
# ============================================================

def infer_market_status() -> str:
    now     = datetime.now(timezone.utc)
    weekday = now.weekday()
    hm      = now.hour * 60 + now.minute

    if weekday >= 5:
        return "Closed (Weekend)"
    if 570 <= hm < 870:
        return "Pre-Market"
    elif 870 <= hm < 1260:
        return "Open"
    elif hm >= 1260 or hm < 60:
        return "After-Hours"
    else:
        return "Closed"


# ============================================================
# ITEM BUILDERS
# ============================================================

def build_simple_item(cfg: dict) -> dict | None:
    data = fetch_price_data(cfg["ticker"])
    if data is None:
        return None
    return {
        "label":          cfg["label"],
        "ticker":         cfg.get("display_ticker", cfg["ticker"]),
        "price":          round(data["price"], 2),
        "change":         round(data["change"], 2),
        "percent_change": data["percent_change"],
        "direction":      data["direction"],
    }


def build_rate_item(cfg: dict) -> dict | None:
    """
    Build a rate/yield snapshot item.
    DUMMY_ prefix = use static fallback, no fetch.
    CBOE tickers return yield directly as percent — no scaling needed.
    """
    ticker = cfg["ticker"]

    if ticker.startswith("DUMMY_"):
        return {
            "label":          cfg["label"],
            "ticker":         cfg.get("display_ticker", ticker),
            "price":          cfg.get("fallback", 0.0),
            "change":         0.0,
            "percent_change": 0.0,
            "direction":      "flat",
        }

    data = fetch_price_data(ticker)

    if data is None:
        if "fallback" in cfg:
            return {
                "label":          cfg["label"],
                "ticker":         cfg.get("display_ticker", ticker),
                "price":          cfg["fallback"],
                "change":         0.0,
                "percent_change": 0.0,
                "direction":      "flat",
            }
        return None

    # ^TNX and ^IRX already return percent values directly
    # e.g. 4.63 means 4.63% — no division needed
    price      = round(data["price"], 3)
    prev       = round(data["prev_close"], 3)
    change     = round(price - prev, 3)
    pct_change = round((change / prev) * 100, 2) if prev else 0.0
    direction  = "up" if change > 0 else ("down" if change < 0 else "flat")

    return {
        "label":          cfg["label"],
        "ticker":         cfg.get("display_ticker", ticker),
        "price":          price,
        "change":         change,
        "percent_change": pct_change,
        "direction":      direction,
    }


def build_fx_item(cfg: dict) -> dict | None:
    """Build a snapshot item for an FX pair."""
    data = fetch_price_data(cfg["ticker"])
    if data is None:
        return None
    return {
        "label":          cfg["label"],
        "ticker":         cfg.get("display_ticker", cfg["ticker"]),
        "price":          round(data["price"], 4),
        "change":         round(data["change"], 4),
        "percent_change": data["percent_change"],
        "direction":      data["direction"],
    }


# ============================================================
# MACRO SUMMARY
# ============================================================

def build_macro_summary(equities, rates, commodities, fx) -> str:
    lines = []

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

    us10y_items = [r for r in rates if r.get("ticker") == "US10Y"]
    if us10y_items:
        r10 = us10y_items[0]
        if r10["change"] >= 0.04:
            lines.append(f"10Y yields rising ({r10['price']:.2f}%), pressure on duration assets")
        elif r10["change"] <= -0.04:
            lines.append(f"10Y yields easing ({r10['price']:.2f}%), rate relief emerging")
        else:
            lines.append(f"Rates broadly stable; 10Y at {r10['price']:.2f}%")

    gold_items = [c for c in commodities if c.get("ticker") == "GC"]
    if gold_items:
        g = gold_items[0]
        if g["percent_change"] >= 0.5:
            lines.append(f"Gold firming ({g['percent_change']:+.1f}%), safe-haven demand evident")
        elif g["percent_change"] <= -0.5:
            lines.append(f"Gold retreating ({g['percent_change']:+.1f}%)")

    crude_items = [c for c in commodities if c.get("ticker") == "CL"]
    if crude_items:
        cl = crude_items[0]
        if cl["percent_change"] <= -1.0:
            lines.append(f"Crude sliding ({cl['percent_change']:+.1f}%)")
        elif cl["percent_change"] >= 1.0:
            lines.append(f"Crude firming ({cl['percent_change']:+.1f}%)")

    # FX color
    dxy_items = [f for f in fx if f.get("ticker") == "DXY"]
    if dxy_items:
        dxy = dxy_items[0]
        if dxy["percent_change"] >= 0.4:
            lines.append(f"Dollar strengthening (DXY {dxy['percent_change']:+.1f}%)")
        elif dxy["percent_change"] <= -0.4:
            lines.append(f"Dollar weakening (DXY {dxy['percent_change']:+.1f}%)")

    if not lines:
        return "Market data refreshed. Monitor key levels and upcoming macro releases."

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
        "fx":             [],
        "global_markets": {"asia": [], "europe": []},
    }

    for cfg in SNAPSHOT_SYMBOLS.get("equities", []):
        item = build_simple_item(cfg)
        if item:
            snapshot["equities"].append(item)
        else:
            log.warning(f"Skipping equity: {cfg['label']}")

    for cfg in SNAPSHOT_SYMBOLS.get("rates", []):
        item = build_rate_item(cfg)
        if item:
            snapshot["rates"].append(item)
        else:
            log.warning(f"Skipping rate: {cfg['label']}")

    for cfg in SNAPSHOT_SYMBOLS.get("commodities", []):
        item = build_simple_item(cfg)
        if item:
            snapshot["commodities"].append(item)
        else:
            log.warning(f"Skipping commodity: {cfg['label']}")

    for cfg in SNAPSHOT_SYMBOLS.get("fx", []):
        item = build_fx_item(cfg)
        if item:
            snapshot["fx"].append(item)
        else:
            log.warning(f"Skipping FX: {cfg['label']}")

    for region in ("asia", "europe"):
        for cfg in SNAPSHOT_SYMBOLS["global_markets"].get(region, []):
            item = build_simple_item(cfg)
            if item:
                snapshot["global_markets"][region].append(item)
            else:
                log.warning(f"Skipping {region} index: {cfg['label']}")

    snapshot["macro_summary"] = build_macro_summary(
        snapshot["equities"],
        snapshot["rates"],
        snapshot["commodities"],
        snapshot["fx"],
    )

    log.info(
        f"Market status: {snapshot['market_status']} | "
        f"Equities: {len(snapshot['equities'])} | "
        f"Rates: {len(snapshot['rates'])} | "
        f"FX: {len(snapshot['fx'])} | "
        f"Commodities: {len(snapshot['commodities'])}"
    )

    success = write_json(data_path("market_snapshot.json"), snapshot)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
