"""
generate_portfolio_prices.py

Fetches current prices for all tickers in the portfolio/watchlist universe.
Dynamically merges:
  1. Base ticker list from config.py (PRICE_TICKERS)
  2. Any tickers already in the existing portfolio_prices.json
  3. Any tickers in watchlist.json

This means any ticker added to the portfolio or watchlist will automatically
be priced on the next pipeline run — no config changes needed.
"""

import sys
import json

from common import (
    data_path, read_json, write_json, utc_now_iso,
    fetch_prices_bulk, log,
)
from config import PRICE_TICKERS


def discover_tickers() -> list[str]:
    """
    Build the full ticker universe by merging:
    - PRICE_TICKERS from config (base hardcoded list)
    - Tickers already tracked in portfolio_prices.json
    - Tickers in watchlist.json
    Returns a deduplicated sorted list.
    """
    universe = set(PRICE_TICKERS)

    # Pull any tickers already in the existing portfolio_prices.json
    existing_prices = read_json(data_path("portfolio_prices.json"))
    if existing_prices and isinstance(existing_prices.get("prices"), dict):
        for ticker in existing_prices["prices"]:
            if ticker != "CASH":
                universe.add(ticker.upper())
        log.info(f"Added {len(existing_prices['prices'])} tickers from existing portfolio_prices.json")

    # Pull tickers from watchlist.json
    watchlist_data = read_json(data_path("watchlist.json"))
    if watchlist_data and isinstance(watchlist_data.get("watchlist"), list):
        for item in watchlist_data["watchlist"]:
            ticker = item.get("ticker", "").strip().upper()
            if ticker:
                universe.add(ticker)
        log.info(f"Merged watchlist tickers into universe")

    # Remove CASH — always added manually, never fetched
    universe.discard("CASH")

    result = sorted(universe)
    log.info(f"Total ticker universe: {len(result)} tickers")
    return result


def main():
    log.info("=== generate_portfolio_prices.py ===")

    tickers = discover_tickers()
    log.info(f"Fetching prices for {len(tickers)} tickers...")

    prices = fetch_prices_bulk(tickers)

    # CASH is always exactly 1.00
    prices["CASH"] = 1.00

    fetched_count = len([t for t in tickers if t in prices])
    log.info(f"Successfully fetched {fetched_count}/{len(tickers)} tickers.")

    if fetched_count == 0:
        log.error("Zero tickers resolved — aborting to avoid overwriting with empty data.")
        sys.exit(1)

    if fetched_count < len(tickers) * 0.5:
        log.warning("Fewer than 50% of tickers resolved. Data quality may be degraded.")

    output = {
        "prices":       prices,
        "last_updated": utc_now_iso(),
    }

    success = write_json(data_path("portfolio_prices.json"), output)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
