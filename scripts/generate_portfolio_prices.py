"""
generate_portfolio_prices.py

Fetches current prices for all tickers in the portfolio/watchlist universe
and writes data/portfolio_prices.json.
"""

import sys

from common import (
    data_path, write_json, utc_now_iso,
    fetch_prices_bulk, log,
)
from config import PRICE_TICKERS


def main():
    log.info("=== generate_portfolio_prices.py ===")

    log.info(f"Fetching prices for {len(PRICE_TICKERS)} tickers...")
    prices = fetch_prices_bulk(PRICE_TICKERS)

    # CASH is always exactly 1.00 — never fetched from market data
    prices["CASH"] = 1.00

    if not prices:
        log.error("No prices fetched — aborting to avoid overwriting with empty data.")
        sys.exit(1)

    fetched_count = len([t for t in PRICE_TICKERS if t in prices])
    log.info(f"Successfully fetched {fetched_count}/{len(PRICE_TICKERS)} tickers.")

    if fetched_count < len(PRICE_TICKERS) * 0.5:
        # If fewer than half resolved, something is seriously wrong — warn but still write
        log.warning("Fewer than 50% of tickers resolved. Data quality may be degraded.")

    output = {
        "prices":       prices,
        "last_updated": utc_now_iso(),
    }

    success = write_json(data_path("portfolio_prices.json"), output)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
