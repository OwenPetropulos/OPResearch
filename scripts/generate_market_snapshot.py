"""
generate_market_snapshot.py
Fetches live market data (equities, rates, commodities, FX, global markets).
Outputs: market_snapshot.json
"""

import json
from datetime import datetime
import yfinance as yf
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "dashboard" / "data"
OUTPUT_FILE = OUTPUT_DIR / "market_snapshot.json"

# ─────────────────────────────────────────────────────────────
# TICKER DEFINITIONS
# ─────────────────────────────────────────────────────────────

EQUITIES = [
    {"label": "S&P 500", "ticker": "ES=F"},
    {"label": "Nasdaq", "ticker": "NQ=F"},
    {"label": "Dow Jones", "ticker": "YM=F"},
    {"label": "Russell 2000", "ticker": "RTY=F"},
    {"label": "VIX", "ticker": "^VIX"},
]

RATES = [
    {"label": "US 3M", "ticker": "^IRX"},
    {"label": "US 2Y", "ticker": "^TYX"},
    {"label": "US 10Y", "ticker": "^TNX"},
    {"label": "Japan 10Y", "ticker": "^FTJPY"},
    {"label": "UK 10Y", "ticker": "^FTUK10"},
    {"label": "China 10Y", "ticker": "513500.SH"},
]

COMMODITIES = [
    {"label": "Crude Oil", "ticker": "CL=F"},
    {"label": "Gold", "ticker": "GC=F"},
    {"label": "Silver", "ticker": "SI=F"},
    {"label": "Copper", "ticker": "HG=F"},
]

FX_PAIRS = [
    {"label": "USD/JPY", "ticker": "USDJPY=X"},
    {"label": "EUR/USD", "ticker": "EURUSD=X"},
    {"label": "EUR/JPY", "ticker": "EURJPY=X"},
    {"label": "DXY", "ticker": "DXY=F"},
]

GLOBAL_MARKETS = {
    "asia": [
        {"label": "Nikkei 225", "ticker": "^N225"},
        {"label": "Hang Seng", "ticker": "^HSI"},
        {"label": "Shanghai Comp", "ticker": "000001.SS"},
        {"label": "KOSPI", "ticker": "^KS11"},
    ],
    "europe": [
        {"label": "FTSE 100", "ticker": "^FTSE"},
        {"label": "DAX", "ticker": "^GDAXI"},
        {"label": "CAC 40", "ticker": "^FCHI"},
        {"label": "Euro Stoxx 50", "ticker": "^STOXX50E"},
    ],
}

# ─────────────────────────────────────────────────────────────
# MARKET STATUS
# ─────────────────────────────────────────────────────────────

def get_market_status():
    """Determine market status (Open, Closed, Pre-Market, After-Hours)"""
    now = datetime.utcnow()
    hour_utc = now.hour
    
    # US market hours: 9:30 AM - 4:00 PM ET = 13:30 - 20:00 UTC
    if 13 <= hour_utc < 20:
        return "Open"
    elif 8 <= hour_utc < 13:
        return "Pre-Market"
    elif 20 <= hour_utc < 24:
        return "After-Hours"
    else:
        return "Closed"

# ─────────────────────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────────────────────

def fetch_tickers(ticker_specs):
    """
    Fetch price data for a list of ticker specs.
    Returns dict keyed by ticker with price, change, percent_change, direction.
    """
    if not ticker_specs:
        return {}
    
    tickers_str = " ".join([spec["ticker"] for spec in ticker_specs])
    try:
        data = yf.download(tickers_str, period="5d", interval="1d", progress=False)
    except Exception as e:
        print(f"[ERROR] yfinance fetch failed: {e}")
        return {}
    
    results = {}
    for spec in ticker_specs:
        ticker = spec["ticker"]
        label = spec["label"]
        
        try:
            # Handle single ticker vs multiple
            if len(ticker_specs) == 1:
                close_data = data['Close']
            else:
                close_data = data['Close'][ticker] if ticker in data['Close'].columns else None
            
            if close_data is None or len(close_data) < 2:
                continue
            
            current = close_data.iloc[-1]
            previous = close_data.iloc[-2]
            change = current - previous
            pct = (change / previous * 100) if previous != 0 else 0
            
            results[ticker] = {
                "label": label,
                "ticker": ticker,
                "price": round(current, 2),
                "change": round(change, 2),
                "percent_change": round(pct, 2),
                "direction": "up" if change >= 0 else "down",
            }
        except Exception as e:
            print(f"[WARN] Could not fetch {ticker}: {e}")
    
    return results

def fetch_rates():
    """Fetch yield data (rates are percentages, already scaled)."""
    rate_results = fetch_tickers(RATES)
    
    # Rates from Yahoo are already in percentage form (e.g., 4.37)
    # No scaling needed for Treasury yields
    output = []
    for spec in RATES:
        ticker = spec["ticker"]
        if ticker in rate_results:
            result = rate_results[ticker]
            result["price"] = round(result["price"], 2)  # Keep as-is (already %)
            output.append(result)
    
    return output

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print("generate_market_snapshot.py")
    print(f"{'='*60}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Fetch all data groups
    print("[...] Fetching equities...")
    equities = [fetch_tickers([spec])[spec["ticker"]] for spec in EQUITIES 
                if spec["ticker"] in fetch_tickers([spec])]
    equities = list(fetch_tickers(EQUITIES).values())
    
    print("[...] Fetching rates...")
    rates = fetch_rates()
    
    print("[...] Fetching commodities...")
    commodities = list(fetch_tickers(COMMODITIES).values())
    
    print("[...] Fetching FX...")
    fx = list(fetch_tickers(FX_PAIRS).values())
    
    print("[...] Fetching global markets...")
    global_markets = {
        "asia": list(fetch_tickers(GLOBAL_MARKETS["asia"]).values()),
        "europe": list(fetch_tickers(GLOBAL_MARKETS["europe"]).values()),
    }
    
    # Build snapshot
    snapshot = {
        "last_updated": datetime.utcnow().isoformat() + "Z",
        "market_status": get_market_status(),
        "macro_summary": "Market data loaded. Review equities, rates, commodities for positioning cues.",
        "equities": equities,
        "rates": rates,
        "commodities": commodities,
        "fx": fx,
        "global_markets": global_markets,
    }
    
    # Write output
    OUTPUT_FILE.write_text(json.dumps(snapshot, indent=2))
    print(f"[OK] Snapshot written to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
