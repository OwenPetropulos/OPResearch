"""
generate_portfolio_prices.py
Fetches live prices for portfolio holdings.
Reads portfolio from localStorage (frontend), outputs portfolio_prices.json.
"""

import json
from datetime import datetime
from pathlib import Path
import yfinance as yf

OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "dashboard" / "data"
OUTPUT_FILE = OUTPUT_DIR / "portfolio_prices.json"

def main():
    print(f"\n{'='*60}")
    print("generate_portfolio_prices.py")
    print(f"{'='*60}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # For now, use a default portfolio. In production, this would read from a backend store or API.
    DEFAULT_PORTFOLIO = [
        {"ticker": "AAPL", "shares": 10, "cost_basis": 150.00},
        {"ticker": "MSFT", "shares": 5, "cost_basis": 400.00},
        {"ticker": "NVDA", "shares": 3, "cost_basis": 850.00},
    ]
    
    holdings = []
    
    for position in DEFAULT_PORTFOLIO:
        ticker = position["ticker"]
        shares = position["shares"]
        cost_basis = position["cost_basis"]
        
        try:
            # Fetch latest price
            data = yf.download(ticker, period="1d", progress=False)
            if data.empty:
                print(f"[WARN] No data for {ticker}")
                continue
            
            current_price = data['Close'].iloc[-1]
            position_value = shares * current_price
            cost = shares * cost_basis
            unrealized_gain = position_value - cost
            unrealized_pct = (unrealized_gain / cost * 100) if cost > 0 else 0
            
            holdings.append({
                "ticker": ticker,
                "shares": shares,
                "cost_basis": cost_basis,
                "current_price": round(current_price, 2),
                "position_value": round(position_value, 2),
                "unrealized_gain": round(unrealized_gain, 2),
                "unrealized_pct": round(unrealized_pct, 2),
            })
            
            print(f"[OK] {ticker}: ${current_price:.2f}")
        except Exception as e:
            print(f"[ERROR] Failed to fetch {ticker}: {e}")
    
    output = {
        "last_updated": datetime.utcnow().isoformat() + "Z",
        "holdings": holdings,
        "total_value": round(sum(h["position_value"] for h in holdings), 2),
        "total_cost": round(sum(h["shares"] * h["cost_basis"] for h in holdings), 2),
    }
    
    OUTPUT_FILE.write_text(json.dumps(output, indent=2))
    print(f"[OK] Portfolio prices written to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
