"""
generate_earnings_calendar.py
Generates earnings_calendar.json with next week's earnings & macro events.
Intended to run automatically on Mondays via GitHub Actions.
Sources: yfinance earnings data, macro calendar data.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
import requests

OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "dashboard" / "data"
OUTPUT_FILE = OUTPUT_DIR / "earnings_calendar.json"

# ─────────────────────────────────────────────────────────────
# HARDCODED MACRO CALENDAR (Update as needed)
# ─────────────────────────────────────────────────────────────

MACRO_EVENTS = {
    # YYYY-MM-DD: [{"event": "...", "time": "10:00 AM ET", "importance": "high|medium|low"}]
    "2026-06-29": [
        {"event": "ISM Mfg PMI", "time": "10:00 AM ET", "importance": "high"},
    ],
    "2026-06-30": [
        {"event": "JOLTS Job Openings", "time": "10:00 AM ET", "importance": "medium"},
    ],
    "2026-07-01": [
        {"event": "FOMC Meeting Decision", "time": "2:00 PM ET", "importance": "high"},
        {"event": "Powell Press Conference", "time": "2:30 PM ET", "importance": "high"},
    ],
    "2026-07-02": [
        {"event": "Initial Jobless Claims", "time": "8:30 AM ET", "importance": "medium"},
    ],
    "2026-07-03": [
        {"event": "NFP / Unemployment", "time": "8:30 AM ET", "importance": "high"},
        {"event": "ISM Services PMI", "time": "10:00 AM ET", "importance": "medium"},
    ],
}

# ─────────────────────────────────────────────────────────────
# EARNINGS DATA SOURCE
# ─────────────────────────────────────────────────────────────

# Key companies to track (replace with API if you want dynamic data)
TRACKED_EARNINGS = [
    {"ticker": "GOOGL", "company": "Alphabet", "expected_date": "2026-07-02", "status": "Pending"},
    {"ticker": "MSFT", "company": "Microsoft", "expected_date": "2026-07-02", "status": "Pending"},
    {"ticker": "AAPL", "company": "Apple", "expected_date": "2026-07-03", "status": "Pending"},
    {"ticker": "AMZN", "company": "Amazon", "expected_date": "2026-07-03", "status": "Pending"},
    {"ticker": "META", "company": "Meta", "expected_date": "2026-07-02", "status": "Pending"},
    {"ticker": "NVDA", "company": "NVIDIA", "expected_date": "2026-07-02", "status": "Pending"},
    {"ticker": "TSLA", "company": "Tesla", "expected_date": "2026-06-30", "status": "Pending"},
]

# ─────────────────────────────────────────────────────────────
# BUILD CALENDAR
# ─────────────────────────────────────────────────────────────

def build_calendar():
    """Build next week's earnings & macro calendar."""
    today = datetime.utcnow().date()
    
    # Determine Monday of next week
    days_until_monday = (7 - today.weekday()) % 7
    if days_until_monday == 0:
        days_until_monday = 7  # If today is Monday, start from next Monday
    
    monday = today + timedelta(days=days_until_monday)
    
    calendar = []
    
    # Build Mon-Fri
    for day_offset in range(5):
        current_date = monday + timedelta(days=day_offset)
        date_str = current_date.strftime("%Y-%m-%d")
        day_name = ["MON", "TUE", "WED", "THU", "FRI"][day_offset]
        
        events = []
        
        # Add macro events
        if date_str in MACRO_EVENTS:
            for macro in MACRO_EVENTS[date_str]:
                events.append({
                    "title": macro["event"],
                    "type": "macro-event",
                    "time": macro["time"],
                    "importance": macro.get("importance", "medium"),
                    "url": None,
                })
        
        # Add earnings for this date
        day_earnings = [e for e in TRACKED_EARNINGS if e["expected_date"] == date_str]
        for earnings in day_earnings:
            events.append({
                "title": f"{earnings['company']} ({earnings['ticker']}) Earnings",
                "type": "earnings-pending",
                "time": "After Close",
                "ticker": earnings["ticker"],
                "url": f"https://finance.yahoo.com/quote/{earnings['ticker']}/",
            })
        
        calendar.append({
            "date": date_str,
            "day": day_name,
            "events": events,
        })
    
    return calendar

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print("generate_earnings_calendar.py")
    print(f"{'='*60}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    calendar = build_calendar()
    
    output = {
        "last_updated": datetime.utcnow().isoformat() + "Z",
        "period": "next_week",
        "calendar": calendar,
    }
    
    OUTPUT_FILE.write_text(json.dumps(output, indent=2))
    print(f"[OK] Calendar written to {OUTPUT_FILE}")
    print(f"[OK] Coverage: {len(calendar)} days, {sum(len(d['events']) for d in calendar)} events")

if __name__ == "__main__":
    main()
