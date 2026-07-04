"""
generate_sectors_overview.py
Generates sectors_overview.json from sector_news.json.
Runs AFTER sector_news is generated (dependency).
"""

import json
from pathlib import Path
from collections import Counter

DATA_DIR = Path(__file__).parent.parent / "docs" / "dashboard" / "data"
SECTOR_NEWS_FILE = DATA_DIR / "sector_news.json"
OUTPUT_FILE = DATA_DIR / "sectors_overview.json"

SECTOR_ORDER = [
    "Technology",
    "Financials",
    "Healthcare",
    "Industrials",
    "Consumer",
    "Energy",
    "Materials",
    "Utilities",
    "Real Estate",
    "Communications",
]

def analyze_sector_sentiment(stories: list) -> str:
    """Infer sentiment from story themes (positive/neutral/negative)."""
    if not stories:
        return "NEUTRAL"
    
    positive_keywords = ["rally", "surge", "gains", "beat", "strong", "recovery"]
    negative_keywords = ["plunge", "slump", "miss", "weak", "decline", "pressure"]
    
    pos_count = sum(1 for s in stories if any(kw in s.get("title", "").lower() for kw in positive_keywords))
    neg_count = sum(1 for s in stories if any(kw in s.get("title", "").lower() for kw in negative_keywords))
    
    if pos_count > neg_count:
        return "POSITIVE"
    elif neg_count > pos_count:
        return "NEGATIVE"
    else:
        return "NEUTRAL"

def main():
    print(f"\n{'='*60}")
    print("generate_sectors_overview.py")
    print(f"{'='*60}")
    
    # Load sector news
    if not SECTOR_NEWS_FILE.exists():
        print(f"[ERROR] {SECTOR_NEWS_FILE} not found. Run sector_news generator first.")
        return
    
    sector_news = json.loads(SECTOR_NEWS_FILE.read_text())
    stories = sector_news.get("stories", [])
    
    # Group stories by sector
    sector_map = {}
    for story in stories:
        for sector in story.get("sector_tags", []):
            if sector not in sector_map:
                sector_map[sector] = []
            sector_map[sector].append(story)
    
    # Build sector overview
    sectors = []
    for sector in SECTOR_ORDER:
        if sector not in sector_map:
            continue
        
        stories = sector_map[sector][:10]  # Top 10 stories per sector
        
        # Primary driver (first story headline)
        primary_driver = stories[0].get("title", "Market consolidation") if stories else ""
        
        sectors.append({
            "sector": sector,
            "sentiment": analyze_sector_sentiment(stories),
            "primary_driver": primary_driver,
            "story_count": len(sector_map[sector]),
        })
    
    output = {
        "last_updated": datetime.utcnow().isoformat() + "Z",
        "sectors": sectors,
    }
    
    OUTPUT_FILE.write_text(json.dumps(output, indent=2))
    print(f"[OK] Sectors overview written to {OUTPUT_FILE}")
    print(f"[OK] Coverage: {len(sectors)} sectors")

if __name__ == "__main__":
    from datetime import datetime
    main()
