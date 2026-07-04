"""
generate_macro_and_sector_news.py
Generates both macro_news.json and sector_news.json using multi-source aggregation.
Filters stories by macro relevance vs sector-specific tags.
"""

import json
from pathlib import Path
from news_aggregator import aggregate_all_news

OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "dashboard" / "data"

MACRO_SECTORS = ["Macro", "Financials", "Energy"]
SECTOR_MAPPING = {
    "Technology": ["Technology", "Software"],
    "Financials": ["Financials", "Banking"],
    "Healthcare": ["Healthcare", "Pharma", "Biotech"],
    "Industrials": ["Industrials", "Manufacturing"],
    "Consumer": ["Consumer", "Retail"],
    "Energy": ["Energy", "Oil & Gas"],
}

def classify_story(story: dict) -> tuple:
    """
    Classify a story as macro or sector-specific.
    Returns: (is_macro, assigned_sectors)
    """
    tags = story.get("sector_tags", [])
    
    # Check for macro tags
    is_macro = any(tag in MACRO_SECTORS for tag in tags)
    
    # Map to sectors
    assigned = []
    for sector, keywords in SECTOR_MAPPING.items():
        if any(kw.lower() in str(tag).lower() for kw in keywords for tag in tags):
            assigned.append(sector)
    
    return is_macro, assigned if assigned else ["General"]

def generate_news_files():
    """Fetch all news and split into macro_news.json and sector_news.json."""
    print(f"\n{'='*60}")
    print("generate_macro_and_sector_news.py")
    print(f"{'='*60}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Aggregate news
    print("[...] Aggregating news from all sources...")
    all_stories = aggregate_all_news(limit=100)
    
    # Classify and split
    macro_stories = []
    sector_stories = []
    
    for story in all_stories:
        is_macro, sectors = classify_story(story)
        
        # Add boilerplate why_it_matters if missing
        if not story.get("why_it_matters"):
            story["why_it_matters"] = (
                "Monitor for market implications. Primary transmission channels: "
                "earnings revisions, sector rotation, and macro implications. "
                "Track management commentary and options implied volatility."
            )
        
        if is_macro or "Macro" in story.get("sector_tags", []):
            macro_stories.append(story)
        
        if sectors:
            story["sector_tags"] = sectors
            sector_stories.append(story)
    
    # Limit to reasonable counts
    macro_stories = macro_stories[:30]
    sector_stories = sector_stories[:50]
    
    # Write files
    macro_output = {"stories": macro_stories}
    sector_output = {"stories": sector_stories}
    
    (OUTPUT_DIR / "macro_news.json").write_text(json.dumps(macro_output, indent=2))
    (OUTPUT_DIR / "sector_news.json").write_text(json.dumps(sector_output, indent=2))
    
    print(f"[OK] macro_news.json: {len(macro_stories)} stories")
    print(f"[OK] sector_news.json: {len(sector_stories)} stories")

if __name__ == "__main__":
    generate_news_files()
