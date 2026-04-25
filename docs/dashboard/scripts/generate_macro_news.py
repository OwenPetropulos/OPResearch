"""
generate_macro_news.py

Ingests macro-relevant stories from public RSS feeds, classifies them,
and writes data/macro_news.json.
"""

import sys

from common import (
    data_path, write_json, utc_now_iso,
    fetch_feed, deduplicate_stories, sort_stories_by_time,
    strip_internal_fields, classify_sectors, classify_tickers,
    get_why_it_matters, make_story_id, log,
)
from config import RSS_FEEDS, SECTOR_KEYWORDS

# ============================================================
# MACRO RELEVANCE FILTER
# ============================================================

# Keywords that indicate a story is macro-relevant enough for the Morning Brief.
MACRO_RELEVANCE_KEYWORDS = [
    # Fed / rates
    "federal reserve", "fed", "fomc", "rate cut", "rate hike", "interest rate",
    "treasury", "yield", "bond", "inflation", "cpi", "pce", "core prices",
    # Global macro
    "gdp", "recession", "stagflation", "central bank", "ecb", "boj",
    "bank of england", "pboc", "imf", "world bank",
    # Macro risk
    "china economy", "global growth", "emerging market", "geopolitical",
    "trade war", "tariff", "dollar", "yen", "euro",
    # Markets
    "s&p 500", "nasdaq", "stock market", "equity", "vix", "volatility",
    "crude oil", "gold", "commodities", "risk off", "risk sentiment",
    # Economic data
    "jobs report", "nonfarm payroll", "unemployment", "retail sales",
    "manufacturing", "pmi", "housing", "consumer confidence",
]


def is_macro_relevant(story: dict) -> bool:
    """Return True if the story is likely macro-relevant."""
    text = (story.get("title", "") + " " + story.get("summary", "")).lower()
    return any(kw in text for kw in MACRO_RELEVANCE_KEYWORDS)


# ============================================================
# STORY ENRICHMENT
# ============================================================

def enrich_story(story: dict, index: int) -> dict:
    """Add id, sector_tags, ticker_tags, and why_it_matters to a story."""
    combined = story.get("title", "") + " " + story.get("summary", "")

    story["id"]            = f"mn{str(index + 1).zfill(3)}"
    story["sector_tags"]   = classify_sectors(combined)
    story["ticker_tags"]   = classify_tickers(combined)
    story["why_it_matters"] = get_why_it_matters(combined)

    return story


# ============================================================
# MAIN
# ============================================================

def main():
    log.info("=== generate_macro_news.py ===")

    all_stories = []

    for feed_def in RSS_FEEDS:
        raw = fetch_feed(feed_def, max_entries=25)
        # Filter to macro-relevant stories only
        relevant = [s for s in raw if is_macro_relevant(s)]
        all_stories.extend(relevant)

    # Sort, deduplicate, trim
    sorted_stories    = sort_stories_by_time(all_stories)
    unique_stories    = deduplicate_stories(sorted_stories)
    top_stories       = unique_stories[:10]

    # Enrich each story
    enriched = [enrich_story(s, i) for i, s in enumerate(top_stories)]

    # Strip internal pipeline fields before output
    output_stories = strip_internal_fields(enriched)

    output = {"stories": output_stories}

    if not output_stories:
        log.warning("No macro stories found — writing empty stories list.")

    success = write_json(data_path("macro_news.json"), output)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
