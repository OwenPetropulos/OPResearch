"""
generate_sector_news.py

Ingests stories from public RSS feeds, classifies them into sectors,
and writes data/sector_news.json for use by the sector intelligence page
and the homepage key developments feed.
"""

import sys

from common import (
    data_path, write_json,
    fetch_feed, deduplicate_stories, sort_stories_by_time,
    strip_internal_fields, classify_sectors, classify_tickers,
    get_why_it_matters, make_story_id, log,
)
from config import RSS_FEEDS, SECTOR_KEYWORDS

# ============================================================
# ALL SECTORS WE WANT TO COVER
# ============================================================

ALL_SECTORS = ["Energy", "Financials", "Technology", "Industrials", "Consumer", "Healthcare", "Macro"]

# Maximum stories per sector to prevent one sector dominating the feed.
MAX_PER_SECTOR = 3

# Total story cap across all sectors.
MAX_TOTAL = 30


# ============================================================
# STORY ENRICHMENT
# ============================================================

def enrich_sector_story(story: dict, index: int) -> dict:
    """
    Add sector classification, ticker tags, why_it_matters, and id.
    The primary sector is the first match from classify_sectors().
    """
    combined = story.get("title", "") + " " + story.get("summary", "")
    sectors  = classify_sectors(combined)

    story["id"]             = f"sn{str(index + 1).zfill(3)}"
    story["sector"]         = sectors[0] if sectors else "Macro"
    story["sector_tags"]    = sectors
    story["ticker_tags"]    = classify_tickers(combined)
    story["why_it_matters"] = get_why_it_matters(combined)

    return story


# ============================================================
# MAIN
# ============================================================

def main():
    log.info("=== generate_sector_news.py ===")

    all_raw = []

    for feed_def in RSS_FEEDS:
        raw = fetch_feed(feed_def, max_entries=30)
        all_raw.extend(raw)

    # Sort by recency, deduplicate globally
    sorted_raw = sort_stories_by_time(all_raw)
    unique     = deduplicate_stories(sorted_raw)

    # Enrich all stories (add sector classification)
    enriched_all = []
    for i, story in enumerate(unique):
        enriched_all.append(enrich_sector_story(story, i))

    # Per-sector cap: take up to MAX_PER_SECTOR per sector, prioritizing recency
    sector_counts: dict[str, int] = {s: 0 for s in ALL_SECTORS}
    selected = []

    for story in enriched_all:
        sector = story.get("sector", "Macro")
        if sector not in sector_counts:
            sector_counts[sector] = 0
        if sector_counts[sector] < MAX_PER_SECTOR:
            selected.append(story)
            sector_counts[sector] += 1
        if len(selected) >= MAX_TOTAL:
            break

    # Re-index IDs after filtering
    for i, story in enumerate(selected):
        story["id"] = f"sn{str(i + 1).zfill(3)}"

    output_stories = strip_internal_fields(selected)
    output = {"stories": output_stories}

    if not output_stories:
        log.warning("No sector stories found — writing empty stories list.")

    success = write_json(data_path("sector_news.json"), output)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
