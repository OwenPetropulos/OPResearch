"""
generate_sector_news.py

Ingests stories from public RSS feeds, classifies them into sectors,
and writes data/sector_news.json.
"""

import sys

from common import (
    data_path, write_json,
    fetch_feed, deduplicate_stories, sort_stories_by_time,
    strip_internal_fields, classify_sectors, classify_tickers,
    get_why_it_matters, make_story_id, log,
)
from config import RSS_FEEDS

ALL_SECTORS = ["Energy", "Financials", "Technology", "Industrials", "Consumer", "Healthcare", "Macro"]

# Increased from 3 to 8 — sectors page now shows up to 10
MAX_PER_SECTOR = 8

# Total story cap
MAX_TOTAL = 60


def enrich_sector_story(story: dict, index: int) -> dict:
    combined = story.get("title", "") + " " + story.get("summary", "")
    sectors  = classify_sectors(combined)

    story["id"]             = f"sn{str(index + 1).zfill(3)}"
    story["sector"]         = sectors[0] if sectors else "Macro"
    story["sector_tags"]    = sectors
    story["ticker_tags"]    = classify_tickers(combined)
    story["why_it_matters"] = get_why_it_matters(combined)

    return story


def main():
    log.info("=== generate_sector_news.py ===")

    all_raw = []
    for feed_def in RSS_FEEDS:
        raw = fetch_feed(feed_def, max_entries=40)
        all_raw.extend(raw)

    sorted_raw = sort_stories_by_time(all_raw)
    unique     = deduplicate_stories(sorted_raw)

    enriched_all = []
    for i, story in enumerate(unique):
        enriched_all.append(enrich_sector_story(story, i))

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