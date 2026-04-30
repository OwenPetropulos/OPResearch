"""
generate_sector_news.py

Fetches sector-specific news from:
1. MarketAux API (primary — filtered by sector/topic)
2. RSS feeds (fallback / supplement)

Writes data/sector_news.json
"""

import sys
import os
import requests

from common import (
    data_path, write_json,
    fetch_feed, deduplicate_stories, sort_stories_by_time,
    strip_internal_fields, classify_sectors, classify_tickers,
    get_why_it_matters, log, clean_text, parse_timestamp, to_iso,
)
from config import RSS_FEEDS, MARKETAUX_SECTOR_FILTERS

ALL_SECTORS  = ["Energy", "Financials", "Technology", "Industrials", "Consumer", "Healthcare", "Macro"]
MAX_PER_SECTOR = 8
MAX_TOTAL      = 60


# ============================================================
# MARKETAUX SECTOR FETCHER
# ============================================================

def fetch_marketaux_sector_news(api_key: str) -> list[dict]:
    """
    Fetch sector-specific news from MarketAux.
    Makes one request per sector group to get relevant stories.
    """
    stories = []

    for sector, params_override in MARKETAUX_SECTOR_FILTERS.items():
        params = {
            "api_token":       api_key,
            "language":        "en",
            "filter_entities": "true",
            "limit":           10,
            "sort":            "published_at",
            "sort_order":      "desc",
        }
        params.update(params_override)

        try:
            resp = requests.get(
                "https://api.marketaux.com/v1/news/all",
                params=params,
                timeout=15,
            )
            resp.raise_for_status()
            data     = resp.json()
            articles = data.get("data", [])
            log.info(f"MarketAux {sector}: {len(articles)} articles")

            for article in articles:
                title   = clean_text(article.get("title", ""))
                summary = clean_text(article.get("description", "") or article.get("snippet", ""))
                url     = article.get("url", "")
                source  = clean_text(article.get("source", ""))
                published = article.get("published_at", "")

                if not title:
                    continue

                entities = article.get("entities", [])
                entity_tickers = [
                    e.get("symbol", "").upper()
                    for e in entities
                    if e.get("type") == "equity" and e.get("symbol")
                ]

                dt        = parse_timestamp(published)
                timestamp = to_iso(dt)

                stories.append({
                    "title":            title,
                    "summary":          summary[:600] if summary else "",
                    "url":              url,
                    "source_name":      source or "MarketAux",
                    "source_type":      "Mainstream",
                    "_dt":              dt,
                    "timestamp":        timestamp,
                    "_entity_tickers":  entity_tickers,
                    "_marketaux_sector": sector,  # hint for classification
                })

        except requests.exceptions.HTTPError as e:
            log.error(f"MarketAux {sector} HTTP error: {e}")
        except Exception as e:
            log.error(f"MarketAux {sector} error: {e}")

    return stories


# ============================================================
# STORY ENRICHMENT
# ============================================================

def enrich_sector_story(story: dict, index: int) -> dict:
    combined = story.get("title", "") + " " + story.get("summary", "")
    sectors  = classify_sectors(combined)

    # Use MarketAux sector hint as a tiebreaker if classification
    # returns only generic "Macro"
    hint = story.pop("_marketaux_sector", None)
    if len(sectors) == 1 and sectors[0] == "Macro" and hint and hint in ALL_SECTORS:
        sectors = [hint]

    entity_tickers  = story.pop("_entity_tickers", [])
    keyword_tickers = classify_tickers(combined)
    all_tickers     = list(dict.fromkeys(entity_tickers + keyword_tickers))[:6]

    story["id"]             = f"sn{str(index + 1).zfill(3)}"
    story["sector"]         = sectors[0] if sectors else "Macro"
    story["sector_tags"]    = sectors
    story["ticker_tags"]    = all_tickers
    story["why_it_matters"] = get_why_it_matters(combined)

    return story


# ============================================================
# MAIN
# ============================================================

def main():
    log.info("=== generate_sector_news.py ===")

    api_key    = os.environ.get("MARKETAUX_API_KEY", "").strip()
    all_raw    = []

    # 1. MarketAux sector-specific stories
    if api_key:
        marketaux_stories = fetch_marketaux_sector_news(api_key)
        all_raw.extend(marketaux_stories)
        log.info(f"Got {len(marketaux_stories)} sector stories from MarketAux")
    else:
        log.warning("MARKETAUX_API_KEY not set — using RSS only")

    # 2. RSS supplement
    for feed_def in RSS_FEEDS:
        raw = fetch_feed(feed_def, max_entries=40)
        all_raw.extend(raw)

    sorted_raw = sort_stories_by_time(all_raw)
    unique     = deduplicate_stories(sorted_raw)

    enriched_all = [enrich_sector_story(s, i) for i, s in enumerate(unique)]

    # Per-sector cap
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
        log.warning("No sector stories found — writing empty list.")

    success = write_json(data_path("sector_news.json"), output)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
