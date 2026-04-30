"""
generate_macro_news.py

Fetches macro-relevant news from:
1. MarketAux API (primary — real financial news with entity tagging)
2. RSS feeds (fallback / supplement)

Writes data/macro_news.json
"""

import sys
import os
import requests

from common import (
    data_path, write_json, utc_now_iso,
    fetch_feed, deduplicate_stories, sort_stories_by_time,
    strip_internal_fields, classify_sectors, classify_tickers,
    get_why_it_matters, log, clean_text, parse_timestamp, to_iso,
)
from config import RSS_FEEDS, MARKETAUX_MACRO_FILTERS

# ============================================================
# MACRO RELEVANCE KEYWORDS (for RSS fallback filtering)
# ============================================================

MACRO_RELEVANCE_KEYWORDS = [
    "federal reserve", "fed", "fomc", "rate cut", "rate hike", "interest rate",
    "treasury", "yield", "bond", "inflation", "cpi", "pce", "core prices",
    "gdp", "recession", "stagflation", "central bank", "ecb", "boj",
    "bank of england", "pboc", "imf", "world bank",
    "china economy", "global growth", "emerging market", "geopolitical",
    "trade war", "tariff", "dollar", "yen", "euro",
    "s&p 500", "nasdaq", "stock market", "equity", "vix", "volatility",
    "crude oil", "gold", "commodities", "risk off", "risk sentiment",
    "jobs report", "nonfarm payroll", "unemployment", "retail sales",
    "manufacturing", "pmi", "housing", "consumer confidence",
    "exorbitant privilege", "dedollarization", "reserve currency",
    "currency war", "carry trade", "yen carry",
]

def is_macro_relevant(story: dict) -> bool:
    text = (story.get("title", "") + " " + story.get("summary", "")).lower()
    return any(kw in text for kw in MACRO_RELEVANCE_KEYWORDS)


# ============================================================
# MARKETAUX API FETCHER
# ============================================================

def fetch_marketaux_news(api_key: str, max_stories: int = 15) -> list[dict]:
    """
    Fetch macro/financial news from the MarketAux API.
    Returns normalized story dicts compatible with our schema.
    """
    stories = []

    # MarketAux endpoint — general financial news
    # Filter by broad macro/market topics using the topics parameter
    params = {
        "api_token":    api_key,
        "language":     "en",
        "filter_entities": "true",
        "limit":        25,
        "topics":       MARKETAUX_MACRO_FILTERS.get("topics", ""),
        "sort":         "published_at",
        "sort_order":   "desc",
    }

    try:
        log.info("Fetching from MarketAux API...")
        resp = requests.get(
            "https://api.marketaux.com/v1/news/all",
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        articles = data.get("data", [])
        log.info(f"MarketAux returned {len(articles)} articles")

        for article in articles[:max_stories]:
            title     = clean_text(article.get("title", ""))
            summary   = clean_text(article.get("description", "") or article.get("snippet", ""))
            url       = article.get("url", "")
            source    = clean_text(article.get("source", ""))
            published = article.get("published_at", "")

            if not title:
                continue

            # Extract tickers from MarketAux entity data
            entities      = article.get("entities", [])
            entity_tickers = [
                e.get("symbol", "").upper()
                for e in entities
                if e.get("type") == "equity" and e.get("symbol")
            ]

            dt        = parse_timestamp(published)
            timestamp = to_iso(dt)

            stories.append({
                "title":       title,
                "summary":     summary[:600] if summary else "",
                "url":         url,
                "source_name": source or "MarketAux",
                "source_type": "Mainstream",
                "_dt":         dt,
                "timestamp":   timestamp,
                "_entity_tickers": entity_tickers,  # used during enrichment
            })

    except requests.exceptions.HTTPError as e:
        log.error(f"MarketAux HTTP error: {e} — response: {e.response.text[:200] if e.response else ''}")
    except requests.exceptions.RequestException as e:
        log.error(f"MarketAux request failed: {e}")
    except Exception as e:
        log.error(f"MarketAux unexpected error: {e}")

    return stories


# ============================================================
# STORY ENRICHMENT
# ============================================================

def enrich_story(story: dict, index: int) -> dict:
    combined = story.get("title", "") + " " + story.get("summary", "")

    # Merge entity tickers from MarketAux with our keyword-based tickers
    entity_tickers  = story.pop("_entity_tickers", [])
    keyword_tickers = classify_tickers(combined)
    all_tickers     = list(dict.fromkeys(entity_tickers + keyword_tickers))[:6]

    story["id"]             = f"mn{str(index + 1).zfill(3)}"
    story["sector_tags"]    = classify_sectors(combined)
    story["ticker_tags"]    = all_tickers
    story["why_it_matters"] = get_why_it_matters(combined)

    return story


# ============================================================
# MAIN
# ============================================================

def main():
    log.info("=== generate_macro_news.py ===")

    api_key = os.environ.get("MARKETAUX_API_KEY", "").strip()
    all_stories = []

    # 1. Try MarketAux first (primary source)
    if api_key:
        marketaux_stories = fetch_marketaux_news(api_key, max_stories=15)
        all_stories.extend(marketaux_stories)
        log.info(f"Got {len(marketaux_stories)} stories from MarketAux")
    else:
        log.warning("MARKETAUX_API_KEY not set — skipping MarketAux, using RSS only")

    # 2. Supplement with RSS feeds (always run, helps fill gaps)
    for feed_def in RSS_FEEDS:
        raw = fetch_feed(feed_def, max_entries=20)
        relevant = [s for s in raw if is_macro_relevant(s)]
        all_stories.extend(relevant)

    # Sort, deduplicate, trim to top 15
    sorted_stories = sort_stories_by_time(all_stories)
    unique_stories = deduplicate_stories(sorted_stories)
    top_stories    = unique_stories[:15]

    # Enrich
    enriched = [enrich_story(s, i) for i, s in enumerate(top_stories)]

    output_stories = strip_internal_fields(enriched)
    output = {"stories": output_stories}

    if not output_stories:
        log.warning("No macro stories found — writing empty stories list.")

    success = write_json(data_path("macro_news.json"), output)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
