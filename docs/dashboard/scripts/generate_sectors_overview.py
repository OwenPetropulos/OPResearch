"""
generate_sectors_overview.py

Derives sectors_overview.json from the already-generated sector_news.json.
Uses rule-based sentiment scoring and config-defined ticker defaults.
Must run AFTER generate_sector_news.py.
"""

import sys

from common import (
    data_path, read_json, write_json, utc_now_iso,
    score_sentiment, log,
)
from config import SECTOR_DEFAULT_TICKERS, SECTOR_KEYWORDS, TICKER_KEYWORDS

ALL_SECTORS = ["Energy", "Financials", "Technology", "Industrials", "Consumer", "Healthcare", "Macro"]


# ============================================================
# DRIVER EXTRACTION
# ============================================================

def extract_primary_driver(stories: list[dict]) -> str:
    """Return the title of the highest-ranked story as primary driver text."""
    if not stories:
        return "No recent stories"
    # Use the first (most recent) story's title, truncated
    title = stories[0].get("title", "")
    return title[:100] if title else "Recent developments"


def extract_key_drivers(stories: list[dict], max_drivers: int = 4) -> list[str]:
    """
    Extract key driver strings from story titles.
    Returns up to max_drivers short strings.
    """
    drivers = []
    for story in stories[:max_drivers]:
        title = story.get("title", "").strip()
        if title:
            # Truncate long titles to keep UI clean
            drivers.append(title[:120])
    return drivers


def extract_trending_tickers(stories: list[dict], sector_defaults: list[str]) -> list[str]:
    """
    Find tickers that appear most frequently across story ticker_tags.
    Fall back to sector defaults if no tickers are found in stories.
    """
    from collections import Counter
    counter: Counter = Counter()

    for story in stories:
        for ticker in story.get("ticker_tags", []):
            counter[ticker] += 1

    # Top 4 by frequency
    trending = [t for t, _ in counter.most_common(4)]

    if not trending:
        trending = sector_defaults[:3]

    return trending


# ============================================================
# SECTOR TONE TEMPLATES
# ============================================================

TONE_TEMPLATES = {
    "Positive": "{sector} outlook constructive. Recent developments support near-term strength.",
    "Negative": "{sector} facing headwinds. Monitor key support levels and guidance revisions.",
    "Neutral":  "{sector} showing mixed signals. Await clearer catalysts before adding exposure.",
}


def build_sector_tone(sector: str, sentiment: str) -> str:
    template = TONE_TEMPLATES.get(sentiment, TONE_TEMPLATES["Neutral"])
    return template.format(sector=sector)


# ============================================================
# MAIN
# ============================================================

def main():
    log.info("=== generate_sectors_overview.py ===")

    # Load sector_news.json (must exist)
    sector_news_data = read_json(data_path("sector_news.json"))
    if not sector_news_data or "stories" not in sector_news_data:
        log.error("sector_news.json not found or invalid. Run generate_sector_news.py first.")
        sys.exit(1)

    all_stories = sector_news_data["stories"]

    sectors_output = []

    for sector in ALL_SECTORS:
        # Filter stories for this sector
        sector_stories = [
            s for s in all_stories
            if s.get("sector") == sector
            or sector in s.get("sector_tags", [])
        ]

        defaults       = SECTOR_DEFAULT_TICKERS.get(sector, {})
        key_tickers    = defaults.get("key", [])
        default_trend  = defaults.get("trending", [])

        sentiment      = score_sentiment(sector_stories)
        primary_driver = extract_primary_driver(sector_stories)
        key_drivers    = extract_key_drivers(sector_stories)
        trending       = extract_trending_tickers(sector_stories, default_trend)
        tone           = build_sector_tone(sector, sentiment)

        # Pad key_drivers with a generic message if too few stories
        if not key_drivers:
            key_drivers = [f"No major {sector.lower()} stories in current update cycle."]

        sectors_output.append({
            "sector":           sector,
            "sentiment":        sentiment,
            "tone":             tone,
            "story_count":      len(sector_stories),
            "primary_driver":   primary_driver,
            "key_drivers":      key_drivers,
            "key_tickers":      key_tickers,
            "trending_tickers": trending,
        })

        log.info(f"{sector}: {len(sector_stories)} stories, sentiment={sentiment}")

    output  = {"sectors": sectors_output}
    success = write_json(data_path("sectors_overview.json"), output)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
