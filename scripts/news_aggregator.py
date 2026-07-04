"""
news_aggregator.py
Fetches financial news from multiple sources and deduplicates.
Sources: MarketAux API, Yahoo Finance, Alpha Vantage, RSS feeds
"""

import json
import requests
from datetime import datetime, timedelta
from typing import List, Dict
import hashlib
import re

# ─────────────────────────────────────────────────────────────
# API KEYS & ENDPOINTS
# ─────────────────────────────────────────────────────────────

MARKETAUX_API_KEY = "your_marketaux_key_here"  # Set via env var or .env
ALPHA_VANTAGE_API_KEY = "your_av_key_here"

MARKETAUX_BASE = "https://api.marketaux.com/v1"
YAHOO_FINANCE_BASE = "https://query2.finance.yahoo.com"
ALPHA_VANTAGE_BASE = "https://www.alphavantage.co"

# RSS feeds for fallback
RSS_FEEDS = [
    "https://feeds.bloomberg.com/markets/news.rss",
    "https://feeds.cnbc.com/cnbc/financialnews/",
    "https://feeds.bloomberg.com/markets/news.rss",
    "https://feeds.finance.yahoo.com/rss/2.0/headlines",
]

# ─────────────────────────────────────────────────────────────
# DEDUPLICATION
# ─────────────────────────────────────────────────────────────

def story_fingerprint(title: str) -> str:
    """Generate a fingerprint for a story title (lowercase, remove punctuation)."""
    normalized = re.sub(r'[^a-z0-9\s]', '', title.lower()).strip()
    return hashlib.md5(normalized.encode()).hexdigest()

def deduplicate_stories(stories: List[Dict]) -> List[Dict]:
    """
    Remove duplicate stories based on title similarity.
    Keeps first occurrence, merges source attribution.
    """
    seen = {}
    deduplicated = []
    
    for story in stories:
        fp = story_fingerprint(story.get("title", ""))
        
        if fp in seen:
            # Merge sources
            existing = seen[fp]
            existing_sources = existing.get("sources", [existing.get("source_name")])
            new_source = story.get("source_name")
            if new_source not in existing_sources:
                existing_sources.append(new_source)
            existing["sources"] = existing_sources
        else:
            # New story
            story["sources"] = [story.get("source_name")]
            deduplicated.append(story)
            seen[fp] = story
    
    return deduplicated

# ─────────────────────────────────────────────────────────────
# MARKETAUX (Primary Source)
# ─────────────────────────────────────────────────────────────

def fetch_marketaux_news(limit: int = 50) -> List[Dict]:
    """Fetch financial news from MarketAux API."""
    if not MARKETAUX_API_KEY or MARKETAUX_API_KEY == "your_marketaux_key_here":
        print("[WARN] MarketAux API key not set. Skipping MarketAux source.")
        return []
    
    try:
        url = f"{MARKETAUX_BASE}/news/all"
        params = {
            "api_token": MARKETAUX_API_KEY,
            "limit": limit,
            "sort": "latest",
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        stories = []
        for item in data.get("data", []):
            stories.append({
                "title": item.get("title", ""),
                "summary": item.get("description", ""),
                "url": item.get("url", ""),
                "source_name": item.get("source", "MarketAux"),
                "source_type": "News Aggregator",
                "timestamp": item.get("published_at", ""),
                "sector_tags": item.get("entities", []),
                "ticker_tags": item.get("tickers", []),
            })
        
        print(f"[OK] Fetched {len(stories)} stories from MarketAux")
        return stories
    except Exception as e:
        print(f"[ERROR] MarketAux fetch failed: {e}")
        return []

# ─────────────────────────────────────────────────────────────
# YAHOO FINANCE NEWS
# ─────────────────────────────────────────────────────────────

def fetch_yahoo_finance_news(limit: int = 30) -> List[Dict]:
    """Fetch financial news from Yahoo Finance."""
    try:
        url = f"{YAHOO_FINANCE_BASE}/v10/finance/quoteSummary/^GSPC"
        params = {"modules": "news"}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        stories = []
        for item in data.get("quoteSummary", {}).get("result", [{}])[0].get("news", [])[:limit]:
            stories.append({
                "title": item.get("title", ""),
                "summary": item.get("summary", ""),
                "url": item.get("link", ""),
                "source_name": item.get("source", "Yahoo Finance"),
                "source_type": "Mainstream",
                "timestamp": datetime.fromtimestamp(item.get("providerPublishTime")).isoformat() + "Z",
                "sector_tags": [],
                "ticker_tags": item.get("related", []),
            })
        
        print(f"[OK] Fetched {len(stories)} stories from Yahoo Finance")
        return stories
    except Exception as e:
        print(f"[WARN] Yahoo Finance fetch failed: {e}")
        return []

# ─────────────────────────────────────────────────────────────
# ALPHA VANTAGE NEWS
# ─────────────────────────────────────────────────────────────

def fetch_alpha_vantage_news(limit: int = 30) -> List[Dict]:
    """Fetch financial news from Alpha Vantage news sentiment API."""
    if not ALPHA_VANTAGE_API_KEY or ALPHA_VANTAGE_API_KEY == "your_av_key_here":
        print("[WARN] Alpha Vantage API key not set. Skipping.")
        return []
    
    try:
        url = f"{ALPHA_VANTAGE_BASE}/query"
        params = {
            "function": "NEWS_SENTIMENT",
            "apikey": ALPHA_VANTAGE_API_KEY,
            "limit": limit,
            "sort": "LATEST",
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        stories = []
        for item in data.get("feed", [])[:limit]:
            stories.append({
                "title": item.get("title", ""),
                "summary": item.get("summary", ""),
                "url": item.get("url", ""),
                "source_name": item.get("source", "Alpha Vantage"),
                "source_type": "News Feed",
                "timestamp": item.get("time_published", ""),
                "sector_tags": item.get("topics", []),
                "ticker_tags": [t["ticker"] for t in item.get("tickers", [])],
            })
        
        print(f"[OK] Fetched {len(stories)} stories from Alpha Vantage")
        return stories
    except Exception as e:
        print(f"[WARN] Alpha Vantage fetch failed: {e}")
        return []

# ─────────────────────────────────────────────────────────────
# RSS FALLBACK
# ─────────────────────────────────────────────────────────────

def fetch_rss_feeds() -> List[Dict]:
    """Fallback: fetch news from RSS feeds (requires feedparser)."""
    try:
        import feedparser
    except ImportError:
        print("[WARN] feedparser not installed. Skipping RSS feeds.")
        return []
    
    stories = []
    for feed_url in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:10]:
                stories.append({
                    "title": entry.get("title", ""),
                    "summary": entry.get("summary", ""),
                    "url": entry.get("link", ""),
                    "source_name": feed.feed.get("title", "RSS Feed"),
                    "source_type": "RSS",
                    "timestamp": entry.get("published", ""),
                    "sector_tags": [],
                    "ticker_tags": [],
                })
        except Exception as e:
            print(f"[WARN] RSS fetch failed for {feed_url}: {e}")
    
    print(f"[OK] Fetched {len(stories)} stories from RSS feeds")
    return stories

# ─────────────────────────────────────────────────────────────
# AGGREGATE & DEDUPLICATE
# ─────────────────────────────────────────────────────────────

def aggregate_all_news(limit: int = 50) -> List[Dict]:
    """Fetch from all sources and deduplicate."""
    all_stories = []
    
    # Primary sources
    all_stories.extend(fetch_marketaux_news(limit))
    all_stories.extend(fetch_yahoo_finance_news(limit // 2))
    all_stories.extend(fetch_alpha_vantage_news(limit // 2))
    
    # Fallback
    if len(all_stories) < limit // 2:
        all_stories.extend(fetch_rss_feeds())
    
    # Deduplicate
    deduplicated = deduplicate_stories(all_stories)
    
    # Sort by timestamp (newest first)
    deduplicated.sort(
        key=lambda x: x.get("timestamp", ""),
        reverse=True
    )
    
    # Assign IDs
    for i, story in enumerate(deduplicated):
        story["id"] = f"story_{i:04d}"
    
    print(f"[OK] Aggregated {len(deduplicated)} unique stories after deduplication")
    return deduplicated[:limit] 
