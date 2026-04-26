"""
common.py — Shared utilities for the OPResearch data pipeline.
"""

import json
import os
import re
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import yfinance as yf

from config import (
    SECTOR_KEYWORDS,
    TICKER_KEYWORDS,
    POSITIVE_WORDS,
    NEGATIVE_WORDS,
    WHY_IT_MATTERS_TEMPLATES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("opr")

# ============================================================
# PATHS
# ============================================================

REPO_ROOT = Path(__file__).resolve().parent.parent

_env_data_dir = os.environ.get("OPR_DATA_DIR", "").strip()
if _env_data_dir:
    DATA_DIR = REPO_ROOT / _env_data_dir
    log.info(f"DATA_DIR overridden by OPR_DATA_DIR: {DATA_DIR}")
else:
    DATA_DIR = REPO_ROOT / "data"

log.info(f"DATA_DIR resolved to: {DATA_DIR}")


def data_path(filename: str) -> Path:
    return DATA_DIR / filename


# ============================================================
# JSON I/O
# ============================================================

def write_json(path: Path, data: dict | list, indent: int = 2) -> bool:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        tmp.replace(path)
        log.info(f"Wrote {path}")
        return True
    except Exception as e:
        log.error(f"Failed to write {path}: {e}")
        return False


def read_json(path: Path) -> dict | list | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"Could not read {path}: {e}")
        return None


# ============================================================
# TIMESTAMP UTILITIES
# ============================================================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_timestamp(ts_string: str | None) -> datetime | None:
    if not ts_string:
        return None
    formats = [
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S GMT",
        "%Y-%m-%d %H:%M:%S",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(ts_string.strip(), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return None


def to_iso(dt: datetime | None) -> str:
    if dt is None:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# ============================================================
# YFINANCE HELPERS
# ============================================================

def fetch_price_data(ticker: str, period: str = "5d", interval: str = "1d") -> dict | None:
    """
    Fetch price data for a single ticker.
    Returns price, prev_close, change, percent_change, direction.
    NOTE: For CBOE rate indices (^TNX, ^IRX), Yahoo Finance returns
    the yield directly as a percentage (e.g. 4.63 for 4.63%).
    Do NOT divide by 10 — that was incorrect.
    """
    try:
        tk     = yf.Ticker(ticker)
        hist   = tk.history(period=period, interval=interval, auto_adjust=True)
        if hist.empty or len(hist) < 1:
            log.warning(f"No price history for {ticker}")
            return None

        closes     = hist["Close"].dropna()
        if closes.empty:
            log.warning(f"No close prices for {ticker}")
            return None

        price      = float(closes.iloc[-1])
        prev_close = float(closes.iloc[-2]) if len(closes) >= 2 else price
        change     = round(price - prev_close, 4)
        pct_change = round((change / prev_close) * 100, 2) if prev_close else 0.0
        direction  = "up" if change > 0 else ("down" if change < 0 else "flat")

        return {
            "price":          round(price, 4),
            "prev_close":     round(prev_close, 4),
            "change":         round(change, 4),
            "percent_change": pct_change,
            "direction":      direction,
        }
    except Exception as e:
        log.warning(f"yfinance fetch failed for {ticker}: {e}")
        return None


def fetch_prices_bulk(tickers: list[str]) -> dict[str, float]:
    """
    Fetch latest closing price for multiple tickers individually.
    Avoids MultiIndex issues with yfinance v1.3+.
    """
    results = {}
    if not tickers:
        return results

    for ticker in tickers:
        try:
            tk     = yf.Ticker(ticker)
            hist   = tk.history(period="5d", interval="1d", auto_adjust=True)
            if hist.empty:
                log.warning(f"No data for {ticker}")
                continue
            closes = hist["Close"].dropna()
            if closes.empty:
                continue
            price = round(float(closes.iloc[-1]), 4)
            results[ticker] = price
            log.info(f"  {ticker}: {price}")
        except Exception as e:
            log.warning(f"Price fetch failed for {ticker}: {e}")

    return results


# ============================================================
# RSS FEED PARSING
# ============================================================

def fetch_feed(feed_def: dict, max_entries: int = 20) -> list[dict]:
    source_name = feed_def.get("source_name", "Unknown")
    url         = feed_def.get("url", "")
    source_type = feed_def.get("source_type", "Mainstream")

    stories = []
    try:
        parsed = feedparser.parse(url, agent="OPResearch/1.0")
        if parsed.bozo and not parsed.entries:
            log.warning(f"Feed parse error for {source_name}: {parsed.bozo_exception}")
            return []

        for entry in parsed.entries[:max_entries]:
            title   = clean_text(getattr(entry, "title", ""))
            summary = clean_text(getattr(entry, "summary", "") or getattr(entry, "description", ""))
            link    = getattr(entry, "link", "")

            published = getattr(entry, "published", None) or getattr(entry, "updated", None)
            dt        = parse_timestamp(published)
            timestamp = to_iso(dt)

            if not title:
                continue

            stories.append({
                "title":       title,
                "summary":     summary[:600] if summary else "",
                "url":         link,
                "source_name": source_name,
                "source_type": source_type,
                "timestamp":   timestamp,
                "_dt":         dt,
            })

        log.info(f"Fetched {len(stories)} entries from {source_name}")
    except Exception as e:
        log.error(f"Failed to fetch feed {source_name} ({url}): {e}")

    return stories


def deduplicate_stories(stories: list[dict]) -> list[dict]:
    seen_urls   = set()
    seen_hashes = set()
    unique      = []

    for s in stories:
        url   = s.get("url", "")
        title = s.get("title", "")

        if url and url in seen_urls:
            continue

        title_key = re.sub(r"\s+", " ", title.lower().strip())[:80]
        h         = hashlib.md5(title_key.encode()).hexdigest()
        if h in seen_hashes:
            continue

        if url:
            seen_urls.add(url)
        seen_hashes.add(h)
        unique.append(s)

    return unique


def sort_stories_by_time(stories: list[dict]) -> list[dict]:
    return sorted(
        stories,
        key=lambda s: s.get("_dt") or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )


def strip_internal_fields(stories: list[dict]) -> list[dict]:
    return [{k: v for k, v in s.items() if not k.startswith("_")} for s in stories]


# ============================================================
# TEXT UTILITIES
# ============================================================

def clean_text(raw: str) -> str:
    if not raw:
        return ""
    text = re.sub(r"<[^>]+>", " ", raw)
    entities = {
        "&amp;": "&", "&lt;": "<", "&gt;": ">",
        "&quot;": '"', "&#39;": "'", "&nbsp;": " ",
    }
    for ent, char in entities.items():
        text = text.replace(ent, char)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def make_story_id(title: str, source_name: str) -> str:
    raw = (title + source_name).lower().encode()
    return hashlib.md5(raw).hexdigest()[:8]


# ============================================================
# CLASSIFICATION HELPERS
# ============================================================

def classify_sectors(text: str) -> list[str]:
    lower   = text.lower()
    matched = []
    for sector, keywords in SECTOR_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            matched.append(sector)
    return matched if matched else ["Macro"]


def classify_tickers(text: str) -> list[str]:
    lower   = text.lower()
    matched = []
    for ticker, keywords in TICKER_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            if ticker not in matched:
                matched.append(ticker)
    return matched[:6]


def classify_macro_category(text: str) -> str:
    """
    Return best-matching category key for why_it_matters template selection.
    Order matters — more specific rules checked first.
    """
    lower = text.lower()
    rules = [
        ("currency_fx",      [
            "exorbitant privilege", "dedollarization", "reserve currency",
            "dollar hegemony", "currency war", "yen carry", "carry trade",
            "usd/jpy", "eur/usd", "dxy", "dollar index", "currency debasement",
            "devaluation", "dollar weakness", "dollar strength", "petrodollar",
            "forex", "exchange rate", "fx market",
        ]),
        ("geopolitical",     [
            "sanctions", "war", "conflict", "nato", "iran", "russia", "ukraine",
            "taiwan", "geopolitical", "shadow fleet", "weapons",
        ]),
        ("inflation",        ["cpi", "inflation", "pce", "core prices", "price index", "stagflation"]),
        ("fed_rates",        ["federal reserve", "fed", "fomc", "rate cut", "rate hike", "dot plot", "powell", "warsh", "fed chair"]),
        ("china_growth",     ["china", "pboc", "beijing", "chinese economy", "shanghai", "caixin"]),
        ("crude_oil",        ["crude", "wti", "brent", "opec", "oil price", "tanker", "oil inventory"]),
        ("treasury_auction", ["treasury auction", "bond auction", "bid-to-cover", "10-year note", "debt ceiling"]),
        ("risk_sentiment",   ["vix", "risk off", "put/call", "volatility", "gamma", "dealer positioning"]),
        ("tech_ai",          ["nvidia", "ai", "artificial intelligence", "semiconductor", "cloud", "azure", "gpu", "data center"]),
        ("financials",       ["bank", "jpmorgan", "goldman", "nim", "lending", "deposit", "credit", "fomc"]),
        ("healthcare",       ["fda", "pharma", "glp-1", "biotech", "drug approval", "obesity", "clinical"]),
        ("industrials",      ["defense", "manufacturing", "pmi", "aerospace", "lockheed", "pentagon"]),
        ("consumer",         ["retail", "consumer spending", "walmart", "nike", "discretionary", "personal finance"]),
    ]
    for category, keywords in rules:
        if any(kw in lower for kw in keywords):
            return category
    return "default"


def get_why_it_matters(text: str) -> str:
    category = classify_macro_category(text)
    return WHY_IT_MATTERS_TEMPLATES.get(category, WHY_IT_MATTERS_TEMPLATES["default"])


def score_sentiment(stories: list[dict]) -> str:
    pos_score = 0
    neg_score = 0
    for story in stories:
        text = (story.get("title", "") + " " + story.get("summary", "")).lower()
        pos_score += sum(1 for w in POSITIVE_WORDS if w in text)
        neg_score += sum(1 for w in NEGATIVE_WORDS if w in text)

    if pos_score > neg_score + 1:
        return "Positive"
    if neg_score > pos_score + 1:
        return "Negative"
    return "Neutral"
