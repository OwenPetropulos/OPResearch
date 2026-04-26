"""
config.py — Central configuration for OPResearch data pipeline.

Edit this file to adjust symbols, feeds, sector mappings, and ticker universes.
All scripts import from here — nothing is hardcoded in the generators.
"""

# ============================================================
# MARKET SNAPSHOT SYMBOLS
# Uses Yahoo Finance ticker symbols.
# ============================================================

SNAPSHOT_SYMBOLS = {
    "equities": [
        {"label": "S&P 500",       "ticker": "SPY",   "display_ticker": "ES"},
        {"label": "Nasdaq",        "ticker": "QQQ",   "display_ticker": "NQ"},
        {"label": "Dow Jones",     "ticker": "DIA",   "display_ticker": "YM"},
        {"label": "Russell 2000",  "ticker": "IWM",   "display_ticker": "RTY"},
        {"label": "VIX",           "ticker": "^VIX",  "display_ticker": "VIX"},
    ],
    "rates": [
        # ETF proxies — direct sovereign yield feeds are unreliable from free sources.
        # SHY ~ 1-3Y UST, IEF ~ 7-10Y UST. Yield is approximated from price movement.
        # For reference yield levels, we use static fallbacks with live delta adjustments.
        {"label": "US 2Y",       "ticker": "^IRX",  "display_ticker": "US2Y",  "is_yield": True},
        {"label": "US 10Y",      "ticker": "^TNX",  "display_ticker": "US10Y", "is_yield": True},
        {"label": "Japan 10Y",   "ticker": "^JGBL", "display_ticker": "JP10Y", "is_yield": True,  "fallback": 1.55},
        {"label": "UK 10Y",      "ticker": "^TMBMKGB-10Y", "display_ticker": "UK10Y", "is_yield": True, "fallback": 4.40},
        {"label": "Eurozone 10Y","ticker": "^TMBMKDE-10Y", "display_ticker": "DE10Y", "is_yield": True, "fallback": 2.50},
        {"label": "China 10Y",   "ticker": "^TMBMKCN-10Y", "display_ticker": "CN10Y", "is_yield": True, "fallback": 2.30},
    ],
    "commodities": [
        {"label": "Crude Oil",  "ticker": "CL=F",  "display_ticker": "CL"},
        {"label": "Gold",       "ticker": "GC=F",  "display_ticker": "GC"},
        {"label": "Silver",     "ticker": "SI=F",  "display_ticker": "SI"},
        {"label": "Copper",     "ticker": "HG=F",  "display_ticker": "HG"},
    ],
    "global_markets": {
        "asia": [
            {"label": "Nikkei 225",   "ticker": "^N225"},
            {"label": "Hang Seng",    "ticker": "^HSI"},
            {"label": "Shanghai Comp","ticker": "000001.SS"},
            {"label": "KOSPI",        "ticker": "^KS11"},
        ],
        "europe": [
            {"label": "FTSE 100",     "ticker": "^FTSE"},
            {"label": "DAX",          "ticker": "^GDAXI"},
            {"label": "CAC 40",       "ticker": "^FCHI"},
            {"label": "Euro Stoxx 50","ticker": "^STOXX50E"},
        ],
    },
}

# ============================================================
# PORTFOLIO PRICE TICKER UNIVERSE
# All tickers we want live prices for in portfolio_prices.json.
# ============================================================

PRICE_TICKERS = [
    # Tech
    "NVDA", "MSFT", "GOOGL", "AMZN", "META", "AAPL", "AMD",
    # Financials
    "JPM", "GS", "BAC", "MS", "KRE", "ZION", "WAL",
    # Healthcare
    "LLY", "NVO", "AMGN", "PFE", "VRTX",
    # Energy
    "XOM", "CVX", "OXY", "LNG", "RIG", "VAL",
    # Industrials
    "GE", "RTX", "LMT", "NOC", "HON", "CAT",
    # Consumer
    "NKE", "WMT", "COST", "TGT", "ONON", "LULU",
    # Macro ETFs
    "GLD", "TLT", "TIP", "SPY", "QQQ", "IWM", "GDX",
    # Commodity ETFs
    "USO", "SLV", "COPX",
]
# CASH is always 1.00 — added programmatically, not fetched.

# ============================================================
# RSS FEED DEFINITIONS
# source_name, url, source_type
# source_type must be one of: Mainstream | Reddit | Macro Data | Filing / IR | Niche / Blog
# ============================================================

RSS_FEEDS = [
    # Mainstream financial news
    {
        "source_name": "Reuters Business",
        "url": "https://feeds.reuters.com/reuters/businessNews",
        "source_type": "Mainstream",
    },
    {
        "source_name": "Reuters Markets",
        "url": "https://feeds.reuters.com/reuters/financialsNews",
        "source_type": "Mainstream",
    },
    {
        "source_name": "CNBC Top News",
        "url": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
        "source_type": "Mainstream",
    },
    {
        "source_name": "MarketWatch",
        "url": "https://feeds.marketwatch.com/marketwatch/topstories/",
        "source_type": "Mainstream",
    },
    {
        "source_name": "Seeking Alpha",
        "url": "https://seekingalpha.com/feed.xml",
        "source_type": "Niche / Blog",
    },
    {
        "source_name": "Calculated Risk",
        "url": "https://feeds.feedburner.com/CalculatedRisk",
        "source_type": "Niche / Blog",
    },
    {
        "source_name": "Federal Reserve",
        "url": "https://www.federalreserve.gov/feeds/press_all.xml",
        "source_type": "Macro Data",
    },
    {
        "source_name": "U.S. Treasury",
        "url": "https://home.treasury.gov/system/files/rss/press-releases.rss",
        "source_type": "Macro Data",
    },
    {
        "source_name": "SEC EDGAR",
        "url": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&dateb=&owner=include&count=20&output=atom",
        "source_type": "Filing / IR",
    },
]

# ============================================================
# SECTOR KEYWORD MAPS
# Used to classify stories into sectors by keyword matching.
# Order matters: first match wins for primary sector assignment.
# ============================================================

SECTOR_KEYWORDS = {
    "Energy": [
        "oil", "crude", "wti", "brent", "opec", "natural gas", "lng", "shale",
        "permian", "offshore", "driller", "refiner", "exxon", "chevron", "oxy",
        "pioneer", "halliburton", "schlumberger", "transocean", "valaris",
        "energy sector", "fossil fuel", "petroleum", "pipeline", "midstream",
    ],
    "Financials": [
        "fed", "federal reserve", "interest rate", "rate cut", "rate hike",
        "jpmorgan", "goldman sachs", "bank of america", "morgan stanley",
        "regional bank", "cre", "commercial real estate", "nim", "net interest",
        "credit card", "private equity", "investment bank", "hedge fund",
        "insurance", "fintech", "treasury yield", "yield curve", "bond market",
        "lending", "deposit", "fdic", "basel",
    ],
    "Technology": [
        "nvidia", "microsoft", "google", "amazon", "meta", "apple", "amd",
        "semiconductor", "chip", "ai", "artificial intelligence", "cloud",
        "azure", "aws", "software", "data center", "gpu", "machine learning",
        "openai", "anthropic", "llm", "cyber", "cybersecurity", "saas",
        "tech sector", "venture capital", "startup", "ipo tech",
    ],
    "Industrials": [
        "defense", "aerospace", "boeing", "lockheed", "raytheon", "northrop",
        "general electric", "caterpillar", "deere", "honeywell", "pmi",
        "manufacturing", "supply chain", "reshoring", "infrastructure",
        "nato", "military", "procurement", "industrial production",
        "freight", "railroad", "logistics", "construction",
    ],
    "Consumer": [
        "retail", "consumer spending", "walmart", "target", "amazon retail",
        "nike", "lululemon", "consumer confidence", "holiday sales",
        "e-commerce", "discretionary", "staples", "restaurants", "travel",
        "airline", "hotel", "leisure", "tariff consumer", "price hike",
        "inflation consumer", "spending",
    ],
    "Healthcare": [
        "fda", "drug approval", "pharmaceutical", "biotech", "clinical trial",
        "eli lilly", "pfizer", "novo nordisk", "amgen", "glp-1", "obesity",
        "weight loss", "cancer", "oncology", "vaccine", "medicare", "medicaid",
        "health insurance", "hospital", "medical device", "nih",
    ],
    "Macro": [
        "gdp", "inflation", "cpi", "pce", "unemployment", "jobs report",
        "nonfarm payroll", "recession", "stagflation", "central bank",
        "ecb", "boj", "bank of england", "pboc", "imf", "world bank",
        "china economy", "global growth", "emerging market", "currency",
        "dollar", "yen", "euro", "geopolitical", "trade war", "tariff",
        "gold rally", "vix", "risk off", "risk sentiment",
    ],
}

# ============================================================
# TICKER KEYWORD MAPS
# Maps keywords in headlines/summaries to ticker symbols.
# Used to populate ticker_tags in story output.
# ============================================================

TICKER_KEYWORDS = {
    "NVDA":  ["nvidia", "nvda"],
    "MSFT":  ["microsoft", "azure", "msft"],
    "GOOGL": ["google", "alphabet", "googl"],
    "AMZN":  ["amazon", "aws", "amzn"],
    "META":  ["meta", "facebook", "instagram"],
    "AAPL":  ["apple", "aapl", "iphone"],
    "AMD":   ["amd", "advanced micro"],
    "JPM":   ["jpmorgan", "jp morgan", "jpm"],
    "GS":    ["goldman sachs", "goldman"],
    "BAC":   ["bank of america", "bac"],
    "MS":    ["morgan stanley"],
    "KRE":   ["regional bank", "kre"],
    "LLY":   ["eli lilly", "lilly", "lly", "orforglipron"],
    "NVO":   ["novo nordisk", "ozempic", "wegovy", "semaglutide", "nvo"],
    "AMGN":  ["amgen", "amgn"],
    "PFE":   ["pfizer", "pfe"],
    "XOM":   ["exxon", "xom"],
    "CVX":   ["chevron", "cvx"],
    "OXY":   ["occidental", "oxy"],
    "LNG":   ["cheniere", "lng"],
    "RIG":   ["transocean", "rig"],
    "GE":    ["general electric", "ge aerospace"],
    "RTX":   ["raytheon", "rtx"],
    "LMT":   ["lockheed", "lmt"],
    "NOC":   ["northrop", "noc"],
    "NKE":   ["nike", "nke"],
    "WMT":   ["walmart", "wmt"],
    "COST":  ["costco", "cost"],
    "TGT":   ["target", "tgt"],
    "GLD":   ["gold etf", "gld", "gold price", "gold rally"],
    "TLT":   ["tlt", "20-year treasury", "long bond"],
    "TIP":   ["tips", "tip etf", "inflation-linked"],
    "SPY":   ["s&p 500", "sp500", "spy"],
    "QQQ":   ["nasdaq 100", "qqq"],
    "VIX":   ["vix", "volatility index"],
    "CL=F":  ["wti", "crude oil", "oil price"],
    "GC=F":  ["gold futures", "comex gold"],
}

# ============================================================
# SECTOR DEFAULT TICKERS
# Fallback key/trending tickers per sector for sectors_overview.
# ============================================================

SECTOR_DEFAULT_TICKERS = {
    "Energy":      {"key": ["XOM", "CVX", "OXY", "LNG", "RIG"],     "trending": ["OXY", "RIG"]},
    "Financials":  {"key": ["JPM", "GS", "BAC", "MS", "KRE"],       "trending": ["JPM", "GS"]},
    "Technology":  {"key": ["NVDA", "MSFT", "GOOGL", "AMZN", "AMD"],"trending": ["NVDA", "MSFT"]},
    "Industrials": {"key": ["GE", "RTX", "LMT", "NOC", "HON"],      "trending": ["LMT", "RTX"]},
    "Consumer":    {"key": ["NKE", "WMT", "COST", "TGT", "ONON"],   "trending": ["NKE", "WMT"]},
    "Healthcare":  {"key": ["LLY", "NVO", "AMGN", "PFE", "VRTX"],   "trending": ["LLY", "NVO"]},
    "Macro":       {"key": ["TLT", "GLD", "TIP", "SPY", "VIX"],     "trending": ["GLD", "VIX"]},
}

# ============================================================
# SECTOR SENTIMENT KEYWORDS
# Used to score sentiment in story headlines/summaries.
# ============================================================

POSITIVE_WORDS = [
    "beat", "surge", "rise", "rally", "gain", "strong", "record", "expansion",
    "growth", "outperform", "upgrade", "approval", "approved", "bullish",
    "recovery", "rebound", "positive", "exceed", "upside", "acceleration",
]

NEGATIVE_WORDS = [
    "miss", "fall", "drop", "decline", "weak", "concern", "risk", "pressure",
    "downgrade", "recession", "slowdown", "cut", "loss", "negative", "bearish",
    "slump", "contraction", "stress", "default", "warning", "tariff hit",
]

# ============================================================
# WHY IT MATTERS TEMPLATES
# Deterministic templates keyed by story category.
# ============================================================

WHY_IT_MATTERS_TEMPLATES = {
    "inflation": (
        "Persistent inflation constrains Fed policy optionality and pressures "
        "duration assets via higher discount rates. Watch CPI components for "
        "breadth signals and their effect on rate cut timelines."
    ),
    "fed_rates": (
        "Fed policy signals directly set the risk-free rate baseline for all "
        "asset valuations. Rate path repricing creates volatility in both "
        "equities and fixed income. Monitor forward guidance and dot plot revisions."
    ),
    "china_growth": (
        "China's growth trajectory drives demand for commodities, luxury goods, "
        "and EM assets. Policy stimulus announcements from Beijing tend to lift "
        "commodity-linked equities and risk sentiment globally."
    ),
    "crude_oil": (
        "Oil price moves cascade through energy equities, transportation costs, "
        "and headline inflation. OPEC+ supply discipline and global demand signals "
        "are the key variables to monitor."
    ),
    "treasury_auction": (
        "Treasury auction demand sets marginal price discovery for U.S. sovereign "
        "yields. Weak demand can accelerate yield selloffs and create headwinds "
        "for rate-sensitive equities. Watch bid-to-cover ratios and foreign participation."
    ),
    "risk_sentiment": (
        "Broad risk sentiment shifts affect cross-asset positioning simultaneously. "
        "VIX elevation and put/call ratio moves can signal near-term volatility risk "
        "or contrarian opportunities depending on accompanying fundamental context."
    ),
    "tech_ai": (
        "AI infrastructure spending drives semiconductor, cloud, and data center demand. "
        "Hyperscaler capex commitments are the clearest leading indicator for "
        "AI-adjacent equities. Monitor GPU supply chains and inference cost curves."
    ),
    "financials": (
        "Bank earnings and net interest margin trends reflect the transmission of "
        "monetary policy to the real economy. IB fee recovery signals M&A and "
        "capital markets cycle health."
    ),
    "healthcare": (
        "FDA approvals and clinical data releases create binary event risk and "
        "opportunity in healthcare equities. GLP-1 competitive dynamics remain "
        "a dominant cross-sector theme with consumer and payer implications."
    ),
    "industrials": (
        "Defense procurement and reshoring capex are durable secular tailwinds "
        "for industrial names with government contract exposure. PMI readings "
        "provide the clearest leading indicator of order book direction."
    ),
    "consumer": (
        "Consumer spending data reveals the health of the most important driver "
        "of U.S. GDP. Bifurcation between value and premium cohorts remains a "
        "key theme. Watch credit card delinquency trends for stress signals."
    ),
    "default": (
        "This development may affect near-term price action and sector positioning. "
        "Monitor related earnings guidance, macro data, and management commentary "
        "for confirming signals before adjusting portfolio exposure."
    ),
}
