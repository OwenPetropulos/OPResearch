"""
config.py — Central configuration for OPResearch data pipeline.
"""

SNAPSHOT_SYMBOLS = {
    "equities": [
        {"label": "S&P 500",      "ticker": "^GSPC", "display_ticker": "ES"},
        {"label": "Nasdaq",       "ticker": "^IXIC", "display_ticker": "NQ"},
        {"label": "Dow Jones",    "ticker": "^DJI",  "display_ticker": "YM"},
        {"label": "Russell 2000", "ticker": "^RUT",  "display_ticker": "RTY"},
        {"label": "VIX",          "ticker": "^VIX", "display_ticker": "VIX"},
    ],
    "rates": [
        # These Yahoo Finance tickers report in percent directly (e.g. 4.63)
        # NOT in tenths — do NOT divide by 10
      
        {"label": "US 3M",        "ticker": "^IRX",     "display_ticker": "US3M",  "is_yield": True},
        {"label": "US 2Y",        "ticker": "^FVX",     "display_ticker": "US2Y",  "is_yield": True},
        {"label": "US 10Y",       "ticker": "^TNX",     "display_ticker": "US10Y", "is_yield": True},
        {"label": "Japan 10Y",    "ticker": "^JP10Y", "display_ticker": "JP10Y", "is_yield": True, "fallback": 1.55},
        {"label": "UK 10Y",       "ticker": "^UK10Y", "display_ticker": "UK10Y", "is_yield": True, "fallback": 4.40},
        {"label": "China 10Y",    "ticker": "^CBON", "display_ticker": "CN10Y", "is_yield": True, "fallback": 2.30},
    ],
    "commodities": [
        {"label": "Crude Oil", "ticker": "CL=F", "display_ticker": "CL"},
        {"label": "Gold",      "ticker": "GC=F", "display_ticker": "GC"},
        {"label": "Silver",    "ticker": "SI=F", "display_ticker": "SI"},
        {"label": "Copper",    "ticker": "HG=F", "display_ticker": "HG"},
    ],
    "fx": [
        # FX pairs: price = units of quote currency per 1 unit of base
        {"label": "USD/JPY", "ticker": "JPY=X",  "display_ticker": "USDJPY"},
        {"label": "EUR/USD", "ticker": "EURUSD=X","display_ticker": "EURUSD"},
        {"label": "EUR/JPY", "ticker": "EURJPY=X","display_ticker": "EURJPY"},
        {"label": "DXY",     "ticker": "DX-Y.NYB","display_ticker": "DXY"},
    ],
    "global_markets": {
        "asia": [
            {"label": "Nikkei 225",    "ticker": "^N225"},
            {"label": "Hang Seng",     "ticker": "^HSI"},
            {"label": "Shanghai Comp", "ticker": "000001.SS"},
            {"label": "KOSPI",         "ticker": "^KS11"},
        ],
        "europe": [
            {"label": "FTSE 100",      "ticker": "^FTSE"},
            {"label": "DAX",           "ticker": "^GDAXI"},
            {"label": "CAC 40",        "ticker": "^FCHI"},
            {"label": "Euro Stoxx 50", "ticker": "^STOXX50E"},
        ],
    },
}

# ============================================================
# CBOE RATE INDEX TICKERS
# These report as direct percentage values (e.g. 4.63 = 4.63%)
# Do NOT divide by 10 — Yahoo fixed this in recent data
# ============================================================
CBOE_RATE_TICKERS = {"^TNX", "^IRX", "^TYX", "^FVX"}

# ============================================================
# PORTFOLIO PRICE TICKER UNIVERSE
# ============================================================
# In config.py — replace just the PRICE_TICKERS list:

PRICE_TICKERS = [
    # Tech
    "NVDA", "MSFT", "GOOGL", "AMZN", "META", "AAPL", "AMD", "ORCL", "CRM", "NOW", "PLTR",
    # Financials
    "JPM", "GS", "BAC", "MS", "KRE", "ZION", "WAL", "BRK-B", "C", "WFC",
    # Healthcare
    "LLY", "NVO", "AMGN", "PFE", "VRTX", "ABBV", "MRK", "JNJ",
    # Energy
    "XOM", "CVX", "OXY", "LNG", "RIG", "VAL", "ET", "SHEL", "BP", "TTE",
    "COP", "EOG", "PXD", "DVN", "FANG", "MPC", "VLO", "PSX",
    # Industrials
    "GE", "RTX", "LMT", "NOC", "HON", "CAT", "DE", "BA", "GD",
    # Consumer
    "NKE", "WMT", "COST", "TGT", "ONON", "LULU", "AMZN", "HD", "MCD",
    # Macro ETFs
    "GLD", "TLT", "TIP", "SPY", "QQQ", "IWM", "GDX", "SHY", "HYG", "LQD",
    # Commodity ETFs
    "USO", "SLV", "COPX", "UNG",
    # FX ETFs
    "FXY", "FXE", "UUP",
    # International / ADRs
    "BABA", "TSM", "ASML", "SAP", "NVS", "RYCEY",
    # Added Manuelly
    "RRR", "PS", 
]

# ============================================================
# RSS FEED DEFINITIONS
# ============================================================
RSS_FEEDS = [
    {
        "source_name": "Reuters Business",
        "url":         "https://feeds.reuters.com/reuters/businessNews",
        "source_type": "Mainstream",
    },
    {
        "source_name": "Reuters Markets",
        "url":         "https://feeds.reuters.com/reuters/financialsNews",
        "source_type": "Mainstream",
    },
    {
        "source_name": "CNBC Top News",
        "url":         "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
        "source_type": "Mainstream",
    },
    {
        "source_name": "MarketWatch",
        "url":         "https://feeds.marketwatch.com/marketwatch/topstories/",
        "source_type": "Mainstream",
    },
    {
        "source_name": "Seeking Alpha",
        "url":         "https://seekingalpha.com/feed.xml",
        "source_type": "Niche / Blog",
    },
    {
        "source_name": "Calculated Risk",
        "url":         "https://feeds.feedburner.com/CalculatedRisk",
        "source_type": "Niche / Blog",
    },
    {
        "source_name": "Federal Reserve",
        "url":         "https://www.federalreserve.gov/feeds/press_all.xml",
        "source_type": "Macro Data",
    },
    {
        "source_name": "U.S. Treasury",
        "url":         "https://home.treasury.gov/system/files/rss/press-releases.rss",
        "source_type": "Macro Data",
    },
    {
        "source_name": "SEC EDGAR",
        "url":         "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&dateb=&owner=include&count=20&output=atom",
        "source_type": "Filing / IR",
    },
]

# ============================================================
# SECTOR KEYWORD MAPS
# ============================================================
SECTOR_KEYWORDS = {
    "Energy": [
        "oil", "crude", "wti", "brent", "opec", "natural gas", "lng", "shale",
        "permian", "offshore", "driller", "refiner", "exxon", "chevron", "oxy",
        "pioneer", "halliburton", "schlumberger", "transocean", "valaris",
        "energy sector", "fossil fuel", "petroleum", "pipeline", "midstream",
        "tanker", "shipping fuel", "iranian oil", "shadow fleet",
    ],
    "Financials": [
        "fed", "federal reserve", "interest rate", "rate cut", "rate hike",
        "jpmorgan", "goldman sachs", "bank of america", "morgan stanley",
        "regional bank", "cre", "commercial real estate", "nim", "net interest",
        "credit card", "private equity", "investment bank", "hedge fund",
        "insurance", "fintech", "treasury yield", "yield curve", "bond market",
        "lending", "deposit", "fdic", "basel", "fomc", "powell", "warsh",
        "fed chair", "monetary policy", "quantitative", "tightening", "easing", "Capex",
        "Depreciaiton", "Free Cash Flow", 
    ],
    "Technology": [
        "nvidia", "microsoft", "google", "amazon", "meta", "apple", "amd",
        "semiconductor", "chip", "ai", "artificial intelligence", "cloud",
        "azure", "aws", "software", "data center", "gpu", "machine learning",
        "openai", "anthropic", "llm", "cyber", "cybersecurity", "saas",
        "tech sector", "venture capital", "startup", "Karp", "Palantir", "Claude", "Anthropic", 
        "OpenAI", 
    ],
    "Industrials": [
        "defense", "aerospace", "boeing", "lockheed", "raytheon", "northrop",
        "general electric", "caterpillar", "deere", "honeywell", "pmi",
        "manufacturing", "supply chain", "reshoring", "infrastructure",
        "nato", "military", "procurement", "industrial production",
        "freight", "railroad", "logistics", "construction",
    ],
    "Consumer": [
        "retail", "consumer spending", "walmart", "target",
        "nike", "lululemon", "consumer confidence", "holiday sales",
        "e-commerce", "discretionary", "staples", "restaurants", "travel",
        "airline", "hotel", "leisure", "tariff consumer", "price hike",
        "inflation consumer", "spending", "personal finance", "retirement",
        "expat", "housing",
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
        "china economy", "global growth", "emerging market",
        "geopolitical", "trade war", "tariff",
        "gold rally", "vix", "risk off", "risk sentiment",
        # Currency / FX keywords
        "dollar", "yen", "euro", "sterling", "yuan", "renminbi",
        "currency", "forex", "exchange rate", "fx", "devaluation",
        "currency war", "exorbitant privilege", "reserve currency",
        "dollar hegemony", "dedollarization", "dollar weakness",
        "dollar strength", "carry trade", "yen carry", "usd", "jpy", "eur",
        "dxy", "dollar index", "currency debasement", "purchasing power",
        "balance of payments", "current account", "capital flows",
        "bretton woods", "petrodollar", "sanctions currency",
    ],
}

# ============================================================
# TICKER KEYWORD MAPS
# ============================================================
TICKER_KEYWORDS = {
    "NVDA":    ["nvidia", "nvda"],
    "PLTR":    ["Palantir", "pltr", "Karp", "Thiel",]
    "MSFT":    ["microsoft", "azure", "msft"],
    "GOOGL":   ["google", "alphabet", "googl"],
    "AMZN":    ["amazon", "aws", "amzn"],
    "META":    ["meta", "facebook", "instagram"],
    "AAPL":    ["apple", "aapl", "iphone"],
    "AMD":     ["amd", "advanced micro"],
    "JPM":     ["jpmorgan", "jp morgan", "jpm"],
    "GS":      ["goldman sachs", "goldman"],
    "BAC":     ["bank of america", "bac"],
    "MS":      ["morgan stanley"],
    "KRE":     ["regional bank", "kre"],
    "LLY":     ["eli lilly", "lilly", "lly", "orforglipron"],
    "NVO":     ["novo nordisk", "ozempic", "wegovy", "semaglutide", "nvo"],
    "AMGN":    ["amgen", "amgn"],
    "PFE":     ["pfizer", "pfe"],
    "XOM":     ["exxon", "xom"],
    "CVX":     ["chevron", "cvx"],
    "OXY":     ["occidental", "oxy"],
    "LNG":     ["cheniere", "lng"],
    "RIG":     ["transocean", "rig"],
    "GE":      ["general electric", "ge aerospace"],
    "RTX":     ["raytheon", "rtx"],
    "LMT":     ["lockheed", "lmt"],
    "NOC":     ["northrop", "noc"],
    "NKE":     ["nike", "nke"],
    "WMT":     ["walmart", "wmt"],
    "COST":    ["costco", "cost"],
    "TGT":     ["target", "tgt"],
    "GLD":     ["gold etf", "gld", "gold price", "gold rally"],
    "TLT":     ["tlt", "20-year treasury", "long bond"],
    "TIP":     ["tips", "tip etf", "inflation-linked"],
    "SPY":     ["s&p 500", "sp500", "spy"],
    "QQQ":     ["nasdaq 100", "qqq"],
    "VIX":     ["vix", "volatility index"],
    "CL=F":    ["wti", "crude oil", "oil price"],
    "GC=F":    ["gold futures", "comex gold"],
    # FX
    "FXY":     ["yen etf", "fxy", "japanese yen"],
    "FXE":     ["euro etf", "fxe", "european euro"],
    "UUP":     ["dollar etf", "uup", "dollar index etf"],
    "JPY=X":   ["usd/jpy", "dollar yen", "usdjpy", "yen weakness", "yen strength"],
    "EURUSD=X":["eur/usd", "euro dollar", "eurusd"],
    "DX-Y.NYB":["dxy", "dollar index", "usdx"],
}

# ============================================================
# MARKETAUX API CONFIGURATION
#
# MarketAux free tier: 100 requests/day
# With 3 pipeline runs/day:
#   - generate_macro_news.py:   1 request per run  = 3/day
#   - generate_sector_news.py:  7 requests per run = 21/day
#   Total: ~24 requests/day — well within free limit
#
# Topics reference: https://www.marketaux.com/documentation
# ============================================================

MARKETAUX_MACRO_FILTERS = {
    # Broad macro/market topics for the morning brief
    # These are MarketAux topic slugs
    "topics": "central-banks,monetary-policy,inflation,economic-data,currencies,commodities,bonds,stock-markets,geopolitics",
}

MARKETAUX_SECTOR_FILTERS = {
    # Each entry maps a sector name to MarketAux API params
    # MarketAux supports filtering by industry_group, topics, and symbols
    "Energy": {
        "topics": "energy,oil-and-gas,commodities",
    },
    "Financials": {
        "topics": "banking,financial-services,monetary-policy,central-banks",
    },
    "Technology": {
        "topics": "technology,artificial-intelligence,semiconductors,software",
    },
    "Industrials": {
        "topics": "industrials,defense,aerospace,manufacturing",
    },
    "Consumer": {
        "topics": "retail,consumer-goods,e-commerce",
    },
    "Healthcare": {
        "topics": "healthcare,pharmaceuticals,biotech",
    },
    "Macro": {
        "topics": "central-banks,monetary-policy,inflation,economic-data,currencies,bonds,geopolitics",
    },
}

# ============================================================
# SECTOR DEFAULT TICKERS
# ============================================================
SECTOR_DEFAULT_TICKERS = {
    "Energy":      {"key": ["XOM", "CVX", "OXY", "LNG", "RIG"],      "trending": ["OXY", "RIG"]},
    "Financials":  {"key": ["JPM", "GS", "BAC", "MS", "KRE"],        "trending": ["JPM", "GS"]},
    "Technology":  {"key": ["NVDA", "MSFT", "GOOGL", "AMZN", "AMD", "PLTR",], "trending": ["NVDA", "MSFT"]},
    "Industrials": {"key": ["GE", "RTX", "LMT", "NOC", "HON"],       "trending": ["LMT", "RTX"]},
    "Consumer":    {"key": ["NKE", "WMT", "COST", "TGT", "ONON"],    "trending": ["NKE", "WMT"]},
    "Healthcare":  {"key": ["LLY", "NVO", "AMGN", "PFE", "VRTX"],    "trending": ["LLY", "NVO"]},
    "Macro":       {"key": ["TLT", "GLD", "TIP", "SPY", "DXY"],      "trending": ["GLD", "DXY"]},
}

# ============================================================
# SENTIMENT SCORING KEYWORDS
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
    "devaluation", "debasement", "currency war", "sanctions",
]

# ============================================================
# WHY IT MATTERS TEMPLATES
# Expanded and improved — more specific, less generic.
# ============================================================
WHY_IT_MATTERS_TEMPLATES = {
    "inflation": (
        "Inflation above the Fed's 2% target keeps the policy rate restrictive, "
        "compressing equity multiples via a higher discount rate and squeezing "
        "consumers through real wage erosion. Watch the services component — it "
        "is the stickiest and the Fed's primary focus. Persistent core CPI above "
        "3% effectively rules out near-term cuts and keeps the short end of the "
        "curve elevated. Key watch: breakeven inflation rates and 5Y5Y forward "
        "inflation swaps for market-implied long-run expectations."
    ),
    "fed_rates": (
        "The Fed's rate path is the single most important variable for asset "
        "pricing across every major class. Each 25bps shift reprices risk-free "
        "rates, reshapes the equity risk premium, and resets the cost of capital "
        "for corporate balance sheets. A hawkish pivot crushes duration and "
        "growth equities hardest; a dovish pivot is broadly reflationary. "
        "Monitor the dot plot, FOMC minutes, and Fed speaker tone for forward "
        "guidance. The gap between market pricing and Fed projections is where "
        "volatility is born."
    ),
    "china_growth": (
        "China is the world's largest marginal consumer of commodities and a "
        "critical demand driver for global manufacturing. Stimulus signals from "
        "Beijing historically move copper, iron ore, and bulk shipping within "
        "days. For equity investors, the transmission runs through commodity "
        "producers, luxury goods names with China exposure, and EM sovereign "
        "spreads. Key risk: stimulus may be structural rather than cyclical, "
        "meaning headline numbers recover while consumer demand remains "
        "impaired. Watch the Caixin PMI and retail sales for ground truth."
    ),
    "crude_oil": (
        "Oil prices are simultaneously an input cost, an inflation driver, and "
        "a geopolitical signal. A sustained move above $90 WTI historically "
        "starts to weigh on consumer spending and airline margins; a break below "
        "$70 pressures E&P cash flows and sovereign budgets in OPEC+ nations. "
        "The current supply-demand balance is fragile — OPEC+ compliance and "
        "U.S. shale production response are the key swing variables. "
        "Key watch: EIA weekly inventory builds and IEA demand revision cycles."
    ),
    "treasury_auction": (
        "U.S. Treasury auctions are the mechanism through which sovereign debt "
        "is priced at the margin. A weak auction — low bid-to-cover, high "
        "tail — signals that the market demands a higher yield to absorb supply, "
        "which can trigger a rapid repricing across the entire curve. With the "
        "U.S. running trillion-dollar deficits, auction demand is structurally "
        "critical. Foreign central bank participation is especially important — "
        "a withdrawal of Japanese or Chinese demand would be a systemic event. "
        "Key watch: bid-to-cover ratio, indirect bidder allocation, and the "
        "stop-through or tail versus the when-issued yield."
    ),
    "risk_sentiment": (
        "Broad risk sentiment determines the beta of every portfolio. When "
        "VIX spikes above 20, correlations across assets converge toward 1 — "
        "diversification fails precisely when you need it most. Dealer gamma "
        "positioning amplifies moves: a negative gamma environment means market "
        "makers must sell into declines, accelerating drawdowns. Put/call ratios "
        "at extremes can be contrarian signals, but only when accompanied by "
        "capitulation in fund flows. Key watch: VIX term structure, SKEW index, "
        "and equity put/call ratio versus its 20-day moving average."
    ),
    "tech_ai": (
        "The AI infrastructure buildout is the dominant capex cycle of this "
        "decade. Hyperscaler spending on GPUs, networking, and data center power "
        "is running at a pace that dwarfs prior investment cycles. The investment "
        "thesis has two phases: the picks-and-shovels phase (benefiting Nvidia, "
        "TSMC, power equipment names) and the monetization phase (benefiting "
        "software and application layer companies that can convert AI spend into "
        "revenue). The key risk is whether ROI on AI infrastructure materializes "
        "fast enough to justify continued capex. Key watch: hyperscaler capex "
        "guidance in earnings calls and enterprise software ARR growth rates."
    ),
    "financials": (
        "Bank earnings are a real-time readout on the health of the credit cycle "
        "and the transmission of monetary policy. Net interest margin tells you "
        "how well banks are monetizing the rate environment; provisions for loan "
        "losses tell you what management actually thinks about credit quality "
        "ahead. Investment banking fee recovery is a leading indicator for "
        "M&A and capital markets activity — when deal volumes recover, advisory "
        "boutiques and prime brokers benefit disproportionately. Key watch: "
        "NIM trajectory, deposit beta, CRE reserve builds, and IB fee backlog."
    ),
    "healthcare": (
        "FDA approval decisions create binary, event-driven return profiles. "
        "A major drug launch can shift a company's revenue trajectory for a "
        "decade; a rejection can destroy 40-60% of market cap overnight. The "
        "GLP-1 obesity drug market is reshaping the entire healthcare landscape "
        "— downstream impacts include reduced demand for diabetes devices, "
        "bariatric surgery, and cardiovascular interventions. Payer dynamics "
        "(insurance coverage, formulary placement) determine whether clinical "
        "efficacy translates into commercial success. Key watch: CMS coverage "
        "decisions, pharmacy benefit manager formulary updates, and "
        "manufacturing capacity announcements."
    ),
    "industrials": (
        "The defense sector operates on multi-year procurement cycles that are "
        "largely immune to economic downturns — government budgets replace "
        "consumer demand as the primary driver. NATO's 2% GDP spending "
        "commitment creates a structural tailwind for U.S. defense primes that "
        "extends well into the next decade. On the commercial side, reshoring "
        "capex is building a multi-year order book for automation, "
        "semiconductor fabs, and energy infrastructure. Key watch: DoD budget "
        "requests, program-of-record funding, and ISM new orders subindex."
    ),
    "consumer": (
        "The U.S. consumer accounts for roughly 70% of GDP, making consumption "
        "data the most important real-time economic signal available. The current "
        "bifurcation — value and discount formats outperforming, premium "
        "struggling — reflects K-shaped dynamics where lower-income cohorts "
        "are under pressure from food and shelter inflation while higher-income "
        "consumers remain resilient. Credit card delinquency rates are the "
        "earliest warning signal for consumer stress. Key watch: real wage "
        "growth versus CPI, revolving credit balances, and savings rate."
    ),
    "currency_fx": (
        "Currency moves are often the first market to price geopolitical and "
        "macro regime shifts. The dollar's reserve currency status — what Giscard "
        "d'Estaing called the 'exorbitant privilege' — means U.S. financial "
        "conditions are exported globally via the dollar funding market. "
        "A strong dollar tightens financial conditions in EM economies that "
        "borrow in USD, compresses S&P 500 earnings from overseas revenue, "
        "and makes U.S. exports less competitive. Yen carry unwind risk is "
        "particularly acute — an estimated $4 trillion in yen-funded positions "
        "could be forced to unwind rapidly if BOJ normalizes rates aggressively. "
        "Key watch: DXY trend, USD/JPY 145-150 range, and EUR/USD parity risk."
    ),
    "geopolitical": (
        "Geopolitical risk creates fat-tail scenarios that standard risk models "
        "systematically underprice. Commodity supply disruptions, sanctions "
        "regimes, and reserve currency fragmentation are the three primary "
        "transmission channels into financial markets. Energy price spikes "
        "from conflict are stagflationary — they raise inflation while "
        "simultaneously depressing growth. Safe-haven flows into gold, U.S. "
        "Treasuries, and the Swiss franc are the typical first-order response. "
        "Key watch: oil price volatility premium, gold/oil ratio, and "
        "credit default swap spreads on exposed sovereigns."
    ),
    "default": (
        "This development has direct or indirect implications for asset prices "
        "and portfolio positioning. The primary transmission channels to watch "
        "are: earnings revisions for directly exposed companies, sector rotation "
        "flows as capital reprices risk, and any second-order macro effects "
        "on the rate or inflation outlook. Monitor management commentary on "
        "the next earnings call for forward guidance revisions, and track "
        "options market implied volatility for the affected names as a "
        "real-time risk gauge."
    ),
}
