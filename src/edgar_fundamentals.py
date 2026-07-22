"""
SEC EDGAR Point-in-Time Fundamentals Fetcher
----------------------------------------------
Fixes the look-ahead bias in backtest_oil_dcf.py's fair-value engine:
instead of applying today's revenue/shares/net-debt across the entire
historical backtest window, this pulls each company's full XBRL filing
history from SEC EDGAR (completely free, no API key) and reconstructs
what was actually PUBLIC as of any given historical date -- i.e. only
using a filing once its own `filed` date has passed.

COVERAGE NOTE: US domestic filers (10-K/10-Q) get quarterly point-in-time
granularity. Foreign private issuers (20-F/40-F -- SHEL, TTE, BP, CNQ,
SU, PBR in this project's universe) only file annually with the SEC, so
these will have annual-only granularity, and some may use the IFRS
taxonomy instead of us-gaap (handled below, but coverage/tag consistency
is less reliable than for US domestic filers). This is a structural
limit of what foreign issuers report to the SEC, not a bug in this code.

REQUIRED: SEC's fair-access policy requires every request to carry a
descriptive User-Agent identifying you (name + email). Requests without
one, or with a generic/fake one, can get rate-limited or blocked. Fill
in EDGAR_USER_AGENT below with your own name and email before running.
"""

import time
import warnings

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings('ignore')

# TODO(Owen): fill in your own name and email -- SEC requires a real,
# descriptive User-Agent on every request. Example:
# EDGAR_USER_AGENT = "Owen Petropulos owen@example.com"
EDGAR_USER_AGENT = "Owen Petropulos owenpetropulos@gmail.com"

TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# SEC asks for no more than 10 requests/second; we stay well under that.
REQUEST_DELAY_SECONDS = 0.15

# Candidate XBRL tags per concept, tried in priority order across both
# us-gaap (US domestic filers) and ifrs-full (foreign filers using IFRS)
# taxonomies. Company XBRL tagging is inconsistent -- different filers
# use different tags for economically similar concepts -- so this list
# is deliberately broad. Extend it if a ticker comes back empty.
CONCEPT_TAGS = {
    'revenue': [
        ('us-gaap', 'Revenues'),
        ('us-gaap', 'RevenueFromContractWithCustomerExcludingAssessedTax'),
        ('us-gaap', 'RevenueFromContractWithCustomerIncludingAssessedTax'),
        ('us-gaap', 'SalesRevenueNet'),
        ('us-gaap', 'SalesRevenueGoodsNet'),
        ('ifrs-full', 'Revenue'),
    ],
    'shares_outstanding': [
        ('dei', 'EntityCommonStockSharesOutstanding'),
        ('us-gaap', 'CommonStockSharesOutstanding'),
        ('us-gaap', 'CommonStockSharesIssued'),
    ],
    'total_debt': [
        ('us-gaap', 'DebtLongtermAndShorttermCombinedAmount'),
        ('us-gaap', 'LongTermDebt'),
        ('us-gaap', 'LongTermDebtNoncurrent'),
        ('ifrs-full', 'BorrowingsNoncurrent'),
    ],
    'cash': [
        ('us-gaap', 'CashAndCashEquivalentsAtCarryingValue'),
        ('us-gaap', 'CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents'),
        ('ifrs-full', 'CashAndCashEquivalents'),
    ],
}

# Fallback keyword search used when none of the hardcoded CONCEPT_TAGS
# candidates produce a reliable series. Many foreign filers (and some
# US ones) report under a custom/company-specific extension taxonomy
# instead of the standard us-gaap/ifrs-full tags -- e.g. TotalEnergies,
# Suncor, and Petrobras all came back "frozen" (one stale value forever)
# under the hardcoded candidates, almost certainly because their real
# recurring disclosure lives under a tag we never checked. Rather than
# hand-researching each company's actual tag name, this scans EVERY
# taxonomy present in that company's filing history for any tag whose
# name contains one of these keywords, tests each one, and keeps the
# best-scoring result. Keyword matching is on the lowercased tag name
# with no separators (XBRL tag names are CamelCase, e.g. "Revenues"),
# so keep these as lowercase substrings.
CONCEPT_KEYWORD_FALLBACKS = {
    'revenue': ['revenue', 'salesrevenue', 'turnover', 'totalrevenues'],
    'shares_outstanding': ['sharesoutstanding', 'sharesissued'],
    'total_debt': ['debt', 'borrowing'],
    'cash': ['cashandcashequivalents', 'cashequivalents'],
}


def _get(url, params=None):
    headers = {'User-Agent': EDGAR_USER_AGENT}
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    time.sleep(REQUEST_DELAY_SECONDS)
    return resp.json()


# Manual CIK overrides for tickers where SEC's own ticker_cik file
# resolves to the wrong entity. Confirmed case: "XOM" resolves to
# "ExxonMobil Holdings Corp" (CIK 2115436, no real XBRL history) instead
# of the actual public company "Exxon Mobil Corp" (CIK 34088) -- these
# two entities both use "XOM" in SEC's file, and there's no reliable
# ordering to pick the right one automatically. If you see a ticker
# resolve to a company name that doesn't match what you expect (printed
# alongside the CIK when fetching), look up the correct CIK yourself at
# https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany and add it
# here.
CIK_OVERRIDES = {
    'XOM': ('0000034088', 'Exxon Mobil Corp (manual override)'),
}


def load_ticker_cik_map():
    """
    SEC's own ticker->CIK lookup file. Returns dict[ticker] -> (cik, title)
    tuples, where title is the registrant's name -- useful for eyeballing
    that a ticker resolved to the company you actually expect.

    IMPORTANT: this file can contain more than one entry for the same
    ticker string (e.g. an unrelated small/OTC registrant that happens to
    reuse a well-known ticker elsewhere). SEC's file convention generally
    lists the primary/most-liquid listing first, so this keeps the FIRST
    match per ticker rather than the last -- fixes a real bug where a
    well-known ticker (XOM) was getting silently overwritten by an
    unrelated later entry in the file, resolving to the wrong CIK.
    """
    data = _get(TICKER_CIK_URL)
    mapping = {}
    for _, entry in data.items():
        ticker = entry['ticker'].upper()
        cik = str(entry['cik_str']).zfill(10)
        title = entry.get('title', '')
        if ticker not in mapping:  # first match wins
            mapping[ticker] = (cik, title)
    return mapping


def fetch_company_facts(cik):
    """One call returns a company's ENTIRE XBRL filing history -- no
    need to query per period or per week."""
    return _get(COMPANY_FACTS_URL.format(cik=cik))


def _records_from_units(units, unit_key_pref):
    """Pulls the raw (filed, start, end, val, form, fy, fp) records out
    of one tag's `units` dict, preferring the given unit key (USD or
    shares) but falling back to whatever unit is present."""
    if unit_key_pref not in units:
        if not units:
            return []
        unit_key_pref = next(iter(units))
    records = units[unit_key_pref]
    return [
        {
            'filed': r.get('filed'), 'start': r.get('start'), 'end': r.get('end'),
            'val': r.get('val'), 'form': r.get('form'), 'fy': r.get('fy'), 'fp': r.get('fp'),
        }
        for r in records
        if r.get('filed') and r.get('val') is not None
    ]


def _apply_annual_filter(candidates):
    """Keeps only records whose reporting period is roughly a full year
    (340-386 days). See _extract_concept_series docstring for why this
    matters for flow concepts like revenue."""
    filtered = []
    for c in candidates:
        if not c['start'] or not c['end']:
            continue
        duration_days = (pd.Timestamp(c['end']) - pd.Timestamp(c['start'])).days
        if 340 <= duration_days <= 386:
            filtered.append(c)
    return filtered


def _score_candidate(records):
    """Data-quality score for one candidate tag's records: how many
    distinct filing dates it covers and how much the values actually
    vary. A tag that only ever reports one stale number (the TTE/SU/PBR
    failure pattern -- almost certainly the wrong tag, or one that
    stopped being used) scores low even if it technically "has data."
    """
    if not records:
        return (0, 0)
    n_dates = len(set(r['filed'] for r in records))
    n_unique_vals = len(set(r['val'] for r in records))
    return (n_unique_vals, n_dates)


def _discover_tags_by_keyword(facts_json, keywords):
    """
    Scans every taxonomy present in this company's filing history (not
    just us-gaap/ifrs-full) for any tag whose name contains one of the
    given keywords. This is the auto-discovery fallback for filers using
    a custom/extension taxonomy for their real recurring disclosure --
    rather than hand-researching each company's actual tag name, this
    finds candidates automatically so they can be scored and compared
    the same way as the hardcoded CONCEPT_TAGS list.
    """
    discovered = []
    facts = facts_json.get('facts', {})
    for taxonomy, tags in facts.items():
        for tag in tags:
            tag_lower = tag.lower()
            if any(kw in tag_lower for kw in keywords):
                discovered.append((taxonomy, tag))
    return discovered


def _extract_concept_series(facts_json, concept_key, annual_only=False):
    """
    Tries every hardcoded CONCEPT_TAGS candidate for this concept, scores
    each by data quality (_score_candidate), and if NONE of them score
    well (empty, or fewer than 3 distinct values), automatically expands
    the search to every tag in the company's filing history matching
    CONCEPT_KEYWORD_FALLBACKS -- catching custom/extension taxonomy tags
    that the hardcoded list doesn't know about. Returns the best-scoring
    candidate's records, or an empty list if nothing usable was found.

    annual_only: when True, filters to records whose reporting period
    (end - start) is roughly a full year (340-386 days, allowing for
    52/53-week fiscal calendars). This matters for flow concepts like
    revenue -- without it, a 10-Q's quarterly figure (~3 months) and a
    10-K's full-year figure get mixed together inconsistently, which
    silently corrupts the resulting series (e.g. COP alternating between
    ~$16B-quarter and ~$59B-year figures depending on which filing type
    was most recently filed, rather than a consistent annual scale).
    Not applied to instantaneous/balance-sheet concepts (shares, debt,
    cash), since those aren't a duration -- a single filing's disclosed
    balance is valid regardless of whether it came from a 10-Q or 10-K.

    Returns (records, source_label). source_label is None if nothing
    usable was found (score fell below the reliability threshold).
    """
    unit_key_pref = 'shares' if concept_key == 'shares_outstanding' else 'USD'

    def try_candidate(taxonomy, tag):
        try:
            units = facts_json['facts'][taxonomy][tag]['units']
        except KeyError:
            return []
        records = _records_from_units(units, unit_key_pref)
        if annual_only:
            records = _apply_annual_filter(records)
        return records

    best_records, best_score, best_source = [], (0, 0), None

    # Pass 1: hardcoded candidates, in priority order.
    for taxonomy, tag in CONCEPT_TAGS[concept_key]:
        records = try_candidate(taxonomy, tag)
        score = _score_candidate(records)
        if score > best_score:
            best_records, best_score, best_source = records, score, f"{taxonomy}:{tag}"

    # Pass 2: only bother with keyword auto-discovery if pass 1 didn't
    # find anything reliable (fewer than 3 distinct values) -- keeps the
    # common case (major US filers, whose hardcoded tags work fine) fast.
    if best_score[0] < 3 and concept_key in CONCEPT_KEYWORD_FALLBACKS:
        keywords = CONCEPT_KEYWORD_FALLBACKS[concept_key]
        for taxonomy, tag in _discover_tags_by_keyword(facts_json, keywords):
            if (taxonomy, tag) in CONCEPT_TAGS[concept_key]:
                continue  # already tried in pass 1
            records = try_candidate(taxonomy, tag)
            score = _score_candidate(records)
            if score > best_score:
                best_records, best_score, best_source = records, score, f"{taxonomy}:{tag} [auto-discovered]"

    source = best_source if best_score[0] >= 3 else None
    return best_records, source



def build_point_in_time_table(facts_json, verbose=False):
    """
    Combines revenue, shares outstanding, total debt, and cash into one
    DataFrame indexed by filing date, forward-fillable so any historical
    date can look up "what was most recently known as of then."

    verbose: if True, prints which tag ended up being used for each
    concept -- useful for spotting when auto-discovery kicked in.
    """
    series_by_concept = {}
    for key in CONCEPT_TAGS:
        records, source = _extract_concept_series(facts_json, key, annual_only=(key == 'revenue'))
        series_by_concept[key] = records
        if verbose:
            print(f"      [{key}] using tag: {source or 'NONE FOUND'}")

    frames = []
    for concept, records in series_by_concept.items():
        if not records:
            continue
        df = pd.DataFrame(records)
        df['filed'] = pd.to_datetime(df['filed'])
        # Same filing can restate multiple periods (e.g. a 10-K showing
        # prior-year comparatives) -- keep only the record whose `end`
        # (period end) is latest for each filing date, i.e. the freshest
        # period disclosed in that filing.
        df = df.sort_values(['filed', 'end']).drop_duplicates('filed', keep='last')
        df = df.set_index('filed')[['val']].rename(columns={'val': concept})
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, axis=1).sort_index()
    # Forward-fill: once a figure is filed, it remains "the known value"
    # until superseded by a later filing.
    combined = combined.ffill()
    return combined


def as_of(point_in_time_table, date):
    """
    Returns the fundamentals dict known as of `date` (i.e. using only
    filings with filed <= date). Returns None if nothing was filed yet
    by that date, or if the table is empty.
    """
    if point_in_time_table is None or point_in_time_table.empty:
        return None
    eligible = point_in_time_table[point_in_time_table.index <= pd.Timestamp(date)]
    if eligible.empty:
        return None
    row = eligible.iloc[-1]
    revenue = row.get('revenue')
    shares = row.get('shares_outstanding')
    debt_raw = row.get('total_debt', 0.0)
    cash_raw = row.get('cash', 0.0)
    if pd.isna(revenue) or pd.isna(shares):
        return None
    debt_val = float(debt_raw) if not pd.isna(debt_raw) else 0.0
    cash_val = float(cash_raw) if not pd.isna(cash_raw) else 0.0
    return {
        'current_revenue': float(revenue) / 1e6,       # normalize to $mm
        'current_production': 100,                       # scale-free anchor
        'shares_outstanding': float(shares) / 1e6,        # normalize to mm shares
        'net_debt': (debt_val - cash_val) / 1e6,
    }


def is_reliable_series(table, min_unique_values=3):
    """
    Sanity check for a ticker's point-in-time table: real companies'
    revenue genuinely changes year to year, especially for oil producers
    across a 2019-2026 window that includes a price crash, a spike, and
    a slow grind -- so a revenue series that's missing entirely, or
    frozen at a single value across the whole history, almost certainly
    means the XBRL tag matching failed to find that company's real
    recurring revenue disclosure (e.g. a custom/extension tag not in
    CONCEPT_TAGS), not that revenue was actually flat for years.

    Returns False for tables that are empty or whose revenue column has
    fewer than `min_unique_values` distinct values -- these tickers
    should fall back to a static snapshot rather than be trusted as
    genuinely point-in-time.
    """
    if table is None or table.empty or 'revenue' not in table.columns:
        return False
    n_unique = table['revenue'].nunique(dropna=True)
    return n_unique >= min_unique_values


def get_point_in_time_fundamentals(tickers, verbose=True):
    """
    Main entry point. Returns dict[ticker] -> point-in-time DataFrame
    (or None if the ticker couldn't be resolved/fetched). Pass the
    result's per-ticker table to as_of(table, date) inside the backtest
    loop instead of a single static fundamentals dict.
    """
    if EDGAR_USER_AGENT.startswith("REPLACE_ME"):
        raise RuntimeError(
            "Set EDGAR_USER_AGENT to your own name and email before "
            "running -- SEC requires a real identifying User-Agent on "
            "every request."
        )

    if verbose:
        print("Loading SEC ticker->CIK map...")
    ticker_cik_map = load_ticker_cik_map()

    tables = {}
    for ticker in tickers:
        match = CIK_OVERRIDES.get(ticker.upper()) or ticker_cik_map.get(ticker.upper())
        if match is None:
            if verbose:
                print(f"   {ticker}: not found in SEC ticker map, skipping")
            tables[ticker] = None
            continue
        cik, title = match
        try:
            if verbose:
                # Prints the registrant name alongside the CIK so you can
                # eyeball that it resolved to the company you expect --
                # e.g. confirm "XOM" -> "Exxon Mobil Corp", not some
                # unrelated entity that happens to share the ticker.
                print(f"   {ticker}: fetching (CIK {cik}, \"{title}\")...", end=' ')
            facts = fetch_company_facts(cik)
            table = build_point_in_time_table(facts, verbose=verbose)
            if table.empty:
                if verbose:
                    print("no usable XBRL facts found")
                tables[ticker] = None
            else:
                if verbose:
                    print(f"OK ({len(table)} filing dates, "
                          f"{table.index.min().date()} to {table.index.max().date()})")
                tables[ticker] = table
        except requests.exceptions.RequestException as e:
            if verbose:
                print(f"FAILED ({e})")
            tables[ticker] = None

    return tables


if __name__ == "__main__":
    # Quick standalone smoke test -- prints a fundamentals snapshot for
    # a couple of dates so you can eyeball whether it's behaving sanely
    # before wiring it into the full backtest. Covers the full universe
    # so any other CIK collisions or missing-data tickers surface now,
    # not mid-integration.
    test_tickers = ['XOM', 'CVX', 'SHEL', 'TTE', 'BP',
                     'COP', 'OXY', 'CNQ', 'SU', 'FANG', 'EOG', 'PBR']
    tables = get_point_in_time_fundamentals(test_tickers)
    print("\n" + "=" * 70)
    print("RELIABILITY CHECK (would this ticker use EDGAR point-in-time,")
    print("or fall back to a static snapshot in the full backtest?)")
    print("=" * 70)
    for ticker, table in tables.items():
        reliable = is_reliable_series(table)
        n_unique = table['revenue'].nunique(dropna=True) if (table is not None and not table.empty and 'revenue' in table.columns) else 0
        print(f"   {ticker:5s} {'RELIABLE' if reliable else 'UNRELIABLE -> fallback needed':30s} "
              f"({n_unique} distinct revenue values in history)")

    print("\n" + "=" * 70)
    for ticker, table in tables.items():
        print(f"\n=== {ticker} ===")
        if table is None:
            print("   No data.")
            continue
        for date in ['2019-06-01', '2022-06-01', '2026-06-01']:
            snap = as_of(table, date)
            print(f"   as_of {date}: {snap}")
