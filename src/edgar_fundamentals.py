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
EDGAR_USER_AGENT = "REPLACE_ME name@example.com"

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


def _get(url, params=None):
    headers = {'User-Agent': EDGAR_USER_AGENT}
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    time.sleep(REQUEST_DELAY_SECONDS)
    return resp.json()


def load_ticker_cik_map():
    """
    SEC's own ticker->CIK lookup file. Returns dict[ticker] -> zero-padded
    10-digit CIK string (as required by the companyfacts URL).
    """
    data = _get(TICKER_CIK_URL)
    mapping = {}
    for _, entry in data.items():
        ticker = entry['ticker'].upper()
        cik = str(entry['cik_str']).zfill(10)
        mapping[ticker] = cik
    return mapping


def fetch_company_facts(cik):
    """One call returns a company's ENTIRE XBRL filing history -- no
    need to query per period or per week."""
    return _get(COMPANY_FACTS_URL.format(cik=cik))


def _extract_concept_series(facts_json, concept_key):
    """
    Tries each candidate (taxonomy, tag) pair for this concept in
    priority order. Returns the first one with data, as a list of dicts:
    {filed, start, end, val, form, fy, fp}. Empty list if none matched.
    """
    for taxonomy, tag in CONCEPT_TAGS[concept_key]:
        try:
            units = facts_json['facts'][taxonomy][tag]['units']
        except KeyError:
            continue
        # Prefer USD for monetary concepts, 'shares' for share counts.
        unit_key = 'shares' if concept_key == 'shares_outstanding' else 'USD'
        if unit_key not in units:
            # fall back to whatever unit is present
            if not units:
                continue
            unit_key = next(iter(units))
        records = units[unit_key]
        if records:
            return [
                {
                    'filed': r.get('filed'),
                    'start': r.get('start'),
                    'end': r.get('end'),
                    'val': r.get('val'),
                    'form': r.get('form'),
                    'fy': r.get('fy'),
                    'fp': r.get('fp'),
                }
                for r in records
                if r.get('filed') and r.get('val') is not None
            ]
    return []


def build_point_in_time_table(facts_json):
    """
    Combines revenue, shares outstanding, total debt, and cash into one
    DataFrame indexed by filing date, forward-fillable so any historical
    date can look up "what was most recently known as of then."
    """
    series_by_concept = {
        key: _extract_concept_series(facts_json, key) for key in CONCEPT_TAGS
    }

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
        cik = ticker_cik_map.get(ticker.upper())
        if cik is None:
            if verbose:
                print(f"   {ticker}: not found in SEC ticker map, skipping")
            tables[ticker] = None
            continue
        try:
            if verbose:
                print(f"   {ticker}: fetching (CIK {cik})...", end=' ')
            facts = fetch_company_facts(cik)
            table = build_point_in_time_table(facts)
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
    # before wiring it into the full backtest.
    test_tickers = ['XOM', 'COP']
    tables = get_point_in_time_fundamentals(test_tickers)
    for ticker, table in tables.items():
        print(f"\n=== {ticker} ===")
        if table is None:
            print("   No data.")
            continue
        for date in ['2019-06-01', '2022-06-01', '2026-06-01']:
            snap = as_of(table, date)
            print(f"   as_of {date}: {snap}")
