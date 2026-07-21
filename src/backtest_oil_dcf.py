"""
Weekly Oil-DCF Rebalancing Backtest
------------------------------------
Every week (anchored to Monday's WTI crude price), runs a fast Monte Carlo
version of the Stochastic Macro-Driven DCF (v1.2 logic) for each stock in
the universe, ranks stocks by how far price sits below simulated fair
value, and builds a long-only, conviction-weighted portfolio tilted
toward the cheapest names -- capped per a graduated schedule based on how
many names are eligible (positive signal) that week.

This intentionally re-implements the v1.2 JS engine's math in Python
(correlated shocks, kinked margin, distress-linked capex pullback and
debt spread widening, smoothed terminal value) rather than approximating
it, so the backtest signal is mechanically consistent with the tool.

STRUCTURE:
  1. Configuration (universe, fundamentals, cap schedule)
  2. Data download (WTI + stock prices)
  3. Fast Monte Carlo DCF engine (Python port of v1.2 script.js)
  4. Signal construction (fair value -> weights, capped)
  5. Backtest loop
  6. Benchmark + metrics
  7. Save results
  8. Main
"""

import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

try:
    import yfinance as yf
except ImportError:
    import subprocess
    subprocess.check_call(['pip3', 'install', '--upgrade', 'yfinance'])
    import yfinance as yf


# ============================================================================
# 1. CONFIGURATION
# ============================================================================

# Upstream oil / E&P universe. Deliberately excludes: utilities (no
# direct oil-price revenue link), refiners (crack-spread economics run
# in the opposite direction from producer economics), and Saudi Aramco
# (thin free float, Tadawul's Sun-Thu trading week doesn't align with a
# Monday-anchored weekly strategy, and yfinance coverage is unreliable).
UNIVERSE = ['XOM', 'CVX', 'SHEL', 'TTE', 'BP',      # integrated majors
            'COP', 'OXY', 'CNQ', 'SU', 'FANG', 'EOG',  # pure-play upstream
            'PBR']                                   # EM state-risk name

WTI_TICKER = 'CL=F'
START_DATE = '2019-01-01'
END_DATE = '2026-07-01'
INITIAL_CAPITAL = 100_000

# Number of simulated paths per ticker per week. Kept low deliberately --
# see the note in section 3 on why full-precision Monte Carlo isn't
# needed for a cross-sectional ranking signal, only relative accuracy
# between stocks in the same week.
FAST_MC_PATHS = 300
FORECAST_YEARS = 5

# Graduated single-name weight cap, keyed by number of eligible
# (positive-signal) names that week. Beyond 6 eligible names, no cap is
# applied. See cap_and_redistribute() for the mechanics.
CAP_SCHEDULE = {1: 0.30, 2: 0.50, 3: 0.60, 4: 0.65, 5: 0.75, 6: 0.85}

# Default operating assumptions applied to every ticker unless overridden
# in COMPANY_OVERRIDES below. These mirror the v1.2 tool's own defaults --
# calibrate per-company once real data/analysis is available; treat these
# as a reasonable generic upstream-producer starting point, not researched
# figures for any specific company.
DEFAULT_ASSUMPTIONS = {
    'base_operating_margin': 0.30,
    'capex_percent_revenue': 0.18,
    'da_percent_revenue': 0.06,
    'working_capital_percent_revenue': 0.05,
    'revenue_sensitivity_to_commodity': 0.70,
    'revenue_sensitivity_to_demand': 0.30,
    'margin_sensitivity_to_commodity': 0.20,
    'cash_breakeven_price': 40,
    'below_breakeven_multiplier': 2.0,
    'structural_margin_floor': -0.20,
    'capex_sensitivity_to_commodity': 0.10,
    'production_growth_sensitivity_to_reinvestment': 0.05,
    'tax_rate': 0.25,
    'beta': 1.20,
    'equity_risk_premium': 0.055,
    'debt_spread': 0.015,
    'debt_weight': 0.30,
    'equity_weight': 0.70,
    'terminal_growth': 0.025,
    'terminal_smoothing_years': 3,
    'distress_price_threshold': 45,
    'distress_trailing_years': 2,
    'capex_pullback_sensitivity': 0.10,
    'debt_spread_distress_sensitivity': 0.02,
}

# Per-ticker overrides, built from qualitative group differentiation
# rather than the flat defaults every ticker shared before. These are
# directional starting points reflecting each group's business model --
# NOT regression-fitted or analyst-researched figures. TODO(Owen): refine
# with real sensitivity regressions (revenue growth vs. WTI) and updated
# betas once you have time to calibrate properly.
#
#   Integrated majors (XOM, CVX, SHEL, TTE, BP): downstream/chemicals
#     segments dampen pure upstream oil-price sensitivity relative to
#     pure-plays; scale and diversification also support a lower beta.
#   Pure-play upstream (COP, OXY, CNQ, FANG, EOG): revenue and margins
#     track oil price much more directly with no downstream buffer.
#   Suncor (SU): oil sands operator -- structurally higher extraction
#     cost curve than conventional producers, so a higher cash breakeven.
#   Petrobras (PBR): EM state-controlled -- Brazilian government has a
#     documented history of intervening in domestic fuel pricing for
#     political reasons, weakening the clean price-follows-WTI link the
#     model otherwise assumes. Modeled here as extra equity risk premium
#     and a wider base debt spread rather than touching the commodity
#     sensitivities directly, since the mechanism is political risk, not
#     an operating cost-curve difference.

MAJORS = ['XOM', 'CVX', 'SHEL', 'TTE', 'BP']
PURE_PLAY = ['COP', 'OXY', 'CNQ', 'FANG', 'EOG']

MAJOR_OVERRIDES = {
    'revenue_sensitivity_to_commodity': 0.50,
    'margin_sensitivity_to_commodity': 0.14,
    'beta': 1.00,
}
PURE_PLAY_OVERRIDES = {
    'revenue_sensitivity_to_commodity': 0.85,
    'margin_sensitivity_to_commodity': 0.26,
    'beta': 1.35,
}
SUNCOR_OVERRIDES = {
    'revenue_sensitivity_to_commodity': 0.75,
    'margin_sensitivity_to_commodity': 0.22,
    'cash_breakeven_price': 50,
    'distress_price_threshold': 55,
    'beta': 1.30,
}
PETROBRAS_OVERRIDES = {
    'equity_risk_premium': 0.075,
    'debt_spread': 0.030,
    'beta': 1.40,
}

COMPANY_OVERRIDES = {}
for _t in MAJORS:
    COMPANY_OVERRIDES[_t] = dict(MAJOR_OVERRIDES)
for _t in PURE_PLAY:
    COMPANY_OVERRIDES[_t] = dict(PURE_PLAY_OVERRIDES)
COMPANY_OVERRIDES['SU'] = dict(SUNCOR_OVERRIDES)
COMPANY_OVERRIDES['PBR'] = dict(PETROBRAS_OVERRIDES)

# Macro process assumptions shared across all tickers each week (only
# commodity_current is re-anchored weekly to that Monday's WTI price).
MACRO_ASSUMPTIONS = {
    'commodity_vol': 0.20,
    'commodity_long_run_mean': 70,
    'commodity_reversion_speed': 0.02,
    'rate_drift': 0.30,
    'rate_vol': 0.008,
    'rate_long_run_mean': 0.040,
    'demand_current': 100,
    'demand_vol': 0.06,
    'demand_growth': 0.02,
    'corr_oil_rate': -0.20,
    'corr_oil_demand': 0.30,
    'corr_rate_demand': 0.20,
}


# ============================================================================
# 2. DATA DOWNLOAD
# ============================================================================

def download_series(ticker, start, end, label=None):
    label = label or ticker
    try:
        print(f"   {label}...", end=' ')
        hist = yf.Ticker(ticker).history(start=start, end=end)
        if len(hist) > 0:
            s = hist['Close']
            s.name = ticker
            print(f"OK {len(s)} days")
            return s
        print("No data")
        return None
    except Exception as e:
        print(f"FAILED ({e})")
        return None


def download_all_data(tickers, wti_ticker, start, end):
    print("Downloading price data...")
    print(f"   Period: {start} to {end}\n")

    all_prices = []
    for ticker in tickers:
        s = download_series(ticker, start, end)
        if s is not None:
            all_prices.append(s)
        time.sleep(0.5)

    if not all_prices:
        print("\nFailed to download any stock data")
        return None, None

    prices_df = pd.concat(all_prices, axis=1)

    wti = download_series(wti_ticker, start, end, label='WTI Crude')
    if wti is None:
        print("\nFailed to download WTI data")
        return None, None

    print(f"\nDownloaded {len(all_prices)} stocks + WTI")
    print(f"   Date range: {prices_df.index[0].date()} to {prices_df.index[-1].date()}\n")

    return prices_df, wti


def fetch_static_fundamentals(tickers):
    """
    Pull the fundamentals that are reasonably reliable from yfinance
    (shares outstanding, revenue, net debt, current price) at the start
    of the backtest run. These are treated as roughly static over the
    backtest window -- a simplification worth flagging: real fundamentals
    obviously drift over a multi-year backtest, but re-pulling them
    historically per week isn't available through this data source.
    Production is intentionally normalized to 100 for every ticker since
    the DCF only uses it as a scale-free ratio (prod_t / prod_0).
    """
    fundamentals = {}
    print("Fetching static fundamentals...")
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            shares = info.get('sharesOutstanding')
            revenue = info.get('totalRevenue')
            total_debt = info.get('totalDebt', 0) or 0
            total_cash = info.get('totalCash', 0) or 0
            price = info.get('currentPrice') or info.get('regularMarketPrice')

            if not shares or not revenue or not price:
                print(f"   {ticker}: missing required fields, skipping")
                continue

            fundamentals[ticker] = {
                'current_revenue': revenue / 1e6,       # normalize to $mm
                'current_production': 100,               # scale-free anchor
                'shares_outstanding': shares / 1e6,       # normalize to mm shares
                'net_debt': (total_debt - total_cash) / 1e6,
                'current_stock_price': price,
            }
            print(f"   {ticker}: OK")
        except Exception as e:
            print(f"   {ticker}: FAILED ({e})")
        time.sleep(0.3)

    return fundamentals


# ============================================================================
# 3. FAST MONTE CARLO DCF ENGINE (Python port of v1.2 script.js)
#
# Deliberately uses far fewer paths (FAST_MC_PATHS) than the interactive
# tool's default. For a single valuation, precision matters. For a
# cross-sectional weekly ranking signal across N stocks, what matters is
# relative accuracy between stocks in the same week, not point-precision
# on any one of them -- so a smaller path count is an acceptable and
# necessary tradeoff to keep a multi-year weekly backtest tractable.
# ============================================================================

def cholesky_3x3(r12, r13, r23):
    L21 = r12
    L22 = np.sqrt(max(1 - L21 ** 2, 1e-8))
    L31 = r13
    L32 = (r23 - L21 * L31) / L22
    L33 = np.sqrt(max(1 - L31 ** 2 - L32 ** 2, 1e-8))
    return L21, L22, L31, L32, L33


def simulate_state_paths(inputs, n_sims, T, rng):
    L21, L22, L31, L32, L33 = cholesky_3x3(
        inputs['corr_oil_rate'], inputs['corr_oil_demand'], inputs['corr_rate_demand']
    )

    comm_kappa = inputs['commodity_reversion_speed']
    comm_theta = inputs['commodity_long_run_mean']
    comm_sigma = inputs['commodity_vol'] * inputs['commodity_current']

    rate_kappa = inputs['rate_drift']
    rate_theta = inputs['rate_long_run_mean']
    rate_sigma = inputs['rate_vol']

    demand_sigma = inputs['demand_vol']
    demand_drift = inputs['demand_growth'] - 0.5 * demand_sigma ** 2

    e = rng.standard_normal((n_sims, T, 3))
    z1 = e[:, :, 0]
    z2 = L21 * e[:, :, 0] + L22 * e[:, :, 1]
    z3 = L31 * e[:, :, 0] + L32 * e[:, :, 1] + L33 * e[:, :, 2]

    commodity = np.zeros((n_sims, T))
    rate = np.zeros((n_sims, T))
    demand = np.zeros((n_sims, T))

    comm = np.full(n_sims, inputs['commodity_current'])
    rt = np.full(n_sims, inputs['rate_current'])
    dem = np.full(n_sims, inputs['demand_current'])

    for t in range(T):
        comm = comm + comm_kappa * (comm_theta - comm) + comm_sigma * z1[:, t]
        comm = np.maximum(comm, 1.0)
        commodity[:, t] = comm

        rt = rt + rate_kappa * (rate_theta - rt) + rate_sigma * z2[:, t]
        rt = np.maximum(rt, 0.001)
        rate[:, t] = rt

        dem = dem * np.exp(demand_drift + demand_sigma * z3[:, t])
        demand[:, t] = dem

    return commodity, rate, demand


def trailing_average(path_row, t, window):
    start = max(0, t - window + 1)
    return path_row[start:t + 1].mean()


def build_operating_and_wacc_paths(commodity, rate, demand, inputs):
    n_sims, T = commodity.shape

    rev0, comm0, demand0 = inputs['current_revenue'], inputs['commodity_current'], inputs['demand_current']
    base_margin = inputs['base_operating_margin']
    base_capex = inputs['capex_percent_revenue']
    da_pct = inputs['da_percent_revenue']
    wc_pct = inputs['working_capital_percent_revenue']
    tax = inputs['tax_rate']

    rev_sens_comm = inputs['revenue_sensitivity_to_commodity']
    rev_sens_dem = inputs['revenue_sensitivity_to_demand']
    margin_sens = inputs['margin_sensitivity_to_commodity']
    breakeven = inputs['cash_breakeven_price']
    below_mult = inputs['below_breakeven_multiplier']
    margin_floor = inputs['structural_margin_floor']
    capex_sens = inputs['capex_sensitivity_to_commodity']
    breakeven_dev = (breakeven - comm0) / comm0

    distress_threshold = inputs['distress_price_threshold']
    distress_window = inputs['distress_trailing_years']
    capex_pullback_sens = inputs['capex_pullback_sensitivity']
    spread_distress_sens = inputs['debt_spread_distress_sensitivity']

    beta, erp = inputs['beta'], inputs['equity_risk_premium']
    debt_spread = inputs['debt_spread']
    debt_weight, equity_weight = inputs['debt_weight'], inputs['equity_weight']

    # Revenue depends only on the contemporaneous commodity/demand
    # deviation in this port (production is a scale-free constant, unlike
    # the JS engine where production compounds path-dependently through
    # reinvestment) -- so it's fully vectorizable in one pass.
    comm_dev = (commodity - comm0) / comm0
    dem_dev = (demand - demand0) / demand0
    revenue = np.maximum(rev0 * (1 + rev_sens_comm * comm_dev) * (1 + rev_sens_dem * dem_dev), 0)

    fcf = np.zeros((n_sims, T))
    wacc = np.zeros((n_sims, T))

    for i in range(n_sims):
        for t in range(T):
            comm_t = commodity[i, t]
            rev_t = revenue[i, t]
            cdev_t = comm_dev[i, t]

            # Kinked margin: normal slope above breakeven, steeper below
            # (mirrors the A2 logic in script.js exactly).
            if comm_t >= breakeven:
                margin_t = base_margin + margin_sens * cdev_t
            else:
                normal_portion = margin_sens * breakeven_dev
                steep_portion = margin_sens * below_mult * (cdev_t - breakeven_dev)
                margin_t = base_margin + normal_portion + steep_portion
            margin_t = min(max(margin_t, margin_floor), 0.95)

            ebit_t = rev_t * margin_t
            da_t = da_pct * rev_t

            # A4: capex pulls back further under sustained trailing distress.
            trail_price = trailing_average(commodity[i], t, distress_window)
            distress_gap = max(0.0, (distress_threshold - trail_price) / distress_threshold)
            capex_pct_t = base_capex + capex_sens * cdev_t - capex_pullback_sens * distress_gap
            capex_pct_t = min(max(capex_pct_t, 0), 0.60)
            capex_t = capex_pct_t * rev_t

            wc_t = wc_pct * rev_t
            wc_prev_t = wc_pct * (rev0 if t == 0 else revenue[i, t - 1])
            delta_wc_t = wc_t - wc_prev_t

            fcf[i, t] = ebit_t * (1 - tax) + da_t - capex_t - delta_wc_t

            # A3: debt spread widens on the same trailing distress signal.
            rf = rate[i, t]
            ke = rf + beta * erp
            spread_t = debt_spread + spread_distress_sens * distress_gap
            kd = rf + spread_t
            wacc[i, t] = max(equity_weight * ke + debt_weight * kd * (1 - tax), 0.01)

    return fcf, wacc


def discount_to_price(fcf, wacc, inputs):
    n_sims, T = fcf.shape
    g = inputs['terminal_growth']
    net_debt = inputs['net_debt']
    shares = inputs['shares_outstanding']
    smooth_years = max(1, min(inputs['terminal_smoothing_years'], T))

    cum_disc = np.cumprod(1 + wacc, axis=1)
    pv_fcf = (fcf / cum_disc).sum(axis=1)

    wacc_terminal = wacc[:, -1]
    fcf_terminal = fcf[:, T - smooth_years:T].mean(axis=1)
    tv_denom = wacc_terminal - g
    tv = np.where(tv_denom > 0.001, fcf_terminal * (1 + g) / np.maximum(tv_denom, 1e-9), 0.0)

    ev = pv_fcf + tv / cum_disc[:, -1]
    prices = (ev - net_debt) / shares
    return prices


def fair_value(inputs, wti_price, n_sims=FAST_MC_PATHS, T=FORECAST_YEARS, seed=None):
    """
    Returns the median simulated per-share fair value for one ticker,
    given that week's WTI price as the commodity starting point.
    """
    run_inputs = dict(inputs)
    run_inputs['commodity_current'] = wti_price
    run_inputs.setdefault('rate_current', 0.045)

    rng = np.random.default_rng(seed)
    commodity, rate, demand = simulate_state_paths(run_inputs, n_sims, T, rng)
    fcf, wacc = build_operating_and_wacc_paths(commodity, rate, demand, run_inputs)
    prices = discount_to_price(fcf, wacc, run_inputs)
    return float(np.median(prices))


# ============================================================================
# 4. SIGNAL CONSTRUCTION
# ============================================================================

def get_cap(n_eligible):
    return CAP_SCHEDULE.get(n_eligible, 1.0)


def cap_and_redistribute(raw_weights, cap, max_iter=50, tol=1e-9):
    weights = dict(raw_weights)
    for _ in range(max_iter):
        over = {k: v for k, v in weights.items() if v > cap + tol}
        if not over:
            break
        excess = sum(v - cap for v in over.values())
        for k in over:
            weights[k] = cap
        under = {k: v for k, v in weights.items() if v < cap - tol}
        under_sum = sum(under.values())
        if under_sum <= tol or not under:
            break
        for k in under:
            weights[k] += excess * (weights[k] / under_sum)
    return weights


def build_weekly_weights(fair_values, prices):
    """
    fair_values, prices: dict[ticker] -> float, for a single week.
    Returns dict[ticker] -> weight (may sum to < 1.0; remainder is cash).
    """
    raw_signal = {}
    for ticker, fv in fair_values.items():
        price = prices.get(ticker)
        if price is None or fv is None or fv <= 0:
            continue
        signal = (fv - price) / fv
        if signal > 0:
            raw_signal[ticker] = signal

    if not raw_signal:
        return {}

    total_signal = sum(raw_signal.values())
    raw_weights = {k: v / total_signal for k, v in raw_signal.items()}

    cap = get_cap(len(raw_weights))
    return cap_and_redistribute(raw_weights, cap)


# ============================================================================
# 5. BACKTEST LOOP
# ============================================================================

def run_backtest(prices_df, wti, universe, fundamentals, initial_capital=INITIAL_CAPITAL):
    weekly_prices = prices_df.resample('W-MON').first()
    weekly_wti = wti.resample('W-MON').first()

    common_idx = weekly_prices.index.intersection(weekly_wti.index)
    weekly_prices = weekly_prices.loc[common_idx]
    weekly_wti = weekly_wti.loc[common_idx]

    weights_history = pd.DataFrame(0.0, index=common_idx, columns=universe)
    portfolio_value = pd.Series(index=common_idx, dtype=float)

    print(f"Running oil-DCF backtest over {len(common_idx)} weeks...\n")

    for i, date in enumerate(common_idx):
        wti_price = weekly_wti.loc[date]
        if pd.isna(wti_price):
            continue

        fair_values = {}
        for ticker in universe:
            if ticker not in fundamentals:
                continue
            inputs = {**DEFAULT_ASSUMPTIONS, **MACRO_ASSUMPTIONS,
                      **fundamentals[ticker], **COMPANY_OVERRIDES.get(ticker, {})}
            try:
                fair_values[ticker] = fair_value(inputs, wti_price, seed=hash((ticker, str(date))) % (2**32))
            except Exception:
                continue

        week_prices = {t: weekly_prices.loc[date, t] for t in universe if t in weekly_prices.columns}
        week_prices = {t: p for t, p in week_prices.items() if not pd.isna(p)}

        weights = build_weekly_weights(fair_values, week_prices)
        for ticker, w in weights.items():
            weights_history.loc[date, ticker] = w

        if i == 0:
            portfolio_value.iloc[i] = initial_capital
        else:
            prev_date = common_idx[i - 1]
            price_changes = (weekly_prices.loc[date] / weekly_prices.loc[prev_date] - 1).fillna(0)
            prev_weights = weights_history.loc[prev_date]
            period_return = (prev_weights * price_changes).sum()  # un-invested weight (cash) earns 0
            portfolio_value.iloc[i] = portfolio_value.iloc[i - 1] * (1 + period_return)

        if (i + 1) % 25 == 0:
            print(f"   ...week {i+1}/{len(common_idx)} ({date.date()})")

    print("\nBacktest complete.\n")
    return portfolio_value, weights_history, weekly_prices


def create_benchmark(weekly_prices, initial_capital=INITIAL_CAPITAL):
    benchmark_weights = pd.DataFrame(index=weekly_prices.index, columns=weekly_prices.columns, dtype=float)
    for date in weekly_prices.index:
        valid = weekly_prices.loc[date].dropna()
        if len(valid) > 0:
            benchmark_weights.loc[date, valid.index] = 1.0 / len(valid)
    benchmark_weights = benchmark_weights.fillna(0)

    value = pd.Series(index=weekly_prices.index, dtype=float)
    for i, date in enumerate(weekly_prices.index):
        if i == 0:
            value.iloc[i] = initial_capital
        else:
            prev_date = weekly_prices.index[i - 1]
            price_changes = (weekly_prices.loc[date] / weekly_prices.loc[prev_date] - 1).fillna(0)
            ret = (benchmark_weights.loc[prev_date] * price_changes).sum()
            value.iloc[i] = value.iloc[i - 1] * (1 + ret)
    return value


# ============================================================================
# 6. METRICS
# ============================================================================

def calculate_turnover(weights_history):
    diffs = weights_history.diff().abs().sum(axis=1)
    return diffs.mean()  # average weekly one-way turnover


def _series_metrics(value_series):
    """Raw-decimal metrics for one value series (strategy or benchmark),
    matching the schema used by docs/projects/moving_average_v1/metrics.json:
    total_return/cagr/volatility/max_drawdown as raw decimals (7.20 = +720%,
    not 720), sharpe as a unitless ratio.
    """
    total_return = value_series.iloc[-1] / value_series.iloc[0] - 1
    years = (value_series.index[-1] - value_series.index[0]).days / 365.25
    cagr = (value_series.iloc[-1] / value_series.iloc[0]) ** (1 / years) - 1
    volatility = value_series.pct_change().std() * np.sqrt(52)
    sharpe = cagr / volatility if volatility > 0 else 0
    running_max = value_series.expanding().max()
    max_drawdown = ((value_series - running_max) / running_max).min()
    return {
        'total_return': float(total_return),
        'cagr': float(cagr),
        'volatility': float(volatility),
        'sharpe': float(sharpe),
        'max_drawdown': float(max_drawdown),
    }


def calculate_metrics(portfolio_value, benchmark_value, weights_history):
    strategy = _series_metrics(portfolio_value)
    benchmark = _series_metrics(benchmark_value)
    turnover = calculate_turnover(weights_history)

    print("=" * 70)
    print("PERFORMANCE RESULTS")
    print("=" * 70)
    print(f"\nTotal return -- Strategy: {strategy['total_return']*100:.2f}%  "
          f"Benchmark: {benchmark['total_return']*100:.2f}%  "
          f"Alpha: {(strategy['total_return']-benchmark['total_return'])*100:+.2f}%")
    print(f"CAGR -- Strategy: {strategy['cagr']*100:.2f}%  Benchmark: {benchmark['cagr']*100:.2f}%")
    print(f"Volatility -- Strategy: {strategy['volatility']*100:.2f}%  Benchmark: {benchmark['volatility']*100:.2f}%")
    print(f"Sharpe (return/vol) -- Strategy: {strategy['sharpe']:.2f}  Benchmark: {benchmark['sharpe']:.2f}")
    print(f"Max drawdown -- Strategy: {strategy['max_drawdown']*100:.2f}%  Benchmark: {benchmark['max_drawdown']*100:.2f}%")
    print(f"Avg weekly turnover: {turnover*100:.1f}%")
    print("=" * 70 + "\n")

    return {
        'period': {
            'start': str(portfolio_value.index[0].date()),
            'end': str(portfolio_value.index[-1].date()),
        },
        'metrics': {
            'strategy': strategy,
            'benchmark': benchmark,
        },
        'avg_weekly_turnover': float(turnover),
    }


# ============================================================================
# 7. SAVE RESULTS
# ============================================================================

def save_results(metrics_dict, weights_history):
    """
    Writes metrics.json in the same schema as
    docs/projects/moving_average_v1/metrics.json, so it can be dropped
    directly into a new docs/projects/oil-dcf-backtest/ folder. Also
    saves the weekly weights to CSV for your own debugging/analysis --
    not required by the site, just useful to have.
    """
    import json
    output_dir = '../data' if os.path.exists('../data') else ('data' if os.path.exists('data') else '.')

    with open(os.path.join(output_dir, 'oil_dcf_metrics.json'), 'w') as f:
        json.dump(metrics_dict, f, indent=2)

    weights_history.to_csv(os.path.join(output_dir, 'oil_dcf_backtest_weights.csv'))
    print(f"Saved oil_dcf_metrics.json and oil_dcf_backtest_weights.csv to {output_dir}/")


# ============================================================================
# 8. MAIN
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("  WEEKLY OIL-DCF REBALANCING BACKTEST")
    print("=" * 70 + "\n")

    prices_df, wti = download_all_data(UNIVERSE, WTI_TICKER, START_DATE, END_DATE)
    if prices_df is None:
        return

    fundamentals = fetch_static_fundamentals(UNIVERSE)
    if not fundamentals:
        print("No fundamentals available -- cannot proceed.")
        return

    active_universe = [t for t in UNIVERSE if t in fundamentals]
    print(f"\nActive universe ({len(active_universe)}): {active_universe}\n")

    portfolio_value, weights_history, weekly_prices = run_backtest(
        prices_df, wti, active_universe, fundamentals
    )
    benchmark_value = create_benchmark(weekly_prices[active_universe])
    metrics_dict = calculate_metrics(portfolio_value, benchmark_value, weights_history)
    save_results(metrics_dict, weights_history)

    print("BACKTEST COMPLETE\n")


if __name__ == "__main__":
    main()
