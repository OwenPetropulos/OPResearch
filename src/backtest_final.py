"""
Stock Rebalancing Backtest - Robust yfinance Version
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Try importing yfinance
try:
    import yfinance as yf
except ImportError:
    print("Installing yfinance...")
    import subprocess
    subprocess.check_call(['pip3', 'install', '--upgrade', 'yfinance'])
    import yfinance as yf

# ============================================================================
# CONFIGURATION
# ============================================================================

UNIVERSE = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 
            'XOM', 'BRK-B', 'JPM', 'JNJ', 'WMT']

START_DATE = '2015-01-01'
END_DATE = '2026-01-31'
LOOKBACK_DAYS = 252

# ============================================================================
# DOWNLOAD DATA
# ============================================================================

def download_stock_individually(ticker, start, end):
    """Download one stock at a time"""
    try:
        print(f"   {ticker}...", end=' ')
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start, end=end)
        
        if len(hist) > 0:
            # Get adjusted close (or close if adjusted not available)
            if 'Close' in hist.columns:
                prices = hist['Close']
            else:
                prices = hist['Adj Close']
            
            prices.name = ticker
            print(f"✅ {len(prices)} days")
            return prices
        else:
            print("❌ No data")
            return None
    except Exception as e:
        print(f"❌ {str(e)}")
        return None

def download_all_data(tickers, start, end):
    """Download all stocks one by one"""
    print("📊 Downloading price data...")
    print(f"   Period: {start} to {end}")
    print()
    
    all_prices = []
    
    for ticker in tickers:
        prices = download_stock_individually(ticker, start, end)
        if prices is not None:
            all_prices.append(prices)
        
        # Small delay to avoid rate limiting
        import time
        time.sleep(0.5)
    
    if len(all_prices) == 0:
        print("\n❌ Failed to download any data")
        return None
    
    # Combine all stocks
    prices_df = pd.concat(all_prices, axis=1)
    
    # Rename BRK-B to BRK.B
    if 'BRK-B' in prices_df.columns:
        prices_df.rename(columns={'BRK-B': 'BRK.B'}, inplace=True)
    
    print(f"\n✅ Successfully downloaded {len(all_prices)} stocks")
    print(f"   Total days: {len(prices_df)}")
    print(f"   Date range: {prices_df.index[0].date()} to {prices_df.index[-1].date()}")
    print(f"   Stocks: {', '.join(prices_df.columns)}")
    print()
    
    return prices_df

# ============================================================================
# BACKTEST FUNCTIONS
# ============================================================================

def calculate_trailing_returns(prices, lookback=252):
    """Calculate 12-month trailing returns"""
    print("📈 Calculating 12-month trailing returns...")
    returns_12m = pd.DataFrame(index=prices.index, columns=prices.columns)
    for ticker in prices.columns:
        returns_12m[ticker] = (prices[ticker] / prices[ticker].shift(lookback)) - 1
    print("✅ Done\n")
    return returns_12m

def generate_signals(returns_12m, prices):
    """Generate weekly signals"""
    print("🎯 Generating weekly signals...")
    weekly_returns = returns_12m.resample('W-FRI').last()
    weekly_prices = prices.resample('W-FRI').last()
    weights = pd.DataFrame(0.0, index=weekly_returns.index, columns=weekly_returns.columns)
    
    for date in weekly_returns.index:
        returns_row = weekly_returns.loc[date]
        valid_returns = returns_row.dropna()
        if len(valid_returns) == 0:
            continue
        
        avg_return = valid_returns.mean()
        distances = avg_return - valid_returns
        underperformers = distances[distances > 0]
        
        if len(underperformers) > 0:
            weight_sum = underperformers.sum()
            for ticker in underperformers.index:
                weights.loc[date, ticker] = underperformers[ticker] / weight_sum
        else:
            for ticker in valid_returns.index:
                weights.loc[date, ticker] = 1.0 / len(valid_returns)
    
    print(f"✅ Generated {len(weights)} weeks\n")
    return weights, weekly_prices

def run_backtest(weights, weekly_prices, initial_capital=100000):
    """Run the backtest"""
    print(f"🚀 Running backtest (${initial_capital:,.0f})...\n")
    portfolio_value = pd.Series(index=weights.index, dtype=float)
    portfolio_value.iloc[0] = initial_capital
    holdings = pd.DataFrame(0.0, index=weights.index, columns=weights.columns)
    
    for i in range(len(weights)):
        current_date = weights.index[i]
        if i == 0:
            current_value = initial_capital
        else:
            prev_date = weights.index[i-1]
            price_changes = (weekly_prices.loc[current_date] / weekly_prices.loc[prev_date]) - 1
            price_changes = price_changes.fillna(0)
            portfolio_return = (holdings.loc[prev_date] * price_changes).sum()
            current_value = portfolio_value.iloc[i-1] * (1 + portfolio_return)
        
        portfolio_value.iloc[i] = current_value
        for ticker in weights.columns:
            holdings.loc[current_date, ticker] = weights.loc[current_date, ticker]
    
    print("✅ Done\n")
    return portfolio_value, holdings

def create_benchmark(weekly_prices, initial_capital=100000):
    """Equal-weight benchmark"""
    print("📊 Creating benchmark...")
    benchmark_weights = pd.DataFrame(index=weekly_prices.index, columns=weekly_prices.columns)
    for date in weekly_prices.index:
        valid_stocks = weekly_prices.loc[date].dropna()
        n_stocks = len(valid_stocks)
        if n_stocks > 0:
            for ticker in valid_stocks.index:
                benchmark_weights.loc[date, ticker] = 1.0 / n_stocks
    benchmark_value, _ = run_backtest(benchmark_weights, weekly_prices, initial_capital)
    return benchmark_value

def calculate_metrics(portfolio_value, benchmark_value):
    """Calculate performance metrics"""
    print("\n" + "=" * 70)
    print("📈 PERFORMANCE RESULTS")
    print("=" * 70)
    
    strat_ret = (portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1) * 100
    bench_ret = (benchmark_value.iloc[-1] / benchmark_value.iloc[0] - 1) * 100
    
    print(f"\n💰 TOTAL RETURNS:")
    print(f"   Strategy:  {strat_ret:>8.2f}%")
    print(f"   Benchmark: {bench_ret:>8.2f}%")
    print(f"   Alpha:     {strat_ret - bench_ret:>+8.2f}%")
    
    years = (portfolio_value.index[-1] - portfolio_value.index[0]).days / 365.25
    strat_cagr = ((portfolio_value.iloc[-1] / portfolio_value.iloc[0]) ** (1/years) - 1) * 100
    bench_cagr = ((benchmark_value.iloc[-1] / benchmark_value.iloc[0]) ** (1/years) - 1) * 100
    
    print(f"\n📊 ANNUALIZED RETURNS:")
    print(f"   Strategy:  {strat_cagr:>8.2f}%")
    print(f"   Benchmark: {bench_cagr:>8.2f}%")
    
    strat_vol = portfolio_value.pct_change().std() * np.sqrt(52) * 100
    bench_vol = benchmark_value.pct_change().std() * np.sqrt(52) * 100
    
    print(f"\n📉 VOLATILITY:")
    print(f"   Strategy:  {strat_vol:>8.2f}%")
    print(f"   Benchmark: {bench_vol:>8.2f}%")
    
    strat_sharpe = strat_cagr / strat_vol if strat_vol > 0 else 0
    bench_sharpe = bench_cagr / bench_vol if bench_vol > 0 else 0
    
    print(f"\n⚡ SHARPE RATIO:")
    print(f"   Strategy:  {strat_sharpe:>8.2f}")
    print(f"   Benchmark: {bench_sharpe:>8.2f}")
    
    strat_dd = ((portfolio_value - portfolio_value.expanding().max()) / portfolio_value.expanding().max() * 100).min()
    bench_dd = ((benchmark_value - benchmark_value.expanding().max()) / benchmark_value.expanding().max() * 100).min()
    
    print(f"\n📉 MAX DRAWDOWN:")
    print(f"   Strategy:  {strat_dd:>8.2f}%")
    print(f"   Benchmark: {bench_dd:>8.2f}%")
    print("\n" + "=" * 70 + "\n")

def save_results(portfolio_value, benchmark_value, weights, holdings):
    """Save to CSV"""
    print("💾 Saving results...")
    
    results = pd.DataFrame({
        'Date': portfolio_value.index,
        'Strategy_Value': portfolio_value.values,
        'Benchmark_Value': benchmark_value.values,
        'Strategy_Return_%': ((portfolio_value / portfolio_value.iloc[0]) - 1) * 100,
        'Benchmark_Return_%': ((benchmark_value / benchmark_value.iloc[0]) - 1) * 100
    })
    results.to_csv('portfolio_performance.csv', index=False)
    weights.to_csv('weekly_weights.csv')
    holdings.to_csv('weekly_holdings.csv')
    print("   ✅ Saved all files\n")

import plotly.graph_objects as go
from pathlib import Path
import pandas as pd

out_dir = Path("docs/figures")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("data/portfolio_performance.csv")
df["Date"] = pd.to_datetime(df["Date"])

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df["Date"],
    y=df["Portfolio"],
    mode="lines",
    name="Strategy"
))

if "Benchmark" in df.columns:
    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["Benchmark"],
        mode="lines",
        name="Benchmark"
    ))

fig.update_layout(
    title="Moving Average Strategy Backtest",
    xaxis_title="Date",
    yaxis_title="Portfolio Value",
    template="plotly_white"
)

fig.write_html(out_dir / "moving_average_backtest.html", include_plotlyjs="cdn")
print("Interactive plot written to docs/figures/")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("  STOCK REBALANCING BACKTEST")
    print("  Strategy: Overweight 12-Month Underperformers")
    print("=" * 70)
    print()
    
    prices = download_all_data(UNIVERSE, START_DATE, END_DATE)
    
    if prices is None or len(prices.columns) < 3:
        print("\n❌ Not enough data downloaded to run backtest")
        print("   Need at least 3 stocks")
        return
    
    returns_12m = calculate_trailing_returns(prices, LOOKBACK_DAYS)
    weights, weekly_prices = generate_signals(returns_12m, prices)
    portfolio_value, holdings = run_backtest(weights, weekly_prices)
    benchmark_value = create_benchmark(weekly_prices)
    calculate_metrics(portfolio_value, benchmark_value)
    save_results(portfolio_value, benchmark_value, weights, holdings)
    
    print("🎉 BACKTEST COMPLETE!\n")

if __name__ == "__main__":
    main()