"""
Stock Rebalancing Backtest Algorithm
Universe: 10 high-quality equities
Strategy: Overweight 12-month underperformers vs group average
Period: January 2016 - January 2026
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

UNIVERSE = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 
            'XOM', 'BRK-B', 'JPM', 'JNJ', 'WMT']

START_DATE = '2015-01-01'
END_DATE = '2026-01-31'
LOOKBACK_DAYS = 252
REBALANCE_FREQUENCY = 'W-FRI'

# ============================================================================
# STEP 1: DOWNLOAD PRICE DATA
# ============================================================================

def download_data(tickers, start, end):
    """Download daily adjusted close prices from Yahoo Finance"""
    print("📊 Downloading price data from Yahoo Finance...")
    print(f"   Tickers: {', '.join(tickers)}")
    print(f"   Period: {start} to {end}")
    print()
    
    # Download data
    data = yf.download(tickers, start=start, end=end, progress=False)
    
    # Handle different yfinance return structures
    if isinstance(data.columns, pd.MultiIndex):
        # Multiple tickers - extract Adj Close
        prices = data['Adj Close'].copy()
    else:
        # Single ticker case (shouldn't happen with our list, but just in case)
        prices = pd.DataFrame(data['Adj Close'])
        prices.columns = tickers
    
    # Rename BRK-B to BRK.B for consistency
    if 'BRK-B' in prices.columns:
        prices.rename(columns={'BRK-B': 'BRK.B'}, inplace=True)
    
    print(f"✅ Downloaded {len(prices)} days of data")
    print(f"   Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"   Stocks: {', '.join(prices.columns)}")
    print()
    
    return prices

# ============================================================================
# STEP 2: CALCULATE 12-MONTH RETURNS
# ============================================================================

def calculate_trailing_returns(prices, lookback=252):
    """Calculate 12-month trailing returns"""
    print("📈 Calculating 12-month trailing returns...")
    
    returns_12m = pd.DataFrame(index=prices.index, columns=prices.columns)
    
    for ticker in prices.columns:
        returns_12m[ticker] = (prices[ticker] / prices[ticker].shift(lookback)) - 1
    
    print(f"✅ Calculated trailing returns")
    print()
    
    return returns_12m

# ============================================================================
# STEP 3: GENERATE WEEKLY SIGNALS
# ============================================================================

def generate_signals(returns_12m, prices):
    """Generate weekly rebalancing signals"""
    print("🎯 Generating weekly rebalancing signals...")
    
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
    
    print(f"✅ Generated signals for {len(weights)} weeks")
    print()
    
    return weights, weekly_prices

# ============================================================================
# STEP 4: RUN BACKTEST
# ============================================================================

def run_backtest(weights, weekly_prices, initial_capital=100000):
    """Execute the backtest with weekly rebalancing"""
    print("🚀 Running backtest...")
    print(f"   Initial capital: ${initial_capital:,.0f}")
    print()
    
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
        current_weights = weights.loc[current_date]
        
        for ticker in weights.columns:
            holdings.loc[current_date, ticker] = current_weights[ticker]
    
    print(f"✅ Backtest complete")
    print()
    
    return portfolio_value, holdings

# ============================================================================
# STEP 5: CREATE BENCHMARK
# ============================================================================

def create_benchmark(weekly_prices, initial_capital=100000):
    """Equal-weight buy-and-hold benchmark"""
    print("📊 Creating equal-weight benchmark...")
    
    benchmark_weights = pd.DataFrame(index=weekly_prices.index, columns=weekly_prices.columns)
    
    for date in weekly_prices.index:
        valid_stocks = weekly_prices.loc[date].dropna()
        n_stocks = len(valid_stocks)
        if n_stocks > 0:
            for ticker in valid_stocks.index:
                benchmark_weights.loc[date, ticker] = 1.0 / n_stocks
    
    benchmark_value, _ = run_backtest(benchmark_weights, weekly_prices, initial_capital)
    
    return benchmark_value

# ============================================================================
# STEP 6: CALCULATE PERFORMANCE METRICS
# ============================================================================

def calculate_metrics(portfolio_value, benchmark_value):
    """Calculate performance statistics"""
    print("📈 PERFORMANCE RESULTS")
    print("=" * 70)
    
    strategy_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1) * 100
    benchmark_return = (benchmark_value.iloc[-1] / benchmark_value.iloc[0] - 1) * 100
    
    print(f"\n💰 TOTAL RETURNS:")
    print(f"   Strategy:  {strategy_return:>8.2f}%")
    print(f"   Benchmark: {benchmark_return:>8.2f}%")
    print(f"   Difference: {strategy_return - benchmark_return:>7.2f}%")
    
    years = (portfolio_value.index[-1] - portfolio_value.index[0]).days / 365.25
    strategy_cagr = ((portfolio_value.iloc[-1] / portfolio_value.iloc[0]) ** (1/years) - 1) * 100
    benchmark_cagr = ((benchmark_value.iloc[-1] / benchmark_value.iloc[0]) ** (1/years) - 1) * 100
    
    print(f"\n📊 ANNUALIZED RETURNS (CAGR):")
    print(f"   Strategy:  {strategy_cagr:>8.2f}%")
    print(f"   Benchmark: {benchmark_cagr:>8.2f}%")
    
    strategy_weekly_returns = portfolio_value.pct_change().dropna()
    benchmark_weekly_returns = benchmark_value.pct_change().dropna()
    
    strategy_vol = strategy_weekly_returns.std() * np.sqrt(52) * 100
    benchmark_vol = benchmark_weekly_returns.std() * np.sqrt(52) * 100
    
    print(f"\n📉 ANNUALIZED VOLATILITY:")
    print(f"   Strategy:  {strategy_vol:>8.2f}%")
    print(f"   Benchmark: {benchmark_vol:>8.2f}%")
    
    strategy_sharpe = strategy_cagr / strategy_vol if strategy_vol > 0 else 0
    benchmark_sharpe = benchmark_cagr / benchmark_vol if benchmark_vol > 0 else 0
    
    print(f"\n⚡ SHARPE RATIO:")
    print(f"   Strategy:  {strategy_sharpe:>8.2f}")
    print(f"   Benchmark: {benchmark_sharpe:>8.2f}")
    
    strategy_cummax = portfolio_value.expanding().max()
    strategy_dd = ((portfolio_value - strategy_cummax) / strategy_cummax * 100).min()
    
    benchmark_cummax = benchmark_value.expanding().max()
    benchmark_dd = ((benchmark_value - benchmark_cummax) / benchmark_cummax * 100).min()
    
    print(f"\n📉 MAXIMUM DRAWDOWN:")
    print(f"   Strategy:  {strategy_dd:>8.2f}%")
    print(f"   Benchmark: {benchmark_dd:>8.2f}%")
    
    print("\n" + "=" * 70)
    print()

# ============================================================================
# STEP 7: SAVE RESULTS
# ============================================================================

def save_results(portfolio_value, benchmark_value, weights, holdings):
    """Save results to CSV files"""
    print("💾 Saving results to files...")
    
    results = pd.DataFrame({
        'Date': portfolio_value.index,
        'Strategy_Value': portfolio_value.values,
        'Benchmark_Value': benchmark_value.values,
        'Strategy_Return_%': ((portfolio_value / portfolio_value.iloc[0]) - 1) * 100,
        'Benchmark_Return_%': ((benchmark_value / benchmark_value.iloc[0]) - 1) * 100
    })
    results.to_csv('portfolio_performance.csv', index=False)
    print("   ✅ Saved: portfolio_performance.csv")
    
    weights.to_csv('weekly_weights.csv')
    print("   ✅ Saved: weekly_weights.csv")
    
    holdings.to_csv('weekly_holdings.csv')
    print("   ✅ Saved: weekly_holdings.csv")
    
    print()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run the complete backtest"""
    print("\n" + "=" * 70)
    print("  STOCK REBALANCING BACKTEST")
    print("  Strategy: Overweight 12-Month Underperformers")
    print("  Universe: 10 High-Quality Equities")
    print("=" * 70)
    print()
    
    try:
        prices = download_data(UNIVERSE, START_DATE, END_DATE)
        returns_12m = calculate_trailing_returns(prices, LOOKBACK_DAYS)
        weights, weekly_prices = generate_signals(returns_12m, prices)
        portfolio_value, holdings = run_backtest(weights, weekly_prices)
        benchmark_value = create_benchmark(weekly_prices)
        calculate_metrics(portfolio_value, benchmark_value)
        save_results(portfolio_value, benchmark_value, weights, holdings)
        
        print("🎉 BACKTEST COMPLETE!")
        print()
        print("Next steps:")
        print("  1. Check 'portfolio_performance.csv' for equity curves")
        print("  2. Check 'weekly_weights.csv' for rebalancing weights")
        print("  3. Check 'weekly_holdings.csv' for position sizes")
        print()
    
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("\nTrying alternative download method...")
        
        # Try downloading one at a time as fallback
        prices_list = []
        for ticker in UNIVERSE:
            try:
                print(f"   Downloading {ticker}...")
                data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
                prices_list.append(data['Adj Close'].rename(ticker))
            except Exception as e2:
                print(f"   ⚠️  Failed to download {ticker}: {str(e2)}")
        
        if len(prices_list) > 0:
            prices = pd.concat(prices_list, axis=1)
            print(f"\n✅ Successfully downloaded {len(prices_list)} stocks")
            
            # Continue with backtest
            returns_12m = calculate_trailing_returns(prices, LOOKBACK_DAYS)
            weights, weekly_prices = generate_signals(returns_12m, prices)
            portfolio_value, holdings = run_backtest(weights, weekly_prices)
            benchmark_value = create_benchmark(weekly_prices)
            calculate_metrics(portfolio_value, benchmark_value)
            save_results(portfolio_value, benchmark_value, weights, holdings)
            
            print("🎉 BACKTEST COMPLETE!")

if __name__ == "__main__":
    main()