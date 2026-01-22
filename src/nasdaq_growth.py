"""
Calculate hypothetical growth of $100,000 invested in Nasdaq 100
ALIGNED WITH YOUR STRATEGY - Starting Jan 2016
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os

print("\n" + "=" * 70)
print("  NASDAQ 100 INVESTMENT GROWTH CALCULATOR")
print("  Initial Investment: $100,000")
print("  START DATE: January 2016 (aligned with your strategy)")
print("=" * 70)
print()

# Download Nasdaq 100 data
print("📊 Downloading Nasdaq 100 data (QQQ)...")

qqq = yf.Ticker('QQQ')
# Start from 2016 to match when your strategy actually starts trading
data = qqq.history(start='2016-01-04', end='2026-01-09')

if len(data) == 0:
    print("❌ Failed to download data")
    exit()

print(f"✅ Downloaded {len(data)} days of data")
print()

# Use Close prices
prices = data['Close']

# Initial investment
initial_investment = 100000

# Calculate number of shares you could buy on day 1
initial_price = prices.iloc[0]
shares = initial_investment / initial_price

# Calculate portfolio value over time
portfolio_value = shares * prices

# Final value
final_value = portfolio_value.iloc[-1]
total_return = ((final_value / initial_investment) - 1) * 100

# Calculate CAGR
start_date = prices.index[0]
end_date = prices.index[-1]
years = (end_date - start_date).days / 365.25
cagr = ((final_value / initial_investment) ** (1/years) - 1) * 100

# Calculate max drawdown
cummax = portfolio_value.expanding().max()
drawdown = ((portfolio_value - cummax) / cummax * 100)
max_drawdown = drawdown.min()

# Calculate volatility
daily_returns = portfolio_value.pct_change().dropna()
annual_volatility = daily_returns.std() * np.sqrt(252) * 100

# Print results
print("=" * 70)
print("📈 RESULTS")
print("=" * 70)
print()
print(f"💵 INITIAL INVESTMENT:")
print(f"   Date:        {start_date.date()}")
print(f"   Amount:      ${initial_investment:,.2f}")
print(f"   QQQ Price:   ${initial_price:.2f}")
print(f"   Shares:      {shares:.4f}")
print()
print(f"💰 FINAL VALUE:")
print(f"   Date:        {end_date.date()}")
print(f"   QQQ Price:   ${prices.iloc[-1]:.2f}")
print(f"   Portfolio:   ${final_value:,.2f}")
print()
print(f"📊 PERFORMANCE:")
print(f"   Total Return:      {total_return:>8.2f}%")
print(f"   Annualized (CAGR): {cagr:>8.2f}%")
print(f"   Time Period:       {years:.2f} years")
print()
print(f"📉 RISK:")
print(f"   Max Drawdown:      {max_drawdown:>8.2f}%")
print(f"   Volatility:        {annual_volatility:>8.2f}%")
print()
print(f"⚡ SHARPE RATIO:       {cagr / annual_volatility if annual_volatility > 0 else 0:>8.2f}")
print()
print(f"💡 PROFIT:")
print(f"   Gain/Loss:         ${final_value - initial_investment:>,.2f}")
print()
print("=" * 70)
print()

# Save results to data folder
data_dir = '../data' if os.path.exists('../data') else 'data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

results_df = pd.DataFrame({
    'Date': portfolio_value.index,
    'QQQ_Price': prices.values,
    'Portfolio_Value': portfolio_value.values,
    'Return_%': ((portfolio_value / initial_investment - 1) * 100).values,
    'Drawdown_%': drawdown.values
})

output_path = os.path.join(data_dir, 'nasdaq100_growth_aligned.csv')
results_df.to_csv(output_path, index=False)
print(f"💾 Saved detailed results to: {output_path}")
print()

# Weekly summary
weekly_value = portfolio_value.resample('W-FRI').last()
weekly_df = pd.DataFrame({
    'Date': weekly_value.index,
    'Portfolio_Value': weekly_value.values,
    'Return_%': ((weekly_value / initial_investment - 1) * 100).values
})

weekly_path = os.path.join(data_dir, 'nasdaq100_weekly_growth_aligned.csv')
weekly_df.to_csv(weekly_path, index=False)
print(f"💾 Saved weekly summary to: {weekly_path}")
print()

# Compare to your strategy
try:
    # Try to find portfolio_performance.csv in data folder
    perf_path = os.path.join(data_dir, 'portfolio_performance.csv')
    strategy_results = pd.read_csv(perf_path)
    
    # Get the first date from your strategy
    strategy_start = pd.to_datetime(strategy_results['Date'].iloc[0])
    strategy_final = strategy_results['Strategy_Value'].iloc[-1]
    benchmark_final = strategy_results['Benchmark_Value'].iloc[-1]
    
    print("=" * 70)
    print("📊 APPLES-TO-APPLES COMPARISON")
    print("=" * 70)
    print()
    print(f"All starting from: {strategy_start.date()}")
    print(f"Initial investment: $100,000")
    print()
    print(f"   Your Strategy (10 stocks):    ${strategy_final:>12,.2f}   ({(strategy_final/100000-1)*100:>6.2f}%)")
    print(f"   Your Benchmark (equal-wt):    ${benchmark_final:>12,.2f}   ({(benchmark_final/100000-1)*100:>6.2f}%)")
    print(f"   Nasdaq 100 (QQQ):             ${final_value:>12,.2f}   ({total_return:>6.2f}%)")
    print()
    
    # Calculate CAGRs
    strat_cagr = ((strategy_final / 100000) ** (1/years) - 1) * 100
    bench_cagr = ((benchmark_final / 100000) ** (1/years) - 1) * 100
    
    print(f"📊 ANNUALIZED RETURNS (CAGR):")
    print(f"   Your Strategy:     {strat_cagr:>6.2f}%")
    print(f"   Your Benchmark:    {bench_cagr:>6.2f}%")
    print(f"   Nasdaq 100:        {cagr:>6.2f}%")
    print()
    
    if strategy_final > final_value:
        diff = strategy_final - final_value
        pct_better = ((strategy_final / final_value) - 1) * 100
        print(f"🏆 Your strategy BEAT Nasdaq 100 by ${diff:,.2f} ({pct_better:+.2f}%)")
    else:
        diff = final_value - strategy_final
        pct_worse = ((final_value / strategy_final) - 1) * 100
        print(f"📉 Nasdaq 100 beat your strategy by ${diff:,.2f} ({pct_worse:+.2f}%)")
    
    print()
    print("=" * 70)
    
except Exception as e:
    print(f"💡 Could not load strategy results from data folder: {e}")
    print(f"   Looking for: {perf_path}")

print()
print("🎉 Analysis complete!")
print()