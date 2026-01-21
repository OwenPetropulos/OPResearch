"""
Calculate hypothetical growth of $100,000 invested in Nasdaq 100
From Jan 1, 2016 to Jan 6, 2026
"""

import yfinance as yf
import pandas as pd
import numpy as np

print("\n" + "=" * 70)
print("  NASDAQ 100 INVESTMENT GROWTH CALCULATOR")
print("  Initial Investment: $100,000")
print("=" * 70)
print()

# Download Nasdaq 100 data (using QQQ ETF as proxy)
print("📊 Downloading Nasdaq 100 data (QQQ)...")

# Use the Ticker method (more reliable)
qqq = yf.Ticker('QQQ')
data = qqq.history(start='2016-01-01', end='2026-01-09')

if len(data) == 0:
    print("❌ Failed to download data")
    exit()

print(f"✅ Downloaded {len(data)} days of data")
print()

# Use Close prices (already adjusted in yfinance's history method)
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

# Calculate CAGR (Compound Annual Growth Rate)
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

# Create detailed CSV
results_df = pd.DataFrame({
    'Date': portfolio_value.index,
    'QQQ_Price': prices.values,
    'Portfolio_Value': portfolio_value.values,
    'Return_%': ((portfolio_value / initial_investment - 1) * 100).values,
    'Drawdown_%': drawdown.values
})

results_df.to_csv('nasdaq100_growth.csv', index=False)
print("💾 Saved detailed results to: nasdaq100_growth.csv")
print()

# Weekly summary
weekly_value = portfolio_value.resample('W-FRI').last()
weekly_df = pd.DataFrame({
    'Date': weekly_value.index,
    'Portfolio_Value': weekly_value.values,
    'Return_%': ((weekly_value / initial_investment - 1) * 100).values
})
weekly_df.to_csv('nasdaq100_weekly_growth.csv', index=False)
print("💾 Saved weekly summary to: nasdaq100_weekly_growth.csv")
print()

# Show key milestones
print("🎯 KEY MILESTONES:")
print()

milestones = [150000, 200000, 250000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000]
for milestone in milestones:
    if final_value >= milestone:
        dates_reached = portfolio_value[portfolio_value >= milestone]
        if len(dates_reached) > 0:
            first_date = dates_reached.index[0]
            days_to_reach = (first_date - start_date).days
            years_to_reach = days_to_reach / 365.25
            print(f"   ${milestone:>9,} reached on {first_date.date()} ({years_to_reach:.1f} years)")

print()

# Compare to your strategy
try:
    strategy_results = pd.read_csv('portfolio_performance.csv')
    strategy_final = strategy_results['Strategy_Value'].iloc[-1]
    benchmark_final = strategy_results['Benchmark_Value'].iloc[-1]
    
    print("=" * 70)
    print("📊 COMPARISON TO YOUR STRATEGY")
    print("=" * 70)
    print()
    print(f"Investment: $100,000 (Jan 2016 - Jan 2026)")
    print()
    print(f"   Your Strategy (10 stocks):    ${strategy_final:>12,.2f}   ({(strategy_final/100000-1)*100:>6.2f}%)")
    print(f"   Your Benchmark (equal-wt):    ${benchmark_final:>12,.2f}   ({(benchmark_final/100000-1)*100:>6.2f}%)")
    print(f"   Nasdaq 100 (QQQ):             ${final_value:>12,.2f}   ({total_return:>6.2f}%)")
    print()
    
    if strategy_final > final_value:
        diff = strategy_final - final_value
        print(f"🏆 Your strategy BEAT Nasdaq 100 by ${diff:,.2f}!")
    else:
        diff = final_value - strategy_final
        print(f"📉 Nasdaq 100 beat your strategy by ${diff:,.2f}")
    
    print()
    print("=" * 70)
    
except:
    print("💡 Run your backtest first to see comparison!")

print()
print("🎉 Analysis complete!")
print()