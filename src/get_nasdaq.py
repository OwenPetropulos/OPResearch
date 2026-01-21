import yfinance as yf
import pandas as pd

# Download Nasdaq 100 (ticker: QQQ or ^NDX)
data = yf.download('QQQ', start='2015-01-02', end='2026-01-06', interval='1wk')

# Save to CSV
data.to_csv('nasdaq100_weekly.csv')

print(f"✅ Downloaded {len(data)} weeks of Nasdaq 100 data")
print(f"   Saved to: nasdaq100_weekly.csv")
print(f"\nFirst few rows:")
print(data.head())
print(f"\nLast few rows:")
print(data.tail())