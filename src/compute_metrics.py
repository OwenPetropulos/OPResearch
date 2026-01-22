import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "docs" / "projects" / "moving_average_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PORTFOLIO_CSV = DATA_DIR / "portfolio_performance.csv"
NASDAQ_ALIGNED_CSV = DATA_DIR / "nasdaq100_weekly_growth_aligned.csv"

OUT_JSON = OUT_DIR / "metrics.json"


def _read_datesafe(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    s = pd.to_datetime(df[col], errors="coerce", utc=True)
    df[col] = s.dt.tz_convert(None)
    return df.dropna(subset=[col])


def _weekly_returns_from_values(values: pd.Series) -> pd.Series:
    # simple returns
    r = values.pct_change()
    return r.dropna()


def _max_drawdown(values: pd.Series) -> float:
    v = values.astype(float)
    peak = v.cummax()
    dd = (v / peak) - 1.0
    return float(dd.min())


def _metrics_from_weekly_values(values: pd.Series) -> dict:
    v = values.dropna().astype(float)
    if len(v) < 3:
        return {
            "total_return": None,
            "cagr": None,
            "volatility": None,
            "sharpe": None,
            "max_drawdown": None,
        }

    rets = _weekly_returns_from_values(v)

    total_return = float(v.iloc[-1] / v.iloc[0] - 1.0)

    # Weekly -> annual (52)
    n_weeks = len(v) - 1
    years = n_weeks / 52.0 if n_weeks > 0 else None
    cagr = float((v.iloc[-1] / v.iloc[0]) ** (1.0 / years) - 1.0) if years and years > 0 else None

    vol = float(rets.std(ddof=1) * math.sqrt(52)) if len(rets) > 1 else None

    # Sharpe with rf = 0
    sharpe = None
    if len(rets) > 1 and rets.std(ddof=1) != 0:
        sharpe = float((rets.mean() / rets.std(ddof=1)) * math.sqrt(52))

    mdd = _max_drawdown(v)

    return {
        "total_return": total_return,
        "cagr": cagr,
        "volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
    }


def main():
    if not PORTFOLIO_CSV.exists():
        raise FileNotFoundError(f"Missing {PORTFOLIO_CSV}")

    df = pd.read_csv(PORTFOLIO_CSV)
    if "Date" not in df.columns:
        raise ValueError(f"{PORTFOLIO_CSV} must contain a 'Date' column. Found: {list(df.columns)}")

    df = _read_datesafe(df, "Date").sort_values("Date")

    # Expect these columns (based on your screenshot error earlier)
    # Date, Strategy_Value, Benchmark_Value, ...
    needed = {"Strategy_Value", "Benchmark_Value"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Expected {needed} in {PORTFOLIO_CSV}. Missing: {missing}. Found: {list(df.columns)}")

    # Nasdaq aligned weekly file
    if not NASDAQ_ALIGNED_CSV.exists():
        raise FileNotFoundError(f"Missing {NASDAQ_ALIGNED_CSV}")

    qqq = pd.read_csv(NASDAQ_ALIGNED_CSV)
    if "Date" not in qqq.columns or "Portfolio_Value" not in qqq.columns:
        raise ValueError(
            f"{NASDAQ_ALIGNED_CSV} must contain columns: Date, Portfolio_Value. Found: {list(qqq.columns)}"
        )
    qqq = _read_datesafe(qqq, "Date").sort_values("Date").rename(columns={"Portfolio_Value": "Nasdaq100_Value"})

    # Merge on Date (inner = aligned sample)
    merged = pd.merge(df[["Date", "Strategy_Value", "Benchmark_Value"]], qqq[["Date", "Nasdaq100_Value"]], on="Date", how="inner")
    if merged.empty:
        raise ValueError("Merged series is empty. Check that your Date formats align across files.")

    start = merged["Date"].iloc[0].date().isoformat()
    end = merged["Date"].iloc[-1].date().isoformat()

    strat_m = _metrics_from_weekly_values(merged["Strategy_Value"])
    bench_m = _metrics_from_weekly_values(merged["Benchmark_Value"])
    nasdaq_m = _metrics_from_weekly_values(merged["Nasdaq100_Value"])

    payload = {
        "period": {"start": start, "end": end},
        "metrics": {
            "strategy": strat_m,
            "benchmark": bench_m,
            "nasdaq": nasdaq_m,
        },
    }

    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
