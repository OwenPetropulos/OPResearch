from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

INPUT_FILE = Path("data/portfolio_performance.csv")
OUTPUT_FILE = Path("docs/projects/moving_average_v1/metrics.json")

WEEKS_PER_YEAR = 52
RISK_FREE_ANNUAL = 0.0  # keep 0.0 for now


def _max_drawdown(values: np.ndarray) -> float:
    peak = np.maximum.accumulate(values)
    dd = values / peak - 1.0
    return float(dd.min())


def _compute_metrics(values: np.ndarray, dates: pd.Series) -> dict:
    # weekly returns
    rets = pd.Series(values, index=dates).pct_change().dropna().values
    if len(rets) < 10:
        raise ValueError("Not enough weekly observations to compute stable metrics.")

    total_return = values[-1] / values[0] - 1.0

    years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
    cagr = (values[-1] / values[0]) ** (1 / years) - 1.0 if years > 0 else float("nan")

    vol_ann = np.std(rets, ddof=1) * np.sqrt(WEEKS_PER_YEAR)
    sharpe = (cagr - RISK_FREE_ANNUAL) / vol_ann if vol_ann != 0 else float("nan")

    return {
        "total_return": float(total_return),
        "annualized_return": float(cagr),
        "volatility": float(vol_ann),
        "sharpe": float(sharpe),
        "max_drawdown": _max_drawdown(values),
    }


def _read_portfolio_performance(path: Path) -> pd.DataFrame:
    """
    Robust loader:
    - If file has headers, looks for date/value columns.
    - If file has no headers, assumes:
        col0 = date
        col1 = strategy_value
        col2 = benchmark_value
      and ignores extra columns.
    """
    df = pd.read_csv(path)

    # Detect headerless CSV: first column name looks like a date string
    first_col_name = str(df.columns[0])
    headerless = first_col_name.startswith("20") or "00:00:00" in first_col_name

    if headerless:
        df = pd.read_csv(path, header=None)
        if df.shape[1] < 3:
            raise ValueError(f"{path} needs at least 3 columns (date, strategy, benchmark). Found {df.shape[1]}")
        df = df.iloc[:, :3].copy()
        df.columns = ["date", "strategy_value", "benchmark_value"]
    else:
        # Normalize column names
        df.columns = [c.strip() for c in df.columns]

        # Must have a date column
        date_candidates = [c for c in df.columns if c.lower() in ["date", "datetime", "time"]]
        if not date_candidates:
            raise ValueError(f"Could not find a date column in: {list(df.columns)}")
        date_col = date_candidates[0]

        # Try to identify strategy/benchmark value columns
        # Prefer obvious names first; fall back to "Portfolio" and "NASDAQ 100" (from your earlier data format).
        strat_candidates = [c for c in df.columns if c.lower() in ["strategy", "portfolio", "portfolio_value", "strategy_value"]]
        bench_candidates = [c for c in df.columns if c.lower() in ["benchmark", "nasdaq 100", "nasdaq100", "qqq", "benchmark_value"]]

        # If not found by names, use the first two numeric columns after date
        tmp = df.copy()
        tmp[date_col] = tmp[date_col]
        numeric_cols = [c for c in df.columns if c != date_col]

        if not strat_candidates or not bench_candidates:
            # try numeric coercion to identify numeric columns
            numeric_like = []
            for c in numeric_cols:
                s = pd.to_numeric(df[c], errors="coerce")
                if s.notna().sum() > 0.9 * len(df):  # mostly numeric
                    numeric_like.append(c)
            if len(numeric_like) < 2:
                raise ValueError(f"Could not infer strategy/benchmark columns from: {list(df.columns)}")
            strat_col, bench_col = numeric_like[0], numeric_like[1]
        else:
            strat_col = strat_candidates[0]
            bench_col = bench_candidates[0]

        df = df[[date_col, strat_col, bench_col]].copy()
        df.columns = ["date", "strategy_value", "benchmark_value"]

    # Parse dates (handles your timezone suffix like -05:00)
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_convert(None)
    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"])

    df["strategy_value"] = pd.to_numeric(df["strategy_value"], errors="coerce")
    df["benchmark_value"] = pd.to_numeric(df["benchmark_value"], errors="coerce")
    df = df.dropna(subset=["strategy_value", "benchmark_value"])

    return df


def main():
    df = _read_portfolio_performance(INPUT_FILE)

    # Overlap window is inherently enforced by rows where both values exist
    start = df["date"].iloc[0]
    end = df["date"].iloc[-1]

    strat = _compute_metrics(df["strategy_value"].values.astype(float), df["date"])
    bench = _compute_metrics(df["benchmark_value"].values.astype(float), df["date"])

    out = {
        "period_start": start.strftime("%Y-%m-%d"),
        "period_end": end.strftime("%Y-%m-%d"),
        "strategy": {k: round(v, 4) for k, v in strat.items()},
        "benchmark": {k: round(v, 4) for k, v in bench.items()},
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
