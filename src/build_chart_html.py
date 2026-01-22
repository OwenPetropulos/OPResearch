# src/build_chart_html.py
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from pathlib import Path


PORTFOLIO_CSV = Path("data/portfolio_performance.csv")
NASDAQ_CSV = Path("data/nasdaq100_weekly_growth_aligned.csv")  # must have Date + Portfolio_Value
OUT_HTML = Path("docs/projects/moving_average_v1/moving_average_backtest.html")


def _read_datesafe(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()

    # Robust parsing for strings like "2016-01-08 00:00:00-05:00"
    # Force UTC, then drop timezone so we can merge/plot cleanly.
    s = pd.to_datetime(df[col], errors="coerce", utc=True)

    # Convert to naive timestamps (no tz)
    df[col] = s.dt.tz_convert(None)

    # Drop any rows that still failed to parse
    df = df.dropna(subset=[col])
    return df



def main() -> None:
    if not PORTFOLIO_CSV.exists():
        raise FileNotFoundError(f"Missing: {PORTFOLIO_CSV}")

    if not NASDAQ_CSV.exists():
        raise FileNotFoundError(f"Missing: {NASDAQ_CSV}")

    # Strategy + Benchmark (weekly)
    df = pd.read_csv(PORTFOLIO_CSV)
    required_cols = {"Date", "Strategy_Value", "Benchmark_Value"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{PORTFOLIO_CSV} missing columns: {sorted(missing)}. Found: {list(df.columns)}")

    df = _read_datesafe(df, "Date").sort_values("Date")

    # Nasdaq weekly aligned growth file (your script output)
    ndx = pd.read_csv(NASDAQ_CSV)
    # Expecting: Date, Portfolio_Value, Return_% (based on your screenshot)
    if "Date" not in ndx.columns or "Portfolio_Value" not in ndx.columns:
        raise ValueError(
            f"{NASDAQ_CSV} must contain columns Date and Portfolio_Value. Found: {list(ndx.columns)}"
        )

    ndx = _read_datesafe(ndx, "Date").sort_values("Date")
    ndx = ndx.rename(columns={"Portfolio_Value": "Nasdaq100_Value"})

    # Merge on weekly dates (inner ensures perfect alignment)
    merged = pd.merge(df, ndx[["Date", "Nasdaq100_Value"]], on="Date", how="inner")

    if merged.empty:
        raise ValueError("Merged dataset is empty. Dates are not aligning between portfolio and Nasdaq series.")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=merged["Date"],
            y=merged["Strategy_Value"],
            mode="lines",
            name="Strategy",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=merged["Date"],
            y=merged["Benchmark_Value"],
            mode="lines",
            name="Benchmark (Equal-Weight)",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=merged["Date"],
            y=merged["Nasdaq100_Value"],
            mode="lines",
            name="Nasdaq 100 (QQQ)",
        )
    )

    fig.update_layout(
    title=dict(
        text="Strategy vs Benchmark vs Nasdaq 100 (QQQ)",
        x=0.5,
        xanchor="center",
        font=dict(
            size=22,
            family="Inter, system-ui, -apple-system, sans-serif",
            color="#0f172a"
        )
    ),

    # Light blue background to match site theme
    paper_bgcolor="#e6f0fa",
    plot_bgcolor="#e6f0fa",

    xaxis_title="Date",
    yaxis_title="Portfolio Value",

    xaxis=dict(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.05)"
    ),

    yaxis=dict(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.05)"
    ),

    margin=dict(l=40, r=40, t=120, b=50),

    legend=dict(
        orientation="h",
        x=0.5,
        xanchor="center",
        y=1.02,
        yanchor="bottom",
        font=dict(
            size=13,
            color="#2563eb"  # blue to signal interactivity
        )
    ),

    annotations=[
        dict(
            text="Click legend items to toggle series visibility",
            x=0.5,
            y=1.10,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(
                size=12,
                color="#2563eb"
            )
        )
    ]
)


    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)

    # Include Plotly via CDN to keep file size smaller
    html = fig.to_html(full_html=True, include_plotlyjs="cdn")
    OUT_HTML.write_text(html, encoding="utf-8")

    print(f"Wrote: {OUT_HTML}")
    print(f"Rows plotted: {len(merged)}")
    print(f"Date range: {merged['Date'].min().date()} to {merged['Date'].max().date()}")


if __name__ == "__main__":
    main()
