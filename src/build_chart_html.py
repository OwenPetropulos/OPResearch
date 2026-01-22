from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

# MAIN STRATEGY OUTPUTS
INPUT = Path("data/portfolio_performance.csv")

# QQQ SERIES (Nasdaq 100 proxy)
QQQ_INPUT = Path("data/nasdaq100_weekly_growth_aligned.csv")

OUTPUT = Path("docs/projects/moving_average_v1/moving_average_backtest.html")

def normalize_date(s):
    # robust for timezone strings like "2016-01-08 00:00:00-05:00"
    return pd.to_datetime(s, utc=True, errors="coerce").dt.tz_convert(None)

def main():
    # --- Load strategy + benchmark ---
    df = pd.read_csv(INPUT)

    # Required columns in portfolio_performance.csv
    DATE_COL = "Date"
    STRAT_COL = "Strategy_Value"
    BENCH_COL = "Benchmark_Value"

    missing = [c for c in [DATE_COL, STRAT_COL, BENCH_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {INPUT}: {missing}. Found: {list(df.columns)}")

    df[DATE_COL] = normalize_date(df[DATE_COL])
    df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL)

    # --- Load QQQ weekly series ---
    qqq = pd.read_csv(QQQ_INPUT)

    # Try common column names
    qqq_date_candidates = [c for c in qqq.columns if c.strip().lower() in ("date", "datetime", "time")]
    if not qqq_date_candidates:
        raise ValueError(f"Could not find a Date column in {QQQ_INPUT}. Found: {list(qqq.columns)}")
    QQQ_DATE_COL = qqq_date_candidates[0]

    # QQQ value column: handle common names
    # If your file already has a Portfolio_Value column, that is fine (it represents QQQ growth portfolio).
    possible_val_cols = ["Portfolio_Value", "QQQ_Value", "Value", "Adj Close", "Adj_Close", "Close"]
    QQQ_VAL_COL = None
    for c in possible_val_cols:
        if c in qqq.columns:
            QQQ_VAL_COL = c
            break
    if QQQ_VAL_COL is None:
        raise ValueError(
            f"Could not find a QQQ value column in {QQQ_INPUT}. "
            f"Tried {possible_val_cols}. Found: {list(qqq.columns)}"
        )

    qqq[QQQ_DATE_COL] = normalize_date(qqq[QQQ_DATE_COL])
    qqq = qqq.dropna(subset=[QQQ_DATE_COL]).sort_values(QQQ_DATE_COL)
    qqq = qqq[[QQQ_DATE_COL, QQQ_VAL_COL]].rename(columns={QQQ_DATE_COL: DATE_COL, QQQ_VAL_COL: "Nasdaq100_QQQ"})

    # --- Align on dates (inner join so all lines share the same x-axis points) ---
    merged = df[[DATE_COL, STRAT_COL, BENCH_COL]].merge(qqq, on=DATE_COL, how="inner")

    # --- Plot ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merged[DATE_COL], y=merged[STRAT_COL], mode="lines", name="Strategy"))
    fig.add_trace(go.Scatter(x=merged[DATE_COL], y=merged[BENCH_COL], mode="lines", name="Benchmark (Equal-Weight)"))
    fig.add_trace(go.Scatter(x=merged[DATE_COL], y=merged["Nasdaq100_QQQ"], mode="lines", name="Nasdaq 100 (QQQ)"))

    fig.update_layout(
        title="Strategy vs Benchmark vs Nasdaq 100 (QQQ)",
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        template="plotly_white",
        height=650,
        margin=dict(l=40, r=20, t=90, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="left", x=0.5)
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(OUTPUT), include_plotlyjs="cdn", full_html=True)
    print(f"Wrote: {OUTPUT}")
    print(f"Points plotted: {len(merged)} (after date alignment)")

if __name__ == "__main__":
    main()
