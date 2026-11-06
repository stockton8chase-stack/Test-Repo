#!/usr/bin/env python3
"""
Record historical stock prices and dividend-adjusted % change.

Default ticker is AGNC (AGNC Investment Corp., a monthly dividend payer),
but any ticker works.

For each trading day it records:
  close              - closing price (as traded, split-adjusted only)
  dividend           - cash dividend going ex on that day (0 if none)
  price_pct_change   - % change in closing price only (ignores dividends)
  total_pct_change   - % change adjusted for dividends (total return):
                         (close_t + dividend_t) / close_{t-1} - 1
  adj_close          - dividend-adjusted price series, scaled so the most
                       recent value equals the latest close. % changes in
                       this column equal total_pct_change.
  cumulative_total_return_pct - total return since the first recorded day,
                       assuming dividends are reinvested

Usage:
  pip install yfinance pandas
  python stock_history.py                         # AGNC, last 5 years
  python stock_history.py --start 2015-01-01      # from a given date
  python stock_history.py --period max            # full history
  python stock_history.py --interval 1mo          # monthly rows
  python stock_history.py --ticker AGNC --out agnc.csv

Re-running with the same --out file merges new rows into the existing
file, so it can be run periodically to keep the record up to date.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def fetch_history(ticker: str, start: str | None, end: str | None,
                  period: str, interval: str) -> pd.DataFrame:
    """Download raw (unadjusted-for-dividends) prices plus dividend events."""
    import yfinance as yf

    kwargs = dict(interval=interval, auto_adjust=False, actions=True)
    if start or end:
        kwargs.update(start=start, end=end)
    else:
        kwargs.update(period=period)

    raw = yf.Ticker(ticker).history(**kwargs)
    if raw.empty:
        raise RuntimeError(f"No data returned for {ticker!r}. "
                           "Check the ticker symbol and date range.")
    return raw


def compute_returns(raw: pd.DataFrame) -> pd.DataFrame:
    """Build the output table from a DataFrame with 'Close' and 'Dividends'."""
    df = pd.DataFrame(index=raw.index)
    df["close"] = raw["Close"].astype(float)
    df["dividend"] = (raw["Dividends"].astype(float).fillna(0.0)
                      if "Dividends" in raw else 0.0)

    prev_close = df["close"].shift(1)
    df["price_pct_change"] = (df["close"] / prev_close - 1) * 100
    df["total_pct_change"] = ((df["close"] + df["dividend"]) / prev_close - 1) * 100

    # Chain total-return factors into an adjusted price series, then rescale
    # so the latest adjusted close equals the latest actual close.
    growth = (1 + df["total_pct_change"].fillna(0) / 100).cumprod()
    df["adj_close"] = growth * (df["close"].iloc[-1] / growth.iloc[-1])
    df["cumulative_total_return_pct"] = (growth / growth.iloc[0] - 1) * 100

    # Clean index: plain dates, no timezone
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "date"
    return df.round({"close": 4, "dividend": 4, "price_pct_change": 4,
                     "total_pct_change": 4, "adj_close": 4,
                     "cumulative_total_return_pct": 4})


def save(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Write to CSV, merging with an existing file if present.

    Raw closes and dividends are merged (newest download wins on overlapping
    dates) and all derived columns are recomputed over the full history, so
    the adjusted series stays consistent.
    """
    if out.exists():
        old = pd.read_csv(out, index_col="date", parse_dates=["date"])
        merged = pd.concat([old[["close", "dividend"]], df[["close", "dividend"]]])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        df = compute_returns(merged.rename(columns={"close": "Close",
                                                    "dividend": "Dividends"}))
    df.to_csv(out)
    return df


def summarize(ticker: str, df: pd.DataFrame) -> None:
    first, last = df.iloc[0], df.iloc[-1]
    price_ret = (last["close"] / first["close"] - 1) * 100
    divs = df[df["dividend"] > 0]
    print(f"\n{ticker}: {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  Start close:            ${first['close']:.2f}")
    print(f"  End close:              ${last['close']:.2f}")
    print(f"  Price-only return:      {price_ret:+.2f}%")
    print(f"  Total return (w/ divs): {last['cumulative_total_return_pct']:+.2f}%")
    print(f"  Dividends paid:         {len(divs)} totaling ${divs['dividend'].sum():.2f}/share")
    print("\nMost recent rows:")
    with pd.option_context("display.width", 120, "display.max_columns", None):
        print(df.tail(10).to_string())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ticker", default="AGNC")
    p.add_argument("--start", help="Start date YYYY-MM-DD (overrides --period)")
    p.add_argument("--end", help="End date YYYY-MM-DD (exclusive)")
    p.add_argument("--period", default="5y",
                   help="Lookback if no --start/--end: 1mo,6mo,1y,5y,10y,max (default 5y)")
    p.add_argument("--interval", default="1d",
                   help="Bar size: 1d, 1wk, 1mo (default 1d)")
    p.add_argument("--out", type=Path,
                   help="CSV output path (default <TICKER>_history.csv)")
    args = p.parse_args(argv)

    ticker = args.ticker.upper()
    out = args.out or Path(f"{ticker}_history.csv")

    try:
        raw = fetch_history(ticker, args.start, args.end, args.period, args.interval)
    except Exception as e:  # network errors, bad ticker, etc.
        print(f"Error fetching data: {e}", file=sys.stderr)
        return 1

    df = save(compute_returns(raw), out)
    summarize(ticker, df)
    print(f"\nSaved to {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
