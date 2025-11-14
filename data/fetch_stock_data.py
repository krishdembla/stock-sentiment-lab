import yfinance as yf
import pandas as pd


def get_stock_data(ticker: str, period: str = "4y", interval: str = "1d", start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Fetch historical OHLCV data for a ticker.

    If start is provided, it takes precedence over period.
    Ensures a 'Date' column exists for downstream processing.
    """
    if start is not None:
        df = yf.download(ticker, start=start, end=end, interval=interval, progress=False, auto_adjust=False)
    else:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)

    # Basic cleanup
    df.dropna(inplace=True)
    df["Ticker"] = ticker

    # Ensure a Date column for time-based splits
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df.reset_index(inplace=True)
    if "index" in df.columns and "Date" not in df.columns:
        df.rename(columns={"index": "Date"}, inplace=True)

    return df