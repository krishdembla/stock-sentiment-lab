import yfinance as yf
import pandas as pd
from pathlib import Path


def get_stock_data(ticker: str, period: str = "4y", interval: str = "1d", start: str | None = None, end: str | None = None, use_cache: bool = True) -> pd.DataFrame:
    """
    grabs historical stock data (open, high, low, close, volume) for a ticker
    
    checks cache first if use_cache=True, otherwise downloads from yahoo finance.
    if you pass a start date, it uses that instead of the period.
    
    returns a dataframe with columns: Date, Open, High, Low, Close, Volume, Adj Close, Ticker
    """
    # try loading from cache first if it's there
    if use_cache:
        cache_path = Path(__file__).parent.parent / "cache" / f"{ticker}.csv"
        if cache_path.exists():
            try:
                df = pd.read_csv(cache_path, parse_dates=['Date'])
                
                # filter by date range if we need a specific window
                if start is not None:
                    df = df[df['Date'] >= pd.to_datetime(start)]
                if end is not None:
                    df = df[df['Date'] <= pd.to_datetime(end)]
                
                # make sure ticker column is there
                if 'Ticker' not in df.columns:
                    df['Ticker'] = ticker
                
                return df
            except Exception as e:
                print(f"Warning: Cache read failed for {ticker}, downloading fresh data...")
    
    # download from yahoo finance if cache misses or is disabled
    if start is not None:
        df = yf.download(ticker, start=start, end=end, interval=interval, progress=False, auto_adjust=False)
    else:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)

    # clean up missing data
    df.dropna(inplace=True)
    df["Ticker"] = ticker

    # make sure we have a date column (some data comes with date as index)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df.reset_index(inplace=True)
    if "index" in df.columns and "Date" not in df.columns:
        df.rename(columns={"index": "Date"}, inplace=True)

    return df