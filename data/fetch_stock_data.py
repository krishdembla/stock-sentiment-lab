import yfinance as yf
import pandas as pd

def get_stock_data(ticker: str, start="2019-01-01", interval="1d"):
    df = yf.download(ticker, start=start, interval=interval)
    df.dropna(inplace=True)
    df['Ticker'] = ticker
    return df