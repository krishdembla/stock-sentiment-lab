import yfinance as yf
import pandas as pd
import os
import time

TICKERS = [
    'AAPL', 'MSFT', 'TSLA', 'NVDA', 'AMZN',
]

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def download_all_data(period: str = "4y", max_retries: int = 5):
    """
    Downloads historical data for all tickers and saves it to a local cache.
    Includes a retry mechanism with exponential backoff for rate limiting.
    """
    for ticker in TICKERS:
        cache_path = os.path.join(CACHE_DIR, f"{ticker}.csv")
        if os.path.exists(cache_path):
            print(f"Data for {ticker} already cached. Skipping.")
            continue

        print(f"Downloading data for {ticker}...")
        for attempt in range(max_retries):
            try:
                df = yf.download(ticker, period=period, progress=False, auto_adjust=False)
                if df.empty:
                    # yfinance returns an empty dataframe for failed downloads
                    raise ValueError("No data found")
                
                # Ensure data is numeric
                numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                df.dropna(subset=numeric_cols, inplace=True)

                df.to_csv(cache_path)
                print(f"Saved data for {ticker} to {cache_path}")
                break  # Success, exit retry loop
            except Exception as e:
                if "Too Many Requests" in str(e) or "No data found" in str(e):
                    if attempt < max_retries - 1:
                        delay = 2 ** (attempt + 1)
                        print(f"Rate limited. Retrying in {delay} seconds...")
                        time.sleep(delay)
                    else:
                        print(f"Failed to download {ticker} after {max_retries} attempts.")
                else:
                    print(f"An unexpected error occurred for {ticker}: {e}")
                    break # Don't retry on unexpected errors
        
        time.sleep(2) # Delay between different tickers

if __name__ == "__main__":
    download_all_data()
