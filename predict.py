import argparse
import pandas as pd
import joblib
import json

from data.fetch_stock_data import get_stock_data
from utils.features import add_features, FEATURE_COLUMNS_DEFAULT

# Load volatility/regression model (optional)
vol_model = None
try:
    vol_model = joblib.load("models/general_short_term_model.pkl")
except Exception:
    pass

# Load direction model (optional)
dir_model = None
try:
    dir_model = joblib.load("models/direction_model.pkl")
except Exception:
    pass

# Load encoder
encoder = joblib.load("models/ticker_encoder.pkl")

# Load feature columns if available
try:
    with open("models/feature_columns.json", "r") as f:
        FEATURE_COLUMNS = json.load(f)
except Exception:
    FEATURE_COLUMNS = list(FEATURE_COLUMNS_DEFAULT) + ['Encoded_Ticker']


def predict_summary(ticker: str, period: str = '2y'):
    if vol_model is None and dir_model is None:
        raise RuntimeError("No trained models found. Please train first.")

    if ticker not in encoder.classes_:
        supported = ", ".join(sorted(encoder.classes_))
        raise ValueError(f"Ticker '{ticker}' was not seen during training. Supported tickers: {supported}")

    # Fetch historical data and generate features
    df = get_stock_data(ticker, period=period)
    # Use default target_type='volatility' for consistent features
    df = add_features(df)

    latest = df.iloc[-1:].copy()
    latest['Encoded_Ticker'] = encoder.transform([ticker])[0]
    X_latest = latest[FEATURE_COLUMNS]

    # Volatility prediction (if available)
    vol_pred = None
    if vol_model is not None:
        vol_pred = float(vol_model.predict(X_latest)[0])

    # Direction prediction (if available)
    dir_pred = None
    dir_prob = None
    if dir_model is not None:
        dir_prob = float(dir_model.predict_proba(X_latest)[0, 1])
        dir_pred = "up" if dir_prob >= 0.5 else "down"

    # Output
    print("")
    print(f"Ticker: {ticker}")
    if vol_pred is not None:
        print(f"Predicted 3-day volatility: {vol_pred*100:.2f}%")
    if dir_prob is not None:
        print(f"Direction probability: P(up)={dir_prob*100:.1f}%, P(down)={(1-dir_prob)*100:.1f}% (bias: {dir_pred})")
    if vol_pred is not None and dir_prob is not None:
        print(f"Summary: {ticker} bias {dir_pred} with confidence {dir_prob*100:.1f}% and expected choppiness ~{vol_pred*100:.2f}% over 3 days.")
    print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict 3-day volatility and direction (if models available).")
    parser.add_argument("--ticker", required=True, help="Stock ticker symbol (e.g., TSLA)")
    args = parser.parse_args()
    predict_summary(args.ticker.upper())
