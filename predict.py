import argparse
import pandas as pd
import joblib

from data.fetch_stock_data import get_stock_data
from utils.features import add_features

#loading the trained model and ticker numbers
model = joblib.load("models/general_short_term_model.pkl")
encoder = joblib.load("models/ticker_encoder.pkl")


def predict_return(ticker):
    #fetching historical data and adding features
    df = get_stock_data(ticker)
    df = add_features(df)
    #fetching latest row for prediction
    latest = df.iloc[-1:].copy()
    latest['Encoded_Ticker'] = encoder.transform([ticker])[0]

    #selecting features for prediction
    features = ['Return', 'Volatility', 'Momentum', 'MA10', 'MA50', 'RSI', 'MACD', 'OBV', 'Encoded_Ticker']
    X_latest = latest[features]

    #making the prediction
    predicted_return = model.predict(X_latest)[0]
    percent = round(predicted_return * 100, 2)

    direction = "up" if percent > 0 else "down"
    print(f"\n{ticker} is predicted to go {direction} by {abs(percent):.2f}% over the next 3 days.\n")
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict stock return for the next 3 days.")
    parser.add_argument("--ticker", required=True, help = "Stock ticker symbol (e.g., TSLA)")
    args = parser.parse_args()
    predict_return(args.ticker.upper())
