import pandas as pd
import numpy as np
import joblib
import sys
import os

from xgboost import XGBRegressor
from utils.features import add_features
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data.fetch_stock_data import get_stock_data

sys.path.append("..")

TICKERS = [
    'AAPL', 'MSFT', 'TSLA', 'NVDA', 'AMZN',  # Tech
    'GOOG', 'META', 'CRM', 'ADBE', 'INTC',   # More Tech
    'JPM', 'V', 'MA',                        # Financials
    'JNJ', 'UNH',                            # Healthcare
    'WMT', 'PG', 'MCD',                      # Consumer
    'XOM', 'CVX'                             # Energy
]


def prepare_data():
    all_dfs = []
    for ticker in TICKERS:
        try:
            df = get_stock_data(ticker)
            df = add_features(df)
            all_dfs.append(df)
            print(f"Processed data for {ticker} - {len(df)} rows")
        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df



def train_model():
    df = prepare_data()
    # Encode the ticker symbol
    le = LabelEncoder()
    df['Encoded_Ticker'] = le.fit_transform(df['Ticker']).squeeze()
    
    features = ['Return', 'Volatility', 'Momentum', 'MA10', 'MA50', 'RSI', 'MACD', 'OBV', 'Encoded_Ticker']
    X = df[features]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = False)

    model = XGBRegressor(n_estimators = 150, learning_rate = 0.1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("MSE:", mean_squared_error(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("R2 Score:", r2_score(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/general_short_term_model.pkl")
    joblib.dump(le, "models/ticker_encoder.pkl")
    print("Model and encoder saved successfully.")

    return model



if __name__ == "__main__":
    train_model()



    


    

