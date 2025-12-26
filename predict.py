#!/usr/bin/env python3
"""
position sizing calculator based on predicted volatility

uses xgboost to forecast daily volatility and calculates optimal position sizes
for risk management. each position is sized so it risks the same dollar amount.

basic usage:
    python predict.py --tickers AAPL MSFT TSLA
    python predict.py --tickers AAPL MSFT --account 100000 --risk 0.02
    python predict.py --tickers AAPL --output positions.csv
"""

import argparse
import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta

# need to add project root so imports work properly
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data.fetch_stock_data import get_stock_data
from utils.features import add_features, add_market_context, add_realized_volatility_features


def load_model(model_path):
    """loads the trained xgboost model from disk"""
    try:
        import joblib
        model = joblib.load(str(model_path))
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        sys.exit(1)


def load_feature_columns(features_path):
    """gets the list of features the model expects"""
    try:
        with open(features_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading feature columns from {features_path}: {e}")
        sys.exit(1)


def load_ticker_encoder(encoder_path):
    """loads the encoder that converts ticker symbols to numbers"""
    try:
        import joblib
        encoder = joblib.load(str(encoder_path))
        return encoder
    except Exception as e:
        print(f"Error loading ticker encoder from {encoder_path}: {e}")
        sys.exit(1)


def prepare_features(ticker, start_date, end_date, feature_columns, ticker_encoder, use_cache=True):
    """
    grabs stock data and calculates all the technical indicators the model needs
    
    use_cache=True loads from cache, False fetches live from yahoo finance
    
    returns a dataframe ready for prediction, or None if something goes wrong
    """
    try:
        # get historical data from cache or download from yahoo
        df = get_stock_data(ticker, start=start_date, end=end_date, use_cache=use_cache)
        
        if df is None or len(df) == 0:
            print(f"Warning: No data available for {ticker}")
            return None
        
        # calculate all the features (moving averages, rsi, bollinger bands, etc)
        df = add_features(df)
        
        # add market context like spy and vix to help the model understand broader trends
        df = add_market_context(df)
        
        # add volatility measurements over different time windows
        df = add_realized_volatility_features(df)
        
        # drop rows where we couldn't calculate all the features yet
        df = df.dropna()
        
        if len(df) == 0:
            print(f"Warning: Insufficient data for {ticker} after feature engineering")
            return None
        
        # grab the most recent data point for prediction
        latest = df.iloc[-1:].copy()
        
        # the model needs the ticker as a number so we encode it
        try:
            encoded_ticker = ticker_encoder.transform([ticker])[0]
            latest['Encoded_Ticker'] = encoded_ticker
        except Exception as e:
            print(f"Warning: Unknown ticker {ticker}, not in training set")
            return None
        
        # check if any features are missing and fill them with zeros
        missing_features = set(feature_columns) - set(latest.columns)
        if missing_features:
            # this handles cases where market context or sentiment data isn't available
            for feat in missing_features:
                latest[feat] = 0.0
        
        # select only the features the model expects, in the right order
        X = latest[feature_columns]
        
        # quirk: the trained model expects feature names with trailing spaces
        X.columns = [f"{col} " for col in X.columns]
        
        return X
        
    except Exception as e:
        print(f"Error preparing features for {ticker}: {e}")
        return None


def predict_volatility(model, X):
    """
    runs the xgboost model to predict daily volatility
    
    returns predicted volatility as a decimal (0.02 = 2% daily movement)
    """
    try:
        # model loaded with joblib can predict directly
        prediction = model.predict(X)[0]
        return float(prediction)
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None


def calculate_position_size(predicted_vol, account_value, risk_percent):
    """
    figures out how much to invest based on volatility - higher vol means smaller position
    
    the math: position_size = (account * risk%) / (2 * volatility)
    
    why 2x? we're using 2 standard deviations as a safety buffer. if a stock normally
    moves 1% per day, there's a chance it could move 2% in a bad day.
    
    returns position size in dollars, percentage, and other risk metrics
    """
    risk_amount = account_value * risk_percent
    potential_loss_per_dollar = 2 * predicted_vol  # 2 standard deviations
    
    position_size = risk_amount / potential_loss_per_dollar
    
    # calculate as percentage of account
    position_percent = (position_size / account_value) * 100
    
    # stop loss distance (2 sigma)
    stop_loss_percent = potential_loss_per_dollar * 100
    
    return {
        'position_size': position_size,
        'position_percent': position_percent,
        'risk_amount': risk_amount,
        'stop_loss_percent': stop_loss_percent,
        'predicted_vol': predicted_vol * 100  # convert to percentage
    }


def get_current_price(ticker, use_cache=True):
    """fetches the latest price for a stock"""
    try:
        # Get broader date range to ensure we get cached data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        df = get_stock_data(ticker, start=start_date, end=end_date, use_cache=use_cache)
        if df is not None and len(df) > 0:
            return float(df['Close'].iloc[-1])
        return None
    except:
        return None


def format_output(results, account_value, risk_percent):
    """formats prediction results for console display"""
    print("\n" + "="*80)
    print("VOLATILITY-BASED POSITION SIZING CALCULATOR")
    print("="*80)
    print(f"\nAccount Value: ${account_value:,.0f}")
    print(f"Risk Per Position: {risk_percent*100:.1f}% (${account_value * risk_percent:,.0f})")
    print("\n" + "-"*80)
    
    for result in results:
        ticker = result['ticker']
        current_price = result.get('current_price')
        pos = result['position_sizing']
        
        print(f"\n{ticker}")
        print(f"   Current Price: ${current_price:.2f}" if current_price else "   Current Price: N/A")
        print(f"   Predicted Volatility: {pos['predicted_vol']:.2f}%")
        print(f"   → Position Size: ${pos['position_size']:,.0f} ({pos['position_percent']:.1f}% of account)")
        
        if current_price:
            shares = int(pos['position_size'] / current_price)
            stop_loss_price = current_price * (1 - pos['stop_loss_percent'] / 100)
            print(f"   → Shares: {shares:,}")
            print(f"   → Stop Loss: ${stop_loss_price:.2f} (-{pos['stop_loss_percent']:.2f}%)")
        
        print(f"   → Risk on Position: ${pos['risk_amount']:,.0f}")
    
    # Portfolio summary
    print("\n" + "-"*80)
    total_position_value = sum(r['position_sizing']['position_size'] for r in results)
    total_deployed = (total_position_value / account_value) * 100
    
    print(f"\n📈 PORTFOLIO SUMMARY")
    print(f"   Total Capital Deployed: ${total_position_value:,.0f} ({total_deployed:.1f}%)")
    print(f"   Remaining Cash: ${account_value - total_position_value:,.0f} ({100 - total_deployed:.1f}%)")
    print(f"   Total Positions: {len(results)}")
    
    avg_vol = np.mean([r['position_sizing']['predicted_vol'] for r in results])
    print(f"   Average Predicted Volatility: {avg_vol:.2f}%")
    
    print("\n" + "="*80)


def save_to_csv(results, output_path, account_value, risk_percent):
    """Save results to CSV file"""
    rows = []
    for result in results:
        ticker = result['ticker']
        current_price = result.get('current_price', 0)
        pos = result['position_sizing']
        
        shares = int(pos['position_size'] / current_price) if current_price else 0
        stop_loss_price = current_price * (1 - pos['stop_loss_percent'] / 100) if current_price else 0
        
        rows.append({
            'Ticker': ticker,
            'Current_Price': current_price,
            'Predicted_Volatility_Pct': pos['predicted_vol'],
            'Position_Size_Dollars': pos['position_size'],
            'Position_Percent_of_Account': pos['position_percent'],
            'Shares': shares,
            'Stop_Loss_Price': stop_loss_price,
            'Stop_Loss_Percent': pos['stop_loss_percent'],
            'Risk_Amount': pos['risk_amount']
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Predict volatility and calculate position sizes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py --tickers AAPL MSFT TSLA
  python predict.py --tickers AAPL MSFT --account 100000 --risk 0.02
  python predict.py --tickers NVDA META GOOG --output positions.csv
        """
    )
    
    parser.add_argument(
        '--tickers',
        nargs='+',
        required=True,
        help='Stock tickers to analyze (space-separated)'
    )
    
    parser.add_argument(
        '--account',
        type=float,
        default=100000,
        help='Account value in dollars (default: 100000)'
    )
    
    parser.add_argument(
        '--risk',
        type=float,
        default=0.01,
        help='Risk per position as decimal (default: 0.01 = 1%%)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output CSV file path (optional)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='models/general_short_term_model.pkl',
        help='Path to trained model (default: models/general_short_term_model.pkl)'
    )
    
    parser.add_argument(
        '--features',
        type=str,
        default='models/feature_columns.json',
        help='Path to feature columns JSON (default: models/feature_columns.json)'
    )
    
    parser.add_argument(
        '--encoder',
        type=str,
        default='models/ticker_encoder.pkl',
        help='Path to ticker encoder (default: models/ticker_encoder.pkl)'
    )
    
    parser.add_argument(
        '--live',
        action='store_true',
        help='Fetch live data from Yahoo Finance (bypasses cache, may hit rate limits)'
    )
    
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable cache (same as --live)'
    )
    
    args = parser.parse_args()
    
    # Handle --no-cache as alias for --live
    if args.no_cache:
        args.live = True
    
    # Validate inputs
    if args.risk <= 0 or args.risk > 0.1:
        print("Error: Risk must be between 0 and 0.1 (0% to 10%)")
        sys.exit(1)
    
    if args.account <= 0:
        print("Error: Account value must be positive")
        sys.exit(1)
    
    print("\nLoading model and configuration...")
    
    # Load model and feature columns
    model_path = project_root / args.model
    features_path = project_root / args.features
    encoder_path = project_root / args.encoder
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("   Please train the model first: python models/train_general_model.py")
        sys.exit(1)
    
    if not features_path.exists():
        print(f"❌ Feature columns not found at {features_path}")
        sys.exit(1)
    
    if not encoder_path.exists():
        print(f"❌ Ticker encoder not found at {encoder_path}")
        sys.exit(1)
    
    model = load_model(model_path)
    feature_columns = load_feature_columns(features_path)
    ticker_encoder = load_ticker_encoder(encoder_path)
    
    print(f"Model loaded: {len(feature_columns)} features")
    
    # Display cache mode
    cache_mode = "LIVE MODE (no cache)" if args.live else "CACHE MODE (fast)"
    print(f"Data Source: {cache_mode}")
    
    if args.live:
        print("Warning: Yahoo Finance may rate limit. Use cache for bulk analysis.")
    
    print(f"\nFetching data for {len(args.tickers)} ticker(s)...\n")
    
    # Process each ticker
    results = []
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')  # 1 year
    
    for ticker in args.tickers:
        print(f"Processing {ticker}...", end=" ")
        
        # Prepare features (use_cache = not args.live)
        X = prepare_features(ticker, start_date, end_date, feature_columns, ticker_encoder, use_cache=not args.live)
        
        if X is None:
            print(f"❌ Skipped")
            continue
        
        # Predict volatility
        predicted_vol = predict_volatility(model, X)
        
        if predicted_vol is None:
            print(f"❌ Prediction failed")
            continue
        
        # Get current price
        current_price = get_current_price(ticker, use_cache=not args.live)
        
        # Calculate position sizing
        position_sizing = calculate_position_size(predicted_vol, args.account, args.risk)
        
        results.append({
            'ticker': ticker,
            'current_price': current_price,
            'position_sizing': position_sizing
        })
        
        print(f"Predicted volatility: {position_sizing['predicted_vol']:.2f}%")
    
    if not results:
        print("\nError: No valid predictions generated. Check ticker symbols and data availability.")
        sys.exit(1)
    
    # Output results
    format_output(results, args.account, args.risk)
    
    # Save to CSV if requested
    if args.output:
        save_to_csv(results, args.output, args.account, args.risk)
    
    print("\n✨ Done!\n")


if __name__ == "__main__":
    main()
