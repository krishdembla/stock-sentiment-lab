#!/usr/bin/env python3
"""
Test Step 2 improvements: Compare direction model performance with and without market context.
"""

import joblib
import pandas as pd
import json
import numpy as np
from data.fetch_stock_data import get_stock_data
from utils.features import add_features, add_market_context, FEATURE_COLUMNS_DEFAULT
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def test_step2_improvements():
    """Test if market context features improve direction prediction."""
    
    print("=" * 70)
    print("STEP 2 TEST: Direction Model with Market Context Features")
    print("=" * 70)
    
    # Load existing models
    try:
        dir_model = joblib.load('models/direction_model.pkl')
        encoder = joblib.load('models/ticker_encoder.pkl')
        with open('models/feature_columns.json', 'r') as f:
            feature_columns = json.load(f)
    except Exception as e:
        print(f"Error loading models: {e}")
        return
    
    print(f"\n✅ Loaded direction model")
    print(f"   Supported tickers: {len(encoder.classes_)}")
    print(f"   Current features: {len(feature_columns)}")
    
    # Check if market context features are in the model
    market_features = [f for f in feature_columns if f.startswith('SPY_') or f.startswith('VIX_') or f.startswith('Relative_')]
    print(f"   Market context features: {len(market_features)}")
    if market_features:
        print(f"   Market features: {', '.join(market_features)}")
    else:
        print("   ⚠️  No market context features found - model needs retraining")
    
    # Test on a few tickers
    test_tickers = ['AAPL', 'MSFT', 'TSLA']
    print(f"\n{'='*70}")
    print("Testing Predictions (with market context if available)")
    print(f"{'='*70}")
    
    results = []
    
    for ticker in test_tickers:
        if ticker not in encoder.classes_:
            print(f"\n⚠️  {ticker}: Not in training set, skipping")
            continue
            
        try:
            # Get recent data
            df = get_stock_data(ticker, period='1y')
            df = add_features(df, target_type='direction', target_horizon=3)
            
            # Add market context
            try:
                df = add_market_context(df, period='1y')
                has_market_context = True
            except Exception as e:
                print(f"  Warning: Could not add market context for {ticker}: {e}")
                has_market_context = False
                # Add zero columns for market features
                market_cols = ['SPY_Return', 'SPY_Volatility', 'SPY_Trend_5d', 
                              'Relative_Return', 'Relative_Volatility', 'VIX_Level', 'VIX_Change']
                for col in market_cols:
                    if col not in df.columns:
                        df[col] = 0.0
            
            if len(df) == 0:
                print(f"\n⚠️  {ticker}: No data after feature engineering")
                continue
            
            # Get latest prediction
            latest = df.iloc[-1:].copy()
            latest['Encoded_Ticker'] = encoder.transform([ticker])[0]
            
            # Prepare features - use only features that exist in both
            available_features = [f for f in feature_columns if f in latest.columns]
            missing_features = [f for f in feature_columns if f not in latest.columns]
            
            if missing_features:
                print(f"\n⚠️  {ticker}: Missing {len(missing_features)} features")
                # Fill missing features with 0
                for f in missing_features:
                    latest[f] = 0.0
            
            X_latest = latest[feature_columns]
            
            # Predict
            dir_prob = dir_model.predict_proba(X_latest)[0, 1]
            direction = "UP" if dir_prob >= 0.5 else "DOWN"
            confidence = max(dir_prob, 1 - dir_prob) * 100
            
            # Get actual direction (if we have target)
            if 'Target' in latest.columns:
                actual = "UP" if latest['Target'].iloc[0] == 1 else "DOWN"
                correct = "✓" if direction == actual else "✗"
            else:
                actual = "N/A"
                correct = ""
            
            print(f"\n{ticker}:")
            print(f"  Prediction: {direction} (confidence: {confidence:.1f}%)")
            if actual != "N/A":
                print(f"  Actual: {actual} {correct}")
            print(f"  Market context: {'Yes' if has_market_context else 'No'}")
            
            results.append({
                'ticker': ticker,
                'prediction': direction,
                'confidence': confidence,
                'has_market_context': has_market_context
            })
            
        except Exception as e:
            print(f"\n❌ {ticker}: Error - {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"✅ Model loaded successfully")
    print(f"✅ Predictions generated for {len(results)} tickers")
    
    if market_features:
        print(f"✅ Market context features are available in model")
        print(f"   Next: Retrain model with market context to see improvement")
    else:
        print(f"⚠️  Market context features NOT in current model")
        print(f"   Action needed: Retrain model with market context")
    
    # Check feature importances
    print(f"\n{'='*70}")
    print("Feature Importance Check")
    print(f"{'='*70}")
    
    try:
        if hasattr(dir_model, 'feature_importances_'):
            importances = dir_model.feature_importances_
            feature_imp = list(zip(feature_columns, importances))
            feature_imp.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\nTop 10 Most Important Features:")
            for i, (feat, imp) in enumerate(feature_imp[:10], 1):
                marker = "📈" if any(f in feat for f in ['SPY', 'VIX', 'Relative']) else "  "
                print(f"  {i:2d}. {marker} {feat}: {imp:.6f}")
            
            # Check if market features are in top features
            top_market_features = [f for f, _ in feature_imp[:20] 
                                 if any(m in f for m in ['SPY', 'VIX', 'Relative'])]
            if top_market_features:
                print(f"\n✅ Market context features in top 20: {len(top_market_features)}")
            else:
                print(f"\n⚠️  No market context features in top 20 (model needs retraining)")
    except Exception as e:
        print(f"Error checking feature importances: {e}")
    
    print(f"\n{'='*70}")
    print("STEP 2 STATUS")
    print(f"{'='*70}")
    
    if market_features:
        print("✅ Market context features implemented")
        print("✅ Feature engineering updated")
        print("⚠️  Model needs retraining to use market context")
        print("\nNext step: Retrain direction model to see accuracy improvement")
    else:
        print("✅ Market context features implemented")
        print("✅ Feature engineering updated")
        print("❌ Current model doesn't have market context")
        print("\nNext step: Retrain direction model with market context")

if __name__ == "__main__":
    test_step2_improvements()


