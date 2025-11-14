#!/usr/bin/env python3
"""
Test different target horizons for direction prediction to find the most predictable one.
"""

import pandas as pd
import numpy as np
from data.fetch_stock_data import get_stock_data
from utils.features import add_features
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt

def test_direction_horizons():
    """Test 1-day, 3-day, 5-day, and 10-day direction predictions."""
    
    # Test tickers
    tickers = ['AAPL', 'MSFT', 'TSLA', 'NVDA', 'AMZN']
    
    horizons = [1, 3, 5, 10]
    results = {}
    
    print("Testing Direction Prediction Horizons")
    print("=" * 60)
    
    for horizon in horizons:
        print(f"\n{'='*60}")
        print(f"Testing {horizon}-day direction prediction")
        print(f"{'='*60}")
        
        all_data = []
        
        for ticker in tickers:
            try:
                df = get_stock_data(ticker, period='2y')
                df = add_features(df, target_type='direction', target_horizon=horizon)
                df['Ticker'] = ticker
                all_data.append(df)
                print(f"  {ticker}: {len(df)} rows")
            except Exception as e:
                print(f"  {ticker}: Error - {e}")
        
        if not all_data:
            continue
            
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Analyze target distribution
        target_stats = combined_df['Target'].describe()
        up_pct = (combined_df['Target'] == 1).mean() * 100
        print(f"  Target distribution: {up_pct:.1f}% up, {100-up_pct:.1f}% down")
        
        # Prepare features
        feature_cols = ['Return', 'Volatility', 'Momentum', 'MA10', 'MA50', 'RSI', 'MACD', 'OBV',
                       'High_Low_Range', 'Price_Range_5d', 'Volatility_10d', 'Volatility_20d', 'Volatility_Ratio',
                       'Trend_5d', 'Trend_10d', 'Trend_20d', 'MA_Ratio', 'Price_vs_MA10', 'Price_vs_MA50',
                       'Volume_Ratio', 'Volume_Price_Trend', 'MACD_Signal', 'MACD_Histogram',
                       'BB_Width', 'BB_Position', 'ATR_Ratio', 'Stoch_K', 'Stoch_D', 'Williams_R',
                       'Volatility_Regime', 'High_Volatility', 'Low_Volatility']
        
        X = combined_df[feature_cols]
        y = combined_df['Target']
        
        # Remove rows with NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        if len(X) == 0:
            print("  No valid data after cleaning")
            continue
        
        # Split data (time-based would be better, but for quick test use random)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        baseline = max(y_test.mean(), 1 - y_test.mean())
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
        
        print(f"  Accuracy: {acc:.4f} (Baseline: {baseline:.4f})")
        print(f"  Improvement: {((acc - baseline) / baseline * 100):.2f}%")
        print(f"  Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        
        # Feature importance
        importances = model.feature_importances_
        feature_importance = dict(zip(feature_cols, importances))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"  Top 5 features: {[f[0] for f in top_features]}")
        
        results[horizon] = {
            'accuracy': acc,
            'baseline': baseline,
            'improvement': ((acc - baseline) / baseline * 100),
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'up_pct': up_pct
        }
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY - Direction Prediction by Horizon")
    print("=" * 60)
    print(f"{'Horizon':<10} {'Accuracy':<12} {'Improvement':<15} {'F1 Score':<12}")
    print("-" * 60)
    
    for horizon, result in sorted(results.items()):
        print(f"{horizon} days{'':<6} {result['accuracy']:.4f}      {result['improvement']:>6.2f}%         {result['f1']:.4f}")
    
    # Find best horizon
    best_horizon = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n✅ BEST HORIZON: {best_horizon[0]} days (Accuracy: {best_horizon[1]['accuracy']:.4f})")
    
    return results, best_horizon[0]

if __name__ == "__main__":
    results, best_horizon = test_direction_horizons()

