#!/usr/bin/env python3
"""
Test different target formulations to find the most predictable one.
"""

import pandas as pd
import numpy as np
from data.fetch_stock_data import get_stock_data
from utils.features import add_features
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import matplotlib.pyplot as plt

def test_target_formulations():
    """Test different target formulations and compare their predictability."""
    
    # Test tickers
    tickers = ['AAPL', 'MSFT', 'TSLA', 'NVDA', 'GOOG']
    
    target_types = ['return', 'direction', 'volatility', 'momentum']
    results = {}
    
    print("Testing Target Formulations")
    print("=" * 50)
    
    for target_type in target_types:
        print(f"\nTesting: {target_type.upper()}")
        print("-" * 30)
        
        all_data = []
        
        for ticker in tickers:
            try:
                df = get_stock_data(ticker, period='2y')
                df = add_features(df, target_type=target_type)
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
        print(f"  Target stats: mean={target_stats['mean']:.6f}, std={target_stats['std']:.6f}")
        
        # Prepare features (excluding target and ticker)
        feature_cols = ['Return', 'Volatility', 'Momentum', 'MA10', 'MA50', 'RSI', 'MACD', 'OBV']
        X = combined_df[feature_cols]
        y = combined_df['Target']
        
        # Remove rows with NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        if len(X) == 0:
            print("  No valid data after cleaning")
            continue
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        if target_type == 'direction':
            # Classification problem
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            baseline = max(y_test.mean(), 1 - y_test.mean())  # Majority class baseline
            print(f"  Accuracy: {score:.4f}")
            print(f"  Baseline: {baseline:.4f}")
            print(f"  Improvement: {((score - baseline) / baseline * 100):.2f}%")
            
            # Feature importance
            importances = model.feature_importances_
            feature_importance = dict(zip(feature_cols, importances))
            print(f"  Top features: {sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]}")
            
        else:
            # Regression problem
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            baseline_rmse = y_test.std()
            r2 = model.score(X_test, y_test)
            
            print(f"  RMSE: {rmse:.6f}")
            print(f"  Baseline RMSE: {baseline_rmse:.6f}")
            print(f"  R²: {r2:.4f}")
            print(f"  Improvement: {((baseline_rmse - rmse) / baseline_rmse * 100):.2f}%")
            
            # Feature importance
            importances = model.feature_importances_
            feature_importance = dict(zip(feature_cols, importances))
            print(f"  Top features: {sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]}")
        
        results[target_type] = {
            'score': score if target_type == 'direction' else r2,
            'baseline': baseline if target_type == 'direction' else baseline_rmse,
            'improvement': ((score - baseline) / baseline * 100) if target_type == 'direction' else ((baseline_rmse - rmse) / baseline_rmse * 100),
            'feature_importance': feature_importance
        }
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    for target_type, result in results.items():
        print(f"{target_type.upper()}: {result['improvement']:.2f}% improvement over baseline")
    
    # Find best target
    best_target = max(results.items(), key=lambda x: x[1]['improvement'])
    print(f"\nBEST TARGET: {best_target[0].upper()} ({best_target[1]['improvement']:.2f}% improvement)")
    
    return results, best_target[0]

if __name__ == "__main__":
    results, best_target = test_target_formulations() 