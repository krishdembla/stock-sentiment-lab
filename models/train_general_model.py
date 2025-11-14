import sys
import os

# Add the project root to the Python path
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
import joblib
import json
import random
import time
from datetime import datetime

import pandas as pd
import numpy as np
import joblib
import json
import random
import time
from datetime import datetime

from xgboost import XGBRegressor
from utils.features import add_features, add_market_context, FEATURE_COLUMNS_DEFAULT
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data.fetch_stock_data import get_stock_data
import matplotlib.pyplot as plt

TICKERS = [
    # Tech
    'AAPL', 'MSFT', 'TSLA', 'NVDA', 'AMZN', 'GOOG', 'META', 'CRM', 'ADBE', 'INTC',
    # Financials
    'JPM', 'V', 'MA', 'BAC', 'WFC',
    # Healthcare
    'JNJ', 'UNH', 'PFE', 'MRK',
    # Consumer
    'WMT', 'PG', 'MCD', 'COST', 'HD', 'NKE',
    # Energy/Materials
    'XOM', 'CVX', 'COP'
]

CACHE_DIR = os.path.join(project_root, "data", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def prepare_data(target_type: str = 'volatility', period: str = '3y', use_cache: bool = True) -> pd.DataFrame:
    """Prepare data for training. Uses cache if available, otherwise fetches from yfinance."""
    all_dfs: list[pd.DataFrame] = []
    
    for ticker in TICKERS:
        cache_path = os.path.join(CACHE_DIR, f"{ticker}.csv")
        
        # Try to load from cache first
        if use_cache and os.path.exists(cache_path):
            try:
                df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                df.dropna(inplace=True)
                df["Ticker"] = ticker
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index, errors="coerce")
                df.reset_index(inplace=True)
                if "index" in df.columns and "Date" not in df.columns:
                    df.rename(columns={"index": "Date"}, inplace=True)
                df = add_features(df, target_type=target_type)
                all_dfs.append(df)
                print(f"Processed {ticker} from cache - {len(df)} rows")
                continue
            except Exception as e:
                print(f"Error loading {ticker} from cache: {e}. Fetching fresh data...")
        
        # Fetch fresh data if cache doesn't exist or failed
        try:
            print(f"Fetching data for {ticker}...")
            df = get_stock_data(ticker, period=period)
            df = add_features(df, target_type=target_type)
            all_dfs.append(df)
            
            # Save to cache for future use
            if use_cache:
                df_to_cache = df.copy()
                if 'Date' in df_to_cache.columns:
                    df_to_cache.set_index('Date', inplace=True)
                df_to_cache.to_csv(cache_path)
                print(f"  Cached {ticker} data")
            
            print(f"Processed {ticker} - {len(df)} rows")
        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    if not all_dfs:
        raise ValueError("No data to concatenate. Check ticker symbols and network connection.")

    combined_df = pd.concat(all_dfs, ignore_index=True)
    # Ensure global chronological order to avoid leakage
    combined_df = combined_df.sort_values(['Date', 'Ticker']).reset_index(drop=True)
    
    # Add market context features (SPY, VIX) - do this after combining all tickers
    # to avoid redundant API calls
    print("Adding market context features (SPY, VIX)...")
    try:
        combined_df = add_market_context(combined_df, period=period)
        print("Market context features added successfully")
    except Exception as e:
        print(f"Warning: Could not add market context features: {e}")
        # Add zero columns so feature list is consistent
        market_cols = ['SPY_Return', 'SPY_Volatility', 'SPY_Trend_5d', 'Relative_Return', 
                      'Relative_Volatility', 'VIX_Level', 'VIX_Change']
        for col in market_cols:
            if col not in combined_df.columns:
                combined_df[col] = 0.0
    
    return combined_df


def _compute_ticker_balanced_weights(tickers: pd.Series) -> np.ndarray:
    counts = tickers.value_counts().to_dict()
    num_unique = len(counts)
    total_rows = len(tickers)
    # Make total weight per ticker equal
    per_ticker_total = total_rows / num_unique
    weights = tickers.map({t: per_ticker_total / c for t, c in counts.items()}).astype(float).to_numpy()
    return weights


def _time_based_splits(dates: pd.Series, train_frac: float = 0.7, val_frac: float = 0.15):
    assert 0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1
    unique_dates = np.sort(dates.unique())
    n_dates = len(unique_dates)
    train_end = unique_dates[int(n_dates * train_frac)]
    val_end = unique_dates[int(n_dates * (train_frac + val_frac))]

    train_idx = dates <= train_end
    val_idx = (dates > train_end) & (dates <= val_end)
    test_idx = dates > val_end
    return train_idx, val_idx, test_idx


def _evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred))
    }


def _random_param() -> dict:
    return {
        'n_estimators': random.choice([200, 400, 600, 800]),
        'learning_rate': random.choice([0.02, 0.05, 0.1]),
        'max_depth': random.choice([3, 4, 5, 6, 8]),
        'min_child_weight': random.choice([1, 3, 5, 7]),
        'subsample': random.choice([0.7, 0.8, 0.9, 1.0]),
        'colsample_bytree': random.choice([0.6, 0.8, 1.0]),
        'reg_alpha': random.choice([0.0, 0.001, 0.01, 0.1]),
        'reg_lambda': random.choice([0.5, 1.0, 1.5, 2.0]),
        'gamma': random.choice([0.0, 0.01, 0.1])
    }


def train_model(num_param_samples: int = 50, early_stopping_rounds: int = 50, target_type: str = 'volatility', period: str = '3y'):
    df = prepare_data(target_type=target_type, period=period)

    # Encode the ticker symbol
    le = LabelEncoder()
    df['Encoded_Ticker'] = le.fit_transform(df['Ticker']).squeeze()

    feature_columns = list(FEATURE_COLUMNS_DEFAULT) + ['Encoded_Ticker']
    X = df[feature_columns]
    y = df['Target']

    # Time-based split across global timeline
    train_idx, val_idx, test_idx = _time_based_splits(df['Date'])

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Sample weights to avoid per-ticker dominance
    weights = _compute_ticker_balanced_weights(df['Ticker'])
    w_train = weights[train_idx]
    w_val = weights[val_idx]

    os.makedirs("models", exist_ok=True)

    if target_type == 'direction':
        # Classification branch
        from xgboost import XGBClassifier
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

        # Handle class imbalance via scale_pos_weight
        pos_ratio = float((y_train == 1).mean())
        neg_ratio = 1.0 - pos_ratio
        scale_pos_weight = (neg_ratio / max(pos_ratio, 1e-6)) if pos_ratio > 0 else 1.0

        best_score = -1.0
        best_params = None
        best_model = None

        for i in range(num_param_samples):
            params = _random_param()
            model = XGBClassifier(
                objective='binary:logistic',
                tree_method='hist',
                n_estimators=params['n_estimators'],
                learning_rate=params['learning_rate'],
                max_depth=params['max_depth'],
                min_child_weight=params['min_child_weight'],
                subsample=params['subsample'],
                colsample_bytree=params['colsample_bytree'],
                reg_alpha=params['reg_alpha'],
                reg_lambda=params['reg_lambda'],
                gamma=params['gamma'],
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X_train, y_train, sample_weight=w_train, verbose=False)
            pred_val = model.predict(X_val)
            acc = float(accuracy_score(y_val, pred_val))
            if acc > best_score:
                best_score = acc
                best_params = params | {'scale_pos_weight': scale_pos_weight}
                best_model = model
            print(f"Trial {i+1}/{num_param_samples} - val ACC: {acc:.4f}")

        # Evaluate test set
        y_pred_test = best_model.predict(X_test)
        y_proba_test = best_model.predict_proba(X_test)[:, 1]
        acc_test = float(accuracy_score(y_test, y_pred_test))
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred_test, average='binary', zero_division=0)
        cm = confusion_matrix(y_test, y_pred_test).tolist()

        # Persist artifacts
        joblib.dump(best_model, "models/direction_model.pkl")
        joblib.dump(le, "models/ticker_encoder.pkl")
        with open("models/feature_columns.json", "w") as f:
            json.dump(feature_columns, f)

        with open("models/metrics_direction.json", "w") as f:
            json.dump({
                'best_params': best_params,
                'val': {'accuracy': best_score},
                'test': {
                    'accuracy': acc_test,
                    'precision': float(prec),
                    'recall': float(rec),
                    'f1': float(f1),
                    'confusion_matrix': cm
                },
                'num_rows': len(df),
                'period': period,
            }, f, indent=2)

        # Feature importances - use multiple methods
        importances_data = []
        
        # Method 1: Built-in feature_importances_ (gain-based)
        if hasattr(best_model, 'feature_importances_'):
            for col, imp in zip(feature_columns, best_model.feature_importances_):
                importances_data.append({
                    'feature': col,
                    'gain': float(imp),
                    'weight': 0.0,
                    'cover': 0.0
                })
        
        # Method 2: Get from booster with proper feature names
        booster = best_model.get_booster()
        # Set feature names so we can map them properly
        booster.feature_names = feature_columns
        
        # Try different importance types
        for imp_type in ['gain', 'weight', 'cover']:
            try:
                score_map = booster.get_score(importance_type=imp_type)
                for idx, col in enumerate(feature_columns):
                    # Try both f{idx} and actual feature name
                    score = score_map.get(f'f{idx}', score_map.get(col, 0.0))
                    if idx < len(importances_data):
                        importances_data[idx][imp_type] = float(score)
            except:
                pass
        
        imp_df = pd.DataFrame(importances_data)
        if len(imp_df) > 0:
            # Sort by gain (most important)
            imp_df = imp_df.sort_values('gain', ascending=False)
            imp_df.to_csv('models/feature_importances_direction.csv', index=False)

        # Plot importances
        plt.figure(figsize=(8, 5))
        plt.barh(imp_df['feature'], imp_df['gain'])
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('models/feature_importances_direction.png', dpi=150)
        plt.close()

        print("Direction model and metrics saved to models/. Test accuracy:", acc_test)
        return best_model

    # Regression branch (volatility/return/momentum)
    best_score = float('inf')
    best_params = None
    best_model = None

    for i in range(num_param_samples):
        params = _random_param()
        model = XGBRegressor(
            objective='reg:squarederror',
            tree_method='hist',
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            max_depth=params['max_depth'],
            min_child_weight=params['min_child_weight'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            reg_alpha=params['reg_alpha'],
            reg_lambda=params['reg_lambda'],
            gamma=params['gamma'],
            eval_metric='rmse',
            random_state=42,
            n_jobs=-1,
        )

        model.fit(
            X_train, y_train,
            sample_weight=w_train,
            verbose=False,
        )
        pred_val = model.predict(X_val)
        metrics_val = _evaluate_regression(y_val, pred_val)
        if metrics_val['rmse'] < best_score:
            best_score = metrics_val['rmse']
            best_params = params
            best_model = model
        print(f"Trial {i+1}/{num_param_samples} - val RMSE: {metrics_val['rmse']:.6f}")

    # Evaluate on test using the best model
    y_pred_test = best_model.predict(X_test)
    metrics_test = _evaluate_regression(y_test, y_pred_test)

    # Also compute train/val for reporting
    y_pred_train = best_model.predict(X_train)
    y_pred_val = best_model.predict(X_val)
    metrics_train = _evaluate_regression(y_train, y_pred_train)
    metrics_val = _evaluate_regression(y_val, y_pred_val)

    # Persist artifacts
    joblib.dump(best_model, "models/general_short_term_model.pkl")
    joblib.dump(le, "models/ticker_encoder.pkl")
    with open("models/feature_columns.json", "w") as f:
        json.dump(feature_columns, f)

    with open("models/metrics.json", "w") as f:
        json.dump({
            'best_params': best_params,
            'train': metrics_train,
            'val': metrics_val,
            'test': metrics_test,
            'num_rows': len(df),
            'period': period,
            'splits': {
                'train_frac': 0.7,
                'val_frac': 0.15,
                'test_frac': 0.15
            }
        }, f, indent=2)

    # Feature importances - use multiple methods
    importances_data = []
    
    # Method 1: Built-in feature_importances_ (gain-based)
    if hasattr(best_model, 'feature_importances_'):
        for col, imp in zip(feature_columns, best_model.feature_importances_):
            importances_data.append({
                'feature': col,
                'gain': float(imp),
                'weight': 0.0,
                'cover': 0.0
            })
    
    # Method 2: Get from booster with proper feature names
    booster = best_model.get_booster()
    # Set feature names so we can map them properly
    booster.feature_names = feature_columns
    
    # Try different importance types
    for imp_type in ['gain', 'weight', 'cover']:
        try:
            score_map = booster.get_score(importance_type=imp_type)
            for idx, col in enumerate(feature_columns):
                # Try both f{idx} and actual feature name
                score = score_map.get(f'f{idx}', score_map.get(col, 0.0))
                if idx < len(importances_data):
                    importances_data[idx][imp_type] = float(score)
        except:
            pass
    
    imp_df = pd.DataFrame(importances_data)
    if len(imp_df) > 0:
        # Sort by gain (most important)
        imp_df = imp_df.sort_values('gain', ascending=False)
        imp_df.to_csv('models/feature_importances.csv', index=False)

    # Plot importances
    plt.figure(figsize=(8, 5))
    plt.barh(imp_df['feature'], imp_df['gain'])
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('models/feature_importances.png', dpi=150)
    plt.close()

    # SHAP values (sample for speed)
    try:
        import shap
        shap.initjs()
        sample_n = min(1000, len(X_test))
        X_shap = X_test.tail(sample_n)
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_shap)
        # Save summary plot
        plt.figure(figsize=(8, 5))
        shap.summary_plot(shap_values, X_shap, show=False)
        plt.tight_layout()
        plt.savefig('models/shap_summary.png', dpi=150)
        plt.close()
    except Exception as e:
        print(f"SHAP computation failed: {e}")

    print("Model, encoder, features, metrics, and interpretability artifacts saved to models/.")
    print("Test metrics:", metrics_test)

    return best_model


if __name__ == "__main__":
    train_model()



    


    

