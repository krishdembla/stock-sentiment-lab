import ta
import pandas as pd
import numpy as np
from typing import Sequence, Tuple
import yfinance as yf
import requests
from datetime import datetime, timedelta
import time


def _ensure_series(obj: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(obj, pd.DataFrame):
        return obj.iloc[:, 0].astype(float)
    return pd.Series(obj).astype(float)


# ============================================================================
# SENTIMENT FEATURES - Fear & Greed Index
# ============================================================================

def get_fear_greed_index(date: pd.Timestamp = None) -> float:
    """
    Fetch Fear & Greed Index from Alternative.me API.
    
    Returns value 0-100:
    - 0-25: Extreme Fear
    - 25-45: Fear
    - 45-55: Neutral
    - 55-75: Greed
    - 75-100: Extreme Greed
    
    Args:
        date: Date to fetch index for (defaults to today)
    
    Returns:
        Fear & Greed Index value (0-100), or 50 (neutral) on error
    """
    try:
        # API endpoint (free, no key required)
        url = "https://api.alternative.me/fng/?limit=2000&format=json"
        
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if not data.get('data'):
            return 50.0  # Neutral default
        
        # If no date specified, return latest
        if date is None:
            return float(data['data'][0]['value'])
        
        # Find closest date
        target_timestamp = int(date.timestamp())
        closest_value = 50.0
        min_diff = float('inf')
        
        for entry in data['data']:
            entry_timestamp = int(entry['timestamp'])
            diff = abs(entry_timestamp - target_timestamp)
            
            if diff < min_diff:
                min_diff = diff
                closest_value = float(entry['value'])
            
            # If we found exact match or within 1 day, use it
            if diff < 86400:  # 1 day in seconds
                return closest_value
        
        return closest_value
        
    except Exception as e:
        # Return neutral (50) on any error
        print(f"Warning: Could not fetch Fear & Greed Index: {e}")
        return 50.0


def add_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    adds sentiment features using the fear & greed index
    
    pulls historical fear & greed data (0-100 scale) and calculates:
    - current reading
    - 7-day moving average
    - day-over-day change
    - extreme fear indicator (below 25)
    - extreme greed indicator (above 75)
    - overall sentiment regime
    
    returns the dataframe with sentiment columns added
    """
    df = df.copy()
    
    # make sure we have a date column to work with
    if 'Date' not in df.columns:
        raise ValueError("DataFrame must have a 'Date' column")
    
    print("Fetching Fear & Greed Index data...")
    
    # Fetch once for all dates (API returns historical data)
    try:
        url = "https://api.alternative.me/fng/?limit=2000&format=json"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        api_data = response.json()
        
        # Create lookup dictionary: date -> value
        fg_lookup = {}
        for entry in api_data.get('data', []):
            entry_date = pd.to_datetime(int(entry['timestamp']), unit='s').date()
            fg_lookup[entry_date] = float(entry['value'])
        
        # Map to dataframe dates
        df['Fear_Greed'] = df['Date'].apply(
            lambda d: fg_lookup.get(pd.to_datetime(d).date(), 50.0)
        )
        
        print(f"  Fetched {len(fg_lookup)} days of Fear & Greed data")
        
    except Exception as e:
        print(f"  Warning: Error fetching Fear & Greed data: {e}")
        print(f"  Using neutral default (50) for all dates")
        df['Fear_Greed'] = 50.0
    
    # Derived sentiment features
    df['Fear_Greed_MA7'] = df['Fear_Greed'].rolling(7, min_periods=1).mean()
    df['Fear_Greed_MA30'] = df['Fear_Greed'].rolling(30, min_periods=1).mean()
    df['Fear_Greed_Change'] = df['Fear_Greed'].diff()
    df['Fear_Greed_Change_7d'] = df['Fear_Greed'].diff(7)
    
    # Regime indicators (binary flags)
    df['Extreme_Fear'] = (df['Fear_Greed'] < 25).astype(int)
    df['Extreme_Greed'] = (df['Fear_Greed'] > 75).astype(int)
    df['Fear_Zone'] = (df['Fear_Greed'] < 45).astype(int)  # Fear or extreme fear
    df['Greed_Zone'] = (df['Fear_Greed'] > 55).astype(int)  # Greed or extreme greed
    
    # Trend: Is fear/greed increasing or decreasing?
    df['Fear_Greed_Rising'] = (df['Fear_Greed_MA7'] > df['Fear_Greed_MA30']).astype(int)
    
    # Volatility of sentiment (how quickly sentiment changes)
    df['Fear_Greed_Volatility',
    # Realized volatility features (added by add_realized_volatility_features) - NEW for Phase 1
    'Realized_Vol_5d', 'Realized_Vol_20d', 'Realized_Vol_60d', 'Vol_Ratio_20_60', 'Vol_Distance_Mean',
    'Vol_Persistence', 'Vol_Momentum', 'Vol_of_Vol', 'Vol_Trend_20d', 'Vol_Regime_Low', 
    'Vol_Regime_High', 'Realized_Vol_EMA_20'] = df['Fear_Greed'].rolling(14, min_periods=1).std()
    
    print(f"  ✓ Created 10 sentiment features")
    
    return df



def add_features(df: pd.DataFrame, rsi_window: int = 14, vol_window: int = 5, momentum_lag: int = 5, ma_short: int = 10, ma_long: int = 50, target_horizon: int = 3, target_type: str = 'volatility') -> pd.DataFrame:
    """
    calculates all the technical indicators for a stock
    
    takes raw ohlcv data and adds features like:
    - moving averages, rsi, macd, bollinger bands
    - volume indicators (obv, money flow)
    - volatility measurements
    - momentum and trend signals
    
    expects columns: date, open, high, low, close, adj close, volume, ticker
    
    returns dataframe with ~40 new feature columns
    """
    df = df.copy()

    # convert columns to 1d series (sometimes they come as dataframes)
    close_series = _ensure_series(df['Close'])
    volume_series = _ensure_series(df['Volume'])
    high_series = _ensure_series(df['High'])
    low_series = _ensure_series(df['Low'])

    # Basic price-derived features
    df['Return'] = close_series.pct_change()
    df['Volatility'] = df['Return'].rolling(window=vol_window).std()
    df['Momentum'] = close_series - close_series.shift(momentum_lag)
    df[f'MA{ma_short}'] = close_series.rolling(window=ma_short).mean()
    df[f'MA{ma_long}'] = close_series.rolling(window=ma_long).mean()

    # Enhanced volatility-specific features
    df['High_Low_Range'] = (high_series - low_series) / close_series
    df['Price_Range_5d'] = (high_series.rolling(5).max() - low_series.rolling(5).min()) / close_series
    df['Volatility_10d'] = df['Return'].rolling(window=10).std()
    df['Volatility_20d'] = df['Return'].rolling(window=20).std()
    df['Volatility_Ratio'] = df['Volatility_10d'] / df['Volatility_20d']
    
    # Trend features
    df['Trend_5d'] = (close_series - close_series.shift(5)) / close_series.shift(5)
    df['Trend_10d'] = (close_series - close_series.shift(10)) / close_series.shift(10)
    df['Trend_20d'] = (close_series - close_series.shift(20)) / close_series.shift(20)
    
    # Moving average relationships
    df['MA_Ratio'] = df[f'MA{ma_short}'] / df[f'MA{ma_long}']
    df['Price_vs_MA10'] = (close_series / df[f'MA{ma_short}'] - 1)
    df['Price_vs_MA50'] = (close_series / df[f'MA{ma_long}'] - 1)
    
    # Volume features
    df['Volume_MA'] = volume_series.rolling(window=20).mean()
    df['Volume_Ratio'] = volume_series / df['Volume_MA']
    df['Volume_Price_Trend'] = volume_series * df['Return']

    # Indicators using ta
    df['RSI'] = ta.momentum.RSIIndicator(close=close_series, window=rsi_window).rsi()
    macd = ta.trend.MACD(close=close_series)
    df['MACD'] = macd.macd_diff()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Histogram'] = macd.macd_diff() - macd.macd_signal()
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=close_series, volume=volume_series).on_balance_volume()

    # Additional technical indicators
    bb = ta.volatility.BollingerBands(close=close_series)
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Lower'] = bb.bollinger_lband()
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / close_series
    bb_range = (df['BB_Upper'] - df['BB_Lower']).replace(0, np.nan)
    df['BB_Position'] = (close_series - df['BB_Lower']) / bb_range
    
    # ATR needs at least 14 data points
    try:
        if len(close_series) >= 14:
            atr = ta.volatility.AverageTrueRange(high=high_series, low=low_series, close=close_series)
            df['ATR'] = atr.average_true_range()
            df['ATR_Ratio'] = (df['ATR'] / close_series).fillna(0)
        else:
            df['ATR'] = 0.0
            df['ATR_Ratio'] = 0.0
    except Exception:
        df['ATR'] = 0.0
        df['ATR_Ratio'] = 0.0
    
    stoch = ta.momentum.StochasticOscillator(high=high_series, low=low_series, close=close_series)
    df['Stoch_K'] = stoch.stoch()
    df['Stoch_D'] = stoch.stoch_signal()
    
    df['Williams_R'] = ta.momentum.WilliamsRIndicator(high=high_series, low=low_series, close=close_series).williams_r()
    
    # Volatility clustering features
    df['Volatility_Regime'] = (df['Volatility'] > df['Volatility'].rolling(50).mean()).astype(int)
    df['High_Volatility'] = (df['Volatility'] > df['Volatility'].rolling(50).quantile(0.8)).astype(int)
    df['Low_Volatility'] = (df['Volatility'] < df['Volatility'].rolling(50).quantile(0.2)).astype(int)

    # Chaikin Money Flow (CMF)
    df['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(high=high_series, low=low_series, close=close_series, volume=volume_series).chaikin_money_flow()

    # Ichimoku Cloud
    ichimoku = ta.trend.IchimokuIndicator(high=high_series, low=low_series)
    df['Ichimoku_A'] = ichimoku.ichimoku_a()
    df['Ichimoku_B'] = ichimoku.ichimoku_b()
    df['Ichimoku_Base'] = ichimoku.ichimoku_base_line()
    df['Ichimoku_Conversion'] = ichimoku.ichimoku_conversion_line()

    # Rate of Change (ROC)
    df['ROC'] = ta.momentum.ROCIndicator(close=close_series).roc()

    # Commodity Channel Index (CCI)
    df['CCI'] = ta.trend.CCIIndicator(high=high_series, low=low_series, close=close_series).cci()

    # Interaction Features
    df['RSI_x_Volume'] = df['RSI'] * df['Volume_Ratio']
    df['Volatility_x_Momentum'] = df['Volatility'] * df['Momentum']

    # Time-Based Features
    df['Day_of_Week'] = pd.to_datetime(df['Date']).dt.dayofweek
    df['Month'] = pd.to_datetime(df['Date']).dt.month

    # Fourier Transforms for Seasonality
    close_fft = np.fft.fft(close_series.fillna(0))
    df['FFT_Real'] = np.real(close_fft)
    df['FFT_Imag'] = np.imag(close_fft)
    
    # Add realized volatility features (for enhanced volatility prediction)
    df = add_realized_volatility_features(df)

    # Multiple target formulations
    if target_type == 'return':
        df['Target'] = (close_series.shift(-target_horizon) - close_series) / close_series
    elif target_type == 'direction':
        future_return = (close_series.shift(-target_horizon) - close_series) / close_series
        df['Target'] = (future_return > 0).astype(int)
    elif target_type == 'volatility':
        future_returns = close_series.pct_change().shift(-1).rolling(window=target_horizon).std()
        df['Target'] = future_returns
    elif target_type == 'momentum':
        df['Target'] = close_series.shift(-target_horizon) / close_series - 1
    else:
        raise ValueError(f"Unknown target_type: {target_type}")

    # Drop rows with any NaNs introduced by rolling/shift ops
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    return df


# ============================================================================
# VOLATILITY-SPECIFIC FEATURES - For Enhanced Volatility Prediction
# ============================================================================

def add_realized_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    adds realized volatility features to help predict future volatility
    
    these features capture how volatile the stock has been recently:
    - short-term (5-day), medium (20-day), long-term (60-day) volatility
    - whether current volatility is higher or lower than usual
    - if volatility is trending up or down
    - volatility of volatility (how unstable the volatility itself is)
    
    why this works: volatility has momentum - if it's been volatile lately, 
    it'll probably stay volatile. but it also mean-reverts - extreme volatility 
    doesn't last forever.
    
    returns dataframe with ~12 new volatility-focused features
    """
    df = df.copy()
    
    # make sure we have returns calculated first
    if 'Return' not in df.columns:
        df['Return'] = df['Close'].pct_change()
    
    # calculate realized volatility over different time windows
    # (standard deviation of returns)
    
    # 5 days = short-term volatility
    df['Realized_Vol_5d'] = df['Return'].rolling(window=5).std()
    
    # Medium-term volatility (20 trading days = 1 month)
    df['Realized_Vol_20d'] = df['Return'].rolling(window=20).std()
    
    # Long-term volatility (60 trading days = 3 months)
    df['Realized_Vol_60d'] = df['Return'].rolling(window=60).std()
    
    # 2. Mean Reversion Signal: Current vol vs long-term average
    # When ratio > 1: volatility is elevated (likely to decrease)
    # When ratio < 1: volatility is suppressed (likely to increase)
    df['Vol_Ratio_20_60'] = df['Realized_Vol_20d'] / (df['Realized_Vol_60d'] + 1e-8)
    
    # Alternative mean reversion: distance from long-term mean
    long_term_mean_vol = df['Realized_Vol_60d'].rolling(window=252, min_periods=60).mean()
    df['Vol_Distance_Mean'] = (df['Realized_Vol_20d'] - long_term_mean_vol) / (long_term_mean_vol + 1e-8)
    
    # 3. Volatility Persistence: Autocorrelation of volatility
    # High persistence means volatility trends continue
    # Calculate as correlation between recent vol and previous vol
    df['Vol_Persistence'] = df['Realized_Vol_20d'].rolling(window=40).apply(
        lambda x: x.autocorr(lag=1) if len(x) >= 2 else 0, raw=False
    )
    
    # Alternative persistence: ratio of short-term to medium-term vol
    df['Vol_Momentum'] = df['Realized_Vol_5d'] / (df['Realized_Vol_20d'] + 1e-8)
    
    # 4. Volatility of Volatility (Vol-of-Vol): Uncertainty measure
    # High vol-of-vol means volatility itself is unstable
    df['Vol_of_Vol'] = df['Realized_Vol_20d'].rolling(window=20).std()
    
    # 5. Volatility Trend: Is volatility increasing or decreasing?
    df['Vol_Trend_20d'] = df['Realized_Vol_20d'] - df['Realized_Vol_20d'].shift(20)
    
    # 6. Volatility Regime Indicators (quartile-based)
    # Identify if current volatility is low, medium, or high vs historical
    vol_q25 = df['Realized_Vol_60d'].rolling(window=252, min_periods=60).quantile(0.25)
    vol_q75 = df['Realized_Vol_60d'].rolling(window=252, min_periods=60).quantile(0.75)
    
    df['Vol_Regime_Low'] = (df['Realized_Vol_20d'] < vol_q25).astype(int)
    df['Vol_Regime_High'] = (df['Realized_Vol_20d'] > vol_q75).astype(int)
    
    # 7. Exponential Moving Average of volatility (gives more weight to recent data)
    df['Realized_Vol_EMA_20'] = df['Return'].ewm(span=20).std()
    
    # Fill any remaining NaNs with 0 (for initial rows where windows aren't full)
    volatility_cols = [
        'Realized_Vol_5d', 'Realized_Vol_20d', 'Realized_Vol_60d',
        'Vol_Ratio_20_60', 'Vol_Distance_Mean', 'Vol_Persistence',
        'Vol_Momentum', 'Vol_of_Vol', 'Vol_Trend_20d',
        'Vol_Regime_Low', 'Vol_Regime_High', 'Realized_Vol_EMA_20'
    ]
    
    for col in volatility_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    return df


def add_market_context(df: pd.DataFrame, period: str = '3y') -> pd.DataFrame:
    """
    adds market context features using spy and vix
    
    gives the model information about what the broader market is doing:
    - spy return, volatility, and trend (market benchmark)
    - vix level and change (fear gauge)
    - relative metrics (how this stock compares to spy)
    
    helps the model understand if high volatility is stock-specific or market-wide
    
    returns dataframe with ~7 market context features
    """
    df = df.copy()
    
    # make sure we have a date column
    if 'Date' not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df.reset_index(inplace=True)
            df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
        else:
            raise ValueError("DataFrame must have a Date column or DatetimeIndex")
    
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Check if we have valid dates
    if df['Date'].isna().all() or len(df) == 0:
        print("Warning: No valid dates in dataframe, skipping market context")
        market_cols = ['SPY_Return', 'SPY_Volatility', 'SPY_Trend_5d', 'Relative_Return', 
                      'Relative_Volatility', 'VIX_Level', 'VIX_Change']
        for col in market_cols:
            df[col] = 0.0
        return df
    
    start_date = df['Date'].min()
    end_date = df['Date'].max()
    
    # Convert to string format for yfinance
    start_str = pd.Timestamp(start_date).strftime('%Y-%m-%d') if pd.notna(start_date) else None
    end_str = pd.Timestamp(end_date).strftime('%Y-%m-%d') if pd.notna(end_date) else None
    
    if not start_str or not end_str:
        print("Warning: Invalid date range, skipping market context")
        market_cols = ['SPY_Return', 'SPY_Volatility', 'SPY_Trend_5d', 'Relative_Return', 
                      'Relative_Volatility', 'VIX_Level', 'VIX_Change']
        for col in market_cols:
            df[col] = 0.0
        return df
    
    # Fetch SPY (market) data
    try:
        # Try cache first
        from pathlib import Path
        cache_path = Path(__file__).parent.parent / "cache" / "SPY.csv"
        if cache_path.exists():
            spy = pd.read_csv(cache_path, parse_dates=['Date'])
        else:
            spy = yf.download('SPY', start=start_str, end=end_str, progress=False, auto_adjust=False)
            spy.reset_index(inplace=True)
            if 'index' in spy.columns:
                spy.rename(columns={'index': 'Date'}, inplace=True)
        
        if not spy.empty:
            spy['Date'] = pd.to_datetime(spy['Date'])
            spy['SPY_Return'] = spy['Close'].pct_change()
            spy['SPY_Volatility'] = spy['SPY_Return'].rolling(5).std()
            spy['SPY_Trend_5d'] = (spy['Close'] - spy['Close'].shift(5)) / spy['Close'].shift(5)
            
            # Merge SPY features
            df = df.merge(spy[['Date', 'SPY_Return', 'SPY_Volatility', 'SPY_Trend_5d']], 
                         on='Date', how='left')
            
            # Relative performance features
            if 'Return' in df.columns:
                df['Relative_Return'] = df['Return'] - df['SPY_Return']
            if 'Volatility' in df.columns:
                df['Relative_Volatility'] = df['Volatility'] - df['SPY_Volatility']
    except Exception as e:
        print(f"Warning: Could not fetch SPY data: {e}")
        df['SPY_Return'] = 0.0
        df['SPY_Volatility'] = 0.0
        df['SPY_Trend_5d'] = 0.0
        df['Relative_Return'] = 0.0
        df['Relative_Volatility'] = 0.0
    
    # Fetch VIX (volatility index) data
    try:
        # Try cache first
        from pathlib import Path
        cache_path = Path(__file__).parent.parent / "cache" / "VIX.csv"
        if cache_path.exists():
            vix = pd.read_csv(cache_path, parse_dates=['Date'])
        else:
            vix = yf.download('^VIX', start=start_str, end=end_str, progress=False, auto_adjust=False)
            vix.reset_index(inplace=True)
            if 'index' in vix.columns:
                vix.rename(columns={'index': 'Date'}, inplace=True)
        
        if not vix.empty:
            vix['Date'] = pd.to_datetime(vix['Date'])
            vix['VIX_Level'] = vix['Close']
            vix['VIX_Change'] = vix['Close'].pct_change()
            
            # Merge VIX features
            df = df.merge(vix[['Date', 'VIX_Level', 'VIX_Change']], 
                         on='Date', how='left')
    except Exception as e:
        print(f"Warning: Could not fetch VIX data: {e}")
        df['VIX_Level'] = 20.0  # Default VIX level
        df['VIX_Change'] = 0.0
    
    # Fill NaN values from market data (forward fill then backward fill)
    market_cols = [str(col) for col in df.columns if str(col).startswith('SPY_') or str(col).startswith('VIX_') or str(col).startswith('Relative_')]
    for col in market_cols:
        if col in df.columns:
            df[col] = df[col].ffill().bfill().fillna(0.0)
    
    return df


FEATURE_COLUMNS_DEFAULT: Tuple[str, ...] = (
    'Return', 'Volatility', 'Momentum', 'MA10', 'MA50', 'RSI', 'MACD', 'OBV',
    'High_Low_Range', 'Price_Range_5d', 'Volatility_10d', 'Volatility_20d', 'Volatility_Ratio',
    'Trend_5d', 'Trend_10d', 'Trend_20d', 'MA_Ratio', 'Price_vs_MA10', 'Price_vs_MA50',
    'Volume_Ratio', 'Volume_Price_Trend', 'MACD_Signal', 'MACD_Histogram',
    'BB_Width', 'BB_Position', 'ATR_Ratio', 'Stoch_K', 'Stoch_D', 'Williams_R',
    'Volatility_Regime', 'High_Volatility', 'Low_Volatility',
    'CMF', 'Ichimoku_A', 'Ichimoku_B', 'Ichimoku_Base', 'Ichimoku_Conversion', 'ROC', 'CCI',
    # New features
    'RSI_x_Volume', 'Volatility_x_Momentum', 'Day_of_Week', 'Month', 'FFT_Real', 'FFT_Imag',
    # Market context features (added by add_market_context)
    'SPY_Return', 'SPY_Volatility', 'SPY_Trend_5d', 'Relative_Return', 'Relative_Volatility',
    'VIX_Level', 'VIX_Change',
    # Sentiment features (added by add_sentiment_features)
    'Fear_Greed', 'Fear_Greed_MA7', 'Fear_Greed_MA30', 'Fear_Greed_Change', 'Fear_Greed_Change_7d',
    'Extreme_Fear', 'Extreme_Greed', 'Fear_Zone', 'Greed_Zone', 'Fear_Greed_Rising', 'Fear_Greed_Volatility',
    # Realized volatility features (added by add_realized_volatility_features) - NEW for Phase 1
    'Realized_Vol_5d', 'Realized_Vol_20d', 'Realized_Vol_60d', 'Vol_Ratio_20_60', 'Vol_Distance_Mean',
    'Vol_Persistence', 'Vol_Momentum', 'Vol_of_Vol', 'Vol_Trend_20d', 'Vol_Regime_Low', 
    'Vol_Regime_High', 'Realized_Vol_EMA_20'
)