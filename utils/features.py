import ta
import pandas as pd
import numpy as np
from typing import Sequence, Tuple
import yfinance as yf


def _ensure_series(obj: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(obj, pd.DataFrame):
        # Take the first column if a DataFrame slips through (e.g., duplicate column names)
        return obj.iloc[:, 0].astype(float)
    return pd.Series(obj).astype(float)


def add_features(df: pd.DataFrame, rsi_window: int = 14, vol_window: int = 5, momentum_lag: int = 5, ma_short: int = 10, ma_long: int = 50, target_horizon: int = 3, target_type: str = 'volatility') -> pd.DataFrame:
    """Add technical features and target to a price DataFrame.

    Expects columns: ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Ticker']
    Returns a new DataFrame with NaNs dropped after all feature construction.
    
    target_type options:
    - 'return': (future_price - current_price) / current_price
    - 'direction': 1 if future_return > 0, 0 otherwise
    - 'volatility': future_price volatility over target_horizon (RECOMMENDED)
    - 'momentum': future_price momentum (trend strength)
    """
    df = df.copy()

    # Coerce base columns to 1D series
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


def add_market_context(df: pd.DataFrame, period: str = '3y') -> pd.DataFrame:
    """Add market context features (SPY, VIX) to the dataframe.
    
    Args:
        df: DataFrame with Date column and stock data
        period: Period to fetch market data (should match stock data period)
    
    Returns:
        DataFrame with added market context features
    """
    df = df.copy()
    
    # Ensure Date column exists and is datetime
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
        spy = yf.download('SPY', start=start_str, end=end_str, progress=False, auto_adjust=False)
        if not spy.empty:
            spy.reset_index(inplace=True)
            if 'index' in spy.columns:
                spy.rename(columns={'index': 'Date'}, inplace=True)
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
        vix = yf.download('^VIX', start=start_str, end=end_str, progress=False, auto_adjust=False)
        if not vix.empty:
            vix.reset_index(inplace=True)
            if 'index' in vix.columns:
                vix.rename(columns={'index': 'Date'}, inplace=True)
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
    # Market context features (added by add_market_context)
    'SPY_Return', 'SPY_Volatility', 'SPY_Trend_5d', 'Relative_Return', 'Relative_Volatility',
    'VIX_Level', 'VIX_Change'
)