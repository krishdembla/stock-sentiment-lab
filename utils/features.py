import ta


def add_features(df):
    print("Close shape:", df['Close'].shape)

    df['Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Return'].rolling(window=5).std()
    df['Momentum'] = df['Close'] - df['Close'].shift(5)
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()

    # RSI
    rsi = ta.momentum.RSIIndicator(close=df['Close'], window=14)
    df['RSI'] = rsi.rsi().values.flatten()

    # MACD
    macd = ta.trend.MACD(close=df['Close'])
    df['MACD'] = macd.macd_diff().values.flatten()

    # OBV
    obv = ta.volume.OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume'])
    df['OBV'] = obv.on_balance_volume().values.flatten()

    df['Target'] = (df['Close'].shift(-3) - df['Close']) / df['Close']

    return df.dropna()



