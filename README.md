# stock volatility prediction & position sizing

predicts how much stocks will move (not which direction) and uses that to figure out position sizes for risk management.

**[🚀 Try Live Demo](https://your-app.streamlit.app)** | [View Code](https://github.com/krishdembla/stock-sentiment-lab)

## what it does

most people size positions based on gut feel or arbitrary percentages. this uses machine learning to adjust position sizes based on predicted volatility:

- **low volatility stock** (like WMT) → bigger position
- **high volatility stock** (like TSLA) → smaller position
- **each position risks the same dollar amount**

example with $100k account, risking 1% ($1,000) per position:
```
TSLA (3% daily volatility):  $16,779 position  →  risks $1,000
WMT  (0.9% daily volatility): $56,180 position  →  risks $1,000
```

same risk, different position sizes. that's the point.

## how it works

the xgboost model predicts daily volatility using 76 features:
- technical indicators (rsi, macd, bollinger bands, moving averages)
- market context (spy and vix movements)
- realized volatility over different time windows

model performance: **R² = 0.533** (explains 53% of volatility variation)

position sizing formula:
```
position = (account × risk%) / (2 × predicted_volatility)
```

the 2x multiplier is a safety buffer (2 standard deviations).

## quick start

**web app (easiest):**
```bash
streamlit run app.py
```
Then open http://localhost:8501 in your browser.

**command line:**

run the demo (no downloads needed):
```bash
python demo_position_sizing.py
```

or get predictions for specific stocks:
```bash
python predict.py --tickers AAPL MSFT TSLA --account 100000 --risk 0.01
```

the cache already has data for 28 stocks (aapl, msft, tsla, nvda, etc). to refresh:
```bash
python download_yahooquery.py
```

## what's in the repo

- **predict.py**: cli tool for position sizing calculations
- **demo_position_sizing.py**: standalone demo with example predictions
- **download_yahooquery.py**: downloads fresh data for all 28 tickers
- **models/**: trained xgboost model and metadata
- **cache/**: pre-downloaded stock data (4 years for each ticker)
- **utils/features.py**: feature engineering pipeline

## dependencies

```bash
pip install -r requirements.txt
```

main libraries: pandas, xgboost, yahooquery, scikit-learn

## model details

trained on 28 large-cap stocks with 4 years of data each. uses time-based train/val/test splits (70/15/15) to avoid lookahead bias.

the model does better at predicting volatility than stock direction because volatility has patterns - it clusters (high vol follows high vol) and mean-reverts (extreme vol doesn't last). price direction is basically random.

## why volatility prediction

tried predicting if stocks go up or down → got 50% accuracy (coin flip).

switched to predicting volatility → got R² of 0.533 (actually works).

turns out you can't predict direction, but you can predict how much stocks will move. that's useful for risk management even if you don't know which way they'll go.

## the 76 features

technical indicators (moving averages, rsi, macd, bollinger bands), market context (spy/vix correlation), realized volatility over different windows, volume patterns, and time features.

top predictors are short-term realized volatility (5-day, 20-day) and current volatility vs historical average. the model basically learns that volatility clusters - if yesterday was volatile, today probably will be too.

## known issues

- only works with the 28 stocks it was trained on (labelencoder limitation)
- needs 60+ days of data per stock (can't predict new ipos)
- yahoo finance api occasionally rate limits (use cache or wait)
- model trained on data through dec 2024, should be retrained periodically

## what i learned

started trying to predict if stocks go up or down. wasted time. learned that's basically impossible.

switched to predicting volatility instead. actually works. learned that some patterns in markets are predictable even when direction isn't.

built a practical tool that portfolio managers could actually use instead of just another ml model without purpose.
