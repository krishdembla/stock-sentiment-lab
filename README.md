# Risk Parity Portfolio Builder

An ML-powered portfolio construction tool that uses predicted volatility to build equal-risk portfolios. Instead of sizing positions by dollar amount or arbitrary percentages, this tool adjusts position sizes so each stock contributes the same amount of risk to your portfolio.

Try it here: https://riskparityportfoliobuilder.streamlit.app

## What It Does

Traditional portfolio allocation gives each position the same dollar weight (equal weighting) or weights by market cap. Both ignore risk. A $10,000 position in a volatile tech stock carries far more risk than a $10,000 position in a stable utility stock.

Risk parity fixes this by using volatility predictions to size positions:

- Low volatility stock (WMT, JNJ, PG) gets a larger position
- High volatility stock (TSLA, NVDA, META) gets a smaller position  
- Each position risks the same dollar amount

Example with $100,000 account risking 1% ($1,000) per position:

```
TSLA (predicted 3% daily volatility):   $16,667 position
WMT  (predicted 0.9% daily volatility): $55,556 position
```

Both positions risk $1,000, but TSLA gets 1/3 the capital because it's 3x more volatile.

## How It Works

The app uses an XGBoost regression model trained to predict daily volatility. The model analyzes 76 technical features including:

- Realized volatility over multiple time windows (5-day, 20-day, 60-day)
- Technical indicators (RSI, MACD, Bollinger Bands, ATR)
- Market context (S&P 500 and VIX movements)
- Volume patterns and price momentum

Model performance: R² = 0.533 on test data (explains 53% of volatility variation). This is on par with academic benchmarks for volatility forecasting.

Once volatility is predicted, positions are sized using:

```
position_size = (account_value * risk_per_trade) / (2 * predicted_volatility)
```

The 2x multiplier provides a safety buffer (roughly 2 standard deviations).

## Features

- Real-time volatility predictions for 28 large-cap stocks
- Interactive portfolio construction with customizable account size and risk tolerance
- Backtest analysis showing historical performance vs equal-weight benchmark
- Performance metrics including Sharpe ratio, max drawdown, and return statistics
- Visual equity curve comparison
- ML model transparency (feature importance, prediction confidence, model metrics)

## Repository Structure

- **app.py**: Streamlit web interface
- **predict.py**: Core prediction and position sizing logic
- **backtest.py**: Historical backtest engine
- **data/fetch_stock_data.py**: Data download utilities
- **models/**: Trained XGBoost model and feature metadata
- **utils/features.py**: Feature engineering pipeline
- **cache/**: Pre-downloaded stock data (28 tickers, 4 years each)

## Running Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

The app will open at http://localhost:8501.

## Model Training

The model was trained on 28 large-cap stocks with 4 years of daily data (19,684 total observations). Training used time-based splits (70% train, 15% validation, 15% test) to avoid lookahead bias.

Volatility is more predictable than price direction because it exhibits clustering (high volatility tends to follow high volatility) and mean reversion (extreme volatility doesn't persist). The model captures these patterns through short-term realized volatility features and technical indicators.

Key predictors: 5-day realized volatility, 20-day realized volatility, ATR (Average True Range), and volatility ratio (current vs historical average).

## Why Volatility Instead of Direction

Initial attempts to predict price direction achieved 50% accuracy (no better than random). Volatility prediction reached R² of 0.533, which is meaningful for risk management even without knowing which direction prices will move.

This aligns with empirical finance research: returns are largely unpredictable, but volatility shows autocorrelation and can be forecast with moderate accuracy.

## Technologies

Python, Streamlit, XGBoost, scikit-learn, pandas, yfinance, plotly, SHAP

## known issues

- only works with the 28 stocks it was trained on (labelencoder limitation)
- needs 60+ days of data per stock (can't predict new ipos)
- yahoo finance api occasionally rate limits (use cache or wait)
- model trained on data through dec 2024, should be retrained periodically

## what i learned

started trying to predict if stocks go up or down. wasted time. learned that's basically impossible.

switched to predicting volatility instead. actually works. learned that some patterns in markets are predictable even when direction isn't.

built a practical tool that portfolio managers could actually use instead of just another ml model without purpose.
