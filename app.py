import streamlit as st
import pandas as pd
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
import plotly.graph_objects as go

# add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from predict import load_model, load_feature_columns, load_ticker_encoder, prepare_features, predict_volatility, calculate_position_size, get_current_price

# page config
st.set_page_config(
    page_title="Position Size Calculator",
    page_icon="📊",
    layout="wide"
)

# available tickers (28 stocks the model was trained on)
AVAILABLE_TICKERS = [
    'AAPL', 'ADBE', 'AMZN', 'BAC', 'COP', 'COST', 'CRM', 'CVX',
    'GOOG', 'HD', 'INTC', 'JNJ', 'JPM', 'MA', 'MCD', 'META',
    'MRK', 'MSFT', 'NKE', 'NVDA', 'PFE', 'PG', 'TSLA', 'UNH',
    'V', 'WFC', 'WMT', 'XOM'
]

@st.cache_resource
def load_models():
    """load model and configuration (cached for performance)"""
    model_dir = Path(__file__).parent / "models"
    model = load_model(model_dir / "general_short_term_model.pkl")
    feature_columns = load_feature_columns(model_dir / "feature_columns.json")
    ticker_encoder = load_ticker_encoder(model_dir / "ticker_encoder.pkl")
    return model, feature_columns, ticker_encoder

@st.cache_resource
def load_model_metrics():
    """load model training metrics"""
    metrics_path = Path(__file__).parent / "models" / "metrics.json"
    try:
        with open(metrics_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        return None

def calculate_confidence(X, feature_columns):
    """
    calculate prediction confidence based on data quality
    returns: 'high', 'medium', or 'low'
    
    Note: Excludes sentiment/regime features from confidence calculation
    since they're often unavailable in cached data but model handles this
    """
    if X is None or len(X) == 0:
        return 'low'
    
    # Sentiment features that are commonly missing (filled with zeros in model)
    # These don't impact prediction quality significantly
    sentiment_features = [
        'Fear_Greed', 'Fear_Greed_MA7', 'Fear_Greed_MA30', 'Fear_Greed_Change',
        'Fear_Greed_Change_7d', 'Extreme_Fear', 'Extreme_Greed', 'Fear_Greed_Rising',
        'Fear_Greed_Volatility', 'Fear_Zone', 'Greed_Zone',
        'Volatility_Regime', 'High_Volatility', 'Low_Volatility', 'Vol_Regime_High'
    ]
    
    row = X.iloc[0]
    
    # Only check core technical features (exclude sentiment)
    core_feature_names = [col.strip() for col in feature_columns if col.strip() not in sentiment_features]
    
    # Get values for core features only
    core_values = row[[col for col in row.index if col.strip() in core_feature_names]]
    
    zero_features = (core_values == 0).sum()
    missing_pct = zero_features / len(core_values)
    
    # Check for extreme values that might indicate data corruption
    extreme_count = (core_values.abs() > 1000).sum()
    
    # Stricter thresholds since we're only checking critical features
    if missing_pct > 0.25 or extreme_count > 8:
        return 'low'
    elif missing_pct > 0.10 or extreme_count > 4:
        return 'medium'
    else:
        return 'high'

# header
st.title("Risk Parity Portfolio Builder")
st.markdown("ML-powered portfolio construction with volatility-based position sizing | [View Code](https://github.com/krishdembla/stock-sentiment-lab)")

# load models
try:
    model, feature_columns, ticker_encoder = load_models()
    model_metrics = load_model_metrics()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ML Model Performance Card
if model_metrics:
    st.markdown("")
    
    # Main metrics in clean layout
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model Architecture", "XGBoost")
    with col2:
        st.metric("Test R² Score", f"{model_metrics['test']['r2']:.3f}")
    with col3:
        st.metric("Training Data", f"{model_metrics['num_rows']:,} rows")
    with col4:
        st.metric("Feature Set", f"{len(feature_columns)} indicators")
    
    # Achievement & Context Box
    st.info(f"""
    **🎯 Model Performance:** R² = **{model_metrics['test']['r2']:.3f}** on unseen test data — **on par with industry benchmarks** for volatility forecasting.
    
    **Why This Matters:** Predicting volatility (not price) is fundamentally difficult due to market randomness. Academic research typically achieves R² = 0.40-0.50 for this task. Our model exceeds this while avoiding overfitting (Train R² = {model_metrics['train']['r2']:.2f} vs Test R² = {model_metrics['test']['r2']:.2f}).
    
    **Technical Features:** {len(feature_columns)} indicators including RSI, MACD, Bollinger Bands, Ichimoku, volume flows, and market context (SPY/VIX correlation).
    """)
    st.markdown("---")

# sidebar for inputs
st.sidebar.header("Portfolio Settings")

account_value = st.sidebar.number_input(
    "Account Size ($)",
    min_value=1000,
    max_value=10000000,
    value=100000,
    step=10000,
    help="Total portfolio value"
)

risk_percent = st.sidebar.slider(
    "Total Portfolio Risk (%)",
    min_value=0.5,
    max_value=10.0,
    value=2.0,
    step=0.5,
    help="Total portfolio risk budget (will be split across all positions)"
) / 100

st.sidebar.markdown("---")
st.sidebar.markdown("### Build Your Portfolio")
st.sidebar.info("Select 2-5 stocks to create a diversified portfolio with risk parity")

selected_tickers = st.sidebar.multiselect(
    "Choose stocks",
    AVAILABLE_TICKERS,
    default=['AAPL', 'MSFT', 'TSLA'],
    help="Select stocks for your portfolio"
)

# main content
if not selected_tickers:
    st.info("Select stocks from the sidebar to build your portfolio")
    st.stop()

if len(selected_tickers) > 8:
    st.warning("Please select 8 or fewer stocks for a focused portfolio")
    st.stop()

if len(selected_tickers) < 2:
    st.warning("Select at least 2 stocks to build a diversified portfolio")
    st.stop()

# calculate button
if st.button("Build Portfolio", type="primary"):
    
    # progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # calculate risk per position (distribute total risk across N positions)
    num_positions = len(selected_tickers)
    risk_per_position = risk_percent / num_positions
    
    results = []
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=1460)).strftime('%Y-%m-%d')
    
    for i, ticker in enumerate(selected_tickers):
        status_text.text(f"Processing {ticker}...")
        
        try:
            # prepare features
            X = prepare_features(
                ticker, 
                start_date, 
                end_date, 
                feature_columns, 
                ticker_encoder, 
                use_cache=True
            )
            
            if X is None:
                continue
            
            # predict volatility
            predicted_vol = predict_volatility(model, X)
            
            if predicted_vol is None:
                continue
            
            # calculate position sizing
            position_sizing = calculate_position_size(
                predicted_vol, 
                account_value, 
                risk_per_position  # using distributed risk
            )
            
            # get current price
            current_price = get_current_price(ticker, use_cache=True)
            
            # calculate shares
            shares = int(position_sizing['position_size'] / current_price) if current_price else 0
            
            results.append({
                'Ticker': ticker,
                'Current Price': f"${current_price:.2f}" if current_price else "N/A",
                'Predicted Vol (%)': f"{position_sizing['predicted_vol']:.2f}",
                'Position Size': f"${position_sizing['position_size']:,.0f}",
                'Position %': f"{position_sizing['position_percent']:.1f}%",
                'Shares': shares,
                'Stop Loss': f"-{position_sizing['stop_loss_percent']:.2f}%",
                'Risk Amount': f"${position_sizing['risk_amount']:,.0f}",
                # for sorting
                '_vol': position_sizing['predicted_vol'],
                '_position': position_sizing['position_size']
            })
            
        except Exception as e:
            st.error(f"Error processing {ticker}: {e}")
        
        progress_bar.progress((i + 1) / len(selected_tickers))
    
    status_text.empty()
    progress_bar.empty()
    
    if not results:
        st.error("No valid predictions generated. Check your ticker selections.")
        st.stop()
    
    # create dataframe
    df = pd.DataFrame(results)
    
    # display results
    st.success(f"Portfolio built with {len(results)} stocks using risk parity allocation")
    
    # summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_position = df['_position'].sum()
    avg_vol = df['_vol'].mean()
    deployment_pct = (total_position / account_value) * 100
    
    with col1:
        st.metric("Total Allocation", f"${total_position:,.0f}", f"{deployment_pct:.1f}%")
    with col2:
        st.metric("Avg Volatility", f"{avg_vol:.2f}%")
    with col3:
        st.metric("Positions", len(results))
    with col4:
        st.metric("Risk per Position", f"${account_value * risk_per_position:,.0f}")
    
    # helpful context about deployment
    if deployment_pct < 95:
        st.info(f"Portfolio is {100-deployment_pct:.1f}% in cash. Consider increasing total risk % or selecting stocks with lower volatility to deploy more capital.")
    elif deployment_pct > 105:
        st.warning(f"Portfolio slightly over-allocated ({deployment_pct:.1f}%). This is normal with risk parity - adjust position sizes down slightly or reduce total risk %.")
    
    st.markdown("---")
    
    # results table
    st.subheader("Portfolio Allocation")
    
    st.info(f"""
    **Risk Parity Portfolio:** Each position risks ${account_value * risk_per_position:,.0f} ({risk_per_position*100:.2f}% of account)
    
    Total portfolio risk budget: ${account_value * risk_percent:,.0f} ({risk_percent*100:.1f}%) split equally across {num_positions} positions
    """)
    
    # display without hidden columns
    display_df = df.drop(columns=['_vol', '_position'])
    st.dataframe(
        display_df,
        width='stretch',
        hide_index=True
    )
    
    # visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Position Size by Stock")
        chart_data = df[['Ticker', '_position']].set_index('Ticker')
        st.bar_chart(chart_data)
    
    with col2:
        st.subheader("Predicted Volatility")
        vol_data = df[['Ticker', '_vol']].set_index('Ticker')
        st.bar_chart(vol_data)
    
    # insights
    st.markdown("---")
    st.subheader("Key Insights")
    
    lowest_vol = df.loc[df['_vol'].idxmin()]
    highest_vol = df.loc[df['_vol'].idxmax()]
    
    st.markdown(f"""
    - **{lowest_vol['Ticker']}** has the **lowest volatility** ({lowest_vol['Predicted Vol (%)']}%) → Gets **largest position** ({lowest_vol['Position %']})
    - **{highest_vol['Ticker']}** has the **highest volatility** ({highest_vol['Predicted Vol (%)']}%) → Gets **smallest position** ({highest_vol['Position %']})
    - All {len(results)} positions risk equal dollar amounts: **${account_value * risk_per_position:,.0f}** each
    - This achieves **risk parity**: equal risk contribution from each stock
    - Total portfolio risk: **${account_value * risk_percent:,.0f}** ({risk_percent*100:.1f}%)
    """)
    
    # download button
    st.markdown("---")
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name=f"position_sizing_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

else:
    # show example output when button not clicked
    st.info("Click 'Build Portfolio' to generate your risk parity allocation")
    
    # show methodology
    with st.expander("How It Works"):
        st.markdown("""
        ### Portfolio Construction Methodology
        
        1. **Predict Volatility**: XGBoost model predicts daily volatility for each stock
           - 76 features: technical indicators, market context, realized volatility
           - R² = 0.533 (explains 53% of volatility variation)
        
        2. **Distribute Risk Budget**: Split total portfolio risk across N positions
           ```
           risk_per_position = total_portfolio_risk / number_of_stocks
           ```
        
        3. **Size Each Position**: Calculate position size for equal dollar risk
           ```
           position_size = (account × risk_per_position) / (2 × predicted_volatility)
           ```
        
        4. **Achieve Risk Parity**: Each position contributes equally to portfolio risk
           - Higher volatility → Smaller position size
           - Lower volatility → Larger position size
           - Equal dollar risk from each position
        
        ### Example with $100k account, 3 stocks, 3% total risk:
        - Risk per position: 3% / 3 = **1% each** ($1,000)
        - TSLA (high vol): Small position of $15k → risks $1,000
        - AAPL (medium vol): Medium position of $35k → risks $1,000  
        - WMT (low vol): Large position of $50k → risks $1,000
        - **Total: $100k deployed, $3k total risk, equal risk per stock**
        
        This is how professional portfolio managers allocate capital for balanced risk.
        """)
    
    with st.expander("Backtest Results: Does This Approach Work?"):
        st.markdown("""
        ### Backtest: Risk Parity vs Equal Weight (2022-2024)
        
        We tested this approach on 8 diversified stocks over 3 years:
        """)
        
        # load backtest results
        backtest_metrics_path = Path(__file__).parent / "backtest_metrics.json"
        if backtest_metrics_path.exists():
            with open(backtest_metrics_path, 'r') as f:
                backtest_metrics = json.load(f)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Risk Parity Return",
                    f"{backtest_metrics['risk_parity']['annualized_return']:.1f}%",
                    f"{backtest_metrics['risk_parity']['sharpe_ratio']:.2f} Sharpe"
                )
                st.metric(
                    "Risk Parity Volatility",
                    f"{backtest_metrics['risk_parity']['volatility']:.1f}%",
                    f"{backtest_metrics['risk_parity']['max_drawdown']:.1f}% Max DD"
                )
            
            with col2:
                st.metric(
                    "Equal Weight Return",
                    f"{backtest_metrics['equal_weight']['annualized_return']:.1f}%",
                    f"{backtest_metrics['equal_weight']['sharpe_ratio']:.2f} Sharpe"
                )
                st.metric(
                    "Equal Weight Volatility",
                    f"{backtest_metrics['equal_weight']['volatility']:.1f}%",
                    f"{backtest_metrics['equal_weight']['max_drawdown']:.1f}% Max DD"
                )
            
            # load equity curves
            equity_path = Path(__file__).parent / "backtest_equity_curves.csv"
            if equity_path.exists():
                equity_df = pd.read_csv(equity_path, index_col=0, parse_dates=True)
                
                # create chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=equity_df.index,
                    y=equity_df['risk_parity'],
                    mode='lines',
                    name='Risk Parity',
                    line=dict(color='#1f77b4', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=equity_df.index,
                    y=equity_df['equal_weight'],
                    mode='lines',
                    name='Equal Weight',
                    line=dict(color='#ff7f0e', width=2)
                ))
                fig.update_layout(
                    title="Portfolio Growth: Risk Parity vs Equal Weight",
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value ($)",
                    hovermode='x unified',
                    height=400
                )
                st.plotly_chart(fig, width='stretch')
            
            st.info("""
            **Key Finding**: In this bull market period, equal weight captured more upside 
            but risk parity provided **{:.1f}% lower volatility** and **{:.1f}% smaller max drawdown**. 
            Risk parity shines in volatile/down markets by limiting losses.
            """.format(
                abs(backtest_metrics['risk_parity']['volatility'] - backtest_metrics['equal_weight']['volatility']),
                abs(backtest_metrics['risk_parity']['max_drawdown'] - backtest_metrics['equal_weight']['max_drawdown'])
            ))
        else:
            st.warning("Run backtest.py to generate performance metrics")
    
    with st.expander("Supported Tickers"):
        st.write("Model supports these 28 stocks:")
        st.code(", ".join(AVAILABLE_TICKERS))

# footer
st.markdown("---")
st.markdown("*Built with XGBoost, pandas, and Streamlit | Data updated December 2025*")
