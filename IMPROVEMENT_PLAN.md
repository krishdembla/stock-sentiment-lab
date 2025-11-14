# 🚀 Stock Sentiment Lab - Improvement Plan

## ✅ Step 1: Fixed Feature Importance Calculation (COMPLETED)

### What We Fixed:
1. **Feature Importance Extraction**: 
   - Now uses XGBoost's built-in `feature_importances_` attribute
   - Also extracts `gain`, `weight`, and `cover` importance types
   - Properly maps feature names to importance scores

2. **Data Fetching Improvements**:
   - Added automatic data fetching if cache doesn't exist
   - Created caching system to avoid repeated API calls
   - Added period parameter support (1y, 2y, 3y, 4y)

3. **Error Handling**:
   - Fixed ATR calculation error (handles insufficient data)
   - Better error messages and fallbacks

### Files Modified:
- `models/train_general_model.py`: Fixed feature importance calculation for both models
- `utils/features.py`: Fixed ATR calculation error
- `models/train_general_model.py`: Enhanced data preparation with caching

### Next Steps for Step 1:
- [ ] Retrain models to verify feature importances are now non-zero
- [ ] Review which features are most important
- [ ] Use feature importances to guide feature engineering improvements

---

## 📋 Step 2: Improve Direction Model (IN PROGRESS)

### Current Status:
- Direction model accuracy: **52.9%** (essentially random)
- Need to improve to at least **55-60%** to be useful

### Planned Improvements:

#### Option A: Try Different Target Horizons
- [ ] Test 1-day direction prediction
- [ ] Test 5-day direction prediction  
- [ ] Test 10-day direction prediction
- [ ] Compare which horizon is most predictable

#### Option B: Add Market Context Features
- [ ] Add SPY (market) performance features
- [ ] Add VIX (volatility index) features
- [ ] Add sector ETF performance (XLK, XLF, etc.)
- [ ] Add relative performance (stock vs market, stock vs sector)

#### Option C: Try Different Algorithms
- [ ] Logistic Regression with regularization
- [ ] Random Forest Classifier
- [ ] LightGBM Classifier
- [ ] Ensemble of multiple models

#### Option D: Feature Engineering
- [ ] Add momentum features (trend strength)
- [ ] Add regime indicators (bull/bear market)
- [ ] Add volume-price divergence indicators
- [ ] Normalize/scale features properly

### Success Criteria:
- Direction accuracy > 55%
- Precision and Recall both > 60%
- F1 score > 60%

---

## 📋 Step 3: Expand Ticker Coverage (PENDING)

### Current Status:
- Only 5 tickers: AAPL, AMZN, MSFT, NVDA, TSLA
- Goal: All 28 tickers across sectors

### Planned Actions:
- [ ] Retrain models with all 28 tickers
- [ ] Ensure data quality across all tickers
- [ ] Handle ticker-specific edge cases
- [ ] Update ticker encoder to include all tickers

### Tickers to Add:
- Tech: GOOG, META, CRM, ADBE, INTC
- Financials: JPM, V, MA, BAC, WFC
- Healthcare: JNJ, UNH, PFE, MRK
- Consumer: WMT, PG, MCD, COST, HD, NKE
- Energy: XOM, CVX, COP

---

## 📋 Step 4: Fix Data Fetching Issues (PENDING)

### Current Issues:
- Rate limiting from Yahoo Finance
- ATR errors with insufficient data (FIXED)
- Network timeouts

### Planned Solutions:
- [ ] Add retry logic with exponential backoff
- [ ] Implement request throttling
- [ ] Add better error handling for API failures
- [ ] Consider alternative data sources (Alpha Vantage, etc.)
- [ ] Improve caching strategy

---

## 📋 Step 5: Feature Engineering Improvements (PENDING)

### Planned Enhancements:
- [ ] Feature scaling/normalization (StandardScaler)
- [ ] Remove redundant features
- [ ] Add interaction features
- [ ] Add lagged features
- [ ] Add rolling statistics (mean, std, min, max)

---

## 📊 Progress Tracking

### Completed:
- ✅ Fixed feature importance calculation
- ✅ Enhanced data fetching with caching
- ✅ Fixed ATR calculation error

### In Progress:
- 🔄 Improving direction model

### Pending:
- ⏳ Expand ticker coverage
- ⏳ Fix data fetching issues
- ⏳ Feature engineering improvements

---

## 🎯 Immediate Next Actions

1. **Retrain models** with fixed feature importance to verify it works
2. **Review feature importances** to see which features matter most
3. **Test different direction targets** (1-day, 5-day) to find best horizon
4. **Add market context features** to improve direction model

---

## 📝 Notes

- Rate limiting is currently blocking full retraining
- Consider training during off-peak hours or using cached data
- Feature importance fix should reveal which features are actually being used
- Direction model improvements are critical for user value

