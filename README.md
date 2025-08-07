# LSTM-w-Sentiment-Analysis

## 📈 Tesla Stock Prediction Using LSTM + Twitter Sentiment Analysis

This project combines a Long Short-Term Memory (LSTM) neural network with sentiment analysis from Twitter to enhance the prediction of **Tesla (TSLA)** stock prices. Sentiment derived from public opinion adds an edge to traditional time-series forecasting.

---

## 🔍 Project Overview

1. **LSTM Model**  
   A deep learning model trained on daily historical TSLA stock prices from 2010 to 2023, using a 60-day lookback window.

2. **Twitter Sentiment Analysis**  
   Sentiment is extracted using a BERT-based transformer (`nlptown/bert-base-multilingual-uncased-sentiment`), applied to tweets mentioning Tesla. Scores are in the range [0, 1] and are averaged daily.

3. **Sentiment-Based Adjustment**  
   Final predictions from the LSTM are modified based on **daily sentiment**, using a **confidence-weighted multiplier**. This boosts or dampens predictions when sentiment is strongly positive or negative.

---

## 📊 Model Evaluation Results

| Metric        | LSTM Only | LSTM + Sentiment |
|---------------|-----------|------------------|
| **RMSE**      | 11.612    |   **11.541**     |
| **R² Score**  | 0.8353    |   **0.8373**     |

>  The sentiment-adjusted model showed improved accuracy on unseen data.

---

## Sentiment Multiplier Logic

- A **neutral sentiment band** is defined from **0.4 to 0.6**.
- Multiplier is only applied if sentiment score falls **outside** this range.
- Formula:

```python
alpha = 0.2
threshold = 0.1
if score < (0.5 - threshold) or score > (0.5 + threshold):
    multiplier = 1 + alpha * (score - 0.5)
else:
    multiplier = 1  # No adjustment
```

---

## Works Even Better on Weaker Models

When a less complex model was used:

| Metric       | Weaker Model | With Sentiment | Improvement |
|--------------|--------------|----------------|-------------|
| **R² Score** |    0.770     |   **0.785**    |  **+0.015** |

> Sentiment correction had **higher impact** when the base model had **more room for improvement**.

---

## Future Work
### Cross-Asset Signals
Incorporate price trends of competitors or supplier/parent companies.

### Derived Features
Add volatility, moving averages, or trading volume patterns.

### Multi-Input Model
Train an LSTM that jointly learns from both price data and daily sentiment as input features.

### News Sentiment
Integrate news articles to enrich sentiment data.


