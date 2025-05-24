
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Define tickers and metadata
tickers = {
    "^GSPC": {"name": "S&P 500", "currency": "USD"},
    "^FTSE": {"name": "FTSE 100", "currency": "GBP"},
    "^N225": {"name": "Nikkei 225", "currency": "JPY"},
    "^GDAXI": {"name": "DAX", "currency": "EUR"}
}

# Download stock data
data = yf.download(" ".join(tickers.keys()), start="2020-01-01", end="2025-02-28", interval="1d", group_by='ticker', auto_adjust=True)

# Extract close and volume
close_prices = pd.DataFrame({
    info["name"]: data[ticker]["Close"] for ticker, info in tickers.items()
})
volumes = pd.DataFrame({
    info["name"]: data[ticker]["Volume"] for ticker, info in tickers.items()
})

# Clean data
close_prices.dropna(inplace=True)
volumes.dropna(inplace=True)

# Metric 1: Market return
market_return = ((close_prices.iloc[-1] - close_prices.iloc[0]) / close_prices.iloc[0] * 100).to_dict()

# Metric 2: P/E ratio (mocked)
pe_ratios = {
    "S&P 500": 25.1,
    "FTSE 100": 14.5,
    "Nikkei 225": 15.3,
    "DAX": 16.7
}

# Metric 3: Dividend Yield %
dividends = {
    "S&P 500": 1.5,
    "FTSE 100": 3.8,
    "Nikkei 225": 1.7,
    "DAX": 2.5
}

# Metric 4: Liquidity (avg volume)
avg_volumes = volumes.mean().to_dict()

# Metric 5: Macro strength (GDP - Inflation, mocked)
macro_strength = {
    "S&P 500": 2.1 - 3.2,
    "FTSE 100": 0.6 - 3.8,
    "Nikkei 225": 1.0 - 2.7,
    "DAX": 0.8 - 3.0
}

# Metric 6: Sentiment (Bullish if 50MA > 200MA)
ma_50 = close_prices.rolling(50).mean()
ma_200 = close_prices.rolling(200).mean()
sentiment = {
    index: 1 if ma_50[index].iloc[-1] > ma_200[index].iloc[-1] else 0
    for index in close_prices.columns
}

# Normalize and score out of 10
def normalize_scores(values, reverse=False):
    vals = list(values.values())
    min_v, max_v = min(vals), max(vals)
    scores = {}
    for k, v in values.items():
        if max_v == min_v:
            scores[k] = 5
        else:
            scores[k] = 10 * (max_v - v) / (max_v - min_v) if reverse else 10 * (v - min_v) / (max_v - min_v)
    return scores

# Score each metric
score_return = normalize_scores(market_return)
score_pe = normalize_scores(pe_ratios, reverse=True)
score_dividends = normalize_scores(dividends)
score_liquidity = normalize_scores(avg_volumes)
score_macro = normalize_scores(macro_strength)
score_sentiment = {k: v * 10 for k, v in sentiment.items()}

# Weighted scoring
indices = list(close_prices.columns)
total_score = {}
for index in indices:
    total_score[index] = (
        0.25 * score_return[index] +
        0.15 * score_pe[index] +
        0.15 * score_dividends[index] +
        0.15 * score_liquidity[index] +
        0.15 * score_macro[index] +
        0.15 * score_sentiment[index]
    )

# Reason to invest
reasons = {
    "S&P 500": "Strong historical returns, excellent liquidity, and solid macroeconomic fundamentals make the S&P 500 a great choice for growth investors.",
    "FTSE 100": "Low valuation (P/E), high dividend yield, and stable sectors make the FTSE 100 a solid pick for conservative or income-focused investors.",
    "Nikkei 225": "Moderate valuation and positive sentiment indicate potential upside in the Japanese market, especially if you're diversifying in Asia.",
    "DAX": "Balanced valuation and dividend yield, but lower growth and sentiment make it a neutral or secondary pick."
}

# Find the best index
best_index = max(total_score, key=total_score.get)

# Final Output
print("\n=== 📊 Final Investment Recommendation ===")
print(f"✅ Invest in: **{best_index}**")
print(f"💡 Reason: {reasons[best_index]}")

print("\n--- 🔢 Score Breakdown ---")
for idx in sorted(total_score, key=total_score.get, reverse=True):
    print(f"{idx}: {total_score[idx]:.2f}")

# Optional: Visualize
plt.figure(figsize=(10, 5))
plt.bar(total_score.keys(), total_score.values(), color='skyblue')
plt.title("Total Investment Scores per Index")
plt.ylabel("Score (Out of 10)")
plt.grid(axis='y')
plt.savefig("static/images/recommendation_plot.png")
plt.close()
