from flask import Flask, render_template, request
import pandas as pd
from datetime import datetime, timedelta, date
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX

import matplotlib
matplotlib.use("Agg")        # headless backend, perfect for Flask

import matplotlib.pyplot as plt
import io
import base64
import warnings

from forecasting import forecast_ftse

import base64
from io import BytesIO

import os



warnings.filterwarnings("ignore")

app = Flask(__name__)



#===============================================================================================================
#Ftse Forcasting
#---------------------------------------------------------------------------------------------------------------

def forecast_ftse(end_date_input):

    today = date.today()
    yesterday = today - timedelta(days=1)

    # -------------- download & prep -----------------
    raw_data = yf.download(
        tickers="^GSPC ^FTSE ^N225 ^GDAXI",
        start="2020-03-01",
        end=yesterday,
        interval="1d",
        progress=False
    )

    df_comp = raw_data["Close"][["^GSPC", "^GDAXI", "^FTSE", "^N225"]].copy()
    df_comp.columns = ["spx", "dax", "ftse", "nikkei"]
    df_comp = df_comp.asfreq("B").ffill()

    df_comp["ret_ftse"] = df_comp["ftse"].pct_change() * 100

    # -------------- SARIMAX model -------------------
    model   = SARIMAX(df_comp["ret_ftse"].iloc[1:],
                      order=(3, 0, 4), seasonal_order=(3, 0, 2, 5))
    results = model.fit(disp=False)

    # -------------- forecast series -----------------
    future_dates = pd.date_range(start=today, end=end_date_input, freq="B")
    predictions  = results.predict(start=len(df_comp),
                                   end=len(df_comp)+len(future_dates)-1)
    predictions.index = future_dates

    last_close = df_comp["ftse"].iloc[-1]
    ftse_level = last_close * (1 + predictions/100).cumprod()
    ftse_level.index = future_dates

    # -------------- build combined DataFrame --------
    forecast_df = pd.DataFrame({
        "Predicted_return_%": predictions,
        "Predicted FTSE Closing Price": ftse_level
    })
    forecast_df.index.name = "date"


    # ---------- 1) closing‑price plot ----------
    fig1, ax1 = plt.subplots(figsize=(12, 4), facecolor="black")
    ax1.set_facecolor("black")

    ftse_level.plot(ax=ax1, label="Forecast", color="#4fe8ff", linestyle="--")
    ax1.set_title("Predicted FTSE Closing Prices", color="white")
    ax1.legend(facecolor="black", edgecolor="none", labelcolor="white")
    ax1.grid(color="#555555", alpha=0.3)
    ax1.tick_params(colors="white")
    for spine in ax1.spines.values():
        spine.set_color("white")

    fig1.tight_layout()
    buf1 = io.BytesIO()
    plt.savefig(buf1, format="png", facecolor=fig1.get_facecolor())
    buf1.seek(0)
    plot_url_price = base64.b64encode(buf1.getvalue()).decode()
    plt.close(fig1)

    # ---------- 2) daily‑return plot ----------
    fig2, ax2 = plt.subplots(figsize=(12, 4), facecolor="black")
    ax2.set_facecolor("black")

    df_comp["ret_ftse"].plot(ax=ax2, label="Historical", color="#ffcf59")
    predictions.plot(ax=ax2, label="Forecast", color="#4fe8ff", linestyle="--")
    ax2.set_title("FTSE Daily Return Forecast", color="white")
    ax2.legend(facecolor="black", edgecolor="none", labelcolor="white")
    ax2.grid(color="#555555", alpha=0.3)
    ax2.tick_params(colors="white")
    for spine in ax2.spines.values():
        spine.set_color("white")

    fig2.tight_layout()
    buf2 = io.BytesIO()
    plt.savefig(buf2, format="png", facecolor=fig2.get_facecolor())
    buf2.seek(0)
    plot_url_return = base64.b64encode(buf2.getvalue()).decode()
    plt.close(fig2)

    # ----- return everything you need -----
    return forecast_df, plot_url_price, plot_url_return


   


#===============================================================================================================
#S&P Forcasting
#---------------------------------------------------------------------------------------------------------------
def forecast_spx(end_date_input):

    today = date.today()
    yesterday = today - timedelta(days=1)

    # -------------- download & prep -----------------
    raw_data = yf.download(
        tickers="^GSPC ^FTSE ^N225 ^GDAXI",
        start="2020-03-01",
        end=yesterday,
        interval="1d",
        progress=False
    )

    df_comp = raw_data["Close"][["^GSPC", "^GDAXI", "^FTSE", "^N225"]].copy()
    df_comp.columns = ["spx", "dax", "ftse", "nikkei"]
    df_comp = df_comp.asfreq("B").ffill()

    df_comp["ret_spx"] = df_comp["spx"].pct_change() * 100

    # -------------- SARIMAX model -------------------
    model   = SARIMAX(df_comp["ret_spx"].iloc[1:],
                      order=(3, 0, 4), seasonal_order=(3, 0, 2, 5))
    results = model.fit(disp=False)

    # -------------- forecast series -----------------
    future_dates = pd.date_range(start=today, end=end_date_input, freq="B")
    predictions  = results.predict(start=len(df_comp),
                                   end=len(df_comp)+len(future_dates)-1)
    predictions.index = future_dates

    last_close = df_comp["spx"].iloc[-1]
    spx_level = last_close * (1 + predictions/100).cumprod()
    spx_level.index = future_dates

    # -------------- build combined DataFrame --------
    forecast_df = pd.DataFrame({
        "Predicted_return_%": predictions,
        "Predicted S&P Closing Price": spx_level
    })
    forecast_df.index.name = "date"


    # ---------- 1) closing‑price plot ----------
    fig1, ax1 = plt.subplots(figsize=(12, 4), facecolor="black")
    ax1.set_facecolor("black")

    spx_level.plot(ax=ax1, label="Forecast", color="#4fe8ff", linestyle="--")
    ax1.set_title("Predicted S&P Closing Prices", color="white")
    ax1.legend(facecolor="black", edgecolor="none", labelcolor="white")
    ax1.grid(color="#555555", alpha=0.3)
    ax1.tick_params(colors="white")
    for spine in ax1.spines.values():
        spine.set_color("white")

    fig1.tight_layout()
    buf1 = io.BytesIO()
    plt.savefig(buf1, format="png", facecolor=fig1.get_facecolor())
    buf1.seek(0)
    plot_url_price = base64.b64encode(buf1.getvalue()).decode()
    plt.close(fig1)

    # ---------- 2) daily‑return plot ----------
    fig2, ax2 = plt.subplots(figsize=(12, 4), facecolor="black")
    ax2.set_facecolor("black")

    df_comp["ret_spx"].plot(ax=ax2, label="Historical", color="#ffcf59")
    predictions.plot(ax=ax2, label="Forecast", color="#4fe8ff", linestyle="--")
    ax2.set_title("S&P Daily Return Forecast", color="white")
    ax2.legend(facecolor="black", edgecolor="none", labelcolor="white")
    ax2.grid(color="#555555", alpha=0.3)
    ax2.tick_params(colors="white")
    for spine in ax2.spines.values():
        spine.set_color("white")

    fig2.tight_layout()
    buf2 = io.BytesIO()
    plt.savefig(buf2, format="png", facecolor=fig2.get_facecolor())
    buf2.seek(0)
    plot_url_return = base64.b64encode(buf2.getvalue()).decode()
    plt.close(fig2)

    # ----- return everything you need -----
    return forecast_df, plot_url_price, plot_url_return


   


#===============================================================================================================
#Dax Forcasting
#---------------------------------------------------------------------------------------------------------------
def forecast_dax(end_date_input):

    today = date.today()
    yesterday = today - timedelta(days=1)

    # -------------- download & prep -----------------
    raw_data = yf.download(
        tickers="^GSPC ^FTSE ^N225 ^GDAXI",
        start="2020-03-01",
        end=yesterday,
        interval="1d",
        progress=False
    )

    df_comp = raw_data["Close"][["^GSPC", "^GDAXI", "^FTSE", "^N225"]].copy()
    df_comp.columns = ["spx", "dax", "ftse", "nikkei"]
    df_comp = df_comp.asfreq("B").ffill()

    df_comp["ret_dax"] = df_comp["dax"].pct_change() * 100

    # -------------- SARIMAX model -------------------
    model   = SARIMAX(df_comp["ret_dax"].iloc[1:],
                      order=(3, 0, 4), seasonal_order=(3, 0, 2, 5))
    results = model.fit(disp=False)

    # -------------- forecast series -----------------
    future_dates = pd.date_range(start=today, end=end_date_input, freq="B")
    predictions  = results.predict(start=len(df_comp),
                                   end=len(df_comp)+len(future_dates)-1)
    predictions.index = future_dates

    last_close = df_comp["dax"].iloc[-1]
    dax_level = last_close * (1 + predictions/100).cumprod()
    dax_level.index = future_dates

    # -------------- build combined DataFrame --------
    forecast_df = pd.DataFrame({
        "Predicted_return_%": predictions,
        "Predicted DAX Closing Price": dax_level
    })
    forecast_df.index.name = "date"


    # ---------- 1) closing‑price plot ----------
    fig1, ax1 = plt.subplots(figsize=(12, 4), facecolor="black")
    ax1.set_facecolor("black")

    dax_level.plot(ax=ax1, label="Forecast", color="#4fe8ff", linestyle="--")
    ax1.set_title("Predicted DAX Closing Prices", color="white")
    ax1.legend(facecolor="black", edgecolor="none", labelcolor="white")
    ax1.grid(color="#555555", alpha=0.3)
    ax1.tick_params(colors="white")
    for spine in ax1.spines.values():
        spine.set_color("white")

    fig1.tight_layout()
    buf1 = io.BytesIO()
    plt.savefig(buf1, format="png", facecolor=fig1.get_facecolor())
    buf1.seek(0)
    plot_url_price = base64.b64encode(buf1.getvalue()).decode()
    plt.close(fig1)

    # ---------- 2) daily‑return plot ----------
    fig2, ax2 = plt.subplots(figsize=(12, 4), facecolor="black")
    ax2.set_facecolor("black")

    df_comp["ret_dax"].plot(ax=ax2, label="Historical", color="#ffcf59")
    predictions.plot(ax=ax2, label="Forecast", color="#4fe8ff", linestyle="--")
    ax2.set_title("DAX Daily Return Forecast", color="white")
    ax2.legend(facecolor="black", edgecolor="none", labelcolor="white")
    ax2.grid(color="#555555", alpha=0.3)
    ax2.tick_params(colors="white")
    for spine in ax2.spines.values():
        spine.set_color("white")

    fig2.tight_layout()
    buf2 = io.BytesIO()
    plt.savefig(buf2, format="png", facecolor=fig2.get_facecolor())
    buf2.seek(0)
    plot_url_return = base64.b64encode(buf2.getvalue()).decode()
    plt.close(fig2)

    # ----- return everything you need -----
    return forecast_df, plot_url_price, plot_url_return


   


#===============================================================================================================
#Nikkei Forcasting
#---------------------------------------------------------------------------------------------------------------
def forecast_nikkei(end_date_input):

    today = date.today()
    yesterday = today - timedelta(days=1)

    # -------------- download & prep -----------------
    raw_data = yf.download(
        tickers="^GSPC ^FTSE ^N225 ^GDAXI",
        start="2020-03-01",
        end=yesterday,
        interval="1d",
        progress=False
    )

    df_comp = raw_data["Close"][["^GSPC", "^GDAXI", "^FTSE", "^N225"]].copy()
    df_comp.columns = ["spx", "dax", "ftse", "nikkei"]
    df_comp = df_comp.asfreq("B").ffill()

    df_comp["ret_nikkei"] = df_comp["nikkei"].pct_change() * 100

    # -------------- SARIMAX model -------------------
    model   = SARIMAX(df_comp["ret_nikkei"].iloc[1:],
                      order=(3, 0, 4), seasonal_order=(3, 0, 2, 5))
    results = model.fit(disp=False)

    # -------------- forecast series -----------------
    future_dates = pd.date_range(start=today, end=end_date_input, freq="B")
    predictions  = results.predict(start=len(df_comp),
                                   end=len(df_comp)+len(future_dates)-1)
    predictions.index = future_dates

    last_close = df_comp["nikkei"].iloc[-1]
    nikkei_level = last_close * (1 + predictions/100).cumprod()
    nikkei_level.index = future_dates

    # -------------- build combined DataFrame --------
    forecast_df = pd.DataFrame({
        "Predicted_return_%": predictions,
        "Predicted NIKKEI Closing Price": nikkei_level
    })
    forecast_df.index.name = "date"


    # ---------- 1) closing‑price plot ----------
    fig1, ax1 = plt.subplots(figsize=(12, 4), facecolor="black")
    ax1.set_facecolor("black")

    nikkei_level.plot(ax=ax1, label="Forecast", color="#4fe8ff", linestyle="--")
    ax1.set_title("Predicted NIKKEI Closing Prices", color="white")
    ax1.legend(facecolor="black", edgecolor="none", labelcolor="white")
    ax1.grid(color="#555555", alpha=0.3)
    ax1.tick_params(colors="white")
    for spine in ax1.spines.values():
        spine.set_color("white")

    fig1.tight_layout()
    buf1 = io.BytesIO()
    plt.savefig(buf1, format="png", facecolor=fig1.get_facecolor())
    buf1.seek(0)
    plot_url_price = base64.b64encode(buf1.getvalue()).decode()
    plt.close(fig1)

    # ---------- 2) daily‑return plot ----------
    fig2, ax2 = plt.subplots(figsize=(12, 4), facecolor="black")
    ax2.set_facecolor("black")

    df_comp["ret_nikkei"].plot(ax=ax2, label="Historical", color="#ffcf59")
    predictions.plot(ax=ax2, label="Forecast", color="#4fe8ff", linestyle="--")
    ax2.set_title("NIKKEI Daily Return Forecast", color="white")
    ax2.legend(facecolor="black", edgecolor="none", labelcolor="white")
    ax2.grid(color="#555555", alpha=0.3)
    ax2.tick_params(colors="white")
    for spine in ax2.spines.values():
        spine.set_color("white")

    fig2.tight_layout()
    buf2 = io.BytesIO()
    plt.savefig(buf2, format="png", facecolor=fig2.get_facecolor())
    buf2.seek(0)
    plot_url_return = base64.b64encode(buf2.getvalue()).decode()
    plt.close(fig2)

    # ----- return everything you need -----
    return forecast_df, plot_url_price, plot_url_return


   


#=================================================================================================================
#=================================================================================================================
# Calling 
#=================================================================================================================


@app.route('/')
def index():
    return render_template('index.html')

#-------------------------------------------------------------------------------
@app.route('/ftse', methods=['GET', 'POST'])
def ftse_forecast():

    # initialise everything (needed for first GET load)
    forecast_df = price_plot = return_plot = None
    projected_value = return_pct = amount = forecast_end = None

    if request.method == 'POST':
        # ------------- get form inputs -------------
        end_date_str = request.form['end_date']
        amount       = float(request.form['amount'])

        y, m, d = map(int, end_date_str.split('-'))
        end_date = date(y, m, d)

        # ------------- run forecast ----------------
        forecast_df, price_plot, return_plot = forecast_ftse(end_date)

        # ------------- compute ROI -----------------
        start_lvl = forecast_df['Predicted FTSE Closing Price'].iloc[0]
        end_lvl   = forecast_df['Predicted FTSE Closing Price'].iloc[-1]

        growth_factor   = end_lvl / start_lvl
        projected_value = amount * growth_factor
        return_pct      = (growth_factor - 1) * 100
        forecast_end    = end_date.strftime('%Y‑%m‑%d')

    return render_template('ftse.html',
                           table = (forecast_df.to_html(classes="table table-bordered")
                                    if forecast_df is not None else None),
                           price_plot  = price_plot,
                           return_plot = return_plot,
                           amount          = amount,
                           projected_value = projected_value,
                           return_pct      = return_pct,
                           forecast_end    = forecast_end)

#---------------------------------------------------------------------------------------------------------------
@app.route('/spx', methods=['GET', 'POST'])
def spx_forecast():

    # initialise everything (needed for first GET load)
    forecast_df = price_plot = return_plot = None
    projected_value = return_pct = amount = forecast_end = None

    if request.method == 'POST':
        # ------------- get form inputs -------------
        end_date_str = request.form['end_date']
        amount       = float(request.form['amount'])

        y, m, d = map(int, end_date_str.split('-'))
        end_date = date(y, m, d)

        # ------------- run forecast ----------------
        forecast_df, price_plot, return_plot = forecast_spx(end_date)

        # ------------- compute ROI -----------------
        start_lvl = forecast_df['Predicted S&P Closing Price'].iloc[0]
        end_lvl   = forecast_df['Predicted S&P Closing Price'].iloc[-1]

        growth_factor   = end_lvl / start_lvl
        projected_value = amount * growth_factor
        return_pct      = (growth_factor - 1) * 100
        forecast_end    = end_date.strftime('%Y‑%m‑%d')

    return render_template('spx.html',
                           table = (forecast_df.to_html(classes="table table-bordered")
                                    if forecast_df is not None else None),
                           price_plot  = price_plot,
                           return_plot = return_plot,
                           amount          = amount,
                           projected_value = projected_value,
                           return_pct      = return_pct,
                           forecast_end    = forecast_end)

#------------------------------------------------------------------------------------------------------------------
@app.route('/dax', methods=['GET', 'POST'])
def dax_forecast():

    # initialise everything (needed for first GET load)
    forecast_df = price_plot = return_plot = None
    projected_value = return_pct = amount = forecast_end = None

    if request.method == 'POST':
        # ------------- get form inputs -------------
        end_date_str = request.form['end_date']
        amount       = float(request.form['amount'])

        y, m, d = map(int, end_date_str.split('-'))
        end_date = date(y, m, d)

        # ------------- run forecast ----------------
        forecast_df, price_plot, return_plot = forecast_dax(end_date)

        # ------------- compute ROI -----------------
        start_lvl = forecast_df['Predicted DAX Closing Price'].iloc[0]
        end_lvl   = forecast_df['Predicted DAX Closing Price'].iloc[-1]

        growth_factor   = end_lvl / start_lvl
        projected_value = amount * growth_factor
        return_pct      = (growth_factor - 1) * 100
        forecast_end    = end_date.strftime('%Y‑%m‑%d')

    return render_template('dax.html',
                           table = (forecast_df.to_html(classes="table table-bordered")
                                    if forecast_df is not None else None),
                           price_plot  = price_plot,
                           return_plot = return_plot,
                           amount          = amount,
                           projected_value = projected_value,
                           return_pct      = return_pct,
                           forecast_end    = forecast_end)

#-------------------------------------------------------------------------------------------------------------------
@app.route('/nikkei', methods=['GET', 'POST'])
def nikkei_forecast():

    # initialise everything (needed for first GET load)
    forecast_df = price_plot = return_plot = None
    projected_value = return_pct = amount = forecast_end = None

    if request.method == 'POST':
        # ------------- get form inputs -------------
        end_date_str = request.form['end_date']
        amount       = float(request.form['amount'])

        y, m, d = map(int, end_date_str.split('-'))
        end_date = date(y, m, d)

        # ------------- run forecast ----------------
        forecast_df, price_plot, return_plot = forecast_nikkei(end_date)

        # ------------- compute ROI -----------------
        start_lvl = forecast_df['Predicted NIKKEI Closing Price'].iloc[0]
        end_lvl   = forecast_df['Predicted NIKKEI Closing Price'].iloc[-1]

        growth_factor   = end_lvl / start_lvl
        projected_value = amount * growth_factor
        return_pct      = (growth_factor - 1) * 100
        forecast_end    = end_date.strftime('%Y‑%m‑%d')

    return render_template('nikkei.html',
                           table = (forecast_df.to_html(classes="table table-bordered")
                                    if forecast_df is not None else None),
                           price_plot  = price_plot,
                           return_plot = return_plot,
                           amount          = amount,
                           projected_value = projected_value,
                           return_pct      = return_pct,
                           forecast_end    = forecast_end)





#-----------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------
#Recommendation 
#=================================================================================================================
@app.route("/recommendation")
def recommendation():
    # Define tickers and metadata
    tickers = {
        "^GSPC": {"name": "S&P 500", "currency": "USD"},
        "^FTSE": {"name": "FTSE 100", "currency": "GBP"},
        "^N225": {"name": "Nikkei 225", "currency": "JPY"},
        "^GDAXI": {"name": "DAX", "currency": "EUR"}
    }

    # Download data
    data = yf.download(" ".join(tickers.keys()), start="2020-01-01", end="2025-02-28", interval="1d", group_by='ticker', auto_adjust=True)

    # Extract Close prices and Volume
    close_prices = pd.DataFrame({info["name"]: data[ticker]["Close"] for ticker, info in tickers.items()})
    volumes = pd.DataFrame({info["name"]: data[ticker]["Volume"] for ticker, info in tickers.items()})
    close_prices.dropna(inplace=True)
    volumes.dropna(inplace=True)

    # Market return
    market_return = ((close_prices.iloc[-1] - close_prices.iloc[0]) / close_prices.iloc[0] * 100).to_dict()

    # P/E ratio (mocked)
    pe_ratios = {"S&P 500": 25.1, "FTSE 100": 14.5, "Nikkei 225": 15.3, "DAX": 16.7}
    dividends = {"S&P 500": 1.5, "FTSE 100": 3.8, "Nikkei 225": 1.7, "DAX": 2.5}
    avg_volumes = volumes.mean().to_dict()
    macro_strength = {
        "S&P 500": 2.1 - 3.2,
        "FTSE 100": 0.6 - 3.8,
        "Nikkei 225": 1.0 - 2.7,
        "DAX": 0.8 - 3.0
    }

    ma_50 = close_prices.rolling(50).mean()
    ma_200 = close_prices.rolling(200).mean()
    sentiment = {index: 1 if ma_50[index].iloc[-1] > ma_200[index].iloc[-1] else 0 for index in close_prices.columns}

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

    # Score metrics
    score_return = normalize_scores(market_return)
    score_pe = normalize_scores(pe_ratios, reverse=True)
    score_dividends = normalize_scores(dividends)
    score_liquidity = normalize_scores(avg_volumes)
    score_macro = normalize_scores(macro_strength)
    score_sentiment = {k: v * 10 for k, v in sentiment.items()}

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

    reasons = {
        "S&P 500": "Strong historical returns, excellent liquidity, and solid macroeconomic fundamentals make the S&P 500 a great choice for growth investors.",
        "FTSE 100": "Low valuation (P/E), high dividend yield, and stable sectors make the FTSE 100 a solid pick for conservative or income-focused investors.",
        "Nikkei 225": "Moderate valuation and positive sentiment indicate potential upside in the Japanese market, especially if you're diversifying in Asia.",
        "DAX": "Balanced valuation and dividend yield, but lower growth and sentiment make it a neutral or secondary pick."
    }

    best_index = max(total_score, key=total_score.get)

    os.makedirs("static/images", exist_ok=True)

    # 📊 Bar chart
    plt.figure(figsize=(10, 5))
    plt.bar(total_score.keys(), total_score.values(), color='skyblue')
    plt.title("Total Investment Scores per Index")
    plt.ylabel("Score (Out of 10)")
    plt.grid(axis='y')
    plt.savefig("static/images/recommendation_plot.png")
    plt.close()

    # 📈 Close Price Graph
    close_prices.plot(figsize=(12, 6), title="Close Price Trends")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("static/images/close_price_plot.png")
    plt.close()

    # 🥧 Pie Chart for Return %
    plt.figure(figsize=(8, 6))
    plt.pie(market_return.values(), labels=market_return.keys(), autopct='%1.1f%%', startangle=140)
    plt.title("Market Return Contribution")
    plt.tight_layout()
    plt.savefig("static/images/return_pie_chart.png")
    plt.close()

   # return render_template("recommendation.html", best_index=best_index, reason=reasons[best_index])
    return render_template(
        "recommendation.html",
        best_index=best_index,
        reason=reasons[best_index],
        total_score=total_score,
        market_return=market_return,
        recommendation_plot_url="static/images/recommendation_plot.png",
        close_price_plot_url="static/images/close_price_plot.png",
        return_pie_chart_url="static/images/return_pie_chart.png"
    )




if __name__ == '__main__':
    app.run(debug=True)