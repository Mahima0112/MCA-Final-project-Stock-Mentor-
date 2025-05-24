import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from datetime import date, timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX

def forecast_ftse(end_date_input):
    today = date.today()
    yesterday = today - timedelta(days=1)

    # Download stock data
    raw_data = yf.download(
        tickers="^GSPC ^FTSE ^N225 ^GDAXI",
        start="2020-03-01",
        end=yesterday,
        interval="1d"
    )

    # Check for NaN values and drop if any
    df_comp = raw_data['Close'].copy()
    df_comp['spx'] = df_comp['^GSPC']
    df_comp['dax'] = df_comp['^GDAXI']
    df_comp['ftse'] = df_comp['^FTSE']
    df_comp['nikkei'] = df_comp['^N225']
    df_comp = df_comp[['spx', 'dax', 'ftse', 'nikkei']]
    df_comp = df_comp.asfreq('b').fillna(method='ffill')

    # Calculate returns for FTSE
    df_comp['ret_ftse'] = df_comp['ftse'].pct_change(1) * 100

    # Check for NaN values in returns and drop them
    df_comp = df_comp.dropna(subset=['ret_ftse'])

    # Normalize returns to avoid large numbers
    df_comp['ret_ftse'] = (df_comp['ret_ftse'] - df_comp['ret_ftse'].mean()) / df_comp['ret_ftse'].std()

    # Fit SARIMAX model (simplified orders to reduce instability)
    try:
        model = SARIMAX(df_comp['ret_ftse'][1:], order=(1, 0, 1), seasonal_order=(1, 0, 1, 5))
        results = model.fit(method='powell', maxiter=1000, disp=True)  # Using Powell method and increasing iterations
    except Exception as e:
        print(f"Error in SARIMAX model fitting: {e}")
        return None, None

    # Generate future dates for prediction
    future_dates = pd.date_range(start=today, end=end_date_input, freq='B')
    
    # Make predictions
    predictions = results.predict(start=len(df_comp), end=len(df_comp) + len(future_dates) - 1)
    predictions.index = future_dates

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 4))
    df_comp['ret_ftse'].plot(ax=ax, label='Historical', color='blue')
    predictions.plot(ax=ax, label='Forecast', color='red', linestyle='--')
    ax.legend()
    ax.set_title("FTSE Daily Return Forecast")
    fig.tight_layout()

    # Save plot to a BytesIO object and encode in base64
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
    plt.close()

    # Prepare data for the JSON output
    json_df = {
        "dates": [str(date) for date in predictions.index],
        "predictions": predictions.tolist()
    }

    return json_df, plot_url
