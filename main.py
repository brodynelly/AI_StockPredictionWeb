import streamlit as st
from datetime import date
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# -------------------------------
# 1. App Configuration
# -------------------------------
st.set_page_config(page_title="AI Stock Forecasting", layout="centered")
st.title("AI Stock Forecasting App")

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

stocks = ("AAPL", "GOOG", "MSFT", "GME")
selected_stock = st.selectbox("Select a stock for prediction", stocks)

n_years = st.slider("Years of prediction", 1, 4)
period = n_years * 365

# -------------------------------
# 2. Cache‑Clear Control
# -------------------------------
if st.button("Clear cached stock data"):
    st.cache_data.clear()
    st.rerun()

# -------------------------------
# 3. Data Loader (fixed Date column)
# -------------------------------
@st.cache_data(show_spinner=False)
def load_data(ticker: str) -> pd.DataFrame:
    try:
        # Download data from yfinance
        df = yf.download(ticker, START, TODAY, progress=False)

        if df.empty:
            raise ValueError(f"No data found for ticker {ticker}.")

        # Print debug info about the dataframe
        print(f"Downloaded data columns: {df.columns.tolist()}")
        print(f"Index type: {type(df.index)}")

        # Handle MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            print("MultiIndex columns detected, flattening...")
            # For a ticker, yfinance returns MultiIndex columns like (Open, AAPL), (Close, AAPL)
            # We'll flatten to just use 'Open', 'Close', etc.
            df.columns = [col[0] for col in df.columns]
            print(f"New columns: {df.columns.tolist()}")

        # Ensure a proper Date column
        df['Date'] = df.index
        df = df.reset_index(drop=True)

        # Clean and validate dates
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        invalid_dates = df['Date'].isna().sum()
        if invalid_dates > 0:
            print(f"Warning: {invalid_dates} invalid dates found and will be dropped")
        df = df.dropna(subset=['Date'])  # drop any bad dates

        # Clean and validate Close prices
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        invalid_closes = df['Close'].isna().sum()
        if invalid_closes > 0:
            print(f"Warning: {invalid_closes} invalid Close prices found and will be dropped")
        df = df.dropna(subset=['Close'])  # drop non-numeric closes

        # Final validation
        if df.empty:
            raise ValueError("After cleaning, no valid data remains.")

        return df

    except Exception as e:
        # More detailed error handling
        import traceback
        print(f"Error in load_data: {str(e)}")
        print(traceback.format_exc())
        raise ValueError(f"Failed to load data for {ticker}: {str(e)}")

# Attempt to load
try:
    data = load_data(selected_stock)
    if 'Date' not in data.columns:
        st.error(f"Data load failed: Missing 'Date' column in the data")
        st.write("Available columns:", data.columns.tolist())
        st.stop()

    if len(data) == 0:
        st.error(f"No data available for {selected_stock}")
        st.stop()

    st.success(f"Loaded {len(data)} rows: {data['Date'].min().date()} → {data['Date'].max().date()}")
except Exception as e:
    st.error(f"Data load failed: {e}")
    st.write("Try clearing the cache and reloading the app.")
    st.stop()

# -------------------------------
# 4. Raw Data Preview & Date Range
# -------------------------------
st.subheader("🔍 Raw Data Preview")
st.write(data.tail())

st.markdown(f"**Date Range:** {data['Date'].min().date()} → {data['Date'].max().date()}")

# -------------------------------
# 5. Plot Historical Prices
# -------------------------------
def plot_historical(df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Open'],  name="Open"))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Close"))
    fig.update_layout(
        title=f"Historical Prices: {selected_stock}",
        xaxis_rangeslider_visible=True,
        xaxis=dict(range=[START, TODAY])
    )
    st.plotly_chart(fig, use_container_width=True)

plot_historical(data)

# -------------------------------
# 6. Prepare & Fit Prophet
# -------------------------------
st.subheader("Forecasting Model")

# this will fit the training data
df_train = data[['Date', 'Close']].copy()
df_train.columns = ['ds', 'y']
df_train = df_train.dropna(subset=['ds', 'y'])  # ensure no NaNs

model = Prophet()
model.fit(df_train)

# -------------------------------
# 7. Generate & Display Forecast
# -------------------------------
future   = model.make_future_dataframe(periods=period)
forecast = model.predict(future)

st.subheader("Forecasted Values")
st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

st.subheader(f"{selected_stock} Forecast Plot ({n_years} year{'s' if n_years>1 else ''})")
fig_forecast = plot_plotly(model, forecast)
st.plotly_chart(fig_forecast, use_container_width=True)

st.subheader("Forecast Components")
fig_components = model.plot_components(forecast)
st.write(fig_components)
