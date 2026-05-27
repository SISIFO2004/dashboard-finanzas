import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
import yfinance as yf
from yahooquery import Ticker as YQTicker
import requests_cache
import streamlit as st

session = requests_cache.CachedSession('mercado_cache', expire_after=86400)
session.headers.update({'User-Agent': 'Mozilla/5.0'})

@st.cache_data(ttl=43200, show_spinner=False)
def get_macro_context():
    try:
        tnx = yf.Ticker("^TNX", session=session).history(period="5d")
        vix = yf.Ticker("^VIX", session=session).history(period="5d")
        rf_rate = tnx['Close'].iloc[-1] / 100.0 if not tnx.empty else 0.045
        vix_level = vix['Close'].iloc[-1] if not vix.empty else 20.0
        return rf_rate, vix_level
    except Exception:
        return 0.045, 20.0 

def generate_synthetic_data(ticker: str, days: int = 500, trading_days: int = 252) -> pd.DataFrame:
    np.random.seed(42)
    S0, mu, sigma, dt = 150.0, 0.08, 0.20, 1 / trading_days   
    prices = [S0]
    for _ in range(1, days):
        prices.append(prices[-1] * np.exp((mu - (sigma**2) / 2) * dt + sigma * np.sqrt(dt) * np.random.normal()))
    dates = pd.date_range(end=datetime.date.today(), periods=days, freq='D' if trading_days == 365 else 'B')
    return pd.DataFrame({'Close': prices, 'Source': 'SINTÉTICO_EMERGENCIA'}, index=dates)

@st.cache_data(ttl=3600, show_spinner=False)
def load_financial_data(ticker: str, tiingo_key: str = "") -> pd.DataFrame:
    yf_ticker = ticker
    def validate_df(df_temp, source_name):
        if not df_temp.empty and 'Close' in df_temp.columns:
            if df_temp['Close'].isnull().any(): df_temp = df_temp.dropna()
            if len(df_temp) > 50: 
                df_temp['Source'] = source_name
                return df_temp
        return pd.DataFrame()

    try:
        stock = yf.Ticker(yf_ticker, session=session)
        df = stock.history(period="2y")
        if not df.empty:
            col_precio = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
            valid_df = validate_df(df[[col_precio]].rename(columns={col_precio: 'Close'}), 'Yahoo Finance')
            if not valid_df.empty: return valid_df
    except Exception: pass
    raise ValueError("Proveedores caídos o datos corruptos")
