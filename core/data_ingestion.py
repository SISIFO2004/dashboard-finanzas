import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
import yfinance as yf
import requests_cache
import streamlit as st

# Usamos una sesión más anónima para evitar bloqueos
session = requests_cache.CachedSession('mercado_cache', expire_after=86400)
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
})

@st.cache_data(ttl=43200, show_spinner=False)
def get_macro_context():
    try:
        tnx = yf.download("^TNX", period="5d", progress=False)
        vix = yf.download("^VIX", period="5d", progress=False)
        rf_rate = float(tnx['Close'].iloc[-1]) / 100.0 if not tnx.empty else 0.045
        vix_level = float(vix['Close'].iloc[-1]) if not vix.empty else 20.0
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
        if not df_temp.empty:
            # Manejo de la nueva estructura MultiIndex de yfinance
            if isinstance(df_temp.columns, pd.MultiIndex):
                df_temp = df_temp.xs(yf_ticker, level=1, axis=1) if yf_ticker in df_temp.columns.get_level_values(1) else df_temp.droplevel(1, axis=1)
            
            if 'Close' in df_temp.columns:
                df_clean = df_temp[['Close']].copy().dropna()
                if len(df_clean) > 50:
                    df_clean['Source'] = source_name
                    return df_clean
        return pd.DataFrame()

    # 1. Intento primario: yf.download (Más resistente a bloqueos en la nube)
    try:
        df = yf.download(yf_ticker, period="2y", progress=False)
        valid_df = validate_df(df, 'Yahoo Finance (Download)')
        if not valid_df.empty: return valid_df
    except Exception: pass

    # 2. Intento secundario: Ticker API
    try:
        stock = yf.Ticker(yf_ticker, session=session)
        df = stock.history(period="2y")
        valid_df = validate_df(df, 'Yahoo Finance (Ticker)')
        if not valid_df.empty: return valid_df
    except Exception: pass

    raise ValueError("Proveedores caídos o bloqueados por IP")
