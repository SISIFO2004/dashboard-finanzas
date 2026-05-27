import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from yahooquery import Ticker as YQTicker
import requests
import requests_cache
import datetime
from datetime import timedelta
import plotly.graph_objects as go
from fpdf import FPDF
from numba import njit 

# ==============================================================================
# CONFIGURACIÓN DE MEMORIA CACHÉ
# ==============================================================================
session = requests_cache.CachedSession('mercado_cache', expire_after=86400)
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36'
})

# ==============================================================================
# FASE 2: EXTRACCIÓN MACROECONÓMICA DINÁMICA
# ==============================================================================

@st.cache_data(ttl=43200, show_spinner=False) # Caché de 12 horas para datos macro
def get_macro_context():
    """Extrae las variables de entorno: Tasa Libre de Riesgo (^TNX) e Índice de Miedo (^VIX)"""
    try:
        tnx = yf.Ticker("^TNX", session=session).history(period="5d")
        vix = yf.Ticker("^VIX", session=session).history(period="5d")
        
        # El bono cotiza en porcentaje (ej. 4.5), lo convertimos a decimal (0.045)
        rf_rate = tnx['Close'].iloc[-1] / 100.0 if not tnx.empty else 0.045
        vix_level = vix['Close'].iloc[-1] if not vix.empty else 20.0
        return rf_rate, vix_level
    except Exception:
        return 0.045, 20.0 # Valores de contingencia (Fallback)

# ==============================================================================
# FASE 0 & 1: INGESTA RESILIENTE
# ==============================================================================

def generate_synthetic_data(ticker: str, days: int = 500, trading_days: int = 252) -> pd.DataFrame:
    np.random.seed(42)
    S0 = 150.0  
    mu = 0.08   
    sigma = 0.20 
    dt = 1 / trading_days   
    prices = [S0]
    for _ in range(1, days):
        Z = np.random.normal()
        S_t = prices[-1] * np.exp((mu - (sigma**2) / 2) * dt + sigma * np.sqrt(dt) * Z)
        prices.append(S_t)
    end_date = datetime.date.today()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, periods=days, freq='D' if trading_days == 365 else 'B')
    
    if len(dates) > len(prices): dates = dates[:len(prices)]
    elif len(prices) > len(dates): prices = prices[:len(dates)]
        
    df = pd.DataFrame(index=dates)
    df['Close'] = prices
    df['Source'] = 'SINTÉTICO_EMERGENCIA'
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def load_financial_data(ticker: str, tiingo_key: str = "") -> pd.DataFrame:
    yf_ticker = ticker
    t_ticker = ticker.replace("-USD", "") if "USD" in ticker else ticker

    def validate_df(df_temp, source_name):
        if not df_temp.empty and 'Close' in df_temp.columns:
            if df_temp['Close'].isnull().any(): df_temp = df_temp.dropna()
            if len(df_temp) > 50: 
                df_temp['Source'] = source_name
                return df_temp
        return pd.DataFrame()

    if tiingo_key:
        try:
            url = f"https://api.tiingo.com/tiingo/crypto/prices?tickers={t_ticker}usd&resampleFreq=1day&token={tiingo_key}" if "USD" in ticker else f"https://api.tiingo.com/tiingo/daily/{t_ticker}/prices?startDate={(datetime.datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')}&token={tiingo_key}"
            response = session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if len(data) > 0:
                    df = pd.DataFrame(data[0]['priceData']) if "USD" in ticker else pd.DataFrame(data)
                    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
                    df = df.set_index('date').rename(columns={'close': 'Close'}).sort_index().tail(504)
                    cache_status = "Caché Local" if response.from_cache else "Internet"
                    valid_df = validate_df(df[['Close']].copy(), f'Tiingo API ({cache_status})')
                    if not valid_df.empty: return valid_df
        except Exception: pass

    try:
        yq = YQTicker(yf_ticker, session=session)
        df = yq.history(period="2y")
        if not df.empty and isinstance(df, pd.DataFrame) and 'close' in df.columns:
            df = df.reset_index()
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
            df = df.set_index('date').rename(columns={'close': 'Close'})
            valid_df = validate_df(df[['Close']].copy(), 'YahooQuery')
            if not valid_df.empty: return valid_df
    except Exception: pass

    try:
        stock = yf.Ticker(yf_ticker, session=session)
        df = stock.history(period="2y")
        if not df.empty:
            col_precio = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
            if col_precio in df.columns:
                df = df[[col_precio]].rename(columns={col_precio: 'Close'})
                valid_df = validate_df(df.copy(), 'Yahoo Finance (Ajustado)')
                if not valid_df.empty: return valid_df
    except Exception: pass

    raise ValueError("Proveedores caídos o datos corruptos")

# ==============================================================================
# FASE 2: MOTOR DE INGENIERÍA DE CARACTERÍSTICAS (FEATURE ENGINEERING)
# ==============================================================================

def engineer_ml_features(df: pd.DataFrame, trading_days: int) -> pd.DataFrame:
    """
    Expande el DataFrame con variables técnicas, estadísticas y de momento.
    Esta es la matriz de entrenamiento para el futuro modelo de Machine Learning.
    """
    df = df.copy()
    
    # 1. Retornos Clásicos
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # 2. Momentos Estadísticos de Distribución (Riesgo de Cola)
    df['Rolling_Vol_20'] = df['Returns'].rolling(window=20).std() * np.sqrt(trading_days)
    df['Realized_Vol_60'] = df['Returns'].rolling(window=60).std() * np.sqrt(trading_days)
    df['Skewness_60'] = df['Returns'].rolling(window=60).skew()  # Asimetría
    df['Kurtosis_60'] = df['Returns'].rolling(window=60).kurt()  # Colas pesadas
    
    # 3. Indicadores de Dinámica de Caos / Momento
    # RSI (Relative Strength Index - 14 periodos)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    return df.dropna()

# ==============================================================================
# FILTROS ESTOCÁSTICOS (KALMAN Y LOG-VOL)
# ==============================================================================

def apply_kalman_filter_dynamic(returns_array, divisor_inercia):
    n = len(returns_array)
    R = np.var(returns_array)
    if R == 0: return np.full(n, returns_array[-1])
    Q = R / float(divisor_inercia) 
    x_hat, P = np.zeros(n), np.zeros(n)
    x_hat[0] = returns_array[0]
    P[0] = 1.0
    for k in range(1, n):
        x_hat_minus = x_hat[k-1]
        P_minus = P[k-1] + Q
        K = P_minus / (P_minus + R)
        x_hat[k] = x_hat_minus + K * (returns_array[k] - x_hat_minus)
        P[k] = (1 - K) * P_minus
    return x_hat

def calculate_kalman_score(returns_array, divisor_inercia):
    kalman_states = apply_kalman_filter_dynamic(returns_array, divisor_inercia)
    raw_vol = np.std(returns_array)
    kalman_vol = np.std(kalman_states)
    if raw_vol == 0: return float('inf'), 0, 0, 0
    svr = kalman_vol / raw_vol
    penalty_svr = abs(svr - 0.30) * 30
    residuals = returns_array - kalman_states
    suma_cuadrados_res = np.sum(residuals**2)
    dw_stat = np.sum(np.diff(residuals)**2) / suma_cuadrados_res if suma_cuadrados_res != 0 else 0
    return penalty_svr + abs(dw_stat - 2.0) * 20, svr, dw_stat

def optimize_kalman_filter(returns_array):
    candidatos = [10, 20, 50, 100, 200, 300, 500]
    mejor_divisor, menor_error = 100, float('inf')
    for divisor in candidatos:
        error, _, _ = calculate_kalman_score(returns_array, divisor)
        if error < menor_error: menor_error, mejor_divisor = error, divisor
    return mejor_divisor

def apply_stochastic_volatility_filter(returns_array, q_variance=0.1):
    n = len(returns_array)
    y_star = np.log(returns_array**2 + 1e-8)
    R_vol = 4.93 
    Q_vol = q_variance 
    h_hat, P_vol = np.zeros(n), np.zeros(n) 
    h_hat[0] = np.mean(y_star) if n > 0 else 0.0
    P_vol[0] = 1.0
    for k in range(1, n):
        h_minus = h_hat[k-1]
        P_minus = P_vol[k-1] + Q_vol
        K = P_minus / (P_minus + R_vol)
        h_hat[k] = h_minus + K * (y_star[k] - h_minus)
        P_vol[k] = (1 - K) * P_minus
    return np.clip(np.exp(h_hat / 2.0), 0.0001, 1.0)

# ==============================================================================
# SIMULACIÓN ACELERADA (NUMBA JIT)
# ==============================================================================

@njit
def _jit_montecarlo_loop(price_paths, v_t, mu_t, days, simulations, dt, kappa_v, long_term_sigma, vol_v, Z_vol, kappa_mu, long_term_mu, vol_mu, Z_mu, Z_price, jump_returns):
    for t in range(days):
        for s in range(simulations):
            v_new = np.abs(v_t[s] + kappa_v * (long_term_sigma - v_t[s]) * dt + vol_v * np.sqrt(v_t[s] * dt) * Z_vol[t, s])
            mu_new = mu_t[s] + kappa_mu * (long_term_mu - mu_t[s]) * dt + vol_mu * np.sqrt(dt) * Z_mu[t, s]
            gbm_returns = (mu_new - 0.5 * v_new**2) * dt + v_new * np.sqrt(dt) * Z_price[t, s]
            total_returns = gbm_returns + jump_returns[t, s]
            
            price_paths[t+1, s] = price_paths[t, s] * np.exp(total_returns)
            v_t[s] = v_new
            mu_t[s] = mu_new
    return price_paths

def run_montecarlo_advanced_stochastic(S0, current_mu, long_term_mu, current_sigma, long_term_sigma, days, simulations, lambda_j, mu_j, sigma_j, trading_days):
    dt = 1 / trading_days  
    kappa_v, vol_v = 5.0, 0.2  
    kappa_mu, vol_mu = 2.0, 0.1   
    
    price_paths = np.zeros((days + 1, simulations))
    price_paths[0] = S0
    v_t = np.full(simulations, current_sigma)
    mu_t = np.full(simulations, current_mu)
    
    Z_vol = np.random.standard_normal((days, simulations))
    Z_mu = np.random.standard_normal((days, simulations))
    Z_price = np.random.standard_t(df=4, size=(days, simulations)) * np.sqrt(0.5)
    
    poisson_jumps = np.random.poisson(lambda_j * dt, (days, simulations))
    jump_magnitudes = np.random.normal(mu_j, sigma_j, (days, simulations))
    jump_returns = poisson_jumps * jump_magnitudes
    
    return _jit_montecarlo_loop(price_paths, v_t, mu_t, days, simulations, dt, kappa_v, long_term_sigma, vol_v, Z_vol, kappa_mu, long_term_mu, vol_mu, Z_mu, Z_price, jump_returns)

# ==============================================================================
# SANEAMIENTO MATEMÁTICO
# ==============================================================================

def calculate_risk_metrics_phase1(S0, paths, conf_level):
    final_prices = paths[-1, :]
    prob_pos = np.mean(final_prices > S0)
    
    intra_period_mins = np.amin(paths, axis=0)
    var_price = np.percentile(intra_period_mins, 100 - conf_level)
    
    cvar_price = np.mean(final_prices[final_prices <= var_price])
    if np.isnan(cvar_price): cvar_price = var_price
    
    return prob_pos, var_price, (var_price - S0) / S0, cvar_price, np.median(final_prices), np.percentile(final_prices, 90)

def generate_directive_common(prob_pos, days, rend_esp, tp_price, var_price, conf_level):
    intro = f"Proyección Estocástica Condicionada ({days} días):\n• **Capital Proyectado (Mediana):** Cambio de **{rend_esp:+,.2f}**.\n"
    if prob_pos > 0.62: return "Alcista", intro + f"• **Directriz:** COMPRA ESTRUCTURAL. • **Objetivo Clínico (TP):** ${tp_price:,.2f}. • **Barrera Touch VaR ({conf_level}%):** ${var_price:,.2f}.\n"
    elif prob_pos < 0.38: return "Bajista", intro + f"• **Directriz:** LIQUIDACIÓN PREVENTIVA. • **Soporte de Estrés:** ${var_price:,.2f}.\n"
    return "Neutral", intro + f"• **Directriz:** RETENCIÓN / CONDICIÓN LATERAL. Riesgo simétrico.\n"

# ==============================================================================
# EXPORTACIÓN PDF
# ==============================================================================

def create_pdf_report(data: dict) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, txt=f"REPORTE CUANTITATIVO: {data['ticker']}", ln=True, align='C')
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 8, txt=f"Fecha: {datetime.datetime.now().strftime('%Y-%m-%d')} | Modelo: Hibrido Fase 2", ln=True, align='C')
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 12)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 8, txt=" 1. NIVELES OPERATIVOS", ln=True, fill=True)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 6, txt=f"   - ESTADO: {data['estado'].upper()}", ln=True)
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 6, txt=f"   - Entrada: ${data['S0']:.2f}", ln=True)
    pdf.cell(0, 6, txt=f"   - Take Profit: ${data['tp_price']:.2f}", ln=True)
    pdf.cell(0, 6, txt=f"   - Touch Stop-Loss (VaR): ${data['var_price']:.2f}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, txt=" 2. EXPOSICION DE CAPITAL", ln=True, fill=True)
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 6, txt=f"   - Inversion Inicial: ${data['capital_inicial']:,.2f}", ln=True)
    pdf.cell(0, 6, txt=f"   - Retorno Esperado (Mediana): ${data['rendimiento_esperado']:+,.2f}", ln=True)
    pdf.cell(0, 6, txt=f"   - Capital en Riesgo (VaR): ${data['capital_var']:,.2f}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, txt=" 3. VARIABLES MACROECONOMICAS", ln=True, fill=True)
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 6, txt=f"   - Tasa Libre de Riesgo (^TNX): {data['rf_rate']*100:.2f}%", ln=True)
    pdf.cell(0, 6, txt=f"   - Indice de Miedo (^VIX): {data['vix_level']:.2f}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, txt=" 4. DIRECTRIZ INSTITUCIONAL", ln=True, fill=True)
    pdf.set_font("Arial", '', 10)
    clean_text = data['recomendacion'].replace('**', '').replace('•', '-')
    sustituciones = {'á':'a', 'é':'e', 'í':'i', 'ó':'o', 'ú':'u', 'Á':'A', 'É':'E', 'Í':'I', 'Ó':'O', 'Ú':'U'}
    for acento, sin_acento in sustituciones.items():
        clean_text = clean_text.replace(acento, sin_acento)
    for p in clean_text.split('\n'):
        if p.strip():
            pdf.multi_cell(0, 6, txt="   " + p.strip())
            pdf.ln(1)

    pdf_string = pdf.output(dest='S')
    return pdf_string.encode('latin-1', errors='replace')

# ==============================================================================
# UI Y DASHBOARD
# ==============================================================================

def render_dashboard():
    st.set_page_config(page_title="Quant Risk Engine", layout="wide", page_icon="📈")
    st.title("📊 Motor Cuantitativo de Riesgo (Fase 2: Datos y Macro)")

    # FASE 2: Extracción Macro al inicio de la app
    with st.spinner("Sincronizando Entorno Macroeconómico..."):
        risk_free_rate, current_vix = get_macro_context()

    with st.sidebar:
        st.header("1. Datos y Conexión")
        tiingo_key_input = st.text_input("Tiingo API Key (Opcional):", type="password")
        st.divider()
        st.header("2. Selección de Activo")
        
        ASSET_UNIVERSE = {
            "🔍 Entrada Manual (Ticker)": "MANUAL",
            "--- ÍNDICES Y ETFS ---": "HEADER",
            "📊 S&P 500 ETF (SPY)": "SPY",
            "📊 Nasdaq 100 ETF (QQQ)": "QQQ",
            "--- TECNOLOGÍA (MAG 7) ---": "HEADER",
            "🇺🇸 Apple Inc. (AAPL)": "AAPL",
            "🇺🇸 Microsoft Corp. (MSFT)": "MSFT",
            "🇺🇸 NVIDIA Corp. (NVDA)": "NVDA",
            "🇺🇸 Alphabet Inc. (GOOGL)": "GOOGL",
            "🇺🇸 Amazon.com Inc. (AMZN)": "AMZN",
            "🇺🇸 Meta Platforms (META)": "META",
            "🇺🇸 Tesla Inc. (TSLA)": "TSLA",
            "--- FINANZAS E INDUSTRIA ---": "HEADER",
            "🇺🇸 JPMorgan Chase (JPM)": "JPM",
            "🇺🇸 Berkshire Hathaway (BRK-B)": "BRK-B",
            "--- CRIPTOMONEDAS ---": "HEADER",
            "₿ Bitcoin (BTC-USD)": "BTC-USD",
            "⟠ Ethereum (ETH-USD)": "ETH-USD",
            "☀️ Solana (SOL-USD)": "SOL-USD",
            "--- MATERIAS PRIMAS ---": "HEADER",
            "🥇 Oro ETF (GLD)": "GLD",
            "🥈 Plata ETF (SLV)": "SLV",
            "🛢️ Petróleo ETF (USO)": "USO"
        }
        
        sel_asset = st.selectbox("Seleccione un Activo:", list(ASSET_UNIVERSE.keys()))
        
        if ASSET_UNIVERSE[sel_asset] == "HEADER":
            st.warning("⚠️ Selecciona un activo válido abajo.")
            st.stop()
            
        custom_asset = st.text_input("...o ingrese Ticker Manual:", "")
        ticker = custom_asset.upper() if custom_asset else ASSET_UNIVERSE[sel_asset]
        
        st.divider()
        capital_inicial = st.number_input("Capital ($):", min_value=10.0, value=10000.0, step=1000.0)
        days_to_project = st.slider("Días de Proyección:", 10, 252, 21)
        simulations = {"1k": 1000, "5k": 5000, "10k": 10000}[st.selectbox("Simulaciones:", ["1k", "5k", "10k"])]
        conf_level = st.slider("Límite VaR (%):", 90.0, 99.9, 95.0, 0.1)

    trading_days = 365 if "USD" in ticker else 252

    try:
        df_hist = load_financial_data(ticker, tiingo_key_input)
    except ValueError:
        df_hist = pd.DataFrame() 
        
    if not df_hist.empty: df_hist = df_hist.iloc[:-21] 
        
    if df_hist.empty:
        df_hist = generate_synthetic_data(ticker, days=500, trading_days=trading_days)
        st.error("🚨 **ALERTA CRÍTICA:** Conexión fallida. Desplegando simulación teórica base.")
    else:
        st.caption(f"✅ Conexión estable: {df_hist['Source'].iloc[0]} | Base temporal: {trading_days} días")
    
    # FASE 2: Ingeniería de Características (Pipeline ML)
    df_features = engineer_ml_features(df_hist, trading_days)
    
    daily_returns = df_hist['Close'].pct_change().dropna()
    S0 = df_hist['Close'].iloc[-1]
    
    raw_mu = daily_returns.mean() * trading_days
    
    with st.spinner("Calibrando Algoritmos Estocásticos..."):
        mejor_divisor = optimize_kalman_filter(daily_returns.values)
        kalman_states = apply_kalman_filter_dynamic(daily_returns.values, mejor_divisor)
        
        pesos_ewma = np.exp(np.linspace(-1, 0, min(20, len(kalman_states))))
        pesos_ewma /= pesos_ewma.sum()
        smoothed_kalman_return = np.dot(kalman_states[-len(pesos_ewma):], pesos_ewma)
        current_mu = smoothed_kalman_return * trading_days
        
        dynamic_sigma_daily = apply_stochastic_volatility_filter(daily_returns.values)
        current_sigma_ann = dynamic_sigma_daily[-1] * np.sqrt(trading_days)
        long_term_sigma_ann = np.mean(dynamic_sigma_daily) * np.sqrt(trading_days)

    downside_returns = daily_returns[daily_returns < 0]
    downside_sigma_ann = (downside_returns.std() * np.sqrt(trading_days)) if len(downside_returns) > 0 else current_sigma_ann
    sortino_ratio = (current_mu - risk_free_rate) / downside_sigma_ann if downside_sigma_ann > 0 else 0
    
    with st.spinner("Ejecutando Simulación Acelerada (Numba JIT)..."):
        paths = run_montecarlo_advanced_stochastic(S0, current_mu, raw_mu, current_sigma_ann, long_term_sigma_ann, days_to_project, simulations, 2.0, -0.05, 0.05, trading_days)
        prob_pos, var_price, var_loss, cvar_price, median_price, tp_price = calculate_risk_metrics_phase1(S0, paths, conf_level)

    capital_esperado = (capital_inicial / S0) * median_price
    capital_var = (capital_inicial / S0) * var_price
    rendimiento_esperado = capital_esperado - capital_inicial
    
    # FASE 2: Modificación Visual (Agregando VIX)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precio Base", f"${S0:,.2f}")
    c2.metric("Drift Estabilizado", f"{current_mu*100:.1f}%")
    c3.metric(f"Sortino (TNX: {risk_free_rate*100:.2f}%)", f"{sortino_ratio:.2f}")
    c4.metric("Volatilidad Dinámica", f"{current_sigma_ann*100:.1f}%", f"VIX: {current_vix:.1f}", delta_color="inverse")

    estado, recomendacion = generate_directive_common(prob_pos, days_to_project, rendimiento_esperado, tp_price, var_price, conf_level)
    
    if estado == "Alcista": st.success(recomendacion)
    elif estado == "Bajista": st.error(recomendacion)
    else: st.warning(recomendacion)

    fig = go.Figure()
    for i in range(min(50, paths.shape[1])): fig.add_trace(go.Scatter(x=np.arange(days_to_project + 1), y=paths[:, i], mode='lines', line=dict(color='rgba(0, 100, 255, 0.1)'), showlegend=False))
    fig.add_trace(go.Scatter(x=[0, days_to_project], y=[S0, S0], mode='lines', name='Entrada', line=dict(color='black', dash='dash')))
    fig.add_trace(go.Scatter(x=[0, days_to_project], y=[tp_price, tp_price], mode='lines', name='Take Profit', line=dict(color='green', dash='dot')))
    fig.add_trace(go.Scatter(x=[0, days_to_project], y=[var_price, var_price], mode='lines', name='Touch Stop-Loss (VaR)', line=dict(color='red', dash='dot')))
    st.plotly_chart(fig, use_container_width=True)

    # FASE 2: VISUALIZADOR DE FEATURES PARA ML
    st.markdown("---")
    with st.expander("🔍 Laboratorio Cuantitativo: Matriz de Entrenamiento ML (Fase 2)", expanded=False):
        st.caption("Estas son las variables estructuradas (Features) que alimentarán al Clasificador de Régimen en la Fase 3.")
        st.dataframe(df_features.tail(10).style.format("{:.4f}").background_gradient(cmap='Blues'), use_container_width=True)

    return {
        "ticker": ticker, "S0": S0, "tp_price": tp_price, "var_price": var_price,
        "estado": estado, "recomendacion": recomendacion, "capital_inicial": capital_inicial,
        "rendimiento_esperado": rendimiento_esperado, "capital_var": capital_var,
        "rf_rate": risk_free_rate, "vix_level": current_vix
    }

if __name__ == "__main__":
    report_data = render_dashboard()
    if report_data:
        with st.sidebar:
            st.markdown("---")
            with st.spinner("Compilando Reporte Institucional..."):
                pdf_bytes = create_pdf_report(report_data)
                st.download_button(
                    label="📥 Descargar Reporte PDF",
                    data=pdf_bytes,
                    file_name=f"Quant_Risk_{report_data['ticker']}.pdf",
                    mime="application/pdf",
                    type="primary"
                )
