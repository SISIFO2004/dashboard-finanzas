import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from yahooquery import Ticker as YQTicker
import requests
import requests_cache
import io
import datetime
from datetime import timedelta
import plotly.graph_objects as go
from fpdf import FPDF

# ==============================================================================
# CONFIGURACIÓN DE MEMORIA CACHÉ
# ==============================================================================
session = requests_cache.CachedSession('mercado_cache', expire_after=86400)
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36'
})

# ==============================================================================
# FASE 1: MOTOR DE INGESTA RESILIENTE
# ==============================================================================

def generate_synthetic_data(ticker: str, days: int = 500) -> pd.DataFrame:
    np.random.seed(42)
    S0 = 150.0  
    mu = 0.08   
    sigma = 0.20 
    dt = 1/252   
    prices = [S0]
    for _ in range(1, days):
        Z = np.random.normal()
        S_t = prices[-1] * np.exp((mu - (sigma**2) / 2) * dt + sigma * np.sqrt(dt) * Z)
        prices.append(S_t)
    end_date = datetime.date.today()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, periods=days, freq='B')
    df = pd.DataFrame(index=dates)
    df['Close'] = prices
    df['Source'] = 'Sintético (Modo Contingencia)'
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def load_financial_data(ticker: str, tiingo_key: str = "") -> pd.DataFrame:
    yf_ticker = ticker
    t_ticker = ticker.replace("-USD", "") if "USD" in ticker else ticker

    if tiingo_key:
        try:
            if "USD" in ticker:
                url = f"https://api.tiingo.com/tiingo/crypto/prices?tickers={t_ticker}usd&resampleFreq=1day&token={tiingo_key}"
            else:
                start_date = (datetime.datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
                url = f"https://api.tiingo.com/tiingo/daily/{t_ticker}/prices?startDate={start_date}&token={tiingo_key}"
            
            response = session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if len(data) > 0:
                    df = pd.DataFrame(data[0]['priceData']) if "USD" in ticker else pd.DataFrame(data)
                    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
                    df.set_index('date', inplace=True)
                    df = df.rename(columns={'close': 'Close'}).sort_index().tail(504)
                    df_final = df[['Close']].copy()
                    cache_status = "Caché Local" if response.from_cache else "Internet"
                    df_final['Source'] = f'Tiingo API ({cache_status})'
                    return df_final
        except Exception: pass

    try:
        yq = YQTicker(yf_ticker, session=session)
        df = yq.history(period="2y")
        if not df.empty and isinstance(df, pd.DataFrame) and 'close' in df.columns:
            df = df.reset_index()
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
            df = df.set_index('date').rename(columns={'close': 'Close'})
            df_final = df[['Close']].copy()
            df_final['Source'] = 'YahooQuery'
            return df_final
    except Exception: pass

    try:
        stock = yf.Ticker(yf_ticker, session=session)
        df = stock.history(period="2y")
        if not df.empty and 'Close' in df.columns:
            df = df[['Close']].copy()
            df['Source'] = 'Yahoo Finance'
            return df
    except Exception: pass

    return pd.DataFrame() 

# ==============================================================================
# FASE 2: MOTOR MATEMÁTICO ESTOCÁSTICO
# ==============================================================================

def run_montecarlo_jumps(S0, mu, sigma, days, simulations, lambda_j, mu_j, sigma_j):
    dt = 1 / 252  
    Z = np.random.standard_normal((days, simulations))
    gbm_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    poisson_jumps = np.random.poisson(lambda_j * dt, (days, simulations))
    jump_magnitudes = np.random.normal(mu_j, sigma_j, (days, simulations))
    jump_returns = poisson_jumps * jump_magnitudes
    total_returns = gbm_returns + jump_returns
    price_paths = S0 * np.exp(np.cumsum(total_returns, axis=0))
    S0_array = np.full((1, simulations), S0)
    return np.vstack((S0_array, price_paths))

def calculate_risk_metrics(S0, final_prices, conf_level):
    prob_positive = np.mean(final_prices > S0)
    var_price = np.percentile(final_prices, 100 - conf_level)
    var_loss_pct = (var_price - S0) / S0
    cvar_price = np.mean(final_prices[final_prices <= var_price])
    if np.isnan(cvar_price): cvar_price = var_price
    cvar_loss_pct = (cvar_price - S0) / S0
    # Usamos la mediana de los precios finales como la expectativa más realista para distribuciones sesgadas
    median_price = np.median(final_prices) 
    return prob_positive, var_price, var_loss_pct, cvar_price, cvar_loss_pct, median_price

# --- DIRECTRICES CONCISAS Y PRECISAS ---
def generate_directive(prob_pos, sigma, var_loss_pct, days):
    vol = sigma * 100
    riesgo = abs(var_loss_pct) * 100

    if prob_pos > 0.65:
        estado = "Bullish"
        rec = f"• **Tendencia:** Fuerte asimetría positiva ({prob_pos*100:.1f}% prob. éxito).\n"
        rec += f"• **Acción Recomendada:** Operar posiciones en largo o aplicar estrategias de Momentum.\n"
        rec += f"• **Gestión de Riesgo:** Volatilidad del {vol:.1f}%. Fije un Stop-Loss dinámico por debajo de la barrera VaR ({riesgo:.1f}%)."
    elif prob_pos < 0.35:
        estado = "Bearish"
        rec = f"• **Tendencia:** Dominancia de retornos negativos ({100 - prob_pos*100:.1f}% prob. de pérdida).\n"
        rec += f"• **Acción Recomendada:** Reducir exposición. Implementar coberturas (Puts) o posiciones en corto.\n"
        rec += f"• **Gestión de Riesgo:** Peligro estructural. Todo soporte técnico por encima del VaR ({riesgo:.1f}%) es frágil. Cierre agresivo de pérdidas."
    else:
        estado = "Neutral"
        rec = f"• **Tendencia:** Distribución simétrica (Rango lateral / Entropía direccional).\n"
        rec += f"• **Acción Recomendada:** Estrategias Delta-Neutral (Iron Condors) o acumulación pasiva (DCA).\n"
        rec += f"• **Gestión de Riesgo:** Comprar debilidad en el límite VaR ({riesgo:.1f}%) y vender fortaleza en desviaciones estándar superiores."

    return estado, rec

# ==============================================================================
# UI Y DASHBOARD
# ==============================================================================

def render_dashboard():
    st.set_page_config(page_title="Quant Risk Engine", layout="wide", page_icon="📈")
    st.title("📊 Quantitative Risk Analytics Engine")

    # --- BARRA LATERAL ---
    st.sidebar.header("1. Datos y Conexión")
    tiingo_key_input = st.sidebar.text_input("Tiingo API Key (Opcional):", type="password")
    
    st.sidebar.divider()
    st.sidebar.header("2. Selección de Activo")
    ASSET_UNIVERSE = {
        "🔍 Entrada Manual (Ticker)": "MANUAL",
        "--- TECNOLÓGICAS ---": "MANUAL",
        "🇺🇸 Apple Inc. (AAPL)": "AAPL",
        "🇺🇸 Microsoft (MSFT)": "MSFT",
        "🇺🇸 NVIDIA (NVDA)": "NVDA",
        "--- CRIPTOMONEDAS ---": "MANUAL",
        "₿ Bitcoin (BTC-USD)": "BTC-USD",
        "⟠ Ethereum (ETH-USD)": "ETH-USD",
        "--- MATERIAS PRIMAS ---": "MANUAL",
        "🥇 Oro (GC=F)": "GC=F"
    }
    
    selected_asset = st.sidebar.selectbox("Seleccione un Activo:", list(ASSET_UNIVERSE.keys()))
    if "---" in selected_asset:
        st.warning("Selecciona un activo válido.")
        st.stop()
        
    custom_asset = st.sidebar.text_input("...o ingrese Ticker Manual:", "")
    ticker = custom_asset.upper() if custom_asset else ASSET_UNIVERSE[selected_asset]
    
    # --- NUEVO MÓDULO DE INVERSIÓN ---
    st.sidebar.divider()
    st.sidebar.header("3. Simulación de Capital")
    moneda_str = st.sidebar.selectbox("Moneda Base:", ["USD ($)", "EUR (€)"])
    simbolo = "$" if "USD" in moneda_str else "€"
    # Usamos código de moneda explícito para evitar problemas de codificación en el PDF
    codigo_moneda = "USD" if "USD" in moneda_str else "EUR" 
    
    capital_inicial = st.sidebar.number_input(f"Capital a Invertir ({simbolo}):", min_value=10.0, value=10000.0, step=1000.0)

    st.sidebar.divider()
    st.sidebar.header("4. Parámetros del Modelo")
    days_to_project = st.sidebar.slider("Días a Proyectar:", 10, 252, 60)
    simulations = {"1k": 1000, "5k": 5000, "10k": 10000}[st.sidebar.selectbox("Simulaciones:", ["1k", "5k", "10k"])]
    conf_level = st.sidebar.slider("Nivel VaR (%):", 90.0, 99.9, 95.0, 0.1)

    with st.sidebar.expander("📉 Calibración Avanzada (Jumps & Drift)", expanded=False):
        override_drift = st.checkbox("Forzar Drift (μ) Manual")
        manual_drift = st.number_input("Drift Anualizado (μ):", value=0.10, step=0.01) if override_drift else None
        lambda_j = st.slider("Prob. Saltos/Año (λ):", 0.0, 10.0, 2.0)
        mu_j = st.number_input("Media Salto (μ_J):", value=-0.05, step=0.01)
        sigma_j = st.number_input("Vol. Salto (σ_J):", value=0.05, step=0.01)

    # --- EJECUCIÓN CORE ---
    df_hist = load_financial_data(ticker, tiingo_key_input)
    if df_hist.empty:
        df_hist = generate_synthetic_data(ticker, days=500)
    else:
        st.caption(f"Conexión: {df_hist['Source'].iloc[0]}")
    
    daily_returns = df_hist['Close'].pct_change().dropna()
    hist_mu = daily_returns.mean() * 252
    hist_sigma = daily_returns.std() * np.sqrt(252)
    S0 = df_hist['Close'].iloc[-1]
    
    mu = manual_drift if override_drift else hist_mu
    sigma = hist_sigma
    
    with st.spinner("🚀 Computando..."):
        paths = run_montecarlo_jumps(S0, mu, sigma, days_to_project, simulations, lambda_j, mu_j, sigma_j)
        final_prices = paths[-1, :]
        prob_pos, var_price, var_loss, cvar_price, cvar_loss, median_price = calculate_risk_metrics(S0, final_prices, conf_level)

    # --- CÁLCULOS DE PORTAFOLIO REAL ---
    acciones_adquiridas = capital_inicial / S0
    capital_esperado = acciones_adquiridas * median_price
    capital_var = acciones_adquiridas * var_price
    rendimiento_esperado = capital_esperado - capital_inicial

    # --- KPIs DE LA ACCIÓN ---
    st.subheader(f"Métricas del Activo: {ticker}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precio Spot (S₀)", f"${S0:,.2f}")
    c2.metric("Probabilidad de Éxito", f"{prob_pos*100:.1f}%")
    c3.metric(f"Precio VaR ({conf_level}%)", f"${var_price:,.2f}", f"{var_loss*100:.1f}%", delta_color="inverse")
    c4.metric("Volatilidad Anual (σ)", f"{sigma*100:.1f}%")

    # --- KPIs DEL PORTAFOLIO (NUEVO) ---
    st.markdown("---")
    st.subheader(f"Proyección de Inversión ({codigo_moneda}) a {days_to_project} días")
    p1, p2, p3 = st.columns(3)
    p1.metric("Capital Inicial", f"{simbolo}{capital_inicial:,.2f}", f"{acciones_adquiridas:.4f} shares", delta_color="off")
    p2.metric("Valor Proyectado (Mediana)", f"{simbolo}{capital_esperado:,.2f}", f"{simbolo}{rendimiento_esperado:,.2f} Retorno Esperado")
    p3.metric("Capital en Riesgo Máximo (VaR)", f"{simbolo}{capital_var:,.2f}", f"{simbolo}{capital_var - capital_inicial:,.2f} Exposición Crítica", delta_color="inverse")

    # --- DIRECTRICES ---
    estado, recomendacion = generate_directive(prob_pos, sigma, var_loss, days_to_project)
    
    st.markdown("---")
    if estado == "Bullish": st.success(f"### 🟢 Estado: {estado}")
    elif estado == "Bearish": st.error(f"### 🔴 Estado: {estado}")
    else: st.warning(f"### 🟡 Estado: {estado}")
    st.markdown(recomendacion)
    st.markdown("---")

    # --- GRÁFICO ---
    fig = go.Figure()
    visual_paths = paths[:, :100]
    time_axis = np.arange(days_to_project + 1)
    for i in range(visual_paths.shape[1]):
        fig.add_trace(go.Scatter(x=time_axis, y=visual_paths[:, i], mode='lines', line=dict(color='rgba(0, 100, 255, 0.1)'), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=[0, days_to_project], y=[S0, S0], mode='lines', name='Precio Spot', line=dict(color='black', width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=[0, days_to_project], y=[var_price, var_price], mode='lines', name=f'VaR ({conf_level}%)', line=dict(color='red', width=2, dash='dot')))
    fig.update_layout(title=f"Simulación Montecarlo ({simulations} trayectorias)", height=400, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    return {
        "ticker": ticker, "S0": S0, "sigma": sigma, "days": days_to_project, "simulations": simulations,
        "prob_pos": prob_pos, "var_price": var_price, "var_loss": var_loss,
        "capital_inicial": capital_inicial, "moneda": codigo_moneda, "acciones": acciones_adquiridas,
        "capital_esperado": capital_esperado, "capital_var": capital_var,
        "estado": estado, "recomendacion": recomendacion
    }

# ==============================================================================
# FASE 3: GENERACIÓN PDF (CORREGIDA PARA UNICODE)
# ==============================================================================

def create_pdf_report(data: dict) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, txt=f"REPORTE CUANTITATIVO: {data['ticker']}", ln=True, align='C')
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 8, txt=f"Fecha: {datetime.datetime.now().strftime('%Y-%m-%d')}", ln=True, align='C')
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 11)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 8, txt=" 1. SIMULACION DE PORTAFOLIO", ln=True, fill=True)
    pdf.set_font("Arial", '', 10)
    
    # Sanitizamos la moneda para evitar el símbolo € que rompe fpdf
    moneda_pdf = "EUR" if data['moneda'] == "EUR" else "USD"
    
    pdf.cell(0, 6, txt=f"   - Capital Inicial: {data['capital_inicial']:,.2f} {moneda_pdf}", ln=True)
    pdf.cell(0, 6, txt=f"   - Tamano de Posicion: {data['acciones']:.4f} shares (Precio: ${data['S0']:.2f})", ln=True)
    pdf.cell(0, 6, txt=f"   - Capital Proyectado (Mediana): {data['capital_esperado']:,.2f} {moneda_pdf}", ln=True)
    pdf.cell(0, 6, txt=f"   - Exposicion Maxima al VaR: {data['capital_var']:,.2f} {moneda_pdf} (Perdida esperada: {data['var_loss']*100:.1f}%)", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 8, txt=" 2. DIRECTRIZ ESTRATEGICA", ln=True, fill=True)
    pdf.set_font("Arial", 'B', 10)
    
    # Limpiamos acentos y caracteres especiales del estado
    estado_limpio = data['estado'].upper().replace('Ó', 'O').replace('Í', 'I')
    pdf.cell(0, 6, txt=f"   ESTADO: {estado_limpio}", ln=True)
    pdf.set_font("Arial", '', 10)
    
    # Sanitizamos el texto: quitamos negritas (**) y cambiamos viñetas (•) por guiones (-)
    clean_text = data['recomendacion'].replace('**', '').replace('•', '-')
    
    # Limpiamos tildes comunes para evitar errores latin-1
    sustituciones = {'á':'a', 'é':'e', 'í':'i', 'ó':'o', 'ú':'u', 'Á':'A', 'É':'E', 'Í':'I', 'Ó':'O', 'Ú':'U'}
    for acento, sin_acento in sustituciones.items():
        clean_text = clean_text.replace(acento, sin_acento)

    for p in clean_text.split('\n'):
        if p.strip():
            pdf.multi_cell(0, 6, txt="   " + p.strip())
            pdf.ln(1)

    # El encode de fpdf requiere ignore o replace para sobrevivir a caracteres rebeldes
    pdf_string = pdf.output(dest='S')
    return pdf_string.encode('latin-1', errors='replace')

if __name__ == "__main__":
    report_data = render_dashboard()
    if report_data:
        # ATENCIÓN: El botón ahora vive en la barra lateral izquierda
        with st.sidebar:
            st.markdown("---")
            with st.spinner("Generando PDF..."):
                pdf_bytes = create_pdf_report(report_data)
                st.download_button(
                    label="📥 Descargar PDF Analítico",
                    data=pdf_bytes,
                    file_name=f"Quant_Report_{report_data['ticker']}.pdf",
                    mime="application/pdf",
                    type="primary"
                )
