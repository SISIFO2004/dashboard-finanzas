import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import requests
import io
import datetime
from datetime import timedelta
import plotly.graph_objects as go
from fpdf import FPDF

# ==============================================================================
# FASE 1: MOTOR DE INGESTA HÍBRIDO Y RESILIENTE
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
    df['Source'] = 'Sintético (Modo Failsafe)'
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def load_financial_data(ticker: str) -> pd.DataFrame:
    # Intento 1: Yahoo Finance
    try:
        session = requests.Session()
        session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/115.0.0.0 Safari/537.36'})
        yf_ticker = yf.Ticker(ticker, session=session)
        df = yf_ticker.history(period="2y")
        if not df.empty and 'Close' in df.columns:
            df = df[['Close']].copy()
            df['Source'] = 'Yahoo Finance (Masked API)'
            return df
    except Exception: pass

    # Intento 2: Stooq Directo (Con parche .US)
    try:
        tickers_to_try = [ticker, f"{ticker}.US"]
        for t_try in tickers_to_try:
            stooq_url = f"https://stooq.com/q/d/l/?s={t_try}&i=d"
            response = requests.get(stooq_url, timeout=10)
            if response.status_code == 200 and "Date" in response.text:
                df = pd.read_csv(io.StringIO(response.text), index_col='Date', parse_dates=True).sort_index().tail(504)
                if not df.empty and 'Close' in df.columns:
                    df = df[['Close']].copy()
                    df['Source'] = 'Stooq Direct HTTP'
                    return df
    except Exception: pass

    # Intento 3: Sintético
    st.warning(f"⚠️ Red bloqueada por proveedores. Activando 'Degradación Graciosa' con datos sintéticos para {ticker}.")
    return generate_synthetic_data(ticker, days=500)

# ==============================================================================
# FASE 2: MOTOR MATEMÁTICO ESTOCÁSTICO Y ESTRUCTURA DE STREAMLIT (UI)
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
    return prob_positive, var_price, var_loss_pct, cvar_price, cvar_loss_pct

def generate_directive(prob_positive):
    if prob_positive > 0.65: return "Bullish", "Sesgo Alcista Fuerte. Considere estrategias de Momentum. Mantener Stop-Loss holgado."
    elif prob_positive < 0.35: return "Bearish", "Sesgo Bajista Peligroso. Prioridad Absoluta: Estrategias de Hedging o reducción de exposición. Ajustar Stop-Loss."
    else: return "Neutral", "Distribución no direccional. Recomendado: Estrategias Delta-Neutral o Dollar-Cost Averaging."

def render_dashboard():
    st.set_page_config(page_title="Quant Risk Engine", layout="wide", page_icon="📈")
    st.title("📊 Quantitative Risk Analytics Engine")
    st.markdown("Motor de simulaciones de Montecarlo con Movimiento Browniano Geométrico y Difusión de Saltos de Merton.")

    st.sidebar.header("⚙️ Parámetros del Modelo")
    asset_list =["AAPL", "MSFT", "TSLA", "SPY", "BTC-USD", "GC=F"]
    selected_asset = st.sidebar.selectbox("Seleccione un Activo:", asset_list)
    custom_asset = st.sidebar.text_input("...o ingrese un Ticker Manual:", "")
    ticker = custom_asset.upper() if custom_asset else selected_asset
    
    days_to_project = st.sidebar.slider("Días Hábiles a Proyectar:", 10, 252, 60)
    simulations = {"1k": 1000, "5k": 5000, "10k": 10000}[st.sidebar.selectbox("Número de Simulaciones:", ["1k", "5k", "10k"])]
    
    st.sidebar.subheader("🧮 Ajustes Cuantitativos")
    override_drift = st.sidebar.checkbox("Forzar Drift (μ) Manual")
    manual_drift = st.sidebar.number_input("Drift Anualizado (μ):", value=0.10, step=0.01) if override_drift else None
    conf_level = st.sidebar.slider("Nivel de Confianza VaR (%):", 90.0, 99.9, 95.0, 0.1)
    
    st.sidebar.subheader("📉 Merton Jumps (Cisnes Negros)")
    lambda_j = st.sidebar.slider("Prob. Saltos/Año (λ):", 0.0, 10.0, 2.0)
    mu_j = st.sidebar.number_input("Media del Salto (μ_J):", value=-0.05, step=0.01)
    sigma_j = st.sidebar.number_input("Volatilidad del Salto (σ_J):", value=0.05, step=0.01)

    df_hist = load_financial_data(ticker)
    if df_hist.empty:
        st.error(f"Imposible obtener datos para {ticker}.")
        st.stop()
        
    st.write(f"***Fuente de datos actual:** `{df_hist['Source'].iloc[0]}`*")
    
    daily_returns = df_hist['Close'].pct_change().dropna()
    hist_mu = daily_returns.mean() * 252
    hist_sigma = daily_returns.std() * np.sqrt(252)
    S0 = df_hist['Close'].iloc[-1]
    
    mu = manual_drift if override_drift else hist_mu
    sigma = hist_sigma
    
    with st.spinner("🚀 Computando campo estocástico..."):
        paths = run_montecarlo_jumps(S0, mu, sigma, days_to_project, simulations, lambda_j, mu_j, sigma_j)
        final_prices = paths[-1, :]
        prob_pos, var_price, var_loss, cvar_price, cvar_loss = calculate_risk_metrics(S0, final_prices, conf_level)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Precio Spot (S₀)", f"${S0:,.2f}")
    col2.metric("Probabilidad de Éxito", f"{prob_pos*100:.1f}%")
    col3.metric(f"VaR ({conf_level}%)", f"${var_price:,.2f}", f"{var_loss*100:.1f}%", delta_color="inverse")
    col4.metric(f"CVaR", f"${cvar_price:,.2f}", f"{cvar_loss*100:.1f}%", delta_color="inverse")
    
    estado, recomendacion = generate_directive(prob_pos)
    # Colores visuales en Streamlit
    color_map = {"Bullish": "success", "Bearish": "error", "Neutral": "warning"}
    if estado == "Bullish": st.success(f"**Directriz [{estado}]:** {recomendacion}")
    elif estado == "Bearish": st.error(f"**Directriz [{estado}]:** {recomendacion}")
    else: st.warning(f"**Directriz [{estado}]:** {recomendacion}")

    fig = go.Figure()
    visual_paths = paths[:, :100]
    time_axis = np.arange(days_to_project + 1)
    
    for i in range(visual_paths.shape[1]):
        fig.add_trace(go.Scatter(x=time_axis, y=visual_paths[:, i], mode='lines', line=dict(color='rgba(0, 100, 255, 0.1)'), showlegend=False, hoverinfo='skip'))
        
    fig.add_trace(go.Scatter(x=[0, days_to_project], y=[S0, S0], mode='lines', name='Precio Actual (S₀)', line=dict(color='black', width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=[0, days_to_project], y=[var_price, var_price], mode='lines', name=f'VaR Limite ({conf_level}%)', line=dict(color='red', width=2, dash='dot')))
                             
    fig.update_layout(title=f"Simulación Montecarlo ({simulations} trayectorias) - Muestra visual: 100", xaxis_title="Días a Futuro", yaxis_title="Precio Simulado ($)", template="plotly_white", height=500)
    st.plotly_chart(fig, use_container_width=True)

    return {
        "ticker": ticker, "S0": S0, "mu": mu, "sigma": sigma, "days": days_to_project,
        "simulations": simulations, "prob_pos": prob_pos, "var_price": var_price, 
        "cvar_price": cvar_price, "estado": estado, "recomendacion": recomendacion
    }

# ==============================================================================
# FASE 3: GENERACIÓN DE REPORTES PDF EN MEMORIA
# ==============================================================================

def create_pdf_report(report_data: dict) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Quantitative Risk Analytics Report", ln=True, align='C')
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(200, 10, txt=f"Generado el: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="1. Parametros del Modelo y Activo", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.cell(200, 8, txt=f"- Activo Analizado: {report_data['ticker']}", ln=True)
    pdf.cell(200, 8, txt=f"- Precio Spot Actual: ${report_data['S0']:.2f}", ln=True)
    pdf.cell(200, 8, txt=f"- Horizonte de Proyeccion: {report_data['days']} dias", ln=True)
    pdf.cell(200, 8, txt=f"- Simulaciones Ejecutadas: {report_data['simulations']}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="2. Metricas de Riesgo", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.cell(200, 8, txt=f"- Probabilidad de Retorno Positivo: {report_data['prob_pos']*100:.2f}%", ln=True)
    pdf.cell(200, 8, txt=f"- Value at Risk (VaR): ${report_data['var_price']:.2f}", ln=True)
    pdf.cell(200, 8, txt=f"- Conditional VaR (CVaR): ${report_data['cvar_price']:.2f}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="3. Directriz Estrategica", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 8, txt=f"ESTADO: {report_data['estado']}")
    pdf.multi_cell(0, 8, txt=f"RECOMENDACION: {report_data['recomendacion']}")
    pdf.ln(10)

    pdf.set_font("Arial", 'I', 8)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 5, txt="DISCLAIMER: Este reporte es generado por algoritmos con fines de analisis cuantitativo. No constituye asesoria de inversion. Los modelos basados en Montecarlo y Jump Diffusion no garantizan rendimientos futuros.")

    return pdf.output(dest='S').encode('latin-1')

# ==============================================================================
# EJECUCIÓN PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    report_data = render_dashboard()
    
    if report_data:
        st.markdown("---")
        st.subheader("📄 Generación de Reporte Ejecutivo")
        with st.spinner("Generando PDF..."):
            pdf_bytes = create_pdf_report(report_data)
            st.download_button(
                label="📥 Descargar Reporte en PDF",
                data=pdf_bytes,
                file_name=f"Quant_Risk_Report_{report_data['ticker']}.pdf",
                mime="application/pdf"
            )
