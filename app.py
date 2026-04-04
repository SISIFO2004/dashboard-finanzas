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
# CONFIGURACIÓN DE MEMORIA CACHÉ AVANZADA
# ==============================================================================
session = requests_cache.CachedSession('mercado_cache', expire_after=86400)
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
})

# ==============================================================================
# FASE 1: MOTOR DE INGESTA ULTRA-RESILIENTE (Tiingo + YahooQuery + Caché)
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
                    df_final['Source'] = f'Tiingo API Oficial ({cache_status})'
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
            df_final['Source'] = 'YahooQuery Backend'
            return df_final
    except Exception: pass

    try:
        stock = yf.Ticker(yf_ticker, session=session)
        df = stock.history(period="2y")
        if not df.empty and 'Close' in df.columns:
            df = df[['Close']].copy()
            df['Source'] = 'Yahoo Finance Tradicional'
            return df
    except Exception: pass

    return pd.DataFrame() 

# ==============================================================================
# FASE 2: MOTOR MATEMÁTICO ESTOCÁSTICO Y UI
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

# --- NUEVO MOTOR DE GENERACIÓN DE DIRECTRICES PROFUNDAS ---
def generate_directive(prob_pos, sigma, var_loss_pct, days):
    vol_text = "Alta Volatilidad" if sigma > 0.35 else "Volatilidad Moderada/Estable"
    riesgo_text = f"El modelo proyecta una exposición a pérdidas máximas esperadas (VaR) del {abs(var_loss_pct)*100:.1f}% para el horizonte de {days} días."

    if prob_pos > 0.65:
        estado = "Bullish (Alcista)"
        rec = f"**Análisis Estocástico:** El modelo muestra una fuerte asimetría positiva ({prob_pos*100:.1f}% de probabilidad de éxito). Las simulaciones indican que la tendencia estructural (Drift) está superando significativamente al ruido del mercado, proyectando un escenario de expansión de precios.\n\n"
        rec += f"**Estrategia Recomendada:** Se sugiere un enfoque de Momentum o la apertura de posiciones en largo. Para portafolios avanzados, es un entorno propicio para la compra de Call Options o el incremento escalonado de la exposición neta al activo.\n\n"
        rec += f"**Gestión de Riesgo:** Régimen de {vol_text}. {riesgo_text} Se recomienda fijar un 'Trailing Stop-Loss' (Stop de arrastre) justo por debajo del límite del VaR para proteger las ganancias acumuladas sin asfixiar las fluctuaciones normales del mercado."
    elif prob_pos < 0.35:
        estado = "Bearish (Bajista)"
        rec = f"**Análisis Estocástico:** El campo probabilístico detecta un deterioro agudo en la acción del precio ({100 - prob_pos*100:.1f}% de probabilidad de caída). El peso de los retornos negativos domina las proyecciones de Montecarlo, indicando una tendencia descendente sostenida.\n\n"
        rec += f"**Estrategia Recomendada:** Prioridad Absoluta de Defensa. Se recomienda una reducción drástica del apalancamiento. Es el momento estadístico para implementar coberturas (Hedging) mediante la compra de Put Options o abrir posiciones en corto si el perfil de inversión lo permite.\n\n"
        rec += f"**Gestión de Riesgo:** Entorno de {vol_text}. {riesgo_text} Cualquier posición comprada restante debe contar con un Stop-Loss rígido y ajustado al milímetro. Evitar promediar a la baja (comprar más mientras cae)."
    else:
        estado = "Neutral (Consolidación / Incertidumbre)"
        rec = f"**Análisis Estocástico:** La distribución de trayectorias es altamente simétrica. No existe una dominancia clara entre fuerzas compradoras y vendedoras. El activo se encuentra en una fase de alta entropía direccional o rango lateral.\n\n"
        rec += f"**Estrategia Recomendada:** Evitar apuestas direccionales agresivas. El entorno estadístico favorece estrategias neutrales al mercado (Delta-Neutral) como Iron Condors o Straddles. Para inversores pasivos, es zona de acumulación conservadora vía Dollar-Cost Averaging (DCA).\n\n"
        rec += f"**Gestión de Riesgo:** Régimen de {vol_text}. {riesgo_text} La estrategia de supervivencia aquí es operar los extremos: acumular cerca de la línea del VaR (soporte estadístico) y tomar beneficios en las desviaciones estándar superiores."

    return estado, rec

def render_dashboard():
    st.set_page_config(page_title="Quant Risk Engine", layout="wide", page_icon="📈")
    st.title("📊 Quantitative Risk Analytics Engine")
    st.markdown("Motor de simulaciones con Arquitectura de Evasión (YahooQuery) y Caché Persistente.")

    st.sidebar.header("1. Conexión de Datos")
    tiingo_key_input = st.sidebar.text_input("Tiingo API Key (Opcional - 500/día):", type="password", help="Tiingo ofrece 500 descargas diarias gratis. Regístrate en tiingo.com.")
    
    st.sidebar.divider()
    st.sidebar.header("2. Selección de Activo")
    
    ASSET_UNIVERSE = {
        "🔍 Entrada Manual (Ticker)": "MANUAL",
        "--- TECNOLÓGICAS ---": "MANUAL",
        "🇺🇸 Apple Inc. (AAPL)": "AAPL",
        "🇺🇸 Microsoft (MSFT)": "MSFT",
        "🇺🇸 NVIDIA (NVDA)": "NVDA",
        "🇺🇸 Tesla (TSLA)": "TSLA",
        "🇺🇸 Meta Platforms (META)": "META",
        "🇺🇸 Alphabet/Google (GOOGL)": "GOOGL",
        "🇺🇸 Amazon (AMZN)": "AMZN",
        "--- CRIPTOMONEDAS ---": "MANUAL",
        "₿ Bitcoin (BTC-USD)": "BTC-USD",
        "⟠ Ethereum (ETH-USD)": "ETH-USD",
        "◎ Solana (SOL-USD)": "SOL-USD",
        "--- ÍNDICES Y ETFS ---": "MANUAL",
        "📊 S&P 500 ETF (SPY)": "SPY",
        "📊 Nasdaq 100 ETF (QQQ)": "QQQ",
        "--- MATERIAS PRIMAS ---": "MANUAL",
        "🥇 Oro (GC=F)": "GC=F",
        "🥈 Plata (SI=F)": "SI=F",
        "🛢️ Petróleo Crudo (CL=F)": "CL=F"
    }
    
    selected_asset = st.sidebar.selectbox("Seleccione un Activo:", list(ASSET_UNIVERSE.keys()))
    if "---" in selected_asset:
        st.warning("Por favor selecciona un activo válido de la lista.")
        st.stop()
        
    custom_asset = st.sidebar.text_input("...o ingrese Ticker Manual (Ej: AMD, INTC):", "")
    ticker = custom_asset.upper() if custom_asset else ASSET_UNIVERSE[selected_asset]
    
    st.sidebar.divider()
    st.sidebar.header("3. Simulación")
    days_to_project = st.sidebar.slider("Días Hábiles a Proyectar:", 10, 252, 60)
    simulations = {"1k": 1000, "5k": 5000, "10k": 10000}[st.sidebar.selectbox("Número de Simulaciones:", ["1k", "5k", "10k"])]
    
    st.sidebar.divider()
    st.sidebar.header("4. Ajustes Cuantitativos")
    override_drift = st.sidebar.checkbox("Forzar Drift (μ) Manual")
    manual_drift = st.sidebar.number_input("Drift Anualizado (μ):", value=0.10, step=0.01) if override_drift else None
    conf_level = st.sidebar.slider("Nivel de Confianza VaR (%):", 90.0, 99.9, 95.0, 0.1)
    
    with st.sidebar.expander("📉 Calibración de Cisnes Negros (Merton Jumps)", expanded=False):
        lambda_j = st.slider("Prob. Saltos/Año (λ):", 0.0, 10.0, 2.0)
        mu_j = st.number_input("Media del Salto (μ_J):", value=-0.05, step=0.01)
        sigma_j = st.number_input("Volatilidad del Salto (σ_J):", value=0.05, step=0.01)

    with st.spinner("Descargando datos financieros (Buscando en Caché primero)..."):
        df_hist = load_financial_data(ticker, tiingo_key_input)
    
    if df_hist.empty:
        st.warning(f"⚠️ Servidores bloqueados permanentemente. Activando 'Modo Contingencia' para {ticker}.")
        df_hist = generate_synthetic_data(ticker, days=500)
    else:
        st.success(f"✅ Conexión Activa: {df_hist['Source'].iloc[0]}")
    
    daily_returns = df_hist['Close'].pct_change().dropna()
    hist_mu = daily_returns.mean() * 252
    hist_sigma = daily_returns.std() * np.sqrt(252)
    S0 = df_hist['Close'].iloc[-1]
    
    mu = manual_drift if override_drift else hist_mu
    sigma = hist_sigma
    
    with st.spinner("🚀 Computando campo estocástico multidimensional..."):
        paths = run_montecarlo_jumps(S0, mu, sigma, days_to_project, simulations, lambda_j, mu_j, sigma_j)
        final_prices = paths[-1, :]
        prob_pos, var_price, var_loss, cvar_price, cvar_loss = calculate_risk_metrics(S0, final_prices, conf_level)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Precio Spot (S₀)", f"${S0:,.2f}")
    col2.metric("Probabilidad de Éxito", f"{prob_pos*100:.1f}%")
    col3.metric(f"VaR ({conf_level}%)", f"${var_price:,.2f}", f"{var_loss*100:.1f}%", delta_color="inverse")
    col4.metric(f"CVaR", f"${cvar_price:,.2f}", f"{cvar_loss*100:.1f}%", delta_color="inverse")
    
    estado, recomendacion = generate_directive(prob_pos, sigma, var_loss, days_to_project)
    
    # Renderizado estilizado de la recomendación en UI
    st.markdown("---")
    if estado.startswith("Bullish"):
        st.success(f"### 📈 Directriz Estratégica: {estado}")
    elif estado.startswith("Bearish"):
        st.error(f"### 📉 Directriz Estratégica: {estado}")
    else:
        st.warning(f"### ⚖️ Directriz Estratégica: {estado}")
        
    st.markdown(recomendacion)
    st.markdown("---")

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
# FASE 3: GENERACIÓN DE REPORTES PDF EN MEMORIA (ESTRUCTURA DE CONSULTORÍA)
# ==============================================================================

def create_pdf_report(report_data: dict) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    
    # Título Principal
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, txt="REPORTE DE RIESGO CUANTITATIVO", ln=True, align='C')
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 8, txt=f"Generado el: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
    pdf.ln(5)

    # 1. Parámetros
    pdf.set_font("Arial", 'B', 12)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 8, txt=" 1. PARAMETROS DEL ACTIVO Y MODELO", ln=True, fill=True)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 7, txt=f"   - Activo Analizado (Ticker): {report_data['ticker']}", ln=True)
    pdf.cell(0, 7, txt=f"   - Precio Spot Actual (S0): ${report_data['S0']:.2f}", ln=True)
    pdf.cell(0, 7, txt=f"   - Volatilidad Anualizada (Sigma): {report_data['sigma']*100:.2f}%", ln=True)
    pdf.cell(0, 7, txt=f"   - Horizonte de Proyeccion: {report_data['days']} dias habiles", ln=True)
    pdf.cell(0, 7, txt=f"   - Simulaciones (Montecarlo): {report_data['simulations']}", ln=True)
    pdf.ln(5)

    # 2. Riesgo
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, txt=" 2. METRICAS DE RIESGO ESTOCASTICO", ln=True, fill=True)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 7, txt=f"   - Probabilidad de Retorno Positivo: {report_data['prob_pos']*100:.2f}%", ln=True)
    pdf.cell(0, 7, txt=f"   - Value at Risk (VaR): ${report_data['var_price']:.2f}", ln=True)
    pdf.cell(0, 7, txt=f"   - Conditional VaR (CVaR): ${report_data['cvar_price']:.2f}", ln=True)
    pdf.ln(5)

    # 3. Conclusiones (Texto extendido)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, txt=" 3. ANALISIS Y DIRECTRIZ ESTRATEGICA", ln=True, fill=True)
    pdf.ln(2)
    
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 7, txt=f"ESTADO DEL MERCADO: {report_data['estado']}", ln=True)
    pdf.ln(2)
    
    pdf.set_font("Arial", '', 10)
    # Limpiar formato Markdown (**) para que FPDF no lo imprima feo
    clean_text = report_data['recomendacion'].replace('**', '')
    
    # Imprimir párrafo por párrafo
    paragraphs = clean_text.split('\n\n')
    for p in paragraphs:
        if p.strip():
            pdf.multi_cell(0, 6, txt=p.strip())
            pdf.ln(3)

    pdf.ln(5)
    pdf.set_font("Arial", 'I', 8)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 4, txt="DISCLAIMER: Este reporte es generado mediante algoritmos cuantitativos (Movimiento Browniano Geometrico y Saltos de Merton) con fines academicos y de analisis de escenarios. No constituye asesoria financiera ni recomendacion de compra/venta. El mercado posee variables exogenas imposibles de modelar matematicamente con precision absoluta.")

    return pdf.output(dest='S').encode('latin-1')

if __name__ == "__main__":
    report_data = render_dashboard()
    if report_data:
        with st.sidebar:
            st.markdown("---")
            st.subheader("📄 Reporte Ejecutivo")
            with st.spinner("Generando PDF..."):
                pdf_bytes = create_pdf_report(report_data)
                st.download_button(
                    label="📥 Descargar PDF Analítico",
                    data=pdf_bytes,
                    file_name=f"Quant_Risk_Report_{report_data['ticker']}.pdf",
                    mime="application/pdf",
                    type="primary"
                )
