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
from pykalman import KalmanFilter

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
# FASE 2: FILTRO DE KALMAN Y MOTOR ESTOCÁSTICO
# ==============================================================================

def apply_kalman_filter(returns_array):
    """
    Aplica el Filtro de Kalman usando el algoritmo Expectation-Maximization 
    para extraer la tendencia oculta (Drift) eliminando el ruido del mercado.
    """
    kf = KalmanFilter(transition_matrices=[1],
                      observation_matrices=[1],
                      initial_state_mean=returns_array.mean(),
                      initial_state_covariance=returns_array.var(),
                      observation_covariance=returns_array.var(),
                      transition_covariance=returns_array.var() / 100)
    
    # Auto-calibración matemática (EM)
    kf = kf.em(returns_array, n_iter=5)
    state_means, _ = kf.filter(returns_array)
    
    # El último estado es la tendencia "verdadera" purificada
    kalman_daily_drift = state_means[-1][0]
    return kalman_daily_drift

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
    median_price = np.median(final_prices) 
    tp_price = np.percentile(final_prices, 90)
    return prob_positive, var_price, var_loss_pct, cvar_price, cvar_loss_pct, median_price, tp_price

def generate_trading_signal(prob_pos, sharpe):
    if prob_pos >= 0.65 and sharpe >= 1.0:
        return "🟢 COMPRA FUERTE", "Condiciones excelentes. El activo tiene tendencia a subir y el riesgo está justificado."
    elif prob_pos >= 0.55 and sharpe > 0:
        return "🟡 COMPRA CAUTA", "Hay posibilidades de ganar, pero el riesgo es alto. Invierte con precaución."
    elif prob_pos <= 0.35:
        return "🔴 VENTA / ALERTA", "El activo está en deterioro estructural. Busca salidas seguras."
    else:
        return "⚪ MANTENER / ESPERAR", "Mercado con alta entropía direccional. El riesgo no compensa el rendimiento."

def generate_directive_common(prob_pos, sigma, var_loss_pct, days, capital, rend_esp, tp_price, var_price, conf_level):
    vol = sigma * 100
    riesgo = abs(var_loss_pct) * 100
    intro_txt = f"Basado en {days} días de proyección estocástica para tu inversión de {capital:,.2f}:\n"
    crecimiento_txt = f"• **Tu capital a futuro:** En el escenario probabilístico medio, proyectas un cambio de **{rend_esp:+,.2f}**.\n"
    
    if prob_pos > 0.65:
        estado = "Alcista (Positivo)"
        rec = intro_txt
        rec += f"• **Señal de Compra:** Favorable ({prob_pos*100:.0f}% prob. éxito). El Drift estructural empuja el precio al alza.\n"
        rec += f"• **Take Profit (Cobro):** Si el precio toca **${tp_price:,.2f}**, el algoritmo indica sobrecompra. Asegura liquidez.\n"
        rec += f"• **Stop-Loss (Corte):** Liquidación obligatoria si cae a **${var_price:,.2f}** (Barrera VaR al {conf_level}%).\n"
        rec += crecimiento_txt
    elif prob_pos < 0.35:
        estado = "Bajista (Peligro)"
        rec = intro_txt
        rec += f"• **Señal de Compra:** Bloqueada. ({100 - prob_pos*100:.0f}% prob. de continuación a la baja).\n"
        rec += f"• **Acción Inmediata:** Si hay exposición, liquidar en rebotes. Soporte crítico en **${var_price:,.2f}**.\n"
        rec += crecimiento_txt
    else:
        estado = "Neutral (Indeciso)"
        rec = intro_txt
        rec += f"• **Señal de Compra:** Retenida. Riesgo simétrico.\n"
        rec += f"• **Take Profit (Cobro):** Vender picos en **${tp_price:,.2f}**.\n"
        rec += f"• **Stop-Loss (Corte):** Proteger capital estrictamente en **${var_price:,.2f}**.\n"
        rec += crecimiento_txt

    return estado, rec

# ==============================================================================
# UI Y DASHBOARD
# ==============================================================================

def render_dashboard():
    st.set_page_config(page_title="Quant Risk Engine", layout="wide", page_icon="📈")
    st.title("📊 Motor Cuantitativo de Riesgo")
    st.markdown("Procesamiento Estocástico y Filtrado de Ruido (Kalman)")

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
    
    st.sidebar.divider()
    st.sidebar.header("3. Simulación de Capital")
    moneda_str = st.sidebar.selectbox("Moneda Base:", ["USD ($)", "EUR (€)"])
    simbolo = "$" if "USD" in moneda_str else "€"
    codigo_moneda = "USD" if "USD" in moneda_str else "EUR" 
    capital_inicial = st.sidebar.number_input(f"Capital ({simbolo}):", min_value=10.0, value=10000.0, step=1000.0)

    st.sidebar.divider()
    st.sidebar.header("4. Arquitectura del Modelo")
    days_to_project = st.sidebar.slider("Días de Proyección:", 10, 252, 60)
    simulations = {"1k": 1000, "5k": 5000, "10k": 10000}[st.sidebar.selectbox("Simulaciones:", ["1k", "5k", "10k"])]
    conf_level = st.sidebar.slider("Límite VaR (%):", 90.0, 99.9, 95.0, 0.1)
    
    st.sidebar.divider()
    st.sidebar.header("5. Procesamiento Avanzado")
    use_kalman = st.sidebar.checkbox("🧠 Activar Filtro de Kalman (Filtrar Ruido)", value=True)
    
    with st.sidebar.expander("📉 Inyección Estocástica (Saltos)", expanded=False):
        lambda_j = st.slider("Prob. Saltos (λ):", 0.0, 10.0, 2.0)
        mu_j = st.number_input("Impacto (μ_J):", value=-0.05, step=0.01)
        sigma_j = st.number_input("Vol. Salto (σ_J):", value=0.05, step=0.01)

    # --- INGESTA Y CÁLCULO ---
    df_hist = load_financial_data(ticker, tiingo_key_input)
    if df_hist.empty:
        df_hist = generate_synthetic_data(ticker, days=500)
    else:
        st.caption(f"Conexión estable: {df_hist['Source'].iloc[0]}")
    
    daily_returns = df_hist['Close'].pct_change().dropna()
    raw_mu = daily_returns.mean() * 252
    sigma = daily_returns.std() * np.sqrt(252) # Mantenemos volatilidad cruda para no subestimar el riesgo extremo
    S0 = df_hist['Close'].iloc[-1]
    
    # --- FILTRO DE KALMAN ---
    ruido_eliminado = 0
    if use_kalman:
        with st.spinner("Procesando Filtro de Kalman..."):
            kalman_daily_mu = apply_kalman_filter(daily_returns.values)
            mu = kalman_daily_mu * 252
            ruido_eliminado = abs(raw_mu - mu) * 100
    else:
        mu = raw_mu

    risk_free_rate = 0.045 
    sharpe_ratio = (mu - risk_free_rate) / sigma if sigma > 0 else 0
    
    with st.spinner("Ejecutando Montecarlo Vectorizado..."):
        paths = run_montecarlo_jumps(S0, mu, sigma, days_to_project, simulations, lambda_j, mu_j, sigma_j)
        final_prices = paths[-1, :]
        prob_pos, var_price, var_loss, cvar_price, cvar_loss, median_price, tp_price = calculate_risk_metrics(S0, final_prices, conf_level)

    # --- KPIs ---
    acciones_adquiridas = capital_inicial / S0
    capital_esperado = acciones_adquiridas * median_price
    capital_var = acciones_adquiridas * var_price
    rendimiento_esperado = capital_esperado - capital_inicial

    st.subheader(f"Telemetría del Activo: {ticker}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precio Spot", f"${S0:,.2f}")
    c2.metric("Tendencia (Drift Anual)", f"{mu*100:.1f}%", f"{ruido_eliminado:.1f}% Ruido Filtrado" if use_kalman else "Datos Crudos", delta_color="normal" if use_kalman else "off")
    c3.metric(f"Ratio de Sharpe", f"{sharpe_ratio:.2f}")
    c4.metric("Volatilidad (σ)", f"{sigma*100:.1f}%")

    st.subheader(f"Plan de Ejecución ({days_to_project} días)")
    señal, señal_desc = generate_trading_signal(prob_pos, sharpe_ratio)
    
    col_sig, col_tp, col_sl = st.columns(3)
    with col_sig:
        st.info(f"**DIRECTRIZ:**\n\n{señal}\n\n_{señal_desc}_")
    with col_tp:
        st.success(f"**🟢 TAKE PROFIT:**\n\n**${tp_price:,.2f}**\n\n_Venta parcial o total._")
    with col_sl:
        st.error(f"**🔴 STOP-LOSS:**\n\n**${var_price:,.2f}**\n\n_Cierre de emergencia._")

    estado, recomendacion = generate_directive_common(prob_pos, sigma, var_loss, days_to_project, capital_inicial, rendimiento_esperado, tp_price, var_price, conf_level)
    st.markdown("---")
    st.markdown(recomendacion)
    st.markdown("---")

    # --- PORTAFOLIO Y GRÁFICA ---
    p1, p2, p3 = st.columns(3)
    p1.metric("Capital Invertido", f"{simbolo}{capital_inicial:,.2f}")
    p2.metric("Valor Esperado", f"{simbolo}{capital_esperado:,.2f}", f"{simbolo}{rendimiento_esperado:,.2f}")
    p3.metric("Capital en Riesgo (VaR)", f"{simbolo}{capital_var:,.2f}", f"{simbolo}{capital_var - capital_inicial:,.2f}", delta_color="inverse")

    fig = go.Figure()
    visual_paths = paths[:, :100]
    time_axis = np.arange(days_to_project + 1)
    
    for i in range(visual_paths.shape[1]):
        fig.add_trace(go.Scatter(x=time_axis, y=visual_paths[:, i], mode='lines', line=dict(color='rgba(0, 100, 255, 0.1)'), showlegend=False, hoverinfo='skip'))
    
    fig.add_trace(go.Scatter(x=[0, days_to_project], y=[S0, S0], mode='lines', name='Entrada', line=dict(color='black', width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=[0, days_to_project], y=[tp_price, tp_price], mode='lines', name='Take Profit', line=dict(color='green', width=2, dash='dashdot')))
    fig.add_trace(go.Scatter(x=[0, days_to_project], y=[var_price, var_price], mode='lines', name='Stop-Loss (VaR)', line=dict(color='red', width=2, dash='dot')))
    
    fig.update_layout(title=f"Matriz de Montecarlo ({simulations} proyecciones)", height=400, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    return {
        "ticker": ticker, "S0": S0, "sigma": sigma, "days": days_to_project, "simulations": simulations,
        "prob_pos": prob_pos, "var_price": var_price, "var_loss": var_loss, "conf_level": conf_level,
        "capital_inicial": capital_inicial, "moneda": codigo_moneda, "acciones": acciones_adquiridas,
        "capital_esperado": capital_esperado, "capital_var": capital_var, "rendimiento_esperado": rendimiento_esperado,
        "estado": estado, "recomendacion": recomendacion, "sharpe": sharpe_ratio, "tp_price": tp_price, "signal": señal,
        "kalman": use_kalman
    }

# ==============================================================================
# FASE 3: GENERACIÓN PDF
# ==============================================================================

def create_pdf_report(data: dict) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, txt=f"REPORTE CUANTITATIVO: {data['ticker']}", ln=True, align='C')
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 8, txt=f"Fecha: {datetime.datetime.now().strftime('%Y-%m-%d')} | Filtro Kalman: {'ACTIVO' if data['kalman'] else 'INACTIVO'}", ln=True, align='C')
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 12)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 8, txt=" 1. NIVELES OPERATIVOS", ln=True, fill=True)
    pdf.set_font("Arial", 'B', 10)
    signal_clean = data['signal'].replace('🟢', '').replace('🟡', '').replace('🔴', '').replace('⚪', '').strip()
    pdf.cell(0, 6, txt=f"   - SEÑAL: {signal_clean}", ln=True)
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 6, txt=f"   - Entrada: ${data['S0']:.2f}", ln=True)
    pdf.cell(0, 6, txt=f"   - Take Profit: ${data['tp_price']:.2f}", ln=True)
    pdf.cell(0, 6, txt=f"   - Stop-Loss: ${data['var_price']:.2f}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, txt=" 2. EXPOSICION DE CAPITAL", ln=True, fill=True)
    pdf.set_font("Arial", '', 10)
    m = "EUR" if data['moneda'] == "EUR" else "USD"
    pdf.cell(0, 6, txt=f"   - Inversion Inicial: {data['capital_inicial']:,.2f} {m}", ln=True)
    pdf.cell(0, 6, txt=f"   - Retorno Esperado: {data['rendimiento_esperado']:+,.2f} {m}", ln=True)
    pdf.cell(0, 6, txt=f"   - Riesgo Maximo (VaR): {data['capital_inicial'] - data['capital_var']:,.2f} {m}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, txt=" 3. DIRECTRIZ", ln=True, fill=True)
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

if __name__ == "__main__":
    report_data = render_dashboard()
    if report_data:
        with st.sidebar:
            st.markdown("---")
            with st.spinner("Compilando PDF..."):
                pdf_bytes = create_pdf_report(report_data)
                st.download_button(
                    label="📥 Descargar Reporte Kalman PDF",
                    data=pdf_bytes,
                    file_name=f"Quant_Kalman_{report_data['ticker']}.pdf",
                    mime="application/pdf",
                    type="primary"
                )
