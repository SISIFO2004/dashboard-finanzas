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
# FASE 2: MOTOR MATEMÁTICO ESTOCÁSTICO Y TRADING
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
    median_price = np.median(final_prices) 
    tp_price = np.percentile(final_prices, 90)
    
    return prob_positive, var_price, var_loss_pct, cvar_price, cvar_loss_pct, median_price, tp_price

def generate_trading_signal(prob_pos, sharpe):
    if prob_pos >= 0.65 and sharpe >= 1.0:
        return "🟢 COMPRA FUERTE", "Condiciones excelentes. El activo tiene tendencia a subir y el riesgo está justificado por la posible ganancia."
    elif prob_pos >= 0.55 and sharpe > 0:
        return "🟡 COMPRA CAUTA", "Hay posibilidades de ganar, pero el riesgo es alto. Invierte poco dinero si decides entrar."
    elif prob_pos <= 0.35:
        return "🔴 VENTA / ALERTA", "El activo está cayendo con fuerza. Vende tus acciones o evita comprar por ahora."
    else:
        return "⚪ MANTENER / ESPERAR", "El mercado está indeciso. No es un buen momento para tomar decisiones arriesgadas."

# --- NUEVO MOTOR DE LENGUAJE COMÚN ---
def generate_directive_common(prob_pos, sigma, var_loss_pct, days, capital, rend_esp, tp_price, var_price, conf_level):
    vol = sigma * 100
    riesgo = abs(var_loss_pct) * 100
    
    # Textos comunes
    intro_txt = f"Basado en {days} días de proyección y múltiples escenarios simulados, esta es la lectura de tu inversión de {capital:,.2f}:\n"
    crecimiento_txt = f"• **Tu dinero a futuro:** En un escenario normal (mediana), podrías ver un cambio de **{rend_esp:+,.2f}** en tu capital.\n"
    
    if prob_pos > 0.65:
        estado = "Alcista (Positivo)"
        rec = intro_txt
        rec += f"• **¿Cuándo Comprar?:** ¡Ahora mismo es un buen momento! El modelo indica un **{prob_pos*100:.0f}% de probabilidad** de que el precio suba impulsado por su tendencia natural (Drift).\n"
        rec += f"• **¿Cuándo Vender y Cobrar (Take Profit)?:** Si el precio toca los **${tp_price:,.2f}**, el modelo sugiere que has alcanzado el límite superior esperado. Vende una parte para asegurar tus ganancias.\n"
        rec += f"• **¿Cuándo Vender y Salir (Stop-Loss)?:** Si ocurre algo malo en el mercado, vende **todo** si el precio cae a **${var_price:,.2f}**. (Esto representa el límite de pérdida máxima calculada o VaR al {conf_level}%).\n"
        rec += crecimiento_txt
        rec += f"• **Consejo sobre Riesgo:** El activo se mueve con una volatilidad (cambios de precio) del {vol:.1f}%. Es seguro mantener, pero respeta estrictamente el precio de salida (${var_price:,.2f}) para no perder más de la cuenta."
        
    elif prob_pos < 0.35:
        estado = "Bajista (Peligro)"
        rec = intro_txt
        rec += f"• **¿Cuándo Comprar?:** **No compres ahora.** El modelo detecta un {100 - prob_pos*100:.0f}% de probabilidad de que el precio siga cayendo.\n"
        rec += f"• **¿Cuándo Vender (Urgente)?:** Si ya tienes acciones, considera vender de inmediato. Si decides aguantar, tu límite absoluto de salida debe ser **${var_price:,.2f}**.\n"
        rec += crecimiento_txt
        rec += f"• **Consejo sobre Riesgo:** El mercado está castigando a este activo. La volatilidad ({vol:.1f}%) está empujando el precio hacia abajo de forma agresiva. Proteger tu dinero es más importante que buscar ganancias aquí."
        
    else:
        estado = "Neutral (Indeciso)"
        rec = intro_txt
        rec += f"• **¿Cuándo Comprar?:** Si vas a largo plazo, puedes comprar de a poco cada semana (estrategia Dollar-Cost Averaging). Si buscas ganancias rápidas, **espera**. El mercado no sabe si subir o bajar.\n"
        rec += f"• **¿Cuándo Vender (Take Profit)?:** Aprovecha los rebotes cortos. Si el precio sube a **${tp_price:,.2f}**, vende y toma tu ganancia antes de que vuelva a caer.\n"
        rec += f"• **¿Cuándo Vender (Stop-Loss)?:** Si el activo se rompe y cae hasta los **${var_price:,.2f}**, abandona la inversión inmediatamente.\n"
        rec += crecimiento_txt
        rec += f"• **Consejo sobre Riesgo:** El activo está estancado. Usa el precio límite de pérdida (${var_price:,.2f}) como tu seguro de vida financiero."

    return estado, rec

# ==============================================================================
# UI Y DASHBOARD
# ==============================================================================

def render_dashboard():
    st.set_page_config(page_title="Quant Risk Engine", layout="wide", page_icon="📈")
    st.title("📊 Análisis Cuantitativo de Inversión")
    st.markdown("Proyecciones matemáticas traducidas a estrategias simples.")

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
    
    st.sidebar.divider()
    st.sidebar.header("3. Simulación de Capital")
    moneda_str = st.sidebar.selectbox("Moneda Base:", ["USD ($)", "EUR (€)"])
    simbolo = "$" if "USD" in moneda_str else "€"
    codigo_moneda = "USD" if "USD" in moneda_str else "EUR" 
    capital_inicial = st.sidebar.number_input(f"Capital a Invertir ({simbolo}):", min_value=10.0, value=10000.0, step=1000.0)

    st.sidebar.divider()
    st.sidebar.header("4. Parámetros del Modelo")
    days_to_project = st.sidebar.slider("Días a Proyectar:", 10, 252, 60)
    simulations = {"1k": 1000, "5k": 5000, "10k": 10000}[st.sidebar.selectbox("Simulaciones (Montecarlo):", ["1k", "5k", "10k"])]
    conf_level = st.sidebar.slider("Nivel VaR (%) - Límite de Riesgo:", 90.0, 99.9, 95.0, 0.1)
    
    risk_free_rate = 0.045 

    with st.sidebar.expander("📉 Calibración Avanzada (Jumps & Drift)", expanded=False):
        override_drift = st.checkbox("Forzar Tendencia (μ) Manual")
        manual_drift = st.number_input("Tendencia Anualizada (μ):", value=0.10, step=0.01) if override_drift else None
        lambda_j = st.slider("Prob. Saltos (Cisnes Negros) λ:", 0.0, 10.0, 2.0)
        mu_j = st.number_input("Impacto del Salto (μ_J):", value=-0.05, step=0.01)
        sigma_j = st.number_input("Volatilidad del Salto (σ_J):", value=0.05, step=0.01)

    # --- EJECUCIÓN CORE ---
    df_hist = load_financial_data(ticker, tiingo_key_input)
    if df_hist.empty:
        df_hist = generate_synthetic_data(ticker, days=500)
    else:
        st.caption(f"Conexión de datos: {df_hist['Source'].iloc[0]}")
    
    daily_returns = df_hist['Close'].pct_change().dropna()
    hist_mu = daily_returns.mean() * 252
    hist_sigma = daily_returns.std() * np.sqrt(252)
    S0 = df_hist['Close'].iloc[-1]
    
    mu = manual_drift if override_drift else hist_mu
    sigma = hist_sigma
    sharpe_ratio = (mu - risk_free_rate) / sigma if sigma > 0 else 0
    
    with st.spinner("🚀 Calculando proyecciones a futuro..."):
        paths = run_montecarlo_jumps(S0, mu, sigma, days_to_project, simulations, lambda_j, mu_j, sigma_j)
        final_prices = paths[-1, :]
        prob_pos, var_price, var_loss, cvar_price, cvar_loss, median_price, tp_price = calculate_risk_metrics(S0, final_prices, conf_level)

    # --- CÁLCULOS DE PORTAFOLIO REAL ---
    acciones_adquiridas = capital_inicial / S0
    capital_esperado = acciones_adquiridas * median_price
    capital_var = acciones_adquiridas * var_price
    rendimiento_esperado = capital_esperado - capital_inicial

    # --- NIVELES OPERATIVOS (TRADING PLAN) ---
    st.subheader(f"Plan de Acción para {ticker} ({days_to_project} días)")
    señal, señal_desc = generate_trading_signal(prob_pos, sharpe_ratio)
    
    col_sig, col_tp, col_sl = st.columns(3)
    with col_sig:
        st.info(f"**CONSEJO PRINCIPAL:**\n\n{señal}\n\n_{señal_desc}_")
    with col_tp:
        st.success(f"**🟢 PRECIO PARA COBRAR GANANCIA:**\n\n**${tp_price:,.2f}**\n\n_(Take Profit al 90%)_")
    with col_sl:
        st.error(f"**🔴 PRECIO PARA HUIR SIN PERDER TODO:**\n\n**${var_price:,.2f}**\n\n_(Stop-Loss / Límite de Riesgo)_")

    # --- DIRECTRICES EN LENGUAJE COMÚN ---
    estado, recomendacion = generate_directive_common(prob_pos, sigma, var_loss, days_to_project, capital_inicial, rendimiento_esperado, tp_price, var_price, conf_level)
    
    st.markdown("---")
    if "Alcista" in estado: st.success(f"### 📈 Escenario General: {estado}")
    elif "Bajista" in estado: st.error(f"### 📉 Escenario General: {estado}")
    else: st.warning(f"### ⚖️ Escenario General: {estado}")
    st.markdown(recomendacion)
    st.markdown("---")

    # --- KPIs DEL PORTAFOLIO ---
    st.subheader(f"¿Qué pasará con tus {simbolo}{capital_inicial:,.0f}?")
    p1, p2, p3 = st.columns(3)
    p1.metric("Dinero Invertido", f"{simbolo}{capital_inicial:,.2f}", f"Compraste {acciones_adquiridas:.2f} unidades", delta_color="off")
    p2.metric("Escenario Probable", f"{simbolo}{capital_esperado:,.2f}", f"{simbolo}{rendimiento_esperado:,.2f} (Ganancia estimada)")
    p3.metric("Peor Escenario Calculado", f"{simbolo}{capital_var:,.2f}", f"{simbolo}{capital_var - capital_inicial:,.2f} (Pérdida máxima)", delta_color="inverse")

    # --- GRÁFICO ---
    fig = go.Figure()
    visual_paths = paths[:, :100]
    time_axis = np.arange(days_to_project + 1)
    
    for i in range(visual_paths.shape[1]):
        fig.add_trace(go.Scatter(x=time_axis, y=visual_paths[:, i], mode='lines', line=dict(color='rgba(0, 100, 255, 0.1)'), showlegend=False, hoverinfo='skip'))
    
    fig.add_trace(go.Scatter(x=[0, days_to_project], y=[S0, S0], mode='lines', name='Precio Actual (Tu entrada)', line=dict(color='black', width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=[0, days_to_project], y=[tp_price, tp_price], mode='lines', name='Precio para Cobrar (Take Profit)', line=dict(color='green', width=2, dash='dashdot')))
    fig.add_trace(go.Scatter(x=[0, days_to_project], y=[var_price, var_price], mode='lines', name=f'Precio para Huir (Stop-Loss)', line=dict(color='red', width=2, dash='dot')))
    
    fig.update_layout(title=f"Visión del Futuro: {simulations} caminos posibles calculados", height=400, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    return {
        "ticker": ticker, "S0": S0, "sigma": sigma, "days": days_to_project, "simulations": simulations,
        "prob_pos": prob_pos, "var_price": var_price, "var_loss": var_loss, "conf_level": conf_level,
        "capital_inicial": capital_inicial, "moneda": codigo_moneda, "acciones": acciones_adquiridas,
        "capital_esperado": capital_esperado, "capital_var": capital_var, "rendimiento_esperado": rendimiento_esperado,
        "estado": estado, "recomendacion": recomendacion,
        "sharpe": sharpe_ratio, "tp_price": tp_price, "signal": señal
    }

# ==============================================================================
# FASE 3: GENERACIÓN PDF (FORMATO GUÍA DE INVERSIÓN)
# ==============================================================================

def create_pdf_report(data: dict) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    
    # TITULO Y CABECERA
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, txt=f"GUIA DE INVERSION: {data['ticker']}", ln=True, align='C')
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 8, txt=f"Fecha de analisis: {datetime.datetime.now().strftime('%Y-%m-%d')}", ln=True, align='C')
    pdf.ln(5)

    # 1. PLAN DE ACCION DIRECTO
    pdf.set_font("Arial", 'B', 12)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 8, txt=" 1. TU PLAN DE ACCION EXACTO", ln=True, fill=True)
    pdf.set_font("Arial", 'B', 10)
    
    signal_clean = data['signal'].replace('🟢', '').replace('🟡', '').replace('🔴', '').replace('⚪', '').strip()
    pdf.cell(0, 6, txt=f"   - SEÑAL DEL SISTEMA: {signal_clean}", ln=True)
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 6, txt=f"   - Precio Actual de Compra: ${data['S0']:.2f}", ln=True)
    pdf.cell(0, 6, txt=f"   - Cuando Vender y Cobrar: Si el precio toca los ${data['tp_price']:.2f}", ln=True)
    pdf.cell(0, 6, txt=f"   - Cuando Vender y Huir: Si el precio cae a los ${data['var_price']:.2f}", ln=True)
    pdf.ln(5)

    # 2. TU DINERO
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, txt=f" 2. QUE PASARA CON TU DINERO ({data['days']} DIAS)", ln=True, fill=True)
    pdf.set_font("Arial", '', 10)
    moneda_pdf = "EUR" if data['moneda'] == "EUR" else "USD"
    
    pdf.cell(0, 6, txt=f"   - Inversion Inicial: {data['capital_inicial']:,.2f} {moneda_pdf}", ln=True)
    pdf.cell(0, 6, txt=f"   - El Escenario Mas Probable: Tu dinero cambiaria en {data['rendimiento_esperado']:+,.2f} {moneda_pdf}", ln=True)
    pdf.cell(0, 6, txt=f"   - El Peor Escenario (Peligro): Podrias llegar a perder {data['capital_inicial'] - data['capital_var']:,.2f} {moneda_pdf}", ln=True)
    pdf.ln(5)

    # 3. EXPLICACIÓN COMÚN
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, txt=" 3. CONSEJOS E INTERPRETACION", ln=True, fill=True)
    pdf.set_font("Arial", 'B', 10)
    
    estado_limpio = data['estado'].upper().replace('Ó', 'O').replace('Í', 'I')
    pdf.cell(0, 6, txt=f"   ESCENARIO: {estado_limpio}", ln=True)
    pdf.set_font("Arial", '', 10)
    
    # Sanitizado de texto para el PDF
    clean_text = data['recomendacion'].replace('**', '').replace('•', '-')
    sustituciones = {'á':'a', 'é':'e', 'í':'i', 'ó':'o', 'ú':'u', 'Á':'A', 'É':'E', 'Í':'I', 'Ó':'O', 'Ú':'U'}
    for acento, sin_acento in sustituciones.items():
        clean_text = clean_text.replace(acento, sin_acento)

    for p in clean_text.split('\n'):
        if p.strip():
            pdf.multi_cell(0, 6, txt="   " + p.strip())
            pdf.ln(1)

    # EXPLICACION MATEMATICA AL FINAL
    pdf.ln(5)
    pdf.set_font("Arial", 'I', 8)
    pdf.set_text_color(100, 100, 100)
    diccionario_txt = (
        "DICCIONARIO MATEMATICO: "
        "- Simulacion Montecarlo: El sistema calcula miles de futuros posibles lanzando "
        "dados matematicos basados en el pasado del activo. "
        "- Drift: Es la tendencia natural del activo (hacia arriba o abajo). "
        f"- VaR (Value at Risk): En este reporte fijado al {data['conf_level']}%, "
        "es una linea de peligro matematico. Si el precio cae de ahi, significa "
        "que algo se ha roto estructuralmente y debes vender para protegerte."
    )
    pdf.multi_cell(0, 4, txt=diccionario_txt)

    pdf_string = pdf.output(dest='S')
    return pdf_string.encode('latin-1', errors='replace')

if __name__ == "__main__":
    report_data = render_dashboard()
    if report_data:
        with st.sidebar:
            st.markdown("---")
            with st.spinner("Creando tu Guía Personalizada..."):
                pdf_bytes = create_pdf_report(report_data)
                st.download_button(
                    label="📥 Descargar tu Plan en PDF",
                    data=pdf_bytes,
                    file_name=f"Guia_Inversion_{report_data['ticker']}.pdf",
                    mime="application/pdf",
                    type="primary"
                )
