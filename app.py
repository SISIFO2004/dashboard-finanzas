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
            columna_precio = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
            df = df[[columna_precio]].copy()
            df = df.rename(columns={columna_precio: 'Close'})
            df['Source'] = 'Yahoo Finance (Ajustado)'
            return df
    except Exception: pass

    return pd.DataFrame() 

# ==============================================================================
# FASE 2: MOTOR MATEMÁTICO (KALMAN + VOLATILIDAD DINÁMICA)
# ==============================================================================

def apply_kalman_filter_dynamic(returns_array, divisor_inercia):
    n = len(returns_array)
    R = np.var(returns_array)
    if R == 0: return np.full(n, returns_array[-1])
    Q = R / float(divisor_inercia) 
    x_hat = np.zeros(n)
    P = np.zeros(n)
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
    diff_residuals = np.diff(residuals)
    suma_cuadrados_res = np.sum(residuals**2)
    dw_stat = np.sum(diff_residuals**2) / suma_cuadrados_res if suma_cuadrados_res != 0 else 0
    penalty_dw = abs(dw_stat - 2.0) * 20
    shock_array = np.array([0.001, 0.001, 0.001, 0.001, 0.001, -0.05, -0.05, -0.01, -0.01, -0.01])
    shock_states = apply_kalman_filter_dynamic(shock_array, divisor_inercia)
    lag_days = 5 
    for i in range(5, len(shock_states)):
        if shock_states[i] < 0:
            lag_days = i - 4 
            break
    penalty_lag = 1000 if lag_days >= 3 else 0
    return penalty_svr + penalty_dw + penalty_lag, svr, dw_stat, lag_days

def optimize_kalman_filter(returns_array):
    candidatos = [10, 20, 50, 100, 200, 300, 500]
    mejor_divisor, menor_error, mejores_metricas = 100, float('inf'), {}
    for divisor in candidatos:
        error, svr, dw, lag = calculate_kalman_score(returns_array, divisor)
        if error < menor_error:
            menor_error, mejor_divisor, mejores_metricas = error, divisor, {'svr': svr, 'dw': dw, 'lag': lag, 'error': error}
    return mejor_divisor, mejores_metricas

def apply_stochastic_volatility_filter(returns_array, q_variance=0.1):
    n = len(returns_array)
    y_star = np.log(returns_array**2 + 1e-8)
    R_vol = 4.93 
    Q_vol = q_variance 
    h_hat = np.zeros(n) 
    P_vol = np.zeros(n) 
    h_hat[0] = np.mean(y_star) if n > 0 else 0.0
    P_vol[0] = 1.0
    for k in range(1, n):
        h_minus = h_hat[k-1]
        P_minus = P_vol[k-1] + Q_vol
        K = P_minus / (P_minus + R_vol)
        h_hat[k] = h_minus + K * (y_star[k] - h_minus)
        P_vol[k] = (1 - K) * P_minus
    dynamic_sigma = np.exp(h_hat / 2.0)
    dynamic_sigma = np.clip(dynamic_sigma, 0.0001, 1.0)
    return dynamic_sigma

# ==============================================================================
# FASE 3: SIMULACIÓN ESTOCÁSTICA AVANZADA (DRIFT + VOLATILIDAD + COLAS PESADAS)
# ==============================================================================

def run_montecarlo_advanced_stochastic(S0, current_mu, long_term_mu, current_sigma, long_term_sigma, days, simulations, lambda_j, mu_j, sigma_j):
    """
    Simulador Avanzado Estocástico.
    - Drift con Reversión a la Media (Frena burbujas).
    - Volatilidad Dinámica (Heston-Lite).
    - Colas Pesadas (t-Student) para Cisnes Negros.
    """
    dt = 1 / 252  
    
    # Parámetros de Reversión para la Volatilidad
    kappa_v = 5.0    
    vol_v = 0.2  
    
    # Parámetros de Reversión para la Tendencia (Drift)
    kappa_mu = 2.0 # Velocidad a la que la euforia regresa a la normalidad
    vol_mu = 0.1   # Ruido intrínseco de la tendencia
    
    price_paths = np.zeros((days + 1, simulations))
    price_paths[0] = S0
    
    # Vectores de estado iniciales
    v_t = np.full(simulations, current_sigma)
    mu_t = np.full(simulations, current_mu)
    
    # 1. Ruido para Volatilidad y Drift (Gausiano clásico)
    Z_vol = np.random.standard_normal((days, simulations))
    Z_mu = np.random.standard_normal((days, simulations))
    
    # 2. Ruido para Precio: Distribución t-Student (Colas Pesadas)
    # df=4 da colas extremas. Multiplicamos por sqrt(0.5) para estandarizar la varianza a 1.
    Z_price = np.random.standard_t(df=4, size=(days, simulations)) * np.sqrt(0.5)
    
    # Proceso de Saltos (Poisson)
    poisson_jumps = np.random.poisson(lambda_j * dt, (days, simulations))
    jump_magnitudes = np.random.normal(mu_j, sigma_j, (days, simulations))
    jump_returns = poisson_jumps * jump_magnitudes
    
    for t in range(days):
        # A. Evolucionar Volatilidad Estocástica
        v_t = np.abs(v_t + kappa_v * (long_term_sigma - v_t) * dt + vol_v * np.sqrt(v_t * dt) * Z_vol[t])
        
        # B. Evolucionar Drift Estocástico (Freno de Euforia)
        mu_t = mu_t + kappa_mu * (long_term_mu - mu_t) * dt + vol_mu * np.sqrt(dt) * Z_mu[t]
        
        # C. Proyección Final del Precio con Colas Pesadas
        gbm_returns = (mu_t - 0.5 * v_t**2) * dt + v_t * np.sqrt(dt) * Z_price[t]
        total_returns = gbm_returns + jump_returns[t]
        
        price_paths[t+1] = price_paths[t] * np.exp(total_returns)
        
    return price_paths

def calculate_risk_metrics(S0, final_prices, conf_level):
    prob_positive = np.mean(final_prices > S0)
    var_price = np.percentile(final_prices, 100 - conf_level)
    var_loss_pct = (var_price - S0) / S0
    cvar_price = np.mean(final_prices[final_prices <= var_price])
    if np.isnan(cvar_price): cvar_price = var_price
    median_price = np.median(final_prices) 
    tp_price = np.percentile(final_prices, 90)
    return prob_positive, var_price, var_loss_pct, cvar_price, (cvar_price - S0)/S0, median_price, tp_price

def generate_trading_signal(prob_pos, sharpe):
    if prob_pos >= 0.65 and sharpe >= 1.0: return "🟢 COMPRA FUERTE", "Condiciones excelentes. Tendencia estructural validada."
    elif prob_pos >= 0.55 and sharpe > 0: return "🟡 COMPRA CAUTA", "Riesgo alto. Precaución con la reversión a la media."
    elif prob_pos <= 0.35: return "🔴 VENTA / ALERTA", "Deterioro estructural. Presión bajista pesada."
    else: return "⚪ MANTENER / ESPERAR", "Alta entropía direccional (Mercado Lateral)."

def generate_directive_common(prob_pos, current_sigma, var_loss_pct, days, capital, rend_esp, tp_price, var_price, conf_level):
    intro_txt = f"Basado en {days} días de proyección estocástica pura (Drift + Vol + Colas Pesadas):\n"
    crecimiento_txt = f"• **Capital Proyectado (Media):** Cambio esperado de **{rend_esp:+,.2f}**.\n"
    if prob_pos > 0.65:
        estado = "Alcista"
        rec = intro_txt + f"• **Señal:** Favorable. • **Take Profit:** ${tp_price:,.2f}. • **Stop-Loss (VaR {conf_level}%):** ${var_price:,.2f}.\n" + crecimiento_txt
    elif prob_pos < 0.35:
        estado = "Bajista"
        rec = intro_txt + f"• **Señal:** Bloqueada. • **Soporte Crítico:** ${var_price:,.2f}.\n" + crecimiento_txt
    else:
        estado = "Neutral"
        rec = intro_txt + f"• **Señal:** Retenida. Entorno simétrico.\n" + crecimiento_txt
    return estado, rec

# ==============================================================================
# UI Y DASHBOARD (INTERFAZ DE USUARIO)
# ==============================================================================

def render_dashboard():
    st.set_page_config(page_title="Quant Risk Engine", layout="wide", page_icon="📈")
    st.title("📊 Motor Cuantitativo de Riesgo (Modo Backtest)")

    st.sidebar.header("1. Datos y Conexión")
    tiingo_key_input = st.sidebar.text_input("Tiingo API Key (Opcional):", type="password")
    
    st.sidebar.divider()
    st.sidebar.header("2. Selección de Activo")
    
    ASSET_UNIVERSE = {
        "🔍 Entrada Manual (Ticker)": "MANUAL",
        "--- ACCIONES DE EE.UU. ---": "HEADER",
        "🇺🇸 Apple Inc. (AAPL)": "AAPL",
        "🇺🇸 Microsoft Corp. (MSFT)": "MSFT",
        "🇺🇸 NVIDIA Corp. (NVDA)": "NVDA",
        "🇺🇸 Amazon.com Inc. (AMZN)": "AMZN",
        "🇺🇸 Alphabet Inc. (GOOGL)": "GOOGL",
        "🇺🇸 Meta Platforms (META)": "META",
        "🇺🇸 Tesla Inc. (TSLA)": "TSLA",
        "--- CRIPTOMONEDAS ---": "HEADER",
        "₿ Bitcoin (BTC-USD)": "BTC-USD",
        "⟠ Ethereum (ETH-USD)": "ETH-USD",
        "☀️ Solana (SOL-USD)": "SOL-USD",
        "🔶 Binance Coin (BNB-USD)": "BNB-USD",
        "--- MATERIAS PRIMAS ---": "HEADER",
        "🥇 Oro (GLD - ETF Trust)": "GLD",
        "🥈 Plata (SLV - ETF Trust)": "SLV",
        "🛢️ Petróleo Brent (USO - ETF)": "USO"
    }
    
    selected_asset = st.sidebar.selectbox("Seleccione un Activo:", list(ASSET_UNIVERSE.keys()))
    if ASSET_UNIVERSE[selected_asset] == "HEADER":
        st.warning("Seleccione un activo válido.")
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
    
    # VALOR POR DEFECTO A 21 DÍAS PARA LA PRUEBA EXACTA DE 1 MES
    days_to_project = st.sidebar.slider("Días de Proyección:", 10, 252, 21)
    simulations = {"1k": 1000, "5k": 5000, "10k": 10000}[st.sidebar.selectbox("Simulaciones:", ["1k", "5k", "10k"])]
    conf_level = st.sidebar.slider("Límite VaR (%):", 90.0, 99.9, 95.0, 0.1)
    
    st.sidebar.divider()
    st.sidebar.header("5. Procesamiento Avanzado")
    use_kalman = st.sidebar.checkbox("🧠 Activar Motor Avanzado (Deriva y Volatilidad)", value=True)
    
    with st.sidebar.expander("📉 Inyección Estocástica (Saltos)", expanded=False):
        lambda_j = st.slider("Prob. Saltos (λ):", 0.0, 10.0, 2.0)
        mu_j = st.number_input("Impacto (μ_J):", value=-0.05, step=0.01)
        sigma_j = st.number_input("Vol. Salto (σ_J):", value=0.05, step=0.01)

    df_hist = load_financial_data(ticker, tiingo_key_input)
    
    # === INICIO DE MÁQUINA DEL TIEMPO (-30 DÍAS CALENDARIO / 21 DÍAS HÁBILES) ===
    if not df_hist.empty:
        df_hist = df_hist.iloc[:-21]
    # === FIN DE MÁQUINA DEL TIEMPO ===
        
    if df_hist.empty:
        df_hist = generate_synthetic_data(ticker, days=500)
    else:
        st.caption(f"Conexión estable: {df_hist['Source'].iloc[0]} (Modo Backtest Activo)")
    
    daily_returns = df_hist['Close'].pct_change().dropna()
    S0 = df_hist['Close'].iloc[-1]
    
    # Cálculos Clásicos (Estáticos y Medias a Largo Plazo)
    raw_mu = daily_returns.mean() * 252
    static_sigma_ann = daily_returns.std() * np.sqrt(252)
    
    ruido_eliminado = 0
    divisor_usado = "Crudo"
    
    # --- FILTRADO AVANZADO ---
    if use_kalman:
        with st.spinner("Calibrando Algoritmos Estocásticos..."):
            mejor_divisor, metricas = optimize_kalman_filter(daily_returns.values)
            divisor_usado = f"D={mejor_divisor}"
            
            # Tendencia de Corto Plazo (Euforia actual)
            kalman_states = apply_kalman_filter_dynamic(daily_returns.values, divisor_inercia=mejor_divisor)
            current_mu = kalman_states[-1] * 252
            long_term_mu = raw_mu # El ancla de gravedad hacia donde revertirá
            ruido_eliminado = abs(raw_mu - current_mu) * 100
            
            # Volatilidad Dinámica
            dynamic_sigma_daily = apply_stochastic_volatility_filter(daily_returns.values)
            current_sigma_ann = dynamic_sigma_daily[-1] * np.sqrt(252)
            long_term_sigma_ann = np.mean(dynamic_sigma_daily) * np.sqrt(252)
    else:
        current_mu = raw_mu
        long_term_mu = raw_mu
        current_sigma_ann = static_sigma_ann
        long_term_sigma_ann = static_sigma_ann

    risk_free_rate = 0.045 
    sharpe_ratio = (current_mu - risk_free_rate) / current_sigma_ann if current_sigma_ann > 0 else 0
    
    with st.spinner("Ejecutando Simulación Avanzada (Drift Estocástico + Colas Pesadas)..."):
        paths = run_montecarlo_advanced_stochastic(S0, current_mu, long_term_mu, current_sigma_ann, long_term_sigma_ann, days_to_project, simulations, lambda_j, mu_j, sigma_j)
        final_prices = paths[-1, :]
        prob_pos, var_price, var_loss, cvar_price, cvar_loss, median_price, tp_price = calculate_risk_metrics(S0, final_prices, conf_level)

    # --- KPIs ---
    acciones_adquiridas = capital_inicial / S0
    capital_esperado = acciones_adquiridas * median_price
    capital_var = acciones_adquiridas * var_price
    rendimiento_esperado = capital_esperado - capital_inicial

    st.subheader(f"Telemetría del Activo: {ticker}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precio Base (Backtest)", f"${S0:,.2f}")
    c2.metric("Tendencia de Corto Plazo", f"{current_mu*100:.1f}%", f"{ruido_eliminado:.1f}% Ruido Filtrado ({divisor_usado})" if use_kalman else "Datos Crudos", delta_color="normal" if use_kalman else "off")
    c3.metric(f"Ratio de Sharpe", f"{sharpe_ratio:.2f}")
    
    vol_label = "Volatilidad Dinámica (σ)" if use_kalman else "Volatilidad Estática (σ)"
    vol_delta = f"Ancla Largo Plazo: {long_term_sigma_ann*100:.1f}%" if use_kalman else ""
    c4.metric(vol_label, f"{current_sigma_ann*100:.1f}%", vol_delta, delta_color="off")

    st.subheader(f"Plan de Ejecución ({days_to_project} días)")
    señal, señal_desc = generate_trading_signal(prob_pos, sharpe_ratio)
    
    col_sig, col_tp, col_sl = st.columns(3)
    with col_sig:
        st.info(f"**DIRECTRIZ:**\n\n{señal}\n\n_{señal_desc}_")
    with col_tp:
        st.success(f"**🟢 TAKE PROFIT:**\n\n**${tp_price:,.2f}**\n\n_Venta parcial o total._")
    with col_sl:
        st.error(f"**🔴 STOP-LOSS:**\n\n**${var_price:,.2f}**\n\n_Cierre de emergencia._")

    estado, recomendacion = generate_directive_common(prob_pos, current_sigma_ann, var_loss, days_to_project, capital_inicial, rendimiento_esperado, tp_price, var_price, conf_level)
    st.markdown("---")
    st.markdown(recomendacion)
    st.markdown("---")

    # --- PORTAFOLIO Y GRÁFICA ---
    p1, p2, p3 = st.columns(3)
    p1.metric("Capital Invertido", f"{simbolo}{capital_inicial:,.2f}")
    p2.metric("Valor Esperado (Mediana)", f"{simbolo}{capital_esperado:,.2f}", f"{simbolo}{rendimiento_esperado:,.2f}")
    p3.metric("Capital en Riesgo (VaR)", f"{simbolo}{capital_var:,.2f}", f"{simbolo}{capital_var - capital_inicial:,.2f}", delta_color="inverse")

    fig = go.Figure()
    visual_paths = paths[:, :100]
    time_axis = np.arange(days_to_project + 1)
    
    for i in range(visual_paths.shape[1]):
        fig.add_trace(go.Scatter(x=time_axis, y=visual_paths[:, i], mode='lines', line=dict(color='rgba(0, 100, 255, 0.1)'), showlegend=False, hoverinfo='skip'))
    
    fig.add_trace(go.Scatter(x=[0, days_to_project], y=[S0, S0], mode='lines', name='Entrada', line=dict(color='black', width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=[0, days_to_project], y=[tp_price, tp_price], mode='lines', name='Take Profit', line=dict(color='green', width=2, dash='dashdot')))
    fig.add_trace(go.Scatter(x=[0, days_to_project], y=[var_price, var_price], mode='lines', name='Stop-Loss (VaR)', line=dict(color='red', width=2, dash='dot')))
    
    fig.update_layout(title=f"Matriz de Supervivencia ({simulations} proyecciones con Reversión a la Media y Colas Pesadas)", height=400, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    return {
        "ticker": ticker, "S0": S0, "sigma": current_sigma_ann, "days": days_to_project, "simulations": simulations,
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
    pdf.cell(0, 8, txt=f"Fecha: {datetime.datetime.now().strftime('%Y-%m-%d')} | Modelo: Estocastico Avanzado", ln=True, align='C')
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
    pdf.cell(0, 6, txt=f"   - Stop-Loss (VaR): ${data['var_price']:.2f}", ln=True)
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
    pdf.cell(0, 8, txt=" 3. DIRECTRIZ INSTITUCIONAL", ln=True, fill=True)
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
                    label="📥 Descargar Reporte PDF",
                    data=pdf_bytes,
                    file_name=f"Quant_Risk_{report_data['ticker']}.pdf",
                    mime="application/pdf",
                    type="primary"
                )
