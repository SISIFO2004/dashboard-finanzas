import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import datetime
import unicodedata
from fpdf import FPDF

# ==============================================================================
# IMPORTACIONES MODULARES (LAS 4 FASES DEL BACKEND)
# ==============================================================================
from core.data_ingestion import get_macro_context, load_financial_data, generate_synthetic_data
from features.structural_engine import engineer_structural_features
from models.inference_bgmm import identify_bayesian_regimes
from simulation.stoch_generators import apply_auxiliary_particle_filter, apply_stochastic_volatility_filter, run_montecarlo_advanced_stochastic
from diagnostics.model_governance import calculate_model_diagnostics, calculate_risk_metrics_phase1, generate_directive

# ==============================================================================
# LÓGICA DE PDF Y SANITIZACIÓN INSTITUCIONAL
# ==============================================================================
def clean_text_for_pdf(text):
    text = str(text)
    for e in ['🔴', '🟢', '⚪', '⚠️', '🛡️', '📉', '🔥', '🌪️', '✅', '⚙️', '🔬', '📊', '🧬', '📥']: 
        text = text.replace(e, '')
    return unicodedata.normalize('NFKD', text).encode('latin-1', 'ignore').decode('latin-1').strip()

def create_pdf_report(data: dict) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, txt=clean_text_for_pdf(f"MEMORANDUM CUANTITATIVO: {data['ticker']}"), ln=True, align='C')
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 8, txt=clean_text_for_pdf(f"Fecha de Inferencia: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"), ln=True, align='C')
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 12)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 8, txt=" 1. DIAGNOSTICO ESPECTRAL Y VARIACIONAL", ln=True, fill=True)
    pdf.set_font("Arial", '', 10)
    pdf.multi_cell(0, 6, txt=clean_text_for_pdf(f"El activo {data['ticker']} opera bajo un regimen de {data['estado_ml']}. "
        f"Persistencia del manifold: {data['persistencia']:.1f} dias. Volatilidad APF: {data['vol_dyn']:.2f}%. "
        f"Clima macroeconomico cruzado: VIX {data['vix']:.2f} (T-Yield {data['tnx']:.2f}%)."))
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, txt=" 2. ESTRUCTURA ESTOCASTICA (CAPITAL EN RIESGO)", ln=True, fill=True)
    pdf.set_font("Arial", '', 10)
    sym = data['sym']
    pdf.cell(0, 6, txt=clean_text_for_pdf(f"   - Capital base asignado: {sym}{data['capital_inicial']:,.2f}"), ln=True)
    pdf.cell(0, 6, txt=clean_text_for_pdf(f"   - Valor esperado (mediana {data['days']}d): {sym}{data['capital_esperado']:,.2f} ({data['rend_esp']:+,.2f})"), ln=True)
    pdf.cell(0, 6, txt=clean_text_for_pdf(f"   - Capital en Riesgo (Touch Stop-Loss {data['conf_level']}%): {sym}{data['capital_var']:,.2f}"), ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, txt=" 3. TRADUCCION CLINICA", ln=True, fill=True)
    pdf.set_font("Arial", '', 10)
    for insight in data['insights']:
        pdf.multi_cell(0, 6, txt=f"   * {clean_text_for_pdf(insight).replace('**', '')}")
        pdf.ln(2)
    
    pdf.ln(3)
    pdf.set_font("Arial", 'B', 10)
    pdf.multi_cell(0, 6, txt=clean_text_for_pdf(f"DIRECTRIZ INSTITUCIONAL: {data['directriz']} - {data['justificacion']}"))
    return pdf.output(dest='S').encode('latin-1', errors='replace')

def interpret_structural_features(latest_features: pd.Series) -> list:
    insights = []
    if latest_features['Kurtosis_20'] > 3.0: insights.append("⚠️ **Alerta de Cisne Negro:** Alta Kurtosis. Riesgo de shocks extremos.")
    if latest_features['DD_Velocity'] < -0.05: insights.append("📉 **Aceleración Bajista:** Drawdown acelerado. Riesgo de liquidación en cascada.")
    if latest_features['Realized_Vol_20'] > 0.40 and latest_features['Returns'] > 0: insights.append("🔥 **Melt-Up:** Euforia volátil detectada.")
    if 'Entropy_20' in latest_features and latest_features['Entropy_20'] > 2.0: insights.append("🌪️ **Alta Entropía:** Mercado desordenado. Baja predictibilidad.")
    if not insights: insights.append("✅ **Estructura Nominal:** Métricas operativas estables.")
    return insights

# ==============================================================================
# ORQUESTADOR UI (FULL-STACK QUANT)
# ==============================================================================
def render_dashboard():
    st.set_page_config(page_title="Quant Lab | Dinámica Estocástica", layout="wide", page_icon="🧬")
    
    # --- RADAR MACROECONÓMICO (CONTEXTO SISTÉMICO) ---
    with st.spinner("Sincronizando Radar Macro..."):
        risk_free_rate, current_vix = get_macro_context()
        
    st.markdown("<h4 style='text-align: center; color: #666;'>Radar Sistémico Global</h4>", unsafe_allow_html=True)
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Costo de Liquidez (US 10Y)", f"{risk_free_rate:.2f}%")
    col_m2.metric("Miedo Institucional (VIX)", f"{current_vix:.2f}", "Alerta Sistémica" if current_vix > 25 else "Nominal", delta_color="inverse" if current_vix > 25 else "normal")
    col_m3.metric("Entorno Cuantitativo", "Contracción" if risk_free_rate > 4.5 and current_vix > 20 else "Expansión")
    st.divider()

    # --- PANEL LATERAL: CONFIGURACIÓN Y OVERRIDE ---
    with st.sidebar:
        st.header("1. Configuración de Activo")
        ASSET_UNIVERSE = {
            "🔍 Entrada Manual (Ticker)": "MANUAL",
            
            "--- ÍNDICES GLOBALES ---": "HEADER",
            "📊 S&P 500 ETF (SPY)": "SPY", 
            "📊 Nasdaq 100 (QQQ)": "QQQ", 
            "📊 Russell 2000 (IWM)": "IWM", 
            "📊 Dow Jones (DIA)": "DIA",
            
            "--- MERCADO PERUANO (ADRs & ETFs) ---": "HEADER",
            "🇵🇪 iShares MSCI Peru ETF (EPU)": "EPU",
            "🇵🇪 Credicorp Ltd. (BAP)": "BAP", 
            "🇵🇪 Cia. de Minas Buenaventura (BVN)": "BVN", 
            "🇵🇪 Southern Copper (SCCO)": "SCCO", 
            "🇵🇪 Intercorp Financial (IFS)": "IFS", 
            "🇵🇪 Cementos Pacasmayo (CPAC)": "CPAC", 
            
            "--- OTROS EMERGENTES ---": "HEADER",
            "🇧🇷 iShares MSCI Brazil (EWZ)": "EWZ", 
            "🇲🇽 iShares MSCI Mexico (EWW)": "EWW",
            "🇦🇷 Grupo Financiero Galicia (GGAL)": "GGAL",
            
            "--- TECNOLOGÍA (MAG 7) ---": "HEADER",
            "🇺🇸 NVIDIA (NVDA)": "NVDA", 
            "🇺🇸 Apple (AAPL)": "AAPL", 
            "🇺🇸 Microsoft (MSFT)": "MSFT", 
            "🇺🇸 Alphabet (GOOGL)": "GOOGL", 
            "🇺🇸 Amazon (AMZN)": "AMZN", 
            "🇺🇸 Meta (META)": "META", 
            "🇺🇸 Tesla (TSLA)": "TSLA",
            
            "--- FINANZAS & BLUE CHIPS ---": "HEADER",
            "🇺🇸 JPMorgan (JPM)": "JPM", 
            "🇺🇸 Berkshire Hathaway (BRK-B)": "BRK-B", 
            "🇺🇸 Visa (V)": "V", 
            "🇺🇸 Johnson & Johnson (JNJ)": "JNJ",
            
            "--- CRIPTOMONEDAS ---": "HEADER",
            "₿ Bitcoin (BTC-USD)": "BTC-USD", 
            "⟠ Ethereum (ETH-USD)": "ETH-USD", 
            "☀️ Solana (SOL-USD)": "SOL-USD", 
            "💠 Cardano (ADA-USD)": "ADA-USD",
            
            "--- MATERIAS PRIMAS ---": "HEADER",
            "🥇 Oro (GLD)": "GLD", 
            "🥈 Plata (SLV)": "SLV", 
            "🛢️ Petróleo Crudo (USO)": "USO", 
            "🥉 Cobre (COPX)": "COPX",
            "🌾 Trigo (WEAT)": "WEAT"
        }
        sel_asset = st.selectbox("Mercado:", list(ASSET_UNIVERSE.keys()), label_visibility="collapsed")
        if ASSET_UNIVERSE[sel_asset] == "HEADER": st.stop()
        ticker = st.text_input("...o ingrese Ticker Manual:", "").upper() or ASSET_UNIVERSE[sel_asset]
        
        st.divider()
        st.header("2. Arquitectura Estocástica")
        moneda_sel = st.selectbox("Moneda:", ["USD ($)", "EUR (€)", "PEN (S/)"])
        sym = moneda_sel.split(" ")[1].replace("(", "").replace(")", "")
        capital_inicial = st.number_input("Capital a Invertir:", value=10000.0, step=1000.0)
        days_to_project = st.slider("Horizonte (Días):", 10, 252, 21)
        simulations = {"1k": 1000, "5k": 5000, "10k": 10000}[st.selectbox("Rutas (Monte Carlo):", ["1k", "5k", "10k"], index=1)]
        conf_level = st.slider("Confianza VaR (%):", 90.0, 99.9, 95.0, 0.1)
        
        st.divider()
        st.header("3. Gobernanza de Riesgo")
        override_ia = st.toggle("☢️ Stress Test (Ignorar IA)")
        if override_ia:
            st.error("Motor Variacional en Bypass. Simulando colas pesadas manuales (Hawkes).")
            manual_lambda = st.slider("Intensidad de Réplicas (λ)", 0.1, 10.0, 5.0)
            manual_mu = st.slider("Profundidad de Salto (μ)", -0.20, 0.0, -0.05)
            manual_sigma = st.slider("Volatilidad de Salto (σ)", 0.01, 0.20, 0.08)

    trading_days = 365 if "USD" in ticker else 252
    
    # --- INGESTA Y CONTINGENCIA ---
    try:
        df_hist = load_financial_data(ticker, "")
        # Eliminado el recorte de backtest para sincronización exacta en tiempo real
    except Exception:
        df_hist = generate_synthetic_data(ticker, days=500, trading_days=trading_days)
        st.error("🚨 **ALERTA:** Conexión offline. Desplegando simulación teórica.")

    # --- FASE 1 & 2: CEREBRO (FFT + INFERENCIA VARIACIONAL) ---
    with st.spinner("Ejecutando Diferenciación Espectral e Inferencia Variacional..."):
        df_features = engineer_structural_features(df_hist, trading_days)
        current_regime, feature_list, regime_history, calib_params, trans_matrix = identify_bayesian_regimes(df_features)
    
    daily_returns = df_hist['Close'].pct_change().dropna()
    S0 = df_hist['Close'].iloc[-1]
    raw_mu = daily_returns.mean() * trading_days
    
    # --- FASE 3: FILTRO DE PARTÍCULAS (APF) ---
    with st.spinner("Extrayendo Tendencia No Lineal (Auxiliary Particle Filter)..."):
        apf_states = apply_auxiliary_particle_filter(daily_returns.values, num_particles=1000)
        pesos_ewma = np.exp(np.linspace(-1, 0, min(20, len(apf_states))))
        current_mu = np.dot(apf_states[-len(pesos_ewma):], pesos_ewma / pesos_ewma.sum()) * trading_days
        
        dynamic_sigma_daily = apply_stochastic_volatility_filter(daily_returns.values)
        current_sigma_ann = dynamic_sigma_daily[-1] * np.sqrt(trading_days)
        long_term_sigma_ann = np.mean(dynamic_sigma_daily) * np.sqrt(trading_days)

    # --- ACOPLE DE CALIBRACIÓN ML -> MONTE CARLO ---
    if override_ia:
        ml_lambda_j, ml_mu_j, ml_sigma_j = manual_lambda, manual_mu, manual_sigma
    else:
        vi_params = calib_params.get(current_regime, {'lambda_j': 1.0, 'mu_regime': -0.01, 'sigma_regime': 0.05})
        ml_lambda_j = max(0.1, vi_params['lambda_j']) 
        ml_mu_j = np.clip(vi_params['mu_regime'], -0.20, 0.20)
        ml_sigma_j = np.clip(vi_params['sigma_regime'], 0.01, 0.30)

    # --- FASE 4: MOTOR DEL CAOS (NON-MARKOVIAN HAWKES NUMBA JIT) ---
    with st.spinner("Compilando Escenarios Estocásticos Exactos..."):
        paths = run_montecarlo_advanced_stochastic(
            S0, current_mu, raw_mu, current_sigma_ann, long_term_sigma_ann, 
            days_to_project, simulations, ml_lambda_j, ml_mu_j, ml_sigma_j, trading_days
        )
        prob_pos, var_price, median_price, tp_price = calculate_risk_metrics_phase1(S0, paths, conf_level)
        
        # Auditoría del Modelo
        aligned_returns = daily_returns.iloc[-len(regime_history):].values
        aligned_apf = apf_states[-len(regime_history):]
        persistencia, durbin_watson, exceedance = calculate_model_diagnostics(aligned_returns, regime_history, aligned_apf)

    # --- CÁLCULOS DE CAPITAL ---
    capital_esperado = (capital_inicial / S0) * median_price
    capital_var = (capital_inicial / S0) * var_price
    rend_esp = capital_esperado - capital_inicial
    rend_var = capital_var - capital_inicial
    estado_ml_txt = "🔴 Estrés Estructural" if current_regime == -1 else "🟢 Expansión Estable" if current_regime == 1 else "⚪ Transición Latente"
    
    # --- RENDERIZADO VISUAL ---
    st.title(f"Telemetría Cuantitativa: {ticker}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precio Base", f"${S0:,.2f}")
    c2.metric("Régimen Inferred (VI)", estado_ml_txt.split(' ', 1)[1])
    c3.metric("Fricción VaR", f"{exceedance*100:.1f}%", delta_color="inverse")
    c4.metric("Deriva APF Ajustada", f"{current_mu*100:.1f}%")

    tab1, tab2 = st.tabs(["📈 Proyección Generativa", "🔬 Auditoría de Manifold Histórico"])
    
    with tab1:
        dir_t, just_t = generate_directive(prob_pos, current_regime)
        col_dir, col_tp, col_sl = st.columns(3)
        col_dir.info(f"**DIRECTRIZ:**\n\n**{dir_t}**\n\n*{just_t}*")
        col_tp.success(f"**TAKE PROFIT (Probabilístico):**\n\n**${tp_price:,.2f}**")
        col_sl.error(f"**STOP-LOSS (VaR {conf_level}%):**\n\n**${var_price:,.2f}**")
        
        fig = go.Figure()
        for i in range(min(50, paths.shape[1])): 
            fig.add_trace(go.Scatter(x=np.arange(days_to_project + 1), y=paths[:, i], mode='lines', line=dict(color='rgba(0,100,255,0.05)'), showlegend=False))
        fig.add_trace(go.Scatter(x=[0, days_to_project], y=[S0, S0], mode='lines', line=dict(color='black', dash='dash')))
        fig.add_trace(go.Scatter(x=[0, days_to_project], y=[var_price, var_price], mode='lines', line=dict(color='red', dash='dot')))
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        cap1, cap2, cap3 = st.columns(3)
        cap1.metric(f"Capital Invertido ({sym})", f"{sym}{capital_inicial:,.2f}")
        cap2.metric(f"Valor Esperado ({sym})", f"{sym}{capital_esperado:,.2f}", f"{sym}{rend_esp:+,.2f}")
        cap3.metric(f"Capital en Riesgo ({sym})", f"{sym}{capital_var:,.2f}", f"{sym}{rend_var:+,.2f}", delta_color="inverse")

    with tab2:
        # Gobernanza Visual: Renderizado del Manifold Histórico
        aligned_dates = df_features.index
        aligned_prices = df_hist.loc[aligned_dates, 'Close']
        
        fig_manifold = go.Figure()
        fig_manifold.add_trace(go.Scatter(x=aligned_dates, y=aligned_prices, mode='lines', line=dict(color='#333', width=1), name='Precio Activo'))
        
        color_map = {-1: 'rgba(255, 0, 0, 0.4)', 0: 'rgba(150, 150, 150, 0.4)', 1: 'rgba(0, 255, 0, 0.4)'}
        for reg_val, reg_name in zip([-1, 0, 1], ['Estrés', 'Transición', 'Expansión']):
            mask = (regime_history == reg_val)
            fig_manifold.add_trace(go.Scatter(
                x=aligned_dates[mask], y=aligned_prices[mask], mode='markers',
                marker=dict(color=color_map[reg_val], size=6), name=f'Reg. {reg_name}'
            ))
        fig_manifold.update_layout(title="Auditoría de Estados Latentes (Inferencia Variacional)", height=350, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_manifold, use_container_width=True)

        st.divider()
        insights = interpret_structural_features(df_features.iloc[-1])
        
        col_insight, col_matrix = st.columns([2, 1])
        with col_insight:
            st.markdown("#### Traducción Clínica Actual")
            for i in insights: st.markdown(f"> {i}")
            st.metric("Pureza de Tendencia (D-W)", f"{durbin_watson:.2f}", "Ideal: ~2.00")
            
        with col_matrix:
            st.markdown("#### Matriz de Inercia de Markov ($A_{ij}$)")
            try:
                df_trans = pd.DataFrame(trans_matrix).round(3)
                df_trans.index.name = "De \\ A"
                st.dataframe(df_trans.style.background_gradient(cmap='Blues'), use_container_width=True)
            except:
                st.caption("Matriz en calibración...")

    # --- EXPORTACIÓN DE REPORTE INSTITUCIONAL ---
    report_data = {
        "ticker": ticker, "S0": S0, "tp_price": tp_price, "var_price": var_price,
        "estado_ml": estado_ml_txt.split(' ', 1)[1], "persistencia": persistencia, "exceedance": exceedance,
        "conf_level": conf_level, "insights": insights, "sym": sym, "directriz": dir_t,
        "justificacion": just_t, "capital_inicial": capital_inicial, 
        "capital_esperado": capital_esperado, "capital_var": capital_var,
        "rend_esp": rend_esp, "vol_dyn": current_sigma_ann * 100, 
        "contexto_riesgo": "Estrés Elevado" if current_regime == -1 else "Estabilidad Operativa",
        "days": days_to_project, "tnx": risk_free_rate, "vix": current_vix
    }
    st.sidebar.divider()
    st.sidebar.download_button("📥 Descargar Memorandum Cuantitativo (PDF)", create_pdf_report(report_data), f"Memorandum_{ticker}.pdf", mime="application/pdf", use_container_width=True)

if __name__ == "__main__": 
    render_dashboard()
