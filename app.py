import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import datetime
import unicodedata
from fpdf import FPDF

# ==============================================================================
# IMPORTACIONES MODULARES
# ==============================================================================
from core.data_ingestion import get_macro_context, load_financial_data, generate_synthetic_data
from features.structural_engine import engineer_structural_features
from models.inference_bgmm import identify_bayesian_regimes
from simulation.stoch_generators import apply_auxiliary_particle_filter, apply_stochastic_volatility_filter, run_montecarlo_advanced_stochastic
from diagnostics.model_governance import calculate_model_diagnostics, calculate_risk_metrics_phase1, generate_directive

# ==============================================================================
# LÓGICA DE PDF Y TRADUCCIÓN CLÍNICA
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
    pdf.cell(0, 10, txt=clean_text_for_pdf(f"REPORTE DE ESTRATEGIA CUANTITATIVA: {data['ticker']}"), ln=True, align='C')
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 8, txt=clean_text_for_pdf(f"Generado: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"), ln=True, align='C')
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 12)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 8, txt=" 1. DIAGNOSTICO INTEGRAL DE MERCADO", ln=True, fill=True)
    pdf.set_font("Arial", '', 10)
    pdf.multi_cell(0, 6, txt=clean_text_for_pdf(f"El activo {data['ticker']} opera bajo un regimen de {data['estado_ml']}. "
        f"Persistencia: {data['persistencia']:.1f} dias. Volatilidad dinamica: {data['vol_dyn']:.2f}%. "
        f"Contexto detectado: {data['contexto_riesgo']}."))
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, txt=" 2. PLAN DE EJECUCION Y ESTRUCTURA DE CAPITAL", ln=True, fill=True)
    pdf.set_font("Arial", '', 10)
    sym = data['sym']
    pdf.cell(0, 6, txt=clean_text_for_pdf(f"   - Capital base: {sym}{data['capital_inicial']:,.2f}"), ln=True)
    pdf.cell(0, 6, txt=clean_text_for_pdf(f"   - Valor esperado (mediana {data['days']}d): {sym}{data['capital_esperado']:,.2f} ({data['rend_esp']:+,.2f})"), ln=True)
    pdf.cell(0, 6, txt=clean_text_for_pdf(f"   - Capital en Riesgo (Stop-Loss VaR {data['conf_level']}%): {sym}{data['capital_var']:,.2f}"), ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, txt=" 3. ANALISIS CLINICO Y RECOMENDACIONES", ln=True, fill=True)
    pdf.set_font("Arial", '', 10)
    for insight in data['insights']:
        pdf.multi_cell(0, 6, txt=f"   * {clean_text_for_pdf(insight).replace('**', '')}")
        pdf.ln(2)
    
    pdf.ln(3)
    pdf.set_font("Arial", 'B', 10)
    pdf.multi_cell(0, 6, txt=clean_text_for_pdf(f"RECOMENDACION INSTITUCIONAL: {data['directriz']} - {data['justificacion']}"))
    return pdf.output(dest='S').encode('latin-1', errors='replace')

def interpret_structural_features(latest_features: pd.Series) -> list:
    insights = []
    if latest_features['Kurtosis_20'] > 3.0: insights.append("⚠️ **Alerta de Cisne Negro:** Alta Kurtosis detectada. Riesgo de shocks extremos.")
    if latest_features['DD_Velocity'] < -0.05: insights.append("📉 **Aceleración Bajista:** Drawdown acelerado. Riesgo de liquidación en cascada.")
    if latest_features['Realized_Vol_20'] > 0.40 and latest_features['Returns'] > 0: insights.append("🔥 **Melt-Up:** Euforia volátil detectada. Posible burbuja.")
    if 'Entropy_20' in latest_features and latest_features['Entropy_20'] > 2.0: insights.append("🌪️ **Alta Entropía:** Mercado desordenado. Baja predictibilidad.")
    if not insights: insights.append("✅ **Estructura Nominal:** Métricas operativas estables.")
    return insights

# ==============================================================================
# ORQUESTADOR UI
# ==============================================================================
def render_dashboard():
    st.set_page_config(page_title="Quant Lab", layout="wide", page_icon="🧬")
    
    with st.spinner("Sincronizando Macro..."):
        risk_free_rate, current_vix = get_macro_context()

    with st.sidebar:
        st.header("1. Configuración de Activo")
        ASSET_UNIVERSE = {
            "🔍 Manual": "MANUAL", 
            "--- INDICES & TECNOLOGIA ---": "HEADER",
            "📊 S&P 500 (SPY)": "SPY", "📊 Nasdaq 100 (QQQ)": "QQQ",
            "🇺🇸 NVIDIA (NVDA)": "NVDA", "🇺🇸 Apple (AAPL)": "AAPL",
            "--- EMERGENTES ---": "HEADER",
            "🇵🇪 Credicorp (BAP)": "BAP", "🇧🇷 iShares Brazil (EWZ)": "EWZ",
            "--- CRYPTO & COMMODITIES ---": "HEADER",
            "₿ Bitcoin (BTC-USD)": "BTC-USD", "⟠ Ethereum (ETH-USD)": "ETH-USD",
            "🥇 Oro (GLD)": "GLD", "🥈 Plata (SLV)": "SLV"
        }
        sel_asset = st.selectbox("Mercado:", list(ASSET_UNIVERSE.keys()))
        if ASSET_UNIVERSE[sel_asset] == "HEADER": st.stop()
        ticker = st.text_input("Ticker:", "").upper() or ASSET_UNIVERSE[sel_asset]
        
        st.divider()
        st.header("2. Arquitectura de Simulación")
        moneda_sel = st.selectbox("Moneda:", ["USD ($)", "EUR (€)", "PEN (S/)"])
        sym = moneda_sel.split(" ")[1].replace("(", "").replace(")", "")
        capital_inicial = st.number_input("Capital a Invertir:", value=10000.0, step=1000.0)
        days_to_project = st.slider("Días:", 10, 252, 21)
        simulations = {"1k": 1000, "5k": 5000, "10k": 10000}[st.selectbox("Rutas:", ["1k", "5k", "10k"], index=1)]
        conf_level = st.slider("Confianza VaR (%):", 90.0, 99.9, 95.0, 0.1)

    trading_days = 365 if "USD" in ticker else 252
    
    # ------------------ INGESTA Y CONTINGENCIA ------------------
    try:
        df_hist = load_financial_data(ticker, "")
        if not df_hist.empty: df_hist = df_hist.iloc[:-21]
    except Exception:
        df_hist = generate_synthetic_data(ticker, days=500, trading_days=trading_days)
        st.error("🚨 **ALERTA CRÍTICA:** Conexión offline. Desplegando simulación sintética.")

    # ------------------ PIPELINE IA VARIACIONAL (FASE 2) ------------------
    with st.spinner("Inferencia Variacional (Auto-Calibración)..."):
        df_features = engineer_structural_features(df_hist, trading_days)
        current_regime, feature_list, regime_history, calib_params, trans_matrix = identify_bayesian_regimes(df_features)
    
    daily_returns = df_hist['Close'].pct_change().dropna()
    S0 = df_hist['Close'].iloc[-1]
    raw_mu = daily_returns.mean() * trading_days
    
    # ------------------ FILTROS MATEMÁTICOS EXTREMOS (FASE 3) ------------------
    with st.spinner("Desplegando Filtro de Partículas Auxiliar..."):
        # Se reemplaza Kalman por APF para extraer la tendencia aislando colas pesadas
        apf_states = apply_auxiliary_particle_filter(daily_returns.values, num_particles=1000)
        pesos_ewma = np.exp(np.linspace(-1, 0, min(20, len(apf_states))))
        current_mu = np.dot(apf_states[-len(pesos_ewma):], pesos_ewma / pesos_ewma.sum()) * trading_days
        
        dynamic_sigma_daily = apply_stochastic_volatility_filter(daily_returns.values)
        current_sigma_ann = dynamic_sigma_daily[-1] * np.sqrt(trading_days)
        long_term_sigma_ann = np.mean(dynamic_sigma_daily) * np.sqrt(trading_days)

    # Parche de transición: Lógica estable hasta introducir proceso Hawkes (Fase 4)
    if current_regime == -1: ml_lambda_j, ml_mu_j, ml_sigma_j = 6.0, -0.08, 0.08 
    elif current_regime == 1: ml_lambda_j, ml_mu_j, ml_sigma_j = 1.0, 0.01, 0.02 
    else: ml_lambda_j, ml_mu_j, ml_sigma_j = 2.0, -0.03, 0.05
    
    # ------------------ MOTOR DE MONTE CARLO ------------------
    with st.spinner("Generación de Escenarios Estocásticos..."):
        paths = run_montecarlo_advanced_stochastic(S0, current_mu, raw_mu, current_sigma_ann, long_term_sigma_ann, days_to_project, simulations, ml_lambda_j, ml_mu_j, ml_sigma_j, trading_days)
        prob_pos, var_price, median_price, tp_price = calculate_risk_metrics_phase1(S0, paths, conf_level)
        persistencia, durbin_watson, exceedance = calculate_model_diagnostics(daily_returns.values, regime_history, apf_states)

    # ------------------ CÁLCULOS DE CAPITAL Y UI ------------------
    capital_esperado = (capital_inicial / S0) * median_price
    capital_var = (capital_inicial / S0) * var_price
    rend_esp = capital_esperado - capital_inicial
    rend_var = capital_var - capital_inicial

    estado_ml_txt = "🔴 Estrés Estructural" if current_regime == -1 else "🟢 Expansión Estable" if current_regime == 1 else "⚪ Transición Latente"
    
    st.title(f"Telemetría Cuantitativa: {ticker}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precio Base", f"${S0:,.2f}")
    c2.metric("Régimen (Inferencia)", estado_ml_txt.split(' ', 1)[1])
    c3.metric("Fricción VaR", f"{exceedance*100:.1f}%")
    c4.metric("Volatilidad Dinámica", f"{current_sigma_ann*100:.1f}%")

    tab1, tab2 = st.tabs(["📈 Proyección y Capital", "🔬 Diagnóstico Clínico"])
    
    with tab1:
        dir_t, just_t = generate_directive(prob_pos, current_regime)
        col_dir, col_tp, col_sl = st.columns(3)
        col_dir.info(f"**DIRECTRIZ:**\n\n**{dir_t}**\n\n*{just_t}*")
        col_tp.success(f"**TAKE PROFIT:**\n\n**${tp_price:,.2f}**")
        col_sl.error(f"**STOP-LOSS:**\n\n**${var_price:,.2f}**")
        
        fig = go.Figure()
        for i in range(min(50, paths.shape[1])): fig.add_trace(go.Scatter(x=np.arange(days_to_project + 1), y=paths[:, i], mode='lines', line=dict(color='rgba(0,100,255,0.05)'), showlegend=False))
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
        insights = interpret_structural_features(df_features.iloc[-1])
        for i in insights: st.markdown(f"> {i}")
        st.divider()
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.metric("Estabilidad de Manifold", f"{persistencia:.1f} días", "Alerta" if persistencia < 10 else "Estable", delta_color="normal" if persistencia > 10 else "inverse")
        with col_v2:
            st.metric("Pureza de Tendencia (D-W)", f"{durbin_watson:.2f}", "Ideal: ~2.00")
        
        st.caption("Matriz de Características Estructurales")
        try: st.dataframe(df_features.tail(5).style.format("{:.4f}").background_gradient(cmap='Purples'), use_container_width=True)
        except Exception: st.dataframe(df_features.tail(5), use_container_width=True)

    # ------------------ EXPORTACIÓN PDF ------------------
    report_data = {
        "ticker": ticker, "S0": S0, "tp_price": tp_price, "var_price": var_price,
        "estado_ml": estado_ml_txt.split(' ', 1)[1], "persistencia": persistencia, "exceedance": exceedance,
        "conf_level": conf_level, "insights": insights, "sym": sym, "directriz": dir_t,
        "justificacion": just_t, "capital_inicial": capital_inicial, 
        "capital_esperado": capital_esperado, "capital_var": capital_var,
        "rend_esp": rend_esp, "vol_dyn": current_sigma_ann * 100, 
        "contexto_riesgo": "Estrés Elevado" if current_regime == -1 else "Estabilidad Operativa",
        "days": days_to_project
    }
    st.sidebar.divider()
    st.sidebar.download_button("📥 Descargar Reporte y Diagnóstico (PDF)", create_pdf_report(report_data), f"Quant_Report_{ticker}.pdf", mime="application/pdf", use_container_width=True)

if __name__ == "__main__": 
    render_dashboard()
