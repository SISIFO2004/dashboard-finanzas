import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import datetime
import unicodedata
from fpdf import FPDF

from core.data_ingestion import get_macro_context, load_financial_data, generate_synthetic_data
from features.structural_engine import engineer_structural_features
from models.inference_bgmm import identify_bayesian_regimes
from simulation.stoch_generators import apply_kalman_filter_dynamic, apply_stochastic_volatility_filter, run_montecarlo_advanced_stochastic
from diagnostics.model_governance import calculate_model_diagnostics, calculate_risk_metrics_phase1, generate_directive

# ==============================================================================
# PDF Y SANITIZACIÓN
# ==============================================================================
def clean_text_for_pdf(text):
    text = str(text)
    for e in ['🔴', '🟢', '⚪', '⚠️', '🛡️', '📉', '🔥', '🌪️', '✅', '⚙️', '🔬', '📊', '🧬', '📥']: text = text.replace(e, '')
    return unicodedata.normalize('NFKD', text).encode('latin-1', 'ignore').decode('latin-1').strip()

def create_pdf_report(data: dict) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, txt=clean_text_for_pdf(f"REPORTE CUANTITATIVO: {data['ticker']}"), ln=True, align='C')
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 8, txt=clean_text_for_pdf(f"Fecha: {datetime.datetime.now().strftime('%Y-%m-%d')} | Laboratorio Bayesiano"), ln=True, align='C')
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 12)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 8, txt=" 1. DIAGNOSTICO DE REGIMEN (GMM)", ln=True, fill=True)
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 6, txt=clean_text_for_pdf(f"   - Estado Detectado: {data['estado_ml']}"), ln=True)
    pdf.cell(0, 6, txt=clean_text_for_pdf(f"   - Directriz: {data['directriz']}"), ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, txt=" 2. SIMULACION DE CAPITAL Y RIESGO", ln=True, fill=True)
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 6, txt=clean_text_for_pdf(f"   - Capital Invertido: ${data['capital_inicial']:,.2f}"), ln=True)
    pdf.cell(0, 6, txt=clean_text_for_pdf(f"   - Valor Esperado (Mediana): ${data['capital_esperado']:,.2f}"), ln=True)
    pdf.cell(0, 6, txt=clean_text_for_pdf(f"   - Capital en Riesgo (VaR {data['conf_level']}%): ${data['capital_var']:,.2f}"), ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, txt=" 3. INTERPRETACION CLINICA", ln=True, fill=True)
    pdf.set_font("Arial", '', 10)
    for insight in data['insights']:
        pdf.multi_cell(0, 6, txt=f"   * {clean_text_for_pdf(insight).replace('**', '')}")
        pdf.ln(2)

    return pdf.output(dest='S').encode('latin-1', errors='replace')

def interpret_structural_features(latest_features: pd.Series) -> list:
    insights = []
    if latest_features['Kurtosis_20'] > 3.0: insights.append("⚠️ **Alerta de Cisne Negro:** Alta Kurtosis. Propenso a movimientos violentos fuera de la campana de Gauss.")
    elif latest_features['Kurtosis_20'] < 1.0: insights.append("🛡️ **Riesgo de Cola Bajo:** Distribución plana. Shocks extremos improbables.")
    if latest_features['DD_Velocity'] < -0.05: insights.append("📉 **Aceleración Bajista:** El drawdown indica riesgo de liquidación en cascada.")
    elif latest_features['Realized_Vol_20'] > 0.40 and latest_features['Returns'] > 0: insights.append("🔥 **Melt-Up:** Alta volatilidad alcista. Posible euforia.")
    if 'Entropy_20' in latest_features and latest_features['Entropy_20'] > 2.0: insights.append("🌪️ **Alta Entropía:** Mercado desordenado. El ruido domina la señal.")
    if not insights: insights.append("✅ **Estructura Nominal:** Métricas de estrés dentro de rangos operativos normales.")
    return insights

# ==============================================================================
# ORQUESTADOR UI
# ==============================================================================
def render_dashboard():
    st.set_page_config(page_title="Quant Lab: Research", layout="wide", page_icon="🧬")
    
    with st.spinner("Sincronizando Macro..."):
        risk_free_rate, current_vix = get_macro_context()

    with st.sidebar:
        st.header("1. Configuración de Entorno")
        ASSET_UNIVERSE = {
            "🔍 Entrada Manual (Ticker)": "MANUAL",
            "--- TECNOLOGÍA ---": "HEADER", "🇺🇸 NVIDIA (NVDA)": "NVDA", "🇺🇸 Apple (AAPL)": "AAPL",
            "--- CRIPTOMONEDAS ---": "HEADER", "₿ Bitcoin (BTC-USD)": "BTC-USD", "⟠ Ethereum (ETH-USD)": "ETH-USD",
            "--- COMMODITIES ---": "HEADER", "🥇 Oro (GLD)": "GLD", "🥈 Plata (SLV)": "SLV"
        }
        sel_asset = st.selectbox("Seleccione un Activo:", list(ASSET_UNIVERSE.keys()))
        if ASSET_UNIVERSE[sel_asset] == "HEADER": st.stop()
        ticker = st.text_input("...o ingrese Ticker Manual:", "").upper() or ASSET_UNIVERSE[sel_asset]
        
        st.divider()
        st.header("2. Arquitectura de Simulación")
        capital_inicial = st.number_input("Capital Base ($):", min_value=10.0, value=10000.0, step=1000.0)
        days_to_project = st.slider("Días de Proyección:", 10, 252, 21)
        simulations = {"1k": 1000, "5k": 5000, "10k": 10000}[st.selectbox("Simulaciones:", ["1k", "5k", "10k"], index=2)]
        conf_level = st.slider("Límite VaR (%):", 90.0, 99.9, 95.0, 0.1)

    trading_days = 365 if "USD" in ticker else 252

    try:
        df_hist = load_financial_data(ticker, "")
    except ValueError:
        df_hist = generate_synthetic_data(ticker, days=500, trading_days=trading_days)
        st.error("🚨 **ALERTA CRÍTICA:** Conexión a mercado fallida o bloqueada. Desplegando simulación de contingencia.")

    if not df_hist.empty: df_hist = df_hist.iloc[:-21]

    with st.spinner("Inferencia Bayesiana..."):
        df_features = engineer_structural_features(df_hist, trading_days)
        current_regime, feature_list, regime_history = identify_bayesian_regimes(df_features)
    
    daily_returns = df_hist['Close'].pct_change().dropna()
    S0 = df_hist['Close'].iloc[-1]
    raw_mu = daily_returns.mean() * trading_days
    
    with st.spinner("Calibrando Filtros Matemáticos..."):
        kalman_states = apply_kalman_filter_dynamic(daily_returns.values, 100)
        pesos_ewma = np.exp(np.linspace(-1, 0, min(20, len(kalman_states))))
        current_mu = np.dot(kalman_states[-len(pesos_ewma):], pesos_ewma / pesos_ewma.sum()) * trading_days
        dynamic_sigma_daily = apply_stochastic_volatility_filter(daily_returns.values)
        current_sigma_ann = dynamic_sigma_daily[-1] * np.sqrt(trading_days)
        long_term_sigma_ann = np.mean(dynamic_sigma_daily) * np.sqrt(trading_days)

    if current_regime == -1: ml_lambda_j, ml_mu_j, ml_sigma_j = 6.0, -0.08, 0.08 
    elif current_regime == 1: ml_lambda_j, ml_mu_j, ml_sigma_j = 1.0, 0.01, 0.02 
    else: ml_lambda_j, ml_mu_j, ml_sigma_j = 2.0, -0.03, 0.05
    
    with st.spinner("Generación de Escenarios..."):
        paths = run_montecarlo_advanced_stochastic(S0, current_mu, raw_mu, current_sigma_ann, long_term_sigma_ann, days_to_project, simulations, ml_lambda_j, ml_mu_j, ml_sigma_j, trading_days)
        prob_pos, var_price, median_price, tp_price = calculate_risk_metrics_phase1(S0, paths, conf_level)
        persistencia, durbin_watson, exceedance = calculate_model_diagnostics(daily_returns.values, regime_history, kalman_states)

    # ---------------- MATEMÁTICA DE CAPITAL ----------------
    capital_esperado = (capital_inicial / S0) * median_price
    capital_var = (capital_inicial / S0) * var_price
    rend_esp = capital_esperado - capital_inicial
    rend_var = capital_var - capital_inicial
    
    directriz, justificacion = generate_directive(prob_pos, current_regime)
    insights = interpret_structural_features(df_features.iloc[-1])
    estado_ml_txt = "🔴 Estrés Estructural" if current_regime == -1 else "🟢 Expansión Estable" if current_regime == 1 else "⚪ Transición Latente"

    # ==============================================================================
    # RENDERIZADO VISUAL
    # ==============================================================================
    st.title(f"Telemetría del Activo: {ticker}")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precio Base", f"${S0:,.2f}")
    c2.metric("Régimen Bayesiano", estado_ml_txt.split(' ', 1)[1])
    c3.metric("Fricción VaR", f"{exceedance*100:.1f}%")
    c4.metric("Volatilidad Dinámica", f"{current_sigma_ann*100:.1f}%")

    tab1, tab2 = st.tabs(["📈 Proyección y Capital", "🔬 Diagnóstico Estructural"])

    with tab1:
        st.subheader(f"Plan de Ejecución ({days_to_project} días)")
        
        # Cajas de Ejecución Clásicas Restauradas
        col_dir, col_tp, col_sl = st.columns(3)
        with col_dir:
            st.info(f"**DIRECTRIZ:**\n\n**{directriz}**\n\n*{justificacion}*")
        with col_tp:
            st.success(f"**TAKE PROFIT:**\n\n**${tp_price:,.2f}**\n\n*Objetivo clínico superior.*")
        with col_sl:
            st.error(f"**STOP-LOSS:**\n\n**${var_price:,.2f}**\n\n*Nivel de liquidación.*")

        fig = go.Figure()
        for i in range(min(50, paths.shape[1])): fig.add_trace(go.Scatter(x=np.arange(days_to_project + 1), y=paths[:, i], mode='lines', line=dict(color='rgba(0, 100, 255, 0.08)'), showlegend=False))
        fig.add_trace(go.Scatter(x=[0, days_to_project], y=[S0, S0], mode='lines', line=dict(color='black', dash='dash')))
        fig.add_trace(go.Scatter(x=[0, days_to_project], y=[var_price, var_price], mode='lines', line=dict(color='red', dash='dot')))
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        # MÉTRICAS DE CAPITAL RESTAURADAS
        cap1, cap2, cap3 = st.columns(3)
        cap1.metric("Capital Invertido", f"${capital_inicial:,.2f}")
        cap2.metric("Valor Esperado (Mediana)", f"${capital_esperado:,.2f}", f"{rend_esp:+,.2f}")
        cap3.metric("Capital en Riesgo (VaR)", f"${capital_var:,.2f}", f"{rend_var:+,.2f}", delta_color="inverse")

    with tab2:
        st.subheader("Traducción Clínica")
        for insight in insights: st.markdown(f"> {insight}")
        st.divider()
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.metric("Estabilidad de Manifold", f"{persistencia:.1f} días", "Alerta" if persistencia < 10 else "Estable", delta_color="normal" if persistencia > 10 else "inverse")
        with col_v2:
            st.metric("Pureza de Tendencia (D-W)", f"{durbin_watson:.2f}", "Ideal: ~2.00")
        
        st.caption("Matriz de Características (GMM)")
        try: st.dataframe(df_features.tail(5).style.format("{:.4f}").background_gradient(cmap='Purples'), use_container_width=True)
        except Exception: st.dataframe(df_features.tail(5), use_container_width=True)

    report_data = {
        "ticker": ticker, "S0": S0, "tp_price": tp_price, "var_price": var_price,
        "estado_ml": estado_ml_txt, "persistencia": persistencia, "exceedance": exceedance,
        "conf_level": conf_level, "insights": insights, "directriz": directriz,
        "capital_inicial": capital_inicial, "capital_esperado": capital_esperado, "capital_var": capital_var
    }
    
    with st.sidebar:
        st.markdown("---")
        st.download_button(label="📥 Descargar Reporte y Diagnóstico (PDF)", data=create_pdf_report(report_data), file_name=f"Quant_Report_{ticker}.pdf", mime="application/pdf", type="primary")

if __name__ == "__main__":
    render_dashboard()
