import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import datetime
from fpdf import FPDF

# ==============================================================================
# IMPORTACIONES DESDE LA ARQUITECTURA MODULAR
# ==============================================================================
from core.data_ingestion import get_macro_context, load_financial_data, generate_synthetic_data
from features.structural_engine import engineer_structural_features
from models.inference_bgmm import identify_bayesian_regimes
from simulation.stoch_generators import apply_kalman_filter_dynamic, apply_stochastic_volatility_filter, run_montecarlo_advanced_stochastic
from diagnostics.model_governance import calculate_model_diagnostics, calculate_risk_metrics_phase1

# ==============================================================================
# EXPORTACIÓN PDF INSTITUCIONAL
# ==============================================================================
def create_pdf_report(data: dict) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, txt=f"REPORTE CUANTITATIVO: {data['ticker']}", ln=True, align='C')
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 8, txt=f"Fecha: {datetime.datetime.now().strftime('%Y-%m-%d')} | Laboratorio Estocastico Bayesiano", ln=True, align='C')
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 12)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 8, txt=" 1. DIAGNOSTICO DE REGIMEN (GMM)", ln=True, fill=True)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 6, txt=f"   - ESTADO DETECTADO: {data['estado_ml']}", ln=True)
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 6, txt=f"   - Estabilidad del Manifold: {data['persistencia']:.1f} dias", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, txt=" 2. NIVELES OPERATIVOS Y RIESGO", ln=True, fill=True)
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 6, txt=f"   - Precio Base: ${data['S0']:.2f}", ln=True)
    pdf.cell(0, 6, txt=f"   - Take Profit Proyectado: ${data['tp_price']:.2f}", ln=True)
    pdf.cell(0, 6, txt=f"   - Touch Stop-Loss (VaR {data['conf_level']}%): ${data['var_price']:.2f}", ln=True)
    pdf.cell(0, 6, txt=f"   - Friccion Historica (VaR Exceedance): {data['exceedance']*100:.1f}%", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, txt=" 3. MOTORES MATEMATICOS ACTIVOS", ln=True, fill=True)
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 6, txt="   [x] Difusion Continua (Drift Estabilizado)", ln=True)
    pdf.cell(0, 6, txt="   [x] Colas Pesadas t-Student (Cisnes Negros)", ln=True)
    pdf.cell(0, 6, txt="   [x] Saltos de Merton Condicionados (Poisson Jumps)", ln=True)

    pdf_string = pdf.output(dest='S')
    return pdf_string.encode('latin-1', errors='replace')

# ==============================================================================
# MOTOR DE INTERPRETACIÓN CLÍNICA DE DATOS (NUEVO)
# ==============================================================================
def interpret_structural_features(latest_features: pd.Series) -> list:
    insights = []
    
    # Análisis de Kurtosis (Cisnes Negros)
    if latest_features['Kurtosis_20'] > 3.0:
        insights.append("⚠️ **Alerta de Cisne Negro:** La Kurtosis es anormalmente alta. El activo está propenso a movimientos violentos e impredecibles que rompen la campana de Gauss.")
    elif latest_features['Kurtosis_20'] < 1.0:
        insights.append("🛡️ **Riesgo de Cola Bajo:** La distribución de retornos es plana. Los movimientos extremos son estadísticamente improbables en este momento.")
        
    # Análisis de Volatilidad y Drawdown
    if latest_features['DD_Velocity'] < -0.05:
        insights.append("📉 **Aceleración Bajista:** La velocidad del Drawdown indica que el activo está en caída libre estructural. Riesgo de cascada de liquidaciones.")
    elif latest_features['Realized_Vol_20'] > 0.40 and latest_features['Returns'] > 0:
        insights.append("🔥 **Euforia Volátil (Melt-Up):** Alta volatilidad combinada con retornos positivos. Posible formación de burbuja a corto plazo.")
        
    # Análisis de Entropía
    if 'Entropy_20' in latest_features and latest_features['Entropy_20'] > 2.0:
        insights.append("🌪️ **Alta Entropía:** El mercado está en estado de desorden. La predictibilidad direccional es nula; el ruido domina a la señal.")
        
    if not insights:
        insights.append("✅ **Estructura Estable:** Las métricas de estrés se encuentran dentro de rangos normales de operación estocástica.")
        
    return insights

# ==============================================================================
# UI Y DASHBOARD ORQUESTADOR
# ==============================================================================
def render_dashboard():
    st.set_page_config(page_title="Quant Lab: Research & Inference", layout="wide", page_icon="🧬")
    
    with st.spinner("Sincronizando Entorno Macroeconómico..."):
        risk_free_rate, current_vix = get_macro_context()

    with st.sidebar:
        st.header("1. Configuración de Entorno")
        
        ASSET_UNIVERSE = {
            "🔍 Entrada Manual (Ticker)": "MANUAL",
            "--- TECNOLOGÍA ---": "HEADER",
            "🇺🇸 NVIDIA Corp. (NVDA)": "NVDA", "🇺🇸 Apple Inc. (AAPL)": "AAPL",
            "--- CRIPTOMONEDAS ---": "HEADER",
            "₿ Bitcoin (BTC-USD)": "BTC-USD", "⟠ Ethereum (ETH-USD)": "ETH-USD",
            "--- MATERIAS PRIMAS ---": "HEADER",
            "🥇 Oro ETF (GLD)": "GLD", "🥈 Plata ETF (SLV)": "SLV"
        }
        
        sel_asset = st.selectbox("Seleccione un Activo:", list(ASSET_UNIVERSE.keys()))
        if ASSET_UNIVERSE[sel_asset] == "HEADER":
            st.warning("⚠️ Selecciona un activo válido abajo.")
            st.stop()
            
        custom_asset = st.text_input("...o ingrese Ticker Manual:", "")
        ticker = custom_asset.upper() if custom_asset else ASSET_UNIVERSE[sel_asset]
        
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
        st.error("🚨 **ALERTA CRÍTICA:** Conexión fallida. Desplegando simulación teórica base.")

    if not df_hist.empty: df_hist = df_hist.iloc[:-21]

    with st.spinner("Inferencia Bayesiana y Construcción de Manifold..."):
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
    
    with st.spinner("Generación Estocástica de Escenarios..."):
        paths = run_montecarlo_advanced_stochastic(S0, current_mu, raw_mu, current_sigma_ann, long_term_sigma_ann, days_to_project, simulations, ml_lambda_j, ml_mu_j, ml_sigma_j, trading_days)
        prob_pos, var_price, median_price, tp_price = calculate_risk_metrics_phase1(S0, paths, conf_level)
        persistencia, durbin_watson, exceedance = calculate_model_diagnostics(daily_returns.values, regime_history, kalman_states)

    # ==============================================================================
    # RENDERIZADO VISUAL RESTAURADO
    # ==============================================================================
    st.title(f"Telemetría Cuantitativa: {ticker}")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precio Base", f"${S0:,.2f}")
    c2.metric("Tendencia (Filtro Kalman)", f"{current_mu*100:.1f}%")
    c3.metric("Fricción VaR (Exceedance)", f"{exceedance*100:.1f}%")
    c4.metric("Volatilidad Dinámica", f"{current_sigma_ann*100:.1f}%")

    estado_ml_txt = "🔴 Estrés Estructural" if current_regime == -1 else "🟢 Expansión Estable" if current_regime == 1 else "⚪ Transición Latente"
    
    st.subheader(f"Plan de Ejecución ({days_to_project} días)")
    
    # Cajas de colores restauradas
    col_dir, col_tp, col_sl = st.columns(3)
    with col_dir:
        st.info(f"**RÉGIMEN BAYESIANO:**\n\n{estado_ml_txt}\n\n*Persistencia actual: {persistencia:.1f} días.*")
    with col_tp:
        st.success(f"**TAKE PROFIT (Probabilístico):**\n\n${tp_price:,.2f}\n\n*Basado en la distribución proyectada.*")
    with col_sl:
        st.error(f"**TOUCH STOP-LOSS (VaR {conf_level}%):**\n\n${var_price:,.2f}\n\n*Límite de dolor institucional.*")

    st.caption("⚙️ **Motores Activos:** Difusión Estocástica Acelerada (Numba JIT) + **Cisnes Negros** (Colas t-Student) + **Saltos de Merton** (Poisson Jumps) condicionados por IA.")

    # Gráfico
    fig = go.Figure()
    for i in range(min(50, paths.shape[1])): 
        fig.add_trace(go.Scatter(x=np.arange(days_to_project + 1), y=paths[:, i], mode='lines', line=dict(color='rgba(0, 100, 255, 0.08)'), showlegend=False))
    fig.add_trace(go.Scatter(x=[0, days_to_project], y=[S0, S0], mode='lines', name='Nivel Entrada', line=dict(color='black', dash='dash')))
    fig.add_trace(go.Scatter(x=[0, days_to_project], y=[var_price, var_price], mode='lines', name='Touch VaR Baseline', line=dict(color='red', dash='dot')))
    fig.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # Diagnóstico e Interpretación
    st.markdown("---")
    st.subheader("🔬 Laboratorio de Diagnóstico Estructural")
    
    try:
        latest_features = df_features.iloc[-1]
        insights = interpret_structural_features(latest_features)
        
        col_table, col_interp = st.columns([2, 1])
        with col_table:
            st.dataframe(df_features.tail(5).style.format("{:.4f}").background_gradient(cmap='Purples'), use_container_width=True)
        with col_interp:
            st.markdown("### Traducción Matemática:")
            for insight in insights:
                st.markdown(insight)
    except Exception:
        st.warning("Esperando sincronización de matriz de características para generar diagnóstico...")

    # Reporte PDF
    report_data = {
        "ticker": ticker, "S0": S0, "tp_price": tp_price, "var_price": var_price,
        "estado_ml": estado_ml_txt, "persistencia": persistencia, "exceedance": exceedance,
        "conf_level": conf_level
    }
    
    with st.sidebar:
        st.markdown("---")
        pdf_bytes = create_pdf_report(report_data)
        st.download_button(label="📥 Descargar Reporte PDF", data=pdf_bytes, file_name=f"Quant_Report_{ticker}.pdf", mime="application/pdf", type="primary")

if __name__ == "__main__":
    render_dashboard()
