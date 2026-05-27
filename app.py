import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ==============================================================================
# IMPORTACIONES DESDE LA ARQUITECTURA MODULAR (SÍSIFO QUANT LAB)
# ==============================================================================
from core.data_ingestion import get_macro_context, load_financial_data, generate_synthetic_data
from features.structural_engine import engineer_structural_features
from models.inference_bgmm import identify_bayesian_regimes
from simulation.stoch_generators import apply_kalman_filter_dynamic, apply_stochastic_volatility_filter, run_montecarlo_advanced_stochastic
from diagnostics.model_governance import calculate_model_diagnostics, calculate_risk_metrics_phase1

def render_dashboard():
    # Configuración de entorno de visualización corporativa
    st.set_page_config(page_title="Quant Lab: Research & Inference", layout="wide", page_icon="🧬")
    st.title("📊 Motor Generativo de Riesgo (GMM Condicionado)")

    # Sincronización asíncrona del entorno macroeconómico global
    with st.spinner("Sincronizando Entorno Macroeconómico..."):
        risk_free_rate, current_vix = get_macro_context()

    # Barra lateral de control operativo de parámetros de simulación
    with st.sidebar:
        st.header("1. Configuración de Entorno")
        ticker = st.text_input("Ticker Activo:", "BTC-USD").upper()
        capital_inicial = st.number_input("Capital Base ($):", min_value=10.0, value=10000.0, step=1000.0)
        days_to_project = st.slider("Horizonte de Proyección (Días):", 10, 252, 21)
        simulations = 5000
        conf_level = st.slider("Límite de Confianza VaR (%):", 90.0, 99.9, 95.0, 0.1)

    # Detección automática del calendario operativo según la naturaleza del activo
    trading_days = 365 if "USD" in ticker else 252

    # Ingesta resiliente de datos históricos con aislamiento de caché
    try:
        df_hist = load_financial_data(ticker, "")
        st.caption(f"✅ Conexión estable a mercado corporativo | Base temporal calibrada: {trading_days} días")
    except ValueError:
        df_hist = generate_synthetic_data(ticker, days=500, trading_days=trading_days)
        st.error("🚨 **ALERTA CRÍTICA DE INFRAESTRUCTURA:** Conexión fallida con proveedores externos. Desplegando simulación teórica base.")

    if not df_hist.empty:
        # Remoción estricta de la última ventana temporal para prevenir fuga de datos (Data Leakage)
        df_hist = df_hist.iloc[:-21]

    # Pipeline de Machine Learning: Extracción de Características Estructurales e Inferencia
    with st.spinner("Inferencia Bayesiana y Construcción de Manifold..."):
        df_features = engineer_structural_features(df_hist, trading_days)
        current_regime, feature_list, regime_history = identify_bayesian_regimes(df_features)
    
    daily_returns = df_hist['Close'].pct_change().dropna()
    S0 = df_hist['Close'].iloc[-1]
    raw_mu = daily_returns.mean() * trading_days
    
    # Procesamiento y suavizado estocástico de las tendencias direccionales y volatilidad
    with st.spinner("Calibrando Filtros Matemáticos..."):
        kalman_states = apply_kalman_filter_dynamic(daily_returns.values, 100)
        pesos_ewma = np.exp(np.linspace(-1, 0, min(20, len(kalman_states))))
        current_mu = np.dot(kalman_states[-len(pesos_ewma):], pesos_ewma / pesos_ewma.sum()) * trading_days
        
        dynamic_sigma_daily = apply_stochastic_volatility_filter(daily_returns.values)
        current_sigma_ann = dynamic_sigma_daily[-1] * np.sqrt(trading_days)
        long_term_sigma_ann = np.mean(dynamic_sigma_daily) * np.sqrt(trading_days)

    # Reglas de Gobernanza: Condicionamiento Estocástico del motor según la inferencia de la IA
    if current_regime == -1:
        ml_lambda_j, ml_mu_j, ml_sigma_j = 6.0, -0.08, 0.08  # Configuración defensiva ante estrés estructural
    elif current_regime == 1:
        ml_lambda_j, ml_mu_j, ml_sigma_j = 1.0, 0.01, 0.02   # Configuración agresiva ante expansión estable
    else:
        ml_lambda_j, ml_mu_j, ml_sigma_j = 2.0, -0.03, 0.05  # Configuración neutral ante transiciones

    # Activación del generador estocástico acelerado por hardware mediante Numba JIT
    with st.spinner("Generación Estocástica de Escenarios..."):
        paths = run_montecarlo_advanced_stochastic(
            S0, current_mu, raw_mu, current_sigma_ann, long_term_sigma_ann, 
            days_to_project, simulations, ml_lambda_j, ml_mu_j, ml_sigma_j, trading_days
        )
        prob_pos, var_price, median_price, tp_price = calculate_risk_metrics_phase1(S0, paths, conf_level)
        persistencia, durbin_watson, exceedance = calculate_model_diagnostics(daily_returns.values, regime_history, kalman_states)

    # Cálculo exacto del costo de oportunidad empleando las tasas de interés reales (Bono TNX)
    downside_returns = daily_returns[daily_returns < 0]
    downside_sigma_ann = (downside_returns.std() * np.sqrt(trading_days)) if len(downside_returns) > 0 else current_sigma_ann
    sortino_ratio = (current_mu - risk_free_rate) / downside_sigma_ann if downside_sigma_ann > 0 else 0

    # ==============================================================================
    # DISEÑO Y DISTRIBUCIÓN DE LA INTERFAZ VISUAL (DASHBOARD SPLIT)
    # ==============================================================================
    col_proj, col_val = st.columns([2, 1])
    
    with col_proj:
        st.subheader("📊 Proyección de Escenarios Generados")
        c1, c2, c3 = st.columns(3)
        c1.metric("Precio Base ($)", f"{S0:,.2f}")
        
        estado_visual = "🔴 Estrés Estructural" if current_regime == -1 else "🟢 Expansión Estable" if current_regime == 1 else "⚪ Transición Latente"
        c2.metric("Régimen Geométrico (ML)", estado_visual)
        c3.metric("Touch Stop-Loss (VaR)", f"${var_price:,.2f}")

        # Renderización de la superficie de dispersión probabilística
        fig = go.Figure()
        for i in range(min(50, paths.shape[1])): 
            fig.add_trace(go.Scatter(
                x=np.arange(days_to_project + 1), y=paths[:, i], 
                mode='lines', line=dict(color='rgba(0, 100, 255, 0.08)'), showlegend=False
            ))
        fig.add_trace(go.Scatter(x=[0, days_to_project], y=[S0, S0], mode='lines', name='Nivel Entrada', line=dict(color='black', dash='dash')))
        fig.add_trace(go.Scatter(x=[0, days_to_project], y=[var_price, var_price], mode='lines', name='Touch VaR Baseline', line=dict(color='red', dash='dot')))
        fig.update_layout(
            height=400, 
            margin=dict(l=10, r=10, t=20, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_val:
        st.subheader("🛡️ Panel de Calibración y Auditoría")
        
        # 1. Métrica de Control de Fricción
        color_friccion = "normal" if abs(exceedance - 0.05) < 0.02 else "inverse"
        st.metric("Fricción Histórica (VaR Exceedance)", f"{exceedance*100:.1f}%", f"Objetivo de Distribución: ~5.0%", delta_color=color_friccion)
        st.caption("Verifica si el motor probabilístico con colas pesadas subestima o mapea correctamente los shocks pasados.")
        st.divider()
        
        # 2. Métrica de Estabilidad Temporal
        color_persistencia = "normal" if persistencia > 10 else "inverse"
        st.metric("Estabilidad de Manifold (Persistencia)", f"{persistencia:.1f} días", "Alerta de Drift Estructural" if persistencia < 10 else "Estructura de Régimen Sólida", delta_color=color_persistencia)
        st.caption("Mide la velocidad de cambio de la geometría latente calculada por el modelo Bayesiano.")
        st.divider()
        
        # 3. Métrica de Autocorrelación de Residuos
        st.metric("Pureza de Tendencia (Durbin-Watson)", f"{durbin_watson:.2f}", "Parámetro de Ruido Blanco Ideal: ~2.00")
        st.caption("Audita matemáticamente el filtro de tendencia para descartar patrones lineales ocultos o sobreajuste.")

    # Despliegue de la matriz de características complejas procesadas
    st.markdown("---")
    with st.expander("🔍 Laboratorio Cuantitativo: Vista Detallada de Características Estructurales", expanded=False):
        st.dataframe(df_features.tail(10).style.format("{:.4f}").background_gradient(cmap='Purples'), use_container_width=True)

if __name__ == "__main__":
    render_dashboard()
