import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ==============================================================================
# IMPORTACIONES DESDE LA ARQUITECTURA MODULAR
# ==============================================================================
from core.data_ingestion import get_macro_context, load_financial_data, generate_synthetic_data
from features.structural_engine import engineer_structural_features
from models.inference_bgmm import identify_bayesian_regimes
from simulation.stoch_generators import apply_kalman_filter_dynamic, apply_stochastic_volatility_filter, run_montecarlo_advanced_stochastic
from diagnostics.model_governance import calculate_model_diagnostics, calculate_risk_metrics_phase1

def render_dashboard():
    st.set_page_config(page_title="Quant Lab: Research & Inference", layout="wide", page_icon="🧬")
    st.title("🧬 Laboratorio Cuantitativo: Inferencia de Riesgo")

    with st.spinner("Sincronizando Entorno Macroeconómico..."):
        risk_free_rate, current_vix = get_macro_context()

    with st.sidebar:
        st.header("1. Configuración de Entorno")
        
        # RESTAURACIÓN DEL UNIVERSO DE ACTIVOS CORPORATIVO
        ASSET_UNIVERSE = {
            "🔍 Entrada Manual (Ticker)": "MANUAL",
            "--- ÍNDICES Y ETFS ---": "HEADER",
            "📊 S&P 500 ETF (SPY)": "SPY",
            "📊 Nasdaq 100 ETF (QQQ)": "QQQ",
            "--- TECNOLOGÍA (MAG 7) ---": "HEADER",
            "🇺🇸 Apple Inc. (AAPL)": "AAPL",
            "🇺🇸 Microsoft Corp. (MSFT)": "MSFT",
            "🇺🇸 NVIDIA Corp. (NVDA)": "NVDA",
            "🇺🇸 Alphabet Inc. (GOOGL)": "GOOGL",
            "🇺🇸 Amazon.com Inc. (AMZN)": "AMZN",
            "🇺🇸 Meta Platforms (META)": "META",
            "🇺🇸 Tesla Inc. (TSLA)": "TSLA",
            "--- FINANZAS E INDUSTRIA ---": "HEADER",
            "🇺🇸 JPMorgan Chase (JPM)": "JPM",
            "🇺🇸 Berkshire Hathaway (BRK-B)": "BRK-B",
            "--- CRIPTOMONEDAS ---": "HEADER",
            "₿ Bitcoin (BTC-USD)": "BTC-USD",
            "⟠ Ethereum (ETH-USD)": "ETH-USD",
            "☀️ Solana (SOL-USD)": "SOL-USD",
            "--- MATERIAS PRIMAS ---": "HEADER",
            "🥇 Oro ETF (GLD)": "GLD",
            "🥈 Plata ETF (SLV)": "SLV",
            "🛢️ Petróleo ETF (USO)": "USO"
        }
        
        sel_asset = st.selectbox("Seleccione un Activo:", list(ASSET_UNIVERSE.keys()))
        
        if ASSET_UNIVERSE[sel_asset] == "HEADER":
            st.warning("⚠️ Selecciona un activo válido abajo.")
            st.stop()
            
        custom_asset = st.text_input("...o ingrese Ticker Manual:", "")
        ticker = custom_asset.upper() if custom_asset else ASSET_UNIVERSE[sel_asset]
        
        st.divider()
        capital_inicial = st.number_input("Capital Base ($):", min_value=10.0, value=10000.0, step=1000.0)
        days_to_project = st.slider("Horizonte de Proyección (Días):", 10, 252, 21)
        simulations = 5000
        conf_level = st.slider("Límite de Confianza VaR (%):", 90.0, 99.9, 95.0, 0.1)

    trading_days = 365 if "USD" in ticker else 252

    try:
        df_hist = load_financial_data(ticker, "")
        st.caption(f"✅ Conexión estable a mercado corporativo | Base temporal: {trading_days} días")
    except ValueError:
        df_hist = generate_synthetic_data(ticker, days=500, trading_days=trading_days)
        st.error("🚨 **ALERTA CRÍTICA DE INFRAESTRUCTURA:** Conexión fallida con proveedores externos. Desplegando simulación teórica base.")

    if not df_hist.empty:
        df_hist = df_hist.iloc[:-21]

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

    col_proj, col_val = st.columns([2, 1])
    
    with col_proj:
        st.subheader("📊 Proyección de Escenarios Generados")
        c1, c2, c3 = st.columns(3)
        c1.metric("Precio Base ($)", f"{S0:,.2f}")
        estado_visual = "🔴 Estrés Estructural" if current_regime == -1 else "🟢 Expansión Estable" if current_regime == 1 else "⚪ Transición Latente"
        c2.metric("Régimen Geométrico (ML)", estado_visual)
        c3.metric("Touch Stop-Loss (VaR)", f"${var_price:,.2f}")

        fig = go.Figure()
        for i in range(min(50, paths.shape[1])): fig.add_trace(go.Scatter(x=np.arange(days_to_project + 1), y=paths[:, i], mode='lines', line=dict(color='rgba(0, 100, 255, 0.08)'), showlegend=False))
        fig.add_trace(go.Scatter(x=[0, days_to_project], y=[S0, S0], mode='lines', name='Nivel Entrada', line=dict(color='black', dash='dash')))
        fig.add_trace(go.Scatter(x=[0, days_to_project], y=[var_price, var_price], mode='lines', name='Touch VaR Baseline', line=dict(color='red', dash='dot')))
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)

    with col_val:
        st.subheader("🛡️ Panel de Calibración y Auditoría")
        color_friccion = "normal" if abs(exceedance - 0.05) < 0.02 else "inverse"
        st.metric("Fricción Histórica (VaR Exceedance)", f"{exceedance*100:.1f}%", f"Objetivo: ~5.0%", delta_color=color_friccion)
        st.divider()
        color_persistencia = "normal" if persistencia > 10 else "inverse"
        st.metric("Estabilidad de Manifold", f"{persistencia:.1f} días", "Alerta de Drift" if persistencia < 10 else "Estructura Sólida", delta_color=color_persistencia)
        st.divider()
        st.metric("Pureza de Tendencia (Durbin-Watson)", f"{durbin_watson:.2f}", "Ideal: ~2.00")

    st.markdown("---")
    with st.expander("🔍 Laboratorio Cuantitativo: Matriz de Características", expanded=False):
        # Utilizamos un fallback seguro en caso de que matplotlib aún no se haya cargado en el entorno
        try:
            st.dataframe(df_features.tail(10).style.format("{:.4f}").background_gradient(cmap='Purples'), use_container_width=True)
        except Exception:
            st.dataframe(df_features.tail(10), use_container_width=True)

if __name__ == "__main__":
    render_dashboard()
