import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import datetime
import unicodedata
from fpdf import FPDF

# Importaciones Core
from core.data_ingestion import get_macro_context, load_financial_data, generate_synthetic_data
from features.structural_engine import engineer_structural_features
from models.inference_bgmm import identify_bayesian_regimes
from simulation.stoch_generators import apply_auxiliary_particle_filter, apply_stochastic_volatility_filter, run_montecarlo_advanced_stochastic
from diagnostics.model_governance import calculate_model_diagnostics, calculate_risk_metrics_phase1, generate_directive

# Importaciones de Validación Avanzada
try:
    from diagnostics.advanced_validation import var_coverage_tests, calculate_pit_returns, berkowitz_tail_test, acerbi_szekely_es_test
except ImportError:
    st.error("Falta el módulo advanced_validation.py")

# ==============================================================================
# GOBERNANZA DE RIESGO (OVERLAY DEFENSIVO)
# ==============================================================================
def apply_defensive_overlay(current_mu_apf, raw_mu_historical, current_sigma, current_regime):
    """Evita la explosión geométrica anclando la deriva matemática a la realidad institucional."""
    blended_mu = (0.2 * current_mu_apf) + (0.8 * raw_mu_historical)
    
    if current_regime == 0:
        safe_mu = 0.0 # Transición: Deriva neutral
    elif current_regime == -1:
        safe_mu = -abs(blended_mu) if blended_mu != 0 else -0.10 # Estrés: Deriva negativa forzada
    else:
        safe_mu = np.clip(blended_mu, -0.40, 0.40) # Expansión: Hard Clipping al 40% anualizado
        
    safe_sigma = max(current_sigma, 0.15) # Piso de volatilidad
    return safe_mu, safe_sigma

# ==============================================================================
# MOTOR DEL TIEMPO (RETROSPECTIVO EXACTO)
# ==============================================================================
def run_time_machine(df_full, days_ago, simulations, trading_days):
    idx_split = -days_ago
    df_past = df_full.iloc[:idx_split]
    df_future = df_full.iloc[idx_split:]
    
    df_features = engineer_structural_features(df_past, trading_days)
    current_regime, _, _, calib_params, _ = identify_bayesian_regimes(df_features)
    
    daily_returns = df_past['Close'].pct_change().dropna()
    S0 = df_past['Close'].iloc[-1]
    raw_mu = daily_returns.mean() * trading_days
    
    apf_states = apply_auxiliary_particle_filter(daily_returns.values, num_particles=1000)
    pesos_ewma = np.exp(np.linspace(-1, 0, min(20, len(apf_states))))
    current_mu_apf = np.dot(apf_states[-len(pesos_ewma):], pesos_ewma / pesos_ewma.sum()) * trading_days
    
    dynamic_sigma = apply_stochastic_volatility_filter(daily_returns.values)
    current_sigma_ann = dynamic_sigma[-1] * np.sqrt(trading_days)
    long_term_sigma_ann = np.mean(dynamic_sigma) * np.sqrt(trading_days)
    
    # APLICAMOS EL FRENO INSTITUCIONAL
    safe_mu, safe_sigma_ann = apply_defensive_overlay(current_mu_apf, raw_mu, current_sigma_ann, current_regime)
    
    vi_params = calib_params.get(current_regime, {'lambda_j': 1.0, 'mu_regime': -0.01, 'sigma_regime': 0.05})
    ml_lambda_j = max(0.1, vi_params['lambda_j']) 
    ml_mu_j = np.clip(vi_params['mu_regime'], -0.20, 0.20)
    ml_sigma_j = np.clip(vi_params['sigma_regime'], 0.01, 0.30)
    
    paths = run_montecarlo_advanced_stochastic(
        S0, safe_mu, raw_mu, safe_sigma_ann, long_term_sigma_ann, 
        days_ago, simulations, ml_lambda_j, ml_mu_j, ml_sigma_j, trading_days
    )
    
    return df_future['Close'].values, paths, S0, df_future.index

# ==============================================================================
# ORQUESTADOR UI
# ==============================================================================
def render_dashboard():
    st.set_page_config(page_title="Quant Lab | Dinámica Estocástica", layout="wide", page_icon="🧬")
    
    with st.spinner("Sincronizando Radar Macro..."):
        risk_free_rate, current_vix = get_macro_context()
        
    st.markdown("<h4 style='text-align: center; color: #666;'>Radar Sistémico Global</h4>", unsafe_allow_html=True)
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Costo de Liquidez (US 10Y)", f"{risk_free_rate:.2f}%")
    col_m2.metric("Miedo Institucional (VIX)", f"{current_vix:.2f}", "Alerta Sistémica" if current_vix > 25 else "Nominal", delta_color="inverse" if current_vix > 25 else "normal")
    col_m3.metric("Entorno Cuantitativo", "Contracción" if risk_free_rate > 4.5 and current_vix > 20 else "Expansión")
    st.divider()

    with st.sidebar:
        st.header("1. Configuración de Activo")
        ASSET_UNIVERSE = {
            "🔍 Entrada Manual (Ticker)": "MANUAL",
            "--- MERCADO PERUANO ---": "HEADER",
            "🇵🇪 Credicorp Ltd. (BAP)": "BAP", "🇵🇪 Intercorp Financial (IFS)": "IFS", "🇵🇪 iShares MSCI Peru ETF (EPU)": "EPU",
            "--- ÍNDICES Y TECNOLOGÍA ---": "HEADER",
            "📊 S&P 500 ETF (SPY)": "SPY", "🇺🇸 NVIDIA (NVDA)": "NVDA", "₿ Bitcoin (BTC-USD)": "BTC-USD"
        }
        sel_asset = st.selectbox("Mercado:", list(ASSET_UNIVERSE.keys()), label_visibility="collapsed")
        if ASSET_UNIVERSE[sel_asset] == "HEADER": st.stop()
        ticker = st.text_input("...o ingrese Ticker Manual:", "").upper() or ASSET_UNIVERSE[sel_asset]
        
        st.divider()
        st.header("2. Arquitectura Estocástica")
        sym = st.selectbox("Moneda:", ["USD ($)", "EUR (€)", "PEN (S/)"]).split(" ")[1].replace("(", "").replace(")", "")
        capital_inicial = st.number_input("Capital a Invertir:", value=10000.0, step=1000.0)
        days_to_project = st.slider("Horizonte Futuro (Días):", 10, 252, 21)
        simulations = {"1k": 1000, "5k": 5000, "10k": 10000}[st.selectbox("Rutas (Monte Carlo):", ["1k", "5k", "10k"], index=1)]
        conf_level = st.slider("Confianza VaR (%):", 90.0, 99.9, 95.0, 0.1)
        
        st.divider()
        st.header("3. Máquina del Tiempo")
        days_ago = st.number_input("Rebobinar mercado (Días):", min_value=10, max_value=150, value=30)

    trading_days = 365 if "USD" in ticker else 252
    
    try:
        df_hist = load_financial_data(ticker, "")
    except Exception:
        df_hist = generate_synthetic_data(ticker, days=500, trading_days=trading_days)
        st.error("🚨 Conexión offline. Desplegando simulación teórica.")

    with st.spinner("Inferencia Variacional y Filtrado (Tiempo Real)..."):
        df_features = engineer_structural_features(df_hist, trading_days)
        current_regime, feature_list, regime_history, calib_params, trans_matrix = identify_bayesian_regimes(df_features)
        
        daily_returns = df_hist['Close'].pct_change().dropna()
        S0 = df_hist['Close'].iloc[-1]
        raw_mu = daily_returns.mean() * trading_days
        
        apf_states = apply_auxiliary_particle_filter(daily_returns.values, num_particles=1000)
        pesos_ewma = np.exp(np.linspace(-1, 0, min(20, len(apf_states))))
        current_mu_apf = np.dot(apf_states[-len(pesos_ewma):], pesos_ewma / pesos_ewma.sum()) * trading_days
        
        dynamic_sigma_daily = apply_stochastic_volatility_filter(daily_returns.values)
        current_sigma_ann = dynamic_sigma_daily[-1] * np.sqrt(trading_days)
        long_term_sigma_ann = np.mean(dynamic_sigma_daily) * np.sqrt(trading_days)

        # Freno Institucional
        safe_mu, safe_sigma_ann = apply_defensive_overlay(current_mu_apf, raw_mu, current_sigma_ann, current_regime)

        vi_params = calib_params.get(current_regime, {'lambda_j': 1.0, 'mu_regime': -0.01, 'sigma_regime': 0.05})
        ml_lambda_j = max(0.1, vi_params['lambda_j']) 
        ml_mu_j = np.clip(vi_params['mu_regime'], -0.20, 0.20)
        ml_sigma_j = np.clip(vi_params['sigma_regime'], 0.01, 0.30)

    with st.spinner("Compilando Escenarios Futuros..."):
        paths = run_montecarlo_advanced_stochastic(
            S0, safe_mu, raw_mu, safe_sigma_ann, long_term_sigma_ann, 
            days_to_project, simulations, ml_lambda_j, ml_mu_j, ml_sigma_j, trading_days
        )
        prob_pos, var_price, median_price, tp_price = calculate_risk_metrics_phase1(S0, paths, conf_level)
        
    capital_esperado = (capital_inicial / S0) * median_price
    capital_var = (capital_inicial / S0) * var_price
    rend_esp = capital_esperado - capital_inicial
    rend_var = capital_var - capital_inicial
    estado_ml_txt = "🔴 Estrés Estructural" if current_regime == -1 else "🟢 Expansión Estable" if current_regime == 1 else "⚪ Transición Latente"
    
    st.title(f"Telemetría Cuantitativa: {ticker}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precio Base", f"${S0:,.2f}")
    c2.metric("Régimen Inferred (VI)", estado_ml_txt.split(' ', 1)[1])
    c3.metric("Fricción VaR", f"{(1 - (var_price/S0))*100:.1f}%", delta_color="inverse")
    c4.metric("Deriva Ajustada (Segura)", f"{safe_mu*100:.1f}%")

    tab1, tab2, tab3 = st.tabs(["📈 Proyección Futura", "🔬 Auditoría de Manifold", "⏱️ Máquina del Tiempo (Validación)"])
    
    with tab1:
        dir_t, just_t = generate_directive(prob_pos, current_regime)
        col_dir, col_tp, col_sl = st.columns(3)
        col_dir.info(f"**DIRECTRIZ:**\n\n**{dir_t}**\n\n*{just_t}*")
        col_tp.success(f"**TAKE PROFIT:**\n\n**${tp_price:,.2f}**")
        col_sl.error(f"**STOP-LOSS (VaR {conf_level}%):**\n\n**${var_price:,.2f}**")
        
        fig = go.Figure()
        for i in range(min(50, paths.shape[1])): 
            fig.add_trace(go.Scatter(x=np.arange(days_to_project + 1), y=paths[:, i], mode='lines', line=dict(color='rgba(0,100,255,0.05)'), showlegend=False))
        fig.add_trace(go.Scatter(x=[0, days_to_project], y=[S0, S0], mode='lines', line=dict(color='black', dash='dash')))
        fig.add_trace(go.Scatter(x=[0, days_to_project], y=[var_price, var_price], mode='lines', line=dict(color='red', dash='dot')))
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        aligned_dates = df_features.index
        aligned_prices = df_hist.loc[aligned_dates, 'Close']
        fig_manifold = go.Figure()
        fig_manifold.add_trace(go.Scatter(x=aligned_dates, y=aligned_prices, mode='lines', line=dict(color='#333', width=1), name='Precio Activo'))
        color_map = {-1: 'rgba(255, 0, 0, 0.4)', 0: 'rgba(150, 150, 150, 0.4)', 1: 'rgba(0, 255, 0, 0.4)'}
        for reg_val, reg_name in zip([-1, 0, 1], ['Estrés', 'Transición', 'Expansión']):
            mask = (regime_history == reg_val)
            fig_manifold.add_trace(go.Scatter(x=aligned_dates[mask], y=aligned_prices[mask], mode='markers', marker=dict(color=color_map[reg_val], size=6), name=f'Reg. {reg_name}'))
        fig_manifold.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_manifold, use_container_width=True)

    with tab3:
        st.markdown(f"### Validación Empírica y Auditoría PnL: {days_ago} días al pasado")
        st.write("Aísla a la IA en el pasado, aplica los frenos institucionales y compara su predicción ciega contra la realidad.")
        
        if st.button(f"Ejecutar Máquina del Tiempo (-{days_ago}d)"):
            with st.spinner("Rebobinando mercado, aplicando overlays y calculando PnL histórico..."):
                real_prices, retro_paths, past_S0, future_dates = run_time_machine(df_hist, days_ago, simulations, trading_days)
                
                median_path = np.median(retro_paths, axis=1)
                upper_bound = np.percentile(retro_paths, 95, axis=1)
                lower_bound = np.percentile(retro_paths, 5, axis=1)
                
                current_real_price = real_prices[-1]
                projected_final_price = median_path[-1]
                
                shares_bought = capital_inicial / past_S0 
                
                actual_final_capital = shares_bought * current_real_price
                actual_pnl = actual_final_capital - capital_inicial
                actual_pnl_pct = (actual_pnl / capital_inicial) * 100
                
                projected_final_capital = shares_bought * projected_final_price
                projected_pnl = projected_final_capital - capital_inicial
                projected_error_margin = abs(actual_final_capital - projected_final_capital)
                
                st.markdown("#### 💰 Auditoría de Capital (Realidad vs. Modelo Seguro)")
                col_retro1, col_retro2, col_retro3 = st.columns(3)
                
                col_retro1.metric(
                    label=f"Inversión hace {days_ago}d (Precio: ${past_S0:,.2f})",
                    value=f"{sym}{capital_inicial:,.2f}",
                    help=f"Equivale a {shares_bought:,.4f} unidades."
                )
                col_retro2.metric(
                    label=f"Capital Hoy REAL (Precio: ${current_real_price:,.2f})",
                    value=f"{sym}{actual_final_capital:,.2f}",
                    delta=f"{sym}{actual_pnl:+,.2f} ({actual_pnl_pct:+.2f}%)",
                    delta_color="normal"
                )
                col_retro3.metric(
                    label=f"Capital PROYECTADO (Precio: ${projected_final_price:,.2f})",
                    value=f"{sym}{projected_final_capital:,.2f}",
                    delta=f"Margen de Error: {sym}{projected_error_margin:,.2f}",
                    delta_color="off"
                )
                
                st.divider()
                
                fig_time = go.Figure()
                x_axis = np.arange(len(median_path))
                
                fig_time.add_trace(go.Scatter(
                    x=np.concatenate([x_axis, x_axis[::-1]]),
                    y=np.concatenate([upper_bound, lower_bound[::-1]]),
                    fill='toself', fillcolor='rgba(0,100,255,0.1)', line=dict(color='rgba(255,255,255,0)'),
                    name="Cono de Probabilidad (IA)"
                ))
                
                fig_time.add_trace(go.Scatter(x=x_axis, y=median_path, mode='lines', line=dict(color='blue', dash='dash'), name='Mediana Proyectada'))
                limit = min(len(x_axis), len(real_prices))
                fig_time.add_trace(go.Scatter(x=x_axis[:limit], y=real_prices[:limit], mode='lines', line=dict(color='black', width=3), name='Realidad Empírica'))
                fig_time.add_trace(go.Scatter(x=[0], y=[past_S0], mode='markers', marker=dict(color='red', size=10), name='Punto de Inferencia Ciega'))
                fig_time.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_time, use_container_width=True)

if __name__ == "__main__": 
    render_dashboard()
