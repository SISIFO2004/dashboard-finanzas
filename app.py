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

try:
    from diagnostics.backtest_engine import run_walk_forward_backtest
except ImportError:
    run_walk_forward_backtest = None

# ==============================================================================
# LÓGICA DE PDF Y SANITIZACIÓN INSTITUCIONAL
# ==============================================================================
def clean_text_for_pdf(text):
    text = str(text)
    for e in ['🔴', '🟢', '⚪', '⚠️', '🛡️', '📉', '🔥', '🌪️', '✅', '⚙️', '🔬', '📊', '🧬', '📥', '⏱️', '🧪', '💰']: 
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
# MOTOR DEL TIEMPO (RETROSPECTIVO)
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
    current_mu = np.dot(apf_states[-len(pesos_ewma):], pesos_ewma / pesos_ewma.sum()) * trading_days
    
    dynamic_sigma = apply_stochastic_volatility_filter(daily_returns.values)
    current_sigma_ann = dynamic_sigma[-1] * np.sqrt(trading_days)
    long_term_sigma_ann = np.mean(dynamic_sigma) * np.sqrt(trading_days)
    
    vi_params = calib_params.get(current_regime, {'lambda_j': 1.0, 'mu_regime': -0.01, 'sigma_regime': 0.05})
    ml_lambda_j = max(0.1, vi_params['lambda_j']) 
    ml_mu_j = np.clip(vi_params['mu_regime'], -0.20, 0.20)
    ml_sigma_j = np.clip(vi_params['sigma_regime'], 0.01, 0.30)
    
    paths = run_montecarlo_advanced_stochastic(
        S0, current_mu, raw_mu, current_sigma_ann, long_term_sigma_ann, 
        days_ago, simulations, ml_lambda_j, ml_mu_j, ml_sigma_j, trading_days
    )
    
    return df_future['Close'].values, paths, S0, df_future.index

# ==============================================================================
# ORQUESTADOR UI (FULL-STACK QUANT)
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
            "--- ÍNDICES GLOBALES ---": "HEADER",
            "📊 S&P 500 ETF (SPY)": "SPY", "📊 Nasdaq 100 (QQQ)": "QQQ", 
            "--- MERCADO PERUANO ---": "HEADER",
            "🇵🇪 iShares MSCI Peru ETF (EPU)": "EPU", "🇵🇪 Credicorp Ltd. (BAP)": "BAP", 
            "🇵🇪 Southern Copper (SCCO)": "SCCO", "🇵🇪 Intercorp Financial (IFS)": "IFS", 
            "--- TECNOLOGÍA ---": "HEADER",
            "🇺🇸 NVIDIA (NVDA)": "NVDA", "🇺🇸 Apple (AAPL)": "AAPL", "🇺🇸 Tesla (TSLA)": "TSLA",
            "--- CRIPTOMONEDAS ---": "HEADER",
            "₿ Bitcoin (BTC-USD)": "BTC-USD", "⟠ Ethereum (ETH-USD)": "ETH-USD"
        }
        sel_asset = st.selectbox("Mercado:", list(ASSET_UNIVERSE.keys()), label_visibility="collapsed")
        if ASSET_UNIVERSE[sel_asset] == "HEADER": st.stop()
        ticker = st.text_input("...o ingrese Ticker Manual:", "").upper() or ASSET_UNIVERSE[sel_asset]
        
        st.divider()
        st.header("2. Arquitectura Estocástica")
        moneda_sel = st.selectbox("Moneda:", ["USD ($)", "EUR (€)", "PEN (S/)"])
        sym = moneda_sel.split(" ")[1].replace("(", "").replace(")", "")
        capital_inicial = st.number_input("Capital a Invertir:", value=10000.0, step=1000.0)
        days_to_project = st.slider("Horizonte Futuro (Días):", 10, 252, 21)
        simulations = {"1k": 1000, "5k": 5000, "10k": 10000}[st.selectbox("Rutas (Monte Carlo):", ["1k", "5k", "10k"], index=1)]
        conf_level = st.slider("Confianza VaR (%):", 90.0, 99.9, 95.0, 0.1)
        
        st.divider()
        st.header("3. Máquina del Tiempo")
        days_ago = st.number_input("Rebobinar mercado (Días):", min_value=10, max_value=150, value=30)
        
        st.divider()
        st.header("4. Gobernanza de Riesgo")
        override_ia = st.toggle("☢️ Stress Test (Ignorar IA)")
        if override_ia:
            st.error("Motor Variacional en Bypass. Simulando colas pesadas (Hawkes).")
            manual_lambda = st.slider("Intensidad (λ)", 0.1, 10.0, 5.0)
            manual_mu = st.slider("Profundidad (μ)", -0.20, 0.0, -0.05)
            manual_sigma = st.slider("Volatilidad (σ)", 0.01, 0.20, 0.08)

    trading_days = 365 if "USD" in ticker else 252
    
    try:
        df_hist = load_financial_data(ticker, "")
    except Exception:
        df_hist = generate_synthetic_data(ticker, days=500, trading_days=trading_days)
        st.error("🚨 Conexión offline. Desplegando simulación teórica.")

    with st.spinner("Inferencia Variacional (Tiempo Real)..."):
        df_features = engineer_structural_features(df_hist, trading_days)
        current_regime, feature_list, regime_history, calib_params, trans_matrix = identify_bayesian_regimes(df_features)
    
    daily_returns = df_hist['Close'].pct_change().dropna()
    S0 = df_hist['Close'].iloc[-1]
    raw_mu = daily_returns.mean() * trading_days
    
    with st.spinner("Extrayendo Tendencia (APF)..."):
        apf_states = apply_auxiliary_particle_filter(daily_returns.values, num_particles=1000)
        pesos_ewma = np.exp(np.linspace(-1, 0, min(20, len(apf_states))))
        current_mu = np.dot(apf_states[-len(pesos_ewma):], pesos_ewma / pesos_ewma.sum()) * trading_days
        
        dynamic_sigma_daily = apply_stochastic_volatility_filter(daily_returns.values)
        current_sigma_ann = dynamic_sigma_daily[-1] * np.sqrt(trading_days)
        long_term_sigma_ann = np.mean(dynamic_sigma_daily) * np.sqrt(trading_days)

    if override_ia:
        ml_lambda_j, ml_mu_j, ml_sigma_j = manual_lambda, manual_mu, manual_sigma
    else:
        vi_params = calib_params.get(current_regime, {'lambda_j': 1.0, 'mu_regime': -0.01, 'sigma_regime': 0.05})
        ml_lambda_j = max(0.1, vi_params['lambda_j']) 
        ml_mu_j = np.clip(vi_params['mu_regime'], -0.20, 0.20)
        ml_sigma_j = np.clip(vi_params['sigma_regime'], 0.01, 0.30)

    with st.spinner("Compilando Escenarios Futuros (Hawkes)..."):
        paths = run_montecarlo_advanced_stochastic(
            S0, current_mu, raw_mu, current_sigma_ann, long_term_sigma_ann, 
            days_to_project, simulations, ml_lambda_j, ml_mu_j, ml_sigma_j, trading_days
        )
        prob_pos, var_price, median_price, tp_price = calculate_risk_metrics_phase1(S0, paths, conf_level)
        
        aligned_returns = daily_returns.iloc[-len(regime_history):].values
        aligned_apf = apf_states[-len(regime_history):]
        persistencia, durbin_watson, exceedance = calculate_model_diagnostics(aligned_returns, regime_history, aligned_apf)

    capital_esperado = (capital_inicial / S0) * median_price
    capital_var = (capital_inicial / S0) * var_price
    rend_esp = capital_esperado - capital_inicial
    rend_var = capital_var - capital_inicial
    estado_ml_txt = "🔴 Estrés Estructural" if current_regime == -1 else "🟢 Expansión Estable" if current_regime == 1 else "⚪ Transición Latente"
    
    st.title(f"Telemetría Cuantitativa: {ticker}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precio Base", f"${S0:,.2f}")
    c2.metric("Régimen Inferred (VI)", estado_ml_txt.split(' ', 1)[1])
    c3.metric("Fricción VaR", f"{exceedance*100:.1f}%", delta_color="inverse")
    c4.metric("Deriva APF", f"{current_mu*100:.1f}%")

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Proyección Futura", "🔬 Auditoría de Manifold", "🧪 Backtest (Walk-Forward)", "⏱️ Máquina del Tiempo"])
    
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
        if run_walk_forward_backtest is not None:
            st.markdown("### Rendimiento Histórico")
            if st.button("Ejecutar Walk-Forward"):
                with st.spinner("Procesando ventanas dinámicas..."):
                    bt_results = run_walk_forward_backtest(df_hist)
                    cum_pnl = bt_results['Pnl'].cumsum()
                    fig_bt = go.Figure()
                    fig_bt.add_trace(go.Scatter(x=cum_pnl.index, y=cum_pnl, line=dict(color='green')))
                    st.plotly_chart(fig_bt, use_container_width=True)
        else:
            st.warning("El módulo de backtest no está disponible.")

    with tab4:
        st.markdown(f"### Validación Empírica: {days_ago} días al pasado")
        st.write("Aísla a la IA en el pasado y compara su predicción ciega contra el rendimiento del capital real.")
        
        if st.button(f"Ejecutar Máquina del Tiempo (-{days_ago}d)"):
            with st.spinner("Rebobinando mercado y calculando PnL histórico..."):
                real_prices, retro_paths, past_S0, future_dates = run_time_machine(df_hist, days_ago, simulations, trading_days)
                
                median_path = np.median(retro_paths, axis=1)
                upper_bound = np.percentile(retro_paths, 95, axis=1)
                lower_bound = np.percentile(retro_paths, 5, axis=1)
                
                # ==========================================================
                # CÁLCULOS FINANCIEROS
                # ==========================================================
                current_real_price = real_prices[-1]
                projected_final_price = median_path[-1]
                
                shares_bought = capital_inicial / past_S0 
                
                actual_final_capital = shares_bought * current_real_price
                actual_pnl = actual_final_capital - capital_inicial
                actual_pnl_pct = (actual_pnl / capital_inicial) * 100
                
                projected_final_capital = shares_bought * projected_final_price
                projected_pnl = projected_final_capital - capital_inicial
                projected_error_margin = abs(actual_final_capital - projected_final_capital)
                
                st.markdown("#### 💰 Auditoría de Capital (Realidad vs. Modelo)")
                col_retro1, col_retro2, col_retro3 = st.columns(3)
                
                col_retro1.metric(
                    label=f"Inversión hace {days_ago}d (Precio: ${past_S0:,.2f})",
                    value=f"{sym}{capital_inicial:,.2f}",
                    help=f"Equivale a {shares_bought:,.4f} unidades del activo."
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
                
                # ==========================================================
                # RENDERIZADO DEL MANIFOLD (GRÁFICA)
                # ==========================================================
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

    # --- EXPORTACIÓN ---
    report_data = {
        "ticker": ticker, "S0": S0, "tp_price": tp_price, "var_price": var_price,
        "estado_ml": estado_ml_txt.split(' ', 1)[1], "persistencia": persistencia, "exceedance": exceedance,
        "conf_level": conf_level, "insights": interpret_structural_features(df_features.iloc[-1]), "sym": sym, "directriz": dir_t,
        "justificacion": just_t, "capital_inicial": capital_inicial, 
        "capital_esperado": capital_esperado, "capital_var": capital_var,
        "rend_esp": rend_esp, "vol_dyn": current_sigma_ann * 100, 
        "contexto_riesgo": "Estrés" if current_regime == -1 else "Nominal",
        "days": days_to_project, "tnx": risk_free_rate, "vix": current_vix
    }
    st.sidebar.divider()
    st.sidebar.download_button("📥 Reporte PDF", create_pdf_report(report_data), f"Mem_{ticker}.pdf", mime="application/pdf", use_container_width=True)

if __name__ == "__main__": 
    render_dashboard()
