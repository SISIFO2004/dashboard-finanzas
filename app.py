import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta

# --- CONFIGURACIÓN DE LA APLICACIÓN ---
st.set_page_config(
    page_title="Quantitative Risk Analytics Engine", 
    layout="wide", 
    page_icon="📈"
)

# --- CATÁLOGO DE ACTIVOS FINANCIEROS ---
ASSET_UNIVERSE = {
    "🔍 Entrada Manual (Ticker)": "MANUAL",
    "🇺🇸 Tesla Inc. (TSLA)": "TSLA",
    "🇺🇸 Apple Inc. (AAPL)": "AAPL",
    "🇺🇸 NVIDIA Corp. (NVDA)": "NVDA",
    "🇺🇸 Microsoft Corp. (MSFT)": "MSFT",
    "🇺🇸 Amazon.com Inc. (AMZN)": "AMZN",
    "🇺🇸 S&P 500 ETF (SPY)": "SPY",
    "🥇 Gold Futures (GC=F)": "GC=F",
    "₿ Bitcoin USD (BTC-USD)": "BTC-USD"
}

# --- SISTEMA DE LOGS (AUDITORÍA DE PROCESOS) ---
def sys_log(message, level="INFO"):
    """Registra eventos del sistema en la consola del servidor con timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] [{level}] {message}")

# --- TÍTULO Y ENCABEZADO ---
st.title("📈 Quantitative Risk Analytics Engine")
st.markdown("""
**Sistema de Simulación Estocástica y Valoración de Riesgo de Mercado.**
Implementación de modelos de *Movimiento Browniano Geométrico (GBM)* y *Difusión con Saltos de Merton* para la proyección de precios y cálculo de VaR.
""")

# --- BARRA LATERAL: PARAMETRIZACIÓN ---
st.sidebar.header("1. Configuración del Activo")
selected_option = st.sidebar.selectbox("Instrumento Financiero:", list(ASSET_UNIVERSE.keys()), index=1)

if ASSET_UNIVERSE[selected_option] == "MANUAL":
    ticker = st.sidebar.text_input("Ingresar Ticker (Yahoo Finance Standard):", value="GOOGL").upper()
else:
    ticker = ASSET_UNIVERSE[selected_option]

st.sidebar.divider()

st.sidebar.header("2. Parámetros de Simulación")
time_horizon = st.sidebar.slider("Horizonte de Proyección (Días)", 5, 365, 30, help="Número de días bursátiles a proyectar (T).")
n_simulations = st.sidebar.selectbox("Iteraciones Montecarlo (N)", [1000, 5000, 10000, 20000], index=1, help="Número de trayectorias aleatorias a generar.")

st.sidebar.divider()

# --- PARÁMETROS AVANZADOS (QUANT) ---
st.sidebar.header("3. Ajuste de Modelo (Quant)")
with st.sidebar.expander("⚙️ Calibración Avanzada", expanded=False):
    st.markdown("### A. Drift (Tendencia)")
    override_drift = st.checkbox("Sobrescribir Drift Histórico (View)")
    manual_drift = st.slider("Drift Anual Esperado (%)", -50.0, 100.0, 10.0, step=0.5) / 100
    
    st.markdown("### B. Componente de Saltos (Merton)")
    enable_jumps = st.checkbox("Habilitar Saltos de Poisson", value=True, help="Añade discontinuidades aleatorias al precio.")
    jump_prob = st.slider("Probabilidad de Salto Anual (Lambda)", 0.0, 50.0, 5.0, step=1.0) / 100
    jump_mean = st.slider("Magnitud Media del Salto (%)", -30.0, 30.0, -5.0, step=0.5) / 100
    jump_std = st.slider("Volatilidad del Salto (%)", 0.0, 50.0, 10.0, step=1.0) / 100
    
    st.markdown("### C. Métricas de Riesgo")
    confidence_level = st.selectbox("Nivel de Confianza (VaR)", [0.90, 0.95, 0.99], index=1)

# --- MÓDULO DE INGESTA DE DATOS ---
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_data(symbol):
    """Descarga datos históricos y calcula parámetros estadísticos base."""
    sys_log(f"Iniciando solicitud de datos para: {symbol}")
    try:
        stock = yf.Ticker(symbol)
        # Se solicitan 2 años para una varianza más robusta
        df = stock.history(period="2y")
        
        if df.empty:
            sys_log(f"ADVERTENCIA: Payload vacío para {symbol}", "WARN")
            return None, None
        
        sys_log(f"Datos recibidos exitosamente: {len(df)} registros.")
        
        # Cálculo de Retornos Logarítmicos Continuos
        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df.dropna(inplace=True)
        
        # Parámetros Anualizados (252 días bursátiles)
        last_price = df['Close'].iloc[-1]
        volatility = df['Log_Ret'].std() * np.sqrt(252)
        historical_drift = df['Log_Ret'].mean() * 252
        
        asset_info = {
            "price": last_price,
            "currency": stock.info.get('currency', 'USD'),
            "name": stock.info.get('longName', symbol)
        }
        
        sys_log(f"Métricas calculadas -> Volatilidad: {volatility:.4f}, Drift Histórico: {historical_drift:.4f}")
        return df, (volatility, historical_drift, asset_info)
        
    except Exception as e:
        sys_log(f"ERROR CRÍTICO en API: {str(e)}", "ERROR")
        return None, None

# --- MOTOR DE SIMULACIÓN (NUMPY VECTORIZADO) ---
def run_simulation(S0, mu, sigma, T, N, dt, jumps, lambda_j, mu_j, sigma_j):
    """
    Ejecuta la simulación estocástica.
    Soporta GBM estándar y Difusión con Saltos.
    """
    sys_log("Inicializando Kernel Estocástico...")
    sys_log(f"Configuración: T={T}, N={N}, Jumps={jumps}")
    
    prices = np.zeros((T + 1, N))
    prices[0] = S0
    
    for t in range(1, T + 1):
        # 1. Componente Difusivo (Browniano)
        z = np.random.normal(0, 1, N)
        # Corrección de Itô para Drift: (mu - 0.5 * sigma^2)
        diffusion_term = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
        
        # 2. Componente de Salto (Poisson)
        jump_term = 0
        if jumps:
            # Poisson: Número de saltos en el intervalo dt
            n_jumps = np.random.poisson(lambda_j * dt, N)
            
            # Si hay saltos, calculamos su magnitud Log-Normal
            if np.any(n_jumps > 0):
                jump_magnitude = np.random.normal(mu_j, sigma_j, N) * n_jumps
                jump_term = jump_magnitude

        # Ecuación Diferencial Estocástica Discretizada
        prices[t] = prices[t-1] * np.exp(diffusion_term + jump_term)
        
    sys_log("Simulación completada. Matriz de precios generada.")
    return prices

# --- CONTROLADOR PRINCIPAL ---
if st.button(f"⚡ EJECUTAR ANÁLISIS PARA {ticker}", type="primary"):
    
    with st.spinner("Procesando datos y ejecutando simulación Montecarlo..."):
        # 1. Obtención de Datos
        historical_data, metrics = fetch_market_data(ticker)
        
    if historical_data is None:
        st.error("Error de conexión con el proveedor de datos de mercado. Verifique el Ticker o intente más tarde.")
    else:
        sigma_base, mu_hist, info = metrics
        S0 = info['price']
        
        # 2. Configuración de Parámetros del Modelo
        mu_final = manual_drift if override_drift else mu_hist
        dt = 1/252  # Paso de tiempo diario
        
        # 3. Ejecución de la Simulación
        simulation_results = run_simulation(
            S0=S0,
            mu=mu_final,
            sigma=sigma_base,
            T=time_horizon,
            N=n_simulations,
            dt=dt,
            jumps=enable_jumps,
            lambda_j=jump_prob,   # Intensidad anual
            mu_j=jump_mean,       # Media del salto
            sigma_j=jump_std      # Desv. Est. del salto
        )
        
        # 4. Procesamiento de Resultados Estadísticos
        final_prices = simulation_results[-1]
        mean_price = np.mean(final_prices)
        median_price = np.median(final_prices)
        
        # Cálculo de Riesgo (VaR & CVaR)
        # Si confianza es 95%, buscamos el percentil 5% de la cola izquierda
        alpha = 1 - confidence_level
        var_value = np.percentile(final_prices, alpha * 100)
        cvar_value = final_prices[final_prices <= var_value].mean()
        
        prob_positive_return = np.sum(final_prices > S0) / n_simulations
        
        # --- DASHBOARD DE RESULTADOS ---
        
        # A. Tarjetas de Métricas (KPIs)
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Precio Spot", f"{S0:.2f} {info['currency']}")
        kpi2.metric("Volatilidad (σ)", f"{sigma_base:.2%}")
        kpi3.metric("Drift Esperado (μ)", f"{mu_final:.2%}", delta="Manual" if override_drift else "Histórico")
        kpi4.metric(f"VaR ({confidence_level:.0%})", f"{var_value:.2f}", delta_color="inverse")
        
        st.divider()
        
        # B. Visualización Gráfica
        tab_sim, tab_dist, tab_data = st.tabs(["📈 Trayectorias Simuladas", "📊 Distribución & Riesgo", "📝 Resumen Técnico"])
        
        # GRÁFICO 1: TRAYECTORIAS
        with tab_sim:
            fig_sim = go.Figure()
            
            # Fechas futuras
            future_dates = [historical_data.index[-1] + timedelta(days=i) for i in range(time_horizon + 1)]
            
            # Muestreo de rutas para optimizar renderizado (max 100 líneas visuales)
            display_routes = 100 if n_simulations > 100 else n_simulations
            indices = np.random.choice(n_simulations, display_routes, replace=False)
            
            for i in indices:
                fig_sim.add_trace(go.Scatter(
                    x=future_dates, 
                    y=simulation_results[:, i], 
                    mode='lines', 
                    line=dict(width=0.5, color='rgba(100, 200, 255, 0.2)'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Líneas de Tendencia Central
            fig_sim.add_trace(go.Scatter(x=future_dates, y=np.mean(simulation_results, axis=1), mode='lines', name='Media Esperada', line=dict(color='orange', width=2)))
            fig_sim.add_trace(go.Scatter(x=future_dates, y=np.percentile(simulation_results, 5, axis=1), mode='lines', name=f'Límite Inferior {(alpha*100):.0f}%', line=dict(color='red', width=1, dash='dash')))
            
            fig_sim.update_layout(
                title=f"Proyección Estocástica: {ticker} ({time_horizon} días)",
                xaxis_title="Fecha",
                yaxis_title="Precio",
                template="plotly_dark",
                height=500
            )
            st.plotly_chart(fig_sim, use_container_width=True)
            
        # GRÁFICO 2: DISTRIBUCIÓN Y VaR
        with tab_dist:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=final_prices, 
                nbinsx=100, 
                name='Frecuencia', 
                marker_color='#636EFA', 
                opacity=0.7
            ))
            
            # Líneas de referencia
            fig_hist.add_vline(x=S0, line_dash="solid", line_color="white", annotation_text="Spot")
            fig_hist.add_vline(x=var_value, line_dash="dash", line_color="red", annotation_text=f"VaR {confidence_level:.0%}")
            fig_hist.add_vline(x=mean_price, line_dash="dash", line_color="orange", annotation_text="Media")
            
            fig_hist.update_layout(
                title="Distribución de Probabilidad de Precios Finales",
                xaxis_title="Precio Simulado",
                yaxis_title="Frecuencia",
                template="plotly_dark",
                height=450
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Análisis textual profesional
            st.markdown(f"""
            #### 🛡️ Informe de Riesgo Cuantitativo
            
            El análisis de **{n_simulations:,} escenarios** proyecta los siguientes resultados para un horizonte de **{time_horizon} días**:
            
            * **Probabilidad de Retorno Positivo:** {prob_positive_return:.1%}
            * **Value at Risk (VaR {confidence_level:.0%}):** Con un nivel de confianza del {confidence_level:.0%}, se estima que la pérdida máxima no excederá un precio suelo de **{var_value:.2f} {info['currency']}**.
            * **Conditional VaR (CVaR):** En el escenario de colapso extremo (peor {(1-confidence_level):.0%} de los casos), el precio promedio esperado es **{cvar_value:.2f} {info['currency']}**.
            """)
            
        # TAB 3: RESUMEN TÉCNICO Y EXPORTACIÓN
        with tab_data:
            st.write("#### 💾 Exportación de Datos")
            
            # Preparar Dataframe para descarga
            df_export = pd.DataFrame(final_prices, columns=["Precio_Simulado_Final"])
            df_export["Escenario"] = df_export.index + 1
            
            # Convertir a CSV
            csv = df_export.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Descargar Resultados de Simulación (CSV)",
                data=csv,
                file_name=f'simulacion_{ticker}_{datetime.now().strftime("%Y%m%d")}.csv',
                mime='text/csv',
                help="Descarga la distribución final de precios para auditoría externa."
            )

            st.write("#### ⚙️ Parámetros del Modelo")
            st.json({
                "Modelo": "Merton Jump Diffusion" if enable_jumps else "Geometric Brownian Motion",
                "Precio Spot (S0)": S0,
                "Drift (mu)": mu_final,
                "Volatilidad (sigma)": sigma_base,
                "Saltos Activos": enable_jumps,
                "Intensidad Saltos (Lambda)": jump_prob if enable_jumps else 0,
                "Horizonte (T)": time_horizon,
                "Iteraciones (N)": n_simulations,
                "Nivel Confianza VaR": confidence_level
            })

        # --- SECCIÓN: DIRECTRIZ ESTRATÉGICA (RESTITUIDA Y MEJORADA) ---
        st.write("---")
        st.subheader("🧭 Directriz Estratégica (Algorithmic Guidance)")
        
        # Lógica de Decisión para la recomendación
        col_strat1, col_strat2 = st.columns(2)
        
        # 1. Determinación de Tendencia
        if prob_positive_return > 0.65:
            signal_color = "🟢"
            signal_text = "TENDENCIA ALCISTA (BULLISH)"
            advice_text = "El modelo sugiere una probabilidad estadística favorable para posiciones LARGAS."
            box_type = "success"
        elif prob_positive_return < 0.35:
            signal_color = "🔴"
            signal_text = "TENDENCIA BAJISTA (BEARISH)"
            advice_text = "La proyección indica deterioro de valor. Se sugiere cobertura (Hedging) o evitar entrada."
            box_type = "error"
        else:
            signal_color = "🟡"
            signal_text = "INCERTIDUMBRE / RANGO (NEUTRAL)"
            advice_text = "No existe una ventaja estadística clara (Ruido > Tendencia). Se recomienda cautela."
            box_type = "warning"
            
        # 2. Perfil de Volatilidad
        if sigma_base < 0.15:
            vol_profile = "Baja (Conservador)"
        elif sigma_base < 0.35:
            vol_profile = "Media (Moderada)"
        else:
            vol_profile = "Alta (Especulativa)"

        with col_strat1:
            st.markdown(f"#### Señal del Modelo: {signal_color} {signal_text}")
            if box_type == "success":
                st.success(advice_text)
            elif box_type == "error":
                st.error(advice_text)
            else:
                st.warning(advice_text)
                
        with col_strat2:
            st.markdown("#### Parámetros de Gestión")
            st.info(f"""
            * **Perfil de Activo:** {vol_profile} (Vol: {sigma_base:.1%})
            * **Stop Loss Técnico (Sugerido):** {var_value:.2f} {info['currency']}
            * **Probabilidad de Éxito:** {prob_positive_return:.1%}
            """)

        st.caption("⚠️ **Disclaimer:** Esta herramienta es un modelo de simulación matemática para apoyo en la toma de decisiones, no constituye asesoramiento financiero certificado.")

else:
    st.info("ℹ️ Seleccione los parámetros en la barra lateral y presione 'EJECUTAR ANÁLISIS' para iniciar el cálculo.")