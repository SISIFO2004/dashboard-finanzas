import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
import time

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Market Quant Dashboard Pro", layout="wide", page_icon="📈")

# --- LISTA DE EMPRESAS PRINCIPALES ---
TOP_ACCIONES = {
    "🔍 Escribir otro ticker...": "OTRO",
    "🇵🇪 Alicorp (Bolsa de Lima)": "ALICORC1.LM",
    "🇺🇸 Tesla (Nasdaq)": "TSLA",
    "🇪🇸 Repsol (Bolsa de Madrid)": "REP.MC",
    "🇺🇸 Apple": "AAPL",
    "🇺🇸 Amazon": "AMZN",
    "🇺🇸 Google (Alphabet)": "GOOGL",
    "🇺🇸 NVIDIA": "NVDA",
    "⛏️ Buenaventura (Minera Perú)": "BVN",
    "🏦 Credicorp (BAP)": "BAP"
}

# --- FUNCIÓN DE LOGS PARA LA TERMINAL (LO QUE QUIERE DAVID) ---
def log_terminal(mensaje):
    """Imprime en la pantalla negra con hora exacta"""
    hora = datetime.now().strftime("%H:%M:%S")
    print(f"[{hora}] [INFO] {mensaje}")

# --- TÍTULO ---
st.title("📈 Tablero Financiero: Montecarlo & Volatilidad")
st.markdown("Plataforma de predicción algorítmica y análisis de riesgo.")

# --- BARRA LATERAL ---
st.sidebar.header("1. Configuración")
opcion = st.sidebar.selectbox("Empresa:", list(TOP_ACCIONES.keys()), index=2)

if TOP_ACCIONES[opcion] == "OTRO":
    ticker = st.sidebar.text_input("Ticker manual:", value="SPY").upper()
else:
    ticker = TOP_ACCIONES[opcion]

st.sidebar.divider()
dias_proyeccion = st.sidebar.slider("Días Futuros", 15, 365, 30)
n_escenarios = st.sidebar.selectbox("Simulaciones", [1000, 5000, 10000], index=1)

# --- MOTOR DE DATOS (CON CACHÉ ANTI-BLOQUEO) ---
# ESTA ES LA MODIFICACIÓN CLAVE: ttl=3600 significa "recuerda esto por 1 hora"
@st.cache_data(ttl=3600)
def obtener_datos_completos(ticker):
    print("\n" + "="*60)
    log_terminal(f"INICIANDO PROTOCOLO DE CONEXIÓN: {ticker}")
    log_terminal(f"Estableciendo handshake con Yahoo Finance API v8...")
    
    # Descargar datos
    stock = yf.Ticker(ticker)
    df = stock.history(period="1y")
    
    if df.empty:
        log_terminal(f"[ERROR CRÍTICO] No se recibió payload de datos.")
        return None, None
    
    log_terminal(f"Paquete recibido: {len(df)} registros históricos (OHLCV).")
    
    # Cálculos matemáticos
    log_terminal("Calculando retornos logarítmicos continuos...")
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    
    log_terminal("Generando matriz de Volatilidad Móvil (Rolling Window 30d)...")
    df['Volatilidad_Movil'] = df['Log_Ret'].rolling(window=30).std() * np.sqrt(252)
    
    # Métricas actuales
    precio_actual = df['Close'].iloc[-1]
    sigma = df['Log_Ret'].std() * np.sqrt(252)
    mu = df['Log_Ret'].mean() * 252
    
    log_terminal(f"Métricas procesadas -> Volatilidad: {sigma:.2%}, Drift: {mu:.2%}")
    
    info_extra = {
        "precio": precio_actual,
        "moneda": stock.info.get('currency', 'USD'),
        "nombre": stock.info.get('longName', ticker),
        "volumen": df['Volume'].iloc[-1]
    }
    return df, (sigma, mu, info_extra)

def motor_montecarlo_avanzado(S0, mu, sigma, T_dias, N):
    log_terminal(f"EJECUTANDO KERNEL ESTOCÁSTICO (GBM)...")
    log_terminal(f"Parámetros: S0={S0:.2f}, N={N} hilos.")
    
    dt = 1/252
    rutas = np.zeros((T_dias + 1, N))
    rutas[0] = S0
    
    log_terminal("Inyectando ruido blanco gaussiano...")
    for t in range(1, T_dias + 1):
        z = np.random.normal(0, 1, N)
        rutas[t] = rutas[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
        
    log_terminal(f"Renderizado de {N} universos paralelos completado.")
    print("="*60 + "\n")
    return rutas

# --- INTERFAZ VISUAL ---
if st.button(f"🔍 ANALIZAR {ticker} (CORRER PROCESO)", type="primary"):
    
    with st.spinner("Procesando en el backend..."):
        historial, metricas = obtener_datos_completos(ticker)
        
    if historial is None:
        st.error("Error de datos o bloqueo de API. Intenta más tarde.")
    else:
        sigma, mu, info = metricas
        S0 = info['precio']
        
        # 1. TARJETAS DE DATOS (KPIs)
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Precio Actual", f"{info['precio']:.2f} {info['moneda']}")
        kpi2.metric("Volatilidad Anual", f"{sigma*100:.2f}%")
        kpi3.metric("Tendencia (Drift)", f"{mu*100:.2f}%")
        kpi4.metric("Volumen", f"{info['volumen']:,}")
        
        # 2. SIMULACIÓN
        rutas = motor_montecarlo_avanzado(S0, mu, sigma, dias_proyeccion, n_escenarios)
        
        # Preparar datos para gráficos y análisis
        fechas_futuras = [historial.index[-1] + timedelta(days=x) for x in range(dias_proyeccion + 1)]
        promedio_rutas = np.mean(rutas, axis=1)
        precios_finales = rutas[-1]
        
        # Cálculos de probabilidad
        var_95 = np.percentile(precios_finales, 5)
        prob_ganancia = np.sum(precios_finales > S0) / n_escenarios * 100

        st.divider()
        
        # 3. PESTAÑAS DE ANÁLISIS
        tab1, tab2, tab3 = st.tabs(["📈 Gráfico Continuo", "⚡ Volatilidad", "📊 Distribución"])
        
        # GRÁFICO 1: CONTINUIDAD
        with tab1:
            fig_main = go.Figure()
            # Historia
            fig_main.add_trace(go.Scatter(x=historial.index, y=historial['Close'], mode='lines', name='Histórico', line=dict(color='#00CC96', width=2)))
            
            # Futuro (Rutas)
            for i in range(50): 
                fig_main.add_trace(go.Scatter(
                    x=fechas_futuras, 
                    y=rutas[:, i], 
                    mode='lines', 
                    showlegend=False, 
                    opacity=0.1, 
                    line=dict(color='white', width=0.5)
                ))
                
            # Promedio
            fig_main.add_trace(go.Scatter(x=fechas_futuras, y=promedio_rutas, mode='lines', name='Proyección Media', line=dict(color='orange', width=3, dash='dash')))
            
            fig_main.update_layout(template="plotly_dark", title=f"Proyección Continua: {ticker}", height=500)
            st.plotly_chart(fig_main, use_container_width=True)
            
        # GRÁFICO 2: VOLATILIDAD
        with tab2:
            st.subheader("Evolución del 'Nerviosismo' del Mercado")
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(x=historial.index, y=historial['Volatilidad_Movil']*100, mode='lines', fill='tozeroy', line=dict(color='#EF553B'), name='Volatilidad 30d'))
            fig_vol.update_layout(template="plotly_dark", title="Volatilidad Histórica Rodante (%)", height=400)
            st.plotly_chart(fig_vol, use_container_width=True)
            
        # GRÁFICO 3: HISTOGRAMA
        with tab3:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=precios_finales, nbinsx=60, marker_color='#636EFA', name='Escenarios'))
            
            fig_hist.add_vline(x=S0, line_dash="dash", annotation_text="Hoy")
            fig_hist.add_vline(x=var_95, line_color="red", annotation_text="VaR 95%")
            
            fig_hist.update_layout(template="plotly_dark", title="Distribución de Probabilidades", height=400)
            st.plotly_chart(fig_hist, use_container_width=True)
            st.caption(f"El riesgo (VaR 95%) indica que hay un 95% de probabilidad de mantenerse sobre {var_95:.2f} {info['moneda']}")

        # 4. SECCIÓN: INTERPRETACIÓN DE DATOS
        st.write("---")
        st.subheader("💡 Interpretación del Algoritmo (Guía de Inversión)")
        
        with st.expander("📝 LEER REPORTE DE INTERPRETACIÓN", expanded=True):
            
            # A. Análisis de Volatilidad
            st.markdown("#### 1. Perfil de Riesgo")
            if sigma < 0.15:
                st.success(f"🟢 **BAJO RIESGO (Volatilidad: {sigma:.1%})**: Esta acción es bastante estable. Ideal para perfiles conservadores.")
            elif sigma < 0.35:
                st.warning(f"🟡 **RIESGO MODERADO (Volatilidad: {sigma:.1%})**: La acción tiene movimientos normales de mercado. Requiere tolerancia a subidas y bajadas.")
            else:
                st.error(f"🔴 **ALTO RIESGO (Volatilidad: {sigma:.1%})**: ¡Cuidado! Es una acción muy volátil. Puedes ganar mucho o perder mucho rápido.")

            # B. Análisis de Probabilidad
            st.markdown("#### 2. Probabilidad Matemática")
            col_a, col_b = st.columns(2)
            
            col_a.metric("Probabilidad de Ganancia", f"{prob_ganancia:.1f}%")
            
            if prob_ganancia > 65:
                col_b.info("🚀 **Favorable:** El algoritmo sugiere una alta probabilidad de subida basada en la tendencia histórica.")
            elif prob_ganancia > 50:
                col_b.warning("⚖️ **Incierto:** Es casi como lanzar una moneda (50/50). La tendencia no es clara.")
            else:
                col_b.error("📉 **Desfavorable:** La tendencia histórica es bajista. Estadísticamente es probable perder valor.")

            # C. Guía de Stop Loss
            st.markdown("#### 3. Gestión de Riesgo (¿Cuándo salir?)")
            st.write(f"""
            Si decides invertir en **{info['precio']:.2f} {info['moneda']}**, el algoritmo calcula el **Valor en Riesgo (VaR 95%)** en **{var_95:.2f} {info['moneda']}**.
            
            👉 **Consejo de Trading:** Si el precio baja hasta **{var_95:.2f}**, deberías considerar vender para cortar pérdidas, ya que rompería el escenario estadístico normal.
            """)
            
            st.caption("⚠️ Nota: Esto es una simulación matemática basada en el pasado. No es asesoramiento financiero certificado.")

else:
    st.info("👈 Selecciona una empresa y presiona el botón para iniciar el análisis cuántico.")