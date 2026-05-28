import pandas as pd
import numpy as np
from scipy.fft import fft, ifft
from statsmodels.tsa.stattools import adfuller
import warnings

# Suprimir advertencias matemáticas no críticas del solver
warnings.filterwarnings("ignore")

def get_weights_fft(d, size):
    """
    Genera los pesos exactos para la diferenciación fraccional
    hasta un tamaño 'size' sin truncar.
    """
    w = np.zeros(size)
    w[0] = 1.0
    for k in range(1, size):
        w[k] = -w[k-1] * (d - k + 1) / k
    return w

def frac_diff_fft(series, d):
    """
    Diferenciación Fraccional Espectral.
    Preserva la memoria infinita utilizando el teorema de convolución
    en el dominio de las frecuencias (Transformada Rápida de Fourier).
    """
    n = len(series)
    # Siguiente potencia de 2 para optimizar el procesamiento de la FFT (Zero-Padding)
    n_fft = 2**(int(np.ceil(np.log2(2 * n - 1))))
    
    # Cálculo de pesos exactos
    w = get_weights_fft(d, n)
    
    # Transformada de Fourier
    series_fft = fft(series.values, n_fft)
    w_fft = fft(w, n_fft)
    
    # Convolución en el espacio de Fourier y transformada inversa (iFFT)
    diff_fft = ifft(series_fft * w_fft, n_fft).real
    
    # Recuperar la dimensión temporal original
    df_diff = pd.Series(diff_fft[:n], index=series.index)
    return df_diff

def find_optimum_d(series, max_d=0.9, step=0.05, p_value_threshold=0.05):
    """
    Optimizador Meta-Learner (Maximum Likelihood/ADF).
    Encuentra el valor mínimo de 'd' que vuelve la serie estacionaria
    perdiendo la menor cantidad de memoria fractal posible.
    """
    for d in np.arange(0.1, max_d + step, step):
        diff_series = frac_diff_fft(series, d).dropna()
        if len(diff_series) > 10:
            # Test de Dickey-Fuller Aumentado para medir estacionariedad
            adf_stat = adfuller(diff_series, maxlag=1, regression='c', autolag=None)
            p_value = adf_stat[1]
            if p_value < p_value_threshold:
                return d
    return max_d

def engineer_structural_features(df: pd.DataFrame, trading_days: int) -> pd.DataFrame:
    df = df.copy()
    
    # 1. OPTIMIZADOR ML DE MEMORIA FRACCIONAL
    # El algoritmo decide el parámetro óptimo sin intervención humana
    optimal_d = find_optimum_d(df['Close'])
    df['Frac_Price'] = frac_diff_fft(df['Close'], d=optimal_d)
    
    # 2. Retornos Clásicos (Benchmark nominal)
    df['Returns'] = df['Close'].pct_change()
    
    # 3. Métricas de Riesgo Estructural (Ventana de 20 periodos)
    df['Realized_Vol_20'] = df['Returns'].rolling(window=20).std() * np.sqrt(trading_days)
    df['Vol_of_Vol'] = df['Realized_Vol_20'].rolling(window=20).std() 
    df['Downside_Vol'] = df['Returns'].where(df['Returns'] < 0, 0).rolling(window=20).std() * np.sqrt(trading_days)
    
    df['Skewness_20'] = df['Returns'].rolling(window=20).skew()
    df['Kurtosis_20'] = df['Returns'].rolling(window=20).kurt()
    
    # 4. Aceleración Fractal de Mercado
    cummax = df['Close'].cummax()
    df['Drawdown'] = (df['Close'] - cummax) / cummax
    df['DD_Velocity'] = df['Drawdown'].diff(3) 
    
    # 5. Autocorrelación Espectral
    df['Autocorr_Frac'] = df['Frac_Price'].rolling(window=20).apply(lambda x: x.autocorr(lag=1) if len(x)>1 else 0)
    
    return df.dropna()
