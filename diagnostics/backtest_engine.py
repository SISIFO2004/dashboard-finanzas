import numpy as np
import pandas as pd
from features.structural_engine import engineer_structural_features
from models.inference_bgmm import identify_bayesian_regimes

def run_walk_forward_backtest(df_hist, window_size=120):
    """
    Backtest de Ventana Deslizante. 
    Evalúa la precisión de la Directriz de la IA contra el retorno real futuro.
    """
    results = []
    # Iteramos desde el punto donde tenemos suficiente data para el primer entrenamiento
    for i in range(window_size, len(df_hist) - 1):
        # 1. Entrenamiento con información SOLO hasta el tiempo 'i'
        df_train = df_hist.iloc[:i]
        df_features = engineer_structural_features(df_train, trading_days=252)
        
        # 2. Inferencia del régimen actual
        regime, _, _, _, _ = identify_bayesian_regimes(df_features)
        
        # 3. La IA emite una directriz teórica
        # (Si regime == 1 es Compra, si -1 es Venta)
        directriz = 1 if regime == 1 else (-1 if regime == -1 else 0)
        
        # 4. Verificamos el retorno real del día siguiente (i+1)
        retorno_real = (df_hist['Close'].iloc[i+1] / df_hist['Close'].iloc[i]) - 1
        
        results.append({
            'Date': df_hist.index[i],
            'Signal': directriz,
            'Real_Return': retorno_real,
            'Pnl': directriz * retorno_real
        })
        
    return pd.DataFrame(results).set_index('Date')
