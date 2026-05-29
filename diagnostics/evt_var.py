import numpy as np
from scipy.stats import genpareto

def calculate_evt_var_es(terminal_prices: np.ndarray, S0: float, alpha: float = 0.05, threshold_percentile: float = 10.0):
    """
    Extreme Value Theory (EVT) - Método Peaks Over Threshold (POT) aplicado a la distribución terminal.
    Retorna el Capital VaR y el Expected Shortfall ajustados por la Distribución Pareto Generalizada.
    """
    # 1. Convertir precios terminales a retornos logarítmicos continuos
    returns = np.log(terminal_prices / S0)
    
    # 2. Definir el dominio de pérdidas (convirtiendo caídas a valores positivos para la GPD)
    losses = -returns
    
    # 3. Extraer el umbral (u) del cuantil empírico
    u = np.percentile(losses, 100 - threshold_percentile)
    
    # 4. Aislar la cola (excesos sobre el umbral)
    excesses = losses[losses > u] - u
    
    N = len(losses)
    N_u = len(excesses)
    p_u = N_u / N
    
    # 5. Fallback Institucional si la cola empírica es demasiado escasa para MLE
    if N_u < 15:
        var_empirical = np.percentile(losses, (1 - alpha) * 100)
        es_empirical = np.mean(losses[losses >= var_empirical]) if np.any(losses >= var_empirical) else var_empirical
        
        var_price = S0 * np.exp(-var_empirical)
        es_price = S0 * np.exp(-es_empirical)
        return var_price, es_price, "Empirical Tail"
        
    # 6. Ajuste MLE de la Distribución Pareto Generalizada a los excesos empíricos
    try:
        # shape (xi), loc (fijado en 0), scale (beta)
        xi, loc, beta = genpareto.fit(excesses, floc=0)
        
        # Límite matemático estructural: Si xi >= 1, el Expected Shortfall es infinito
        if xi >= 1.0:
            xi = 0.99 
            
        # 7. Cuantil EVT-POT (Fórmula analítica de Inversa GPD)
        var_evt_log = u + (beta / xi) * (((alpha / p_u)**-xi) - 1)
        
        # 8. Expected Shortfall EVT
        es_evt_log = (var_evt_log + beta - xi * u) / (1 - xi)
        
        # 9. Traducción a Precios Nominales
        var_price_evt = S0 * np.exp(-var_evt_log)
        es_price_evt = S0 * np.exp(-es_evt_log)
        
        return var_price_evt, es_price_evt, f"GPD (xi: {xi:.3f})"
        
    except Exception:
        # Fallback numérico absoluto en caso de error de optimización MLE
        var_emp = np.percentile(losses, (1 - alpha) * 100)
        return S0 * np.exp(-var_emp), S0 * np.exp(-var_emp * 1.2), "Failed GPD Fit"
