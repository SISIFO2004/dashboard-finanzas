import numpy as np

def calculate_model_diagnostics(returns_array, mapped_regimes_history, apf_states):
    regime_changes = np.diff(mapped_regimes_history) != 0
    num_changes = np.sum(regime_changes)
    persistence_days = len(mapped_regimes_history) / (num_changes + 1) if num_changes > 0 else len(mapped_regimes_history)
    
    residuals = returns_array[-len(apf_states):] - apf_states
    dw_stat = np.sum(np.diff(residuals)**2) / np.sum(residuals**2) if np.sum(residuals**2) != 0 else 0
    
    var_95_hist = np.percentile(returns_array, 5)
    exceedance_ratio = np.mean(returns_array < var_95_hist)
    
    return persistence_days, dw_stat, exceedance_ratio

def calculate_risk_metrics_phase1(S0, paths, conf_level):
    var_price = np.percentile(np.amin(paths, axis=0), 100 - conf_level)
    prob_pos = np.mean(paths[-1, :] > S0)
    median_price = np.median(paths[-1, :])
    tp_price = np.percentile(paths[-1, :], 90)
    return prob_pos, var_price, median_price, tp_price

def generate_directive(prob_pos, current_regime):
    """Traduce la probabilidad y el régimen a una directriz humana"""
    if prob_pos > 0.62 and current_regime != -1:
        return "COMPRA FUERTE", "Condiciones favorables. Tendencia estructural validada."
    elif prob_pos < 0.38 or current_regime == -1:
        return "LIQUIDACIÓN PREVENTIVA", "Estrés estructural detectado. Cierre o protección mandatoria."
    else:
        return "RETENCIÓN / LATERAL", "Entorno de riesgo simétrico. Mantener exposición actual."
