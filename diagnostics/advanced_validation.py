import numpy as np
import pandas as pd
from scipy.stats import chi2, norm, rankdata
import statsmodels.api as sm

def var_coverage_tests(returns: np.ndarray, var_forecasts: np.ndarray, alpha: float = 0.05):
    """Suite de Basilea: Kupiec (Incondicional) y Christoffersen (Condicional)."""
    violations = (returns < var_forecasts).astype(int)
    N = len(violations)
    x = np.sum(violations)
    eps = 1e-10  
    
    p_hat = np.clip(x / N, eps, 1 - eps)
    
    # Kupiec (LR_uc) en Log-Space
    log_L_null_uc = (N - x) * np.log(1 - alpha) + x * np.log(alpha)
    log_L_alt_uc = (N - x) * np.log(1 - p_hat) + x * np.log(p_hat)
    lr_uc = -2 * (log_L_null_uc - log_L_alt_uc)
    p_val_uc = 1 - chi2.cdf(lr_uc, df=1)
    
    # Christoffersen (LR_ind)
    T00 = T01 = T10 = T11 = 0
    for i in range(1, N):
        if violations[i-1] == 0 and violations[i] == 0: T00 += 1
        elif violations[i-1] == 0 and violations[i] == 1: T01 += 1
        elif violations[i-1] == 1 and violations[i] == 0: T10 += 1
        elif violations[i-1] == 1 and violations[i] == 1: T11 += 1
        
    T0 = T00 + T01
    T1 = T10 + T11
    T = T0 + T1
    
    pi_0 = np.clip(T01 / (T0 + eps), eps, 1 - eps)
    pi_1 = np.clip(T11 / (T1 + eps), eps, 1 - eps)
    pi_m = np.clip((T01 + T11) / (T + eps), eps, 1 - eps)
    
    log_L_null_ind = (T00 + T10) * np.log(1 - pi_m) + (T01 + T11) * np.log(pi_m)
    log_L_alt_ind = (T00 * np.log(1 - pi_0) + T01 * np.log(pi_0) + 
                     T10 * np.log(1 - pi_1) + T11 * np.log(pi_1))
    
    lr_ind = -2 * (log_L_null_ind - log_L_alt_ind)
    
    if T11 == 0 and T01 == 0:
        lr_ind, p_val_ind = 0.0, 1.0
    else:
        p_val_ind = 1 - chi2.cdf(lr_ind, df=1)
        
    # Cobertura Condicional Conjunta (LR_cc)
    lr_cc = lr_uc + lr_ind
    p_val_cc = 1 - chi2.cdf(lr_cc, df=2)
    
    return {"Violations": x, "Expected": N * alpha, "p_val_uc": p_val_uc, "p_val_ind": p_val_ind, "p_val_cc": p_val_cc}

def calculate_pit_returns(real_returns: np.ndarray, simulated_returns: np.ndarray):
    """Probability Integral Transform usando ranking empírico sobre retornos."""
    N_days = len(real_returns)
    pit_values = np.zeros(N_days)
    for t in range(N_days):
        combined = np.append(simulated_returns[t, :], real_returns[t])
        rank_real = rankdata(combined)[-1]
        pit_values[t] = rank_real / (len(combined) + 1)
    return pit_values

def berkowitz_tail_test(pit_values: np.ndarray):
    """Test de Normalidad y Autocorrelación de Berkowitz en dominio continuo."""
    eps = 1e-10
    pit_clean = np.clip(pit_values, eps, 1 - eps)
    z_t = norm.ppf(pit_clean)
    
    y = z_t[1:]
    X = sm.add_constant(z_t[:-1])
    try:
        model = sm.OLS(y, X).fit() # Proxy robusto para AR(1)
        mu, rho = model.params
        sigma = np.sqrt(model.mse_resid)
        
        log_L_null = -0.5 * np.sum(z_t[1:]**2 + np.log(2 * np.pi))
        residuals = y - (mu + rho * z_t[:-1])
        log_L_alt = -0.5 * np.sum((residuals / sigma)**2 + np.log(2 * np.pi * sigma**2))
        
        lr_berkowitz = -2 * (log_L_null - log_L_alt)
        p_val_berk = 1 - chi2.cdf(lr_berkowitz, df=3)
        return {"p_val_berk": p_val_berk, "mu": mu, "sigma": sigma}
    except Exception:
        return {"p_val_berk": np.nan, "mu": np.nan, "sigma": np.nan}

def acerbi_szekely_es_test(real_returns: np.ndarray, var_forecasts: np.ndarray, es_forecasts: np.ndarray):
    """Severidad de la cola (Expected Shortfall). Devuelve Z-score de desajuste."""
    violations = real_returns < var_forecasts
    if np.sum(violations) == 0: return np.nan
    tail_returns = real_returns[violations]
    tail_es = es_forecasts[violations]
    z_scores = (tail_returns / tail_es) - 1
    return np.mean(z_scores)
