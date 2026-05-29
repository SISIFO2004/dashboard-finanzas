import numpy as np
import pandas as pd
from scipy.stats import chi2, norm, rankdata
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

def var_coverage_tests(returns: np.ndarray, var_forecasts: np.ndarray, alpha: float = 0.05):
    """Suite de Basilea en Log-Space: Kupiec (Incondicional) y Christoffersen (Condicional)."""
    violations = (returns < var_forecasts).astype(int)
    N = len(violations)
    x = np.sum(violations)
    
    if N == 0 or x == 0:
        return {"Violations": x, "Expected": N * alpha, "p_val_uc": np.nan, "p_val_ind": np.nan, "p_val_cc": np.nan}
        
    eps = 1e-10  
    p_hat = np.clip(x / N, eps, 1 - eps)
    
    log_L_null_uc = (N - x) * np.log(1 - alpha) + x * np.log(alpha)
    log_L_alt_uc = (N - x) * np.log(1 - p_hat) + x * np.log(p_hat)
    lr_uc = -2 * (log_L_null_uc - log_L_alt_uc)
    p_val_uc = 1 - chi2.cdf(lr_uc, df=1)
    
    T00 = T01 = T10 = T11 = 0
    for i in range(1, N):
        if violations[i-1] == 0 and violations[i] == 0: T00 += 1
        elif violations[i-1] == 0 and violations[i] == 1: T01 += 1
        elif violations[i-1] == 1 and violations[i] == 0: T10 += 1
        elif violations[i-1] == 1 and violations[i] == 1: T11 += 1
        
    T0 = T00 + T01
    T1 = T10 + T11
    T = T0 + T1
    
    if T11 == 0 and T01 == 0:
        lr_ind, p_val_ind = 0.0, 1.0
    else:
        pi_0 = np.clip(T01 / (T0 + eps), eps, 1 - eps)
        pi_1 = np.clip(T11 / (T1 + eps), eps, 1 - eps)
        pi_m = np.clip((T01 + T11) / (T + eps), eps, 1 - eps)
        
        log_L_null_ind = (T00 + T10) * np.log(1 - pi_m) + (T01 + T11) * np.log(pi_m)
        log_L_alt_ind = (T00 * np.log(1 - pi_0) + T01 * np.log(pi_0) + T10 * np.log(1 - pi_1) + T11 * np.log(pi_1))
        lr_ind = -2 * (log_L_null_ind - log_L_alt_ind)
        p_val_ind = 1 - chi2.cdf(lr_ind, df=1)
        
    lr_cc = lr_uc + lr_ind
    p_val_cc = 1 - chi2.cdf(lr_cc, df=2)
    
    return {"Violations": x, "Expected": N * alpha, "p_val_uc": p_val_uc, "p_val_ind": p_val_ind, "p_val_cc": p_val_cc}

def calculate_pit_returns(real_returns: np.ndarray, simulated_returns: np.ndarray, rng: np.random.Generator):
    """PIT Randomizado inyectando un RNG global explícito."""
    N_days = len(real_returns)
    pit_values = np.zeros(N_days)
    for t in range(N_days):
        sims = simulated_returns[t, :]
        real = real_returns[t]
        f_minus = np.mean(sims < real)
        f_eq = np.mean(sims == real)
        pit_values[t] = f_minus + rng.uniform(0, 1) * f_eq
    return pit_values

def pit_independence_test(pit_values: np.ndarray, lags: int = 5):
    """Test de Ljung-Box para descartar autocorrelación."""
    try:
        lb_results = acorr_ljungbox(pit_values, lags=[lags], return_df=True)
        return lb_results['lb_pvalue'].iloc[0]
    except Exception:
        return np.nan

def berkowitz_tail_test(pit_values: np.ndarray):
    """Test de Berkowitz usando Máxima Verosimilitud Exacta (SARIMAX sin MA)."""
    eps = 1e-10
    pit_clean = np.clip(pit_values, eps, 1 - eps)
    z_t = norm.ppf(pit_clean)
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            model = sm.tsa.SARIMAX(z_t, order=(1, 0, 0), trend='c').fit(disp=False)
            if any(issubclass(warn.category, (ConvergenceWarning, sm.tools.sm_exceptions.HessianInversionWarning)) for warn in w):
                return {"p_val_berk": np.nan, "status": "Failed Convergence"}
                
            log_L_null = -0.5 * np.sum(z_t**2 + np.log(2 * np.pi))
            log_L_alt = model.llf
            
            lr_berkowitz = max(-2 * (log_L_null - log_L_alt), 0.0)
            p_val = 1 - chi2.cdf(lr_berkowitz, df=3)
            return {"p_val_berk": p_val, "mu": model.params[0], "sigma": np.sqrt(model.params[2]), "status": "Converged"}
        except (np.linalg.LinAlgError, ValueError):
            return {"p_val_berk": np.nan, "status": "LinAlgError"}

def acerbi_szekely_es_test(real_returns: np.ndarray, var_forecasts: np.ndarray, es_forecasts: np.ndarray, alpha: float = 0.05):
    """Test formal Z_2 de Acerbi-Szekely para Expected Shortfall."""
    violations = real_returns < var_forecasts
    if np.sum(violations) == 0: 
        return np.nan
    
    indicator = violations.astype(float)
    eps = 1e-8
    es_clean = np.clip(es_forecasts, -np.inf, -eps)
    Z_AS = np.sum(((real_returns * indicator) / es_clean) - indicator) / (len(real_returns) * alpha)
    return Z_AS
