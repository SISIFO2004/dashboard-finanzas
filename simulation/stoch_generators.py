import numpy as np
from numba import njit

def apply_kalman_filter_dynamic(returns_array, divisor_inercia):
    n, R = len(returns_array), np.var(returns_array)
    if R == 0: return np.full(n, returns_array[-1])
    Q = R / float(divisor_inercia) 
    x_hat, P = np.zeros(n), np.zeros(n)
    x_hat[0], P[0] = returns_array[0], 1.0
    for k in range(1, n):
        K = (P[k-1] + Q) / (P[k-1] + Q + R)
        x_hat[k] = x_hat[k-1] + K * (returns_array[k] - x_hat[k-1])
        P[k] = (1 - K) * (P[k-1] + Q)
    return x_hat

def apply_stochastic_volatility_filter(returns_array):
    y_star = np.log(returns_array**2 + 1e-8)
    h_hat, P_vol = np.zeros(len(y_star)), np.zeros(len(y_star))
    h_hat[0], P_vol[0] = np.mean(y_star) if len(y_star) > 0 else 0.0, 1.0
    for k in range(1, len(y_star)):
        K = (P_vol[k-1] + 0.1) / (P_vol[k-1] + 0.1 + 4.93)
        h_hat[k] = h_hat[k-1] + K * (y_star[k] - h_hat[k-1])
        P_vol[k] = (1 - K) * (P_vol[k-1] + 0.1)
    return np.clip(np.exp(h_hat / 2.0), 0.0001, 1.0)

@njit
def _jit_montecarlo_loop(price_paths, v_t, mu_t, days, simulations, dt, kappa_v, long_term_sigma, vol_v, Z_vol, kappa_mu, long_term_mu, vol_mu, Z_mu, Z_price, jump_returns):
    for t in range(days):
        for s in range(simulations):
            v_new = np.abs(v_t[s] + kappa_v * (long_term_sigma - v_t[s]) * dt + vol_v * np.sqrt(v_t[s] * dt) * Z_vol[t, s])
            mu_new = mu_t[s] + kappa_mu * (long_term_mu - mu_t[s]) * dt + vol_mu * np.sqrt(dt) * Z_mu[t, s]
            price_paths[t+1, s] = price_paths[t, s] * np.exp((mu_new - 0.5 * v_new**2) * dt + v_new * np.sqrt(dt) * Z_price[t, s] + jump_returns[t, s])
            v_t[s], mu_t[s] = v_new, mu_new
    return price_paths

def run_montecarlo_advanced_stochastic(S0, current_mu, long_term_mu, current_sigma, long_term_sigma, days, simulations, lambda_j, mu_j, sigma_j, trading_days):
    dt = 1 / trading_days  
    price_paths = np.zeros((days + 1, simulations))
    price_paths[0] = S0
    Z_vol = np.random.standard_normal((days, simulations))
    Z_mu = np.random.standard_normal((days, simulations))
    Z_price = np.random.standard_t(df=4, size=(days, simulations)) * np.sqrt(0.5)
    jumps = np.random.poisson(lambda_j * dt, (days, simulations)) * np.random.normal(mu_j, sigma_j, (days, simulations))
    return _jit_montecarlo_loop(price_paths, np.full(simulations, current_sigma), np.full(simulations, current_mu), days, simulations, dt, 5.0, long_term_sigma, 0.2, Z_vol, 2.0, long_term_mu, 0.1, Z_mu, Z_price, jumps)
