import numpy as np
from numba import njit, prange

# ==============================================================================
# 1. AUXILIARY PARTICLE FILTER (APF) - Extracción No Lineal
# ==============================================================================
@njit
def systematic_resampling(weights, num_particles):
    """
    Remuestreo sistemático de baja varianza. Evita el colapso de partículas
    clonando las que tienen alta probabilidad y matando el ruido.
    """
    positions = (np.arange(num_particles) + np.random.random()) / num_particles
    indices = np.zeros(num_particles, dtype=np.int32)
    cum_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < num_particles:
        if positions[i] < cum_sum[j]:
            indices[i] = j
            i += 1
        else:
            j += 1
    return indices

@njit
def apply_auxiliary_particle_filter(returns_array, num_particles=1000):
    """
    APF: Filtro de Partículas Auxiliar con verosimilitud de colas pesadas.
    Sobrevive a quiebres estructurales y cisnes negros sin deformar la señal.
    """
    n = len(returns_array)
    filtered_states = np.zeros(n)
    
    # Inicialización del "multiverso" de hipótesis
    particles = np.random.normal(0.0, 0.01, num_particles)
    
    # Parámetros de transición (drift) y observación (ruido)
    sigma_q = 0.005  # Volatilidad estructural de la tendencia
    sigma_r = np.std(returns_array) if len(returns_array) > 1 else 0.02
    
    for t in range(n):
        y_t = returns_array[t]
        
        # PASO 1: Look-ahead (Predicción auxiliar)
        mu_i = particles  # En un Random Walk, el estado esperado es el actual
        
        # PASO 2: Verosimilitud Auxiliar usando distribución Laplaciana (Heavy-Tail)
        diff_aux = y_t - mu_i
        aux_weights = np.exp(-np.abs(diff_aux) / sigma_r)
        aux_weights += 1e-12  # Estabilidad numérica
        aux_weights /= np.sum(aux_weights)
        
        # PASO 3: Primer Remuestreo (Pre-selección de sobrevivientes al shock)
        indices = systematic_resampling(aux_weights, num_particles)
        particles_resampled = particles[indices]
        
        # PASO 4: Propagación estocástica (Mutación)
        particles_prop = particles_resampled + np.random.normal(0.0, sigma_q, num_particles)
        
        # PASO 5: Actualización de pesos reales
        diff_actual = y_t - particles_prop
        actual_likelihood = np.exp(-np.abs(diff_actual) / sigma_r)
        
        diff_aux_resampled = y_t - mu_i[indices]
        aux_likelihood = np.exp(-np.abs(diff_aux_resampled) / sigma_r)
        
        actual_weights = actual_likelihood / (aux_likelihood + 1e-12)
        actual_weights += 1e-12
        actual_weights /= np.sum(actual_weights)
        
        # PASO 6: Segundo Remuestreo (Consolidación)
        final_indices = systematic_resampling(actual_weights, num_particles)
        particles = particles_prop[final_indices]
        
        # Extracción de la señal purificada (Esperanza matemática)
        filtered_states[t] = np.mean(particles)
        
    return filtered_states

# ==============================================================================
# 2. FILTRO DE VOLATILIDAD ESTOCÁSTICA
# ==============================================================================
def apply_stochastic_volatility_filter(returns_array):
    y_star = np.log(returns_array**2 + 1e-8)
    h_hat, P_vol = np.zeros(len(y_star)), np.zeros(len(y_star))
    h_hat[0], P_vol[0] = np.mean(y_star) if len(y_star) > 0 else 0.0, 1.0
    for k in range(1, len(y_star)):
        K = (P_vol[k-1] + 0.1) / (P_vol[k-1] + 0.1 + 4.93)
        h_hat[k] = h_hat[k-1] + K * (y_star[k] - h_hat[k-1])
        P_vol[k] = (1 - K) * (P_vol[k-1] + 0.1)
    return np.clip(np.exp(h_hat / 2.0), 0.0001, 1.0)

# ==============================================================================
# 3. GENERADOR DE ESCENARIOS (MONTE CARLO BASE JIT)
# ==============================================================================
@njit
def _jit_montecarlo_loop(price_paths, v_t, mu_t, days, simulations, dt, kappa_v, long_term_sigma, vol_v, Z_vol, kappa_mu, long_term_mu, vol_mu, Z_mu, Z_price, jump_returns):
    for t in range(days):
        for s in range(simulations):
            v_new = np.abs(v_t[s] + kappa_v * (long_term_sigma - v_t[s]) * dt + vol_v * np.sqrt(v_t[s] * dt) * Z_vol[t, s])
            mu_new = mu_t[s] + kappa_mu * (long_term_mu - mu_t[s]) * dt + vol_mu * np.sqrt(dt) * Z_mu[t, s]
            
            # Compensador de Martingala básico integrado en la deriva
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
    
    # Saltos de Poisson Clásicos (Serán reemplazados por Hawkes en la Fase 4)
    jumps = np.random.poisson(lambda_j * dt, (days, simulations)) * np.random.normal(mu_j, sigma_j, (days, simulations))
    
    return _jit_montecarlo_loop(price_paths, np.full(simulations, current_sigma), np.full(simulations, current_mu), days, simulations, dt, 5.0, long_term_sigma, 0.2, Z_vol, 2.0, long_term_mu, 0.1, Z_mu, Z_price, jumps)
