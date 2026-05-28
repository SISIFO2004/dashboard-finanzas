import numpy as np
from numba import njit, prange

# ==============================================================================
# 1. AUXILIARY PARTICLE FILTER (APF) - Extracción No Lineal
# ==============================================================================
@njit(fastmath=True)
def systematic_resampling(weights, num_particles):
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

@njit(fastmath=True)
def apply_auxiliary_particle_filter(returns_array, num_particles=1000):
    n = len(returns_array)
    filtered_states = np.zeros(n)
    particles = np.random.normal(0.0, 0.01, num_particles)
    sigma_q = 0.005  
    sigma_r = np.std(returns_array) if len(returns_array) > 1 else 0.02
    
    for t in range(n):
        y_t = returns_array[t]
        mu_i = particles  
        
        diff_aux = y_t - mu_i
        aux_weights = np.exp(-np.abs(diff_aux) / sigma_r) + 1e-12
        aux_weights /= np.sum(aux_weights)
        
        indices = systematic_resampling(aux_weights, num_particles)
        particles_resampled = particles[indices]
        
        particles_prop = particles_resampled + np.random.normal(0.0, sigma_q, num_particles)
        
        diff_actual = y_t - particles_prop
        actual_likelihood = np.exp(-np.abs(diff_actual) / sigma_r)
        
        diff_aux_resampled = y_t - mu_i[indices]
        aux_likelihood = np.exp(-np.abs(diff_aux_resampled) / sigma_r)
        
        actual_weights = actual_likelihood / (aux_likelihood + 1e-12) + 1e-12
        actual_weights /= np.sum(actual_weights)
        
        final_indices = systematic_resampling(actual_weights, num_particles)
        particles = particles_prop[final_indices]
        filtered_states[t] = np.mean(particles)
        
    return filtered_states

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
# 2. MOTOR DEL CAOS: HAWKES EXACTO (NO-MARKOVIANO) + HESTON 
# ==============================================================================
@njit(fastmath=True, nogil=True, parallel=True)
def _jit_montecarlo_loop_exact(price_paths, v_t, mu_t, days, simulations, dt, kappa_v, long_term_sigma, vol_v, Z_vol, kappa_mu, long_term_mu, vol_mu, Z_mu, Z_price, base_lambda, mu_j, sigma_j):
    
    # Parámetros de la Ley de Omori-Utsu pura
    alpha = 0.8     # Frecuencia de replicación
    p = 1.2         # Exponente Power-Law (Típicamente entre 1.1 y 1.5 en crisis financieras)
    c = 0.1         # Factor de regularización para evitar divergencia en t=0
    max_jumps = 500 # Límite de memoria pre-asignado para que Numba pueda compilar
    
    # Matriz para almacenar los timestamps EXACTOS de cada salto en la historia
    jump_times = np.full((simulations, max_jumps), -1.0, dtype=np.float64)
    jump_counts = np.zeros(simulations, dtype=np.int32)
    
    for t in range(days):
        current_time = t * dt
        
        for s in prange(simulations):
            
            # 1. Cálculo de intensidad exacta iterando sobre toda la historia previa (O(N^2))
            lambda_t = base_lambda
            n_jumps = jump_counts[s]
            
            for j in range(n_jumps):
                time_diff = current_time - jump_times[s, j]
                # Ecuación Power-Law exacta sin atajos de estado
                lambda_t += alpha / ((c + time_diff) ** p)
            
            # 2. Generación del shock
            jump_count_now = np.random.poisson(lambda_t * dt)
            jump_return = 0.0
            
            if jump_count_now > 0:
                for _ in range(jump_count_now):
                    jump_return += np.random.normal(mu_j, sigma_j)
                    
                    # Grabar el evento en la memoria histórica si hay espacio
                    if jump_counts[s] < max_jumps:
                        jump_times[s, jump_counts[s]] = current_time
                        jump_counts[s] += 1
            
            # 3. Volatilidad Estocástica (Heston Euler Full Truncation)
            v_t_positive = max(v_t[s], 0.0)
            v_new = v_t[s] + kappa_v * (long_term_sigma - v_t_positive) * dt + vol_v * np.sqrt(v_t_positive * dt) * Z_vol[t, s]
            v_new = max(v_new, 1e-8)
            
            # 4. Deriva Estocástica
            mu_new = mu_t[s] + kappa_mu * (long_term_mu - mu_t[s]) * dt + vol_mu * np.sqrt(dt) * Z_mu[t, s]
            
            # 5. Compensador de Martingala Teórico exacto
            compensator = lambda_t * (np.exp(mu_j + 0.5 * sigma_j**2) - 1.0)
            
            # 6. Salto final del precio
            price_paths[t+1, s] = price_paths[t, s] * np.exp((mu_new - compensator - 0.5 * (v_new**2)) * dt + v_new * np.sqrt(dt) * Z_price[t, s] + jump_return)
            
            v_t[s] = v_new
            mu_t[s] = mu_new
            
    return price_paths

def run_montecarlo_advanced_stochastic(S0, current_mu, long_term_mu, current_sigma, long_term_sigma, days, simulations, lambda_j, mu_j, sigma_j, trading_days):
    dt = 1 / trading_days  
    price_paths = np.zeros((days + 1, simulations))
    price_paths[0] = S0
    
    Z_vol = np.random.standard_normal((days, simulations))
    Z_mu = np.random.standard_normal((days, simulations))
    Z_price = np.random.standard_t(df=4, size=(days, simulations)) * np.sqrt(0.5)
    
    return _jit_montecarlo_loop_exact(
        price_paths, 
        np.full(simulations, current_sigma), 
        np.full(simulations, current_mu), 
        days, simulations, dt, 
        5.0, long_term_sigma, 0.4, Z_vol, 
        2.0, long_term_mu, 0.1, Z_mu, 
        Z_price, 
        lambda_j, mu_j, sigma_j
    )
