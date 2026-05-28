import pandas as pd
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

def identify_bayesian_regimes(df_features: pd.DataFrame):
    df_ml = df_features.copy()
    # Variables espectrales generadas en la Fase 1
    features = ['Realized_Vol_20', 'Vol_of_Vol', 'Downside_Vol', 'Skewness_20', 'Kurtosis_20', 'DD_Velocity', 'Frac_Price', 'Autocorr_Frac']
    
    if len(df_ml) < 50:
        return 0, features, np.zeros(len(df_ml)), {}, np.zeros((3,3))
        
    X = df_ml[features].values
    
    # 1. NORMALIZACIÓN DEL ESPACIO LATENTE (Crítico para evitar colapso dimensional)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. INFERENCIA VARIACIONAL (Maximización ELBO con Dirichlet Process)
    vi_model = BayesianGaussianMixture(
        n_components=3,
        covariance_type='full',
        weight_concentration_prior_type='dirichlet_process',
        max_iter=1000,
        tol=1e-4,
        random_state=42
    )
    df_ml['Cluster'] = vi_model.fit_predict(X_scaled)
    
    # 3. AUTO-ETIQUETADO MATEMÁTICO
    cluster_means = df_ml.groupby('Cluster')[['Realized_Vol_20', 'DD_Velocity', 'Kurtosis_20']].mean()
    norm_means = (cluster_means - cluster_means.mean()) / (cluster_means.std() + 1e-8)
    stress_scores = norm_means['Realized_Vol_20'] - norm_means['DD_Velocity'] + norm_means['Kurtosis_20']
    
    stress_cluster = stress_scores.idxmax()
    bull_cluster = stress_scores.idxmin()
    
    def map_regime(r):
        if r == stress_cluster: return -1
        elif r == bull_cluster: return 1
        return 0
        
    df_ml['Mapped_Regime'] = df_ml['Cluster'].apply(map_regime)
    regimes = df_ml['Mapped_Regime'].values
    current_regime = regimes[-1]
    
    # 4. APRENDIZAJE MARKOVIANO: Matriz de Transición Empírica (Inercia temporal)
    reg_series = pd.Series(regimes)
    transition_counts = pd.crosstab(reg_series.shift(), reg_series)
    transition_matrix = transition_counts.div(transition_counts.sum(axis=1), axis=0).fillna(0).to_dict()
    
    # 5. AUTO-CALIBRACIÓN DE PARÁMETROS (Meta-Learning para la simulación)
    calibrated_params = {}
    for reg in [-1, 0, 1]:
        mask = df_ml['Mapped_Regime'] == reg
        if mask.sum() > 5:
            reg_returns = df_ml.loc[mask, 'Returns']
            calibrated_params[reg] = {
                'mu_regime': reg_returns.mean() * 252,
                'sigma_regime': reg_returns.std() * np.sqrt(252),
                # Proxy de Hawkes (Frecuencia de colas pesadas empíricas)
                'lambda_j': (reg_returns < reg_returns.mean() - 2*reg_returns.std()).sum() / (len(reg_returns) / 252 + 1e-8)
            }
        else:
            calibrated_params[reg] = {'mu_regime': 0.0, 'sigma_regime': 0.2, 'lambda_j': 1.0}
            
    return current_regime, features, regimes, calibrated_params, transition_matrix
