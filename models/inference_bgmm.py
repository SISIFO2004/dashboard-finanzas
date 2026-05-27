import pandas as pd
from sklearn.mixture import BayesianGaussianMixture

def identify_bayesian_regimes(df_features: pd.DataFrame):
    df_ml = df_features.copy()
    features = ['Realized_Vol_20', 'Vol_of_Vol', 'Downside_Vol', 'Skewness_20', 'Kurtosis_20', 'DD_Velocity', 'Autocorr_1']
    if len(df_ml) < 50: return 0, {}, []
        
    X = df_ml[features]
    bgm = BayesianGaussianMixture(n_components=3, covariance_type='full', weight_concentration_prior_type='dirichlet_process', random_state=42)
    df_ml['Cluster'] = bgm.fit_predict(X)
    
    cluster_means = df_ml.groupby('Cluster')[['Realized_Vol_20', 'Returns', 'DD_Velocity', 'Kurtosis_20']].mean()
    norm_means = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min() + 1e-8)
    stress_scores = norm_means['Realized_Vol_20'] - norm_means['Returns'] - norm_means['DD_Velocity'] + norm_means['Kurtosis_20']
    
    stress_cluster = stress_scores.idxmax()
    bull_cluster = stress_scores.idxmin()
    
    def map_regime(r):
        if r == stress_cluster: return -1
        elif r == bull_cluster: return 1
        return 0
        
    df_ml['Mapped_Regime'] = df_ml['Cluster'].apply(map_regime)
    return df_ml['Mapped_Regime'].iloc[-1], features, df_ml['Mapped_Regime'].values
