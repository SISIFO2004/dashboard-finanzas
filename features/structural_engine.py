import pandas as pd
import numpy as np
from scipy.stats import entropy

def engineer_structural_features(df: pd.DataFrame, trading_days: int) -> pd.DataFrame:
    df = df.copy()
    df['Returns'] = df['Close'].pct_change()
    
    df['Realized_Vol_20'] = df['Returns'].rolling(window=20).std() * np.sqrt(trading_days)
    df['Vol_of_Vol'] = df['Realized_Vol_20'].rolling(window=20).std() 
    df['Downside_Vol'] = df['Returns'].where(df['Returns'] < 0, 0).rolling(window=20).std() * np.sqrt(trading_days)
    
    df['Skewness_20'] = df['Returns'].rolling(window=20).skew()
    df['Kurtosis_20'] = df['Returns'].rolling(window=20).kurt()
    
    cummax = df['Close'].cummax()
    df['Drawdown'] = (df['Close'] - cummax) / cummax
    df['DD_Velocity'] = df['Drawdown'].diff(3) 
    df['Autocorr_1'] = df['Returns'].rolling(window=20).apply(lambda x: x.autocorr(lag=1) if len(x)>1 else 0)
    
    return df.dropna()
