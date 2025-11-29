# Script path: functions/wsmoter.py

# This script is an adaptation of the 'wsmoter' code, which was developed by: 
# Camacho, L., Bacao, F. WSMOTER: a novel approach for imbalanced regression. 
# Appl Intell 54, 8789–8799 (2024). https://doi.org/10.1007/s10489-024-05608-6

# This script contains the implementation of the WSMOTER algorithm, which is a novel approach for imbalanced regression.
# The WSMOTER algorithm is an extension of the SMOTE algorithm, which is a well-known algorithm for imbalanced classification.


## load dependency - third party
import numpy as np
import pandas as pd
import collections
import warnings

from functions.wsmoter_over_sampling import WSMOTERdense
from functions.wsmoter_over_sampling import WSMOTERNCdense


def y_classes3(y):
    #print(y.describe())
    Q2 = y.describe()['50%']
    
    if y.min() == Q2:
        return np.where(np.array(y) <= Q2, 1, 2)
    else:
        return np.where(np.array(y) < Q2, 1, 2)
    

def apply_wsmoter_dense(Xtrain, ytrain, ratio, alpha, beta, cf):
        
    y_class = y_classes3(ytrain)
   
    count = collections.Counter(y_class)
    
    ss = {1: int(ratio * count[1]), 2: int(ratio * count[2])} 

    if cf == []:
        sm = WSMOTERdense(sampling_strategy = ss, random_state = 42, extreme = 'both', 
                          k_neighbors = 5, weights = 'dense', alpha = alpha, beta = beta)
    else:
        sm = WSMOTERNCdense(categorical_features = cf, sampling_strategy = ss, random_state = 42, extreme = 'both',
                            k_neighbors = 5, weights = 'dense', alpha = alpha, beta = beta)
    
    X_resampled, y_resampled, y_numeric = sm.fit_resample(Xtrain, y_class, ytrain)
    
    return (X_resampled,y_numeric)


def get_nominal_feature_indices(df: pd.DataFrame) -> np.ndarray:
    """
    Returns the indices of nominal (categorical) features in a DataFrame.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
    
    Returns:
        np.ndarray: An array containing the indices of nominal (categorical) columns.
    """
    nominal_indices = np.array([i for i, dtype in enumerate(df.dtypes) if dtype == 'object' or pd.api.types.is_categorical_dtype(dtype)])
    return nominal_indices


def apply_wsmoter_todataset(df, ratio, alpha, beta):
    
    X,y = df.iloc[:,:-1], df.iloc[:,-1]

    nominal_indices = get_nominal_feature_indices(df)
    nominal_indices = get_nominal_feature_indices(df).tolist()
    #print(nominal_indices)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        X_resampled, y_numeric = apply_wsmoter_dense(X, y, ratio, alpha, beta, nominal_indices)
            
    res = X_resampled.copy()
    res['y'] = y_numeric
        
    return res

def do_wsmoter(df, ratio, alpha, beta, drop_na_col=True, drop_na_row=True):
    """
    Applies the WSMOTER algorithm to the given DataFrame.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        ratio (float): The desired ratio of minority to majority samples.
        alpha (float): The alpha parameter for WSMOTER.
        beta (float): The beta parameter for WSMOTER.
    
    Returns:
        pd.DataFrame: The resampled DataFrame after applying WSMOTER.
    """

    df = df.copy()

    if drop_na_col:
        df = df.dropna(axis=1)
    
    if drop_na_row:
        df = df.dropna(axis=0)

    df_wsmoter = apply_wsmoter_todataset(df, ratio, alpha, beta)

    df_wsmoter.columns = df.columns

    return df_wsmoter