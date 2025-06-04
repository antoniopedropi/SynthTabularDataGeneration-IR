# Script path: automated_script_datasets_final.py

# Script that automates the processing of multiple datasets, applies various oversampling strategies, trains regression models, and evaluates their performance.


## SET CURRENT DIRECTORY ## --------------------------------------------------------------------------------------------------

import os
print(os.getcwd())  # Shows the current working directory
print(os.listdir())  # Checks what files are listed in the directory
os.chdir('/Users/antoniopedropi/Library/Mobile Documents/com~apple~CloudDocs/António/Mestrado MSI/2º Ano/Dissertação/Practical Implementation')


## LOAD EXTERNAL PACKAGES ## -------------------------------------------------------------------------------------------------

import importlib 
import numpy as np
import sklearn
import matplotlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels
import collections
import warnings
import itertools
import joblib
import json
import time
import csv
import xgboost
import multiprocessing as mp

from glob import glob
from joblib import Parallel, delayed
from itertools import product

from datetime import datetime

from dataclasses import dataclass
from typing import Dict

from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, GridSearchCV, RepeatedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer

from xgboost import XGBRegressor

from math import sqrt

from sklearn.ensemble import HistGradientBoostingClassifier

from catboost import CatBoostRegressor

from knnor_reg import data_augment


## LOAD INTERNAL PACKAGES/SCRIPTS ## -----------------------------------------------------------------------------------------

from functions import adjBoxplot
from functions import relevance_function_ctrl_pts
from functions import relevance_function_ctrl_pts_normal
from functions import relevance_function
from functions import smogn
from functions import random_under_sampling as ru
from functions import random_over_sampling as ro
from functions import random_over_sampling_normal as ron
from functions import wercs
from functions import gaussian_noise as gn
from functions import smoter
from functions import wsmoter
from functions import david
from functions import cart
from functions import regression_metrics as rm

from functions import aux_functions
from functions import main_functions


## RELOAD INTERNAL PACKAGES/SCRIPTS ## -----------------------------------------------------------------------------------------

# After making changes to a module:

from importlib import reload  # Import the reload function
reload(wercs)  # Reload the module to reflect changes


## RANDOM STATE INITIALISATION FOR REPRODUCIBILITY ## ------------------------------------------------------------------------------------------------------    

seed = 4 # Set the random seed for Python's built-in random module
np.random.seed(seed)  # Set the random seed for NumPy


## DATASET IMPORT AND ANALYSIS ##  ------------------------------------------------------------------------------------------

def load_datasets_info(dataset_folder_path):
    """
    Loads datasets information: (dataset_name, target_variable) tuples.
    
    Parameters:
    - dataset_folder_path: path to the directory where all the datasets are stored

    Returns:
    - datasets_info: list of (dataset_name, target_variable) tuples for each dataset
    """
    
    datasets_info = []
    
    for filename in os.listdir(dataset_folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(dataset_folder_path, filename)
            try:
                df = pd.read_csv(file_path, nrows=0)
                target_variable = df.columns[0]
                datasets_info.append((os.path.splitext(filename)[0], target_variable))
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                
    print("Datasets loaded:")
    for dataset_name, target_variable in datasets_info:
        print(f"  - Dataset: {dataset_name}, Target Variable: {target_variable}")
        
    return datasets_info


def process_datasets(datasets_info, dataset_folder_path, main_functions):
    """
    Process each dataset: load data, calculate stats, detect relevance focus.
    
    Parameters:
    - datasets_info: list of (dataset_name, target_variable)
    - dataset_folder_path: path to the directory where all the datasets are stored
    - main_functions: functions from another script which aid in the processing of the datasets

    Returns:
    - datasets: dictionary with all the datasets
    - datasets_numeric: dictionary with numeric features only of all the datasets
    - datasets_missing_columns: dictionary with all the datasets without columns with missing values
    - datasets_missing_rows: dictionary with all the datasets without rows with missing values
    - datasets_numeric_missing_columns: dictionary with numeric features only of all the datasets without columns with missing values
    - datasets_numeric_missing_rows: dictionary with numeric features only of all the datasets without rows with missing values
    - datasets_stats: dictionary with location statistics (lower whisker, minimum, first quartile, median, third quartile, maximum, higher whisker, outliers) of each dataset
    - datasets_relevance_focus: dictionary with the relvence focus (low, high, both) of each dataset
    """
    
    datasets = {}
    datasets_numeric = {}
    datasets_missing_columns = {}
    datasets_missing_rows = {}
    datasets_numeric_missing_columns = {}
    datasets_numeric_missing_rows = {}
    datasets_stats = {}
    datasets_relevance_focus = {}
    
    for dataset_name, target_variable in datasets_info:
        print(f"Processing {dataset_name}.csv ...")
        
        df, df_numeric, df_missing_columns, df_missing_rows, df_numeric_missing_columns, df_numeric_missing_rows = main_functions.get_dataset(
            f'{dataset_name}.csv', target_variable
        )
        
        stats = main_functions.get_stats(df, target_variable, dataset_name, showBoxplot=True)
        
        datasets[dataset_name] = df
        datasets_numeric[dataset_name] = df_numeric
        datasets_missing_columns[dataset_name] = df_missing_columns
        datasets_missing_rows[dataset_name] = df_missing_rows
        datasets_numeric_missing_columns[dataset_name] = df_numeric_missing_columns
        datasets_numeric_missing_rows[dataset_name] = df_numeric_missing_rows
        datasets_stats[dataset_name] = stats
        
        # Detect relevance focus
        high = any(outlier > stats['med'] for outlier in stats['outliers'])
        low = any(outlier < stats['med'] for outlier in stats['outliers'])
        
        if high and not low:
            datasets_relevance_focus[dataset_name] = "high"
        elif low and not high:
            datasets_relevance_focus[dataset_name] = "low"
        else:
            datasets_relevance_focus[dataset_name] = "both"
    
    return (datasets, datasets_numeric, datasets_missing_columns, datasets_missing_rows,
            datasets_numeric_missing_columns, datasets_numeric_missing_rows, datasets_stats, datasets_relevance_focus)


def check_missing_values(datasets_info, datasets, aux_functions):
    
    """
    Check and store missing values per dataset.
    
    Parameters:
    - datasets_info: list of (dataset_name, target_variable)
    - datasets: dictionary with all the datasets
    - aux_functions: auxiliary functions from another script which aid in the processing of the datasets

    Returns:
    - missing_values_results: dictionary of total values missing and per column in each dataset

    """
    
    missing_values_results = {}
    
    for dataset_name, target_variable in datasets_info:
        dataset = datasets[dataset_name]
        total_missing, summary = aux_functions.check_missing_values(dataset, dataset_name)
        
        missing_values_results[dataset_name] = {
            "total_missing": total_missing,
            "summary": summary
        }
        
    return missing_values_results


# Set dataset folder path
dataset_folder_path = '/Users/antoniopedropi/Library/Mobile Documents/com~apple~CloudDocs/António/Mestrado MSI/2º Ano/Dissertação/Practical Implementation/datasets'

# 1. Load dataset names and target variables
datasets_info = load_datasets_info(dataset_folder_path)

# 2. Process datasets (load dataframes, get stats, determine relevance focus)
(datasets, _ , _ , _ , _ , _ , datasets_stats, datasets_relevance_focus) = process_datasets(datasets_info, dataset_folder_path, main_functions)

# 3. Check for missing values
missing_values_results = check_missing_values(datasets_info, datasets, aux_functions)


## RELEVANCE FUNCTION ## ----------------------------------------------------------------------------------------------------

def compute_phi_points(datasets_info, datasets, datasets_relevance_focus, plot_function=True):
    """
    Computes phi points for each dataset based on the relevance function.
    Parameters:
    - datasets_info: list of (dataset_name, target_variable) tuples
    - datasets: dictionary with all the datasets
    - datasets_relevance_focus: dictionary with relevance focus for each dataset
    - plot_function: boolean to indicate if plots should be generated
    Returns:
    - phi_points_results: dictionary with phi points for each dataset
    """
    
    phi_points_results = {}
    
    for dataset_name, target_variable in datasets_info:
        df = datasets[dataset_name]
        relevance_focus = datasets_relevance_focus[dataset_name]
        
        phi_points = main_functions.get_relevance_function(
            df, 
            target_variable, 
            dataset_name, 
            relevance_focus, 
            plotFunction=plot_function
        )
        
        phi_points_results[dataset_name] = phi_points
        
    return phi_points_results

phi_points_results = compute_phi_points(datasets_info, datasets, datasets_relevance_focus)


def augment_train(base_model, param_dict, strategy, X, y, df, c, dataset_name, target_variable, relevance_focus):
    """
    Augments the training data using the specified oversampling strategy and trains the model.
    Parameters:
    - base_model: the base regression model to be trained
    - param_dict: dictionary of hyperparameters for the model
    - strategy: oversampling strategy to be used
    - X: features of the training dataset
    - y: target variable of the training dataset
    - df: original training dataset
    - c: parameters for the oversampling strategy
    - dataset_name: name of the dataset
    - target_variable: name of the target variable
    - relevance_focus: relevance focus of the dataset (low, high, both)
    Returns:
    - model: trained regression model
    """
    
    #print(f"Augmenting and training model for dataset: {dataset_name}, strategy: {strategy}, parameters: {c}")
    
    # Perform data augmentation based on the specified strategy
    balanced_df = augmentation(df, strategy, c, target_variable, relevance_focus)

    # Separate features and target variable
    X = balanced_df.drop(columns = target_variable, axis=1)
    y = balanced_df[target_variable]
    
    # Do model training
    model = clone(base_model).set_params(**param_dict)
    model.fit(X, y)

    return model
     

def augmentation(df, strategy, c, target_variable, relevance_focus):
    """
    Applies the specified oversampling strategy to the dataset.
    Parameters:
    - df: original dataset as a pandas DataFrame
    - strategy: oversampling strategy to be applied
    - c: parameters for the oversampling strategy
    - target_variable: name of the target variable in the dataset
    - relevance_focus: relevance focus of the dataset (low, high, both)
    Returns:
    - balanced_data: DataFrame with the augmented dataset
    """

    if strategy == "RU":
      balanced_data = ru.random_under(
            data = df,                          ## pandas dataframe
            y = target_variable,                ## string ('header name')
            samp_method=c[0],                   ## string ('balance' or 'extreme')
            drop_na_col=True,                   ## auto drop columns with nan's (bool)
            drop_na_row=True,                   ## auto drop rows with nan's (bool)
            replacement=True,                       ## sampling replacement (bool)
            manual_perc=False,                  ## boolean (True or False)
            perc_u=-1,                          ## positive real number (0 < R < 1) (only assinable if manual_perc == True)
            
            ## phi relevance arguments
            rel_thres=0.80,                     ## positive real number (0 < R < 1)
            rel_method='auto',                  ## string ('auto' or 'manual')
            rel_xtrm_type=relevance_focus)       ## string ('low' or 'both' or 'high'))
      
    elif strategy == "RO":
      balanced_data = ro.ro(
          data = df,                          ## pandas dataframe
          y = target_variable,                ## string ('header name')
          samp_method=c[0],                   ## string ('balance' or 'extreme')
          drop_na_col=True,                   ## auto drop columns with nan's (bool)
          drop_na_row=True,                   ## auto drop rows with nan's (bool)
          replace=True,                       ## sampling replacement (bool)
          manual_perc=False,                  ## boolean (True or False)
          perc_o=-1,                          ## positive real number (0 < R < 1) (only assinable if manual_perc == True)
          
          ## phi relevance arguments
          rel_thres=0.80,                     ## positive real number (0 < R < 1)
          rel_method='auto',                  ## string ('auto' or 'manual')
          rel_xtrm_type=relevance_focus)       ## string ('low' or 'both' or 'high'))
        
    elif strategy == "WERCS":
        
      ctrl_points = relevance_function_ctrl_pts.phi_ctrl_pts(df[target_variable])
      relevance = relevance_function.phi(df[target_variable], ctrl_points)
      
      balanced_data = wercs.do_wercs(
          df = df,                      ## pandas dataframe
          y_label = target_variable,    ## target column name
          phi_points = relevance,       ## relevance function values
          over = c[1],                  ## positive real number (0 < R < 1) (oversampling percentage)
          under = c[0],                 ## positive real number (0 < R < 1) (undersampling percentage)
          seed = seed)                  ## seed for random sampling (pos int or None)

    elif strategy == "GN":
        balanced_data = gn.gn(
            data=df,                        ## pandas dataframe
            y=target_variable,              ## string ('header name')
            pert = c[1],                    ## positive real number (0 < R < 1)
            samp_method=c[0],               ## string ('balance' or 'extreme')
            under_samp = True,              ## under sampling (bool)
            drop_na_col = True,             ## auto drop columns with nan's (bool)
            drop_na_row = True,             ## auto drop rows with nan's (bool)
            replace = False,                ## sampling replacement (bool)
            manual_perc = False,            ## boolean (True or False)
            perc_u = -1,                    ## positive real number (0 < R < 1) (only assinable if manual_perc == True)
            perc_o = -1,                    ## positive real number (0 < R < 1) (only assinable if manual_perc == True)
            
            ## phi relevance arguments
            rel_thres=0.80,                 ## positive real number (0 < R < 1)
            rel_method='auto',              ## string ('auto' or 'manual')
            rel_xtrm_type=relevance_focus   ## string ('low' or 'both' or 'high')
        )
        
    elif strategy == "SMOTER":
        balanced_data = smoter.smote(
            data=df,                            ## pandas dataframe
            y=target_variable,                  ## string ('header name')
            k = c[1],                           ## positive integer (k < n)
            samp_method=c[0],                   ## string ('balance' or 'extreme')
            drop_na_col=True,                   ## auto drop columns with nan's (bool)
            drop_na_row=True,                   ## auto drop rows with nan's (bool)
            
            ## phi relevance arguments
            rel_thres=0.80,                     ## positive real number (0 < R < 1)
            rel_method='auto',                  ## string ('auto' or 'manual')
            rel_xtrm_type=relevance_focus)      ## string ('low' or 'both' or 'high')
        
    elif strategy == "SMOGN":
        balanced_data = smogn.smoter(
            data=df,                        ## pandas dataframe
            y=target_variable,              ## string ('header name'),
            k = c[1],                       ## positive integer (k < n)
            pert = c[2],                    ## positive real number (0 < R < 1)
            samp_method=c[0],               ## string ('balance' or 'extreme')
            under_samp = True,              ## under sampling (bool)
            drop_na_col = True,             ## auto drop columns with nan's (bool)
            drop_na_row = True,             ## auto drop rows with nan's (bool)
            replace = False,                ## sampling replacement (bool)
            seed = seed,                    ## seed for random sampling (pos int or None)
            
            ## phi relevance arguments
            rel_thres=0.80,                 ## positive real number (0 < R < 1)
            rel_method='auto',              ## string ('auto' or 'manual')
            rel_xtrm_type=relevance_focus   ## string ('low' or 'both' or 'high')
        )
        
    elif strategy == "WSMOTER":
        df =  df.dropna(axis=1)
        balanced_data = wsmoter.do_wsmoter(
                ## main arguments
                df = df,                ## pandas dataframe
                ratio = c[1],           ## how many times we want to increase the number of samples 
                alpha = 1.0,            ## DenseWeight hyperparameter (α was defined as 1.0 because paper [Steininger M, Kobs K, Davidson P, Krause A, Hotho A (2021) Density-based weighting for imbalanced regression. Mach Learn 110:2187–2211. https://doi.org/10.1007/s10994-021-06023-5] stated that it improves the model)
                beta = c[0],            ## k is the shift on the y_sorted and k = 10 * beta
                drop_na_col = True,     ## auto drop columns with nan's (bool)
                drop_na_row = True      ## auto drop rows with nan's (bool)
            )
        
    elif strategy == "DAVID":
        try:
            balanced_data = david.david(
                ## main arguments
                data=df,                      # pandas dataframe
                y_label=target_variable,      # target column name
                alfa=c[0],                    # alpha parameter
                proportion=c[1],              # proportion of synthetic samples
                drop_na_col=True,             # drop columns with NaNs
                drop_na_row=True              # drop rows with NaNs
            )
        except Exception as e:
            print(f"⚠️ DAVID oversampling failed: {e}")
            balanced_data = df.copy()

    elif strategy == "KNNOR-REG":
        balanced_data = main_functions.do_knnor_reg(
            ## main arguments
            df,   ## pandas dataframe
        )
        
    elif strategy == "CART":
        synth = cart.RarityWeightedCARTSynthesizer(df, target_column = target_variable)
        balanced_data, _ = synth.generate_synthetic_data(
            sampling_proportion = c[3],
            density_method = c[1],
            alpha = c[0],
            noise = c[2]
        )
        
        balanced_data = balanced_data.drop('global_rarity', axis=1)
        balanced_data = balanced_data.drop('resample_count', axis=1)
        balanced_data = balanced_data.drop('origin', axis=1)
        
    elif strategy == "None":
        balanced_data = df

    return balanced_data
     

def repeatedKfold(df, X, y, dataset_name, target_variable, relevance_focus):
    """
    Applies repeated K-Fold cross-validation to the dataset, trains models with various oversampling strategies, and evaluates their performance.
    Parameters:
    - df: the original dataset
    - X: features of the dataset
    - y: target variable of the dataset
    - dataset_name: name of the dataset
    - target_variable: name of the target variable
    - relevance_focus: relevance focus of the dataset (low, high, both)
    Returns:
    - all_results_df: DataFrame containing all results from the experiments
    - summary_df: DataFrame summarizing the results
    """

    runs = RepeatedKFold(n_splits=5, n_repeats=2, random_state=4)
    
    print(runs)

    strategies = { # Dictionary of oversampling strategies and their parameters
        "None": {},
        "DAVID": {
            "alpha": [1, 2],
            "proportion": [0.9]
        },
        "CART": {
            "alpha": [1, 1.5, 2.0],
            "sampling_proportion": ['balance', 'extreme'],
            "density_method": ['kde_baseline', 'denseweight', 'relevance'],
            "noise": [True, False]
        },
        "RU": {
            "%u": ["balance", "extreme"]
        },
        "RO": {
            "%o": ["balance", "extreme"]
        },
        "WERCS": {
            "%u": [0.5, 0.75],
            "%o": [0.5, 0.75]
        },
        "GN": {
            "%u/%o": ["balance", "extreme"],
            "δ": [0.05, 0.1, 0.5]
        },
        "SMOTER": {
            "%u/%o": ["balance", "extreme"],
            "k": [5]
        },
        "SMOGN": {
            "%u/%o": ["balance", "extreme"],
            "δ": [0.05, 0.1, 0.5],
            "k": [5]
        },
        "WSMOTER": {
            "ratio": [1.5, 1.75],
            "beta": [1, 2]
        },
        "KNNOR-REG": {}  # KNNOR_REG has no tunable parameters
    }


    regressors = { # Dictionary of regression models and their hyperparameters
        'RF': {
            'model': RandomForestRegressor(),
            'param_grid': {
                'n_estimators': [100, 200],
                'max_features': ['sqrt', 'log2']
            }
        },
        'SVM': {
            'model': SVR(),
            'param_grid': {
                'kernel': ['rbf'],
                'C': [1, 10, 100],
                'epsilon': [0.01, 0.1]
            }
        },
        'XGB': {
            'model': XGBRegressor(objective='reg:squarederror', n_jobs=-1, verbosity=0),
            'param_grid': {
                'n_estimators': [100, 200],
                'max_depth': [3, 6]
            }
        }
    }
    
    # Declare a DataFrame to store all results
    all_results_df = pd.DataFrame(columns=['Dataset', 'Fold', 'Strategy', 'Parameter Combination', 'Model', 'Model Parameter Combination', 'MSE', 'RMSE', 'MAE', 'R2', 'WMSE', 'WRMSE', 'WMAE', 'WR2', 'SERA'])
    
    # Iterate through each strategy and its parameters
    for strategy in strategies:
        #print(strategy)
        params = strategies[strategy]
        keys = sorted(params)
        
        combinations = product(*(params[Name] for Name in keys))
      
        for c in combinations:
            
            print(strategy)
            print(c)
            
            for regressor_name, reg_dict in regressors.items():
                  print(regressor_name)
                  
                  base_model = reg_dict['model']
                  
                  param_grid = reg_dict['param_grid']
                  param_names = list(param_grid.keys())
                  param_values = list(param_grid.values())
                  
                  for param_combo in itertools.product(*param_values):
                      
                      param_dict = dict(zip(param_names, param_combo))
                      print(f"    ▶ Params: {param_dict}")
                      
                      for fold, (train_index, test_index) in enumerate(runs.split(X, y)):
                          print("run")
                          print("Fold:", fold)
                          X_train, X_test = X.iloc[train_index].reset_index(drop=True), X.iloc[test_index]
                          y_train, y_test = y.iloc[train_index].reset_index(drop=True), y.iloc[test_index]
                          df_train = df.iloc[train_index].reset_index(drop=True)
                          df_test = df.iloc[test_index].reset_index(drop=True)
                          
                          # Detect numeric and categorical columns
                          num_cols = X_train.select_dtypes(include='number').columns.tolist()
                          cat_cols = X_train.select_dtypes(exclude='number').columns.tolist()
    
                          # Preprocessing
                          preprocessor = ColumnTransformer(transformers=[
                            ('num', StandardScaler(), num_cols),
                            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
                            ])
    
                          # Train on training data and transform both train and test sets
                          X_train_proc = preprocessor.fit_transform(X_train)
                          X_test_proc = preprocessor.transform(X_test)
                          
                          # Attribute names after preprocessing
                          feature_names = preprocessor.get_feature_names_out()
                          
                          df_train_proc = pd.DataFrame(X_train_proc, columns = preprocessor.get_feature_names_out())
                          df_train_proc[target_variable] = y_train.values
                          
                          #print(df_train_proc.columns)
                          #print(df_train_proc.shape)
                          #print(X_test_proc.shape)
                          
                          X_test_proc_df = pd.DataFrame(X_test_proc, columns = preprocessor.get_feature_names_out())
                          
                          model = augment_train(base_model, param_dict, strategy, X_train_proc, y_train, df_train_proc, c, dataset_name, target_variable, relevance_focus)
                          y_pred = model.predict(X_test_proc_df)
                          
                          sera = mse = rmse = mae = r2 = None
                          wmse = wrmse = wmae = wr2 = None

                          sera = rm.sera(y_test, y_pred)
                          mse = mean_squared_error(y_test, y_pred)
                          rmse = sqrt(mse)
                          mae = mean_absolute_error(y_test, y_pred)
                          r2 = r2_score(y_test, y_pred)
                          try:
                            wmse = rm.phi_weighted_mse(y_test, y_pred)
                            wrmse = rm.phi_weighted_root_mse(y_test, y_pred)
                            wmae = rm.phi_weighted_mae(y_test, y_pred)
                            wr2 = rm.phi_weighted_r2(y_test, y_pred)
                          except Exception as e:
                            print(f"[Warning] WMSE/RMSE error: {e}")
                          
                          print("sera", sera)
                          
                          new_row = pd.DataFrame({"Dataset": [dataset_name], "Fold": [fold], "Strategy": [strategy], "Parameter Combination": [c], "Model": [regressor_name], "Model Parameter Combination": [str(param_dict)], "MSE": [mse], "RMSE": [rmse], "MAE": [mae], "R2": [r2], 'WMSE': [wmse], 'WRMSE': [wrmse], 'WMAE': [wmae], 'WR2': [wr2], "SERA": [sera]})
                          all_results_df = pd.concat([all_results_df, new_row], ignore_index=True)
                      
                      
    summary_df = (
        all_results_df
        .groupby(['Dataset', 'Strategy', 'Parameter Combination', 'Model', 'Model Parameter Combination'])[
            ['MSE', 'RMSE', 'MAE', 'R2', 'WMSE', 'WRMSE', 'WMAE', 'WR2', 'SERA']
        ]
        .agg(['mean', 'std'])
        .reset_index()
    )
      
    # Optional: flatten MultiIndex column names
    summary_df.columns = ['_'.join(col).strip('_') for col in summary_df.columns.values]
                    
    return all_results_df, summary_df


# Initialize lists to store results for each dataset         
all_results_list = []
summary_list = []
                
for dataset_name, target_variable in datasets_info:
    """ 
    Process each dataset and apply repeated K-Fold cross-validation with oversampling strategies. 
    Parameters:
    - dataset_name: name of the dataset
    - target_variable: name of the target variable
    Returns:
    - all_results_df: DataFrame containing all results from the experiments for the dataset
    - summary_df: DataFrame summarizing the results for the dataset
    """
    
    df = datasets[dataset_name]
    relevance_focus = datasets_relevance_focus[dataset_name]

    X = df.drop(columns = target_variable, axis = 1)
    y = df[target_variable]

    all_results_df, summary_df = repeatedKfold(df, X, y, dataset_name, target_variable, relevance_focus)
    
    all_results_list.append(all_results_df)
    summary_list.append(summary_df)
    
# Combine all results after the loop
final_all_results_df = pd.concat(all_results_list, ignore_index=True)
final_summary_df = pd.concat(summary_list, ignore_index=True)

# Export the combined results
final_all_results_df.to_csv("all_results_df.csv", index=False, sep=";")
final_summary_df.to_csv("summary_df.csv", index=False, sep=";")