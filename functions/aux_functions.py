# Script path: functions/aux_functions.py

# Description: This script contains auxiliary functions for data preprocessing and feature selection.


## load dependencies - third party
import pandas as pd
import numpy as np


# Function to get the indices of categorical features
def get_nominal_feature_indices(df: pd.DataFrame) -> np.ndarray | None:
    """
    Returns the indices of nominal (categorical) features in a DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        np.ndarray | None: An array containing the indices of nominal (categorical) columns,
                           or None if there are no nominal features.
    """
    nominal_indices = np.array([
        i for i, dtype in enumerate(df.dtypes)
        if dtype == 'object' or pd.api.types.is_categorical_dtype(dtype)
    ])

    if nominal_indices.size == 0:
        return None
    return nominal_indices


# Function to remove non-numeric features
def remove_non_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes non-numeric features (columns) from a given DataFrame.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
    
    Returns:
        pd.DataFrame: A new DataFrame containing only numeric columns.
    """
    return df.select_dtypes(include=['number'])


# Function to find categorical columns
def find_categorical_columns(df):
    """
    Identifies categorical columns in a DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    list: A list of column names that are categorical.
    """
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return cat_cols


# Function to remove missing values (by removing columns)
def remove_missing_columns(df):
    return df.dropna(axis = 1)


# Function to remove missing values (by removing rows)
def remove_missing_rows(df):
    return df.dropna(axis = 0)


# Function to count missing values
def check_missing_values(df, dataset_name):
    """
    Check for missing values in a pandas DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to check.
    
    Returns:
    tuple: (total_missing, missing_summary)
           total_missing (int): Total number of missing values in the dataset.
           missing_summary (pd.DataFrame or None): Summary of missing values per column.
    """
    total_missing = df.isnull().sum().sum()
    missing_summary = df.isnull().sum()
    missing_summary = missing_summary[missing_summary > 0].sort_values(ascending=False)
    
    if total_missing == 0:
        print(f"{dataset_name}: No missing values found.")
        return total_missing, None
    else:
        result = pd.DataFrame({
            'Missing Values': missing_summary,
            'Percentage (%)': (missing_summary / len(df)) * 100
        })
        print(f"{dataset_name}: Total missing values: {total_missing}")
        return total_missing, result