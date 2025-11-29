# Script path: functions/dist_metrics.py
# Script that contains functions that:
#     - import datasets from the 'datasets' folder
#     - get basic dataset statistics
#     - get dataset statistics for a specific attribute
#     - get relevance function for a specific attribute
#     - perform data augmentation using the KNNOR_Reg method


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

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from knnor_reg import data_augment


## LOAD INTERNAL PACKAGES/SCRIPTS ## -----------------------------------------------------------------------------------------

from functions import adjBoxplot
from functions import relevance_function_ctrl_pts
from functions import relevance_function_ctrl_pts_normal
from functions import relevance_function
from functions import smogn
from functions import random_under_sampling as ru
from functions import random_over_sampling as ro
from functions import wercs
from functions import gaussian_noise as gn
from functions import smoter
from functions import wsmoter
from functions import david

from functions import aux_functions


## DATASET IMPORT AND ANALYSIS ##  ------------------------------------------------------------------------------------------

def get_dataset(name, y_label):
    """
    Imports a dataset from the 'datasets' folder, performs basic dataset statistics,
    and visualizes the distribution of the target variable.
    Parameters:
        name (str): The name of the dataset file.
        y_label (str): The label of the target variable in the dataset.
    Returns:
        df (DataFrame): The original dataset.
        df_numeric (DataFrame): The dataset with non-numeric features removed.
        df_missing_columns (DataFrame): The dataset with columns containing missing values removed.
        df_missing_rows (DataFrame): The dataset with rows containing missing values removed.
        df_numeric_missing_columns (DataFrame): The numeric dataset with columns containing missing values removed.
        df_numeric_missing_rows (DataFrame): The numeric dataset with rows containing missing values removed.
    """
    
    # Import dataset
    df = pd.read_csv('datasets/' + name)

    
    # Get basic dataset statistics
    #df.head()
    #df.shape
    #df.info()
    #df.describe()    
    

    # Density Plot
    plt.figure(figsize=(10, 5))
    sns.kdeplot(df[y_label], fill=True, color="skyblue", lw=2)
    
    # Add titles and labels
    plt.title("Distribution of Target Variable \"" + y_label + "\" in " + name[:-4] + " dataset")
    plt.xlabel("Target Value - " + y_label)
    plt.ylabel("Density")
    
    plt.show()
      
    # Dataset Cleaning and Manipulation
    df_numeric = aux_functions.remove_non_numeric_features(df)
    df_missing_columns = aux_functions.remove_missing_columns(df)
    df_missing_rows = aux_functions.remove_missing_rows(df)
    df_numeric_missing_columns = aux_functions.remove_missing_columns(df_numeric)
    df_numeric_missing_rows = aux_functions.remove_missing_rows(df_numeric)  
    
    return df, df_numeric, df_missing_columns, df_missing_rows, df_numeric_missing_columns, df_numeric_missing_rows


def get_stats(df, attribute_label, dataset_name, showBoxplot = True):
    """
    Generates an adjusted boxplot for a specified attribute in the dataset and returns basic statistics.
    Parameters:
        df (DataFrame): The dataset containing the attribute.
        attribute_label (str): The label of the attribute for which the boxplot is generated.
        dataset_name (str): The name of the dataset, used for plot titles.
        showBoxplot (bool): Whether to display the boxplot. Default is True.
    Returns:
        stats (dict): A dictionary containing basic statistics of the attribute, including:
            - 'min': Minimum value
            - 'whislo': Lower whisker value 
            - 'q1': First quartile (25%)
            - 'med': Median (50%)
            - 'q3': Third quartile (75%)
            - 'whishi': Upper whisker value
            - 'max': Maximum value
            - 'outliers': List of outliers
    """
    
    # Constructing the required format for matplotlib
    box_plot_data = {
        'whislo': adjBoxplot.adjusted_boxplot_stats(df[attribute_label])['stats'][0],  # Lower whisker
        'q1': adjBoxplot.adjusted_boxplot_stats(df[attribute_label])['stats'][1],      # First quartile (25%)
        'med': adjBoxplot.adjusted_boxplot_stats(df[attribute_label])['stats'][2],     # Median (50%)
        'q3': adjBoxplot.adjusted_boxplot_stats(df[attribute_label])['stats'][3],      # Third quartile (75%)
        'whishi': adjBoxplot.adjusted_boxplot_stats(df[attribute_label])['stats'][4],  # Upper whisker
        'fliers': adjBoxplot.adjusted_boxplot_stats(df[attribute_label])['xtrms']      # Outliers
    }
    
    if showBoxplot == True:
        
        # Create the figure
        fig, ax = plt.subplots(figsize=(8, 6))

        # Generate the box plot using bxp()
        box = ax.bxp([box_plot_data], showfliers=True, patch_artist=True)

        # Apply colors and styles
        for element in ['boxes', 'whiskers', 'caps', 'medians']:
            for line in box[element]:
                line.set(color='black', linewidth=1.5)  # Make all lines black

        for patch in box['boxes']:
            patch.set(facecolor='lightblue', edgecolor='black', linewidth=1.5)  # Light green fill

        # Customize outliers
        for flier in box['fliers']:
            flier.set(marker='o', color='black', markersize=6)
            
        # Remove the default x-axis label ("1")
        ax.set_xticks([])

        # Set labels and title
        ax.set_title(f'Boxplot - {dataset_name} dataset', fontsize=14, fontweight='bold')
        ax.set_xlabel(attribute_label, fontsize=12)
        ax.set_ylabel('Values', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)  # Light gray grid

        # Show plot
        plt.show()
        
        stats = {
            'min': min(df[attribute_label]),                                               # Minimum
            'whislo': adjBoxplot.adjusted_boxplot_stats(df[attribute_label])['stats'][0],  # Lower whisker
            'q1': adjBoxplot.adjusted_boxplot_stats(df[attribute_label])['stats'][1],      # First quartile (25%)
            'med': adjBoxplot.adjusted_boxplot_stats(df[attribute_label])['stats'][2],     # Median (50%)
            'q3': adjBoxplot.adjusted_boxplot_stats(df[attribute_label])['stats'][3],      # Third quartile (75%)
            'whishi': adjBoxplot.adjusted_boxplot_stats(df[attribute_label])['stats'][4],  # Upper whisker
            'max': max(df[attribute_label]),                                               # Maximum
            'outliers': adjBoxplot.adjusted_boxplot_stats(df[attribute_label])['xtrms']    # Outliers
        }
        
    return stats


def get_relevance_function(df, y_label, dataset_name, relevance_focus, plotFunction = False):
    """
    Computes the relevance function for a specified target variable in the dataset and optionally plots it.
    Parameters:
        df (DataFrame): The dataset containing the target variable.
        y_label (str): The label of the target variable for which the relevance function is computed.
        dataset_name (str): The name of the dataset, used for plot titles.
        relevance_focus (str): The type of extreme values to focus on ('high', 'low', or 'normal').
        plotFunction (bool): Whether to plot the relevance function. Default is False.
    Returns:
        phi_points (array): The computed relevance values for the target variable.
    """
    
    # Get the relevance function control points based on the focus type
    control_points = relevance_function_ctrl_pts.phi_ctrl_pts(y=df[y_label], xtrm_type=relevance_focus)
    
    # Calculate the relevance function using the control points
    phi_points = relevance_function.phi(df[y_label], control_points)
    
    if plotFunction == True:
        
        relevance_dict = {
        y_label: df[y_label],
        'RelevanceValues': phi_points
        }

        relevance_df = pd.DataFrame(relevance_dict)
        
        # Sorting by "SalePrice" in ascending order
        relevance_sorted = relevance_df.sort_values(by=y_label, ascending=True)
        
        
        # Plot y_label on X-axis and "Relevance" on Y-axis
        plt.figure(figsize=(8, 5))
        plt.plot(relevance_sorted[y_label], relevance_sorted['RelevanceValues'], marker='o', linestyle='-')  # Line plot
        plt.xlabel(y_label)
        plt.ylabel('Relevance')
        plt.title('Relevance Function - ' + str(dataset_name) + ' dataset')
        plt.grid(True)
        
        plt.show()
    
    return phi_points


def do_knnor_reg(df, bins = None, target_freq = None):
    """
    Performs data augmentation using the KNNOR_Reg method on a given dataset.
    Parameters:
        df (DataFrame): The dataset containing the features and target variable.
        bins (int or None): Number of bins to use for discretization. If None, no discretization is applied.
        target_freq (int or None): Target frequency for the augmented data. If None, uses the original frequency.
    Returns:
        df_knnor_reg (DataFrame): The augmented dataset with the same structure as the original dataset.
    """
    
    # Separate features and target variable
    X_numeric = df.iloc[:,:-1].values
    Y_numeric = df.iloc[:,-1].values

    # Initialize KNNOR_Reg
    knnor_reg = data_augment.KNNOR_Reg()
    # Perform data augmentation
    X_new_knnor_reg, y_new_knnor_reg = knnor_reg.fit_resample(X_numeric, Y_numeric, bins=bins, target_freq=target_freq)
    y_new_knnor_reg = y_new_knnor_reg.reshape(-1, 1)

    df_knnor_reg = pd.DataFrame(X_new_knnor_reg)
    df_knnor_reg['y'] = pd.DataFrame(y_new_knnor_reg)

    df_knnor_reg.columns = df.columns
    
    return df_knnor_reg
