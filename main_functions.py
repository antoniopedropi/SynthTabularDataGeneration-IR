#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 20:41:43 2025

@author: antoniopedropi
"""

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
from functions import random_over_sampling_normal as ron
from functions import wercs
from functions import gaussian_noise as gn
from functions import smoter
from functions import wsmoter
from functions import david

from functions import aux_functions


## RELOAD INTERNAL PACKAGES/SCRIPTS ## -----------------------------------------------------------------------------------------


# After making changes to a module:

from importlib import reload  # Import the reload function
reload(david)  # Reload the module to reflect changes


## DATASET IMPORT AND ANALYSIS ##  ------------------------------------------------------------------------------------------

def get_dataset(name, y_label):
    
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
    
    # if df_missing_columns.shape[1] == 0:
    #     return df, df_numeric, None, df_missing_rows
    # elif df_missing_rows.shape[0] == 0:
    #     return df, df_numeric, df_missing_columns, None
    # elif (df_missing_columns.shape[1] == 0) and (df_missing_rows.shape[0] == 0):
    #     return df, df_numeric, None, None
    
    
    return df, df_numeric, df_missing_columns, df_missing_rows, df_numeric_missing_columns, df_numeric_missing_rows


#housing_df, housing_df_numeric, housing_df_missing_columns, housing_df_missing_rows, housing_df_numeric_missing_columns, housing_df_numeric_missing_rows = get_dataset('housing.csv', 'SalePrice')


def get_stats(df, attribute_label, dataset_name, showBoxplot = True):
    
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
        
        housing_stats = {
            'min': min(df[attribute_label]),                                               # Minimum
            'whislo': adjBoxplot.adjusted_boxplot_stats(df[attribute_label])['stats'][0],  # Lower whisker
            'q1': adjBoxplot.adjusted_boxplot_stats(df[attribute_label])['stats'][1],      # First quartile (25%)
            'med': adjBoxplot.adjusted_boxplot_stats(df[attribute_label])['stats'][2],     # Median (50%)
            'q3': adjBoxplot.adjusted_boxplot_stats(df[attribute_label])['stats'][3],      # Third quartile (75%)
            'whishi': adjBoxplot.adjusted_boxplot_stats(df[attribute_label])['stats'][4],  # Upper whisker
            'max': max(df[attribute_label]),                                               # Maximum
            'outliers': adjBoxplot.adjusted_boxplot_stats(df[attribute_label])['xtrms']    # Outliers
        }
        
    return housing_stats


def get_relevance_function(df, y_label, dataset_name, relevance_focus, plotFunction = False):
    
    control_points = relevance_function_ctrl_pts.phi_ctrl_pts(y=df[y_label], xtrm_type=relevance_focus)
    
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
