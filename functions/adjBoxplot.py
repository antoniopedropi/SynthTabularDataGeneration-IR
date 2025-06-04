# Script path: functions/adjBoxplot.py

# This script contains functions for calculating adjusted boxplot statistics.
# The adjusted boxplot statistics are calculated similar to R's adjboxStats function.
# The adjusted boxplot statistics are calculated using the medcouple function from the statsmodels package.
# Additionally, we also incorporated the box_plot_stats() function which calculates the box plot five-number summar from the 'SMOGN' Python package, which was developed by:
# Kunz, N. (2020). SMOGN: Synthetic Minority Over-Sampling Technique for Regression with Gaussian Noise (Version 0.1.2) [Software]. PyPI. Retrieved from https://pypi.org/project/smogn/.

# The 'SMOGN' Python package was developed based on the following paper: 
# Branco, P., Torgo, L., Ribeiro, R. (2017). SMOGN: A Pre-Processing Approach for Imbalanced Regression. Proceedings of Machine Learning Research, 74:36-50. http://proceedings.mlr.press/v74/branco17a/branco17a.pdf


## load dependencies - third party
import numpy as np
import pandas as pd
from statsmodels.stats.stattools import medcouple


## adjusted boxplot statistics function
def adjusted_boxplot_stats(y, k=1.5):

    """Compute the adjusted boxplot statistics similar to R's adjboxStats."""

    ## transform the values into a numpy array
    y = np.asarray(y)

    ## compute the adjusted boxplot statistics
    q1, q3 = np.percentile(y, [25, 75], method="midpoint")
    iqr = q3 - q1
    mc = medcouple(y)
    
    # Adjust fences based on MC (medcouple)
    if mc >= 0:
        lower_fence = q1 - k * np.exp(-4 * mc) * iqr
        upper_fence = q3 + k * np.exp(3 * mc) * iqr
    else:
        lower_fence = q1 - k * np.exp(-3 * mc) * iqr
        upper_fence = q3 + k * np.exp(4 * mc) * iqr

    ## compute the minimum and maximum values based on the fences  
    try:
        minimum = min(y[y >= lower_fence])
    except ValueError:
        minimum = np.min(y)

    try:
        maximum = max(y[y <= upper_fence])
    except ValueError:
        maximum = np.max(y)
    
    ## compute the median
    median_q2 = np.median(y)

    ## print the results for initial debugging
    # print("lower_fence: " + str(lower_fence) + 
    #   ", \n upper_fence: " + str(upper_fence) + 
    #   ", \n minimum: " + str(minimum) + 
    #   ", \n q1: " + str(q1) + 
    #   ", \n median/q2: " + str(median_q2) + 
    #   ", \n q3: " + str(q3) + 
    #   ", \n maximum: " + str(maximum) + 
    #   ", \n outliers: " + str(y[(y < lower_fence) | (y > upper_fence)]))

    ## store box plot results in a dictionary
    boxplot_stats = {}
    boxplot_stats["stats"] = np.array([minimum, 
                                       q1, 
                                       median_q2, 
                                       q3, 
                                       maximum])
   
    ## store observations beyond the box plot extremes
    boxplot_stats["xtrms"] = np.array(y[(y < minimum) | 
                                        (y > maximum)])
    
    ## return dictionary        
    return boxplot_stats

## box plot statistics function
def box_plot_stats(
    
    ## arguments / inputs
    x,          ## input array of values 
    coef = 1.5  ## positive real number
                ## (determines how far the whiskers extend from the iqr)
    ):          
    
    """ 
    calculates box plot five-number summary: the lower whisker extreme, the 
    lower ‘hinge’ (observed value), the median, the upper ‘hinge’, and upper 
    whisker extreme (observed value)
    
    returns a results dictionary containing 2 items: "stats" and "xtrms"
    1) the "stats" item contains the box plot five-number summary as an array
    2) the "xtrms" item contains values which lie beyond the box plot extremes
    
    functions much the same as R's 'boxplot.stats()' function for which this
    Python implementation was predicated
    
    ref:
    
    The R Project for Statistical Computing. (2019). Box Plot Statistics. 
    http://finzi.psych.upenn.edu/R/library/grDevices/html/boxplot.stats.html.
    
    Tukey, J. W. (1977). Exploratory Data Analysis. Section 2C.

    McGill, R., Tukey, J.W. and Larsen, W.A. (1978). Variations of Box Plots. 
    The American Statistician, 32:12-16. http://dx.doi.org/10.2307/2683468.

    Velleman, P.F. and Hoaglin, D.C. (1981). Applications, Basics and 
    Computing of Exploratory Data Analysis. Duxbury Press.

    Emerson, J.D. and Strenio, J. (1983). Boxplots and Batch Comparison. 
    Chapter 3 of Understanding Robust and Exploratory Data Analysis, 
    eds. D.C. Hoaglin, F. Mosteller and J.W. Tukey. Wiley.

    Chambers, J.M., Cleveland, W.S., Kleiner, B. and Tukey, P.A. (1983). 
    Graphical Methods for Data Analysis. Wadsworth & Brooks/Cole.
    """
    
    ## quality check for coef
    if coef <= 0:
        raise ValueError("cannot proceed: coef must be greater than zero")
    
    ## convert input to numpy array
    x = np.array(x)
    
    ## determine median, lower ‘hinge’, upper ‘hinge’
    median = np.quantile(a = x, q = 0.50, method = "midpoint")
    first_quart = np.quantile(a = x, q = 0.25, method = "midpoint")
    third_quart = np.quantile(a = x, q = 0.75, method = "midpoint")
    
    ## calculate inter quartile range
    intr_quart_rng = third_quart - first_quart
    
    ## calculate extreme of the lower whisker (observed, not interpolated)
    lower = first_quart - (coef * intr_quart_rng)
    lower_whisk = np.compress(x >= lower, x)
    lower_whisk_obs = np.min(lower_whisk)
    
    ## calculate extreme of the upper whisker (observed, not interpolated)
    upper = third_quart + (coef * intr_quart_rng)
    upper_whisk = np.compress(x <= upper, x)
    upper_whisk_obs = np.max(upper_whisk)
    
    ## store box plot results dictionary
    boxplot_stats = {}
    boxplot_stats["stats"] = np.array([lower_whisk_obs, 
                                       first_quart, 
                                       median, 
                                       third_quart, 
                                       upper_whisk_obs])
   
    ## store observations beyond the box plot extremes
    boxplot_stats["xtrms"] = np.array(x[(x < lower_whisk_obs) | 
                                        (x > upper_whisk_obs)])
    
    ## return dictionary        
    return boxplot_stats