# Script path: functions/random_over_sampling.py

# This script is part of the 'ImbalancedLearningRegression' package, which was developed by: 
# Wu, W., Kunz, N., & Branco, P. (2022). ImbalancedLearningRegression - A Python Package to Tackle the Imbalanced Regression Problem. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases (pp. 645–648). Springer.

# It has been adapted to incorporate the relevance function and control points calculated based on adjusted boxplot statistics, rather than the original boxplot statistics, used by the original developer, to better handle the imbalanced regression problem.

# The 'ImbalancedLearningRegression' Python package was developed based on the following papers: 
# Branco, P., Torgo, L., Ribeiro, R. (2017). SMOGN: A Pre-Processing Approach for Imbalanced Regression. Proceedings of Machine Learning Research, 74:36-50. http://proceedings.mlr.press/v74/branco17a/branco17a.pdf
# Branco, P., Torgo, L., & Ribeiro, R. P. (2019). Pre-processing approaches for imbalanced distributions in regression. Neurocomputing, 343, 76-99. https://www.sciencedirect.com/science/article/abs/pii/S0925231219301638
# Torgo, L., Ribeiro, R. P., Pfahringer, B., & Branco, P. (2013, September). Smote for regression. In Portuguese conference on artificial intelligence (pp. 378-389). Springer, Berlin, Heidelberg. https://link.springer.com/chapter/10.1007/978-3-642-40669-0_33

# This script contains a function for the Random Over-Sampling (RO) technique for regression. 
# RO is a technique that applies over-sampling of the minority class (rare values in a normal distribution of y, typically found at the tails). 


## load dependencies - third party
import numpy as np
import pandas as pd


# load dependencies - internal
from functions.relevance_function import phi
from functions.relevance_function_ctrl_pts_normal import phi_ctrl_pts
from functions.ro_over_sampling import over_sampling_ro


## synthetic minority over-sampling technique for regression with random oversampling
def ro(
    
    ## main arguments / inputs
    data,                     ## training set (pandas dataframe)
    y,                        ## response variable y by name (string)
    samp_method = "balance",  ## over / under sampling ("balance" or extreme")
    drop_na_col = True,       ## auto drop columns with nan's (bool)
    drop_na_row = True,       ## auto drop rows with nan's (bool)
    replace = True,           ## sampling replacement (bool)
    manual_perc = False,      ## user defines percentage of under-sampling and over-sampling  # added
    perc_o = -1,              ## percentage of over-sampling  # added
    
    ## phi relevance function arguments / inputs
    rel_thres = 0.5,          ## relevance threshold considered rare (pos real)
    rel_method = "auto",      ## relevance method ("auto" or "manual")
    rel_xtrm_type = "both",   ## distribution focus ("high", "low", "both")
    rel_coef = 1.5,           ## coefficient for box plot (pos real)
    rel_ctrl_pts_rg = None    ## input for "manual" rel method  (2d array)
    
    ):
    
    """
    the main function, designed to help solve the problem of imbalanced data 
    for regression; RO applies over-sampling of the minority class (rare 
    values in a normal distribution of y, typically found at the tails)
    
    procedure begins with a series of pre-processing steps, and to ensure no 
    missing values (nan's), sorts the values in the response variable y by
    ascending order, and fits a function 'phi' to y, corresponding phi values 
    (between 0 and 1) are generated for each value in y, the phi values are 
    then used to determine if an observation is either normal or rare by the 
    threshold specified in the argument 'rel_thres' 
    
    normal observations are placed into a majority class subset (normal bin), 
    while rare observations are placed in a seperate minority class subset 
    (rare bin) where they're over-sampled
    
    over-sampling is applied by a random sampling from the rare bin based 
    on a calculated percentage control by the argument 'samp_method', if the 
    specified input of 'samp_method' is "balance", less over-sampling is 
    conducted, and if "extreme" is specified more over-sampling is conducted
    
    over-sampling is applied by RO, which randomly duplicate samples from the 
    original samples
    
    procedure concludes by post-processing and returns a modified pandas data
    frame containing over-sampled (synthetic) observations, the distribution 
    of the response variable y should more appropriately reflect the minority 
    class areas of interest in y that are under-represented in the original 
    training set

    note that users can also decide the percentage of over-sampling on each 
    bin by setting manual_perc to True and assigning value to perc_o, this 
    value will directly replace the calculated percentage of each relavant bin

    note also that if the percentage of over-sampling is greater than 1 
    and replace = False, we simply duplicate the original samples i times, 
    where i is the integer part of the percentage of over-sampling, and 
    randomly sample d * n data from the original samples, where d is the 
    decimal part of the percentage of over-sampling and n is the number of 
    the original samples; if replace = True, we randomly sample (i + d) * n 
    data from the original samples with the possibility of reusing the 
    same sample
    
    ref:
    
    Branco, P., Torgo, L., Ribeiro, R. (2017).
    SMOGN: A Pre-Processing Approach for Imbalanced Regression.
    Proceedings of Machine Learning Research, 74:36-50.
    http://proceedings.mlr.press/v74/branco17a/branco17a.pdf.

    Branco, P., Torgo, L., & Ribeiro, R. P. (2019). 
    Pre-processing approaches for imbalanced distributions in regression. 
    Neurocomputing, 343, 76-99. 
    https://www.sciencedirect.com/science/article/abs/pii/S0925231219301638

    Kunz, N., (2019). SMOGN. 
    https://github.com/nickkunz/smogn
    """
    
    ## pre-process missing values
    if bool(drop_na_col) == True:
        data = data.dropna(axis = 1)  ## drop columns with nan's
    
    if bool(drop_na_row) == True:
        data = data.dropna(axis = 0)  ## drop rows with nan's
    
    ## quality check for missing values in dataframe
    if data.isnull().values.any():
        raise ValueError("cannot proceed: data cannot contain NaN values")
    
    ## quality check for y
    if isinstance(y, str) is False:
        raise ValueError("cannot proceed: y must be a string")
    
    if y in data.columns.values is False:
        raise ValueError("cannot proceed: y must be an header name (string) \
               found in the dataframe")
    
    ## quality check for sampling method
    if samp_method in ["balance", "extreme"] is False:
        raise ValueError("samp_method must be either: 'balance' or 'extreme'")

    # added
    ## quality check for sampling percentage
    if manual_perc:
        if perc_o == -1:
            raise ValueError("cannot proceed: require percentage of over-sampling if manual_perc == True")
        if perc_o <= 0:
            raise ValueError("percentage of over-sampling must be a positve real number")
    
    ## quality check for relevance threshold parameter
    if rel_thres == None:
        raise ValueError("cannot proceed: relevance threshold required")
    
    if rel_thres > 1 or rel_thres <= 0:
        raise ValueError("rel_thres must be a real number number: 0 < R < 1")
    
    ## store data dimensions
    n = len(data)
    d = len(data.columns)
    
    ## store original data types
    feat_dtypes_orig = [None] * d
    
    for j in range(d):
        feat_dtypes_orig[j] = data.iloc[:, j].dtype
    
    ## determine column position for response variable y
    y_col = data.columns.get_loc(y)
    
    ## move response variable y to last column
    if y_col < d - 1:
        cols = list(range(d))
        cols[y_col], cols[d - 1] = cols[d - 1], cols[y_col]
        data = data[data.columns[cols]]
    
    ## store original feature headers and
    ## encode feature headers to index position
    feat_names = list(data.columns)
    data.columns = range(d)
    
    ## sort response variable y by ascending order
    y = pd.DataFrame(data[d - 1])
    y_sort = y.sort_values(by = d - 1)
    y_sort = y_sort[d - 1]
    
    ## -------------------------------- phi --------------------------------- ##
    ## calculate parameters for phi relevance function
    ## (see 'phi_ctrl_pts()' function for details)
    phi_params = phi_ctrl_pts(
        
        y = y_sort,                ## y (ascending)
        method = rel_method,       ## defaults "auto" 
        xtrm_type = rel_xtrm_type, ## defaults "both"
        coef = rel_coef,           ## defaults 1.5
        ctrl_pts = rel_ctrl_pts_rg ## user spec
    )
    
    ## calculate the phi relevance function
    ## (see 'phi()' function for details)
    y_phi = phi(
        
        y = y_sort,                ## y (ascending)
        ctrl_pts = phi_params      ## from 'phi_ctrl_pts()'
    )
    
    ## phi relevance quality check
    if all(i == 0 for i in y_phi):
        raise ValueError("redefine phi relevance function: all points are 1")
    
    if all(i == 1 for i in y_phi):
        raise ValueError("redefine phi relevance function: all points are 0")
    ## ---------------------------------------------------------------------- ##
    
    ## determine bin (rare or normal) by bump classification
    bumps = [0]
    
    for i in range(0, len(y_sort) - 1):
        if ((y_phi[i] >= rel_thres and y_phi[i + 1] < rel_thres) or 
            (y_phi[i] < rel_thres and y_phi[i + 1] >= rel_thres)):
                bumps.append(i + 1)
    
    bumps.append(n)
    
    ## number of bump classes
    n_bumps = len(bumps) - 1
    
    ## determine indicies for each bump classification
    b_index = {}
    
    for i in range(n_bumps):
        b_index.update({i: y_sort[bumps[i]:bumps[i + 1]]})
    
    ## calculate over / under sampling percentage according to
    ## bump class and user specified method ("balance" or "extreme")
    b = round(n / n_bumps)
    s_perc = []
    scale = []
    obj = []
    
    if samp_method == "balance":
        for i in b_index:
            s_perc.append(b / len(b_index[i]))
            
    if samp_method == "extreme":
        for i in b_index:
            scale.append(b ** 2 / len(b_index[i]))
        scale = n_bumps * b / sum(scale)
        
        for i in b_index:
            obj.append(round(b ** 2 / len(b_index[i]) * scale, 2))
            s_perc.append(round(obj[i] / len(b_index[i]), 1))
    
    ## conduct over / under sampling and store modified training set
    data_new = pd.DataFrame()
    
    for i in range(n_bumps):
        
        ## no sampling
        if s_perc[i] == 1:
            
            ## simply return no sampling
            ## results to modified training set
            data_new = pd.concat([data.iloc[b_index[i].index], data_new], ignore_index = True)
        
        ## over-sampling
        if s_perc[i] > 1:
            
            ## generate synthetic observations in training set
            ## considered 'minority'
            ## (see 'over_sampling_ro()' function for details)
            synth_obs = over_sampling_ro(
                data = data,
                index = list(b_index[i].index),
                perc = s_perc[i] if not manual_perc else perc_o + 1,  # modified
                replace = replace  # added
            )
            
            ## concatenate over-sampling
            ## results to modified training set
            data_new = pd.concat([synth_obs, data_new], ignore_index = True)

            # added
            ## concatenate original data
            ## to modified training set
            original_obs = data.iloc[list(b_index[i].index)]
            data_new = pd.concat([original_obs, data_new], ignore_index = True)

        ## no sampling
        if s_perc[i] < 1:
            
            # added
            ## concatenate original data
            ## to modified training set
            original_obs = data.iloc[list(b_index[i].index)]
            data_new = pd.concat([original_obs, data_new], ignore_index = True)

    
    ## rename feature headers to originals
    data_new.columns = feat_names
    
    ## restore response variable y to original position
    if y_col < d - 1:
        cols = list(range(d))
        cols[y_col], cols[d - 1] = cols[d - 1], cols[y_col]
        data_new = data_new[data_new.columns[cols]]
    
    ## restore original data types
    for j in range(d):
        data_new.iloc[:, j] = data_new.iloc[:, j].astype(feat_dtypes_orig[j])
    
    ## return modified training set
    return data_new