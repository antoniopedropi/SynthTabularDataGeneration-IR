# Script path: functions/ru_over_sampling.py

# This script is part of the 'ImbalancedLearningRegression' package, which was developed by: 
# Wu, W., Kunz, N., & Branco, P. (2022). ImbalancedLearningRegression - A Python Package to Tackle the Imbalanced Regression Problem. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases (pp. 645–648). Springer.

# It has been adapted to incorporate the relevance function and control points calculated based on adjusted boxplot statistics, rather than the original boxplot statistics, used by the original developer, to better handle the imbalanced regression problem.

# The 'ImbalancedLearningRegression' Python package was developed based on the following papers: 
# Branco, P., Torgo, L., Ribeiro, R. (2017). SMOGN: A Pre-Processing Approach for Imbalanced Regression. Proceedings of Machine Learning Research, 74:36-50. http://proceedings.mlr.press/v74/branco17a/branco17a.pdf
# Branco, P., Torgo, L., & Ribeiro, R. P. (2019). Pre-processing approaches for imbalanced distributions in regression. Neurocomputing, 343, 76-99. https://www.sciencedirect.com/science/article/abs/pii/S0925231219301638
# Torgo, L., Ribeiro, R. P., Pfahringer, B., & Branco, P. (2013, September). Smote for regression. In Portuguese conference on artificial intelligence (pp. 378-389). Springer, Berlin, Heidelberg. https://link.springer.com/chapter/10.1007/978-3-642-40669-0_33

# This script contains a function for generating synthetic observations, which is the primary function underlying the over-sampling technique utilized in the higher main function 'ru()'.


## load dependencies - third party
import numpy as np
import pandas as pd
import random as rd
from tqdm import tqdm


## under-sampling the majority classes by randomly picking samples
def under_sampling_random(
    
    ## arguments / inputs
    data,       ## training set
    index,      ## index of input data
    perc,       ## under sampling
    replacement ## sampling replacement (bool)
    
    ):
    
    """
    Under-sampling the majority classes by randomly picking samples with or 
    without replacement. It is the primary function underlying the under-sampling 
    technique utilized in the higher main function 'random_under()'. 
    The 4 step procedure for under-sampling the majority classes is:
    
    1) pre-processing: temporarily removes features without variation, label 
    encodes nominal / categorical features, and subsets the training set into 
    two data sets by data type: numeric / continuous, and nominal / categorical
    
    2) under-sampling: random undersampling, which randomly chooses majority samples 
    from the original samples and removes them

    3) post processing: restores original values for label encoded features, 
    reintroduces constant features previously removed, converts any interpolated
    negative values to zero in the case of non-negative features
    
    returns a pandas dataframe containing synthetic observations of the training
    set which are then returned to the higher main function 'random_under()'
    
    ref:
    
    Branco, P., Torgo, L., Ribeiro, R. (2017).
    SMOGN: A Pre-Processing Approach for Imbalanced Regression.
    Proceedings of Machine Learning Research, 74:36-50.
    http://proceedings.mlr.press/v74/branco17a/branco17a.pdf.
    
    Branco, P., Ribeiro, R., Torgo, L. (2017). 
    Package 'UBL'. The Comprehensive R Archive Network (CRAN).
    https://cran.r-project.org/web/packages/UBL/UBL.pdf.

    Branco, P., Torgo, L., & Ribeiro, R. P. (2019). 
    Pre-processing approaches for imbalanced distributions in regression. 
    Neurocomputing, 343, 76-99. 
    https://www.sciencedirect.com/science/article/abs/pii/S0925231219301638

    Kunz, N., (2019). SMOGN. 
    https://github.com/nickkunz/smogn
    """
    
    ## subset original dataframe by bump classification index
    data = data.iloc[index]
    
    ## store dimensions of data subset
    n = len(data)
    d = len(data.columns)
    
    ## store original data types
    feat_dtypes_orig = [None] * d
    
    for j in range(d):
        feat_dtypes_orig[j] = data.iloc[:, j].dtype
    
    ## find non-negative numeric features
    feat_non_neg = [] 
    num_dtypes = ["int64", "float64"]
    
    for j in range(d):
        if data.iloc[:, j].dtype in num_dtypes and any(data.iloc[:, j] > 0):
            feat_non_neg.append(j)
    
    ## find features without variation (constant features)
    feat_const = data.columns[data.nunique() == 1]
    
    ## temporarily remove constant features
    if len(feat_const) > 0:
        
        ## create copy of orignal data and omit constant features
        data_orig = data.copy()
        data = data.drop(data.columns[feat_const], axis = 1)
        
        ## store list of features with variation
        feat_var = list(data.columns.values)
        
        ## reindex features with variation
        for i in range(d - len(feat_const)):
            data.rename(columns = {
                data.columns[i]: i
                }, inplace = True)
        
        ## store new dimension of feature space
        d = len(data.columns)
    
    ## create copy of data containing variation
    data_var = data.copy()
    
    ## create global feature list by column index
    feat_list = list(data.columns.values)
    
    ## create nominal feature list and
    ## label encode nominal / categorical features
    ## (strictly label encode, not one hot encode) 
    feat_list_nom = []
    nom_dtypes = ["object", "bool", "datetime64"]

    # Unknown warning, may be handled later
    pd.options.mode.chained_assignment = None
    
    for j in range(d):
        if data.dtypes[j] in nom_dtypes:
            feat_list_nom.append(j)
            data.iloc[:, j] = pd.Categorical(pd.factorize(
                data.iloc[:, j])[0])
    
    data = data.apply(pd.to_numeric)
    
    
    ## total number of majority samples to be removed
    n_synth = int(n * perc)
    
    ## randomly choose index data by the number of majority samples to be removed
    ## "replace" is a parameter of np.random.choice, meaning that if 
    ## a value can be selected multiple times. This is the same as the 
    ## "replacement" parameter defined in "random_under()
    r_index = np.random.choice(
        a = tuple(range(0, n)), 
        size = n_synth,
        replace = replacement, 
        p = None
    )
    
    ## create null matrix to store the remaining synthetic observations
    # under_matrix = np.ndarray(shape = ((n - n_synth), d))
    under_matrix = np.ndarray(shape = (n_synth, d))

    
    # ## create a sub_index for subset data
    # sub_index = []
    # for i in range(n):
    #     sub_index.append(i)
    
    ## find the non-intersecting values of sub_index and r_index  
    # new_index = np.setxor1d(sub_index, r_index)
    new_index = r_index

    # added
    ## store data in the synthetic matrix, data indices are chosen randomly above
    count = 0 
    for i in tqdm(new_index, ascii = True, desc = "new_index"):
        for attr in range(d):
            under_matrix[count, attr] = (data.iloc[i, attr])
        count = count + 1

    ## convert synthetic matrix to dataframe
    data_new = pd.DataFrame(under_matrix)
    
    ## synthetic data quality check
    if sum(data_new.isnull().sum()) > 0:
        raise ValueError("oops! under_sampled data contains missing values")
    
    ## replace label encoded values with original values
    for j in feat_list_nom:
        code_list = data.iloc[:, j].unique()
        cat_list = data_var.iloc[:, j].unique()
        
        for x in code_list:
            data_new.iloc[:, j] = data_new.iloc[:, j].replace(x, cat_list[x])
    
    ## reintroduce constant features previously removed
    if len(feat_const) > 0:
        data_new.columns = feat_var
        
        for j in range(len(feat_const)):
            data_new.insert(
                loc = int(feat_const[j]),
                column = feat_const[j], 
                value = np.repeat(
                    data_orig.iloc[0, feat_const[j]], 
                    len(under_matrix))
            )
    
    ## convert negative values to zero in non-negative features
    for j in feat_non_neg:
        # data_new.iloc[:, j][data_new.iloc[:, j] < 0] = 0
        data_new.iloc[:, j] = data_new.iloc[:, j].clip(lower = 0)
    
    ## return under-sampling results dataframe
    return data_new