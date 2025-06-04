# Script path: functions/wercs.py

# This script is part of the 'resreg' package, which was developed by: 
# Gado, J.E., Beckham, G.T., and Payne, C.M (2020). Improving enzyme optimum temperature prediction with resampling strategies and ensemble learning. J. Chem. Inf. Model. 60(8), 4098-4107

# It has been adapted to incorporate the relevance function and control points calculated based on adjusted boxplot statistics, rather than the original boxplot statistics, used by the original developer, to better handle the imbalanced regression problem.

# The 'resreg' Python package was developed based on the following papers: 
# Branco, P., Torgo, L., Ribeiro, R. (2017). SMOGN: A Pre-Processing Approach for Imbalanced Regression. Proceedings of Machine Learning Research, 74:36-50. http://proceedings.mlr.press/v74/branco17a/branco17a.pdf
# Branco, P., Torgo, L., & Ribeiro, R. P. (2019). Pre-processing approaches for imbalanced distributions in regression. Neurocomputing, 343, 76-99. https://www.sciencedirect.com/science/article/abs/pii/S0925231219301638
# Torgo, L., Ribeiro, R. P., Pfahringer, B., & Branco, P. (2013, September). Smote for regression. In Portuguese conference on artificial intelligence (pp. 378-389). Springer, Berlin, Heidelberg. https://link.springer.com/chapter/10.1007/978-3-642-40669-0_33

# This script contains multiple functions for resampling regression datasets. 
# The functions are used to implement the following resampling methods:
# 1. Random oversampling
# 2. Random undersampling
# 3. SMOTER (Synthetic Minority Over-sampling Technique for Regression)
# 4. Gaussian noise
# 5. WERCS (Weighted Examples using Randomly Created Subsamples)


# Author's comments:
"""
resreg: Resampling strategies for regression in Python
"""


## load dependency - third party
import numpy as np
import pandas as pd
import copy
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

from functions import aux_functions


#======================================#
# Functions for resampling datasets
#======================================#

def get_neighbors(X, k):
    """Return indices of k nearest neighbors for each case in X"""
    
    X = np.asarray(X)
    dist = pdist(X)
    dist_mat = squareform(dist)
    order = [np.argsort(row) for row in dist_mat]
    neighbor_indices = np.array([row[1:k+1] for row in order])
    return neighbor_indices


def smoter_interpolate(X, y, k, size, nominal=None, random_state=None):
    """
    Generate new cases by interpolating between cases in the data and a randomly
    selected nearest neighbor. For nominal features, random selection is carried out, 
    rather than interpolation.
    
    Parameters
    ------------
    X : array-like or sparse matrix 
        Features of the data with shape (n_samples, n_features)
    y : array-like
        The target values
    k : int
        Number of nearest neighbors to use in generating synthetic cases by interpolation
    size : int
        Number of new cases to generate
    nominal : ndarray (default=None)
        Column indices of nominal features. If None, then all features are continuous
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator. If None, 
        the random number generator is the RandomState instance used by np.random.
    
    Returns
    --------
    [X_new, y_new] : list
        List contanining features (X_new) and target values (y_new) of new cases generated.
        Dimensions of X_new and y_new are the same as X and y, respectively.
   
    """
    
    X, y = np.asarray(X), np.squeeze(np.asarray(y))
    assert len(X)==len(y), 'X and y must be of the same length.'
    neighbor_indices = get_neighbors(X, k)  # Get indices of k nearest neighbors
    np.random.seed(seed=random_state)
    sample_indices = np.random.choice(range(len(y)), size, replace=True) 
    X_new, y_new = [], []
        
    for i in sample_indices:
        # Get case and nearest neighbor
        X_case, y_case = X[i,:], y[i]
        neighbor = np.random.choice(neighbor_indices[i,:])
        X_neighbor, y_neighbor = X[neighbor, :], y[neighbor]
        
        # Generate synthetic case by interpolation
        rand = np.random.rand() * np.ones_like(X_case)
        
        if nominal is not None:
            rand = [np.random.choice([0,1]) if x in nominal else rand[x] \
                    for x in range(len(rand))] # Random selection for nominal features, rather than interpolation
            rand = np.asarray(rand)
        diff = (X_case - X_neighbor) * rand
        X_new_case = X_neighbor + diff
        d1 = np.linalg.norm(X_new_case - X_case)
        d2 = np.linalg.norm(X_new_case - X_neighbor)
        y_new_case = (d2 * y_case + d1 * y_neighbor) / (d2 + d1 + 1e-10) # Add 1e-10  to avoid division by zero
        X_new.append(X_new_case)
        y_new.append(y_new_case)
    
    X_new = np.array(X_new)
    y_new = np.array(y_new)
    
    return [X_new, y_new]


def add_gaussian(X, y, delta, size, nominal=None, random_state=None):
    """
    Generate new cases  by adding Gaussian noise to the dataset (X, y) . For nominal 
    features, selection is carried out with weights equal to the probability of the 
    nominal feature.
    
    Parameters
    ------------
    X : array-like or sparse matrix 
        Features of the data with shape (n_samples, n_features)
    y : array-like
        The target values
    delta : float
        Value that determines the magnitude of Gaussian noise added
    size : int
        Number of new cases to generate
    nominal : ndarray (default=None)
        Column indices of nominal features. If None, then all features are continuous
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator. If None, 
        the random number generator is the RandomState instance used by np.random.
    
    Returns
    --------
    [X_new, y_new] : list
        List contanining features (X_new) and target values (y_new) of new cases generated.
        Dimensions of X_new and y_new are the same as X and y, respectively.
    """
    
    X, y = np.asarray(X), np.squeeze(np.asarray(y))
    assert len(X)==len(y), 'X and y must be of the same length.'
    np.random.seed(seed=random_state)
    sample_indices = np.random.choice(range(len(y)), size, replace=True)
    stds_X, std_y = np.std(X, axis=0), np.std(y)
    X_sel, y_sel = X[sample_indices,:], y[sample_indices]
    noise_X = np.array([[np.random.normal(loc=0.0, scale=std*delta) for std in stds_X] \
                         for row in range(X_sel.shape[0])])
    noise_y = np.random.normal(loc=0.0, scale=std_y*delta, size=y_sel.shape)
    X_new = X_sel + noise_X
    y_new  = y_sel + noise_y
    
    # Deal with nominal features (selection with weights, not addition of noise)
    if nominal is not None:
        for i in range(X_sel.shape[1]):
            if i in nominal:
                nom_vals, nom_freqs = np.unique(X[:, i], return_counts=True)
                nom_freqs = nom_freqs/nom_freqs.sum()
                nom_select = np.random.choice(nom_vals, size=X_sel.shape[0], p=nom_freqs,
                                              replace=True)
                X_new[:,i] = nom_select
            
    return [X_new, y_new]


def wercs_oversample(X, y, relevance, size, random_state=None):  
    """
    Generate new cases by selecting samples from the original dataset using the 
    relevance as weights. Samples with with high relevance are more likely to be selected
    for oversampling.
    
    
    Parameters
    -------------
    X : array-like or sparse matrix 
        Features of the data with shape (n_samples, n_features)
    y : array-like
        The target values
    relevance : array_like
        Values ranging from 0 to 1 that indicate the relevance of target values. 
    size : int
        Number of new cases to generate
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator. If None, 
        the random number generator is the RandomState instance used by np.random
    
    
   Returns
    --------
    [X_new, y_new] : list
        List contanining features (X_new) and target values (y_new) of new cases generated.
        Dimensions of X_new and y_new are the same as X and y, respectively.
    """
    
    X, y,  = np.asarray(X), np.squeeze(np.asarray(y))
    relevance = np.squeeze(np.asarray(relevance))
    assert len(X)==len(y), 'X and y must be of the same length.'
    assert len(y)==len(relevance), 'y and relevance must be of the same length'
    prob = np.abs(relevance/np.sum(relevance))  # abs to remove very small negative values
    np.random.seed(seed=random_state)
    sample_indices = np.random.choice(range(len(X)), size=size, p=prob, replace=True)
    X_new, y_new = X[sample_indices,:], y[sample_indices]
    
    return X_new, y_new


def wercs_undersample(X, y, relevance, size, random_state=None):
    """Undersample dataset by removing samples selected using the relevance as weights.
    Samples with low relevance are more likely to be removed in undersampling.
    
    Parameters
    -------------
    X : array-like or sparse matrix 
        Features of the data with shape (n_samples, n_features)
    y : array-like
        The target values
    relevance : array_like
        Values ranging from 0 to 1 that indicate the relevance of target values. 
    size : int
        Number of samples in new undersampled dataset (i.e. after removing samples)
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator. If None, 
        the random number generator is the RandomState instance used by np.random.
    
    Returns
    --------
    [X_new, y_new] : list
        List contanining features (X_new) and target values (y_new) of cases after 
        removing samples. Dimensions of X_new and y_new are the same as X and y, 
        respectively.
    """
    
    X, y,  = np.asarray(X), np.squeeze(np.asarray(y))
    relevance = np.squeeze(np.asarray(relevance))
    assert len(X)==len(y), 'X and y must be of the same length.'
    assert 0 < size < len(y), 'size must be smaller than the length of y'
    assert len(y)==len(relevance), 'y and relevance must be of the same length'
    prob = 1 - relevance
    prob = abs(prob/prob.sum())   # abs to remove very small negative numbers
    remove = len(y) - size
    np.random.seed(seed=random_state)
    sample_indices = np.random.choice(range(len(X)), size=remove, p=prob, replace=False)
    sample_indices = list(set(range(len(X))) - set(sample_indices))
    X_new, y_new = X[sample_indices,:], y[sample_indices]
    
    return X_new, y_new
    

def undersample(X, y, size, random_state=None):
    """
    Randomly undersample a dataset (X, y), and return a smaller dataset (X_new, y_new). 
    
    Parameters
    ------------
    X : array-like or sparse matrix 
        Features of the data with shape (n_samples, n_features)
    y : array-like
        The target values
    size : int
        Number of samples in new undersampled dataset.
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator. If None, 
        the random number generator is the RandomState instance used by np.random
    
    Returns
    ----------
    [X_new, y_new] : list
        List contanining features (X_new) and target values (y_new) after undersampling.
    """
        
    X, y = np.asarray(X), np.squeeze(np.asarray(y))
    assert len(X)==len(y), 'X and y must be of the same length.'
    if size >= len(y):
        raise ValueError('size must be smaller than the length of y')
    np.random.seed(random_state)
    new_indices = np.random.choice(range(len(y)), size, replace=False)
    X_new, y_new = X[new_indices, :], y[new_indices]
    return [X_new, y_new]  

    
def oversample(X, y, size, method, k=None, delta=None, relevance=None, nominal=None,
               random_state=None):
    """
    Randomly oversample a dataset (X, y) and return a larger dataset (X_new, y_new)
    according to specified method.
    
    Parameters
    -------------
    X : array-like or sparse matrix 
        Features of the data with shape (n_samples, n_features)
    y : array-like
        The target values
    size : int
        Number of samples in new oversampled dataset.
    method : str, {'random_oversample' | 'smoter' | 'gaussian' | 'wercs' | 'wercs-gn'}
        Method for generating new samples. 
        
        If 'random_oversample', samples are duplicated.
        
        If 'smoter', new synthetic samples are generated by interpolation with the SMOTER
        algorithm. 
        
        If 'gaussian', new synthetic samples are generated by addition of Gaussian noise. 
        
        If 'wercs', relevance values are used as weights to select values for duplication. 
        
        If 'wercs-gn', values are selected with relevance values as weights and then 
        Gaussian noise is added.
        
    k : int (default=None)
        Number of nearest neighbors to use in generating synthetic cases by interpolation.
        Must be specified if method is 'smoter'. 
    delta : float (default=None)
        Value that determines the magnitude of Gaussian noise added. Must be specified if
        method is 'gaussian'
    relevance : array_like (default=None)
        Values ranging from 0 to 1 that indicate the relevance of target values. Must be
        specified if method is 'wercs' or 'wercs-gn'
    nominal : ndarray (default=None)
        Column indices of nominal features. If None, then all features are continuous.
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator. If None, 
        the random number generator is the RandomState instance used by np.random
    
    Returns
    ----------
    [X_new, y_new] : list
        List contanining features (X_new) and target values (y_new) of new samples after
        oversampling.
    """
    
    # Prepare data 
    X, y = np.asarray(X), np.squeeze(np.asarray(y))
    assert len(X)==len(y), 'X and y must be of the same length.'
    moresize = int(size - len(y))
    if moresize <=0:
        raise ValueError('size must be larger than the length of y')
    
    
    # Generate extra samples for oversampling
    np.random.seed(seed=random_state)
    if method=='duplicate':
        more_indices = np.random.choice(np.arange(len(y)), moresize, replace=True)
        X_more, y_more = X[more_indices,:], y[more_indices]
        
    elif method=='smoter':
        if k is None:
            raise ValueError("Must specify k if method is 'smoter'")
        [X_more, y_more] = smoter_interpolate(X, y, k, size=moresize, nominal=nominal, 
                                              random_state=random_state)
    
    elif method=='gaussian':
        if delta is None:
            raise ValueError("Must specify delta if method is 'gaussian'")
        [X_more, y_more] = add_gaussian(X, y, delta, size=moresize, nominal=nominal,
                                        random_state=random_state)
    
    elif method=='wercs' or method=='wercs-gn':
        if relevance is None:
            raise ValueError("Must specify relevance if method is 'wercs' or 'wercs-gn'")
        else:
            assert len(y)==len(relevance), 'y and relevance must be of the same length'
            
        [X_more, y_more] = wercs_oversample(X, y, relevance, size=moresize, 
                                            random_state=random_state)
        if method=='wercs-gn':
            if delta is None:
                raise ValueError("Must specify delta if method is 'wercs-gn'")
           
            [X_more, y_more] = add_gaussian(X_more, y_more, delta, size=moresize, 
                                            nominal=nominal, random_state=random_state)
    else:
        raise ValueError('Wrong method specified.')
    
    # Combine old dataset with extrasamples
    X_new = np.append(X, X_more, axis=0)
    y_new = np.append(y, y_more, axis=0)
    
    return [X_new, y_new]

    
def split_domains(X, y, relevance, relevance_threshold):
    """
    Split a dataset (X,y) into rare and normal domains according to the relevance of the
    target values. Target values with relevance values below the relevance threshold
    form the normal domain, while other target values form the rare domain.
    
    Parameters
    -------------
    X : array-like or sparse matrix 
        Features of the data with shape (n_samples, n_features)
    y : array-like
        The target values
    relevance : array_like
        Values ranging from 0 to 1 that indicate the relevance of target values. 
    relevance_threshold : float
        Threshold of relevance for forming rare and normal domains. Target values with 
        relevance less than relevance_threshold form the normal domain, while target 
        values with relevance greater than or equal to relevance_threshold form the rare
        domain.
    
    Returns
    -----------
    [X_norm, y_norm, X_rare, y_rare] : list
        List containing features (X_norm, X_rare) and target values (y_norm, y_rare) of 
        the normal and rare domains.
    """
    
    X, y = np.asarray(X), np.squeeze(np.asarray(y))
    relevance = np.squeeze(np.asarray(relevance))
    assert len(X) == len(y) == len(relevance), ('X, y, and relevance must have the same '
                                                  'length')
    rare_indices = np.where(relevance >= relevance_threshold)[0]
    norm_indices = np.where(relevance < relevance_threshold)[0]
    assert len(rare_indices) < len(norm_indices), ('Rare domain must be smaller than '
              'normal domain. Adjust your relevance values or relevance threshold so '
              'that the there are fewer samples in the rare domain.')
    X_rare, y_rare = X[rare_indices,:], y[rare_indices]
    X_norm, y_norm = X[norm_indices,:], y[norm_indices]
    
    return [X_norm, y_norm, X_rare, y_rare]
    

#===========================================#
# Functions to implement resampling methods
#===========================================#
 
def random_undersample(X, y, relevance, relevance_threshold=0.5, under='balance',
                       random_state=None):
    """
    Resample imbalanced dataset by undersampling. The dataset is split into a rare and 
    normal domain using relevance values. Target values with relevance below the relevance
    threshold form the normal domain, and other target values form the rare domain. The 
    normal domain is randomly undersampled and the rare domain is left intact.
    
    Parameters
    ------------
    X : array-like or sparse matrix 
        Features of the data with shape (n_samples, n_features)
    y : array-like
        The target values
    relevance : 1d array-like
        Values ranging from 0 to 1 that indicate the relevance of target values. 
    relevance_threshold : float (default=0.5)
        Threshold of relevance for forming rare and normal domains. Target values with 
        relevance less than relevance_threshold form the normal domain, while target 
        values with relevance greater than or equal to relevance_threshold form the rare
        domain.
    under : float or str, {'balance' | 'extreme' | 'average'} (default='balance')
        Value that determines the amount of undersampling. If float, under is the fraction
        of normal samples removed in undersampling. Half of normal samples are removed if
        under=0.5. 
        
        Otherwise, if 'balance', the normal domain is undersampled so that it has the same
        number of samples as the rare domain. 
        
        If 'extreme', the normal domain is undersampled so that the ratio between the 
        sizes of the normal and rare domain is inverted. 
        
        If 'average', the extent of undersampling is intermediate between 'balance' and 
        'extreme'.
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator. If None, 
        the random number generator is the RandomState instance used by np.random.
    
    Returns
    ---------
    [X_new, y_new] : list
        List contanining features (X_new) and target values (y_new) of resampled dataset
        (both normal and rare samples).
        
    """
    
    X, y = np.asarray(X), np.squeeze(np.asarray(y))
    relevance = np.squeeze(np.asarray(relevance))
    [X_norm, y_norm, X_rare, y_rare] = split_domains(X, y, relevance, relevance_threshold)

    # Determine size of normal domain after undersampling
    if type(under)==float:
        assert 0 < under < 1, "under must be between 0 and 1"
        new_norm_size = int((1 - under) * len(y_norm))
    elif under=='balance':
       new_norm_size = int(len(y_rare))
    elif under=='extreme':
         new_norm_size = int(len(y_rare)**2 / len(y_norm))
         if new_norm_size <= 1:
             raise ValueError("under='extreme' results in a normal domain with {0} "
                              "samples".format(new_norm_size))
    elif under=='average':
        new_norm_size = int((len(y_rare) + len(y_rare)**2 / len(y_norm)) / 2)
    else:
        raise ValueError("Incorrect value of 'under' specified.")
   
    # Undersample normal domain
    [X_norm_new, y_norm_new] = undersample(X_norm, y_norm, size=new_norm_size, 
                                            random_state=random_state)
    X_new = np.append(X_norm_new, X_rare, axis=0)
    y_new = np.append(y_norm_new, y_rare, axis=0)
    
    return [X_new, y_new]
        
    
def random_oversample(X, y, relevance, relevance_threshold=0.5, over='balance', 
                      random_state=None):
    """
    Resample imbalanced dataset by oversampling. The dataset is split into a rare and 
    normal domain using relevance values. Target values with relevance below the relevance
    threshold form the normal domain, and other target values form the rare domain. The 
    rare domain is randomly oversampled (duplicated) and the normal domain is left intact.
    
    Parameters
    ------------
    X : array-like or sparse matrix 
        Features of the data with shape (n_samples, n_features)
    y : array-like
        The target values
    relevance : 1d array-like
        Values ranging from 0 to 1 that indicate the relevance of target values. 
    relevance_threshold : float (default=0.5)
        Threshold of relevance for forming rare and normal domains. Target values with 
        relevance less than relevance_threshold form the normal domain, while target 
        values with relevance greater than or equal to relevance_threshold form the rare
        domain.
    over : float or str, {'balance' | 'extreme' | 'average'} (default='balance')
        Value that determines the amount of oversampling. If float, over is the fraction
        of rare samples duplicated in oversampling. Half of rare samples are duplicated if
        over=0.5. 
        
        Otherwise, if 'balance', the rare domain is oversampled so that it has the same 
        number of samples as the normal domain. 
        
        If 'extreme', the rare domain is oversampled so that the ratio between the sizes 
        of the normal and rare domain is inverted. 
        
        If 'medium', the extent of oversampling is intermediate between 'balance' and 
        'extreme'.
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator. If None, 
        the random number generator is the RandomState instance used by np.random.
    
    Returns
    ---------
    [X_new, y_new] : list
        List contanining features (X_new) and target values (y_new) of resampled dataset
        (both normal and rare samples).
    
    References
    -----------
    ..  [1] Branco, P., Torgo, L., and Ribeiro, R.P. (2019). Pre-processing approaches for
        imbalanced distributions in regression.
    """
    
    X, y = np.asarray(X), np.squeeze(np.asarray(y))
    relevance = np.squeeze(np.asarray(relevance))
    [X_norm, y_norm, X_rare, y_rare] = split_domains(X, y, relevance, relevance_threshold)
    
    # Determine size of rare domain after oversampling
    if type(over)==float:
        assert over >=0 , "over must be non-negative"
        new_rare_size = int((1 + over) * len(y_rare))
    elif over=='balance':
       new_rare_size = len(y_norm)
    elif over=='extreme':
         new_rare_size = int(len(y_norm)**2 / len(y_rare))
    elif over=='average':
        new_rare_size = int((len(y_norm) + len(y_norm)**2 / len(y_rare)) / 2)
    else:
        raise ValueError('Incorrect value of over specified')
   
    # Oversample rare domain
    [X_rare_new, y_rare_new] = oversample(X_rare, y_rare, size=new_rare_size,
                                            method='duplicate', random_state=random_state)
    X_new = np.append(X_norm, X_rare_new, axis=0)
    y_new = np.append(y_norm, y_rare_new, axis=0)
    
    return [X_new, y_new]


def smoter(X, y, relevance, relevance_threshold=0.5, k=5, over='balance', under=None, 
		   nominal=None, random_state=None):
    """
    Resample imbalanced dataset with the SMOTER algorithm. The dataset is split into a 
    rare normal domain using relevance values. Target values with relevance below the 
    relevance threshold form the normal domain, and other target values form the rare 
    domain. The rare domain is oversampled by interpolating between samples, and the 
    normal domain is undersampled.
    
    Parameters
    ------------
    X : array-like or sparse matrix 
        Features of the data with shape (n_samples, n_features)
    y : array-like
        The target values
    relevance : 1d array-like
        Values ranging from 0 to 1 that indicate the relevance of target values. 
    relevance_threshold : float (default=0.5)
        Threshold of relevance for forming rare and normal domains. Target values with 
        relevance less than relevance_threshold form the normal domain, while target 
        values with relevance greater than or equal to relevance_threshold form the rare
        domain.
    k : int (default=5)
        Number of nearest neighbors to use in generating synthetic cases by interpolation.
    over : float or str, {'balance' | 'extreme' | 'average'} (default='balance')
        Value that determines the amount of oversampling. If float, over is the fraction
        of new rare samples generated in oversampling.  
        
        Otherwise, if string, over indicates the amount of both oversampling and 
        undersampling. If 'balance', the rare domain is oversampled and the normal domain 
        is undersampled so that they are equal in size.
        
        If 'extreme', oversampling and undersampling are done so that the ratio of the 
        sizes of rare domain to normal domain is inverted. 
        
        If 'average' the extent of oversampling and undersampling is intermediate between 
        'balance' and 'extreme'.
    under : float (default=None)
        Value that determines the amount of undersampling. Should only be specified if
        over is float. One-third of normal samples are removed if under=0.33.
    nominal : ndarray (default=None)
        Column indices of nominal features. If None, then all features are continuous.
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator. If None, 
        the random number generator is the RandomState instance used by np.random.
    
    Returns
    ---------
    [X_new, y_new] : list
        List contanining features (X_new) and target values (y_new) of resampled dataset
        (both normal and rare samples).
    
    References
    -----------
    ..  [1] Torgo, L. et al (2015). Resampling strategies for regression. 
        [2] Branco, P., Torgo, L., and Ribeiro, R.P. (2019). Pre-processing approaches for
        imbalanced distributions in regression.       
    """
    
    # Split data into rare and normal dormains
    X, y = np.asarray(X), np.squeeze(np.asarray(y))
    relevance = np.squeeze(np.asarray(relevance))
    [X_norm, y_norm, X_rare, y_rare] = split_domains(X, y, relevance, relevance_threshold)
    norm_size, rare_size = len(y_norm), len(y_rare)
    
    # Determine new sizes for rare and normal domains after oversampling
    if type(over)==float:
        assert type(under)==float, 'under must also be a float if over is a float'
        assert 0 <= under <= 1, 'under must be between 0 and 1'
        assert over >=0 , "over must be non-negative"
        new_rare_size = int((1 + over) * rare_size)
        new_norm_size = int((1 - under) * norm_size)
    elif over=='balance':
        new_rare_size = new_norm_size = int((norm_size + rare_size)/2)
    elif over == 'extreme':
        new_rare_size, new_norm_size = norm_size, rare_size
    elif over == 'average':
        new_rare_size = int(((norm_size + rare_size)/2 + norm_size)/2)
        new_norm_size = int(((norm_size + rare_size)/2 + rare_size)/2)
    else:
        raise ValueError("Incorrect value of over, must be a float or  "
                         "'balance', 'extreme', or 'average'")
        
    # Oversample rare domain
    y_median = np.median(y)
    low_indices = np.where(y_rare < y_median)[0]
    high_indices = np.where(y_rare >= y_median)[0]
    
    # First oversample low rare cases
    if len(low_indices) != 0:
        size = int(len(low_indices)/rare_size * new_rare_size)
        X_low_rare, y_low_rare = oversample(X_rare[low_indices,:], y_rare[low_indices], 
                                     size=size, method='smoter', k=k, relevance=relevance,
                                     nominal=nominal, random_state=random_state)
        
    # Then do high rare cases
    if len(high_indices) != 0:
        size = int(len(high_indices)/rare_size * new_rare_size)
        X_high_rare, y_high_rare = oversample(X_rare[high_indices], y_rare[high_indices],
                                     size=size, method='smoter', k=k, relevance=relevance,
                                     nominal=nominal, random_state=random_state)
    
    # Combine oversampled low and high rare cases
    if min(len(low_indices), len(high_indices)) != 0:
        X_rare_new = np.append(X_low_rare, X_high_rare, axis=0)
        y_rare_new = np.append(y_low_rare, y_high_rare, axis=0)
    elif len(low_indices) == 0:
        X_rare_new =  X_high_rare
        y_rare_new =  y_high_rare
    elif len(high_indices) == 0:
        X_rare_new =  X_low_rare
        y_rare_new = y_low_rare
        
    # Undersample normal cases
    X_norm_new, y_norm_new = undersample(X_norm, y_norm, size=new_norm_size, 
                                         random_state=random_state)
    
    # Combine resampled rare and normal cases
    X_new = np.append(X_rare_new, X_norm_new, axis=0)
    y_new = np.append(y_rare_new, y_norm_new, axis=0)
    
    return (X_new, y_new)


def gaussian_noise(X, y, relevance, relevance_threshold=0.5, delta=0.05, over=None, under=None,
                   nominal=None, random_state=None):
    """
    Resample imbalanced dataset by introduction of Gaussian noise. The dataset is split 
    into a rare and normal domain using relevance values. Target values with relevance 
    below the relevance threshold form the normal domain, and other target values form the
    rare domain. The rare domain is oversampled by addition of Gaussian noise, and the 
    normal domain is undersampled.
    
    Parameters
    ------------
    X : array-like or sparse matrix 
        Features of the data with shape (n_samples, n_features)
    y : array-like
        The target values
    relevance : 1d array-like
        Values ranging from 0 to 1 that indicate the relevance of target values. 
    relevance_threshold : float (default=0.5)
        Threshold of relevance for forming rare and normal domains. Target values with 
        relevance less than relevance_threshold form the normal domain, while target 
        values with relevance greater than or equal to relevance_threshold form the rare
        domain.
    delta : float (default=0.05)
        Value that determines the magnitude of Gaussian noise added
    over : float or str, {'balance' | 'medium' | 'extreme'} (default='balance')
        Value that determines the amount of oversampling. If float, over is the fraction
        of new rare samples generated in oversampling.  
        
        Otherwise, if string, over indicates the amount of both oversampling and 
        undersampling. If 'balance', the rare domain is oversampled and the normal domain 
        is undersampled so that they are equal in size.
        
        If 'extreme', oversampling and undersampling are done so that the ratio of the 
        sizes of rare domain to normal domain is inverted. 
        
        If 'medium' the extent of oversampling and undersampling is intermediate between 
        'balance' and 'extreme'.
    under : float (default=None)
        Value that determines the amount of undersampling. Should only be specified if
        over is float. One-third of normal samples are removed if under=0.33.
    nominal : ndarray (default=None)
        Column indices of nominal features. If None, then all features are continuous.
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator. If None, 
        the random number generator is the RandomState instance used by np.random.
    
    Returns
    ---------
    [X_new, y_new] : list
        List contanining features (X_new) and target values (y_new) of resampled dataset
        (both normal and rare samples).
    
    References
    -----------
    ..  [1] Branco, P., Torgo, L., and Ribeiro, R.P. (2019). Pre-processing approaches for
        imbalanced distributions in regression. 
    """
    
    # Split data into rare and normal dormains
    X, y = np.asarray(X), np.squeeze(np.asarray(y))
    relevance = np.squeeze(np.asarray(relevance))
    [X_norm, y_norm, X_rare, y_rare] = split_domains(X, y, relevance, relevance_threshold)
    norm_size, rare_size = len(y_norm), len(y_rare)
    
    # Determine new sizes for rare and normal domains after oversampling
    if type(over)==float:
        assert type(under)==float, 'under must be a float if over is a float'
        assert 0 <= under <= 1, 'under must be between 0 and 1'
        assert over >=0 , "over must be non-negative"
        new_rare_size = int((1 + over) * rare_size)
        new_norm_size = int((1 - under) * norm_size)
    elif over=='balance':
        new_rare_size = new_norm_size = int((norm_size + rare_size)/2)
    elif over == 'extreme':
        new_rare_size, new_norm_size = norm_size, rare_size
    elif over == 'average':
        new_rare_size = int(((norm_size + rare_size)/2 + norm_size)/2)
        new_norm_size = int(((norm_size + rare_size)/2 + rare_size)/2)
    else:
        raise ValueError("Incorrect value of over specified, must be a float or  "
                         "'balance', 'extreme', or 'average'")
        
    # Oversample rare domain
    y_median = np.median(y)
    low_indices = np.where(y_rare < y_median)[0]
    high_indices = np.where(y_rare >= y_median)[0]
    
    # First oversample low rare cases
    if len(low_indices) != 0:
        size = int(len(low_indices)/rare_size * new_rare_size)
        X_low_rare, y_low_rare = oversample(X_rare[low_indices,:], y_rare[low_indices], 
                                           size=size, method='gaussian', delta=delta, 
                                           relevance=relevance, nominal=nominal, 
                                           random_state=random_state)
        
    # Then do high rare cases
    if len(high_indices) != 0:
        size = int(len(high_indices)/rare_size * new_rare_size)
        X_high_rare, y_high_rare = oversample(X_rare[high_indices], y_rare[high_indices],
                                     size=size, method='gaussian', delta=delta, 
                                     relevance=relevance, nominal=nominal, 
                                     random_state=random_state)
    
    # Combine oversampled low and high rare cases
    if min(len(low_indices), len(high_indices)) != 0:
        X_rare_new = np.append(X_low_rare, X_high_rare, axis=0)
        y_rare_new = np.append(y_low_rare, y_high_rare, axis=0)
    elif len(low_indices) == 0:
        X_rare_new =  X_high_rare
        y_rare_new =  y_high_rare
    elif len(high_indices) == 0:
        X_rare_new =  X_low_rare
        y_rare_new = y_low_rare
        
    # Undersample normal cases
    X_norm_new, y_norm_new = undersample(X_norm, y_norm, size=new_norm_size, 
                                         random_state=random_state)
    
    # Combine resampled rare and normal cases
    X_new = np.append(X_rare_new, X_norm_new, axis=0)
    y_new = np.append(y_rare_new, y_norm_new, axis=0)
    
    return (X_new, y_new)


def wercs(X, y, relevance, over=0.5, under=0.5, noise=False, delta=0.05, nominal=None,
          random_state=None):
    """
    Resample imbalanced dataset with the WERCS algorithm. The relevance values are used
    as weights to select samples for oversampling and undersampling such that samples with
    high relevance are more likely to be selected for oversampling and less likely to be
    selected for undersampling.
    
    Parameters
    ------------
    X : array-like or sparse matrix 
        Features of the data with shape (n_samples, n_features)
    y : array-like
        The target values
    relevance : 1d array-like
        Values ranging from 0 to 1 that indicate the relevance of target values. 
    over : float (default=0.5)
        Fraction of new samples generated in oversampling.
    under : float (default=0.5)
        Fraction of samples removed in undersampling.
    noise : bool
        Whether to add Gaussian noise to samples selected for oversampling (wercs-gn).
    delta : float (default=0.05)
        Value that determines the magnitude of Gaussian noise added.
    nominal : ndarray (default=None)
        Column indices of nominal features. If None, then all features are continuous.
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator. If None, 
        the random number generator is the RandomState instance used by np.random.
    
    Returns
    ---------
    [X_new, y_new] : list
        List contanining features (X_new) and target values (y_new) of resampled dataset
        (both normal and rare samples).
    
    References
    -----------
    ..  [1] Branco, P., Torgo, L., and Ribeiro, R.P. (2019). Pre-processing approaches for
        imbalanced distributions in regression.
    """
    
    X, y = np.asarray(X), np.squeeze(np.asarray(y))
    relevance = np.squeeze(np.asarray(relevance))
    over_size = int(over * len(y))
    under_size = int((1 - under) * len(y))
    X_over, y_over = wercs_oversample(X, y, relevance=relevance, size=over_size, 
                                      random_state=random_state) # Oversample
    X_under, y_under = wercs_undersample(X, y, relevance=relevance, size=under_size,
                                         random_state=random_state)
    if noise:
        X_under, y_under = add_gaussian(X_under, y_under, delta=delta, size=under_size,
                                        nominal=nominal, random_state=random_state)
    X_new = np.append(X_over, X_under, axis=0)
    y_new = np.append(y_over, y_under, axis=0)
    
    return [X_new, y_new] 


def do_wercs(df, y_label, phi_points, over, under, seed=None):

    """
    Function to apply WERCS resampling method to a dataset. The function takes a pandas
    dataframe and a list of relevance values for each sample in the dataset. The function
    returns a new dataframe with the resampled dataset."""

    df = df.copy()
    
    relevance_array_wercs = np.asarray(phi_points)

    nominal_indices = aux_functions.get_nominal_feature_indices(df)
    #print(nominal_indices)  # Output: array([0, 2])

    result_wercs = wercs(
        
        ## main arguments
        X = df.iloc[:,:-1],                    ## pandas dataframe
        y = df.iloc[:,-1],                     ## pandas dataframe
        relevance = relevance_array_wercs,     ## array with relevance values from relevance function
        over = over,                           ## positive real number (0 < R < 1)
        under = under,                         ## positive real number (0 < R < 1)
        noise = False,                         ## positive real number (0 < R < 1)
        nominal = nominal_indices,             ## array with nominal indices,
        random_state = seed                    ## positive number     
    )

    df_wercs = pd.DataFrame(result_wercs[0])
    df_wercs[y_label] = result_wercs[1]

    df_wercs.columns = df.columns
    df_wercs = df_wercs.astype(df.dtypes.to_dict())

    return df_wercs