# Script path: functions/wsmoter_over_sampling.py

# This script is part of the 'wsmoter' code, which was developed by: 
# Camacho, L., Bacao, F. WSMOTER: a novel approach for imbalanced regression. 
# Appl Intell 54, 8789–8799 (2024). https://doi.org/10.1007/s10489-024-05608-6

# This script contains the implementation of the WSMOTER algorithm, which is a novel approach for imbalanced regression.
# The WSMOTER algorithm is an extension of the SMOTE algorithm, which is a well-known algorithm for imbalanced classification.


# Author's comments :

"""Base class for sampling"""

# WSMOTER code
# WSMOTER is suitable for imbalanced regression

# For the implementation of WSMOTER, we used the code of SMOTE and SMOTENC as baseline. 
# The original code is from:

# Lemaitre, G., Nogueira, F., Aridas, C.K.: Imbalanced-learn: A python toolbox to
# tackle the curse of imbalanced datasets in machine learning. J. Mach. Learn. Res.
# 18(1), 559–563 (2017)

# All changes in the code were made by Luis Camacho


## load dependency - third party
from abc import ABCMeta, abstractmethod

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.preprocessing import label_binarize
from sklearn.utils.multiclass import check_classification_targets

from imblearn.utils import check_sampling_strategy, check_target_type
from imblearn.utils._validation import ArraysTransformer
from imblearn.utils._validation import _deprecate_positional_args

import math
from collections import Counter

import pandas as pd
import scipy.interpolate
from scipy import sparse

from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import paired_distances

from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_random_state
from sklearn.utils import _safe_indexing
from sklearn.utils import check_array
from sklearn.utils.sparsefuncs_fast import csr_mean_variance_axis0
from sklearn.utils.sparsefuncs_fast import csc_mean_variance_axis0

from imblearn.utils import check_neighbors_object
from imblearn.utils import Substitution
from imblearn.utils._docstring import _n_jobs_docstring
from imblearn.utils._docstring import _random_state_docstring

import statsmodels.stats.stattools as sss
from denseweight import DenseWeight
from sklearn.preprocessing import MinMaxScaler


class SamplerMixin(BaseEstimator, metaclass=ABCMeta):

    _estimator_type = "sampler"

    def fit(self, X, y):

        X, y, _ = self._check_X_y(X, y)
        self.sampling_strategy_ = check_sampling_strategy(
            self.sampling_strategy, y, self._sampling_type
        )
        return self

    def fit_resample(self, X, y, y_num = None):

        check_classification_targets(y)
        arrays_transformer = ArraysTransformer(X, y)
        X, y, binarize_y = self._check_X_y(X, y)
        self.sampling_strategy_ = check_sampling_strategy(
            self.sampling_strategy, y, self._sampling_type
        )

        if y_num is not None:
            ynum = np.array(y_num)

        if y_num is not None:          
            output = self._fit_resample(X, y, ynum)
        else:
            output = self._fit_resample(X, y)

        y_ = (label_binarize(output[1], np.unique(y))
              if binarize_y else output[1])
        X_, y_ = arrays_transformer.transform(output[0], y_)
        
        if y_num is not None:
            ynum_ = np.array(output[2])
            
        return (X_, y_) if len(output) == 2 else (X_, y_, ynum_)

    fit_sample = fit_resample


    @abstractmethod
    def _fit_resample(self, X, y, y_num = None):

        pass


class BaseSampler(SamplerMixin):
    
    def __init__(self, sampling_strategy="auto"):
        self.sampling_strategy = sampling_strategy

    def _check_X_y(self, X, y, accept_sparse=None):
        if accept_sparse is None:
            accept_sparse = ["csr", "csc"]
        y, binarize_y = check_target_type(y, indicate_one_vs_all=True)
        X, y = self._validate_data(
            X, y, reset=True, accept_sparse=accept_sparse
        )
        return X, y, binarize_y


class BaseOverSampler(BaseSampler):

    _sampling_type = "over-sampling"

    _sampling_strategy_docstring = """sampling_strategy : float, str, dict or callable, default='auto'
        Sampling information to resample the data set.
        """.strip()


#WSMOTER
"""Class to perform over-sampling using WSMOTER."""

class BaseSMOTE(BaseOverSampler):

    def __init__(
        self,
        sampling_strategy="auto",
        random_state=None,
        extreme='right',
        k_neighbors=5,
        weights='dense',
        alpha=1.0,
        beta=1,
        n_jobs=None,
    ):    
        super().__init__(sampling_strategy=sampling_strategy)
        self.random_state = random_state
        self.extreme = extreme
        self.k_neighbors = k_neighbors
        self.weights = weights
        self.alpha = alpha
        self.beta = beta
        self.n_jobs = n_jobs

    def _validate_estimator(self):
        
        self.nn_k_ = check_neighbors_object(
            "k_neighbors", self.k_neighbors, additional_neighbor=1
        )

    def _make_samples(
        self, X, y_dtype, y_type, nn_data, nn_num, n_samples, step_size=1.0,
                y_num = None, yweights = None):

        random_state = check_random_state(self.random_state)

        y_weights_relativo = yweights / yweights.sum()

        rows = np.random.RandomState(self.random_state).choice(len(yweights), n_samples, p=y_weights_relativo)
 
        samples_indices = random_state.randint(
            low=0, high=nn_num.size, size=n_samples)
        
        cols = np.mod(samples_indices, nn_num.shape[1])


        steps = step_size * random_state.uniform(size=n_samples)[:, np.newaxis]

        if y_num is not None:
            X_new = self._generate_samples(X, nn_data, nn_num, rows, cols,
                                           steps, y_num)
        else:
            X_new = self._generate_samples(X, nn_data, nn_num, rows, cols,
                                           steps)      
        
        y_new = np.full(n_samples, fill_value=y_type, dtype=y_dtype)


        if y_num is not None:
            distances_1 = paired_distances(X[rows], X_new, metric = 'euclidean')
            distances_2 = paired_distances(nn_data[nn_num[rows, cols]], X_new, metric = 'euclidean')
        
            distances_extremes = distances_1 + distances_2
        
        if y_num is not None:    
            distances1_per = np.divide(distances_1, distances_extremes, out=np.full(distances_1.shape, 0.5), where=distances_extremes!=0)
            distances2_per = np.divide(distances_2, distances_extremes, out=np.full(distances_2.shape, 0.5), where=distances_extremes!=0)

        if y_num is not None:
            y_num_new = ((1 - distances1_per) * y_num[rows] + (1 - distances2_per) * y_num[nn_num[rows, cols]])                     
        
        if y_num is not None:
            return X_new, y_new, y_num_new
        else:
            return X_new, y_new 

    def _generate_samples(self,X,nn_data,nn_num,rows,cols,steps,y_num = None):
        r"""Generate a synthetic sample.
        """

        diffs = nn_data[nn_num[rows, cols]] - X[rows]        

        if sparse.issparse(X):
            sparse_func = type(X).__name__
            steps = getattr(sparse, sparse_func)(steps)
            X_new = X[rows] + steps.multiply(diffs)
        else:
            X_new = X[rows] + steps * diffs

        return X_new.astype(X.dtype)
    


@Substitution(
    sampling_strategy=BaseOverSampler._sampling_strategy_docstring,
    n_jobs=_n_jobs_docstring,
    random_state=_random_state_docstring,
)
class WSMOTERdense(BaseSMOTE):

    @_deprecate_positional_args
    def __init__(
        self,
        *,
        sampling_strategy="auto",
        random_state=None,
        extreme = 'right',
        k_neighbors=5,
        weights='dense',
        alpha=1.0,
        beta=1,
        n_jobs=None,
    ):
        super().__init__(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            extreme = extreme,
            k_neighbors=k_neighbors,
            weights=weights,
            alpha = alpha,
            beta = beta,
            n_jobs=n_jobs,          
        )

    def y_relevance(self, y_num, extreme, y_num2 = None):
        
        def medcouple_(y_):
            return float(sss.medcouple(y_))
                
        Q1 = pd.Series(y_num).describe()['25%']
        Q2 = pd.Series(y_num).describe()['50%']
        Q3 = pd.Series(y_num).describe()['75%']
       
        medc = medcouple_(y_num)
        if medc >= 0:
            adjL = Q1-1.5*math.exp(-4*medc)*(Q3-Q1)
            adjH = Q3+1.5*math.exp(3*medc)*(Q3-Q1)
        else:
            adjL = Q1-1.5*math.exp(-3*medc)*(Q3-Q1)
            adjH = Q3+1.5*math.exp(4*medc)*(Q3-Q1)
        
        m = y_num.min() - 1
        M = y_num.max() + 1
    
        xx_left = np.array([min(m,adjL-1),adjL,Q2,M])
        yy_left = np.array([1.0,1.0,0.0,0.0])    
        xx_right = np.array([m,Q2,adjH,max(adjH+1,M)])
        yy_right = np.array([0.0,0.0,1.0,1.0])


        if extreme == 'left' or extreme == 'both':  
            fr0 = scipy.interpolate.PchipInterpolator(xx_left,yy_left) 
        if extreme == 'right' or extreme == 'both':    
            fr1 = scipy.interpolate.PchipInterpolator(xx_right,yy_right)

        if y_num2 is not None:
            if extreme == 'left':
                return fr0(y_num2)
            elif extreme == 'right':
                return fr1(y_num2)
            elif extreme == 'both':  
                return np.where(y_num2 < Q2, fr0(y_num2), fr1(y_num2))
   
        if extreme == 'left':
            return fr0(y_num)
        elif extreme == 'right':
            return fr1(y_num)
        elif extreme == 'both':  
            return np.where(y_num < Q2, fr0(y_num), fr1(y_num))


    def y_dense(self, y_num):
        
        scaler_y_num = MinMaxScaler()
        y_num_scaled = scaler_y_num.fit_transform(np.array(y_num).reshape(-1, 1)).flatten() 
        dw = DenseWeight(alpha=self.alpha)
        y_weights = dw.fit(np.array(y_num_scaled))
        
        return y_weights


    def _fit_resample(self, X, y, y_num = None):
        
        self._validate_estimator()

        X_resampled = [X.copy()]
        y_resampled = [y.copy()]

        if y_num is not None:
            y_numeric_resampled = [y_num.copy()]

   
        if self.weights == 'relevance':
            y_weights = self.y_relevance(y_num, self.extreme)
          
        elif self.weights == 'dense': 
            y_weights = self.y_dense(y_num)  

        for class_sample, n_samples in self.sampling_strategy_.items():
            
            if n_samples == 0:
                continue
            target_class_indices = np.flatnonzero(y == class_sample)
            X_class = _safe_indexing(X, target_class_indices)

            ynum = y_num[target_class_indices] 
                                            
            if self.weights == 'relevance':
                yweights = y_weights[target_class_indices]
                
            elif self.weights == 'dense':               
                yweights = y_weights[target_class_indices]
                
            if self.weights == 'dense': 
                y_indices = []
                ynum_sorted = np.sort(ynum)
                y_nneighbors = int(self.beta * 10)
                if class_sample == 1:
                    for ivalue in range(len(ynum)):
                        yvalue = ynum[ivalue]
                        yvalue_position = np.where(ynum_sorted == yvalue)[0][0]
                        if yvalue_position - y_nneighbors >= 0:
                            yvalue1 = ynum_sorted[yvalue_position - y_nneighbors]
                            yvalue2 = yvalue
                        else:
                            yvalue1 = ynum_sorted[0] 
                            yvalue2 = ynum_sorted[y_nneighbors]
                        ynum_interval = np.where((ynum >= yvalue1) & (ynum <= yvalue2))[0]
                        y_indices.append(list(np.delete(ynum_interval,np.where(ynum_interval==ivalue))))
                elif class_sample == 2:
                    for ivalue in range(len(ynum)):
                        yvalue = ynum[ivalue]
                        yvalue_position = np.where(ynum_sorted == yvalue)[0][-1]
                        if yvalue_position + y_nneighbors < len(ynum_sorted):
                            yvalue1 = yvalue
                            yvalue2 = ynum_sorted[yvalue_position + y_nneighbors]
                        else:
                            yvalue1 = ynum_sorted[-1 * y_nneighbors - 1] 
                            yvalue2 = ynum_sorted[-1]
                        ynum_interval = np.where((ynum >= yvalue1) & (ynum <= yvalue2))[0]
                        y_indices.append(list(np.delete(ynum_interval,np.where(ynum_interval==ivalue))))
       
            elif self.weights == 'relevance':                
                y_indices = []
                for i in range(len(yweights)):
                    rel_indices_row = []
                    for j in np.delete(range(len(yweights)),i):
                        if ((yweights[j] >= yweights[i]) & (yweights[j] <= yweights[i] + 0.2)):
                            rel_indices_row.append(j)
 
                    kstep = 0.2                    
                    while len(rel_indices_row) < 5: 
                        rel_indices_row = []
                        for j in np.delete(range(len(yweights)),i):
                            if ((yweights[j] >= yweights[i] - kstep) & (yweights[j] <= yweights[i] + 0.2)):
                                rel_indices_row.append(j)
                        kstep = kstep + 0.1
                    
                    y_indices.append(rel_indices_row)

            neighbors_indices_aux = []
            for i in range(X_class.shape[0]):

                neighbors_indices_aux.append(list(np.argpartition(pairwise_distances(X_class[i].reshape(1,-1), X_class[y_indices[i],:], 
                                        metric = 'euclidean')[0],self.k_neighbors - 1)[:self.k_neighbors]))
            
            nns = np.zeros((X_class.shape[0],self.k_neighbors),dtype=int)
            for i in range(len(y_indices)):
                nns[i] = np.array(y_indices[i])[neighbors_indices_aux[i]]
                   
          
            if y_num is not None:
                X_new, y_new, y_numeric = self._make_samples( X_class, y.dtype,
                            class_sample, X_class, nns, n_samples, 1.0, ynum, yweights )
            else:
                X_new, y_new =self._make_samples( X_class,y.dtype,class_sample,
                                        X_class, nns, n_samples, 1.0 )
                
            X_resampled.append(X_new)
            y_resampled.append(y_new)

            if y_num is not None:
                y_numeric_resampled.append(y_numeric)
                

        if sparse.issparse(X):
            X_resampled = sparse.vstack(X_resampled, format=X.format)
        else:
            X_resampled = np.vstack(X_resampled)
            
        y_resampled = np.hstack(y_resampled)

        if y_num is not None:
            y_numeric_resampled = np.hstack(y_numeric_resampled)


        if y_num is not None:
            return X_resampled, y_resampled, y_numeric_resampled
        else:
            return X_resampled, y_resampled
        

# @Substitution(
#     sampling_strategy=BaseOverSampler._sampling_strategy_docstring,
#     random_state=_random_state_docstring)
class WSMOTERNCdense(WSMOTERdense):
    
    _required_parameters = ["categorical_features"]

    @_deprecate_positional_args
    def __init__(
        self,
        categorical_features,
        *,
        sampling_strategy="auto",
        random_state=None,
        extreme='right',
        k_neighbors=5,
        weights='dense',
        alpha=1.0,
        beta=1,
        n_jobs=None,
    ):
        super().__init__(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            extreme = extreme,
            weights=weights,
            alpha = alpha,
            beta = beta,
            k_neighbors=k_neighbors,
        )
        self.categorical_features = categorical_features

    def _check_X_y(self, X, y):
        
        y, binarize_y = check_target_type(y, indicate_one_vs_all=True)
        X, y = self._validate_data(
            X, y, reset=True, dtype=None, accept_sparse=["csr", "csc"]
        )
        return X, y, binarize_y

    def _validate_estimator(self):
        super()._validate_estimator()
        categorical_features = np.asarray(self.categorical_features)
        if categorical_features.dtype.name == "bool":
            self.categorical_features_ = np.flatnonzero(categorical_features)
        else:
            if any(
                [
                    cat not in np.arange(self.n_features_)
                    for cat in categorical_features
                ]
            ):
                raise ValueError(
                    "Some of the categorical indices are out of range. Indices"
                    " should be between 0 and {}".format(self.n_features_)
                )
            self.categorical_features_ = categorical_features
        self.continuous_features_ = np.setdiff1d(
            np.arange(self.n_features_), self.categorical_features_
        )

        if self.categorical_features_.size == self.n_features_in_:
            raise ValueError(
                "SMOTE-NC is not designed to work only with categorical "
                "features. It requires some numerical features."
            )

    def _fit_resample(self, X, y, y_num = None):
        self.n_features_ = X.shape[1]
        self._validate_estimator()

        target_stats = Counter(y)
        class_minority = min(target_stats, key=target_stats.get)

        X_continuous = X[:, self.continuous_features_]
        X_continuous = check_array(X_continuous, accept_sparse=["csr", "csc"])
        X_minority = _safe_indexing(
            X_continuous, np.flatnonzero(y == class_minority)
        )

        if sparse.issparse(X):
            if X.format == "csr":
                _, var = csr_mean_variance_axis0(X_minority)
            else:
                _, var = csc_mean_variance_axis0(X_minority)
        else:
            var = X_minority.var(axis=0)
        self.median_std_ = np.median(np.sqrt(var))

        X_categorical = X[:, self.categorical_features_]
        if X_continuous.dtype.name != "object":
            dtype_ohe = X_continuous.dtype
        else:
            dtype_ohe = np.float64
        self.ohe_ = OneHotEncoder(
            sparse_output=True, handle_unknown="ignore", dtype=dtype_ohe
        )

        X_ohe = self.ohe_.fit_transform(
            X_categorical.toarray()
            if sparse.issparse(X_categorical)
            else X_categorical
        )

        if math.isclose(self.median_std_, 0):
            self._X_categorical_minority_encoded = _safe_indexing(
                X_ohe.toarray(), np.flatnonzero(y == class_minority)
            )

        X_ohe.data = (
            np.ones_like(X_ohe.data, dtype=X_ohe.dtype) * self.median_std_ / 2
        )
        X_encoded = sparse.hstack((X_continuous, X_ohe), format="csr").toarray()

        if y_num is not None:
            X_resampled, y_resampled, y_numeric_resampled = super()._fit_resample(X_encoded, y, y_num)
        else:
            X_resampled, y_resampled = super()._fit_resample(X_encoded, y)

        X_res_cat = X_resampled[:, self.continuous_features_.size:]
        if sparse.issparse(X_res_cat):
            X_res_cat.data = np.ones_like(X_res_cat.data)
        if not sparse.issparse(X_res_cat):
            X_res_cat = sparse.csr_matrix(X_res_cat)
        X_res_cat.data = np.ones_like(X_res_cat.data)
        X_res_cat_dec = self.ohe_.inverse_transform(X_res_cat)

        if sparse.issparse(X):
            X_resampled = sparse.hstack(
                (
                    X_resampled[:, : self.continuous_features_.size],
                    X_res_cat_dec,
                ),
                format="csr",
            )
        else:
            X_resampled = np.hstack(
                (
                    X_resampled[:, : self.continuous_features_.size],
                    X_res_cat_dec,
                )
            )

        indices_reordered = np.argsort(
            np.hstack((self.continuous_features_, self.categorical_features_))
        )
        if sparse.issparse(X_resampled):
            
            col_indices = X_resampled.indices.copy()
            for idx, col_idx in enumerate(indices_reordered):
                mask = X_resampled.indices == col_idx
                col_indices[mask] = idx
            X_resampled.indices = col_indices
        else:
            X_resampled = X_resampled[:, indices_reordered]
   
        if y_num is not None:
            return X_resampled, y_resampled, y_numeric_resampled
        else:
            return X_resampled, y_resampled

    def _generate_samples(self, X, nn_data, nn_num, rows, cols, steps, y_num = None):
        
        rng = check_random_state(self.random_state)
         
        X_new = super()._generate_samples(
            X, nn_data, nn_num, rows, cols, steps, y_num = None  )
        
        X_new = (X_new.tolil() if sparse.issparse(X_new) else X_new)

        nn_data = (nn_data.toarray() if sparse.issparse(nn_data) else nn_data)

        if math.isclose(self.median_std_, 0):
            nn_data[:, self.continuous_features_.size:] = (
                self._X_categorical_minority_encoded
            )

        all_neighbors = nn_data[nn_num[rows]]
              
        categories_size = [self.continuous_features_.size] + [
            cat.size for cat in self.ohe_.categories_
        ]

        for start_idx, end_idx in zip(np.cumsum(categories_size)[:-1],
                                      np.cumsum(categories_size)[1:]):
            col_maxs = all_neighbors[:, :, start_idx:end_idx].sum(axis=1)
            
            is_max = np.isclose(col_maxs, col_maxs.max(axis=1, keepdims=True))
            max_idxs = rng.permutation(np.argwhere(is_max))
            xs, idx_sels = np.unique(max_idxs[:, 0], return_index=True)
            col_sels = max_idxs[idx_sels, 1]

            ys = start_idx + col_sels
            X_new[:, start_idx:end_idx] = 0
            X_new[xs, ys] = 1

        return X_new        