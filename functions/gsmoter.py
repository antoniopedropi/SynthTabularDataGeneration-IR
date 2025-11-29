# Script path: functions/gsmoter.py

# Script that implements G-SMOTER logic for generating synthetic samples
# for imbalanced regression tasks, based on the geometric SMOTE approach.
# This script assumes that all features are numeric (e.g., one-hot encoded if categorical).
# It uses the relevance function to determine which samples to augment.
# The function can be used with either numpy arrays or pandas DataFrames.
# It allows for various parameters to control the sampling process, including
# the number of neighbors, relevance threshold, selection strategy, truncation factor,
# and deformation factor.
# The output is either a numpy array of augmented features and targets or a pandas DataFrame.

# The G-SMOTE algorithm was originally proposed by Douzas and Bacao in 2019 (Douzas, G., & Bacao, F. (2019). Geometric SMOTE a geometrically enhanced drop-in replacement for SMOTE. Information Sciences, 501, 118-135.)
# It was then adapted ro regression tasks by Camacho et al. in 2022 (Camacho, Luís & Douzas, Georgios & Bação, Fernando. (2022). Geometric SMOTE for regression. Expert Systems with Applications. 193. 116387. 10.1016/j.eswa.2021.116387.).

# Since the authors of the original G-SMOTER paper did not provide a code implementation, even after sending requests to each author,
# we implemented our own version based on the description in the paper.
# Therefore, this implementation may differ from the original G-SMOTER implementation in some details, but it follows the same principles and logic as described in the paper as best as we could interpret them.
# However, it follows the same principles and logic as described in the paper.
# The implementation is designed to be flexible and can be adapted for various use cases in imbalanced regression tasks.


"""
G-SMOTER: Geometric SMOTE for Imbalanced Regression
Adapted from SMOTER and Geometric SMOTE (Douzas & Bacao 2019)
Assumes all features are numeric (e.g., one-hot encoded if categorical)
"""


## load dependencies - third party
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state


## load dependencies - internal
from functions.relevance_function_ctrl_pts import phi_ctrl_pts
from functions.relevance_function import phi


def remove_nan_rows(X, y):
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    return X[mask], y[mask]


def gsmoter(X=None, y=None, df=None, target_variable=None, proportion=1.0, k=5, relevance_threshold=0.8,
           selection_strategy='minority', truncation_factor=1.0,
           deformation_factor=1.0, random_state=None):

    """
    Generate synthetic samples using G-SMOTER logic (numeric features only).
    Parameters:
        - X: np.ndarray, shape (n_samples, n_features), feature matrix
        - y: np.ndarray, shape (n_samples,), target variable
        - df: pd.DataFrame, optional, DataFrame containing features and target variable
        - target_variable: str, name of the target variable in df
        - proportion: float, proportion of synthetic samples to generate relative to minority class
        - k: int, number of nearest neighbors to consider
        - relevance_threshold: float, threshold for relevance function
        - selection_strategy: str, 'minority', 'majority', or 'combined'
        - truncation_factor: float, factor to truncate the generated samples
        - deformation_factor: float, factor to deform the generated samples
        - random_state: int, seed for reproducibility
    Returns:
        - X_aug: np.ndarray, shape (n_samples + n_synthetic_samples, n_features), augmented feature matrix
        - y_aug: np.ndarray, shape (n_samples + n_synthetic_samples,), augmented target variable
    """

    if df is not None and target_variable is not None:
        y = df[target_variable].values
        X = df.drop(columns=[target_variable]).values
    elif X is None or y is None:
        raise ValueError("Either provide X and y, or df with target_variable")

    random_state = check_random_state(random_state)

    # Discretisation of the domain based on the relevance function
    ctrl_pts = phi_ctrl_pts(y)
    rel = np.array(phi(pd.Series(y), ctrl_pts=ctrl_pts))
    mask_rare = rel >= relevance_threshold
    X_rare, y_rare = X[mask_rare], y[mask_rare]
    X_normal = X[~mask_rare]

    # Calculate the number of synthetic samples to generate
    n_to_sample = int(proportion * len(X_rare))
    X_syn, y_syn = [], []

    # Determine the surface/center points for sampling based on the selection strategy
    for _ in range(n_to_sample):
        i = random_state.randint(len(X_rare))
        x_center, y_center = X_rare[i], y_rare[i]

        if selection_strategy == 'minority':
            nn = NearestNeighbors(n_neighbors=k + 1).fit(X_rare)
            _, idxs = nn.kneighbors([x_center])
            x_surface = X_rare[random_state.choice(idxs[0][1:])]
            y_surface = y_rare[np.where((X_rare == x_surface).all(axis=1))[0][0]]

        elif selection_strategy == 'majority':
            nn = NearestNeighbors(n_neighbors=1).fit(X_normal)
            _, idxs = nn.kneighbors([x_center])
            x_surface = X_normal[idxs[0][0]]
            y_surface = y[~mask_rare][idxs[0][0]]

        else:  # combined
            nn_r = NearestNeighbors(n_neighbors=k + 1).fit(X_rare)
            _, idxs_r = nn_r.kneighbors([x_center])
            x_min = X_rare[random_state.choice(idxs_r[0][1:])]
            y_min = y_rare[np.where((X_rare == x_min).all(axis=1))[0][0]]

            nn_n = NearestNeighbors(n_neighbors=1).fit(X_normal)
            _, idxs_n = nn_n.kneighbors([x_center])
            x_maj = X_normal[idxs_n[0][0]]
            y_maj = y[~mask_rare][idxs_n[0][0]]

            d_min = np.linalg.norm(x_center - x_min)
            d_maj = np.linalg.norm(x_center - x_maj)

            if d_min < d_maj:
                x_surface, y_surface = x_min, y_min
            else:
                x_surface, y_surface = x_maj, y_maj

        # Generate synthetic sample
        direction = (x_surface - x_center)
        norm = np.linalg.norm(x_surface - x_center)

        if norm == 0:
            # If the center and surface points are the same, skip this iteration
            continue
        e = direction / norm

        # Calculate the radius from the center to the surface point
        radius = np.linalg.norm(x_center - x_surface)

        v = random_state.normal(0, 1, size=X.shape[1])
        v_norm = np.linalg.norm(v)
        if v_norm == 0:
            # If the random vector is zero, skip this iteration
            continue

        # Normalize the random vector
        v /= v_norm
        
        # Generate a random sample in the direction of the surface point
        r = random_state.rand() ** (1 / X.shape[1])
        x_gen = r * v

        x_parallel = np.dot(x_gen, e) * e
        x_perp = x_gen - x_parallel

        x_proj_scalar = np.dot(x_gen, e)

        # Truncate the generated sample if necessary (Truncate function)
        if abs(x_proj_scalar - truncation_factor) > 1:
            x_gen = x_gen - 2 * x_parallel

        # Deform the generated sample if necessary (Deformation function)
        x_gen = x_gen - deformation_factor * x_perp
        x_new = x_center + radius * x_gen

        # Calculate the new target value based on the distances to the center and surface points
        d1 = np.linalg.norm(x_new - x_center)
        d2 = np.linalg.norm(x_new - x_surface)
        y_new = (d2 * y_center + d1 * y_surface) / (d1 + d2) if (d1 + d2) != 0 else (y_center + y_surface) / 2

        X_syn.append(x_new)
        y_syn.append(y_new)

    X_aug = np.vstack([X, np.array(X_syn)])
    y_aug = np.concatenate([y, np.array(y_syn)])

    X_aug, y_aug = remove_nan_rows(X_aug, y_aug)

    if df is not None and target_variable is not None:
        feature_columns = df.drop(columns=[target_variable]).columns
        df_aug = pd.DataFrame(X_aug, columns=feature_columns)
        df_aug[target_variable] = y_aug
        return df_aug

    return X_aug, y_aug
