# Script path: functions/dist_metrics.py

# This script is part of the 'imbalance-metrics' package, which was developed by: 
# imbalance-metrics: A Python package for evaluating imbalanced datasets (version 0.1.6). Available via PyPI (2023)

# It has been adapted to incorporate the relevance function and control points calculated based on adjusted boxplot statistics, rather than the original boxplot statistics, used by the original developer, to better handle the imbalanced regression problem.
# It has also been implemented the DenseWeight weighted root mean squared error and the DenseWeight squared error-relevance areas (SERA) between true and predicted values.
# This results from the adaptation of the original script to include the DenseWeight method, which is a new approach for handling imbalanced regression problems.
# This was originally developed for this study.

# This script contains functions that:
# 1. Calculate the phi value for each element of 'y'.
# 2. Calculate the phi-weighted R^2 score.
# 3. Calculate the phi-weighted mean squared error (MSE).
# 4. Calculate the phi-weighted mean absolute error (MAE).
# 5. Calculate the phi-weighted root mean squared error (RMSE).
# 6. Process series for error calculation.
# 7. Calculate squared error-relevance values at a given threshold.
# 8. Calculate the squared error-relevance areas (SERA) between true and predicted value (phi-weighted).
# 9. Calculate the DenseWeight weighted root mean squared error.
# 10. Calculate the DenseWeight squared error-relevance areas (SERA) between true and predicted values.


## load dependencies - third party
from sklearn.metrics import mean_squared_error, mean_absolute_error , r2_score
import pandas as pd
import numpy as np
from denseweight import DenseWeight


## load dependencies - internal
from functions import relevance_function,relevance_function_ctrl_pts


def calculate_phi(y, method = "auto", xtrm_type = "both", coef = 1.5, ctrl_pts = None):

    """
    Calculates the phi value for each element of 'y'.
    
    Parameters
    ----------
    y : array-like
        Input data for which phi value needs to be calculated.
    method : str, optional (default='auto')
        Relevance method. Either 'auto' or 'manual'.
    xtrm_type : str, optional (default='both')
        Distribution focus. Either 'high', 'low', or 'both'.
    coef : float, optional (default=1.5)
        Coefficient for box plot (pos real)
    ctrl_pts : list, optional (default=None)
        Input for "manual" rel method  (2d array).
    
    Returns
    -------
    y_phi : array-like
        Phi values for each element of 'y'.
    
    """
    
    if not isinstance(y, pd.DataFrame):
        y = pd.core.series.Series(y)
    try:    
        phi_params = relevance_function_ctrl_pts.phi_ctrl_pts(y = y,method = method, xtrm_type = xtrm_type, coef = coef, ctrl_pts = ctrl_pts)
    except Exception as e:
        raise Exception(e)
    else:
        try: 
            y_phi = relevance_function.phi(y = y,ctrl_pts = phi_params)
        except Exception as e:
            raise Exception(e)
        else:
            return y_phi
        

def phi_weighted_r2(y, y_pred, method = "auto", xtrm_type = "both", coef = 1.5, ctrl_pts = None):

    """
    Calculates the R^2 score between 'y' and 'y_pred' with weighting by phi.
    
    Parameters
    ----------
    y : array-like
        True target values.
    y_pred : array-like
        Predicted target values.
    method : str, optional (default='auto')
        Relevance method. Either 'auto' or 'manual'.
    xtrm_type : str, optional (default='both')
        Distribution focus. Either 'high', 'low', or 'both'.
    coef : float, optional (default=1.5)
        Coefficient for box plot (pos real)
    ctrl_pts : list, optional (default=None)
        Input for "manual" rel method  (2d array).

    Returns
    -------
    r2 : float
        Phi weighted R^2 score.
    
    """

    y_phi=calculate_phi(y, method, xtrm_type , coef, ctrl_pts)
    return r2_score(y, y_pred, sample_weight=y_phi)


def phi_weighted_mse(y, y_pred, method = "auto", xtrm_type = "both", coef = 1.5, ctrl_pts = None):

    """
    Calculates the mean squared error between 'y' and 'y_pred' with weighting by phi.
    
    Parameters
    ----------
    y : array-like
        True target values.
    y_pred : array-like
        Predicted target values.
    method : str, optional (default='auto')
        Relevance method. Either 'auto' or 'manual'.
    xtrm_type : str, optional (default='both')
        Distribution focus. Either 'high', 'low', or 'both'.
    coef : float, optional (default=1.5)
        Coefficient for box plot (pos real)
    ctrl_pts : list, optional (default=None)
        Input for "manual" rel method  (2d array).
        
    Returns
    -------
    mse : float
        Phi weighted Mean squared error.
    
    """

    y_phi=calculate_phi(y, method, xtrm_type , coef, ctrl_pts)
    return mean_squared_error(y, y_pred, sample_weight=y_phi)


def phi_weighted_mae(y, y_pred, method = "auto", xtrm_type = "both", coef = 1.5, ctrl_pts = None):

    """
    Calculates the mean absolute error between 'y' and 'y_pred' with weighting by phi.
    
    Parameters
    ----------
    y : array-like
        True target values.
    y_pred : array-like
        Predicted target values.
    method : str, optional (default='auto')
        Relevance method. Either 'auto' or 'manual'.
    xtrm_type : str, optional (default='both')
        Distribution focus. Either 'high', 'low', or 'both'.
    coef : float, optional (default=1.5)
        Coefficient for box plot (pos real)
    ctrl_pts : list, optional (default=None)
        Input for "manual" rel method  (2d array).
        
    Returns
    -------
    phi_weighted_mae : float
       Phi weighted Mean absolute error.
    
    """

    y_phi=calculate_phi(y, method, xtrm_type , coef, ctrl_pts)
    return mean_absolute_error(y, y_pred, sample_weight=y_phi)


def phi_weighted_root_mse(y, y_pred, method = "auto", xtrm_type = "both", coef = 1.5, ctrl_pts = None):

    """
    Calculates the root mean squared error between 'y' and 'y_pred' with weighting by phi.
    
    Parameters
    ----------
    y : array-like
        True target values.
    y_pred : array-like
        Predicted target values.
    method : str, optional (default='auto')
        Relevance method. Either 'auto' or 'manual'.
    xtrm_type : str, optional (default='both')
        Distribution focus. Either 'high', 'low', or 'both'.
    coef : float, optional (default=1.5)
        Coefficient for box plot (pos real)
    ctrl_pts : list, optional (default=None)
        Input for "manual" rel method  (2d array).
        
    Returns
    -------
    mae : float
        Phi weighted Root Mean squared error.
    """

    y_phi=calculate_phi(y, method, xtrm_type , coef, ctrl_pts)
    return np.sqrt(mean_squared_error(y, y_pred, sample_weight=y_phi))


def ser_process(trues, preds, method = "auto", xtrm_type = "both", coef = 1.5, ctrl_pts = None):

    """
    Processes the true and predicted values for error calculation.
    Parameters
    ----------
    trues : array-like
        True target values.
    preds : array-like
        Predicted target values.
    method : str, optional (default='auto')
        Relevance method. Either 'auto' or 'manual'.
    xtrm_type : str, optional (default='both')
        Distribution focus. Either 'high', 'low', or 'both'.
    coef : float, optional (default=1.5)
        Coefficient for box plot (pos real)
    ctrl_pts : list, optional (default=None)
        Input for "manual" rel method  (2d array).
    Returns
    -------
    tbl : DataFrame
        A DataFrame containing the true values, their phi values, and the predicted values.
    ms : list
        A list of column names from the DataFrame, excluding the first two columns.
    """
        
    if not isinstance(preds, pd.DataFrame):
        preds = pd.core.series.Series(preds)
    if not isinstance(trues, pd.DataFrame):
        trues = pd.core.series.Series(trues)
    
    trues=trues.reset_index(drop=True)
    phi_trues= calculate_phi(trues, method, xtrm_type , coef, ctrl_pts) 

    #trues = trues.values
    tbl = pd.DataFrame(
        {'trues': trues,
         'phi_trues': phi_trues,
         })
    tbl = pd.concat([tbl, preds], axis=1)
    ms = list(tbl.columns[2:])
    return tbl,ms


def ser_t(y, y_pred, t, method = "auto", xtrm_type = "both", coef = 1.5,  ctrl_pts = None, s=0):

    """
    Calculates the Squared error-relevance values between 'y' and 'y_pred' with weighting by phi at thershold 't'.
    
    Parameters
    ----------
    y : array-like
        True target values.
    y_pred : array-like
        Predicted target values.
    t : float
        Threshold value.
    method : str, optional (default='auto')
        Relevance method. Either 'auto' or 'manual'.
    xtrm_type : str, optional (default='both')
        Distribution focus. Either 'high', 'low', or 'both'.
    coef : float, optional (default=1.5)
        Coefficient for box plot (pos real)
    ctrl_pts : list, optional (default=None)
        Input for "manual" rel method  (2d array).
        
    Returns
    -------
    ser_t : list
        List of Squared error-relevance values at each threshold t.
    """
    tbl,ms= ser_process(y, y_pred, method, xtrm_type , coef , ctrl_pts)

    error = [sum(tbl.apply(lambda x: ((x['trues'] - x[y]) ** 2) if x['phi_trues'] >= t else 0, axis=1)) for y in ms]
    
    if s==1:
        return error
    else:
        return error[0]


def sera (y, y_pred, step = 0.01,return_err = False, method = "auto", xtrm_type = "both", coef = 1.5, ctrl_pts = None, weight= None) :

    """
    Calculates the Squared error-relevance areas (ERA) between y and y_pred.

    Parameters
    ----------
    y : array-like
        True target values.
    y_pred : array-like
        Predicted target values.
    step : float, optional (default=0.001)
        Step size for threshold values
    return_err : bool, optional (default=False)
        Whether to return the error and thershold values with the SERA value.
    method : str, optional (default='auto')
        Relevance method. Either 'auto' or 'manual'.
    xtrm_type : str, optional (default='both')
        Distribution focus. Either 'high', 'low', or 'both'.
    coef : float, optional (default=1.5)
        Coefficient for box plot (pos real)
    ctrl_pts : list, optional (default=None)
        Input for "manual" rel method  (2d array).
        
    Returns
    -------
    sera : float or dict:
        If `return_err` is False, returns the SERA value as a float. If `return_err` is True, returns a dictionary containing the SERA value, the error values, and the thresholds used in the calculation.

    """
    _,ms= ser_process(y, y_pred)
    th = np.arange(0, 1 + step, step)
    errors = []
    for ind in th:
        errors.append(ser_t(y, y_pred, ind, method , xtrm_type , coef , ctrl_pts , s=1))
        

    areas = []
    for x in range(1, len(th)):
        areas.append([step *(errors[x - 1][y] + errors[x][y]) / 2 for y in range(len(ms))])
    areas = pd.DataFrame(data=areas, columns=ms)
    res = areas.apply(lambda x: sum(x))
    if return_err :
       return {"sera":res, "errors":[item for sublist in errors for item in sublist], "thrs" :th}
    else:
       return res.item()
    
# DenseWeight weighted root mean squared error
def denseweight_weighted_root_mse(y, y_pred):

    """
    Calculates the root mean squared error between 'y' and 'y_pred' with weighting by denseweight.
    
    Parameters
    ----------
    y : array-like
        True target values.
    y_pred : array-like
        Predicted target values.
        
    Returns
    -------
    dw-rmse : float
        DenseWeight weighted Root Mean squared error.
    """
    dw = DenseWeight(alpha = 1.0)
    y_dw = dw.fit(y.values.reshape(-1, 1)).flatten()
    return np.sqrt(mean_squared_error(y, y_pred, sample_weight=y_dw))


def ser_process_dw(trues, preds):
    
    """
    Processes the true and predicted values for DenseWeight calculations.
    Parameters
    ----------
    trues : array-like
        True target values.
    preds : array-like
        Predicted target values.
    Returns
    -------
    tbl : DataFrame
        A DataFrame containing the true values, their phi values, and the predicted values.
    ms : list
        A list of column names from the DataFrame, excluding the first two columns.
    """

    if not isinstance(preds, pd.DataFrame):
        preds = pd.Series(preds)
    if not isinstance(trues, pd.DataFrame):
        trues = pd.Series(trues)

    trues = trues.reset_index(drop=True)

    dw = DenseWeight(alpha=1.0)
    dw_trues = dw.fit(trues.values.reshape(-1, 1)).flatten()

    tbl = pd.DataFrame({
        'trues': trues,
        'phi_trues': dw_trues,
    })
    tbl = pd.concat([tbl, preds], axis=1)
    ms = list(tbl.columns[2:])
    return tbl, ms


# DenseWeight squared error-relevance values (DW-SER) between true and predicted values
def ser_t_dw(y, y_pred, t, s=0):
    
    """
    Calculates the squared error-relevance values between 'y' and 'y_pred' with weighting by DenseWeight at threshold 't'.
    Parameters
    ----------
    y : array-like
        True target values.
    y_pred : array-like
        Predicted target values.
    t : float
        Threshold value.
    s : int, optional (default=0)
        If 1, returns a list of errors for each column; if 0, returns the first error value.
    Returns
    -------
    error : list or float
        If `s` is 1, returns a list of squared error-relevance values at each threshold t. If `s` is 0, returns the first squared error-relevance value.
    """

    tbl, ms = ser_process_dw(y, y_pred)

    error = [sum(tbl.apply(lambda x: ((x['trues'] - x[y]) ** 2) if x['phi_trues'] >= t else 0, axis=1)) for y in ms]

    return error if s == 1 else error[0]


# DenseWeight squared error-relevance areas (DW-SERA) between true and predicted values
def sera_dw(y, y_pred, step=0.01, return_err=False):

    """
    Calculates the Squared error-relevance areas (SERA) between y and y_pred with DenseWeight.
    Parameters
    ----------
    y : array-like
        True target values.
    y_pred : array-like

        Predicted target values.
    step : float, optional (default=0.01)
        Step size for threshold values.
    return_err : bool, optional (default=False)
        Whether to return the error and threshold values with the SERA value.
    Returns
    -------
    sera : float or dict:
        If `return_err` is False, returns the SERA value as a float. If `return_err` is True, returns a dictionary containing the SERA value, the error values, and the thresholds used in the calculation.
    """

    _, ms = ser_process_dw(y, y_pred)
    th = np.arange(0, 1 + step, step)
    errors = []
    for ind in th:
        errors.append(ser_t_dw(y, y_pred, ind, s=1))

    areas = []
    for x in range(1, len(th)):
        areas.append([step * (errors[x - 1][y] + errors[x][y]) / 2 for y in range(len(ms))])
    areas = pd.DataFrame(data=areas, columns=ms)
    res = areas.apply(lambda x: sum(x))

    if return_err:
        return {"sera": res, "errors": [item for sublist in errors for item in sublist], "thrs": th}
    else:
        return res.item()