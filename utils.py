import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def mae_threshold(y_true, y_pred, threshold=2.5):
    mae = mean_absolute_error(y_true, y_pred)
    within_threshold = np.mean(np.abs(y_true - y_pred) <= threshold) * 100
    return mae, within_threshold

def rmse_threshold(y_true, y_pred, threshold=2.5):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    within_threshold = np.mean(np.abs(y_true - y_pred) <= threshold) * 100
    return rmse, within_threshold

def r2_threshold(y_true, y_pred, threshold=2.5):
    r2 = r2_score(y_true, y_pred)
    within_threshold = np.mean(np.abs(y_true - y_pred) <= threshold) * 100
    return r2, within_threshold

__all__ = ['mae_threshold', 'rmse_threshold', 'r2_threshold']