import numpy as np
import pandas as pd
from prefect import task
from xgboost import XGBRegressor

@task(name="Train model")
def train_model(X: pd.DataFrame, y: np.ndarray) -> XGBRegressor:
    """
    Train a XGB regressor model with specified parameters on the given data
    Parameters:
    - X (pd.DataFrame): The input features for the model
    - y (np.ndarray): The target variable
    Returns:
    - XGBRegressor: A trained xgb regressor model
    """
    xgb = XGBRegressor()
    xgb.fit(X, y)
    return xgb