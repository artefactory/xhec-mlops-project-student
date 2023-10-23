import numpy as np
import pandas as pd
from prefect import task
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

@task(name="Make predictions")
def predict(X: pd.DataFrame, 
            model: XGBRegressor) -> np.ndarray:
    """
    Predict the target values using the provided XGBRegressor model
    Parameters:
    - X (pd.DataFrame): The input features for prediction
    - model (XGBRegressor): The trained XGBRegressor model
    Returns:
    - np.ndarray: The predicted target values
    """
    return model.predict(X)

@task(name="Evaluate model")
def evaluate_model(y_true: np.ndarray, 
                   y_pred: np.ndarray
                   ) -> float:
    """
    Evaluate the performance of a regression model using the Root Mean Squared Error (RMSE)
    Parameters:
    - y_true (np.ndarray): True target values
    - y_pred (np.ndarray): Predicted target values by the model
    Returns:
    - float: The RMSE value of the model's predictions
    """
    return mean_squared_error(y_true, y_pred, squared=False)