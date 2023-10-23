import numpy as np
import pandas as pd
from prefect import task
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


@task(name="Train model")
def train_model(X: pd.DataFrame, y: pd.DataFrame) -> np.array:
    """Train and return a linear regression model"""
    lr = LinearRegression()
    lr.fit(X, y)
    return lr


@task(name="Make predictions")
def predict(X: pd.DataFrame, model: LinearRegression) -> np.ndarray:
    """Make predictions with a trained model"""
    return model.predict(X)


@task(name="Evaluate model")
def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate mean squared error for two arrays"""
    return mean_squared_error(y_true, y_pred, squared=False)
