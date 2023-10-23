import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def predict(X: pd.DataFrame, y: pd.DataFrame, model: LinearRegression) -> np.ndarray:
    """Make predictions with a trained model."""
    model = model
    model.fit(X, y)
    return model.predict(X)
