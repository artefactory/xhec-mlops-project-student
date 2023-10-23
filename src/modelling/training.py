import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def train_model(X: pd.DataFrame, y: pd.DataFrame) -> np.array:
    """Train the model and save it."""
    model = LinearRegression()
    return model.fit(X, y)
