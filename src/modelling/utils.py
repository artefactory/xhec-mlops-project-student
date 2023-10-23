# Use this module to code a `pickle_object` function.
# This will be useful to pickle the model
# (and encoder if need be).
import pickle
from typing import Any

import numpy as np
from sklearn.metrics import mean_squared_error

CATEGORICAL_COLS = ["Sexe"]
DROP_COLS = ["Age", "Rings"]


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        loaded_obj = pickle.load(f)
    return loaded_obj


def save_pickle(path: str, obj: Any) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate mean squared error for two arrays."""
    return mean_squared_error(y_true, y_pred, squared=False)
