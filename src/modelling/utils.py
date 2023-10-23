import pickle
from typing import Any

import numpy as np
from sklearn.metrics import mean_squared_error

CATEGORICAL_COLS = ["Sex"]
DROP_COLS = ["Age", "Rings"]


from prefect import task


@task(name="Load pickle")
def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        loaded_obj = pickle.load(f)
    return loaded_obj


@task(name="Save pickle")
def save_pickle(path: str, obj: Any) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)
