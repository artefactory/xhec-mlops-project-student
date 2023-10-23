from typing import List

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator
from sklearn.feature_extraction import DictVectorizer

from lib.models import InputData
from lib.preprocessing import CATEGORICAL_COLS, encode_categorical_cols, scale


def run_inference(input_data: List[InputData], model: BaseEstimator) -> np.ndarray:
    """Run inference on a list of input data.

    Args:
        payload (dict): the data point to run inference on.
        model (BaseEstimator): the fitted model object.

    Returns:
        np.ndarray: the predicted abalone age in years.

    Example Abalone_data:
        {'Sex':M,	'Length':0.388,	'Diameter':0.215,	'Height':0.085,	'Whole weight':0.4990,	'Shucked weight':0.2745,	'Viscera weight':0.1330,	'Shell weight':0.180}
    """
    logger.info(f"Running inference on:\n{input_data}")
    df = pd.DataFrame([x.dict() for x in input_data])
    df = df.rename(columns={"Whole_weight" : "Whole weight",
                            "Shucked_weight" : "Shucked weight",
                            "Viscera_weight" : "Viscera weight",
                            "Shell_weight" : "Shell weight"})
    df = encode_categorical_cols(df, CATEGORICAL_COLS)
    X = scale(df)
    y = model.predict(X)
    logger.info(f"Predicted abalone ages:\n{y}")
    return y
