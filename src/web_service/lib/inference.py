from typing import List

import numpy as np
import pandas as pd
from lib.models import InputData
from lib.preprocessing import scale
from loguru import logger
from sklearn.base import BaseEstimator


def run_inference(input_data: List[InputData], model: BaseEstimator) -> np.ndarray:
    """Run inference on a list of input data.

    Args:
        input_data (List[InputData]): List of input data points.
        model (BaseEstimator): The fitted model object.

    Returns:
        np.ndarray: Predicted abalone ages in years.

    Example InputData:
        [InputData(Sex='M', Length=0.388,
        Diameter=0.215, Height=0.085,
        Whole_weight=0.4990, Shucked_weight=0.2745,
        Viscera_weight=0.1330, Shell_weight=0.180)]
    """
    # Log that the inference is being run
    logger.info(f"Running inference on:\n{input_data}")

    # Convert InputData to a DataFrame and rename columns
    df = pd.DataFrame([x.dict() for x in input_data])
    df = df.rename(
        columns={
            "Whole_weight": "Whole weight",
            "Shucked_weight": "Shucked weight",
            "Viscera_weight": "Viscera weight",
            "Shell_weight": "Shell weight",
        }
    )

    # Encode the 'Sex' column as binary columns 'Sex_I' and 'Sex_M'
    # Note :
    # we did not use the regular get_dummies or label encoder because
    # it would not create two but one column in a 1 row df as the input
    df["Sex_I"] = df["Sex"].apply(lambda x: True if x == "I" else False)
    df["Sex_M"] = df["Sex"].apply(lambda x: True if x == "M" else False)

    # Drop the original 'Sex' column
    df = df.drop(["Sex"], axis=1)

    # Scale the data
    X = scale(df)

    # Make predictions using the model
    y = model.predict(X)

    # Log the predicted abalone ages
    logger.info(f"Predicted abalone ages:\n{y}")

    return y
