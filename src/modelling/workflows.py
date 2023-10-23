import os
from typing import Optional

import numpy as np
from loguru import logger
from modeling import evaluate_model, predict, train_model
from prefect import flow
from preprocessing import process_data
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from utils import load_pickle, save_pickle


@flow(name="Train model")
def train_model_workflow(
    data_filepath: str,
    artifacts_filepath: Optional[str] = None,
) -> dict:
    """Train a model and save it to a file"""
    logger.info("Processing training data...")
    X, y = process_data(data_filepath, for_training=True)
    logger.debug(f"{data_filepath} | Splitting dataset in train and test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logger.info("Training model...")
    model = train_model(X_train, y_train)

    logger.info("Making predictions and evaluating...")
    y_pred = predict(X_test, model)
    rmse = evaluate_model(y_test, y_pred)

    if artifacts_filepath is not None:
        logger.info(f"Saving artifacts to {artifacts_filepath}...")
        save_pickle(os.path.join(artifacts_filepath, "model.pkl"), model)

    return {"model": model, "rmse": rmse}


@flow(name="Batch predict")
def batch_predict_workflow(
    input_filepath: str,
    model: Optional[LinearRegression] = None,
    artifacts_filepath: Optional[str] = None,
) -> np.ndarray:
    """Make predictions on a new dataset"""
    if model is None:
        model = load_pickle(os.path.join(artifacts_filepath, "model.pkl"))
    X, _ = process_data(filepath=input_filepath, for_training=False)
    y_pred = predict(X, model)
    return y_pred


if __name__ == "__main__":
    from config import DATA_DIRPATH, MODELS_DIRPATH

    train_model_workflow(
        data_filepath=os.path.join(DATA_DIRPATH, "abalone.csv"),
        artifacts_filepath=MODELS_DIRPATH,
    )
    batch_predict_workflow(
        input_filepath=os.path.join(
            DATA_DIRPATH, "abalone.csv"
        ),  # in reality another file would be used here
        artifacts_filepath=MODELS_DIRPATH,
    )
