from typing import List

import pandas as pd
from config import CATEGORICAL_COLS, DROP_COLS
from loguru import logger
from prefect import flow, task
from sklearn.preprocessing import StandardScaler


@task
def compute_target(df: pd.DataFrame) -> pd.DataFrame:
    # Compute target
    df["Age"] = df["Rings"] + 1.5
    return df


@task
def encode_categorical_cols(df: pd.DataFrame, categorical_cols: List[str] = None) -> pd.DataFrame:
    # Encode categorical columns
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_COLS

    # print("here")
    # print(df.columns)
    # print(CATEGORICAL_COLS)
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    # print(df.columns)
    # print(df)

    return df


@task
def scale(X: pd.DataFrame) -> pd.DataFrame:
    # StandardScaler
    X_scaled = StandardScaler().fit_transform(X)
    df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    return df_scaled


@task
def extract_x_y(df: pd.DataFrame, no_target=False) -> dict:
    # Extract X and y
    if no_target == True:
        X, y = df.drop(DROP_COLS, axis=1), None
    else:
        X, y = df.drop(DROP_COLS, axis=1).drop(["Age"], axis=1), df["Age"]
    return X, y


@flow(name="Preprocess data")
def process_data(filepath: str, for_training: bool):
    df = pd.read_csv(filepath)

    if for_training:
        logger.debug(f"{filepath} | Computing target...")
        df1 = compute_target(df)
        logger.debug(f"{filepath} | Encoding categorical columns...")
        df2 = encode_categorical_cols(df1)
        logger.debug(f"{filepath} | Scaling the features...")
        df3 = scale(df2)
        logger.debug(f"{filepath} | Extracting X and y...")
        X, y = extract_x_y(df3)
    else:
        logger.debug(f"{filepath} | Encoding categorical columns...")
        df1 = encode_categorical_cols(df)
        logger.debug(f"{filepath} | Scaling the features...")
        df2 = scale(df1)
        logger.debug(f"{filepath} | Extracting X and y...")
        X, y = extract_x_y(df2, no_target=True)

    return X, y
