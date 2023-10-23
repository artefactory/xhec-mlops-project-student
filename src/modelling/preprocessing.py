from typing import List

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import CATEGORICAL_COLS, DROP_COLS

CATEGORICAL_COLS = CATEGORICAL_COLS


def compute_target(df: pd.DataFrame) -> pd.DataFrame:
    # Compute target
    df["Age"] = df["Rings"] + 1.5
    return df


def encode_categorical_cols(df: pd.DataFrame, categorical_cols: List[str] = None) -> pd.DataFrame:
    # Encode categorical columns
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_COLS
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df

def scale(X: pd.DataFrame) -> pd.DataFrame:
    # StandardScaler
    X_scaled = StandardScaler().fit_transform(X)
    return X_scaled

def extract_x_y(df: pd.DataFrame) -> dict:
    # Extract X and y
    X, y = df.drop(DROP_COLS, axis=1), df["Age"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
