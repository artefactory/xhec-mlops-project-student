from typing import List

import pandas as pd
from sklearn.preprocessing import StandardScaler

DROP_COLS = ["Rings", "Age"]
CATEGORICAL_COLS = ["Sex"]


def encode_categorical_cols(df: pd.DataFrame, categorical_cols: List[str] = None) -> pd.DataFrame:
    # Encode categorical columns
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_COLS
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df


def scale(X: pd.DataFrame) -> pd.DataFrame:
    # StandardScaler
    X_scaled = StandardScaler().fit_transform(X)
    df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    return df_scaled
