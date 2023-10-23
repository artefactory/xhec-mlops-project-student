import numpy as np
import pandas as pd
import scipy.sparse
from prefect import flow, task
from loguru import logger
from typing import Dict, List, Tuple
from sklearn.preprocessing import OneHotEncoder, StandardScaler

@task
def compute_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the age of the abalone based on its Rings value and update the DataFrame
    Parameters:
    - df (pd.DataFrame): Input DataFrame containing a 'Rings' column, representing the number of rings of the abalone (which can be used to determine its age)
    Returns:
    - pd.DataFrame: The modified DataFrame with a new 'age' column and the 'Rings' column removed
    Notes:
    - The age of the abalone is calculated as the number of rings plus 1.5 (as per common knowledge about the species)
    - The 'Rings' column is dropped from the DataFrame after calculating the age
    """
    df["age"] = df["Rings"] + 1.5
    df.drop(columns=["Rings"], inplace=True)
    return df

@task
def filter_outliers(
    df: pd.DataFrame,
    filters: Dict[str, List[Tuple[float, float, int]]] = {
        "Viscera weight": [(0.5, ">", 20), (0.5, "<", 25)],
        "Shell weight": [(0.6, ">", 25), (0.8, "<", 25)],
        "Shucked weight": [(1, ">=", 20), (1, "<", 20)],
        "Whole weight": [(2.5, ">=", 25), (2.5, "<", 25)],
        "Diameter": [(0.1, "<", 5), (0.6, "<", 25), (0.6, ">=", 25)],
        "Height": [(0.4, ">", 15), (0.4, "<", 25)],
        "Length": [(0.1, "<", 5), (0.8, "<", 25), (0.8, ">=", 25)],
    },
    ) -> pd.DataFrame:
    """
    Filter outliers from the dataframe based on given conditions
    Parameters:
    - df (pd.DataFrame): Input dataframe with various features including age
    - filters (Dict[str, List[Tuple[float, float, int]]]): Dictionary containing conditions to filter
      outliers. The key is the column name and the value is a list of tuples. Each tuple contains three
      values: the threshold for the column, the comparison direction, and the age limit
    Returns:
    - pd.DataFrame: DataFrame after filtering outliers
    """
    for column, conditions in filters.items():
        for threshold, comparison, age_limit in conditions:
            if comparison == ">":
                df.drop(df[(df[column] > threshold) & (df["age"] < age_limit)].index, inplace=True)
            elif comparison == "<":
                df.drop(df[(df[column] < threshold) & (df["age"] > age_limit)].index, inplace=True)
            elif comparison == ">=":
                df.drop(df[(df[column] >= threshold) & (df["age"] < age_limit)].index, inplace=True)
            elif comparison == "<=":
                df.drop(df[(df[column] <= threshold) & (df["age"] > age_limit)].index, inplace=True)
    return df

@task
def encode_cols(df: pd.DataFrame, 
                categorical_cols: List[str] = ["Sex"]
                ) -> pd.DataFrame:
    """
    Transform the given DataFrame by one-hot encoding specified categorical columns and
    standardizing the remaining numerical columns
    Parameters:
    - df (pd.DataFrame): The input DataFrame to be transformed
    - categorical_cols (List[str], optional): List of categorical column names to be one-hot encoded.
      Defaults to ['sex']
    Returns:
    - pd.DataFrame: Transformed DataFrame with one-hot encoded categorical columns and standardized numerical columns
    """
    # Initialize an empty DataFrame to hold the transformed data
    df_transformed = pd.DataFrame()

    # One-hot encode the specified categorical columns
    if categorical_cols:
        encoder = OneHotEncoder(drop="first")
        cat_encoded = encoder.fit_transform(df[categorical_cols])
        # Convert the encoded sparse matrix to a DataFrame with appropriate column names
        cat_encoded_df = pd.DataFrame(
            cat_encoded.toarray(), columns=encoder.get_feature_names_out(categorical_cols)
        )
        df_transformed = pd.concat([df_transformed, cat_encoded_df], axis=1)

    # Extract the numerical columns by excluding specified categorical columns
    numerical_cols = df.columns.difference(categorical_cols).tolist()

    # Standardize the numerical columns
    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(df[numerical_cols])

    # Convert the scaled array to a DataFrame with appropriate column names
    numeric_scaled_df = pd.DataFrame(numeric_scaled, columns=numerical_cols)
    df_transformed = pd.concat([df_transformed, numeric_scaled_df], axis=1)

    return df_transformed

@flow(name="Preprocess data")
def process_data(filepath: str, 
                 with_target: bool = True
                 ) -> Tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix]:
    """
    Load and preprocess data from a Parquet file.

    Parameters:
    - filepath (str): The path to the Parquet file containing the data to be preprocessed.
    - with_target (bool, optional): Indicates whether the data should be preprocessed with the target (age of abalones). By default, this option is enabled (True).

    Returns:
    - Tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix]: A tuple containing two SciPy CSR matrices. The first is the preprocessed X matrix, and the second is the preprocessed y (target) matrix. These matrices are stored in the SciPy CSR format, suitable for training models.

    Notes:
    - This function loads data from a Parquet file, performs preprocessing operations such as filtering outliers, encoding categorical columns, and computing the target (age of abalones).
    - Preprocessing is performed based on the value of the `with_target` argument. If `with_target` is True, the target (age) is calculated and included in the preprocessed data.
    """
    df = pd.read_csv(filepath)

    if with_target :
        logger.debug(f"{filepath} | Computing target...")
        df1 = compute_target(df)
        logger.debug(f"{filepath} | Filtering outliers...")
        df2 = filter_outliers(df1)
        logger.debug(f"{filepath} | Encoding categorical columns...")
        df3 = encode_cols(df2)
        logger.debug(f"{filepath} | Extracting X and y...")
        y = df3["age"]
        X = df3.drop(columns=["age"])
        return X, y
    
    else :
        logger.debug(f"{filepath} | Encoding categorical columns...")
        df1 = encode_cols(df)
        X = df1.drop(columns=["Rings"])
        logger.debug(f"{filepath} | Extracting X and y...")
        y = []
        return X, y