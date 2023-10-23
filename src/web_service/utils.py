import os
import pickle
from functools import lru_cache
from typing import Any

from loguru import logger


# Function to load a preprocessor object from a file
@lru_cache
def load_preprocessor(filepath: os.PathLike) -> Any:
    """Load a preprocessor object from a file.

    Args:
        filepath (os.PathLike): The path to the file containing the preprocessor object.

    Returns:
        Any: The loaded preprocessor object.

    Note:
        This function uses caching with `@lru_cache` to improve performance.
    """
    logger.info(f"Loading preprocessor from {filepath}")
    with open(filepath, "rb") as f:
        return pickle.load(f)


# Function to load a machine learning model from a file
@lru_cache
def load_model(filepath: os.PathLike) -> Any:
    """Load a machine learning model from a file.

    Args:
        filepath (os.PathLike): The path to the file containing the machine learning model.

    Returns:
        Any: The loaded machine learning model.

    Note:
        This function uses caching with `@lru_cache` to improve performance.
    """
    logger.info(f"Loading model from {filepath}")
    with open(filepath, "rb") as f:
        return pickle.load(f)
