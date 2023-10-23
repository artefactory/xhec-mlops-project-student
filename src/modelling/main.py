# This module is the training flow: it reads the data, preprocesses it, trains a model and saves it.

import argparse
import os

import pandas as pd
from preprocessing import compute_target, encode_categorical_cols, extract_x_y, scale
from training import train_model
from utils import load_pickle, save_pickle


def main(trainset_path, select_train=True) -> None:
    """Train a model using the data at the given path and save the model (pickle)."""
    # Read data
    df = pd.read_csv(trainset_path)

    # Preprocess data
    df = compute_target(df)
    df = encode_categorical_cols(df)
    df = scale(df)
    X_train, X_test, y_train, y_test = extract_x_y(df)

    # Directory paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the directory containing the current script (your current file)
    model_save_path = os.path.join(
        os.path.dirname(os.path.dirname(script_dir), "web_dev", "loaded_model.pkl")
    )  # Save to the "web_dev" directory

    # Train the model and save it
    if select_train:
        # Train model
        model = train_model(X_train, y_train)
        save_pickle(model_save_path, model)

    else:
        model_load_path = os.path.join(script_dir, "training")
        loaded_model = load_pickle(model_load_path)
        save_pickle(model_save_path, loaded_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model using the data at the given path.")
    parser.add_argument("trainset_path", type=str, help="Path to the training set")
    args = parser.parse_args()
    main(args.trainset_path)
