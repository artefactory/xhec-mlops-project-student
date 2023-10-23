# Import required modules and variables from app_config, lib, and utils
import numpy as np  # Import numpy as np for return type
from app_config import APP_DESCRIPTION, APP_TITLE, APP_VERSION, PATH_TO_MODEL
from fastapi import FastAPI
from lib.inference import run_inference
from lib.models import InputData, PredictionOut
from utils import load_model

# Create a FastAPI instance with title, description, and version
app = FastAPI(title=APP_TITLE, description=APP_DESCRIPTION, version=APP_VERSION)

# Define a route for the root endpoint
@app.get("/")
def home() -> dict:
    return {"health_check": "App up and running!"}


# Define a route for making predictions
@app.post("/predict", response_model=PredictionOut, status_code=201)
def predict(payload: InputData) -> dict:
    """Make predictions using the loaded model and input data.

    Args:
        payload (InputData): Input data for making predictions.

    Returns:
        dict: A dictionary containing the predicted age.

    Example:
        {
            "Age": 10.5
        }
    """
    # Load the model using the specified path
    model = load_model(PATH_TO_MODEL)

    # Run inference using the loaded model and input data
    y = run_inference([payload], model)

    # Extract the predicted age from the inference result
    predicted_age = y[0].item() if isinstance(y[0], np.generic) else y[0]

    # Return the predicted age in the response
    return {"Age": predicted_age}
