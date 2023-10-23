from app_config import (
    APP_DESCRIPTION,
    APP_TITLE,
    APP_VERSION,
    PATH_TO_MODEL
)

from fastapi import FastAPI

from lib.models import InputData, PredictionOut
from utils import load_model
from lib.inference import run_inference
import numpy as np

app = FastAPI(title=APP_TITLE, description=APP_DESCRIPTION, version=APP_VERSION)

@app.get("/")
def home():
    return {"health_check": "App up and running!"}

@app.post("/predict", response_model=PredictionOut, status_code=201)
def predict(payload: InputData):
    model = load_model(PATH_TO_MODEL)
    y = run_inference([payload], model)
    predicted_age = y[0].item() if isinstance(y[0], np.generic) else y[0]
    age_dict = {"predicted_age": predicted_age}
    return {"Age": age_dict}