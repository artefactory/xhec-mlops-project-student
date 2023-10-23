from app_config import (
    APP_DESCRIPTION,
    APP_TITLE,
    APP_VERSION,
    MODEL_VERSION,
    PATH_TO_MODEL,
    PATH_TO_PREPROCESSOR,
)
from fastapi import FastAPI

from lib.models import InputData, PredictionOut
from utils import load_objects, load_preprocessor

app = FastAPI(title=APP_TITLE, description=APP_DESCRIPTION, version=APP_VERSION)

@app.get("/")
def home():
    return {"health_check": "App up and running!"}


@app.post("/predict", response_model=PredictionOut, status_code=201)
def predict(payload: InputData):
    dv = load_preprocessor(PATH_TO_PREPROCESSOR)
    model = load_model(PATH_TO_MODEL)
    y = run_inference([payload], dv, model)
    return {"abalone_age_prediction": y[0]}