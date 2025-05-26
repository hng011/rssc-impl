from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import os
from dotenv import load_dotenv

from utils import (
    load_model, 
    predict_image
)


load_dotenv()
PROD = os.getenv("PROD").lower()
API_AUTH = os.getenv("API_AUTH")
VERSION = os.getenv("VERSION")


app = FastAPI(title="Remote Sensing Scene Classification Implementation", version=VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins       = os.getenv("ALLOW_ORIGINS"),
    allow_credentials   = os.getenv("ALLOW_CREDENTIALS"),
    allow_methods       = os.getenv("ALLOW_METHODS"),
    allow_headers       = os.getenv("ALLOW_HEADERS"),
)


# === REQUEST SCHEMA ===
class PredictRequest(BaseModel):
    model_selection: str
    image_data: str  # base64
    api_auth: str


# === ROUTES ===
@app.get("/")
async def root():
    return {"status": "success", "msg": "ready for inference", "version": VERSION}


@app.post("/predict")
def predict(request: PredictRequest):
    if API_AUTH and request.api_auth != API_AUTH:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        model = load_model(request.model_selection, prod=PROD)

        pred, acc_score, infer_time = predict_image(request.image_data, model)

        return {
            "pred": pred,
            "acc_score": f"{acc_score:.2f}%",
            "infer_time": f"{infer_time:.2f} Seconds"
        }

    except Exception as e:
        print("ERROR:\t", e)
        raise HTTPException(status_code=500, detail=str(e))