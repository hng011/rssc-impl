from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import os
import io
import time
import tarfile
import requests
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


@app.on_event("startup")
async def startup_event():
    models_path = "./models/"
    if not PROD == "no":
        if not os.path.isdir(models_path):
            os.makedirs(models_path)
            
        if len(os.listdir(models_path)) < 1:
            load_dotenv()
                    
            print("INFO:\tDownloading models...")
            
            try:
                start_t = time.time()
                r = requests.get(os.getenv("MODELS_URL"))
                
                with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz") as tar:
                    tar.extractall(models_path)
                download_t = time.time() - start_t
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))        
            
            print(f"INFO:\tSuccessfully downloaded the models | {download_t:.2f} seconds")
        
        else:
            print("INFO:\tModels ready")


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
        pred, acc_score = predict_image(request.image_data, model)
        
        return {
            "pred": pred,
            "acc_score": f"{acc_score:.2f}%",
        }

    except Exception as e:
        print("ERROR:\t", e)
        raise HTTPException(status_code=500, detail=str(e))