from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import mlflow.tensorflow
from pydantic import BaseModel

import os
import time
import mlflow
from mlflow.artifacts import download_artifacts
from mlflow.client import MlflowClient
from dotenv import load_dotenv

from utils import predict_image


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
    
    global models
                
    print("INFO:\tDownloading models...")
        
    try:
        
        start_dt = time.time()

        mlflow.set_tracking_uri(os.getenv("MLFOW_TRACKING_SERVER_URL"))
        mlflow.set_experiment(os.getenv("EXPERIMENT_RUN_NAME"))
        
        run_ids = [
            os.getenv("MLFLOW_BEST_RESNET50V2_RUNID"), 
            os.getenv("MLFLOW_BEST_CONVNEXTTINY_RUNID")
        ]
        
        client = MlflowClient()
        
        models = {
            '1': None,
            '2': None,
        }
        
        for i, x in enumerate(run_ids):
            dst_path = f"./models/model{x[:5]}"
            print(f"INFO:\t {client.list_artifacts(run_id=x)}")
            print(f"INFO:\t Downloading artifacts for run {x} to {dst_path}")
            
            path = download_artifacts(run_id=x, artifact_path="model", dst_path=dst_path)
            models[str(i+1)] = mlflow.tensorflow.load_model(path)
                
        download_t = time.time() - start_dt 
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))        
        
    print(f"INFO:\tSuccessfully downloaded the models | {download_t:.2f} seconds")


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
        raise HTTPException(status_code=401, detail="Unauthorized Access :p")

    try:
        pred, acc_score = predict_image(request.image_data, models[request.model_selection])
        
        return {
            "pred": pred,
            "acc_score": f"{acc_score:.2f}%",
        }

    except Exception as e:
        print("ERROR:\t", e)
        raise HTTPException(status_code=500, detail=str(e))