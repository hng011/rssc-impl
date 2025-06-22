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
from transformers import ViTFeatureExtractor


load_dotenv()
PROD = os.getenv("PROD").lower()
API_AUTH = os.getenv("API_AUTH")
VERSION = os.getenv("VERSION")


app = FastAPI(title="API for RSSC with Transformers 4.52.4", version=VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins       = os.getenv("ALLOW_ORIGINS"),
    allow_credentials   = os.getenv("ALLOW_CREDENTIALS"),
    allow_methods       = os.getenv("ALLOW_METHODS"),
    allow_headers       = os.getenv("ALLOW_HEADERS"),
)


@app.on_event("startup")
async def startup_event():
    
    global models, model_names, fe
                
    print("INFO:\tDownloading models...")
        
    try:
        
        start_dt = time.time()

        mlflow.set_tracking_uri(os.getenv("MLFOW_TRACKING_SERVER_URL"))
        mlflow.set_experiment(os.getenv("EXPERIMENT_RUN_NAME"))
        
        run_ids = [
            os.getenv("MLFLOW_BEST_VIT_RUNID"), 
        ]
        
        client = MlflowClient()
        
        models = {
            '3': None,
        }

        model_names = {k: None for k in models.keys()}
        
        for k, x in zip(models, run_ids):
            dst_path = f"./models/model{x[:5]}"
            is_model_ready = os.path.exists(dst_path)
            
            model_atr = client.search_model_versions(f"run_id='{x}'")
            for atr in model_atr:
                model_names[k] = f"{atr.name}_v{atr.version}"

            print(f"INFO:\t Downloading artifacts for {model_names[k]} to {dst_path}")
            
            if not is_model_ready:
                print(f"INFO: Creating {dst_path}")
                path = download_artifacts(run_id=x, artifact_path="model", dst_path=dst_path)
            else:
                print("INFO:\t Models ready :)")
            
            models[k] = mlflow.tensorflow.load_model(f"{dst_path}/model" if is_model_ready else path)
                
        download_t = time.time() - start_dt
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))        
        
    print(f"INFO:\t Successfully downloaded the models | {download_t:.2f} seconds")
    print(f"INFO:\t list model {models}")
    # Download feature extractor
    print("INFO:\tDownloading VIT feature extractor...")
    
    try:
        start_dt = time.time()
        fe = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
        download_t = time.time() - start_dt
        print(f"INFO:\tSuccessfully downloaded the VIT feature extractor | {(download_t):.2f} seconds")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
            

    
# === REQUEST SCHEMA ===
class PredictRequest(BaseModel):
    model_selection: str
    image_data: str  # base64
    api_auth: str


# === ROUTES ===
@app.get("/")
async def root():
    return {"name":app.title, "status": "success", "msg": "ready for inference", "available_model":model_names, "version": app.version}


@app.post("/predict")
def predict(request: PredictRequest):
    if API_AUTH and request.api_auth != API_AUTH:
        raise HTTPException(status_code=401, detail="Unauthorized Access :p")

    try:
        pred, acc_score = predict_image(request.image_data, models[request.model_selection], feature_extractor=fe)
        
        return {
            "pred": pred,
            "acc_score": f"{acc_score:.2f}%",
            "model_name": model_names[request.model_selection]
        }

    except Exception as e:
        print("ERROR:\t", e)
        raise HTTPException(status_code=500, detail=str(e))