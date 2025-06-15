from PIL import Image
import tensorflow as tf
import numpy as np

import base64
import io
import os


CLASSES = ['airplane', 'airport', 'baseball_diamond', 'basketball_court', 'beach', 'bridge', 'chaparral', 'church', 'circular_farmland', 'cloud', 'commercial_area', 'dense_residential', 'desert', 'forest', 'freeway', 'golf_course', 'ground_track_field', 'harbor', 'industrial_area', 'intersection', 'island', 'lake', 'meadow', 'medium_residential', 'mobile_home_park', 'mountain', 'overpass', 'palace', 'parking_lot', 'railway', 'railway_station', 'rectangular_farmland', 'river', 'roundabout', 'runway', 'sea_ice', 'ship', 'snowberg', 'sparse_residential', 'stadium', 'storage_tank', 'tennis_court', 'terrace', 'thermal_power_station', 'wetland']


model_names = {
    '1':"best_resnet50v2_rssc.keras",
    '2':None, # ViT
    '3':"best_convnext-tiny_rssc.keras", # ConvNeXt-Tiny
}


def load_model(model_selection: str, prod: str):
    model_dir = None    
    
    if prod.lower() == "no":
        model_dir = os.path.join("../../../outputs/models/", model_names[model_selection])
    
    else:            
        model_dir = f"./models/{model_names[model_selection]}"
    
    model = tf.keras.models.load_model(model_dir)
    
    return model


def preprocess_image(image_data_b64):
    decoded = base64.b64decode(image_data_b64)
    image = Image.open(io.BytesIO(decoded)).convert("RGB")
    image = image.resize((224, 224))  
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)  # add batch dim
    img_array = img_array / 255.0
    
    return img_array


def predict_image(image_data_b64, model):
    img_array = preprocess_image(image_data_b64)
    preds = model.predict(img_array)[0]
    idx = np.argmax(preds)
    
    return (
        CLASSES[idx], # class name
        float(preds[idx]) * 100, # acc score
    )