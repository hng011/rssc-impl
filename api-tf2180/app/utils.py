from PIL import Image
import tensorflow as tf
import numpy as np

import base64
import io


CLASSES = ['airplane', 'airport', 'baseball_diamond', 'basketball_court', 'beach', 'bridge', 'chaparral', 'church', 'circular_farmland', 'cloud', 'commercial_area', 'dense_residential', 'desert', 'forest', 'freeway', 'golf_course', 'ground_track_field', 'harbor', 'industrial_area', 'intersection', 'island', 'lake', 'meadow', 'medium_residential', 'mobile_home_park', 'mountain', 'overpass', 'palace', 'parking_lot', 'railway', 'railway_station', 'rectangular_farmland', 'river', 'roundabout', 'runway', 'sea_ice', 'ship', 'snowberg', 'sparse_residential', 'stadium', 'storage_tank', 'tennis_court', 'terrace', 'thermal_power_station', 'wetland']


def predict_image(image_data_b64, model):
    
    decoded = base64.b64decode(image_data_b64)
    image = Image.open(io.BytesIO(decoded)).convert("RGB")
    image = image.resize((224, 224))  
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)  # add batch dim
    img_array = img_array / 255.0
    
    logits = model.predict(img_array)[0]
    probs = tf.nn.softmax(logits, axis=-1).numpy()
    idx = np.argmax(probs)
    
    return (
        CLASSES[idx], 
        float(probs[idx]) * 100, 
    )