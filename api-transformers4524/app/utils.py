from PIL import Image
import tensorflow as tf
import numpy as np

import base64
import io


id2label = {
        0: 'airplane', 1: 'cloud', 10: 'ship', 11: 'airport', 12: 'river', 13: 'golf_course', 14: 'roundabout', 15: 'church', 16: 'circular_farmland', 17: 'overpass', 18: 'railway', 19: 'wetland', 2: 'mountain', 20: 'lake', 21: 'parking_lot', 22: 'intersection', 23: 'tennis_court', 24: 'runway', 25: 'industrial_area', 26: 'chaparral', 27: 'bridge', 28: 'sparse_residential', 29: 'freeway', 3: 'medium_residential', 30: 'sea_ice', 31: 'beach', 32: 'palace', 33: 'snowberg', 34: 'meadow', 35: 'ground_track_field', 36: 'harbor', 37: 'rectangular_farmland', 38: 'island', 39: 'basketball_court', 4: 'thermal_power_station', 40: 'desert', 41: 'stadium', 42: 'forest',43: 'storage_tank', 44: 'railway_station', 5: 'terrace', 6: 'commercial_area', 7: 'dense_residential', 8: 'baseball_diamond', 9: 'mobile_home_park'
    }

def predict_image(image_data_b64, model, feature_extractor):
    
    decoded = base64.b64decode(image_data_b64)
    image = Image.open(io.BytesIO(decoded)).convert("RGB")
    
    inputs = feature_extractor(images=image, return_tensors="tf")
    outputs = model({"pixel_values": inputs["pixel_values"]}, training=False)
    logits = outputs["logits"] if isinstance(outputs, dict) else outputs
    
    probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
    predicted_class_idx = int(tf.argmax(probs))
    acc = float(probs[predicted_class_idx])

    predicted_label = id2label[predicted_class_idx]
    
    return (
        predicted_label, 
        acc * 100
    )