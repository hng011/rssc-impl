import streamlit as st
import requests
import base64
import time
from PIL import Image
import io
import os
from dotenv import load_dotenv


load_dotenv()


def get_prediction(model_selection, img_b64):
    API_URL_TF2180 = os.getenv("ENDPOINT_API-TF2180")
    API_URL_TRANSFORMERS4524 = os.getenv("ENDPOINT_TRANSFORMERS4524")
    
    start_t = time.time()
    response = requests.post(
        API_URL_TRANSFORMERS4524 if model_selection=='3' else API_URL_TF2180, 
        json={
            "model_selection": model_selection,
            "image_data": img_b64,
            "api_auth": os.getenv("API_AUTH"),
        }
    )
    infer_time = time.time() - start_t

    if response.status_code == 200:
        data = response.json()
        st.write(f"🔎 PREDICTION\t: {data.get("pred")}")
        st.write(f"📏 ACCURACY\t: {data.get("acc_score")}")
        st.write(f"🤖 Model\t: {data.get("model_name")}")
        st.write(f"⏳ INFERENCE TIME\t: {infer_time:.2f} seconds")
    else:
        st.error(f"API Error: {response.status_code}")


if __name__ == "__main__":
    st.title("🛰️ Remote Sensing Scene Classification Implementation")

    MODEL_OPTIONS = {"ResNet50-V2":'1', "ConvNeXt-Tiny":'2', "Visual Transformer (ViT)":'3'}

    model_selection = st.selectbox("🤖 Model Selection", list(MODEL_OPTIONS.keys()))
    uploaded_file = st.file_uploader("🖼️ Upload an Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        
        cols = st.columns([1, 2, 2]) 
        with cols[1]:
            st.image(image, caption="Uploaded Image", width=256)
            # clf_btn = st.button("Classify")
            
        with cols[2]:
            if clf_btn:
                try:
                    buffered = io.BytesIO()
                    image.save(buffered, format="JPEG")
                    img_bytes = buffered.getvalue()
                    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

                    with st.spinner("Predicting..."):
                        try:
                            get_prediction(MODEL_OPTIONS[model_selection], img_b64)
                        except Exception as e:
                            st.error(f"Request failed: {e}")
                except Exception as E:
                    st.error(E)
