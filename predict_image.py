import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "crop_disease_model.h5"))

# Load model
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}. Set MODEL_PATH or train the model first.")

model = load_model(str(MODEL_PATH))
print("✅ Model loaded.")

# Class names (must match training order)
class_names = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'PlantVillage', 'Potato___Early_blight',
               'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight',
               'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
               'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
               'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']

# Function to preprocess and predict
def predict_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0  # normalize
        img_array = np.expand_dims(img_array, axis=0)  # batch dimension

        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        print(f"\n🌿 Predicted Class: {predicted_class}")
        print(f"📈 Confidence: {confidence:.2f}%")

    except Exception as e:
        print("❌ Error during prediction:", str(e))


# ---- Main Execution ----
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("⚠️ Please provide image path: python predict_image.py path_to_image.jpg")
    else:
        image_path = sys.argv[1]
        if not os.path.exists(image_path):
            print("❌ Image file not found:", image_path)
        else:
            predict_image(image_path)
