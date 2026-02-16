from pathlib import Path
import numpy as np
from PIL import Image
import joblib

IMG_SIZE = (224, 224)

MODEL_PATH = Path("models/model.pkl")


def load_model():
    return joblib.load(MODEL_PATH)


def preprocess_image(image_bytes) -> np.ndarray:
    """
    Preprocess single image for inference.
    """
    img = Image.open(image_bytes).convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.reshape(1, -1)  # flatten like training


def predict(model, X):
    proba = model.predict_proba(X)[0]
    label = int(np.argmax(proba))
    return {
        "label": "dog" if label == 1 else "cat",
        "confidence": float(proba[label])
    }