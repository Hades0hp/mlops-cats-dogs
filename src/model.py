from pathlib import Path
import joblib
import numpy as np
from PIL import Image

from src.preprocess import load_image_rgb, IMG_SIZE

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.pkl"

LABELS = {0: "cat", 1: "dog"}
_model = None

def load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Train first: python scripts/train.py"
            )
        _model = joblib.load(MODEL_PATH)
    return _model

def preprocess_pil(img: Image.Image) -> np.ndarray:
    # Keep it identical to training: RGB resize(224,224) float32 [0,1] flatten
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = (np.asarray(img, dtype=np.float32) / 255.0).reshape(1, -1)
    return arr

def predict_pil(img: Image.Image) -> dict:
    model = load_model()
    x = preprocess_pil(img)

    proba = model.predict_proba(x)[0]
    pred = int(proba.argmax())

    return {
        "label": LABELS[pred],
        "class_id": pred,
        "proba_cat": float(proba[0]),
        "proba_dog": float(proba[1]),
    }
