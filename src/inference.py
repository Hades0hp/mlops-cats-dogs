from pathlib import Path
from io import BytesIO

import numpy as np
from PIL import Image
import joblib

IMG_SIZE = (224, 224)

# repo root = .../mlops-cats-dogs
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.pkl"


def load_model(model_path: Path = MODEL_PATH):
    """
    Load trained model from disk.
    Uses absolute path so it works from any working directory (CI included).
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            f"Train first or fetch it via DVC/MLflow, or mount/copy it into the container."
        )
    return joblib.load(model_path)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess a single image (raw bytes) for inference.
    Returns shape (1, 150528) float32 in [0,1].
    """
    img = Image.open(BytesIO(image_bytes)).convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.reshape(1, -1)


def predict(model, X: np.ndarray):
    """
    Predict label + probabilities.
    """
    proba = model.predict_proba(X)[0]  # [p_cat, p_dog]
    pred = int(np.argmax(proba))
    return {
        "label": "dog" if pred == 1 else "cat",
        "probs": {"cat": float(proba[0]), "dog": float(proba[1])},
        "confidence": float(proba[pred]),
    }