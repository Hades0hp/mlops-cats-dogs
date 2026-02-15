from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np

from src.preprocess import load_image_rgb, IMG_SIZE

def load_model(model_path: str | Path):
    model_path = Path(model_path)
    return joblib.load(model_path)

def predict_proba_from_image(model, image_path: str | Path) -> Dict[str, Any]:
    arr = load_image_rgb(image_path, size=IMG_SIZE)   # (224,224,3) in [0,1]
    x = arr.reshape(1, -1)                            # (1, D)
    proba = model.predict_proba(x)[0]                 # [P(cat), P(dog)]
    label = "cat" if proba[0] >= proba[1] else "dog"
    return {"label": label, "prob_cat": float(proba[0]), "prob_dog": float(proba[1])}