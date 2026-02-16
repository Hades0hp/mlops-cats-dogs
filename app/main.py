from pathlib import Path
import time

import joblib
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from src.preprocess import load_image_rgb

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.pkl"

app = FastAPI(title="Cats vs Dogs Inference Service", version="1.0")

# ---- simple in-app metrics (useful for M5 later)
REQUEST_COUNT = 0
TOTAL_LATENCY_SEC = 0.0

_model = None

def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run training first or pull the model."
            )
        _model = joblib.load(MODEL_PATH)
    return _model

@app.get("/")
def root():
    return {"message": "API is running. Visit /docs for Swagger UI."}

@app.get("/health")
def health():
    ok = MODEL_PATH.exists()
    return {"status": "ok" if ok else "model_missing", "model_path": str(MODEL_PATH)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an image file and returns:
      - label: "cat" or "dog"
      - probs: {"cat": p0, "dog": p1}
    """
    global REQUEST_COUNT, TOTAL_LATENCY_SEC

    start = time.time()
    REQUEST_COUNT += 1

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file")

        # PIL can read from bytes via BytesIO
        from io import BytesIO
        img_arr = load_image_rgb(BytesIO(content))     # (224,224,3) float32 [0,1]
        x = img_arr.reshape(1, -1)                     # (1, 150528)

        model = get_model()
        proba = model.predict_proba(x)[0]              # [p_cat, p_dog]
        pred = int(np.argmax(proba))

        label = "cat" if pred == 0 else "dog"

        return {
            "label": label,
            "probs": {"cat": float(proba[0]), "dog": float(proba[1])},
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
    finally:
        TOTAL_LATENCY_SEC += (time.time() - start)

@app.get("/metrics")
def metrics():
    avg = (TOTAL_LATENCY_SEC / REQUEST_COUNT) if REQUEST_COUNT else 0.0
    return {
        "request_count": REQUEST_COUNT,
        "avg_latency_sec": avg,
    }