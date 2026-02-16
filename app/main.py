from pathlib import Path
import time
import logging
from collections import deque

import joblib
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse

from src.preprocess import load_image_rgb

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.pkl"

# ----- logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("cats-dogs-api")

app = FastAPI(title="Cats vs Dogs Inference Service", version="1.0")

# ---- simple in-app metrics
REQUEST_COUNT = 0
ERROR_COUNT = 0
TOTAL_LATENCY_SEC = 0.0
LATENCIES = deque(maxlen=1000)  # rolling window for p95

_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run training first or pull the model."
            )
        _model = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded from: {MODEL_PATH}")
    return _model


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    try:
        response = await call_next(request)
        return response
    finally:
        latency = time.time() - start
        logger.info(
            f"{request.method} {request.url.path} status={getattr(response,'status_code', 'NA')} latency={latency:.4f}s"
        )


@app.get("/health")
def health():
    ok = MODEL_PATH.exists()
    return {"status": "ok" if ok else "model_missing", "model_path": str(MODEL_PATH)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Returns:
      - label: "cat" or "dog"
      - probs: {"cat": p0, "dog": p1}
      - latency_sec
    """
    global REQUEST_COUNT, ERROR_COUNT, TOTAL_LATENCY_SEC

    start = time.time()
    REQUEST_COUNT += 1

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file")

        from io import BytesIO
        img_arr = load_image_rgb(BytesIO(content))  # (224,224,3)
        x = img_arr.reshape(1, -1)

        model = get_model()
        proba = model.predict_proba(x)[0]
        pred = int(np.argmax(proba))

        label = "cat" if pred == 0 else "dog"
        latency = time.time() - start

        TOTAL_LATENCY_SEC += latency
        LATENCIES.append(latency)

        logger.info(
            f"predict file={file.filename} label={label} p_cat={proba[0]:.4f} p_dog={proba[1]:.4f} latency={latency:.4f}s"
        )

        return {
            "label": label,
            "probs": {"cat": float(proba[0]), "dog": float(proba[1])},
            "latency_sec": latency,
        }

    except HTTPException:
        ERROR_COUNT += 1
        raise
    except Exception as e:
        ERROR_COUNT += 1
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")


@app.get("/metrics")
def metrics():
    avg = (TOTAL_LATENCY_SEC / REQUEST_COUNT) if REQUEST_COUNT else 0.0
    # p95 from rolling window
    if len(LATENCIES) > 0:
        p95 = float(np.percentile(np.array(LATENCIES), 95))
    else:
        p95 = 0.0

    return {
        "request_count": REQUEST_COUNT,
        "error_count": ERROR_COUNT,
        "avg_latency_sec": avg,
        "p95_latency_sec": p95,
        "window_size": len(LATENCIES),
    }