from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import time

from .inference import load_model, preprocess_image, predict

app = FastAPI(title="Cats vs Dogs Classifier")

# In-memory app metrics
REQUEST_COUNT = 0
TOTAL_LATENCY_SEC = 0.0

# Lazy-loaded singleton
_model = None


def get_model():
    """Load the model once, on first request (or first health check)."""
    global _model
    if _model is None:
        _model = load_model()
    return _model


@app.get("/health")
def health():
    """
    Health endpoint:
    - status=ok if model loads successfully
    - status=model_missing if model file is not present
    """
    try:
        _ = get_model()
        return {"status": "ok"}
    except FileNotFoundError as e:
        return {"status": "model_missing", "detail": str(e)}


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    global REQUEST_COUNT, TOTAL_LATENCY_SEC

    start = time.time()
    REQUEST_COUNT += 1

    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        X = preprocess_image(image_bytes)

        model = get_model()
        result = predict(model, X)

        latency = time.time() - start
        TOTAL_LATENCY_SEC += latency

        return JSONResponse(
            {
                "prediction": result,
                "latency": latency,
            }
        )

    except HTTPException:
        raise
    except FileNotFoundError as e:
        # clear error if model is missing in container/CI
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")


@app.get("/metrics")
def metrics():
    avg = (TOTAL_LATENCY_SEC / REQUEST_COUNT) if REQUEST_COUNT else 0.0
    return {
        "request_count": REQUEST_COUNT,
        "avg_latency_sec": avg,
    }