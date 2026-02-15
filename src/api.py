import os
import time
import logging
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("catsdogs-api")

app = FastAPI(title="Cats vs Dogs API")

# Simple in-memory counters (basic monitoring)
REQUEST_COUNT = 0

@app.get("/health")
def health():
    return {"status": "ok"}

def dummy_predict(img: Image.Image):
    """
    Temporary placeholder until model training is wired.
    Returns random probabilities that sum to 1.
    """
    probs = np.random.rand(2)
    probs = probs / probs.sum()
    label = "cat" if probs[0] >= probs[1] else "dog"
    return {"label": label, "prob_cat": float(probs[0]), "prob_dog": float(probs[1])}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global REQUEST_COUNT
    REQUEST_COUNT += 1
    t0 = time.time()

    img = Image.open(file.file).convert("RGB")
    out = dummy_predict(img)

    latency_ms = (time.time() - t0) * 1000.0
    logger.info(f"req={REQUEST_COUNT} filename={file.filename} pred={out['label']} latency_ms={latency_ms:.2f}")
    out["latency_ms"] = latency_ms
    out["request_count"] = REQUEST_COUNT
    return out
