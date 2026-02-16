from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import io
import time

from .inference import load_model, preprocess_image, predict

app = FastAPI(title="Cats vs Dogs Classifier")

model = load_model()

REQUEST_COUNT = 0


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    global REQUEST_COUNT
    start = time.time()

    image_bytes = io.BytesIO(await file.read())
    X = preprocess_image(image_bytes)

    result = predict(model, X)

    latency = time.time() - start
    REQUEST_COUNT += 1

    print(f"[LOG] requests={REQUEST_COUNT} latency={latency:.4f}s")

    return JSONResponse({
        "prediction": result,
        "latency": latency
    })