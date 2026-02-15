from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

from src.model import predict_pil

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    img = Image.open(io.BytesIO(content))
    return predict_pil(img)
