import numpy as np
from PIL import Image
from io import BytesIO

from src.preprocess import load_image_rgb, IMG_SIZE

def test_load_image_rgb_shape_and_range():
    # create a small dummy RGB image in memory
    img = Image.new("RGB", (50, 50), color=(10, 20, 30))
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    arr = load_image_rgb(buf)

    assert arr.shape == (IMG_SIZE[1], IMG_SIZE[0], 3) or arr.shape == (IMG_SIZE[0], IMG_SIZE[1], 3)
    assert arr.dtype == np.float32
    assert float(arr.min()) >= 0.0
    assert float(arr.max()) <= 1.0