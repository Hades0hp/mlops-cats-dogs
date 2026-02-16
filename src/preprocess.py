# src/preprocess.py
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageFile
from typing import Union, BinaryIO

# Helps with partially downloaded / truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_SIZE = (224, 224)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_image_rgb(path_or_file: Union[str, Path, BinaryIO], size: Tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """Load image -> resize -> RGB -> float32 in [0,1]. Accepts a path or a file-like object."""
    img = Image.open(path_or_file).convert("RGB").resize(size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def build_xy_from_folders(
    cats_dir: str | Path,
    dogs_dir: str | Path,
    limit_per_class: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    cats_dir = Path(cats_dir)
    dogs_dir = Path(dogs_dir)

    def list_images(d: Path) -> List[Path]:
        return sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])

    cat_paths = list_images(cats_dir)
    dog_paths = list_images(dogs_dir)

    if limit_per_class is not None:
        cat_paths = cat_paths[:limit_per_class]
        dog_paths = dog_paths[:limit_per_class]

    paths = cat_paths + dog_paths
    y = np.array([0] * len(cat_paths) + [1] * len(dog_paths), dtype=np.int64)

    n = len(paths)
    d = IMG_SIZE[0] * IMG_SIZE[1] * 3

    print(f"[DATA] Allocating X array: (n={n}, d={d}) ~ {(n*d*4)/1e9:.2f} GB float32")
    X = np.empty((n, d), dtype=np.float32)

    kept_paths = []
    dropped = 0
    write_i = 0

    for idx, p in enumerate(paths, start=1):
        try:
            arr = load_image_rgb(p)
            X[write_i] = arr.reshape(-1)
            kept_paths.append(str(p))
            write_i += 1
        except Exception as e:
            dropped += 1
            print(f"[WARN] Skipping unreadable image: {p} ({type(e).__name__}: {e})")

        if idx % 200 == 0:
            print(f"[DATA] processed={idx}/{n} kept={write_i} dropped={dropped}")

    # trim if we dropped anything
    X = X[:write_i]
    y = y[:write_i]

    return X, y, kept_paths


def save_npz(out_path: str | Path, X: np.ndarray, y: np.ndarray, compress: bool = False) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        np.savez_compressed(out_path, X=X, y=y)
    else:
        np.savez(out_path, X=X, y=y)