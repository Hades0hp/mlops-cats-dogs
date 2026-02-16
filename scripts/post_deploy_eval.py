import random
import json
from pathlib import Path

import requests

API = "http://127.0.0.1:8000"
DATA = Path("data/raw")  # cats/ dogs/
OUT = Path("reports")
OUT.mkdir(exist_ok=True)

SAMPLE_PER_CLASS = 10  # total 20


def pick_images(folder: Path, n: int):
    imgs = list(folder.glob("*.jpg"))
    random.shuffle(imgs)
    return imgs[:n]


def main():
    cats = pick_images(DATA / "cats", SAMPLE_PER_CLASS)
    dogs = pick_images(DATA / "dogs", SAMPLE_PER_CLASS)

    items = []
    correct = 0
    total = 0
    # confusion: true rows, pred cols
    # cat=0 dog=1
    cm = [[0, 0], [0, 0]]

    for img_path in cats + dogs:
        true_label = "cat" if img_path.parent.name == "cats" else "dog"
        files = {"file": open(img_path, "rb")}
        r = requests.post(f"{API}/predict", files=files, timeout=30)
        r.raise_for_status()
        pred = r.json()["label"]

        total += 1
        if pred == true_label:
            correct += 1

        t = 0 if true_label == "cat" else 1
        p = 0 if pred == "cat" else 1
        cm[t][p] += 1

        items.append(
            {
                "file": str(img_path),
                "true": true_label,
                "pred": pred,
                "probs": r.json().get("probs", {}),
                "latency_sec": r.json().get("latency_sec"),
            }
        )

    acc = correct / total if total else 0.0
    report = {"accuracy": acc, "total": total, "correct": correct, "confusion_matrix": cm, "samples": items}

    (OUT / "post_deploy_report.json").write_text(json.dumps(report, indent=2))
    print("Saved:", OUT / "post_deploy_report.json")
    print("Accuracy:", acc)
    print("Confusion Matrix [true rows cat,dog][pred cols cat,dog]:", cm)


if __name__ == "__main__":
    main()