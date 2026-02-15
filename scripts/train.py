# scripts/train.py
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from src.preprocess import build_xy_from_folders, save_npz

ROOT = Path(__file__).resolve().parents[1]
RAW_CATS = ROOT / "data" / "raw" / "cats"
RAW_DOGS = ROOT / "data" / "raw" / "dogs"

PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "model.pkl"


def plot_confusion(cm: np.ndarray, out_path: Path) -> None:
    fig = plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix (cat=0, dog=1)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main():
    t0 = time.time()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("[RUN] Building dataset (this can take time)...")
    # NOTE: build_xy_from_folders will read + resize + flatten all images
    # 2000 per class => 4000 total (manageable)
    X, y, paths = build_xy_from_folders(RAW_CATS, RAW_DOGS, limit_per_class=2000)

    print(f"[RUN] Dataset ready: X={X.shape} y={y.shape} (cats={int((y==0).sum())}, dogs={int((y==1).sum())})")

    print("[RUN] Splitting train/val/test (80/10/10)...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    # Saving processed arrays can be slow if compressed; keep it uncompressed
    print("[RUN] Saving processed datasets to data/processed (uncompressed .npz)...")
    save_npz(PROCESSED_DIR / "train.npz", X_train, y_train, compress=False)
    save_npz(PROCESSED_DIR / "val.npz", X_val, y_val, compress=False)
    save_npz(PROCESSED_DIR / "test.npz", X_test, y_test, compress=False)

    # MLflow local tracking (default ./mlruns)
    mlflow.set_experiment("cats-vs-dogs-baseline")

    with mlflow.start_run():
        params = {
            "model": "LogisticRegression",
            # IMPORTANT: saga is best for iterative warm-start progress tracking
            "solver": "saga",
            "img_size": "224x224x3_flatten",
            "train_size": int(len(X_train)),
            "val_size": int(len(X_val)),
            "test_size": int(len(X_test)),
            # step-wise training settings
            "warm_start": True,
            "iter_budget_per_step": 20,
            "num_steps": 10,
            "tol": 1e-3,
            "random_state": 42,
        }
        mlflow.log_params(params)

        # Fixed budget per step; warm_start=True continues from previous state.
        clf = LogisticRegression(
            solver=params["solver"],
            max_iter=params["iter_budget_per_step"],  # keep FIXED (do not grow)
            warm_start=True,
            random_state=params["random_state"],
            tol=params["tol"],
            verbose=1,  # prints solver progress so you can SEE it's training
            n_jobs=-1,
        )

        print("[TRAIN] Starting step-wise training...")
        for step in range(1, params["num_steps"] + 1):
            step_start = time.time()

            clf.fit(X_train, y_train)

            # quick validation after each step
            y_val_pred = clf.predict(X_val)
            val_acc = accuracy_score(y_val, y_val_pred)
            val_f1 = f1_score(y_val, y_val_pred)

            elapsed = time.time() - step_start
            approx_total_iter = step * params["iter_budget_per_step"]

            print(
                f"[TRAIN] step={step}/{params['num_steps']} "
                f"iter_budget={params['iter_budget_per_step']} "
                f"approx_total_iter~{approx_total_iter} "
                f"val_acc={val_acc:.4f} val_f1={val_f1:.4f} time={elapsed:.1f}s"
            )

            # Step-wise MLflow logging
            mlflow.log_metric("val_accuracy", float(val_acc), step=step)
            mlflow.log_metric("val_f1", float(val_f1), step=step)
            mlflow.log_metric("elapsed_step_sec", float(elapsed), step=step)
            mlflow.log_metric("approx_total_iter", float(approx_total_iter), step=step)

        print("[EVAL] Final evaluation on test set...")
        y_test_pred = clf.predict(X_test)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)

        mlflow.log_metrics({
            "test_accuracy": float(test_acc),
            "test_f1": float(test_f1),
        })

        cm = confusion_matrix(y_test, y_test_pred)
        cm_path = MODEL_DIR / "confusion_matrix.png"
        plot_confusion(cm, cm_path)
        mlflow.log_artifact(str(cm_path), artifact_path="artifacts")

        joblib.dump(clf, MODEL_PATH)
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="model")

        print(f"[DONE] Saved model to: {MODEL_PATH}")
        print(f"[DONE] Test acc={test_acc:.4f} Test f1={test_f1:.4f}")
        print(f"[DONE] Total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()