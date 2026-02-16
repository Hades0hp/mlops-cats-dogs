import numpy as np
from src.inference import predict

class DummyModel:
    def predict_proba(self, X):
        # always returns cat=0.2, dog=0.8
        return np.array([[0.2, 0.8]], dtype=np.float32)

def test_predict_returns_label_and_confidence():
    model = DummyModel()
    X = np.zeros((1, 224 * 224 * 3), dtype=np.float32)

    out = predict(model, X)

    assert out["label"] in ["cat", "dog"]
    assert out["label"] == "dog"
    assert abs(out["confidence"] - 0.8) < 1e-6