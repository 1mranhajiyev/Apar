"""Submission faylı yaratmaq üçün skript.

İstifadə:
    python src/predict.py
"""
import pandas as pd
import numpy as np
import pickle
import os
import sys

SEED = 42
np.random.seed(SEED)

DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
MODEL_PATH = os.path.join(OUTPUT_DIR, "model.pkl")


def main():
    # Modeli yüklə
    if not os.path.exists(MODEL_PATH):
        print("Model tapılmadı. Əvvəlcə 03_model.ipynb-i icra edin.")
        sys.exit(1)

    with open(MODEL_PATH, "rb") as f:
        pipeline = pickle.load(f)

    # Test dataseti
    test = pd.read_csv(os.path.join(DATA_DIR, "test_processed.csv"))
    X_test = test["text"].fillna("")

    # Proqnozlar
    preds = pipeline.predict(X_test)

    # Submission
    submission = pd.DataFrame({"id": test["id"], "label": preds})
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "submission.csv")
    submission.to_csv(out_path, index=False)
    print(f"Submission yaradıldı: {out_path}")
    print(submission["label"].value_counts())


if __name__ == "__main__":
    main()
