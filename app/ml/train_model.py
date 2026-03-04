"""Offline training utility for the IDS Isolation Forest model."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest


def train_and_save_model(output_path: str = "app/ml/iforest_model.joblib") -> None:
    rng = np.random.default_rng(seed=42)

    benign = np.column_stack(
        [
            rng.integers(1, 4, 5000),
            rng.normal(900, 300, 5000).clip(60, 1800),
            rng.integers(1024, 65000, 5000),
            rng.choice([53, 80, 123, 443, 8080], 5000),
            rng.normal(5, 2, 5000).clip(0, 40),
            rng.normal(8, 4, 5000).clip(0, 120),
        ]
    )

    anomaly = np.column_stack(
        [
            rng.integers(1, 4, 400),
            rng.normal(1800, 300, 400).clip(500, 2500),
            rng.integers(1, 1024, 400),
            rng.choice([22, 23, 3389, 445, 3306], 400),
            rng.normal(50, 10, 400).clip(15, 200),
            rng.normal(2, 1.5, 400).clip(0, 10),
        ]
    )

    training_data = np.vstack([benign, anomaly])

    model = IsolationForest(n_estimators=300, contamination=0.08, random_state=42)
    model.fit(training_data)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)
    print(f"Model saved to {out}")


if __name__ == "__main__":
    train_and_save_model()
