from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
try:
    from sklearn.ensemble import IsolationForest
except Exception:
    IsolationForest = None  # type: ignore[assignment]


class AnomalyDetector:
    """Isolation Forest wrapper used for network anomaly detection."""

    def __init__(self, model_path: str) -> None:
        self.model_path = Path(model_path)
        self.model: Any = None
        self.fallback_mode = IsolationForest is None

    @staticmethod
    def vectorize(features: dict[str, Any]) -> np.ndarray:
        return np.array(
            [
                [
                    float(features.get("protocol_id", 0)),
                    float(features.get("packet_size", 0)),
                    float(features.get("source_port", 0)),
                    float(features.get("destination_port", 0)),
                    float(features.get("frequency", 0)),
                    float(features.get("session_duration", 0)),
                ]
            ],
            dtype=float,
        )

    def fit_baseline(self) -> None:
        if IsolationForest is None:
            self.fallback_mode = True
            self.model = None
            return

        rng = np.random.default_rng(seed=42)
        baseline = np.column_stack(
            [
                rng.integers(1, 4, 600),
                rng.normal(700, 230, 600).clip(60, 1800),
                rng.integers(1024, 65000, 600),
                rng.choice([21, 22, 53, 80, 123, 443, 3306, 3389, 8080], 600),
                rng.normal(4, 2, 600).clip(0, 40),
                rng.normal(8, 5, 600).clip(0, 120),
            ]
        )
        model = IsolationForest(n_estimators=250, contamination=0.08, random_state=42)
        model.fit(baseline)
        self.model = model

    def load_or_initialize(self) -> None:
        if IsolationForest is None:
            self.fallback_mode = True
            self.model = None
            return

        if self.model_path.exists():
            self.model = joblib.load(self.model_path)
            return
        self.fit_baseline()

    def predict(self, features: dict[str, Any]) -> tuple[bool, float]:
        if self.model is None and not self.fallback_mode:
            self.load_or_initialize()
        if self.fallback_mode or self.model is None:
            size = float(features.get("packet_size", 0))
            frequency = float(features.get("frequency", 0))
            risk = min(1.0, max(0.0, (size / 2200.0) * 0.4 + (frequency / 90.0) * 0.6))
            return risk >= 0.65, risk

        vector = self.vectorize(features)
        prediction = self.model.predict(vector)[0]
        raw_score = float(self.model.decision_function(vector)[0])
        normalized_risk = max(0.0, min(1.0, (0.2 - raw_score) / 0.4))
        return prediction == -1, normalized_risk
