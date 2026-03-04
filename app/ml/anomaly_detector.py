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
    """Isolation Forest wrapper for traffic anomaly detection."""

    def __init__(self, model_path: str) -> None:
        self.model_path = Path(model_path)
        self.model: Any = None
        self.fallback_mode = IsolationForest is None

    def _vectorize(self, features: dict[str, Any]) -> np.ndarray:
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
        """Train a fallback baseline model using synthetic benign traffic."""
        if IsolationForest is None:
            self.fallback_mode = True
            self.model = None
            return

        rng = np.random.default_rng(seed=42)
        baseline = np.column_stack(
            [
                rng.integers(1, 4, 500),
                rng.normal(800, 250, 500).clip(60, 1600),
                rng.integers(1024, 65000, 500),
                rng.choice([53, 80, 123, 443, 8080], 500),
                rng.normal(4, 2, 500).clip(0, 30),
                rng.normal(7, 4, 500).clip(0, 90),
            ]
        )
        model = IsolationForest(n_estimators=200, contamination=0.08, random_state=42)
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
            destination_port = float(features.get("destination_port", 0))
            risk = min(1.0, max(0.0, (size / 2200.0) * 0.35 + (frequency / 90.0) * 0.55 + (1.0 if destination_port in {22, 23, 3389, 3306} else 0.0) * 0.1))
            return risk >= 0.65, risk

        vector = self._vectorize(features)
        prediction = self.model.predict(vector)[0]  # -1 anomaly, 1 normal
        raw_score = float(self.model.decision_function(vector)[0])
        normalized_risk = max(0.0, min(1.0, (0.2 - raw_score) / 0.4))
        return prediction == -1, normalized_risk
