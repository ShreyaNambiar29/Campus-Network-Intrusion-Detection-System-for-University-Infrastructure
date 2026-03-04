from __future__ import annotations


class SecurityScoreEngine:
    """Computes current network security score from open incidents."""

    SEVERITY_WEIGHTS = {
        "LOW": 1,
        "MEDIUM": 3,
        "HIGH": 6,
        "CRITICAL": 10,
    }

    def calculate(self, severities: list[str]) -> dict[str, int | str]:
        score_penalty = sum(self.SEVERITY_WEIGHTS.get(item.upper(), 1) for item in severities)
        score = max(0, min(100, 100 - score_penalty))
        return {
            "score": score,
            "status": self._status(score),
        }

    @staticmethod
    def _status(score: int) -> str:
        if score <= 40:
            return "CRITICAL"
        if score <= 60:
            return "HIGH"
        if score <= 80:
            return "MEDIUM"
        return "SAFE"
