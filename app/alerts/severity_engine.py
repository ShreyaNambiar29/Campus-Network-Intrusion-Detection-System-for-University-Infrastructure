from __future__ import annotations


class SeverityEngine:
    """Computes severity from event characteristics and behavior patterns."""

    def calculate(self, base_severity: str, attempts: int, target_count: int) -> str:
        if target_count >= 4:
            return "CRITICAL"
        if attempts >= 15:
            return "HIGH"

        normalized = base_severity.upper()
        if normalized not in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}:
            return "LOW"
        return normalized
