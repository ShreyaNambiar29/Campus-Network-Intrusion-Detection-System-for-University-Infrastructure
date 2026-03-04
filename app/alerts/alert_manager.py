from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.models.incident import Incident
from app.models.threat_event import ThreatEvent


class AlertManager:
    """Handles persistence and serialization of alert lifecycle records."""

    @staticmethod
    def _resolve_time(timestamp: Any) -> datetime:
        if isinstance(timestamp, (int, float)):
            return datetime.fromtimestamp(timestamp, tz=timezone.utc).replace(tzinfo=None)
        if isinstance(timestamp, datetime):
            return timestamp
        return datetime.utcnow()

    def persist_incident(self, db: Session, alert: dict[str, Any]) -> Incident:
        incident = Incident(
            timestamp=self._resolve_time(alert.get("timestamp")),
            source_ip=alert.get("source_ip", "0.0.0.0"),
            destination_ip=alert.get("destination_ip", "0.0.0.0"),
            attack_type=alert.get("attack_type", "Unknown"),
            severity=alert.get("severity", "LOW"),
            description=alert.get("description", "No description"),
            status=alert.get("status", "open"),
        )
        db.add(incident)
        db.flush()
        return incident

    def persist_threat_event(self, db: Session, alert: dict[str, Any], incident_id: int | None) -> ThreatEvent:
        threat_event = ThreatEvent(
            timestamp=self._resolve_time(alert.get("timestamp")),
            source_ip=alert.get("source_ip", "0.0.0.0"),
            destination_ip=alert.get("destination_ip", "0.0.0.0"),
            attack_type=alert.get("attack_type", "Unknown"),
            severity=alert.get("severity", "LOW"),
            threat_score=int(alert.get("threat_score", 0)),
            protocol=alert.get("protocol", "OTHER"),
            destination_port=int(alert.get("destination_port", 0)),
            incident_id=incident_id,
        )
        db.add(threat_event)
        db.flush()
        return threat_event
