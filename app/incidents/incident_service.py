from __future__ import annotations

from datetime import datetime

from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.models.incident import Incident


class IncidentService:
    """CRUD service for security incidents."""

    def list_incidents(self, db: Session, status: str | None = None, limit: int = 100) -> list[Incident]:
        query = db.query(Incident)
        if status:
            query = query.filter(Incident.status == status)
        return query.order_by(desc(Incident.timestamp)).limit(limit).all()

    def resolve_incident(self, db: Session, incident_id: int) -> Incident | None:
        incident = db.query(Incident).filter(Incident.id == incident_id).first()
        if incident is None:
            return None
        incident.status = "resolved"
        incident.resolved_at = datetime.utcnow()
        db.commit()
        db.refresh(incident)
        return incident
