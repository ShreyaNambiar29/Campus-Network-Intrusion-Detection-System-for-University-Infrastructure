from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.incidents.incident_service import IncidentService


router = APIRouter(prefix="/incidents", tags=["incidents"])
incident_service = IncidentService()


class ResolveIncidentRequest(BaseModel):
    incident_id: int


@router.get("")
def get_incidents(
    status: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    db: Session = Depends(get_db),
) -> list[dict]:
    incidents = incident_service.list_incidents(db, status=status, limit=limit)
    return [
        {
            "id": incident.id,
            "timestamp": incident.timestamp.isoformat(),
            "source_ip": incident.source_ip,
            "destination_ip": incident.destination_ip,
            "attack_type": incident.attack_type,
            "severity": incident.severity,
            "description": incident.description,
            "status": incident.status,
            "resolved_at": incident.resolved_at.isoformat() if incident.resolved_at else None,
        }
        for incident in incidents
    ]


@router.post("/resolve")
def resolve_incident(payload: ResolveIncidentRequest, db: Session = Depends(get_db)) -> dict:
    incident = incident_service.resolve_incident(db, payload.incident_id)
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found")

    return {
        "message": "Incident resolved",
        "incident": {
            "id": incident.id,
            "status": incident.status,
            "resolved_at": incident.resolved_at.isoformat() if incident.resolved_at else None,
        },
    }
