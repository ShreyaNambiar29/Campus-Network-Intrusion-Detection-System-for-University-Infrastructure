from __future__ import annotations

from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.incident import Incident


router = APIRouter(prefix="/incidents", tags=["incidents"])


class ResolveIncidentRequest(BaseModel):
    incident_id: int


@router.get("")
def get_incidents(
    status: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    db: Session = Depends(get_db),
) -> list[dict]:
    query = db.query(Incident)
    if status:
        query = query.filter(Incident.status == status)
    incidents = query.order_by(desc(Incident.timestamp)).limit(limit).all()
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
        }
        for incident in incidents
    ]


@router.post("/resolve")
def resolve_incident(payload: ResolveIncidentRequest, db: Session = Depends(get_db)) -> dict:
    incident = db.query(Incident).filter(Incident.id == payload.incident_id).first()
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found")

    incident.status = "resolved"
    db.commit()
    db.refresh(incident)

    return {
        "message": "Incident resolved",
        "incident": {
            "id": incident.id,
            "status": incident.status,
        },
    }
