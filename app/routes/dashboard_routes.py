from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.core.security_score import SecurityScoreEngine
from app.database import get_db
from app.models.incident import Incident


router = APIRouter(tags=["dashboard"])
security_score_engine = SecurityScoreEngine()


def _get_runtime(request: Request):
    runtime = getattr(request.app.state, "ids", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="IDS runtime not initialized")
    return runtime


@router.get("/security-score")
def get_security_score(db: Session = Depends(get_db)) -> dict:
    open_incidents = db.query(Incident).filter(Incident.status == "open").all()
    result = security_score_engine.calculate([item.severity for item in open_incidents])
    return {
        "security_score": result["score"],
        "status": result["status"],
        "open_incidents": len(open_incidents),
    }


@router.get("/dashboard/overview")
def get_dashboard_overview(request: Request, db: Session = Depends(get_db)) -> dict:
    runtime = _get_runtime(request)
    open_incidents = db.query(Incident).filter(Incident.status == "open").count()
    resolved_incidents = db.query(Incident).filter(Incident.status == "resolved").count()
    critical_incidents = db.query(Incident).filter(Incident.severity == "CRITICAL").count()

    score = security_score_engine.calculate(
        [item.severity for item in db.query(Incident).filter(Incident.status == "open").all()]
    )
    return {
        "total_packets_seen": runtime.total_packets_seen,
        "active_threats": open_incidents,
        "resolved_incidents": resolved_incidents,
        "critical_incidents": critical_incidents,
        "security_score": score["score"],
        "security_status": score["status"],
    }
