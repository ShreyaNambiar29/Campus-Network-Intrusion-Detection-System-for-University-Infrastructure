from __future__ import annotations

from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.incident import Incident
from app.monitoring.traffic_analyzer import TrafficAnalyzer


router = APIRouter(tags=["traffic"])
traffic_analyzer = TrafficAnalyzer()


def _get_runtime(request: Request):
    runtime = getattr(request.app.state, "ids", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="IDS runtime not initialized")
    return runtime


@router.get("/traffic/live")
def get_live_traffic(request: Request) -> dict:
    runtime = _get_runtime(request)
    packets = runtime.packet_sniffer.get_recent_packets(limit=80)
    summary = traffic_analyzer.summarize(packets)
    return {
        "count": len(packets),
        "latest": packets[-1] if packets else None,
        "packets": packets,
        "summary": summary,
    }


@router.get("/stats")
def get_dashboard_stats(request: Request, db: Session = Depends(get_db)) -> dict:
    runtime = _get_runtime(request)

    open_incidents = db.query(Incident).filter(Incident.status == "open").count()
    resolved_incidents = db.query(Incident).filter(Incident.status == "resolved").count()

    critical_incidents = db.query(Incident).filter(Incident.severity == "CRITICAL").count()
    high_incidents = db.query(Incident).filter(Incident.severity == "HIGH").count()

    one_minute_ago = datetime.utcnow() - timedelta(minutes=1)
    packets_per_minute = sum(1 for packet in runtime.packet_sniffer.get_recent_packets(500) if packet["timestamp"] >= one_minute_ago.timestamp())

    return {
        "total_packets_seen": runtime.total_packets_seen,
        "packets_last_minute": packets_per_minute,
        "active_threats": open_incidents,
        "resolved_incidents": resolved_incidents,
        "critical_incidents": critical_incidents,
        "high_incidents": high_incidents,
        "recent_alerts": runtime.alert_manager.get_recent_alerts(limit=10),
    }
