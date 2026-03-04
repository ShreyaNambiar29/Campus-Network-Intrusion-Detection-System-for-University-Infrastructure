from __future__ import annotations

from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.analytics.statistics_engine import StatisticsEngine
from app.analytics.traffic_metrics import TrafficMetrics
from app.database import get_db
from app.models.incident import Incident
from app.models.traffic_log import TrafficLog


router = APIRouter(prefix="/analytics", tags=["analytics"])
statistics_engine = StatisticsEngine()
traffic_metrics = TrafficMetrics()


@router.get("/traffic")
def get_traffic_analytics(
    lookback_minutes: int = Query(default=60, ge=5, le=10080),
    db: Session = Depends(get_db),
) -> dict:
    since = datetime.utcnow() - timedelta(minutes=lookback_minutes)
    logs = db.query(TrafficLog).filter(TrafficLog.timestamp >= since).order_by(TrafficLog.timestamp.asc()).all()

    return {
        "traffic_volume_over_time": traffic_metrics.volume_over_time(logs),
        "total_packets": len(logs),
        "total_bytes": sum(item.packet_size for item in logs),
    }


@router.get("/attacks")
def get_attack_analytics(
    lookback_days: int = Query(default=7, ge=1, le=365),
    db: Session = Depends(get_db),
) -> dict:
    since = datetime.utcnow() - timedelta(days=lookback_days)
    incidents = db.query(Incident).filter(Incident.timestamp >= since).order_by(Incident.timestamp.asc()).all()

    return {
        "attack_distribution": statistics_engine.attack_distribution(incidents),
        "most_common_attack_types": statistics_engine.attack_types(incidents),
        "daily_threat_trends": statistics_engine.daily_threat_trend(incidents),
        "top_attacking_ips": statistics_engine.top_attacking_ips(incidents),
    }
