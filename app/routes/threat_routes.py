from __future__ import annotations

from collections import Counter

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.threat_event import ThreatEvent


router = APIRouter(tags=["threats"])


@router.get("/threats/top")
def get_top_threats(limit: int = Query(default=10, ge=1, le=100), db: Session = Depends(get_db)) -> dict:
    rows = db.query(ThreatEvent).order_by(ThreatEvent.timestamp.desc()).limit(1000).all()
    top_ips = Counter(event.source_ip for event in rows).most_common(limit)
    top_types = Counter(event.attack_type for event in rows).most_common(limit)
    return {
        "top_attacking_ips": [{"ip": ip, "count": count} for ip, count in top_ips],
        "top_attack_types": [{"attack_type": attack_type, "count": count} for attack_type, count in top_types],
    }
