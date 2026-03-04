from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timezone
from typing import Any

from fastapi import WebSocket
from sqlalchemy.orm import Session

from app.models.incident import Incident


class AlertManager:
    """Broadcasts alerts to connected WebSocket clients and stores recent events."""

    def __init__(self, max_recent_alerts: int = 200) -> None:
        self.connections: set[WebSocket] = set()
        self.recent_alerts: deque[dict[str, Any]] = deque(maxlen=max_recent_alerts)
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self.connections.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self.connections.discard(websocket)

    async def publish(self, alert: dict[str, Any]) -> None:
        self.recent_alerts.appendleft(alert)

        async with self._lock:
            dead_connections: list[WebSocket] = []
            for websocket in self.connections:
                try:
                    await websocket.send_json(alert)
                except Exception:
                    dead_connections.append(websocket)

            for websocket in dead_connections:
                self.connections.discard(websocket)

    def publish_from_thread(self, loop: asyncio.AbstractEventLoop, alert: dict[str, Any]) -> None:
        asyncio.run_coroutine_threadsafe(self.publish(alert), loop)

    def get_recent_alerts(self, limit: int = 50) -> list[dict[str, Any]]:
        return list(self.recent_alerts)[:limit]

    def persist_incident(self, db: Session, alert: dict[str, Any]) -> Incident:
        timestamp = alert.get("timestamp")
        if isinstance(timestamp, (int, float)):
            incident_time = datetime.fromtimestamp(timestamp, tz=timezone.utc).replace(tzinfo=None)
        elif isinstance(timestamp, datetime):
            incident_time = timestamp
        else:
            incident_time = datetime.utcnow()

        incident = Incident(
            timestamp=incident_time,
            source_ip=alert.get("source_ip", "0.0.0.0"),
            destination_ip=alert.get("destination_ip", "0.0.0.0"),
            attack_type=alert.get("attack_type", "Unknown"),
            severity=alert.get("severity", "LOW"),
            description=alert.get("description", "No description"),
            status=alert.get("status", "open"),
        )
        db.add(incident)
        db.commit()
        db.refresh(incident)
        return incident
