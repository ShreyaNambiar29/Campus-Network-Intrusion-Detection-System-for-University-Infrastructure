from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.alerts.alert_manager import AlertManager
from app.alerts.websocket_manager import WebSocketManager
from app.config import settings
from app.database import SessionLocal, init_db
from app.ml.anomaly_detector import AnomalyDetector
from app.models.traffic_log import TrafficLog
from app.incidents.incident_routes import router as incident_router
from app.monitoring.feature_extractor import FeatureExtractor
from app.monitoring.packet_sniffer import PacketSniffer
from app.routes.analytics_routes import router as analytics_router
from app.routes.alert_routes import router as alert_router
from app.routes.dashboard_routes import router as dashboard_router
from app.routes.threat_routes import router as threat_router
from app.routes.traffic_routes import router as traffic_router
from app.services.detection_engine import DetectionEngine
from app.utils.json_store import JsonEventStore
from app.utils.logger import get_logger


logger = get_logger("ids.main")


@dataclass
class IDSRuntime:
    packet_sniffer: PacketSniffer
    feature_extractor: FeatureExtractor
    detection_engine: DetectionEngine
    alert_manager: AlertManager
    websocket_manager: WebSocketManager
    loop: asyncio.AbstractEventLoop
    json_store: JsonEventStore | None = None
    total_packets_seen: int = 0

    def handle_packet(self, packet_data: dict) -> None:
        self.total_packets_seen += 1
        self._persist_traffic(packet_data)

        features = self.feature_extractor.extract(packet_data)
        alerts = self.detection_engine.analyze(features)
        if not alerts:
            return

        for alert in alerts:
            self._store_and_emit(alert)

    def _store_and_emit(self, alert: dict) -> None:
        db = SessionLocal()
        try:
            incident = self.alert_manager.persist_incident(db, alert)
            self.alert_manager.persist_threat_event(db, alert, incident.id)
            db.commit()
            db.refresh(incident)
            alert_payload = {
                "id": incident.id,
                "timestamp": incident.timestamp.replace(tzinfo=timezone.utc).isoformat(),
                "source_ip": incident.source_ip,
                "destination_ip": incident.destination_ip,
                "attack_type": incident.attack_type,
                "severity": incident.severity,
                "description": incident.description,
                "status": incident.status,
                "threat_score": alert.get("threat_score", 0),
                "protocol": alert.get("protocol", "OTHER"),
                "destination_port": alert.get("destination_port", 0),
                "attempts": alert.get("attempts", 0),
                "target_count": alert.get("target_count", 0),
            }
            self.websocket_manager.broadcast_from_thread(self.loop, alert_payload)
            if self.json_store is not None:
                self.json_store.save_alert(alert_payload)
            logger.info("Threat detected", extra={"extra": alert_payload})
        except Exception as exc:
            db.rollback()
            logger.error("Failed to persist or emit alert", extra={"extra": {"error": str(exc)}})
        finally:
            db.close()

    def _persist_traffic(self, packet_data: dict) -> None:
        db = SessionLocal()
        try:
            timestamp_value = packet_data.get("timestamp")
            if isinstance(timestamp_value, (int, float)):
                packet_time = datetime.fromtimestamp(timestamp_value, tz=timezone.utc).replace(tzinfo=None)
            elif isinstance(timestamp_value, datetime):
                packet_time = timestamp_value
            else:
                packet_time = datetime.utcnow()

            traffic_log = TrafficLog(
                timestamp=packet_time,
                source_ip=packet_data.get("source_ip", "0.0.0.0"),
                destination_ip=packet_data.get("destination_ip", "0.0.0.0"),
                protocol=packet_data.get("protocol", "OTHER"),
                source_port=int(packet_data.get("source_port", 0)),
                destination_port=int(packet_data.get("destination_port", 0)),
                packet_size=int(packet_data.get("packet_size", 0)),
            )
            db.add(traffic_log)
            db.commit()
            if self.json_store is not None:
                self.json_store.save_traffic(packet_data)
        except Exception as exc:
            db.rollback()
            logger.error("Failed to persist traffic log", extra={"extra": {"error": str(exc)}})
        finally:
            db.close()


app = FastAPI(title=settings.app_name, version="1.0.0", debug=settings.debug)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(traffic_router, prefix=settings.api_prefix)
app.include_router(incident_router, prefix=settings.api_prefix)
app.include_router(alert_router, prefix=settings.api_prefix)
app.include_router(threat_router, prefix=settings.api_prefix)
app.include_router(analytics_router, prefix=settings.api_prefix)
app.include_router(dashboard_router, prefix=settings.api_prefix)


@app.on_event("startup")
async def on_startup() -> None:
    if settings.auto_create_schema:
        init_db()

    anomaly_detector = AnomalyDetector(model_path=settings.model_path)
    anomaly_detector.load_or_initialize()

    alert_manager = AlertManager()
    websocket_manager = WebSocketManager()
    feature_extractor = FeatureExtractor(window_seconds=settings.feature_window_seconds)
    detection_engine = DetectionEngine(anomaly_detector=anomaly_detector)

    loop = asyncio.get_running_loop()
    json_store = None
    if settings.persist_json_logs:
        json_store = JsonEventStore(
            storage_dir=settings.json_storage_dir,
            traffic_file=settings.traffic_json_file,
            alerts_file=settings.alerts_json_file,
        )

    runtime = IDSRuntime(
        packet_sniffer=PacketSniffer(on_packet=lambda packet: runtime.handle_packet(packet), max_buffer=settings.packet_buffer_size),
        feature_extractor=feature_extractor,
        detection_engine=detection_engine,
        alert_manager=alert_manager,
        websocket_manager=websocket_manager,
        loop=loop,
        json_store=json_store,
    )

    # Replace self-referential lambda target once runtime exists.
    runtime.packet_sniffer.on_packet = runtime.handle_packet

    if settings.enable_packet_sniffer:
        runtime.packet_sniffer.start()

    app.state.ids = runtime
    logger.info("IDS backend started", extra={"extra": {"timestamp": datetime.utcnow().isoformat()}})


@app.on_event("shutdown")
async def on_shutdown() -> None:
    runtime = getattr(app.state, "ids", None)
    if runtime is not None:
        runtime.packet_sniffer.stop()
    logger.info("IDS backend stopped")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": settings.app_name}


@app.websocket("/ws/security-alerts")
async def alerts_websocket(websocket: WebSocket) -> None:
    runtime = getattr(app.state, "ids", None)
    if runtime is None:
        await websocket.close(code=1013)
        return

    await runtime.websocket_manager.connect(websocket)
    try:
        for alert in runtime.websocket_manager.get_recent_alerts(limit=20):
            await websocket.send_json(alert)

        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await runtime.websocket_manager.disconnect(websocket)
    except Exception:
        await runtime.websocket_manager.disconnect(websocket)


@app.websocket("/ws/alerts")
async def alerts_websocket_legacy(websocket: WebSocket) -> None:
    await alerts_websocket(websocket)
