from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from app.alerts.severity_engine import SeverityEngine
from app.config import settings
from app.core.threat_classifier import ThreatClassifier
from app.ml.anomaly_detector import AnomalyDetector


class DetectionEngine:
    """Runs rule-based and ML-based intrusion checks over extracted features."""

    def __init__(self, anomaly_detector: AnomalyDetector) -> None:
        self.anomaly_detector = anomaly_detector
        self.threat_classifier = ThreatClassifier()
        self.severity_engine = SeverityEngine()

        self.port_scan_tracker: dict[str, deque[tuple[float, int]]] = defaultdict(deque)
        self.auth_attempt_tracker: dict[tuple[str, int], deque[float]] = defaultdict(deque)
        self.packet_burst_tracker: dict[str, deque[float]] = defaultdict(deque)
        self.target_tracker: dict[str, deque[tuple[float, str]]] = defaultdict(deque)

    def analyze(self, features: dict[str, Any]) -> list[dict[str, Any]]:
        alerts: list[dict[str, Any]] = []
        timestamp = float(features.get("timestamp", 0))
        source_ip = features["source_ip"]
        destination_ip = features["destination_ip"]
        destination_port = int(features.get("destination_port", 0))

        targets = self.target_tracker[source_ip]
        targets.append((timestamp, destination_ip))
        while targets and timestamp - targets[0][0] > 60:
            targets.popleft()
        target_count = len({target for _, target in targets})

        # Rule 1: Port scanning detection
        scan_deque = self.port_scan_tracker[source_ip]
        scan_deque.append((timestamp, destination_port))
        while scan_deque and timestamp - scan_deque[0][0] > 15:
            scan_deque.popleft()
        unique_ports = {port for _, port in scan_deque}
        if len(unique_ports) >= settings.port_scan_threshold:
            attempts = len(scan_deque)
            alerts.append(
                self._build_alert(
                    features,
                    attack_type="Port Scan",
                    severity=self.severity_engine.calculate("HIGH", attempts, target_count),
                    description=f"Multiple ports probed from {source_ip}",
                    threat_score=min(100, 65 + len(unique_ports)),
                    attempts=attempts,
                    target_count=target_count,
                )
            )

        # Rule 2: Repeated connection attempts (brute-force pattern)
        auth_ports = {21, 22, 23, 25, 110, 143, 3306, 3389}
        if destination_port in auth_ports:
            brute_key = (source_ip, destination_port)
            brute_deque = self.auth_attempt_tracker[brute_key]
            brute_deque.append(timestamp)
            while brute_deque and timestamp - brute_deque[0] > 20:
                brute_deque.popleft()
            if len(brute_deque) >= settings.brute_force_threshold:
                attempts = len(brute_deque)
                alerts.append(
                    self._build_alert(
                        features,
                        attack_type="Repeated Connection Attempts",
                        severity=self.severity_engine.calculate("HIGH", attempts, target_count),
                        description=f"Possible brute-force attempts against port {destination_port}",
                        threat_score=min(100, 75 + len(brute_deque)),
                        attempts=attempts,
                        target_count=target_count,
                    )
                )

        # Rule 3: Suspicious packet burst
        burst_deque = self.packet_burst_tracker[source_ip]
        burst_deque.append(timestamp)
        while burst_deque and timestamp - burst_deque[0] > 5:
            burst_deque.popleft()
        if len(burst_deque) >= settings.packet_burst_threshold:
            attempts = len(burst_deque)
            alerts.append(
                self._build_alert(
                    features,
                    attack_type="Suspicious Packet Burst",
                    severity=self.severity_engine.calculate("MEDIUM", attempts, target_count),
                    description=f"Traffic burst from {source_ip}",
                    threat_score=min(100, 40 + len(burst_deque)),
                    attempts=attempts,
                    target_count=target_count,
                )
            )

        # ML anomaly detection
        is_anomaly, anomaly_risk = self.anomaly_detector.predict(features)
        if is_anomaly:
            ml_score = int(55 + anomaly_risk * 45)
            severity = "MEDIUM" if ml_score < 70 else "HIGH"
            attempts = int(features.get("frequency", 1))
            alerts.append(
                self._build_alert(
                    features,
                    attack_type="ML Anomaly",
                    severity=self.severity_engine.calculate(severity, attempts, target_count),
                    description="Isolation Forest flagged abnormal traffic behavior",
                    threat_score=ml_score,
                    attempts=attempts,
                    target_count=target_count,
                )
            )

        return alerts

    def _build_alert(
        self,
        features: dict[str, Any],
        attack_type: str,
        severity: str,
        description: str,
        threat_score: int,
        attempts: int,
        target_count: int,
    ) -> dict[str, Any]:
        classification = self.threat_classifier.classify(
            attack_type=attack_type,
            protocol=str(features.get("protocol", "OTHER")),
            destination_port=int(features.get("destination_port", 0)),
        )
        return {
            "timestamp": features["timestamp"],
            "source_ip": features["source_ip"],
            "destination_ip": features["destination_ip"],
            "attack_type": classification,
            "raw_attack_type": attack_type,
            "severity": severity,
            "description": description,
            "status": "open",
            "threat_score": threat_score,
            "packet_size": features.get("packet_size", 0),
            "protocol": features.get("protocol", "OTHER"),
            "source_port": features.get("source_port", 0),
            "destination_port": features.get("destination_port", 0),
            "attempts": attempts,
            "target_count": target_count,
        }
