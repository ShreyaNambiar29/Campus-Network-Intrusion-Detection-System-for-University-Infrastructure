from __future__ import annotations


class ThreatClassifier:
    """Maps raw detections into dashboard-friendly attack categories."""

    ATTACK_TYPE_MAP = {
        "port scan": "Port Scan",
        "repeated connection attempts": "Brute Force",
        "suspicious packet burst": "Suspicious Traffic",
        "ml anomaly": "Abnormal Behavior",
        "dns c2": "Malware Communication",
    }

    def classify(self, attack_type: str, protocol: str, destination_port: int) -> str:
        key = attack_type.strip().lower()
        if key in self.ATTACK_TYPE_MAP:
            return self.ATTACK_TYPE_MAP[key]

        if protocol.upper() == "UDP" and destination_port in {53, 123}:
            return "Malware Communication"
        return "Suspicious Traffic"
