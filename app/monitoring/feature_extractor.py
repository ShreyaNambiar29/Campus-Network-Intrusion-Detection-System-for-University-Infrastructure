from __future__ import annotations

from collections import defaultdict, deque
from time import time
from typing import Any


class FeatureExtractor:
    """Converts raw packets into enriched feature vectors for detection."""

    PROTOCOL_MAP = {"TCP": 1, "UDP": 2, "ICMP": 3, "OTHER": 4}

    def __init__(self, window_seconds: int = 15) -> None:
        self.window_seconds = window_seconds
        self.flow_timestamps: dict[tuple[str, str, str], deque[float]] = defaultdict(deque)

    def extract(self, packet: dict[str, Any]) -> dict[str, Any]:
        now = float(packet.get("timestamp", time()))
        source_ip = packet.get("source_ip", "0.0.0.0")
        destination_ip = packet.get("destination_ip", "0.0.0.0")
        protocol = str(packet.get("protocol", "OTHER")).upper()

        key = (source_ip, destination_ip, protocol)
        timestamps = self.flow_timestamps[key]
        timestamps.append(now)

        while timestamps and now - timestamps[0] > self.window_seconds:
            timestamps.popleft()

        frequency = len(timestamps)
        session_duration = now - timestamps[0] if timestamps else 0.0

        return {
            "source_ip": source_ip,
            "destination_ip": destination_ip,
            "protocol": protocol,
            "protocol_id": self.PROTOCOL_MAP.get(protocol, self.PROTOCOL_MAP["OTHER"]),
            "source_port": int(packet.get("source_port", 0) or 0),
            "destination_port": int(packet.get("destination_port", 0) or 0),
            "packet_size": int(packet.get("packet_size", 0) or 0),
            "timestamp": now,
            "frequency": frequency,
            "session_duration": round(session_duration, 3),
        }
