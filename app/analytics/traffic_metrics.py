from __future__ import annotations

from collections import defaultdict
from datetime import datetime

from app.models.traffic_log import TrafficLog


class TrafficMetrics:
    """Computes traffic volume and protocol metrics from traffic logs."""

    def volume_over_time(self, traffic_logs: list[TrafficLog]) -> list[dict]:
        buckets: dict[str, int] = defaultdict(int)
        for log in traffic_logs:
            bucket = log.timestamp.strftime("%Y-%m-%d %H:%M")
            buckets[bucket] += 1
        ordered = sorted(buckets.items(), key=lambda item: datetime.strptime(item[0], "%Y-%m-%d %H:%M"))
        return [{"minute": minute, "count": count} for minute, count in ordered]
