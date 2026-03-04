from __future__ import annotations

from collections import Counter


class TrafficAnalyzer:
    """Aggregates packet streams into dashboard-level traffic metrics."""

    def summarize(self, packets: list[dict]) -> dict:
        total_bytes = sum(int(packet.get("packet_size", 0)) for packet in packets)
        protocol_distribution = Counter(packet.get("protocol", "OTHER") for packet in packets)
        top_sources = Counter(packet.get("source_ip", "0.0.0.0") for packet in packets).most_common(10)
        return {
            "packet_count": len(packets),
            "total_bytes": total_bytes,
            "protocol_distribution": dict(protocol_distribution),
            "top_sources": [{"ip": ip, "count": count} for ip, count in top_sources],
        }
