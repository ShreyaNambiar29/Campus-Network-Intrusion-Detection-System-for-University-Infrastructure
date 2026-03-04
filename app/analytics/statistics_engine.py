from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime

from app.models.incident import Incident


class StatisticsEngine:
    """Computes security analytics aggregates for dashboard widgets."""

    def top_attacking_ips(self, incidents: list[Incident], limit: int = 10) -> list[dict]:
        ranking = Counter(incident.source_ip for incident in incidents)
        return [{"ip": ip, "count": count} for ip, count in ranking.most_common(limit)]

    def attack_distribution(self, incidents: list[Incident]) -> list[dict]:
        distribution = Counter(incident.severity for incident in incidents)
        return [{"severity": severity, "count": count} for severity, count in distribution.items()]

    def attack_types(self, incidents: list[Incident]) -> list[dict]:
        distribution = Counter(incident.attack_type for incident in incidents)
        return [{"attack_type": attack_type, "count": count} for attack_type, count in distribution.items()]

    def daily_threat_trend(self, incidents: list[Incident]) -> list[dict]:
        per_day: dict[str, int] = defaultdict(int)
        for incident in incidents:
            date_key = incident.timestamp.strftime("%Y-%m-%d")
            per_day[date_key] += 1
        ordered = sorted(per_day.items(), key=lambda item: datetime.strptime(item[0], "%Y-%m-%d"))
        return [{"date": date, "count": count} for date, count in ordered]
