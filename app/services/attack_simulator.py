from __future__ import annotations

import random
import time


class AttackSimulator:
    """Generates synthetic malicious packet patterns for demos and testing."""

    def __init__(self, source_ip: str = "203.0.113.250", destination_ip: str = "10.10.1.12") -> None:
        self.source_ip = source_ip
        self.destination_ip = destination_ip

    def simulate_port_scan(self) -> list[dict]:
        now = time.time()
        return [
            {
                "source_ip": self.source_ip,
                "destination_ip": self.destination_ip,
                "protocol": "TCP",
                "source_port": random.randint(10000, 60000),
                "destination_port": port,
                "packet_size": random.randint(60, 150),
                "timestamp": now + idx * 0.05,
            }
            for idx, port in enumerate(range(20, 40))
        ]

    def simulate_brute_force(self) -> list[dict]:
        now = time.time()
        target_port = random.choice([22, 3389, 21])
        return [
            {
                "source_ip": self.source_ip,
                "destination_ip": self.destination_ip,
                "protocol": "TCP",
                "source_port": random.randint(10000, 60000),
                "destination_port": target_port,
                "packet_size": random.randint(80, 220),
                "timestamp": now + idx * 0.2,
            }
            for idx in range(18)
        ]

    def simulate_traffic_spike(self) -> list[dict]:
        now = time.time()
        return [
            {
                "source_ip": self.source_ip,
                "destination_ip": self.destination_ip,
                "protocol": random.choice(["TCP", "UDP"]),
                "source_port": random.randint(10000, 60000),
                "destination_port": random.choice([80, 443, 8080]),
                "packet_size": random.randint(900, 1600),
                "timestamp": now + idx * 0.03,
            }
            for idx in range(90)
        ]

    def generate_all(self) -> list[dict]:
        return [
            *self.simulate_port_scan(),
            *self.simulate_brute_force(),
            *self.simulate_traffic_spike(),
        ]
