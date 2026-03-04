from __future__ import annotations

import random
import threading
import time
from collections import deque
from typing import Callable

from app.config import settings
from app.utils.logger import get_logger

try:
    from scapy.all import ICMP, IP, TCP, UDP, sniff  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency behavior
    ICMP = IP = TCP = UDP = None
    sniff = None


logger = get_logger("ids.packet_sniffer")


class PacketSniffer:
    """Captures packet metadata and forwards packet records for analysis."""

    def __init__(self, on_packet: Callable[[dict], None], max_buffer: int = 500) -> None:
        self.on_packet = on_packet
        self.recent_packets: deque[dict] = deque(maxlen=max_buffer)
        self._running = False
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="packet-sniffer")
        self._thread.start()
        logger.info("Packet sniffer started", extra={"extra": {"simulation_mode": settings.simulation_mode}})

    def stop(self) -> None:
        self._running = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        logger.info("Packet sniffer stopped")

    def get_recent_packets(self, limit: int = 50) -> list[dict]:
        return list(self.recent_packets)[-limit:]

    def _run(self) -> None:
        if settings.simulation_mode or sniff is None:
            self._run_simulation_loop()
            return

        sniff(
            iface=settings.sniffer_interface or None,
            prn=self._handle_scapy_packet,
            store=False,
            stop_filter=lambda _: self._stop_event.is_set(),
        )

    def _run_simulation_loop(self) -> None:
        source_pool = [f"198.51.100.{i}" for i in range(1, 50)]
        destination_pool = [f"10.10.{zone}.{host}" for zone in range(1, 5) for host in range(10, 20)]
        protocols = ["TCP", "UDP", "ICMP"]

        while not self._stop_event.is_set():
            packet = {
                "source_ip": random.choice(source_pool),
                "destination_ip": random.choice(destination_pool),
                "protocol": random.choice(protocols),
                "source_port": random.randint(1000, 65535),
                "destination_port": random.choice([21, 22, 53, 80, 123, 443, 3306, 3389, 8080]),
                "packet_size": random.randint(64, 1800),
                "timestamp": time.time(),
            }
            self._record_packet(packet)
            time.sleep(0.05)

    def _handle_scapy_packet(self, packet) -> None:  # pragma: no cover
        if IP is None or not packet.haslayer(IP):
            return

        protocol = "OTHER"
        source_port = 0
        destination_port = 0

        if TCP and packet.haslayer(TCP):
            protocol = "TCP"
            source_port = int(packet[TCP].sport)
            destination_port = int(packet[TCP].dport)
        elif UDP and packet.haslayer(UDP):
            protocol = "UDP"
            source_port = int(packet[UDP].sport)
            destination_port = int(packet[UDP].dport)
        elif ICMP and packet.haslayer(ICMP):
            protocol = "ICMP"

        payload = {
            "source_ip": packet[IP].src,
            "destination_ip": packet[IP].dst,
            "protocol": protocol,
            "source_port": source_port,
            "destination_port": destination_port,
            "packet_size": int(len(packet)),
            "timestamp": float(getattr(packet, "time", time.time())),
        }
        self._record_packet(payload)

    def _record_packet(self, packet_data: dict) -> None:
        self.recent_packets.append(packet_data)
        logger.info(
            "Packet captured",
            extra={
                "extra": {
                    "source_ip": packet_data["source_ip"],
                    "destination_ip": packet_data["destination_ip"],
                    "protocol": packet_data["protocol"],
                    "destination_port": packet_data["destination_port"],
                    "packet_size": packet_data["packet_size"],
                }
            },
        )
        self.on_packet(packet_data)
