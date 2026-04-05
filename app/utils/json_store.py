from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any


class JsonEventStore:
    """Append-only JSONL store for traffic and alert events."""

    def __init__(self, storage_dir: str, traffic_file: str, alerts_file: str) -> None:
        self.base_dir = Path(storage_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.traffic_path = self.base_dir / traffic_file
        self.alerts_path = self.base_dir / alerts_file

        self._lock = threading.Lock()

    def save_traffic(self, payload: dict[str, Any]) -> None:
        self._append(self.traffic_path, payload)

    def save_alert(self, payload: dict[str, Any]) -> None:
        self._append(self.alerts_path, payload)

    def _append(self, file_path: Path, payload: dict[str, Any]) -> None:
        event = dict(payload)
        event.setdefault("stored_at", datetime.utcnow().isoformat())

        with self._lock:
            with file_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, default=str) + "\n")
