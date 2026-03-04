from __future__ import annotations

import asyncio
from collections import deque
from typing import Any

from fastapi import WebSocket


class WebSocketManager:
    """Manages active WebSocket subscribers for security alerts."""

    def __init__(self, max_recent_alerts: int = 200) -> None:
        self.connections: set[WebSocket] = set()
        self.recent_alerts: deque[dict[str, Any]] = deque(maxlen=max_recent_alerts)
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self.connections.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self.connections.discard(websocket)

    async def broadcast(self, payload: dict[str, Any]) -> None:
        self.recent_alerts.appendleft(payload)
        async with self._lock:
            dead_connections: list[WebSocket] = []
            for websocket in self.connections:
                try:
                    await websocket.send_json(payload)
                except Exception:
                    dead_connections.append(websocket)
            for websocket in dead_connections:
                self.connections.discard(websocket)

    def broadcast_from_thread(self, loop: asyncio.AbstractEventLoop, payload: dict[str, Any]) -> None:
        asyncio.run_coroutine_threadsafe(self.broadcast(payload), loop)

    def get_recent_alerts(self, limit: int = 50) -> list[dict[str, Any]]:
        return list(self.recent_alerts)[:limit]
