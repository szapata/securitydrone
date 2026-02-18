import asyncio
import logging
import json
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable, Awaitable, Optional

logger = logging.getLogger(__name__)


class TriggerType(Enum):
    ALARM_SENSOR = "alarm_sensor"
    MANUAL = "manual"
    API_CALL = "api_call"
    MOTION_SENSOR = "motion_sensor"
    SCHEDULED = "scheduled"


@dataclass
class TriggerEvent:
    trigger_type: TriggerType
    source_id: str
    latitude: float
    longitude: float
    timestamp: float
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "trigger_type": self.trigger_type.value,
            "source_id": self.source_id,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


WakeUpCallback = Callable[[TriggerEvent], Awaitable[None]]


class WakeUpManager:
    """Manages incoming wake-up triggers from multiple sources."""

    def __init__(self):
        self._callbacks: list[WakeUpCallback] = []
        self._trigger_queue: asyncio.Queue[TriggerEvent] = asyncio.Queue()
        self._running = False
        self._http_server = None

    def on_wake_up(self, callback: WakeUpCallback) -> None:
        self._callbacks.append(callback)

    async def fire_trigger(self, event: TriggerEvent) -> None:
        logger.info(
            "Trigger received: type=%s source=%s at (%.6f, %.6f)",
            event.trigger_type.value,
            event.source_id,
            event.latitude,
            event.longitude,
        )
        await self._trigger_queue.put(event)

    async def start(self, http_port: Optional[int] = None) -> None:
        self._running = True
        tasks = [asyncio.create_task(self._process_triggers())]
        if http_port:
            tasks.append(asyncio.create_task(self._start_http_listener(http_port)))
        await asyncio.gather(*tasks)

    async def stop(self) -> None:
        self._running = False
        if self._http_server:
            self._http_server.close()
            await self._http_server.wait_closed()

    async def _process_triggers(self) -> None:
        while self._running:
            try:
                event = await asyncio.wait_for(
                    self._trigger_queue.get(), timeout=1.0
                )
                for callback in self._callbacks:
                    try:
                        await callback(event)
                    except Exception:
                        logger.exception("Error in wake-up callback")
            except asyncio.TimeoutError:
                continue

    async def _start_http_listener(self, port: int) -> None:
        """Minimal HTTP listener for API-based triggers.

        Expects POST requests with JSON body:
        {
            "source_id": "sensor-01",
            "latitude": 40.4168,
            "longitude": -3.7038,
            "trigger_type": "alarm_sensor",
            "metadata": {}
        }
        """
        from aiohttp import web

        async def handle_trigger(request: web.Request) -> web.Response:
            try:
                data = await request.json()
                event = TriggerEvent(
                    trigger_type=TriggerType(data.get("trigger_type", "api_call")),
                    source_id=data["source_id"],
                    latitude=data["latitude"],
                    longitude=data["longitude"],
                    timestamp=asyncio.get_event_loop().time(),
                    metadata=data.get("metadata", {}),
                )
                await self.fire_trigger(event)
                return web.json_response({"status": "accepted"})
            except (KeyError, ValueError) as exc:
                return web.json_response(
                    {"error": str(exc)}, status=400
                )

        async def handle_health(_: web.Request) -> web.Response:
            return web.json_response({"status": "ok"})

        app = web.Application()
        app.router.add_post("/api/trigger", handle_trigger)
        app.router.add_get("/api/health", handle_health)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", port)
        logger.info("HTTP trigger listener started on port %d", port)
        await site.start()

        while self._running:
            await asyncio.sleep(1)

        await runner.cleanup()
