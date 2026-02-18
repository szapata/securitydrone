import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PositionReport:
    drone_id: str
    timestamp: float
    drone_latitude: float
    drone_longitude: float
    drone_altitude: float
    target_id: Optional[str] = None
    target_latitude: Optional[float] = None
    target_longitude: Optional[float] = None
    target_type: Optional[str] = None
    target_speed_mps: Optional[float] = None
    target_heading: Optional[float] = None
    event_type: str = "position_update"
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "drone_id": self.drone_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "drone_position": {
                "latitude": self.drone_latitude,
                "longitude": self.drone_longitude,
                "altitude": self.drone_altitude,
            },
            "target": {
                "id": self.target_id,
                "latitude": self.target_latitude,
                "longitude": self.target_longitude,
                "type": self.target_type,
                "speed_mps": self.target_speed_mps,
                "heading": self.target_heading,
            }
            if self.target_id
            else None,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class ReportTransport:
    """Base transport for sending reports. Subclass for HTTP, MQTT, WebSocket, etc."""

    async def send(self, report: PositionReport) -> bool:
        raise NotImplementedError


class HttpTransport(ReportTransport):
    """Sends reports via HTTP POST to the platform API."""

    def __init__(self, endpoint_url: str, api_key: Optional[str] = None):
        self._endpoint_url = endpoint_url
        self._api_key = api_key

    async def send(self, report: PositionReport) -> bool:
        import aiohttp

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._endpoint_url,
                    data=report.to_json(),
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status < 300:
                        return True
                    logger.warning(
                        "Platform responded with status %d", resp.status
                    )
                    return False
        except Exception:
            logger.exception("Failed to send report to platform")
            return False


class LogTransport(ReportTransport):
    """Logs reports locally. Useful as fallback or for development."""

    async def send(self, report: PositionReport) -> bool:
        logger.info("REPORT: %s", report.to_json())
        return True


class CoordinateReporter:
    """Buffers and sends position reports to the monitoring platform."""

    def __init__(
        self,
        drone_id: str,
        transport: Optional[ReportTransport] = None,
        report_interval: float = 1.0,
        buffer_size: int = 100,
    ):
        self._drone_id = drone_id
        self._transport = transport or LogTransport()
        self._report_interval = report_interval
        self._buffer: list[PositionReport] = []
        self._buffer_size = buffer_size
        self._running = False
        self._send_queue: asyncio.Queue[PositionReport] = asyncio.Queue()

    async def start(self) -> None:
        self._running = True
        asyncio.create_task(self._send_loop())
        logger.info("Coordinate reporter started (interval=%.1fs)", self._report_interval)

    async def stop(self) -> None:
        self._running = False
        await self._flush_buffer()
        logger.info("Coordinate reporter stopped")

    async def report_position(
        self,
        drone_lat: float,
        drone_lon: float,
        drone_alt: float,
        target_id: Optional[str] = None,
        target_lat: Optional[float] = None,
        target_lon: Optional[float] = None,
        target_type: Optional[str] = None,
        target_speed: Optional[float] = None,
        target_heading: Optional[float] = None,
        event_type: str = "position_update",
    ) -> None:
        report = PositionReport(
            drone_id=self._drone_id,
            timestamp=time.time(),
            drone_latitude=drone_lat,
            drone_longitude=drone_lon,
            drone_altitude=drone_alt,
            target_id=target_id,
            target_latitude=target_lat,
            target_longitude=target_lon,
            target_type=target_type,
            target_speed_mps=target_speed,
            target_heading=target_heading,
            event_type=event_type,
        )
        await self._send_queue.put(report)

    async def _send_loop(self) -> None:
        while self._running:
            try:
                report = await asyncio.wait_for(
                    self._send_queue.get(), timeout=self._report_interval
                )
                success = await self._transport.send(report)
                if not success:
                    self._buffer_report(report)
            except asyncio.TimeoutError:
                await self._flush_buffer()

    def _buffer_report(self, report: PositionReport) -> None:
        if len(self._buffer) >= self._buffer_size:
            dropped = self._buffer.pop(0)
            logger.warning("Buffer full, dropped oldest report from %f", dropped.timestamp)
        self._buffer.append(report)
        logger.debug("Report buffered (%d in buffer)", len(self._buffer))

    async def _flush_buffer(self) -> None:
        if not self._buffer:
            return
        remaining = []
        for report in self._buffer:
            success = await self._transport.send(report)
            if not success:
                remaining.append(report)
        self._buffer = remaining
        if remaining:
            logger.warning("%d reports still in buffer after flush", len(remaining))
