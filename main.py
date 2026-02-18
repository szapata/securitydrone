import asyncio
import json
import logging
import sys
from pathlib import Path

from src.core.drone_controller import DroneController, DroneConfig
from src.core.flight_adapter import SimulatedFlightAdapter
from src.triggers.wake_up import WakeUpManager
from src.vision.detector import SuspectDetector, YOLOInferenceBackend
from src.reporting.reporter import CoordinateReporter, LogTransport, HttpTransport
from src.mission import SecurityMission


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class StubCamera:
    """Placeholder camera for development/testing."""

    async def capture_frame(self) -> bytes:
        return b""

    async def start_stream(self) -> None:
        pass

    async def stop_stream(self) -> None:
        pass


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        logger.warning("Config file not found: %s, using defaults", config_path)
        return {}
    with open(path) as f:
        return json.load(f)


def build_system(config: dict) -> SecurityMission:
    drone_cfg = config.get("drone", {})
    detection_cfg = config.get("detection", {})
    reporting_cfg = config.get("reporting", {})

    drone_config = DroneConfig(
        home_latitude=drone_cfg.get("home_latitude", 0.0),
        home_longitude=drone_cfg.get("home_longitude", 0.0),
        patrol_altitude=drone_cfg.get("patrol_altitude", 30.0),
        max_altitude=drone_cfg.get("max_altitude", 120.0),
        scan_radius=drone_cfg.get("scan_radius", 100.0),
        max_tracking_distance=drone_cfg.get("max_tracking_distance", 500.0),
        tracking_timeout_seconds=drone_cfg.get("tracking_timeout_seconds", 300.0),
        scan_timeout_seconds=drone_cfg.get("scan_timeout_seconds", 120.0),
    )

    flight_adapter = SimulatedFlightAdapter()
    drone = DroneController(config=drone_config, flight_adapter=flight_adapter)

    backend_type = detection_cfg.get("model_backend", "stub")
    if backend_type == "yolo":
        inference_backend = YOLOInferenceBackend(
            model_path=detection_cfg.get("model_path", "yolov8n.pt")
        )
    else:
        inference_backend = None

    detector = SuspectDetector(
        confidence_threshold=detection_cfg.get("confidence_threshold", 0.6),
        inference_backend=inference_backend,
    )

    transport_type = reporting_cfg.get("transport", "log")
    if transport_type == "http":
        transport = HttpTransport(
            endpoint_url=reporting_cfg.get("http_endpoint", ""),
            api_key=reporting_cfg.get("api_key"),
        )
    else:
        transport = LogTransport()

    reporter = CoordinateReporter(
        drone_id=drone_cfg.get("id", "security-drone-01"),
        transport=transport,
        report_interval=reporting_cfg.get("report_interval", 1.0),
        buffer_size=reporting_cfg.get("buffer_size", 100),
    )

    camera = StubCamera()
    wake_up_manager = WakeUpManager()

    return SecurityMission(
        drone=drone,
        wake_up_manager=wake_up_manager,
        detector=detector,
        camera=camera,
        reporter=reporter,
    )


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/default.json"
    config = load_config(config_path)

    mission = build_system(config)
    http_port = config.get("triggers", {}).get("http_port", 8000)

    logger.info("Starting Security Drone System")
    logger.info("Trigger API will listen on port %d", http_port)
    logger.info("Send POST to http://localhost:%d/api/trigger to wake up", http_port)

    try:
        asyncio.run(mission.run(http_port=http_port))
    except KeyboardInterrupt:
        logger.info("Shutting down")


if __name__ == "__main__":
    main()
