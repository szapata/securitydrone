import asyncio
import logging
import time
from typing import Optional

from .core.drone_controller import DroneController, DroneState, DroneConfig
from .core.flight_adapter import FlightAdapter
from .triggers.wake_up import WakeUpManager, TriggerEvent
from .vision.detector import SuspectDetector, DetectionResult, CameraSource
from .tracking.tracker import SuspectTracker, TrackingTarget
from .reporting.reporter import CoordinateReporter

logger = logging.getLogger(__name__)


class SecurityMission:
    """Orchestrates a complete security drone mission.

    Lifecycle:
        1. IDLE: Waiting for wake-up trigger
        2. WAKE UP: Trigger received -> initialize systems
        3. TAKE OFF: Ascend to patrol altitude
        4. SCAN: Search area for suspects (persons/vehicles)
        5. TRACK: Follow detected suspect, report coordinates
        6. RETURN: Come back when target lost or timeout
        7. LAND: Touch down and go back to IDLE
    """

    def __init__(
        self,
        drone: DroneController,
        wake_up_manager: WakeUpManager,
        detector: SuspectDetector,
        camera: CameraSource,
        reporter: CoordinateReporter,
    ):
        self._drone = drone
        self._wake_up = wake_up_manager
        self._detector = detector
        self._camera = camera
        self._reporter = reporter
        self._tracker: Optional[SuspectTracker] = None
        self._current_trigger: Optional[TriggerEvent] = None

        self._wake_up.on_wake_up(self._handle_wake_up)

    async def run(self, http_port: Optional[int] = None) -> None:
        logger.info("Security mission system ready. Waiting for triggers...")
        await self._reporter.start()
        await self._wake_up.start(http_port=http_port)

    async def _handle_wake_up(self, event: TriggerEvent) -> None:
        if self._drone.state != DroneState.IDLE:
            logger.warning(
                "Ignoring trigger while drone is in %s state",
                self._drone.state.value,
            )
            return

        self._current_trigger = event
        logger.info("Mission starting for trigger from %s", event.source_id)

        try:
            await self._drone.wake_up(event.to_dict())
            await self._drone.take_off()

            await self._reporter.report_position(
                drone_lat=self._drone.position.latitude,
                drone_lon=self._drone.position.longitude,
                drone_alt=self._drone.position.altitude,
                event_type="mission_started",
            )

            await self._scan_phase(event)

        except Exception:
            logger.exception("Mission error")
            await self._abort_mission()

    async def _scan_phase(self, trigger: TriggerEvent) -> None:
        await self._drone.begin_scan()
        scan_start = time.time()
        timeout = self._drone.config.scan_timeout_seconds

        while self._drone.state == DroneState.SCANNING:
            if time.time() - scan_start > timeout:
                logger.info("Scan timeout reached, returning home")
                break

            detections = await self._detector.detect_from_camera(self._camera)

            if detections:
                best = max(detections, key=lambda d: d.confidence)
                best.estimated_latitude = trigger.latitude
                best.estimated_longitude = trigger.longitude
                await self._start_tracking(best)
                return

            await self._reporter.report_position(
                drone_lat=self._drone.position.latitude,
                drone_lon=self._drone.position.longitude,
                drone_alt=self._drone.position.altitude,
                event_type="scanning",
            )
            await asyncio.sleep(0.5)

        await self._reporter.report_position(
            drone_lat=self._drone.position.latitude,
            drone_lon=self._drone.position.longitude,
            drone_alt=self._drone.position.altitude,
            event_type="scan_complete_no_target",
        )
        await self._return_and_land()

    async def _start_tracking(self, detection: DetectionResult) -> None:
        self._tracker = SuspectTracker(
            drone=self._drone,
            detector=self._detector,
            camera=self._camera,
        )

        self._tracker.on_position_update(self._on_target_position)
        self._tracker.on_target_lost(self._on_target_lost)

        await self._reporter.report_position(
            drone_lat=self._drone.position.latitude,
            drone_lon=self._drone.position.longitude,
            drone_alt=self._drone.position.altitude,
            target_id=f"target-{int(time.time())}",
            target_lat=detection.estimated_latitude,
            target_lon=detection.estimated_longitude,
            target_type=detection.target_type.value,
            event_type="tracking_started",
        )

        await self._tracker.start_tracking(detection)

        # After tracking ends (target lost or max distance)
        await self._return_and_land()

    async def _on_target_position(self, target: TrackingTarget) -> None:
        await self._reporter.report_position(
            drone_lat=self._drone.position.latitude,
            drone_lon=self._drone.position.longitude,
            drone_alt=self._drone.position.altitude,
            target_id=target.target_id,
            target_lat=target.estimated_position.latitude,
            target_lon=target.estimated_position.longitude,
            target_type=target.last_detection.target_type.value,
            target_speed=target.speed_mps,
            target_heading=target.heading_degrees,
            event_type="tracking_update",
        )

    async def _on_target_lost(self, target: TrackingTarget) -> None:
        await self._reporter.report_position(
            drone_lat=self._drone.position.latitude,
            drone_lon=self._drone.position.longitude,
            drone_alt=self._drone.position.altitude,
            target_id=target.target_id,
            target_lat=target.estimated_position.latitude,
            target_lon=target.estimated_position.longitude,
            target_type=target.last_detection.target_type.value,
            event_type="target_lost",
        )

    async def _return_and_land(self) -> None:
        try:
            if self._drone.state not in (DroneState.IDLE, DroneState.LANDING, DroneState.ERROR):
                await self._drone.return_home()
                await self._drone.land()
                await self._reporter.report_position(
                    drone_lat=self._drone.position.latitude,
                    drone_lon=self._drone.position.longitude,
                    drone_alt=self._drone.position.altitude,
                    event_type="mission_complete",
                )
        except Exception:
            logger.exception("Error during return and land")
            await self._abort_mission()

    async def _abort_mission(self) -> None:
        logger.warning("Aborting mission")
        await self._drone.emergency_stop()
        await self._reporter.report_position(
            drone_lat=self._drone.position.latitude,
            drone_lon=self._drone.position.longitude,
            drone_alt=self._drone.position.altitude,
            event_type="mission_aborted",
        )
