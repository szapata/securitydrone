import asyncio
import logging
import math
import time
from dataclasses import dataclass
from typing import Optional

from ..core.drone_controller import DroneController, Coordinates, DroneState
from ..vision.detector import SuspectDetector, DetectionResult, CameraSource

logger = logging.getLogger(__name__)


@dataclass
class TrackingTarget:
    target_id: str
    last_detection: DetectionResult
    estimated_position: Coordinates
    heading_degrees: float = 0.0
    speed_mps: float = 0.0
    last_update: float = 0.0
    lost_count: int = 0


class SuspectTracker:
    """Follows a detected suspect by continuously scanning and
    commanding the drone to keep the target in sight."""

    def __init__(
        self,
        drone: DroneController,
        detector: SuspectDetector,
        camera: CameraSource,
        max_lost_frames: int = 30,
        tracking_interval: float = 0.5,
    ):
        self._drone = drone
        self._detector = detector
        self._camera = camera
        self._max_lost_frames = max_lost_frames
        self._tracking_interval = tracking_interval
        self._current_target: Optional[TrackingTarget] = None
        self._tracking = False
        self._on_position_update: list = []
        self._on_target_lost: list = []

    @property
    def current_target(self) -> Optional[TrackingTarget]:
        return self._current_target

    @property
    def is_tracking(self) -> bool:
        return self._tracking

    def on_position_update(self, callback) -> None:
        self._on_position_update.append(callback)

    def on_target_lost(self, callback) -> None:
        self._on_target_lost.append(callback)

    async def start_tracking(self, initial_detection: DetectionResult) -> None:
        self._current_target = TrackingTarget(
            target_id=f"target-{int(time.time())}",
            last_detection=initial_detection,
            estimated_position=Coordinates(
                initial_detection.estimated_latitude,
                initial_detection.estimated_longitude,
                self._drone.position.altitude,
            ),
            last_update=time.time(),
        )
        self._tracking = True

        if self._drone.state == DroneState.SCANNING:
            await self._drone.begin_tracking()

        logger.info(
            "Tracking started for %s (%s, confidence=%.2f)",
            self._current_target.target_id,
            initial_detection.target_type.value,
            initial_detection.confidence,
        )
        await self._tracking_loop()

    async def stop_tracking(self) -> None:
        self._tracking = False
        self._current_target = None
        logger.info("Tracking stopped")

    async def _tracking_loop(self) -> None:
        while self._tracking and self._current_target:
            try:
                detections = await self._detector.detect_from_camera(self._camera)
                matched = self._match_target(detections)

                if matched:
                    self._current_target.lost_count = 0
                    self._update_target(matched)
                    await self._follow_target()
                    await self._notify_position()
                else:
                    self._current_target.lost_count += 1
                    logger.debug(
                        "Target lost frame %d/%d",
                        self._current_target.lost_count,
                        self._max_lost_frames,
                    )
                    if self._current_target.lost_count >= self._max_lost_frames:
                        await self._handle_target_lost()
                        break

                await asyncio.sleep(self._tracking_interval)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in tracking loop")
                await asyncio.sleep(1.0)

    def _match_target(
        self, detections: list[DetectionResult]
    ) -> Optional[DetectionResult]:
        if not detections or not self._current_target:
            return None

        target = self._current_target
        best_match = None
        best_distance = float("inf")

        for det in detections:
            if det.target_type != target.last_detection.target_type:
                continue
            dist = self._bbox_distance(
                target.last_detection.bounding_box.center,
                det.bounding_box.center,
            )
            if dist < best_distance:
                best_distance = dist
                best_match = det

        return best_match

    def _update_target(self, detection: DetectionResult) -> None:
        now = time.time()
        target = self._current_target
        dt = now - target.last_update

        if dt > 0 and target.last_detection.estimated_latitude != 0:
            dlat = detection.estimated_latitude - target.estimated_position.latitude
            dlon = detection.estimated_longitude - target.estimated_position.longitude
            dist = math.sqrt(dlat ** 2 + dlon ** 2) * 111_000  # rough m/deg
            target.speed_mps = dist / dt
            if dlat != 0 or dlon != 0:
                target.heading_degrees = math.degrees(math.atan2(dlon, dlat)) % 360

        target.last_detection = detection
        target.estimated_position = Coordinates(
            detection.estimated_latitude,
            detection.estimated_longitude,
            self._drone.position.altitude,
        )
        target.last_update = now

    async def _follow_target(self) -> None:
        target = self._current_target
        target_pos = Coordinates(
            target.estimated_position.latitude,
            target.estimated_position.longitude,
            self._drone.config.patrol_altitude,
        )

        home = self._drone.config
        dist_from_home = self._haversine_distance(
            target.estimated_position.latitude,
            target.estimated_position.longitude,
            home.home_latitude,
            home.home_longitude,
        )

        if dist_from_home > self._drone.config.max_tracking_distance:
            logger.warning(
                "Target exceeded max tracking distance (%.0fm). Returning.",
                dist_from_home,
            )
            await self.stop_tracking()
            return

        await self._drone.move_to(target_pos)

    async def _notify_position(self) -> None:
        target = self._current_target
        for callback in self._on_position_update:
            try:
                await callback(target)
            except Exception:
                logger.exception("Error in position update callback")

    async def _handle_target_lost(self) -> None:
        logger.warning("Target lost after %d frames", self._max_lost_frames)
        target = self._current_target
        self._tracking = False
        for callback in self._on_target_lost:
            try:
                await callback(target)
            except Exception:
                logger.exception("Error in target-lost callback")
        self._current_target = None

    @staticmethod
    def _bbox_distance(
        center_a: tuple[int, int], center_b: tuple[int, int]
    ) -> float:
        return math.sqrt(
            (center_a[0] - center_b[0]) ** 2 + (center_a[1] - center_b[1]) ** 2
        )

    @staticmethod
    def _haversine_distance(
        lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        R = 6_371_000  # Earth radius in meters
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)
        a = (
            math.sin(dphi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
        )
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
