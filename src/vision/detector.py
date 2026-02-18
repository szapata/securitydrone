import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Protocol

logger = logging.getLogger(__name__)


class TargetType(Enum):
    PERSON = "person"
    VEHICLE = "vehicle"
    UNKNOWN = "unknown"


@dataclass
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)


@dataclass
class DetectionResult:
    target_type: TargetType
    confidence: float
    bounding_box: BoundingBox
    estimated_latitude: float = 0.0
    estimated_longitude: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "target_type": self.target_type.value,
            "confidence": self.confidence,
            "bounding_box": {
                "x": self.bounding_box.x,
                "y": self.bounding_box.y,
                "width": self.bounding_box.width,
                "height": self.bounding_box.height,
            },
            "estimated_latitude": self.estimated_latitude,
            "estimated_longitude": self.estimated_longitude,
            "timestamp": self.timestamp,
        }


class CameraSource(Protocol):
    """Protocol for camera frame providers."""

    async def capture_frame(self) -> bytes:
        """Capture a single frame and return as raw bytes."""
        ...

    async def start_stream(self) -> None:
        """Begin continuous frame capture."""
        ...

    async def stop_stream(self) -> None:
        """Stop continuous frame capture."""
        ...


class SuspectDetector:
    """Detects persons and vehicles using a vision model.

    This class provides the interface for running object detection
    on camera frames. The actual inference is delegated to a
    configurable backend (YOLO, TensorFlow Lite, cloud API, etc.).
    """

    def __init__(
        self,
        confidence_threshold: float = 0.6,
        inference_backend: Optional["InferenceBackend"] = None,
    ):
        self._confidence_threshold = confidence_threshold
        self._backend = inference_backend or _StubInferenceBackend()

    async def detect(self, frame: bytes) -> list[DetectionResult]:
        raw_detections = await self._backend.run_inference(frame)
        results = []
        for det in raw_detections:
            if det.confidence < self._confidence_threshold:
                continue
            if det.target_type in (TargetType.PERSON, TargetType.VEHICLE):
                results.append(det)
        if results:
            logger.info(
                "Detected %d suspect(s): %s",
                len(results),
                [(r.target_type.value, f"{r.confidence:.2f}") for r in results],
            )
        return results

    async def detect_from_camera(
        self, camera: CameraSource
    ) -> list[DetectionResult]:
        frame = await camera.capture_frame()
        return await self.detect(frame)

    def set_confidence_threshold(self, threshold: float) -> None:
        self._confidence_threshold = max(0.0, min(1.0, threshold))


class InferenceBackend(Protocol):
    async def run_inference(self, frame: bytes) -> list[DetectionResult]:
        ...


class _StubInferenceBackend:
    """Placeholder backend that returns no detections.

    Replace with a real backend (YOLOv8, TFLite, etc.) for production.
    """

    async def run_inference(self, frame: bytes) -> list[DetectionResult]:
        logger.debug("[STUB] No real inference backend configured")
        return []


class YOLOInferenceBackend:
    """Example backend using YOLO for object detection.

    Requires ultralytics package. Loads the model at initialization
    and runs inference on each frame.
    """

    def __init__(self, model_path: str = "yolov8n.pt"):
        self._model_path = model_path
        self._model = None

    def _load_model(self):
        try:
            from ultralytics import YOLO
            self._model = YOLO(self._model_path)
            logger.info("YOLO model loaded from %s", self._model_path)
        except ImportError:
            logger.error(
                "ultralytics package not installed. "
                "Install with: pip install ultralytics"
            )
            raise

    async def run_inference(self, frame: bytes) -> list[DetectionResult]:
        if self._model is None:
            self._load_model()

        import numpy as np

        np_frame = np.frombuffer(frame, dtype=np.uint8)

        loop = asyncio.get_event_loop()
        raw_results = await loop.run_in_executor(
            None, lambda: self._model(np_frame, verbose=False)
        )

        detections = []
        # YOLO COCO class IDs: 0=person, 2=car, 5=bus, 7=truck
        suspect_classes = {0: TargetType.PERSON, 2: TargetType.VEHICLE,
                          5: TargetType.VEHICLE, 7: TargetType.VEHICLE}

        for result in raw_results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in suspect_classes:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    DetectionResult(
                        target_type=suspect_classes[cls_id],
                        confidence=float(box.conf[0]),
                        bounding_box=BoundingBox(
                            x=int(x1),
                            y=int(y1),
                            width=int(x2 - x1),
                            height=int(y2 - y1),
                        ),
                    )
                )
        return detections
