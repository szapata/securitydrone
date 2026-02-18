import asyncio
import pytest
from src.vision.detector import (
    SuspectDetector,
    DetectionResult,
    TargetType,
    BoundingBox,
)


class FakeBackend:
    def __init__(self, detections):
        self._detections = detections

    async def run_inference(self, frame):
        return self._detections


class FakeCamera:
    async def capture_frame(self):
        return b"\x00" * 100

    async def start_stream(self):
        pass

    async def stop_stream(self):
        pass


def make_detection(target_type=TargetType.PERSON, confidence=0.8):
    return DetectionResult(
        target_type=target_type,
        confidence=confidence,
        bounding_box=BoundingBox(x=100, y=100, width=50, height=120),
        estimated_latitude=40.4168,
        estimated_longitude=-3.7038,
    )


@pytest.mark.asyncio
async def test_detect_returns_high_confidence():
    backend = FakeBackend([make_detection(confidence=0.9)])
    detector = SuspectDetector(confidence_threshold=0.6, inference_backend=backend)
    results = await detector.detect(b"\x00")
    assert len(results) == 1
    assert results[0].confidence == 0.9


@pytest.mark.asyncio
async def test_detect_filters_low_confidence():
    backend = FakeBackend([make_detection(confidence=0.3)])
    detector = SuspectDetector(confidence_threshold=0.6, inference_backend=backend)
    results = await detector.detect(b"\x00")
    assert len(results) == 0


@pytest.mark.asyncio
async def test_detect_filters_non_suspect_types():
    det = DetectionResult(
        target_type=TargetType.UNKNOWN,
        confidence=0.95,
        bounding_box=BoundingBox(x=0, y=0, width=10, height=10),
    )
    backend = FakeBackend([det])
    detector = SuspectDetector(confidence_threshold=0.5, inference_backend=backend)
    results = await detector.detect(b"\x00")
    assert len(results) == 0


@pytest.mark.asyncio
async def test_detect_from_camera():
    backend = FakeBackend([make_detection()])
    detector = SuspectDetector(confidence_threshold=0.6, inference_backend=backend)
    camera = FakeCamera()
    results = await detector.detect_from_camera(camera)
    assert len(results) == 1


@pytest.mark.asyncio
async def test_detect_vehicle():
    backend = FakeBackend([make_detection(TargetType.VEHICLE, 0.85)])
    detector = SuspectDetector(confidence_threshold=0.6, inference_backend=backend)
    results = await detector.detect(b"\x00")
    assert len(results) == 1
    assert results[0].target_type == TargetType.VEHICLE


def test_set_confidence_threshold():
    detector = SuspectDetector(confidence_threshold=0.5)
    detector.set_confidence_threshold(0.8)
    assert detector._confidence_threshold == 0.8


def test_set_confidence_threshold_clamped():
    detector = SuspectDetector(confidence_threshold=0.5)
    detector.set_confidence_threshold(1.5)
    assert detector._confidence_threshold == 1.0
    detector.set_confidence_threshold(-0.5)
    assert detector._confidence_threshold == 0.0


def test_detection_result_to_dict():
    det = make_detection()
    d = det.to_dict()
    assert d["target_type"] == "person"
    assert d["confidence"] == 0.8
    assert d["bounding_box"]["width"] == 50


def test_bounding_box_center():
    bb = BoundingBox(x=100, y=200, width=50, height=100)
    assert bb.center == (125, 250)


@pytest.mark.asyncio
async def test_stub_backend_returns_empty():
    detector = SuspectDetector(confidence_threshold=0.1)
    results = await detector.detect(b"\x00")
    assert results == []
