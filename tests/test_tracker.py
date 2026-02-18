import pytest
from src.tracking.tracker import SuspectTracker


def test_haversine_distance_zero():
    dist = SuspectTracker._haversine_distance(40.0, -3.0, 40.0, -3.0)
    assert dist == 0.0


def test_haversine_distance_known():
    # Madrid to Barcelona ~ 504 km
    dist = SuspectTracker._haversine_distance(
        40.4168, -3.7038, 41.3851, 2.1734
    )
    assert 490_000 < dist < 520_000


def test_bbox_distance():
    dist = SuspectTracker._bbox_distance((0, 0), (3, 4))
    assert abs(dist - 5.0) < 0.01


def test_bbox_distance_same_point():
    dist = SuspectTracker._bbox_distance((100, 200), (100, 200))
    assert dist == 0.0
