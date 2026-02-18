import asyncio
import json
import pytest
from src.reporting.reporter import (
    CoordinateReporter,
    PositionReport,
    LogTransport,
    ReportTransport,
)


class MockTransport(ReportTransport):
    def __init__(self, should_fail=False):
        self.sent = []
        self._should_fail = should_fail

    async def send(self, report):
        self.sent.append(report)
        return not self._should_fail


def test_position_report_to_dict():
    report = PositionReport(
        drone_id="drone-01",
        timestamp=1000.0,
        drone_latitude=40.4168,
        drone_longitude=-3.7038,
        drone_altitude=30.0,
        event_type="position_update",
    )
    d = report.to_dict()
    assert d["drone_id"] == "drone-01"
    assert d["drone_position"]["latitude"] == 40.4168
    assert d["target"] is None


def test_position_report_with_target():
    report = PositionReport(
        drone_id="drone-01",
        timestamp=1000.0,
        drone_latitude=40.4168,
        drone_longitude=-3.7038,
        drone_altitude=30.0,
        target_id="target-1",
        target_latitude=40.42,
        target_longitude=-3.71,
        target_type="person",
        event_type="tracking_update",
    )
    d = report.to_dict()
    assert d["target"]["id"] == "target-1"
    assert d["target"]["type"] == "person"


def test_position_report_to_json():
    report = PositionReport(
        drone_id="drone-01",
        timestamp=1000.0,
        drone_latitude=40.4168,
        drone_longitude=-3.7038,
        drone_altitude=30.0,
    )
    j = report.to_json()
    parsed = json.loads(j)
    assert parsed["drone_id"] == "drone-01"


@pytest.mark.asyncio
async def test_reporter_sends_via_transport():
    transport = MockTransport()
    reporter = CoordinateReporter(
        drone_id="drone-01",
        transport=transport,
        report_interval=0.1,
    )
    await reporter.start()
    await reporter.report_position(
        drone_lat=40.4168,
        drone_lon=-3.7038,
        drone_alt=30.0,
    )
    await asyncio.sleep(0.5)
    await reporter.stop()
    assert len(transport.sent) >= 1
    assert transport.sent[0].drone_latitude == 40.4168


@pytest.mark.asyncio
async def test_reporter_buffers_on_failure():
    transport = MockTransport(should_fail=True)
    reporter = CoordinateReporter(
        drone_id="drone-01",
        transport=transport,
        report_interval=0.1,
    )
    await reporter.start()
    await reporter.report_position(
        drone_lat=40.4168,
        drone_lon=-3.7038,
        drone_alt=30.0,
    )
    await asyncio.sleep(0.5)
    reporter._running = False
    assert len(reporter._buffer) >= 1


@pytest.mark.asyncio
async def test_log_transport():
    transport = LogTransport()
    report = PositionReport(
        drone_id="drone-01",
        timestamp=1000.0,
        drone_latitude=40.4168,
        drone_longitude=-3.7038,
        drone_altitude=30.0,
    )
    result = await transport.send(report)
    assert result is True
