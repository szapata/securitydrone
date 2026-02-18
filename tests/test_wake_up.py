import asyncio
import pytest
from src.triggers.wake_up import WakeUpManager, TriggerEvent, TriggerType


@pytest.fixture
def manager():
    return WakeUpManager()


def make_event(trigger_type=TriggerType.MANUAL, source_id="test-sensor"):
    return TriggerEvent(
        trigger_type=trigger_type,
        source_id=source_id,
        latitude=40.4168,
        longitude=-3.7038,
        timestamp=1000.0,
    )


@pytest.mark.asyncio
async def test_fire_trigger_calls_callback(manager):
    received = []

    async def callback(event):
        received.append(event)

    manager.on_wake_up(callback)
    event = make_event()
    await manager.fire_trigger(event)

    # Process one cycle of the trigger queue
    manager._running = True
    task = asyncio.create_task(manager._process_triggers())
    await asyncio.sleep(0.2)
    manager._running = False
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert len(received) == 1
    assert received[0].source_id == "test-sensor"


@pytest.mark.asyncio
async def test_multiple_callbacks(manager):
    results_a = []
    results_b = []

    async def cb_a(event):
        results_a.append(event)

    async def cb_b(event):
        results_b.append(event)

    manager.on_wake_up(cb_a)
    manager.on_wake_up(cb_b)

    await manager.fire_trigger(make_event())

    manager._running = True
    task = asyncio.create_task(manager._process_triggers())
    await asyncio.sleep(0.2)
    manager._running = False
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert len(results_a) == 1
    assert len(results_b) == 1


def test_trigger_event_to_dict():
    event = make_event(TriggerType.ALARM_SENSOR, "alarm-01")
    d = event.to_dict()
    assert d["trigger_type"] == "alarm_sensor"
    assert d["source_id"] == "alarm-01"
    assert d["latitude"] == 40.4168
    assert d["longitude"] == -3.7038


@pytest.mark.asyncio
async def test_callback_error_does_not_stop_processing(manager):
    good_results = []

    async def bad_callback(event):
        raise ValueError("boom")

    async def good_callback(event):
        good_results.append(event)

    manager.on_wake_up(bad_callback)
    manager.on_wake_up(good_callback)

    await manager.fire_trigger(make_event())

    manager._running = True
    task = asyncio.create_task(manager._process_triggers())
    await asyncio.sleep(0.2)
    manager._running = False
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert len(good_results) == 1
