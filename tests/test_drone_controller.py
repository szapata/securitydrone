import asyncio
import pytest
from src.core.drone_controller import (
    DroneController,
    DroneConfig,
    DroneState,
    Coordinates,
    InvalidStateTransition,
)
from src.core.flight_adapter import SimulatedFlightAdapter


@pytest.fixture
def config():
    return DroneConfig(
        home_latitude=40.4168,
        home_longitude=-3.7038,
        patrol_altitude=30.0,
        max_tracking_distance=500.0,
    )


@pytest.fixture
def drone(config):
    adapter = SimulatedFlightAdapter()
    return DroneController(config=config, flight_adapter=adapter)


def test_initial_state(drone):
    assert drone.state == DroneState.IDLE


def test_position_starts_at_home(config, drone):
    assert drone.position.latitude == config.home_latitude
    assert drone.position.longitude == config.home_longitude
    assert drone.position.altitude == 0.0


@pytest.mark.asyncio
async def test_wake_up_transitions(drone):
    await drone.wake_up({"source": "test"})
    assert drone.state == DroneState.WAKING_UP


@pytest.mark.asyncio
async def test_full_takeoff_sequence(drone):
    await drone.wake_up({"source": "test"})
    await drone.take_off()
    assert drone.state == DroneState.TAKING_OFF
    assert drone.position.altitude == 30.0


@pytest.mark.asyncio
async def test_scan_after_takeoff(drone):
    await drone.wake_up({"source": "test"})
    await drone.take_off()
    await drone.begin_scan()
    assert drone.state == DroneState.SCANNING


@pytest.mark.asyncio
async def test_tracking_after_scan(drone):
    await drone.wake_up({"source": "test"})
    await drone.take_off()
    await drone.begin_scan()
    await drone.begin_tracking()
    assert drone.state == DroneState.TRACKING


@pytest.mark.asyncio
async def test_return_and_land(drone, config):
    await drone.wake_up({"source": "test"})
    await drone.take_off()
    await drone.begin_scan()
    await drone.return_home()
    assert drone.state == DroneState.RETURNING
    assert drone.position.latitude == config.home_latitude
    await drone.land()
    assert drone.state == DroneState.IDLE
    assert drone.position.altitude == 0.0


@pytest.mark.asyncio
async def test_invalid_transition_raises(drone):
    with pytest.raises(InvalidStateTransition):
        await drone.take_off()  # Can't take off from IDLE directly


@pytest.mark.asyncio
async def test_emergency_stop(drone):
    await drone.wake_up({"source": "test"})
    await drone.take_off()
    await drone.emergency_stop()
    assert drone.state == DroneState.ERROR


def test_reset_from_error(drone):
    drone._state = DroneState.ERROR
    drone.reset()
    assert drone.state == DroneState.IDLE


def test_reset_from_non_error_raises(drone):
    with pytest.raises(InvalidStateTransition):
        drone.reset()


@pytest.mark.asyncio
async def test_move_to(drone):
    await drone.wake_up({"source": "test"})
    await drone.take_off()
    target = Coordinates(40.42, -3.71, 30.0)
    await drone.move_to(target)
    assert drone.position.latitude == 40.42
    assert drone.position.longitude == -3.71


def test_state_listener_called(drone):
    transitions = []
    drone.add_state_listener(lambda old, new: transitions.append((old, new)))
    asyncio.run(drone.wake_up({"source": "test"}))
    assert len(transitions) == 1
    assert transitions[0] == (DroneState.IDLE, DroneState.WAKING_UP)


def test_coordinates_to_dict():
    c = Coordinates(40.4168, -3.7038, 30.0)
    d = c.to_dict()
    assert d["latitude"] == 40.4168
    assert d["longitude"] == -3.7038
    assert d["altitude"] == 30.0
