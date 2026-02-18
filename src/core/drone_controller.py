import asyncio
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


class DroneState(Enum):
    IDLE = "idle"
    WAKING_UP = "waking_up"
    TAKING_OFF = "taking_off"
    SCANNING = "scanning"
    TRACKING = "tracking"
    RETURNING = "returning"
    LANDING = "landing"
    ERROR = "error"


VALID_TRANSITIONS = {
    DroneState.IDLE: {DroneState.WAKING_UP},
    DroneState.WAKING_UP: {DroneState.TAKING_OFF, DroneState.ERROR},
    DroneState.TAKING_OFF: {DroneState.SCANNING, DroneState.ERROR},
    DroneState.SCANNING: {DroneState.TRACKING, DroneState.RETURNING, DroneState.ERROR},
    DroneState.TRACKING: {DroneState.SCANNING, DroneState.RETURNING, DroneState.ERROR},
    DroneState.RETURNING: {DroneState.LANDING, DroneState.ERROR},
    DroneState.LANDING: {DroneState.IDLE, DroneState.ERROR},
    DroneState.ERROR: {DroneState.IDLE},
}


@dataclass
class Coordinates:
    latitude: float
    longitude: float
    altitude: float

    def to_dict(self) -> dict:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude": self.altitude,
        }


@dataclass
class DroneConfig:
    home_latitude: float = 0.0
    home_longitude: float = 0.0
    patrol_altitude: float = 30.0
    max_altitude: float = 120.0
    scan_radius: float = 100.0
    max_tracking_distance: float = 500.0
    tracking_timeout_seconds: float = 300.0
    scan_timeout_seconds: float = 120.0


class DroneController:
    def __init__(self, config: DroneConfig, flight_adapter=None):
        self._state = DroneState.IDLE
        self._config = config
        self._flight_adapter = flight_adapter
        self._position = Coordinates(
            config.home_latitude, config.home_longitude, 0.0
        )
        self._mission_active = False
        self._state_listeners: list = []

    @property
    def state(self) -> DroneState:
        return self._state

    @property
    def position(self) -> Coordinates:
        return self._position

    @property
    def config(self) -> DroneConfig:
        return self._config

    def add_state_listener(self, listener) -> None:
        self._state_listeners.append(listener)

    def _transition_to(self, new_state: DroneState) -> None:
        allowed = VALID_TRANSITIONS.get(self._state, set())
        if new_state not in allowed:
            raise InvalidStateTransition(
                f"Cannot transition from {self._state.value} to {new_state.value}"
            )
        old_state = self._state
        self._state = new_state
        logger.info("State transition: %s -> %s", old_state.value, new_state.value)
        for listener in self._state_listeners:
            listener(old_state, new_state)

    async def wake_up(self, trigger_info: dict) -> None:
        logger.info("Wake-up received: %s", trigger_info)
        self._transition_to(DroneState.WAKING_UP)
        if self._flight_adapter:
            await self._flight_adapter.initialize()
        self._mission_active = True

    async def take_off(self) -> None:
        self._transition_to(DroneState.TAKING_OFF)
        target_alt = self._config.patrol_altitude
        logger.info("Taking off to altitude %.1fm", target_alt)
        if self._flight_adapter:
            await self._flight_adapter.take_off(target_alt)
        self._position.altitude = target_alt

    async def begin_scan(self) -> None:
        self._transition_to(DroneState.SCANNING)
        logger.info(
            "Scanning area with radius %.1fm at position (%.6f, %.6f)",
            self._config.scan_radius,
            self._position.latitude,
            self._position.longitude,
        )

    async def begin_tracking(self) -> None:
        self._transition_to(DroneState.TRACKING)
        logger.info("Tracking mode activated")

    async def move_to(self, target: Coordinates) -> None:
        logger.info(
            "Moving to (%.6f, %.6f, %.1f)",
            target.latitude,
            target.longitude,
            target.altitude,
        )
        if self._flight_adapter:
            await self._flight_adapter.move_to(
                target.latitude, target.longitude, target.altitude
            )
        self._position = Coordinates(
            target.latitude, target.longitude, target.altitude
        )

    async def return_home(self) -> None:
        self._transition_to(DroneState.RETURNING)
        home = Coordinates(
            self._config.home_latitude,
            self._config.home_longitude,
            self._config.patrol_altitude,
        )
        logger.info("Returning to home position")
        await self.move_to(home)

    async def land(self) -> None:
        self._transition_to(DroneState.LANDING)
        logger.info("Landing")
        if self._flight_adapter:
            await self._flight_adapter.land()
        self._position.altitude = 0.0
        self._mission_active = False
        self._transition_to(DroneState.IDLE)

    async def emergency_stop(self) -> None:
        logger.warning("Emergency stop triggered")
        self._state = DroneState.ERROR
        self._mission_active = False
        if self._flight_adapter:
            await self._flight_adapter.emergency_land()

    def reset(self) -> None:
        if self._state != DroneState.ERROR:
            raise InvalidStateTransition("Can only reset from ERROR state")
        self._state = DroneState.IDLE
        self._mission_active = False
        logger.info("Drone reset to IDLE")


class InvalidStateTransition(Exception):
    pass
