import asyncio
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class FlightAdapter(ABC):
    """Abstract interface for drone flight hardware.

    Implement this class to integrate with specific drone SDKs
    (e.g., DJI Mobile SDK, MAVLink/ArduPilot, PX4).
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Power on motors and perform pre-flight checks."""

    @abstractmethod
    async def take_off(self, altitude: float) -> None:
        """Take off and hover at the specified altitude in meters."""

    @abstractmethod
    async def land(self) -> None:
        """Perform a controlled landing at the current position."""

    @abstractmethod
    async def emergency_land(self) -> None:
        """Immediately cut throttle and descend."""

    @abstractmethod
    async def move_to(self, lat: float, lon: float, alt: float) -> None:
        """Fly to GPS coordinates at the given altitude."""

    @abstractmethod
    async def get_battery_level(self) -> float:
        """Return battery percentage (0.0 - 100.0)."""

    @abstractmethod
    async def get_gps_position(self) -> tuple[float, float, float]:
        """Return current (latitude, longitude, altitude)."""


class SimulatedFlightAdapter(FlightAdapter):
    """Simulated flight adapter for testing without real hardware."""

    def __init__(self):
        self._lat = 0.0
        self._lon = 0.0
        self._alt = 0.0
        self._battery = 100.0
        self._airborne = False

    async def initialize(self) -> None:
        logger.info("[SIM] Initializing drone systems")
        await asyncio.sleep(0.1)

    async def take_off(self, altitude: float) -> None:
        logger.info("[SIM] Taking off to %.1fm", altitude)
        self._alt = altitude
        self._airborne = True
        await asyncio.sleep(0.1)

    async def land(self) -> None:
        logger.info("[SIM] Landing")
        self._alt = 0.0
        self._airborne = False
        await asyncio.sleep(0.1)

    async def emergency_land(self) -> None:
        logger.warning("[SIM] Emergency landing!")
        self._alt = 0.0
        self._airborne = False

    async def move_to(self, lat: float, lon: float, alt: float) -> None:
        logger.info("[SIM] Moving to (%.6f, %.6f, %.1f)", lat, lon, alt)
        self._lat = lat
        self._lon = lon
        self._alt = alt
        self._battery = max(0.0, self._battery - 0.5)
        await asyncio.sleep(0.1)

    async def get_battery_level(self) -> float:
        return self._battery

    async def get_gps_position(self) -> tuple[float, float, float]:
        return (self._lat, self._lon, self._alt)
