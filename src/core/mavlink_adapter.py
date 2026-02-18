import asyncio
import logging
from typing import Optional

from .flight_adapter import FlightAdapter

logger = logging.getLogger(__name__)


class MAVLinkAdapter(FlightAdapter):
    """Flight adapter for ArduPilot/PX4 via MAVLink protocol.

    Uses pymavlink to communicate with the flight controller.
    Supports connection via:
      - Serial:    /dev/ttyACM0 (USB) or /dev/ttyAMA0 (UART on Raspberry Pi)
      - UDP:       udp:127.0.0.1:14550 (SITL simulator or MAVProxy)
      - TCP:       tcp:127.0.0.1:5760

    Hardware setup (Raspberry Pi example):
      Raspberry Pi GPIO UART <--> Pixhawk TELEM2 port
      or
      Raspberry Pi USB <--> Pixhawk USB port

    Install: pip install pymavlink
    """

    # MAVLink mode constants for ArduCopter
    MODE_GUIDED = 4
    MODE_LAND = 9
    MODE_RTL = 6

    def __init__(
        self,
        connection_string: str = "/dev/ttyACM0",
        baud_rate: int = 57600,
        source_system: int = 255,
        source_component: int = 0,
    ):
        self._connection_string = connection_string
        self._baud_rate = baud_rate
        self._source_system = source_system
        self._source_component = source_component
        self._connection = None
        self._armed = False

    async def initialize(self) -> None:
        from pymavlink import mavutil

        logger.info(
            "Connecting to flight controller at %s (baud=%d)",
            self._connection_string,
            self._baud_rate,
        )

        loop = asyncio.get_event_loop()
        self._connection = await loop.run_in_executor(
            None,
            lambda: mavutil.mavlink_connection(
                self._connection_string,
                baud=self._baud_rate,
                source_system=self._source_system,
                source_component=self._source_component,
            ),
        )

        logger.info("Waiting for heartbeat...")
        await loop.run_in_executor(None, self._connection.wait_heartbeat)
        logger.info(
            "Heartbeat received (system %d, component %d)",
            self._connection.target_system,
            self._connection.target_component,
        )

        await self._pre_flight_checks()

    async def take_off(self, altitude: float) -> None:
        logger.info("Arming and taking off to %.1fm", altitude)
        await self._set_mode(self.MODE_GUIDED)
        await self._arm()
        await self._send_takeoff(altitude)
        await self._wait_for_altitude(altitude, tolerance=1.0)
        logger.info("Reached target altitude %.1fm", altitude)

    async def land(self) -> None:
        logger.info("Initiating landing")
        await self._set_mode(self.MODE_LAND)
        await self._wait_for_altitude(0.5, below=True)
        await self._disarm()
        logger.info("Landed and disarmed")

    async def emergency_land(self) -> None:
        logger.warning("EMERGENCY: Switching to LAND mode immediately")
        try:
            await self._set_mode(self.MODE_LAND)
        except Exception:
            logger.exception("Failed to set LAND mode during emergency")

    async def move_to(self, lat: float, lon: float, alt: float) -> None:
        logger.info("Flying to (%.6f, %.6f, %.1fm)", lat, lon, alt)
        conn = self._connection

        conn.mav.set_position_target_global_int_send(
            0,  # time_boot_ms
            conn.target_system,
            conn.target_component,
            6,  # MAV_FRAME_GLOBAL_RELATIVE_ALT_INT
            0b0000111111111000,  # type_mask: position only
            int(lat * 1e7),
            int(lon * 1e7),
            alt,
            0, 0, 0,  # velocity
            0, 0, 0,  # acceleration
            0, 0,     # yaw, yaw_rate
        )

        await self._wait_for_position(lat, lon, tolerance_m=2.0)

    async def get_battery_level(self) -> float:
        loop = asyncio.get_event_loop()
        msg = await loop.run_in_executor(
            None,
            lambda: self._connection.recv_match(
                type="BATTERY_STATUS", blocking=True, timeout=5
            ),
        )
        if msg:
            return msg.battery_remaining
        return -1.0

    async def get_gps_position(self) -> tuple[float, float, float]:
        loop = asyncio.get_event_loop()
        msg = await loop.run_in_executor(
            None,
            lambda: self._connection.recv_match(
                type="GLOBAL_POSITION_INT", blocking=True, timeout=5
            ),
        )
        if msg:
            return (msg.lat / 1e7, msg.lon / 1e7, msg.relative_alt / 1000.0)
        raise RuntimeError("No GPS position available")

    # -- Internal helpers --

    async def _pre_flight_checks(self) -> None:
        battery = await self.get_battery_level()
        if 0 < battery < 20:
            raise RuntimeError(f"Battery too low for flight: {battery}%")
        logger.info("Pre-flight checks passed (battery: %.0f%%)", battery)

        try:
            lat, lon, alt = await self.get_gps_position()
            logger.info("GPS fix: (%.6f, %.6f, %.1fm)", lat, lon, alt)
        except RuntimeError:
            logger.warning("No GPS fix yet - proceed with caution")

    async def _set_mode(self, mode_id: int) -> None:
        conn = self._connection
        conn.mav.set_mode_send(
            conn.target_system,
            1,  # MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
            mode_id,
        )
        loop = asyncio.get_event_loop()
        ack = await loop.run_in_executor(
            None,
            lambda: conn.recv_match(type="COMMAND_ACK", blocking=True, timeout=5),
        )
        if ack:
            logger.info("Mode changed to %d (result=%d)", mode_id, ack.result)

    async def _arm(self) -> None:
        conn = self._connection
        conn.mav.command_long_send(
            conn.target_system,
            conn.target_component,
            400,  # MAV_CMD_COMPONENT_ARM_DISARM
            0,
            1, 0, 0, 0, 0, 0, 0,  # param1=1 means arm
        )
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, lambda: conn.motors_armed_wait()
        )
        self._armed = True
        logger.info("Motors armed")

    async def _disarm(self) -> None:
        conn = self._connection
        conn.mav.command_long_send(
            conn.target_system,
            conn.target_component,
            400,  # MAV_CMD_COMPONENT_ARM_DISARM
            0,
            0, 0, 0, 0, 0, 0, 0,  # param1=0 means disarm
        )
        self._armed = False
        logger.info("Motors disarmed")

    async def _send_takeoff(self, altitude: float) -> None:
        conn = self._connection
        conn.mav.command_long_send(
            conn.target_system,
            conn.target_component,
            22,  # MAV_CMD_NAV_TAKEOFF
            0,
            0, 0, 0, 0, 0, 0,
            altitude,
        )

    async def _wait_for_altitude(
        self, target: float, tolerance: float = 1.0, below: bool = False, timeout: float = 30.0
    ) -> None:
        loop = asyncio.get_event_loop()
        start = loop.time()
        while (loop.time() - start) < timeout:
            try:
                _, _, alt = await self.get_gps_position()
                if below and alt <= target:
                    return
                if not below and abs(alt - target) <= tolerance:
                    return
            except RuntimeError:
                pass
            await asyncio.sleep(0.5)
        logger.warning("Altitude wait timed out (target=%.1f)", target)

    async def _wait_for_position(
        self, target_lat: float, target_lon: float, tolerance_m: float = 2.0, timeout: float = 60.0
    ) -> None:
        from ..tracking.tracker import SuspectTracker

        loop = asyncio.get_event_loop()
        start = loop.time()
        while (loop.time() - start) < timeout:
            try:
                lat, lon, _ = await self.get_gps_position()
                dist = SuspectTracker._haversine_distance(lat, lon, target_lat, target_lon)
                if dist <= tolerance_m:
                    return
            except RuntimeError:
                pass
            await asyncio.sleep(0.5)
        logger.warning("Position wait timed out")
