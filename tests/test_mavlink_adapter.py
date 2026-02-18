import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.core.mavlink_adapter import MAVLinkAdapter


def test_default_connection_string():
    adapter = MAVLinkAdapter()
    assert adapter._connection_string == "/dev/ttyACM0"
    assert adapter._baud_rate == 57600


def test_custom_connection_string():
    adapter = MAVLinkAdapter(
        connection_string="udp:127.0.0.1:14550",
        baud_rate=115200,
    )
    assert adapter._connection_string == "udp:127.0.0.1:14550"
    assert adapter._baud_rate == 115200


def test_initial_state():
    adapter = MAVLinkAdapter()
    assert adapter._connection is None
    assert adapter._armed is False


def test_mode_constants():
    assert MAVLinkAdapter.MODE_GUIDED == 4
    assert MAVLinkAdapter.MODE_LAND == 9
    assert MAVLinkAdapter.MODE_RTL == 6
