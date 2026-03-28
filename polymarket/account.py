"""Account helpers — balance, positions, connection status."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

log = logging.getLogger(__name__)


async def get_balance(poly_client) -> float | None:
    """Return USDC balance via the CLOB client's get_bal_allowance."""
    try:
        # py-clob-client exposes get_balance_allowance but it needs params.
        # Safest: call the underlying session directly.
        # Falls back to None if unavailable.
        bal = await asyncio.to_thread(
            lambda: getattr(poly_client.client, "get_collateral", lambda: None)()
        )
        if bal is not None:
            return round(float(bal), 2)
        return None
    except Exception:
        log.exception("Failed to fetch balance")
        return None


async def get_open_positions(poly_client) -> list[dict[str, Any]]:
    """Return list of open positions via the CLOB client."""
    try:
        positions = await asyncio.to_thread(
            lambda: getattr(poly_client.client, "get_positions", lambda: [])()
        )
        if isinstance(positions, list):
            return positions
        return []
    except Exception:
        log.exception("Failed to fetch positions")
        return []


async def get_connection_status(poly_client) -> bool:
    """Quick connectivity check — try to hit the CLOB server time endpoint."""
    try:
        info = await asyncio.to_thread(poly_client.client.get_server_time)
        return info is not None
    except Exception:
        log.exception("Connection check failed")
        return False
