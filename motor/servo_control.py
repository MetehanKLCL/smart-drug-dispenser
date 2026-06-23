"""
ServoController — backward compatibility wrapper around MotorController.

Provides the same API used by kiosk_app.py and api_server.py,
delegating all motor and gate operations to MotorController.
"""

from __future__ import annotations

import logging
from motor_controller import MotorController, DRY_RUN, TOTAL_SLOTS

log = logging.getLogger(__name__)


class ServoController:
    """Backward-compatible wrapper — delegates to MotorController."""

    def __init__(self):
        self._motor = MotorController()

    # ── Gate API

    def open_gate(self) -> bool:
        """Open the gate — rotates SG90 servo to 90°."""
        return self._motor.open_gate()

    def close_gate(self) -> bool:
        """Close the gate — rotates SG90 servo to 0°."""
        return self._motor.close_gate()

    def dispense_cycle(self) -> bool:
        """Dispense medication — rotate tray one slot CW, then open/close gate."""
        log.info("dispense_cycle → rotating one slot + gate open/close")
        ok = self._motor.rotate_one_slot(cw=True)
        if ok:
            self._motor.open_gate()
            import time; time.sleep(3)  # time for patient to take medication
            self._motor.close_gate()
        return ok

    def dispense_at_slot(self, slot: int) -> bool:
        """Hedef fiziksel slota dön, kapıyı aç/kapa (kiosk çizelgesindeki slot_id)."""
        slot = int(slot) % TOTAL_SLOTS
        log.info("dispense_at_slot → rotate_to_slot(%d) + gate", slot)
        ok = self._motor.rotate_to_slot(slot)
        if not ok:
            return False
        self._motor.open_gate()
        import time

        time.sleep(3)
        self._motor.close_gate()
        return True

    # ── Direct motor access

    def rotate_to_slot(self, slot: int) -> bool:
        return self._motor.rotate_to_slot(slot)

    def full_revolution(self, cw: bool = True) -> bool:
        return self._motor.full_revolution(cw=cw)

    # ── Properties

    @property
    def is_open(self) -> bool:
        return self._motor.is_gate_open

    @property
    def is_hardware(self) -> bool:
        return not DRY_RUN

    @property
    def current_slot(self) -> int:
        return self._motor.current_slot

    # ── Cleanup

    def cleanup(self):
        self._motor.cleanup()

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass