"""
MotorController — 28BYJ-48 stepper motor (14-slot tray) + SG90 gate servo.

Hardware connections:
    28BYJ-48 Stepper Motor (via ULN2003 driver):
        GPIO17 (Pin 11) → IN1
        GPIO18 (Pin 12) → IN2
        GPIO27 (Pin 13) → IN3
        GPIO22 (Pin 15) → IN4
        Pin 2           → VCC (5V)
        Pin 6           → GND

    SG90 Gate Servo:
        GPIO12 (Pin 32) → Signal (orange/yellow)
        External 5V     → VCC (red)
        Pin 14          → GND (brown)

System:
    - 14 slots, 4096 steps/revolution → 293 steps per slot (rounded)
    - Half-step sequence (8 steps) — smoother movement
    - SG90: 0° = closed, 90° = open

Environment variables:
    MOTOR_DRY_RUN=1  → simulate without hardware
    GPIO_CHIP=4      → Pi 5 = 4, Pi 4/3 = 0  (auto-detected if not set)
"""

from __future__ import annotations

import os
import time
from datetime import datetime

# ── Constants ─────────────────────────────────────────────────────────────────

DRY_RUN = os.environ.get("MOTOR_DRY_RUN", "").lower() in ("1", "true", "yes")

# 28BYJ-48 stepper motor pins (BCM)
STEP_PINS = [17, 18, 27, 22]

TOTAL_SLOTS    = 14
STEPS_PER_REV  = 4096
STEPS_PER_SLOT = round(STEPS_PER_REV / TOTAL_SLOTS)   # 293 steps (rounded)
STEP_DELAY     = 0.001   # seconds per step (lower = faster, min ~0.001)

# Half-step sequence (8 steps) — standard for 28BYJ-48
HALF_STEP_SEQ = [
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 0, 0, 1],
    [1, 0, 0, 1],
]

# SG90 gate servo
GATE_PIN        = 12    # BCM GPIO 12
GATE_FREQ       = 50    # Hz
GATE_CLOSE_DUTY = 2.5   # 0°  — gate closed
GATE_OPEN_DUTY  = 7.5   # 90° — gate open
GATE_MOVE_TIME  = 0.5   # seconds for servo to reach position

# ── Logging ───────────────────────────────────────────────────────────────────

def _log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] [MotorCtrl] {msg}", flush=True)

# ── lgpio initialization ──────────────────────────────────────────────────────
#
# Strategy:
#   1. Try GPIO_CHIP env var first (explicit override)
#   2. If not set, auto-detect: try chip 4 (Pi 5), then chip 0 (Pi 4/3)
#   3. On ANY failure: close the chip handle before nulling it (prevents leak)
#   4. Log the exact error so we can diagnose why DRY_RUN was activated

_chip     = None
_lgpio    = None
GPIO_CHIP = None   # resolved below


def _try_open_chip(chip_num: int):
    """Try to open a gpiochip, claim all pins, return (chip_handle, lgpio_module) or raise."""
    import lgpio as _lgp
    _log(f"  Trying gpiochip{chip_num} ...")
    h = _lgp.gpiochip_open(chip_num)
    try:
        for pin in STEP_PINS:
            rc = _lgp.gpio_claim_output(h, pin)
            if rc < 0:
                raise RuntimeError(f"gpio_claim_output(pin {pin}) returned {rc}")
        rc = _lgp.gpio_claim_output(h, GATE_PIN)
        if rc < 0:
            raise RuntimeError(f"gpio_claim_output(gate pin {GATE_PIN}) returned {rc}")
    except Exception:
        # Close handle to avoid resource leak before re-raising
        try:
            _lgp.gpiochip_close(h)
        except Exception:
            pass
        raise
    return h, _lgp, chip_num


if not DRY_RUN:
    _log("Initialising GPIO (lgpio) ...")
    _env_chip = os.environ.get("GPIO_CHIP")

    if _env_chip is not None:
        # Explicit override via env var — use it directly, fail loudly
        _candidates = [int(_env_chip)]
    else:
        # Auto-detect: Pi 5 uses chip 4, Pi 4/3 uses chip 0
        _candidates = [4, 0]

    _last_error = None
    for _cnum in _candidates:
        try:
            _chip, _lgpio, GPIO_CHIP = _try_open_chip(_cnum)
            _log(f"lgpio gpiochip{GPIO_CHIP} opened OK — "
                 f"step pins={STEP_PINS}, gate pin={GATE_PIN}, "
                 f"{STEPS_PER_SLOT} steps/slot")
            break
        except Exception as _e:
            _log(f"  gpiochip{_cnum} failed: {type(_e).__name__}: {_e}")
            _last_error = _e

    if _chip is None:
        DRY_RUN = True
        _log(f"*** lgpio unavailable — switching to DRY_RUN ***")
        _log(f"*** Last error: {_last_error} ***")
        _log("*** Check: sudo needed? Pins already claimed? Wrong chip number? ***")

# ── MotorController ───────────────────────────────────────────────────────────

class MotorController:
    """14-slot stepper motor tray controller + SG90 gate servo."""

    def __init__(self):
        self._current_slot = 0
        self._seq_index    = 0   # current position in half-step sequence
        _log(f"Ready (DRY_RUN={DRY_RUN}, chip={GPIO_CHIP}, "
             f"slots={TOTAL_SLOTS}, {STEPS_PER_SLOT} steps/slot)")

    # ── Internal: stepper motor ───────────────────────────────────────────────

    def _set_step(self, seq: list):
        """Apply a single half-step to the motor coils."""
        if DRY_RUN or _chip is None or _lgpio is None:
            return
        for i, pin in enumerate(STEP_PINS):
            _lgpio.gpio_write(_chip, pin, seq[i])

    def _step_motor(self, steps: int, cw: bool = True):
        """
        Rotate the stepper motor by the given number of steps.
        cw=True → clockwise, cw=False → counter-clockwise.
        """
        direction = 1 if cw else -1
        for _ in range(abs(steps)):
            self._seq_index = (self._seq_index + direction) % 8
            self._set_step(HALF_STEP_SEQ[self._seq_index])
            time.sleep(STEP_DELAY)
        # Power off coils after movement to prevent overheating
        self._motor_off()

    def _motor_off(self):
        """Turn off all motor coils."""
        if DRY_RUN or _chip is None or _lgpio is None:
            return
        for pin in STEP_PINS:
            _lgpio.gpio_write(_chip, pin, 0)

    # ── Internal: SG90 gate servo ─────────────────────────────────────────────

    def _gate_pwm(self, duty: float):
        """Send PWM signal to the gate servo."""
        if DRY_RUN or _chip is None or _lgpio is None:
            return
        _lgpio.tx_pwm(_chip, GATE_PIN, GATE_FREQ, duty)

    # ── Slot rotation ─────────────────────────────────────────────────────────

    def rotate_to_slot(self, target_slot: int) -> bool:
        """Rotate tray to the target slot via the shortest path."""
        if not (0 <= target_slot < TOTAL_SLOTS):
            _log(f"Invalid slot: {target_slot} (valid range: 0–{TOTAL_SLOTS - 1})")
            return False

        if target_slot == self._current_slot:
            _log(f"Already at slot {target_slot}")
            return True

        # Calculate shortest rotation direction
        delta_fwd = (target_slot - self._current_slot) % TOTAL_SLOTS
        delta_rev = TOTAL_SLOTS - delta_fwd

        if delta_fwd <= delta_rev:
            delta, cw = delta_fwd, True
        else:
            delta, cw = delta_rev, False

        steps = delta * STEPS_PER_SLOT
        _log(f"Rotating: slot {self._current_slot} → {target_slot}  "
             f"{'CW' if cw else 'CCW'}  {delta} slots  {steps} steps")

        if DRY_RUN:
            time.sleep(delta * 0.5)  # simulate movement delay
        else:
            self._step_motor(steps, cw=cw)

        self._current_slot = target_slot
        _log(f"Arrived at slot {target_slot}")
        return True

    def rotate_one_slot(self, cw: bool = True) -> bool:
        """Rotate one slot forward or backward."""
        nxt = (self._current_slot + (1 if cw else -1)) % TOTAL_SLOTS
        return self.rotate_to_slot(nxt)

    def full_revolution(self, cw: bool = True) -> bool:
        """Full revolution — 14 slots = 4096 steps."""
        _log(f"Full revolution {'CW' if cw else 'CCW'} — {STEPS_PER_REV} steps")
        if DRY_RUN:
            time.sleep(4.0)
        else:
            self._step_motor(STEPS_PER_REV, cw=cw)
        return True

    # ── Gate servo ────────────────────────────────────────────────────────────

    def open_gate(self) -> bool:
        """Rotate SG90 servo to 90° — open the gate."""
        _log("Opening gate (90°)")
        if DRY_RUN:
            _log("[DRY-RUN] open_gate simulated")
            return True
        try:
            self._gate_pwm(GATE_OPEN_DUTY)
            time.sleep(GATE_MOVE_TIME)
            self._gate_pwm(0)  # stop PWM signal to prevent jitter
            return True
        except Exception as e:
            _log(f"open_gate error: {e}")
            return False

    def close_gate(self) -> bool:
        """Rotate SG90 servo to 0° — close the gate."""
        _log("Closing gate (0°)")
        if DRY_RUN:
            _log("[DRY-RUN] close_gate simulated")
            return True
        try:
            self._gate_pwm(GATE_CLOSE_DUTY)
            time.sleep(GATE_MOVE_TIME)
            self._gate_pwm(0)
            return True
        except Exception as e:
            _log(f"close_gate error: {e}")
            return False

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def current_slot(self) -> int:
        return self._current_slot

    @property
    def is_gate_open(self) -> bool:
        return False

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def cleanup(self):
        """Release all GPIO resources."""
        if not DRY_RUN and _chip is not None and _lgpio is not None:
            try:
                self._motor_off()
                self._gate_pwm(0)
                _lgpio.gpiochip_close(_chip)
                _log("GPIO chip closed")
            except Exception as e:
                _log(f"cleanup error: {e}")
        _log("GPIO cleaned up")
