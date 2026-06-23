"""
test_motor.py — 28BYJ-48 Stepper Motor + SG90 Gate Servo Test

Hardware:
    28BYJ-48 Stepper Motor (via ULN2003):
        GPIO17 (Pin 11) → IN1
        GPIO18 (Pin 12) → IN2
        GPIO27 (Pin 13) → IN3
        GPIO22 (Pin 15) → IN4

    SG90 Gate Servo:
        GPIO12 (Pin 32) → Signal

Usage:
    sudo python3 test_motor.py              # interactive menu
    sudo python3 test_motor.py --slot 3    # go to slot 3
    sudo python3 test_motor.py --sweep     # sweep all slots
    sudo python3 test_motor.py --gate      # test gate open/close
    sudo python3 test_motor.py --dry       # simulate without hardware
"""

from __future__ import annotations

import os
import time
import argparse

# ── Constants

GPIO_CHIP = int(os.environ.get("GPIO_CHIP", "4"))   # Pi 5 = 4, Pi 4 = 0
DRY_RUN   = os.environ.get("MOTOR_DRY_RUN", "").lower() in ("1", "true", "yes")

# 28BYJ-48 stepper motor pins (BCM)
STEP_PINS = [17, 18, 27, 22]

TOTAL_SLOTS    = 14
STEPS_PER_REV  = 4096
STEPS_PER_SLOT = STEPS_PER_REV // TOTAL_SLOTS   # 293 steps per slot
STEP_DELAY     = 0.001   # seconds per step

# Half-step sequence (8 steps)
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
GATE_PIN        = 12
GATE_FREQ       = 50
GATE_CLOSE_DUTY = 2.5   # 0°
GATE_OPEN_DUTY  = 7.5   # 90°

# ── Logging

def _log(msg):
    print(f"[test_motor] {msg}", flush=True)

# ── GPIO setup

_chip     = None
_seq_idx  = 0

def setup(dry: bool):
    global _chip, DRY_RUN
    if dry:
        DRY_RUN = True
        _log("DRY-RUN mode (no hardware)")
        return
    try:
        import lgpio
        _chip = lgpio.gpiochip_open(GPIO_CHIP)
        for pin in STEP_PINS:
            lgpio.gpio_claim_output(_chip, pin)
        lgpio.gpio_claim_output(_chip, GATE_PIN)
        _log(f"lgpio chip {GPIO_CHIP} opened — step pins={STEP_PINS}, gate pin={GATE_PIN}")
    except Exception as e:
        DRY_RUN = True
        _chip = None
        _log(f"GPIO error: {e} → switching to DRY-RUN")

def cleanup():
    if _chip is None:
        return
    try:
        import lgpio
        motor_off()
        lgpio.tx_pwm(_chip, GATE_PIN, GATE_FREQ, 0)
        lgpio.gpiochip_close(_chip)
        _log("GPIO closed")
    except Exception:
        pass

# ── Stepper motor

def set_step(seq: list):
    """Apply a single half-step."""
    global DRY_RUN, _chip
    if DRY_RUN or _chip is None:
        return
    import lgpio
    try:
        for i, pin in enumerate(STEP_PINS):
            lgpio.gpio_write(_chip, pin, seq[i])
    except Exception as e:
        DRY_RUN = True
        _chip = None
        _log(f"set_step gpio error: {e} → switching to DRY-RUN")

def motor_off():
    """Turn off all motor coils."""
    global DRY_RUN, _chip
    if DRY_RUN or _chip is None:
        return
    import lgpio
    try:
        for pin in STEP_PINS:
            lgpio.gpio_write(_chip, pin, 0)
    except Exception as e:
        DRY_RUN = True
        _chip = None
        _log(f"motor_off gpio error: {e} → switching to DRY-RUN")

def step_motor(steps: int, cw: bool = True):
    """Rotate stepper motor by given steps."""
    global _seq_idx
    direction = 1 if cw else -1
    for _ in range(abs(steps)):
        _seq_idx = (_seq_idx + direction) % 8
        set_step(HALF_STEP_SEQ[_seq_idx])
        time.sleep(STEP_DELAY)
    motor_off()

# ── Slot tracking

_current_slot = 0

def rotate_to_slot(target: int):
    global _current_slot

    if not (0 <= target < TOTAL_SLOTS):
        _log(f"Invalid slot: {target} (valid: 0–{TOTAL_SLOTS - 1})")
        return

    if target == _current_slot:
        _log(f"Already at slot {target}")
        return

    delta_fwd = (target - _current_slot) % TOTAL_SLOTS
    delta_rev = TOTAL_SLOTS - delta_fwd
    if delta_fwd <= delta_rev:
        delta, cw = delta_fwd, True
    else:
        delta, cw = delta_rev, False

    steps = delta * STEPS_PER_SLOT
    _log(f"Slot {_current_slot} → {target}  |  "
         f"{'CW' if cw else 'CCW'}  {delta} slots  {steps} steps")

    if DRY_RUN or _chip is None:
        time.sleep(delta * 0.3)
    else:
        step_motor(steps, cw=cw)

    _current_slot = target
    _log(f"Arrived at slot {target}")

# ── Gate servo

def gate_open():
    global DRY_RUN, _chip
    _log(f"Opening gate ({GATE_OPEN_DUTY}% → 90°)")
    if DRY_RUN or _chip is None:
        _log("[DRY-RUN] gate open simulated")
        return
    import lgpio
    try:
        lgpio.tx_pwm(_chip, GATE_PIN, GATE_FREQ, GATE_OPEN_DUTY)
        time.sleep(0.5)
        lgpio.tx_pwm(_chip, GATE_PIN, GATE_FREQ, 0)
    except Exception as e:
        DRY_RUN = True
        _chip = None
        _log(f"gate_open gpio error: {e} → switching to DRY-RUN")

def gate_close():
    global DRY_RUN, _chip
    _log(f"Closing gate ({GATE_CLOSE_DUTY}% → 0°)")
    if DRY_RUN or _chip is None:
        _log("[DRY-RUN] gate close simulated")
        return
    import lgpio
    try:
        lgpio.tx_pwm(_chip, GATE_PIN, GATE_FREQ, GATE_CLOSE_DUTY)
        time.sleep(0.5)
        lgpio.tx_pwm(_chip, GATE_PIN, GATE_FREQ, 0)
    except Exception as e:
        DRY_RUN = True
        _chip = None
        _log(f"gate_close gpio error: {e} → switching to DRY-RUN")

# ── Test routines

def test_single_slot():
    """Single slot movement — calibration check."""
    _log(f"=== Single slot test: slot {_current_slot} → {(_current_slot + 1) % TOTAL_SLOTS} ===")
    _log(f"Motor will move {STEPS_PER_SLOT} steps ({360/TOTAL_SLOTS:.2f}°)")
    input("Ready? Press Enter...")
    rotate_to_slot((_current_slot + 1) % TOTAL_SLOTS)
    _log("Done. Did it move exactly one slot?")

def test_sweep():
    """Sweep through all 14 slots."""
    _log(f"=== Sweep: all {TOTAL_SLOTS} slots ===")
    for s in range(TOTAL_SLOTS):
        rotate_to_slot(s)
        input(f"  Slot {s} — Press Enter to continue...")
    _log("Returning to slot 0...")
    rotate_to_slot(0)

def test_full_revolution():
    """Full revolution — 4096 steps."""
    _log(f"=== Full revolution: {STEPS_PER_REV} steps ===")
    input("Ready? Press Enter...")
    if DRY_RUN or _chip is None:
        time.sleep(4.0)
    else:
        step_motor(STEPS_PER_REV, cw=True)
    _log("Done. Did it return to starting position?")

def test_gate():
    """Gate open/close cycle test."""
    _log("=== Gate servo test ===")
    repeats = int(input("Number of cycles [3]: ").strip() or "3")
    for i in range(repeats):
        _log(f"Cycle {i+1}/{repeats}")
        gate_open()
        time.sleep(1)
        gate_close()
        time.sleep(0.5)
    _log("Gate test complete.")

def test_dispense_cycle():
    """Full dispense: rotate to slot + open gate + close gate."""
    _log("=== Full dispense cycle test ===")
    target = int(input(f"Target slot (0-{TOTAL_SLOTS-1}) [1]: ").strip() or "1")
    rotate_to_slot(target)
    _log("Opening gate...")
    gate_open()
    _log("Waiting 3 seconds (patient takes medication)...")
    time.sleep(3)
    _log("Closing gate...")
    gate_close()
    _log("Dispense cycle complete.")

def interactive():
    print(f"\n=== 14-slot Dispenser Test ===")
    print(f"  Stepper: {STEPS_PER_SLOT} steps/slot  |  Gate: GPIO{GATE_PIN}")
    print(f"  DRY_RUN={DRY_RUN}")
    print("Commands:")
    print("  0-13   → go to slot")
    print("  1s     → single slot test")
    print("  sweep  → sweep all slots")
    print("  full   → full revolution")
    print("  gate   → gate open/close test")
    print("  disp   → full dispense cycle")
    print("  q      → quit\n")

    while True:
        try:
            cmd = input("cmd> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if cmd == "q":
            break
        elif cmd == "1s":
            test_single_slot()
        elif cmd == "sweep":
            test_sweep()
        elif cmd == "full":
            test_full_revolution()
        elif cmd == "gate":
            test_gate()
        elif cmd == "disp":
            test_dispense_cycle()
        elif cmd.isdigit():
            t = int(cmd)
            if 0 <= t < TOTAL_SLOTS:
                rotate_to_slot(t)
            else:
                print(f"  Enter 0-{TOTAL_SLOTS-1}")
        else:
            print(f"  Unknown command: {cmd!r}")

# ── Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slot",  type=int, default=None,
                        help=f"Go to slot (0–{TOTAL_SLOTS-1})")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep all slots")
    parser.add_argument("--full",  action="store_true",
                        help="Full revolution")
    parser.add_argument("--gate",  action="store_true",
                        help="Test gate open/close")
    parser.add_argument("--disp",  action="store_true",
                        help="Full dispense cycle")
    parser.add_argument("--dry",   action="store_true",
                        help="Simulate without hardware")
    args = parser.parse_args()

    global DRY_RUN
    DRY_RUN = args.dry or DRY_RUN
    setup(DRY_RUN)

    try:
        if args.slot is not None:
            rotate_to_slot(args.slot)
        elif args.sweep:
            test_sweep()
        elif args.full:
            test_full_revolution()
        elif args.gate:
            test_gate()
        elif args.disp:
            test_dispense_cycle()
        else:
            interactive()
    finally:
        motor_off()
        cleanup()


if __name__ == "__main__":
    main()