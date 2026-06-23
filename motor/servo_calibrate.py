"""
servo_calibrate.py — SG90 Gate Servo Calibration Tool

Hardware:
    GPIO12 (Pin 32) → Signal (orange/yellow)
    External 5V     → VCC (red)
    Pin 14          → GND (brown)

Usage:
    sudo python3 servo_calibrate.py

Menu options:
    1) Test open position (90°)
    2) Test close position (0°)
    3) Custom angle test
    4) Find exact open/close duty cycles
    5) Full open/close cycle test
"""

import time
import sys

# ── Constants 

GPIO_CHIP  = 4    # Pi 5 = 4, Pi 4 = 0
GATE_PIN   = 12   # BCM GPIO 12 (Board Pin 32)
GATE_FREQ  = 50   # Hz — standard servo frequency

# SG90 duty cycle range
# 0°  = 2.5% duty  (~0.5ms pulse)
# 90° = 7.5% duty  (~1.5ms pulse)
# These are starting values — calibrate to your servo
CLOSE_DUTY = 2.5   # 0°  — gate closed
OPEN_DUTY  = 7.5   # 90° — gate open

# ── lgpio init

try:
    import lgpio
    _chip = lgpio.gpiochip_open(GPIO_CHIP)
    lgpio.gpio_claim_output(_chip, GATE_PIN)
    print(f"[OK] lgpio chip {GPIO_CHIP} opened — BCM GPIO {GATE_PIN} ready")
except Exception as e:
    print(f"[ERROR] lgpio init failed: {e}")
    print("        Check GPIO_CHIP and GATE_PIN values.")
    sys.exit(1)

# ── Helpers

def pwm_set(duty: float):
    """Apply duty cycle to gate servo."""
    lgpio.tx_pwm(_chip, GATE_PIN, GATE_FREQ, duty)

def pwm_off():
    """Stop PWM signal to prevent jitter."""
    lgpio.tx_pwm(_chip, GATE_PIN, GATE_FREQ, 0)

def move_to(duty: float, settle_sec: float = 0.5):
    """Move servo to position and wait for it to arrive."""
    pwm_set(duty)
    time.sleep(settle_sec)
    pwm_off()

def cleanup():
    pwm_off()
    lgpio.gpiochip_close(_chip)
    print("[OK] GPIO closed.")

def ask(prompt: str, default=None):
    try:
        val = input(prompt).strip()
        return val if val else default
    except KeyboardInterrupt:
        print("\nExiting...")
        cleanup()
        sys.exit(0)

def ask_float(prompt: str, default: float) -> float:
    while True:
        try:
            raw = ask(f"{prompt} [{default}]: ", str(default))
            return float(raw)
        except (ValueError, TypeError):
            print("  → Enter a valid number (e.g. 7.5)")

# ── Calibration steps

def test_open():
    """Move servo to open position (90°)."""
    global OPEN_DUTY
    print("\n" + "="*60)
    print("TEST 1 — OPEN POSITION (90°)")
    print("="*60)
    print("Servo will move to the open position.")
    print("Adjust duty until gate is fully open (90°).\n")
    print("Commands:  + = increase 0.1  |  - = decrease 0.1  |  s = save  |  q = skip\n")

    duty = OPEN_DUTY
    pwm_set(duty)
    time.sleep(0.5)

    while True:
        cmd = ask(f"  duty={duty:.2f}% > ").lower()
        if cmd == "s":
            OPEN_DUTY = duty
            pwm_off()
            print(f"\n[✓] Open duty saved: {OPEN_DUTY:.2f}%")
            break
        elif cmd == "q":
            pwm_off()
            break
        elif cmd == "+":
            duty = round(duty + 0.1, 2)
            pwm_set(duty)
        elif cmd == "-":
            duty = round(duty - 0.1, 2)
            pwm_set(duty)
        elif cmd == "++":
            duty = round(duty + 0.5, 2)
            pwm_set(duty)
        elif cmd == "--":
            duty = round(duty - 0.5, 2)
            pwm_set(duty)
        else:
            print("  Commands: + / - / ++ / -- / s / q")


def test_close():
    """Move servo to close position (0°)."""
    global CLOSE_DUTY
    print("\n" + "="*60)
    print("TEST 2 — CLOSE POSITION (0°)")
    print("="*60)
    print("Servo will move to the closed position.")
    print("Adjust duty until gate is fully closed.\n")
    print("Commands:  + = increase 0.1  |  - = decrease 0.1  |  s = save  |  q = skip\n")

    duty = CLOSE_DUTY
    pwm_set(duty)
    time.sleep(0.5)

    while True:
        cmd = ask(f"  duty={duty:.2f}% > ").lower()
        if cmd == "s":
            CLOSE_DUTY = duty
            pwm_off()
            print(f"\n[✓] Close duty saved: {CLOSE_DUTY:.2f}%")
            break
        elif cmd == "q":
            pwm_off()
            break
        elif cmd == "+":
            duty = round(duty + 0.1, 2)
            pwm_set(duty)
        elif cmd == "-":
            duty = round(duty - 0.1, 2)
            pwm_set(duty)
        elif cmd == "++":
            duty = round(duty + 0.5, 2)
            pwm_set(duty)
        elif cmd == "--":
            duty = round(duty - 0.5, 2)
            pwm_set(duty)
        else:
            print("  Commands: + / - / ++ / -- / s / q")


def test_custom_angle():
    """Move servo to a custom angle."""
    print("\n" + "="*60)
    print("TEST 3 — CUSTOM ANGLE")
    print("="*60)

    while True:
        duty = ask_float("Enter duty cycle (2.5=0°, 7.5=90°, q=quit)", 5.0)
        if duty is None:
            break
        print(f"  Moving to duty={duty:.2f}%")
        pwm_set(duty)
        time.sleep(0.5)
        cmd = ask("  Press Enter to stop PWM, q to quit > ")
        pwm_off()
        if cmd and cmd.lower() == "q":
            break


def test_cycle():
    """Full open/close cycle test."""
    print("\n" + "="*60)
    print("TEST 4 — FULL OPEN/CLOSE CYCLE")
    print("="*60)
    print(f"Using: OPEN={OPEN_DUTY:.2f}%  CLOSE={CLOSE_DUTY:.2f}%\n")

    repeats = int(ask_float("Number of cycles", 3))
    settle  = ask_float("Settle time per position (sec)", 0.5)

    input("Ready? Press Enter to start...")

    for i in range(repeats):
        print(f"\n  Cycle {i+1}/{repeats}")
        print(f"    → Opening ({OPEN_DUTY:.2f}%)")
        move_to(OPEN_DUTY, settle)
        time.sleep(0.5)
        print(f"    → Closing ({CLOSE_DUTY:.2f}%)")
        move_to(CLOSE_DUTY, settle)
        time.sleep(0.3)

    print("\n[✓] Cycle test complete.")
    input("Press Enter to continue...")


def print_summary():
    print("\n" + "="*60)
    print("CALIBRATION SUMMARY")
    print("="*60)
    print("Copy these values to motor_controller.py:\n")
    print(f"  GATE_CLOSE_DUTY = {CLOSE_DUTY:.2f}   # 0°  — gate closed")
    print(f"  GATE_OPEN_DUTY  = {OPEN_DUTY:.2f}   # 90° — gate open")
    print(f"  GATE_MOVE_TIME  = 0.5              # seconds to reach position")
    print("="*60)


# ── Main menu

def main():
    print("\n" + "★"*60)
    print("   SG90 GATE SERVO CALIBRATION TOOL")
    print(f"   Pi 5 — BCM GPIO {GATE_PIN} — lgpio")
    print("★"*60)
    print(f"\nStarting values: CLOSE={CLOSE_DUTY}%  OPEN={OPEN_DUTY}%\n")

    menu = {
        "1": ("Test open position (90°)",     test_open),
        "2": ("Test close position (0°)",     test_close),
        "3": ("Custom angle test",            test_custom_angle),
        "4": ("Full open/close cycle test",   test_cycle),
        "s": ("Show calibration summary",     print_summary),
        "q": ("Quit",                         None),
    }

    while True:
        print("\nMENU:")
        for key, (label, _) in menu.items():
            print(f"  [{key}] {label}")

        choice = ask("\nChoice > ", "").lower()

        if choice == "q":
            break
        elif choice in menu:
            label, fn = menu[choice]
            if fn:
                try:
                    fn()
                except KeyboardInterrupt:
                    print("\n  Step cancelled.")
                    pwm_off()
        else:
            print("Invalid choice.")

    cleanup()
    print("\nCalibration complete. Good luck!")


if __name__ == "__main__":
    main()