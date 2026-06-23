"""Modern pygame kiosk display for MediDispense (800×480).

Design: dark material-inspired, two-column IDLE layout,
card-based schedule display with frequency badges.

States:
  IDLE           – clock (left) + today's schedule (right)
  WAITING        – patient name + pulsing ring + countdown bar
  AUTHENTICATING – live camera + corner brackets (TODO: parity with
                   face_authentication/auth.py on Mac — landmarks, Blink/Mouth/Head chips)
  SUCCESS        – green confirmation with confidence bar
  DISPENSING     – animated spinner + medication info
  MONITORING_SWALLOW – live camera + intake hints (hand / cup / mouth)
  TIMEOUT        – red timeout with icon ring
  MISSED         – amber warning with missed dose info
  ERROR          – error message display
"""

from __future__ import annotations

import os
import sys
import math
import time
import logging
import numpy as np
from enum import Enum, auto
from datetime import datetime
from typing import Optional

log = logging.getLogger(__name__)

_pi_backend = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pi_backend"))
if _pi_backend not in sys.path:
    sys.path.insert(0, _pi_backend)
from schedule_weekdays import week_days_python_allowed  # noqa: E402

SCREEN_W = 800
SCREEN_H = 480
STATUS_H = 28          # status bar at very bottom
USABLE_H = SCREEN_H - STATUS_H

# ── Color palette (dark navy / slate) ─────────────────────────────────────────
C_BG          = ( 11,  13,  20)   # page background
C_SURFACE     = ( 20,  22,  32)   # card / panel
C_SURFACE_HI  = ( 30,  33,  48)   # elevated card
C_BORDER      = ( 42,  46,  70)   # subtle border
C_DIVIDER     = ( 28,  30,  46)   # divider lines

C_PRIMARY     = ( 96, 165, 250)   # blue-400
C_PRIMARY_DIM = ( 18,  40,  78)   # dimmed primary bg
C_SUCCESS     = ( 52, 211, 153)   # emerald-400
C_SUCCESS_DIM = ( 10,  48,  35)   # dimmed green bg
C_WARN        = (251, 191,  36)   # amber-400
C_WARN_DIM    = ( 58,  42,   8)   # dimmed amber bg
C_ERROR       = (248, 113, 113)   # red-400
C_ERROR_DIM   = ( 58,  14,  14)   # dimmed red bg
C_PURPLE      = (167, 139, 250)   # violet-400  (weekly freq)
C_ORANGE      = (251, 146,  60)   # orange-400  (alternate freq)

C_TEXT1       = (226, 232, 240)   # slate-200  primary text
C_TEXT2       = (148, 163, 184)   # slate-400  secondary text
C_TEXT3       = ( 71,  85, 105)   # slate-600  muted text
C_WHITE       = (255, 255, 255)

# Frequency display config
_FREQ_COLOR = {"daily": C_PRIMARY, "weekly": C_PURPLE, "alternate": C_ORANGE}
_FREQ_LABEL = {"daily": "Daily",   "weekly": "Weekly",  "alternate": "Every 2d"}
_DAY_ABBR   = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


class DisplayState(Enum):
    IDLE           = auto()
    WAITING        = auto()
    AUTHENTICATING = auto()
    SUCCESS        = auto()
    DISPENSING     = auto()
    MONITORING_SWALLOW = auto()
    TIMEOUT        = auto()
    MISSED         = auto()
    ERROR          = auto()


class DispenserDisplay:
    """Manages the 800×480 kiosk screen."""

    def __init__(self, fullscreen: bool = True):
        self._state   = DisplayState.IDLE
        self._screen  = None
        self._clock   = None
        self._fonts:  dict = {}
        self._fullscreen = fullscreen
        self._running = False

        # State context
        self._patient_name:    str   = ""
        self._medication_name: str   = ""
        self._slot_id:         int   = 0
        self._countdown:       int   = 300
        self._error_msg:       str   = ""
        self._auth_score:      float = 0.0
        self._frame_count:     int   = 0
        self._swallow_status: str   = ""

        # IDLE schedule data
        self._schedule_list: list[dict] = []   # rich list from get_todays_schedules
        self._schedule_str:  str        = ""   # fallback plain string
        self._loaded_slots:  list       = []   # backward-compat for older kiosk_app

        self._init_pygame()

    # ── Pygame initialisation ──────────────────────────────────────────────────

    def _init_pygame(self):
        import pygame

        drivers = []
        if os.environ.get("DISPLAY"):          drivers.append("x11")
        if os.environ.get("WAYLAND_DISPLAY"):  drivers.append("wayland")
        drivers += ["kmsdrm", "fbdev", "offscreen"]

        forced = os.environ.get("SDL_VIDEODRIVER")
        if forced:
            drivers = [forced]

        last_err = None
        for driver in drivers:
            try:
                os.environ["SDL_VIDEODRIVER"] = driver
                pygame.display.quit()
                pygame.display.init()

                flags = pygame.FULLSCREEN if (self._fullscreen and driver != "offscreen") else 0
                self._screen = pygame.display.set_mode((SCREEN_W, SCREEN_H), flags)
                pygame.display.set_caption("MediDispense")
                pygame.font.init()
                try:
                    pygame.mixer.init()
                except Exception:
                    pass

                self._clock = pygame.time.Clock()

                def font(size, bold=False):
                    for name in ("DejaVu Sans", "Liberation Sans", "FreeSans", None):
                        try:
                            return pygame.font.SysFont(name, size, bold=bold)
                        except Exception:
                            continue
                    return pygame.font.Font(None, size)

                self._fonts = {
                    "clock_big": font(88, bold=True),   # IDLE large clock
                    "title":     font(40, bold=True),   # state titles
                    "heading":   font(28, bold=True),   # card headings
                    "medium":    font(24),
                    "body":      font(20),
                    "small":     font(15),
                    "badge":     font(13, bold=True),
                    "icon":      font(72, bold=True),
                }

                if driver != "offscreen":
                    pygame.mouse.set_visible(False)

                self._running = True
                log.info("Display init: driver=%s %dx%d fullscreen=%s",
                         driver, SCREEN_W, SCREEN_H, self._fullscreen)
                if driver == "offscreen":
                    log.warning(
                        "Pygame offscreen — HDMI/monitorde PENCERE YOK (SSH oturumunda "
                        "DISPLAY/WAYLAND genelde ayarlı değil). Çözüm: Pi klavyesinden "
                        "terminalde çalıştır veya SSH ile örn. "
                        "DISPLAY=:0 WAYLAND_DISPLAY=wayland-1 XDG_RUNTIME_DIR=/run/user/$(id -u) "
                        "python3 pi_dispenser.py --windowed"
                    )
                return
            except Exception as e:
                last_err = e
                log.debug("Driver '%s' failed: %s", driver, e)
                try:
                    pygame.display.quit()
                except Exception:
                    pass

        log.error("All video drivers failed. Last: %s", last_err)
        self._running = False

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def running(self) -> bool:
        return self._running

    @property
    def state(self) -> DisplayState:
        return self._state

    def set_idle(self, next_info: str = ""):
        self._state = DisplayState.IDLE
        if next_info:
            self._schedule_str = next_info

    def set_waiting(self, patient_name: str, medication_name: str,
                    slot_id: int, countdown: int = 300):
        self._state           = DisplayState.WAITING
        self._patient_name    = patient_name
        self._medication_name = medication_name
        self._slot_id         = slot_id
        self._countdown       = countdown

    def set_authenticating(self, countdown: int = 300):
        self._state       = DisplayState.AUTHENTICATING
        self._countdown   = countdown
        self._frame_count = 0

    def set_success(self, patient_name: str, score: float):
        self._state        = DisplayState.SUCCESS
        self._patient_name = patient_name
        self._auth_score   = score

    def set_dispensing(self, medication_name: str, slot_id: int):
        self._state           = DisplayState.DISPENSING
        self._medication_name = medication_name
        self._slot_id         = slot_id

    def set_swallow_monitoring(
        self,
        patient_name: str,
        medication_name: str,
        countdown: int = 60,
    ):
        """After dispense: camera + hand/cup/mouth intake hints."""
        self._state = DisplayState.MONITORING_SWALLOW
        self._patient_name = patient_name
        self._medication_name = medication_name
        self._countdown = max(0, int(countdown))
        self._swallow_status = ""
        self._frame_count = 0

    def update_swallow_status(self, text: str):
        self._swallow_status = text

    def set_timeout(self):
        self._state = DisplayState.TIMEOUT

    def set_missed(self, patient_name: str, medication_name: str):
        self._state           = DisplayState.MISSED
        self._patient_name    = patient_name
        self._medication_name = medication_name

    def set_error(self, msg: str):
        self._state     = DisplayState.ERROR
        self._error_msg = msg

    def update_countdown(self, seconds_left: int):
        self._countdown = max(0, seconds_left)

    def update_next_schedule(self, info: str):
        """Plain-string fallback (used when schedule_list is empty)."""
        self._schedule_str = info

    def update_schedule_list(self, schedules: list[dict]):
        """Rich schedule groups from ScheduleMonitor.get_todays_schedules()."""
        self._schedule_list = schedules

    def update_loaded_slots(self, loaded):
        """Backward-compatible no-op hook for older kiosk_app versions."""
        self._loaded_slots = loaded or []

    # ── Main render loop ───────────────────────────────────────────────────────

    def render(self, camera_frame: Optional[np.ndarray] = None,
               face_locations: list | None = None):
        if not self._running:
            return

        import pygame

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self._running = False
                return

        self._screen.fill(C_BG)

        dispatch = {
            DisplayState.IDLE:           self._render_idle,
            DisplayState.WAITING:        self._render_waiting,
            DisplayState.AUTHENTICATING: lambda: self._render_auth(camera_frame, face_locations),
            DisplayState.SUCCESS:        self._render_success,
            DisplayState.DISPENSING:     self._render_dispensing,
            DisplayState.MONITORING_SWALLOW: lambda: self._render_swallow_monitor(camera_frame),
            DisplayState.TIMEOUT:        self._render_timeout,
            DisplayState.MISSED:         self._render_missed,
            DisplayState.ERROR:          self._render_error,
        }
        fn = dispatch.get(self._state)
        if fn:
            fn()

        self._render_status_bar()
        pygame.display.flip()
        self._clock.tick(30)

    # ── Drawing primitives ─────────────────────────────────────────────────────

    def _txt(self, text: str, font_key: str, color: tuple) -> "pygame.Surface":
        return self._fonts[font_key].render(str(text), True, color)

    def _blit_cx(self, surf, cx: int, y: int):
        """Blit surface horizontally centred at cx."""
        self._screen.blit(surf, surf.get_rect(centerx=cx, y=y))

    def _rr(self, color, x, y, w, h, r=8, border_col=None, bw=1):
        """Rounded rectangle, optionally with a border."""
        import pygame
        pygame.draw.rect(self._screen, color, (x, y, w, h), border_radius=r)
        if border_col:
            pygame.draw.rect(self._screen, border_col, (x, y, w, h), bw, border_radius=r)

    def _pill(self, text: str, fg: tuple, bg: tuple, x: int, y: int) -> int:
        """Draw a pill/badge. Returns total width."""
        import pygame
        s   = self._fonts["badge"].render(text, True, fg)
        sw, sh = s.get_size()
        pad = 7
        tw  = sw + pad * 2
        pygame.draw.rect(self._screen, bg, (x, y, tw, sh + 6), border_radius=10)
        self._screen.blit(s, (x + pad, y + 3))
        return tw

    def _bar(self, x, y, w, h, frac, color, bg=None, r=4):
        """Horizontal progress bar (frac 0–1)."""
        import pygame
        pygame.draw.rect(self._screen, bg or C_SURFACE_HI, (x, y, w, h), border_radius=r)
        fw = max(r * 2, int(w * max(0.0, min(1.0, frac))))
        pygame.draw.rect(self._screen, color, (x, y, fw, h), border_radius=r)

    def _pulse_ring(self, cx, cy, base_r, color, speed=2.5):
        """Animated pulsing ring."""
        import pygame
        t = time.time()
        for i in range(3, 0, -1):
            r     = base_r + i * 14 + int(math.sin(t * speed + i) * 4)
            alpha = max(15, 60 - i * 18)
            surf  = pygame.Surface((r * 2 + 4, r * 2 + 4), pygame.SRCALPHA)
            pygame.draw.circle(surf, (*color, alpha), (r + 2, r + 2), r, 2)
            self._screen.blit(surf, (cx - r - 2, cy - r - 2))

    def _icon_ring(self, cx, cy, r, fill_col, ring_col, icon_text, icon_key="icon"):
        """Filled circle with border ring and centred icon text."""
        import pygame
        pygame.draw.circle(self._screen, fill_col, (cx, cy), r)
        pygame.draw.circle(self._screen, ring_col, (cx, cy), r, 3)
        s = self._fonts[icon_key].render(icon_text, True, ring_col)
        self._blit_cx(s, cx, cy - s.get_height() // 2)

    # ── IDLE ───────────────────────────────────────────────────────────────────

    def _render_idle(self):
        import pygame

        LEFT_W = 358          # left panel width
        CX_L   = LEFT_W // 2  # left panel centre-x
        now    = datetime.now()

        # ── Left panel: clock + branding ────────────────────────────────
        # Logo strip
        logo_s = self._txt("MediDispense", "badge", C_PRIMARY)
        self._screen.blit(logo_s, (14, 14))
        pygame.draw.circle(self._screen, C_PRIMARY, (10, 21), 4)

        # Large clock
        time_str = now.strftime("%H:%M")
        t_surf   = self._fonts["clock_big"].render(time_str, True, C_WHITE)
        t_rect   = t_surf.get_rect(centerx=CX_L, y=40)
        self._screen.blit(t_surf, t_rect)

        # Date
        ds = self._txt(now.strftime("%d %B %Y"), "body", C_TEXT2)
        self._blit_cx(ds, CX_L, t_rect.bottom + 10)

        # Day of week
        day_s = self._txt(now.strftime("%A"), "medium", C_TEXT1)
        self._blit_cx(day_s, CX_L, t_rect.bottom + 38)

        # Thin accent divider
        sep_y = t_rect.bottom + 72
        pygame.draw.line(self._screen, C_BORDER, (24, sep_y), (LEFT_W - 24, sep_y), 1)

        # System ready pulse
        dot_y = USABLE_H - 36
        pygame.draw.circle(self._screen, C_SUCCESS, (24, dot_y + 7), 5)
        rdy_s = self._txt("System Ready", "small", C_SUCCESS)
        self._screen.blit(rdy_s, (34, dot_y))

        # Vertical divider
        pygame.draw.line(self._screen, C_DIVIDER, (LEFT_W, 10), (LEFT_W, USABLE_H - 6), 1)

        # ── Right panel: today's schedule ───────────────────────────────
        RX = LEFT_W + 12
        RW = SCREEN_W - RX - 8

        hdr_s = self._txt("TODAY'S SCHEDULE", "small", C_TEXT3)
        self._screen.blit(hdr_s, (RX, 13))
        pygame.draw.line(self._screen, C_BORDER,
                         (RX, 32), (SCREEN_W - 12, 32), 1)

        schedules = self._schedule_list
        if not schedules:
            empty_msg = self._schedule_str or "No medications scheduled for today"
            em = self._txt(empty_msg, "body", C_TEXT3)
            self._blit_cx(em, RX + RW // 2, USABLE_H // 2 - 10)
        else:
            visible  = schedules[:4]
            padding  = 6
            card_h   = (USABLE_H - 42 - padding * (len(visible) - 1)) // len(visible)
            card_h   = min(card_h, 118)

            for i, grp in enumerate(visible):
                cy = 40 + i * (card_h + padding)
                self._draw_idle_card(RX, cy, RW, card_h, grp)

    def _draw_idle_card(self, x, y, w, h, group: dict):
        """Single schedule group card in IDLE right panel."""
        import pygame

        freq      = group.get("frequency_type", "daily")
        med_name  = group.get("medication_name") or "Medication"
        pat_name  = group.get("patient_name", "")
        times     = sorted(group.get("times") or [])
        if not times and group.get("planned_time"):
            times = [str(group["planned_time"])]
        week_days = group.get("week_days", "")

        freq_col = _FREQ_COLOR.get(freq, C_PRIMARY)
        freq_dim = tuple(min(255, int(c * 0.22)) for c in freq_col)

        # Card background + border
        self._rr(C_SURFACE, x, y, w, h, r=10, border_col=C_BORDER)

        # Coloured left accent bar
        pygame.draw.rect(self._screen, freq_col, (x, y + 8, 3, h - 16), border_radius=2)

        # Frequency badge (right side of name row) — draw first to know its width
        freq_label = _FREQ_LABEL.get(freq, freq.title())
        badge_x = x + w - 82
        self._pill(freq_label, freq_col, freq_dim, badge_x, y + 12)

        # Medication name — pixel-based truncation so it never overlaps the badge
        f_heading = self._fonts["heading"]
        max_med_w = badge_x - (x + 14) - 6  # gap before badge
        label = med_name
        while label and f_heading.size(label)[0] > max_med_w:
            label = label[:-1]
        if label != med_name:
            label = label.rstrip() + "…"
        med_s = f_heading.render(label, True, C_TEXT1)
        self._screen.blit(med_s, (x + 14, y + 10))

        # Weekly day tags (small chips below name, if weekly)
        line2_y = y + 44
        if freq == "weekly":
            allowed_set = week_days_python_allowed(week_days)
            tx = x + 14
            for di in range(7):
                active = allowed_set is None or di in allowed_set
                col = freq_col if active else C_TEXT3
                bg  = freq_dim if active else C_SURFACE_HI
                tw  = self._pill(_DAY_ABBR[di], col, bg, tx, line2_y)
                tx += tw + 4
            line2_y += 24

        # Time chips
        now_hm = datetime.now().strftime("%H:%M")
        tx = x + 14
        for t in times[:6]:
            is_past  = (t < now_hm)
            chip_col = C_TEXT3 if is_past else C_PRIMARY
            chip_bg  = C_SURFACE_HI
            ts = self._fonts["badge"].render(t, True, chip_col)
            tw = ts.get_width() + 16
            pygame.draw.rect(self._screen, chip_bg, (tx, line2_y, tw, 22), border_radius=6)
            self._screen.blit(ts, (tx + 8, line2_y + 4))
            tx += tw + 6
            if tx > x + w - 50:
                break

        # Patient name (bottom-left, muted)
        if pat_name and h > 90:
            ps = self._txt(pat_name, "small", C_TEXT3)
            self._screen.blit(ps, (x + 14, y + h - 20))

    # ── WAITING ────────────────────────────────────────────────────────────────

    def _render_waiting(self):
        import pygame

        # Amber header strip
        pygame.draw.rect(self._screen, C_WARN_DIM, (0, 0, SCREEN_W, 58))
        title_s = self._txt("Medication Time!", "title", C_WARN)
        self._blit_cx(title_s, SCREEN_W // 2, 10)

        # Patient name
        name_s = self._txt(self._patient_name, "heading", C_WHITE)
        self._blit_cx(name_s, SCREEN_W // 2, 68)

        # Medication info card
        self._rr(C_SURFACE, 80, 100, SCREEN_W - 160, 54, r=12, border_col=C_BORDER)
        med_line = f"{self._medication_name}  ·  Slot {self._slot_id}"
        med_s = self._txt(med_line, "medium", C_TEXT1)
        self._blit_cx(med_s, SCREEN_W // 2, 118)

        # Pulsing face ring (center)
        cx, cy = SCREEN_W // 2, 248
        self._pulse_ring(cx, cy, base_r=40, color=C_PRIMARY, speed=2.2)
        self._icon_ring(cx, cy, 40, C_PRIMARY_DIM, C_PRIMARY, "[ ]", "medium")
        face_txt = self._txt("FACE", "badge", C_PRIMARY)
        self._blit_cx(face_txt, cx, cy - face_txt.get_height() // 2)

        # Instruction
        inst_s = self._txt("Please look at the camera", "body", C_TEXT2)
        self._blit_cx(inst_s, SCREEN_W // 2, 306)

        # Countdown progress bar
        frac      = self._countdown / 300.0
        bar_color = C_SUCCESS if frac > 0.40 else (C_WARN if frac > 0.15 else C_ERROR)
        self._bar(60, 340, SCREEN_W - 120, 10, frac, bar_color, r=5)

        mins = self._countdown // 60
        secs = self._countdown % 60
        cd_col = C_TEXT2 if self._countdown > 60 else C_ERROR
        cd_s = self._txt(f"{mins:02d}:{secs:02d} remaining", "body", cd_col)
        self._blit_cx(cd_s, SCREEN_W // 2, 358)

    # ── AUTHENTICATING ─────────────────────────────────────────────────────────
    # Roadmap: Pi AUTHENTICATING UI ≈ face_authentication/auth.py (cv2.imshow):
    #   Face Landmarker dots, liveness countdown, Blink/Mouth/Head OK row — see auth.py main().

    def _render_auth(self, camera_frame, face_locations):
        import pygame

        CAM_W, CAM_H = 560, 335
        cam_x = (SCREEN_W - CAM_W) // 2
        cam_y = 28

        if camera_frame is not None:
            self._frame_count += 1
            frame_rgb = camera_frame[:, :, ::-1]
            h, w      = frame_rgb.shape[:2]
            scale     = min(CAM_W / w, CAM_H / h)
            nw, nh    = int(w * scale), int(h * scale)
            off_x     = cam_x + (CAM_W - nw) // 2
            off_y     = cam_y + (CAM_H - nh) // 2

            surf = pygame.surfarray.make_surface(
                np.ascontiguousarray(frame_rgb.swapaxes(0, 1))
            )
            surf = pygame.transform.scale(surf, (nw, nh))
            self._screen.blit(surf, (off_x, off_y))

            if face_locations:
                for (top, right, bottom, left) in face_locations:
                    fx = off_x + int(left * scale)
                    fy = off_y + int(top  * scale)
                    fw = int((right - left)  * scale)
                    fh = int((bottom - top) * scale)
                    pygame.draw.rect(self._screen, C_SUCCESS,
                                     (fx, fy, fw, fh), 2, border_radius=4)
                hint = self._txt(
                    "Yüz göründü — kısa göz kırp, ağız aç veya başı çevir (liveness)",
                    "small",
                    C_SUCCESS,
                )
                self._blit_cx(hint, SCREEN_W // 2, cam_y + CAM_H + 5)
            else:
                hint = self._txt("Yüz aranıyor...", "small", C_WARN)
                self._blit_cx(hint, SCREEN_W // 2, cam_y + CAM_H + 5)
        else:
            self._rr(C_SURFACE, cam_x, cam_y, CAM_W, CAM_H, r=8)
            ws = self._txt("Camera starting...", "medium", C_TEXT3)
            self._blit_cx(ws, SCREEN_W // 2, cam_y + CAM_H // 2 - 14)

        # ── Corner brackets overlay ──────────────────────────────────────
        BL = 28   # bracket leg length
        BW = 3    # bracket line width
        bx1, by1 = cam_x + 10,       cam_y + 10
        bx2, by2 = cam_x + CAM_W - 10, cam_y + CAM_H - 10
        for (xx, yy, dx, dy) in [(bx1, by1, 1, 1), (bx2, by1, -1, 1),
                                  (bx1, by2, 1, -1), (bx2, by2, -1, -1)]:
            pygame.draw.line(self._screen, C_PRIMARY,
                             (xx, yy), (xx + dx * BL, yy), BW)
            pygame.draw.line(self._screen, C_PRIMARY,
                             (xx, yy), (xx, yy + dy * BL), BW)

        # ── Bottom info bar ──────────────────────────────────────────────
        info_y = USABLE_H - 36
        self._rr(C_SURFACE_HI, 0, info_y, SCREEN_W, 36)

        name_s = self._txt(f"Verifying: {self._patient_name}", "body", C_TEXT2)
        self._screen.blit(name_s, (16, info_y + 8))

        mins = self._countdown // 60
        secs = self._countdown % 60
        cd_col = C_TEXT2 if self._countdown > 60 else C_ERROR
        cd_s   = self._txt(f"{mins:02d}:{secs:02d}", "heading", cd_col)
        self._screen.blit(cd_s, cd_s.get_rect(right=SCREEN_W - 16, y=info_y + 5))

        frac  = self._countdown / 300.0
        bcol  = C_SUCCESS if frac > 0.4 else (C_WARN if frac > 0.15 else C_ERROR)
        self._bar(0, info_y, SCREEN_W, 4, frac, bcol, r=0)

    # ── MONITORING SWALLOW (post-dispense intake) ─────────────────────────────

    def _render_swallow_monitor(self, camera_frame):
        import pygame

        CAM_W, CAM_H = 560, 335
        cam_x = (SCREEN_W - CAM_W) // 2
        cam_y = 28

        if camera_frame is not None:
            self._frame_count += 1
            frame_rgb = camera_frame[:, :, ::-1]
            h, w = frame_rgb.shape[:2]
            scale = min(CAM_W / w, CAM_H / h)
            nw, nh = int(w * scale), int(h * scale)
            off_x = cam_x + (CAM_W - nw) // 2
            off_y = cam_y + (CAM_H - nh) // 2

            surf = pygame.surfarray.make_surface(
                np.ascontiguousarray(frame_rgb.swapaxes(0, 1))
            )
            surf = pygame.transform.scale(surf, (nw, nh))
            self._screen.blit(surf, (off_x, off_y))
        else:
            self._rr(C_SURFACE, cam_x, cam_y, CAM_W, CAM_H, r=8)
            ws = self._txt("Camera starting...", "medium", C_TEXT3)
            self._blit_cx(ws, SCREEN_W // 2, cam_y + CAM_H // 2 - 14)

        BL, BW = 28, 3
        bx1, by1 = cam_x + 10, cam_y + 10
        bx2, by2 = cam_x + CAM_W - 10, cam_y + CAM_H - 10
        for (xx, yy, dx, dy) in [(bx1, by1, 1, 1), (bx2, by1, -1, 1),
                                  (bx1, by2, 1, -1), (bx2, by2, -1, -1)]:
            pygame.draw.line(self._screen, C_ORANGE,
                             (xx, yy), (xx + dx * BL, yy), BW)
            pygame.draw.line(self._screen, C_ORANGE,
                             (xx, yy), (xx, yy + dy * BL), BW)

        title = self._txt("İlacı alın — yutkunma izleniyor", "medium", C_ORANGE)
        self._blit_cx(title, SCREEN_W // 2, cam_y + CAM_H + 6)

        hint = self._swallow_status or "Elinizi veya bardağı yüze yaklaştırın, sonra yutkunun"
        hint_s = self._txt(hint[:120], "small", C_TEXT2)
        self._blit_cx(hint_s, SCREEN_W // 2, cam_y + CAM_H + 34)

        info_y = USABLE_H - 36
        self._rr(C_SURFACE_HI, 0, info_y, SCREEN_W, 36)

        med_s = self._txt(self._medication_name or "İlaç", "body", C_TEXT1)
        self._screen.blit(med_s, (16, info_y + 8))

        mins = self._countdown // 60
        secs = self._countdown % 60
        cd_col = C_TEXT2 if self._countdown > 15 else C_WARN
        cd_s = self._txt(f"{mins:02d}:{secs:02d}", "heading", cd_col)
        self._screen.blit(cd_s, cd_s.get_rect(right=SCREEN_W - 16, y=info_y + 5))

        frac = min(1.0, self._countdown / 60.0) if self._countdown else 0.0
        bcol = C_SUCCESS if frac > 0.35 else (C_WARN if frac > 0.15 else C_ERROR)
        self._bar(0, info_y, SCREEN_W, 4, frac, bcol, r=0)

    # ── SUCCESS ────────────────────────────────────────────────────────────────

    def _render_success(self):
        import pygame

        pygame.draw.rect(self._screen, C_SUCCESS_DIM, (0, 0, SCREEN_W, SCREEN_H))

        # Icon ring
        cx, cy = SCREEN_W // 2, 148
        pygame.draw.circle(self._screen, (14, 62, 42), (cx, cy), 72)
        pygame.draw.circle(self._screen, C_SUCCESS, (cx, cy), 72, 3)
        ck = self._txt("OK", "heading", C_SUCCESS)
        self._blit_cx(ck, cx, cy - ck.get_height() // 2)

        # Verify label
        title_s = self._txt("Identity Verified", "title", C_SUCCESS)
        self._blit_cx(title_s, SCREEN_W // 2, 242)

        # Patient name
        name_s = self._txt(self._patient_name, "heading", C_WHITE)
        self._blit_cx(name_s, SCREEN_W // 2, 294)

        # Confidence bar
        pct   = int(self._auth_score * 100)
        bx    = SCREEN_W // 2 - 140
        self._bar(bx, 342, 280, 8, self._auth_score, C_SUCCESS, r=4)
        conf_s = self._txt(f"Confidence: {pct}%", "small", C_TEXT2)
        self._blit_cx(conf_s, SCREEN_W // 2, 358)

        # Dispensing notice
        disp_s = self._txt("Dispensing medication...", "body", C_WARN)
        self._blit_cx(disp_s, SCREEN_W // 2, 405)

    # ── DISPENSING ─────────────────────────────────────────────────────────────

    def _render_dispensing(self):
        import pygame

        pygame.draw.rect(self._screen, C_PRIMARY_DIM, (0, 0, SCREEN_W, SCREEN_H))

        # Animated arc ring
        cx, cy = SCREEN_W // 2, 150
        t = time.time()
        pygame.draw.circle(self._screen, (16, 38, 76), (cx, cy), 68)
        pygame.draw.circle(self._screen, C_BORDER, (cx, cy), 68, 3)

        arc_end = int((t * 160) % 360)
        for a in range(0, arc_end, 4):
            rad = math.radians(a - 90)
            px  = cx + int(68 * math.cos(rad))
            py  = cy + int(68 * math.sin(rad))
            pygame.draw.circle(self._screen, C_PRIMARY, (px, py), 2)

        # Center label
        dot_count = int(t * 1.5) % 4
        dots_s = self._txt("." * dot_count, "heading", C_PRIMARY)
        self._blit_cx(dots_s, cx, cy - dots_s.get_height() // 2)

        # Title
        title_s = self._txt("Dispensing", "title", C_PRIMARY)
        self._blit_cx(title_s, SCREEN_W // 2, 244)

        # Medication name
        med_s = self._txt(self._medication_name, "heading", C_WHITE)
        self._blit_cx(med_s, SCREEN_W // 2, 296)

        # Slot
        slot_s = self._txt(f"Slot {self._slot_id}", "body", C_TEXT2)
        self._blit_cx(slot_s, SCREEN_W // 2, 334)

        # Instruction
        inst_s = self._txt("Please collect your medication", "body", C_TEXT2)
        self._blit_cx(inst_s, SCREEN_W // 2, 395)

    # ── TIMEOUT ────────────────────────────────────────────────────────────────

    def _render_timeout(self):
        import pygame

        pygame.draw.rect(self._screen, C_ERROR_DIM, (0, 0, SCREEN_W, SCREEN_H))

        cx, cy = SCREEN_W // 2, 148
        pygame.draw.circle(self._screen, (66, 12, 12), (cx, cy), 68)
        pygame.draw.circle(self._screen, C_ERROR, (cx, cy), 68, 3)
        x_s = self._txt("X", "heading", C_ERROR)
        self._blit_cx(x_s, cx, cy - x_s.get_height() // 2)

        title_s = self._txt("Time Expired", "title", C_ERROR)
        self._blit_cx(title_s, SCREEN_W // 2, 244)

        if self._patient_name:
            pat_s = self._txt(f"{self._patient_name} — dose missed", "medium", C_TEXT1)
            self._blit_cx(pat_s, SCREEN_W // 2, 298)

        note_s = self._txt(
            "This event has been logged. Returning to standby...", "small", C_TEXT3
        )
        self._blit_cx(note_s, SCREEN_W // 2, 362)

    # ── MISSED ─────────────────────────────────────────────────────────────────

    def _render_missed(self):
        import pygame

        pygame.draw.rect(self._screen, C_ERROR_DIM, (0, 0, SCREEN_W, SCREEN_H))

        cx, cy = SCREEN_W // 2, 140
        pygame.draw.circle(self._screen, C_WARN_DIM, (cx, cy), 62)
        pygame.draw.circle(self._screen, C_WARN, (cx, cy), 62, 3)
        warn_s = self._txt("!", "heading", C_WARN)
        self._blit_cx(warn_s, cx, cy - warn_s.get_height() // 2)

        title_s = self._txt("Dose Missed", "title", C_ERROR)
        self._blit_cx(title_s, SCREEN_W // 2, 230)

        if self._medication_name:
            med_s = self._txt(self._medication_name, "heading", C_TEXT1)
            self._blit_cx(med_s, SCREEN_W // 2, 282)

        if self._patient_name:
            pat_s = self._txt(self._patient_name, "medium", C_TEXT2)
            self._blit_cx(pat_s, SCREEN_W // 2, 320)

        note_s = self._txt(
            "Logged. Please notify your caregiver.", "small", C_TEXT3
        )
        self._blit_cx(note_s, SCREEN_W // 2, 380)

    # ── ERROR ──────────────────────────────────────────────────────────────────

    def _render_error(self):
        err_s = self._txt("System Error", "title", C_ERROR)
        self._blit_cx(err_s, SCREEN_W // 2, 110)

        words = self._error_msg.split()
        lines, cur = [], ""
        for w in words:
            test = f"{cur} {w}".strip()
            if self._fonts["body"].size(test)[0] < SCREEN_W - 80:
                cur = test
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)

        y = 190
        for line in lines[:5]:
            s = self._txt(line, "body", C_TEXT1)
            self._blit_cx(s, SCREEN_W // 2, y)
            y += 32

    # ── Status bar (always visible at bottom) ──────────────────────────────────

    def _render_status_bar(self):
        import pygame

        bar_y = SCREEN_H - STATUS_H
        pygame.draw.rect(self._screen, (7, 8, 13), (0, bar_y, SCREEN_W, STATUS_H))
        pygame.draw.line(self._screen, C_BORDER, (0, bar_y), (SCREEN_W, bar_y), 1)

        # Time (left)
        ts = self._txt(datetime.now().strftime("%H:%M"), "small", C_TEXT3)
        self._screen.blit(ts, (10, bar_y + 7))

        # Brand (centre)
        bs = self._txt("MediDispense", "small", C_TEXT3)
        self._blit_cx(bs, SCREEN_W // 2, bar_y + 7)

        # State indicator dot + label (right)
        state_col = {
            DisplayState.IDLE:           C_SUCCESS,
            DisplayState.WAITING:        C_WARN,
            DisplayState.AUTHENTICATING: C_PRIMARY,
            DisplayState.SUCCESS:        C_SUCCESS,
            DisplayState.DISPENSING:     C_PRIMARY,
            DisplayState.MONITORING_SWALLOW: C_ORANGE,
            DisplayState.TIMEOUT:        C_ERROR,
            DisplayState.MISSED:         C_ERROR,
            DisplayState.ERROR:          C_ERROR,
        }.get(self._state, C_TEXT3)

        pretty = {
            DisplayState.MONITORING_SWALLOW: "Swallow",
        }.get(self._state, self._state.name.replace("_", " ").title())
        ss = self._txt(pretty, "small", state_col)
        sr    = ss.get_rect(right=SCREEN_W - 14, y=bar_y + 7)
        self._screen.blit(ss, sr)
        pygame.draw.circle(self._screen, state_col, (sr.left - 8, bar_y + STATUS_H // 2), 4)

    # ── Shutdown ───────────────────────────────────────────────────────────────

    def shutdown(self):
        self._running = False
        try:
            import pygame
            pygame.quit()
        except Exception:
            pass
        log.info("Display shut down")
