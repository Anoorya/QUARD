"""
ui.py  -  everything that is DRAWN in the QUARD window.

main.py decides *what is happening* (which gesture, which voice command, is
control paused ...); this module only turns that state into pixels, so the
look of the app can be changed without touching the control logic.

Notes
  * Pure OpenCV. Its built-in fonts are ASCII-only, so all text goes through
    ascii_text() (emoji would otherwise show as "???").
  * Colours are BGR tuples.
  * The layout is fixed at W x H (main.py resizes the camera image to it) and
    leaves the centre free for the camera: see gesture_controller.active_zone().
"""

import threading
import time

import cv2
import numpy as np

from gesture_controller import active_zone

# ─── Layout ──────────────────────────────────────────────────────────────────
W, H     = 1280, 720
PAD      = 6
TOP_Y, TOP_H       = 4, 48
BOT_H              = 56
PANEL_Y            = 60
PANEL_H            = 596            # 60 .. 656, matches the active zone height
LEFT_X,  LEFT_W    = PAD, 254       # 6 .. 260
RIGHT_X, RIGHT_W   = W - 220 + 4, 214   # 1064 .. 1278  (zone ends at 1056)

# ─── Palette (BGR) ───────────────────────────────────────────────────────────
BG      = (24, 20, 18)
PANEL   = (38, 31, 28)
CARD    = (54, 45, 41)
LINE    = (84, 72, 66)
TEXT    = (245, 240, 235)
DIM     = (160, 150, 145)
FAINT   = (110, 102, 98)
ACCENT  = (0, 190, 255)     # amber  - gestures / highlights
GREEN   = (110, 210, 90)    # ok / on / actions
RED     = (85, 85, 235)     # off / warnings
VOICE   = (235, 130, 255)   # pink   - voice
BLUE    = (255, 175, 70)

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX
AA        = cv2.LINE_AA

# ─── What the side panels list ───────────────────────────────────────────────
# (gesture id from GestureController, action name, how to do it)
GESTURES = [
    ("pointer", "Move cursor",  "Just move your hand"),
    ("click",   "Left click",   "Tap thumb to index finger"),
    ("drag",    "Drag",         "Hold that pinch, then move"),
    ("right",   "Right click",  "Tap thumb to middle finger"),
    ("double",  "Double click", "Opens files: thumb to ring"),
    ("scroll",  "Scroll",       "Make a fist, move up / down"),
    ("zoom",    "Zoom",         "Two hands apart / together"),
]

# (group title, [(text shown, [phrases it covers], is_risky), ...])
VOICE_GROUPS = [
    ("MOUSE", [
        ("click",            ["click", "left click"],              False),
        ("right click",      ["right click"],                      False),
        ("double click / open", ["double click", "open"],             False),
        ("scroll up / down", ["scroll up", "scroll down"],         False),
    ]),
    ("WINDOWS", [
        ("minimize",         ["minimize"],                         False),
        ("maximize",         ["maximize"],                         False),
        ("close window",     ["close", "close window"],            True),
        ("switch window",    ["switch window"],                    False),
        ("new tab / close tab", ["new tab", "close tab"],          False),
    ]),
    ("EDIT", [
        ("copy",             ["copy"],                             False),
        ("paste",            ["paste"],                            False),
        ("undo",             ["undo"],                             False),
        ("select all",       ["select all"],                       False),
    ]),
    ("SYSTEM", [
        ("zoom in / out",    ["zoom in", "zoom out"],              False),
        ("volume up / down", ["volume up", "volume down"],         False),
        ("mute",             ["mute"],                             False),
        ("screenshot",       ["screenshot"],                       False),
        ("go back / forward", ["go back", "go forward"],           False),
        ("press enter / escape / space", ["press enter", "press escape", "press space"], False),
        ("task manager",     ["task manager"],                     False),
    ]),
]

HELP_STEPS = [
    ("1", "Get set up",
     "Sit an arm's length away in good light, with your hand inside the amber box."),
    ("2", "Use your hand",
     "Move your hand to move. Thumb to index = click, thumb to ring = open files, fist = scroll."),
    ("3", "Use your voice",
     "Say a command from the right-hand list, e.g. \"scroll down\" or \"copy\", then pause."),
]
HELP_KEYS = [
    ("G", "Gestures on / off"), ("V", "Voice on / off"), ("SPACE", "Pause everything"),
    ("H", "Show / hide this help"), ("D", "Show tuning numbers"), ("Q", "Quit"),
]


# ─── Drawing helpers ─────────────────────────────────────────────────────────
def ascii_text(s, limit=80):
    """Strip characters OpenCV's Hershey fonts cannot draw, and cap the length."""
    s = str(s).encode("ascii", "ignore").decode("ascii")
    return s if len(s) <= limit else s[:limit - 3] + "..."


def text(img, s, x, y, scale=0.5, color=TEXT, thick=1, font=FONT, anchor="l", shadow=False, limit=80):
    """Draw anti-aliased text. anchor: 'l'eft, 'c'entre or 'r'ight of x. Returns the text width."""
    s = ascii_text(s, limit)
    (tw, _), _ = cv2.getTextSize(s, font, scale, thick)
    if anchor == "c":
        x -= tw // 2
    elif anchor == "r":
        x -= tw
    if shadow:
        cv2.putText(img, s, (x + 1, y + 1), font, scale, (0, 0, 0), thick + 1, AA)
    cv2.putText(img, s, (x, y), font, scale, color, thick, AA)
    return tw


def text_width(s, scale=0.5, thick=1, font=FONT):
    return cv2.getTextSize(ascii_text(s), font, scale, thick)[0][0]


def _round_mask(shape, x, y, w, h, r):
    mask = np.zeros(shape, np.uint8)
    r = max(0, min(r, h // 2, w // 2))
    cv2.rectangle(mask, (x + r, y), (x + w - r, y + h), 255, -1)
    cv2.rectangle(mask, (x, y + r), (x + w, y + h - r), 255, -1)
    for cx, cy in ((x + r, y + r), (x + w - r, y + r), (x + r, y + h - r), (x + w - r, y + h - r)):
        cv2.circle(mask, (cx, cy), r, 255, -1, AA)
    return mask


def rrect(img, x, y, w, h, color, alpha=1.0, r=10, outline=None):
    """Filled rounded rectangle, optionally translucent and outlined. Only touches its own region."""
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, img.shape[1]), min(y + h, img.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    roi  = img[y0:y1, x0:x1]
    mask = _round_mask(roi.shape[:2], x - x0, y - y0, w, h, r)
    over = np.empty_like(roi)
    over[:] = color
    blended = cv2.addWeighted(over, alpha, roi, 1 - alpha, 0)
    roi[mask > 0] = blended[mask > 0]
    if outline is not None:
        solid = (mask > 127).astype(np.uint8) * 255
        # borderValue=0 so edges that touch the region's border are still found
        inner = cv2.erode(solid, np.ones((3, 3), np.uint8), iterations=2,
                          borderType=cv2.BORDER_CONSTANT, borderValue=0)
        roi[(solid > 0) & (inner == 0)] = outline


def pill(img, x, y, label, color, filled=True, h=26):
    """Small rounded label. Returns its width so callers can lay pills out in a row."""
    tw = text_width(label, 0.45, 1)
    w = tw + 22
    if filled:
        rrect(img, x, y, w, h, color, 0.9, h // 2)
        text(img, label, x + 11, y + h - 8, 0.45, BG, 1, FONT_BOLD)
    else:
        rrect(img, x, y, w, h, PANEL, 0.9, h // 2, outline=color)
        text(img, label, x + 11, y + h - 8, 0.45, color, 1)
    return w


def corner_box(img, x1, y1, x2, y2, color, length=30, thick=3):
    """Four L-shaped corner brackets instead of a full rectangle."""
    for (cx, cy, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * length, cy), color, thick, AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * length), color, thick, AA)


def banner(img, msg, color, cx, y, scale=0.7):
    w = text_width(msg, scale, 1, FONT_BOLD) + 44
    rrect(img, cx - w // 2, y - 24, w, 40, (20, 16, 14), 0.78, 20, outline=color)
    text(img, msg, cx, y + 4, scale, color, 1, FONT_BOLD, "c")


# ─── The HUD ─────────────────────────────────────────────────────────────────
class Hud:
    """Holds the small bits of UI state (toasts, highlights) and draws every frame."""

    TOAST_SECONDS = 2.2
    HIGHLIGHT_SECONDS = 1.6

    def __init__(self):
        self._lock = threading.Lock()          # voice callbacks arrive on another thread
        self._toasts = []                      # [(text, colour, created_at)]
        self._voice_hit = ("", 0.0)            # (phrase, time)
        self._no_hand_since = time.time()
        self.frame_count = 0

    # -- called from main / voice thread ---------------------------------------
    def toast(self, msg, color=GREEN):
        with self._lock:
            self._toasts.append((msg, color, time.time()))
            del self._toasts[:-3]              # keep the last three

    def voice_hit(self, phrase):
        with self._lock:
            self._voice_hit = (phrase, time.time())
        self.toast('Voice: "%s"' % phrase, VOICE)

    # -- drawing -----------------------------------------------------------------
    def draw(self, frame, *, gesture_on, voice_on, paused, fps,
             gesture_id, gesture_label, hand_count, pointer,
             voice_state, voice_text, heard, debug=""):
        """
        gesture_on / voice_on : the G / V toggles
        paused                : SPACE pause (everything off)
        voice_state           : 'listening' | 'busy' | 'off' | 'error'
        voice_text            : status line for the voice section
        heard                 : last phrase understood by the recogniser
        """
        self.frame_count += 1
        now = time.time()
        if hand_count:
            self._no_hand_since = now

        self._draw_zone(frame, gesture_on and not paused, hand_count, pointer, gesture_id, now)
        self._draw_top(frame, gesture_on, voice_on, paused, fps)
        self._draw_gestures(frame, gesture_id, gesture_on and not paused)
        self._draw_voice_panel(frame, voice_on and not paused, now)
        self._draw_toasts(frame, now)
        self._draw_bottom(frame, gesture_label if gesture_on and not paused else "Off",
                          hand_count, voice_state, voice_text, heard, debug)
        return frame

    # -- centre: camera area ----------------------------------------------------------
    def _draw_zone(self, frame, active, hand_count, pointer, gesture_id, now):
        x1, y1, x2, y2 = active_zone(W, H)
        cx = (x1 + x2) // 2
        color = ACCENT if active else FAINT
        corner_box(frame, x1, y1, x2, y2, color)
        label = "ACTIVE ZONE - keep your hand inside"
        rrect(frame, x1 + 8, y1 + 8, text_width(label, 0.45) + 20, 24, (20, 16, 14), 0.7, 12)
        text(frame, label, x1 + 18, y1 + 25, 0.45, color)

        if pointer is not None and active:       # ring on the fingertip that moves the cursor
            ring = GREEN if gesture_id in ("pointer", "drag") else ACCENT if gesture_id in ("click", "right", "double") else DIM
            cv2.circle(frame, (int(pointer[0]), int(pointer[1])), 16, ring, 2, AA)

        if not active:
            banner(frame, "Hand control is OFF", RED, cx, (y1 + y2) // 2 - 10)
        elif hand_count == 0 and now - self._no_hand_since > 1.0:
            banner(frame, "Show your hand inside the box", ACCENT, cx, (y1 + y2) // 2 - 10)
        elif gesture_id == "idle":
            banner(frame, "Hand outside the box - nothing will happen", DIM, cx, y1 + 70, 0.6)

    # -- top bar ---------------------------------------------------------------------------
    def _draw_top(self, frame, gesture_on, voice_on, paused, fps):
        rrect(frame, PAD, TOP_Y, W - 2 * PAD, TOP_H, PANEL, 0.88, 14)
        cv2.circle(frame, (30, 28), 9, ACCENT, -1, AA)
        cv2.circle(frame, (30, 28), 15, ACCENT, 2, AA)
        text(frame, "QUARD", 54, 38, 0.95, TEXT, 2, FONT_BOLD)
        text(frame, "Hand + voice computer control", 170, 36, 0.5, DIM)

        x = 520
        x += pill(frame, x, 15, "[G] Hands " + ("ON" if gesture_on else "OFF"),
                  GREEN if gesture_on else RED, gesture_on) + 10
        x += pill(frame, x, 15, "[V] Voice " + ("ON" if voice_on else "OFF"),
                  GREEN if voice_on else RED, voice_on) + 10
        x += pill(frame, x, 15, "[SPACE] " + ("Resume" if paused else "Pause"),
                  RED if paused else DIM, paused) + 10
        pill(frame, x, 15, "[H] Help", ACCENT, False)

        fps_col = GREEN if fps > 20 else ACCENT if fps > 10 else RED
        text(frame, "%2.0f FPS" % fps, W - 22, 36, 0.5, fps_col, 1, FONT, "r")

    # -- left panel -------------------------------------------------------------------------
    def _draw_gestures(self, frame, active_id, enabled):
        rrect(frame, LEFT_X, PANEL_Y, LEFT_W, PANEL_H, PANEL, 0.88, 14)
        text(frame, "HAND GESTURES", LEFT_X + 14, PANEL_Y + 26, 0.55, ACCENT, 1, FONT_BOLD)
        text(frame, "Either hand works", LEFT_X + 14, PANEL_Y + 46, 0.42, DIM)

        row_h = 62
        y = PANEL_Y + 58
        for i, (gid, action, how) in enumerate(GESTURES):
            on = enabled and gid == active_id
            top = y + i * row_h
            rrect(frame, LEFT_X + 8, top, LEFT_W - 16, row_h - 6,
                  ACCENT if on else CARD, 0.28 if on else 0.6, 10, outline=ACCENT if on else None)
            cv2.circle(frame, (LEFT_X + 30, top + (row_h - 6) // 2), 12, ACCENT if on else LINE, -1, AA)
            text(frame, str(i + 1), LEFT_X + 30, top + (row_h - 6) // 2 + 5, 0.5, BG if on else DIM, 1, FONT_BOLD, "c")
            text(frame, action, LEFT_X + 52, top + 24, 0.55, TEXT if on else (225, 220, 215), 1, FONT_BOLD)
            text(frame, how, LEFT_X + 52, top + 44, 0.4, ACCENT if on else DIM)

        # Tips that don't deserve their own row
        ty = y + len(GESTURES) * row_h + 4
        text(frame, "TIPS", LEFT_X + 14, ty + 12, 0.4, FAINT, 1, FONT_BOLD)
        text(frame, "Or tap thumb+index twice quickly", LEFT_X + 14, ty + 32, 0.4, DIM)
        text(frame, "Hand outside the box = idle", LEFT_X + 14, ty + 50, 0.4, DIM)

    # -- right panel ------------------------------------------------------------------------
    def _draw_voice_panel(self, frame, enabled, now):
        rrect(frame, RIGHT_X, PANEL_Y, RIGHT_W, PANEL_H, PANEL, 0.88, 14)
        text(frame, "VOICE COMMANDS", RIGHT_X + 14, PANEL_Y + 26, 0.55, VOICE, 1, FONT_BOLD)
        text(frame, "Say it, then pause", RIGHT_X + 14, PANEL_Y + 46, 0.42, DIM)

        with self._lock:
            hit, hit_t = self._voice_hit
        hit_on = enabled and (now - hit_t) < self.HIGHLIGHT_SECONDS

        y = PANEL_Y + 62
        for title, entries in VOICE_GROUPS:
            text(frame, title, RIGHT_X + 14, y + 12, 0.4, FAINT, 1, FONT_BOLD)
            cv2.line(frame, (RIGHT_X + 14 + text_width(title, 0.4, 1, FONT_BOLD) + 8, y + 8),
                     (RIGHT_X + RIGHT_W - 14, y + 8), LINE, 1, AA)
            y += 20
            for label, phrases, risky in entries:
                lit = hit_on and hit in phrases
                if lit:
                    rrect(frame, RIGHT_X + 8, y - 1, RIGHT_W - 16, 20, VOICE, 0.35, 7, outline=VOICE)
                col = VOICE if lit else (RED if risky else TEXT)
                text(frame, '"%s"' % label, RIGHT_X + 16, y + 14, 0.42, col)
                y += 21
            y += 4

    # -- toasts (what just happened) --------------------------------------------------------------
    def _draw_toasts(self, frame, now):
        x1, y1, x2, y2 = active_zone(W, H)
        cx = (x1 + x2) // 2
        with self._lock:
            self._toasts = [t for t in self._toasts if now - t[2] < self.TOAST_SECONDS]
            toasts = list(self._toasts)
        for i, (msg, color, t0) in enumerate(reversed(toasts)):       # newest at the bottom
            age = now - t0
            alpha = 1.0 if age < self.TOAST_SECONDS - 0.5 else max(0.0, (self.TOAST_SECONDS - age) / 0.5)
            w = text_width(msg, 0.65, 1, FONT_BOLD) + 40
            y = y2 - 28 - i * 46
            rrect(frame, cx - w // 2, y - 26, w, 38, (20, 16, 14), 0.8 * alpha, 19, outline=color if alpha > 0.5 else None)
            text(frame, msg, cx, y, 0.65, color, 1, FONT_BOLD, "c")

    # -- bottom status bar --------------------------------------------------------------------------
    def _draw_bottom(self, frame, gesture_label, hand_count, voice_state, voice_text, heard, debug=""):
        y0 = H - BOT_H - 4
        rrect(frame, PAD, y0, W - 2 * PAD, BOT_H, PANEL, 0.88, 14)
        mid = W // 2

        # left: hand
        text(frame, "HAND", 22, y0 + 24, 0.45, DIM, 1, FONT_BOLD)
        text(frame, gesture_label, 76, y0 + 24, 0.62, ACCENT, 1, FONT_BOLD, limit=34)
        if debug:       # D key: live numbers to see how close each pinch is to triggering
            text(frame, debug, 22, y0 + 46, 0.42, BLUE, limit=70)
        else:
            hands = "%d hand%s seen" % (hand_count, "" if hand_count == 1 else "s")
            text(frame, hands, 22, y0 + 46, 0.42, DIM)
            text(frame, "G hands   V voice   SPACE pause   H help   D tuning   Q quit", 160, y0 + 46, 0.42, FAINT)

        cv2.line(frame, (mid, y0 + 8), (mid, y0 + BOT_H - 8), LINE, 1)

        # right: voice
        mic_col = {"listening": GREEN, "busy": ACCENT, "off": FAINT, "error": RED}[voice_state]
        mx, my = mid + 28, y0 + 22
        if voice_state == "listening":
            pulse = int(abs(np.sin(self.frame_count * 0.12)) * 8) + 9
            cv2.circle(frame, (mx, my), pulse, mic_col, 2, AA)
        cv2.circle(frame, (mx, my), 7, mic_col, -1, AA)
        text(frame, voice_text, mid + 52, y0 + 26, 0.55, mic_col if voice_state != "off" else DIM, 1, FONT_BOLD, limit=42)
        if heard:
            text(frame, 'Heard: "%s"' % heard, mid + 18, y0 + 46, 0.42, VOICE, limit=60)
        else:
            text(frame, "Nothing heard yet", mid + 18, y0 + 46, 0.42, FAINT)

    # -- help overlay ---------------------------------------------------------------------------------
    def draw_help(self, frame, first_run):
        frame[:] = cv2.addWeighted(np.zeros_like(frame), 0.78, frame, 0.22, 0)
        cw, ch = 860, 540
        x, y = (W - cw) // 2, (H - ch) // 2
        rrect(frame, x, y, cw, ch, PANEL, 0.97, 22, outline=LINE)

        text(frame, "Welcome to QUARD" if first_run else "How QUARD works", x + 40, y + 62, 1.0, TEXT, 2, FONT_BOLD)
        text(frame, "Control your computer with your hand and your voice.", x + 40, y + 92, 0.55, DIM)

        for i, (num, title, body) in enumerate(HELP_STEPS):
            top = y + 120 + i * 82
            rrect(frame, x + 30, top, cw - 60, 72, CARD, 0.8, 12)
            cv2.circle(frame, (x + 66, top + 36), 18, ACCENT, -1, AA)
            text(frame, num, x + 66, top + 43, 0.7, BG, 1, FONT_BOLD, "c")
            text(frame, title, x + 100, top + 28, 0.62, TEXT, 1, FONT_BOLD)
            text(frame, body, x + 100, top + 54, 0.46, DIM, limit=110)

        text(frame, "KEYS", x + 40, y + 392, 0.5, ACCENT, 1, FONT_BOLD)
        for i, (k, desc) in enumerate(HELP_KEYS):
            cx = x + 40 + (i % 3) * 270
            cy = y + 416 + (i // 3) * 36
            kw = pill(frame, cx, cy, k, ACCENT, False, 26)
            text(frame, desc, cx + kw + 10, cy + 18, 0.46, TEXT)

        text(frame, "Safety: 'close window' sends Alt+F4. Press SPACE any time to pause everything.",
             x + 40, y + 498, 0.42, FAINT, limit=100)
        banner(frame, "Press H or SPACE to start", GREEN, W // 2, y + ch + 38, 0.7)
