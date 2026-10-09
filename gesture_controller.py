"""
gesture_controller.py
Hand landmark detection using MediaPipe Tasks API (v1.0+).
Maps hand gestures to OS-level cursor/keyboard actions.

Gestures - deliberately few and forgiving:
  move your hand            move the cursor (any hand shape works)
  tap thumb to index        left click      (tap twice quickly = double click)
  hold that pinch, move     drag, release to drop
  tap thumb to middle       right click
  tap thumb to ring         double click    (opens files and folders)
  make a fist, move up/down scroll
  two hands apart/together  zoom
  hand outside the box      idle (nothing happens)

A thumb+index pinch behaves like a physical mouse button: touching presses it,
letting go releases it. A quick tap is therefore a click, a longer hold followed
by movement is a drag, and two quick taps are a double click. After a tap the
cursor stays put for a moment so the second tap lands on exactly the same
spot - Windows only treats two clicks as a double click if they are within a
few pixels of each other.

Detection uses MediaPipe's 3D "world" landmarks (real-world metres), so it
works however your hand is tilted or far from the camera. Pixel landmarks are
only used for drawing and for the cursor position.
"""

import os
import time
from collections import deque

import cv2
import numpy as np
import pyautogui

from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0

# ── Landmark indices ──────────────────────────────────────────────────────────
WRIST = 0
THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP = 4, 8, 12, 16
PALM_POINTS = (0, 5, 9, 13, 17)          # averaged → steady "hand position"
# (mcp, pip, dip, tip) per finger, for the "is it curled?" test
FINGERS = {
    "index":  (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring":   (13, 14, 15, 16),
    "pinky":  (17, 18, 19, 20),
}
TIP_IDS = [4, 8, 12, 16, 20]

# ── Tunables (distances are relative to hand size, so they work at any range) ─
SMOOTHING       = 5      # frames to average for cursor (higher = steadier but laggier)
EDGE_MARGIN     = 0.08   # fraction of the active zone you don't need to reach to hit the screen edge
PINCH_ON        = 0.40   # thumb-to-fingertip distance that STARTS a pinch (bigger = easier, but a
                         #   relaxed hand sits around 0.6-0.8, so don't go much higher)
PINCH_OFF       = 0.50   # distance that ENDS a pinch (> PINCH_ON so it doesn't flicker; keep it close
                         #   to PINCH_ON so two quick taps register as two separate pinches)
PINCH_REACH     = 0.90   # a pinching fingertip must be this far from the wrist (rules out a fist)
FIST_STRAIGHT   = 0.62   # every finger "straightness" below this = fist (straight finger ≈ 0.95)
DRAG_HOLD_S     = 0.35   # seconds a thumb+index pinch must be held before the cursor follows your hand (drag)
PINCH_COOLDOWN  = 3      # frames before another pinch can start
TAP_LOCK_S      = 0.45   # after a tap the cursor stays put this long, so a quick second tap = double click
TAP_LOCK_BREAK  = 45     # ...unless the hand moves more than this many camera pixels (you meant to move)
ZONE_SLACK      = 40     # pixels outside the active zone still counted as inside
SCROLL_GAIN     = 100    # wheel units per (scroll_distance / frame_height) * SCROLL_SPEED
SCROLL_SPEED    = 25
SCROLL_DEADZONE = 8      # pixels of vertical motion before scrolling starts
ZOOM_STEP_RATIO = 0.02   # two-hand distance change (fraction of frame width) per zoom step
ZOOM_COOLDOWN   = 6      # frames between zoom key presses

# Region of the camera frame (not covered by the HUD panels) that maps to the
# whole screen. Shared with ui.py so the drawn "active zone" is the real one.
ZONE_LEFT, ZONE_TOP        = 264, 60
ZONE_RIGHT_PAD, ZONE_BOTTOM_PAD = 224, 64

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

# Label prefix → gesture id (used by the HUD to highlight the active gesture).
# ids: none, idle, pointer, click, drag, right, double, scroll, zoom
_LABEL_TO_ID = [
    ("No Hand", "none"), ("Idle", "idle"), ("Pointer", "pointer"), ("Dragging", "drag"),
    ("Dropped", "drag"), ("Pinch", "click"), ("Right Click", "right"), ("Double Click", "double"),
    ("Fist", "scroll"), ("Zoom", "zoom"),
]

_PINCH_TIPS = (("click", INDEX_TIP), ("right", MIDDLE_TIP), ("double", RING_TIP))


def active_zone(w: int, h: int):
    """Return (x1, y1, x2, y2) of the active tracking zone for a w x h frame."""
    return ZONE_LEFT, ZONE_TOP, w - ZONE_RIGHT_PAD, h - ZONE_BOTTOM_PAD


class GestureController:
    """Detects hand landmarks per-frame and fires OS actions accordingly."""

    def __init__(self, cam_w: int, cam_h: int):
        self.cam_w, self.cam_h = cam_w, cam_h
        self.screen_w, self.screen_h = pyautogui.size()

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}\n"
                "Download from: https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            )

        opts = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = HandLandmarker.create_from_options(opts)
        self._ts_ms      = 0   # strictly increasing timestamp for VIDEO mode

        self._buf = deque(maxlen=SMOOTHING)   # cursor smoothing buffer

        # State
        self.dragging       = False      # left button held AND following the hand
        self._btn_down      = False      # left button currently held down by us
        self.scroll_ref_y   = None
        self.prev_zoom_dist = None
        self.zoom_cd        = 0
        self._pinch_kind    = None       # None | 'click' | 'right' | 'double' while thumb and finger touch
        self._pinch_t0      = 0.0
        self._pinch_cd      = 0
        self._lock_until    = 0.0        # cursor held still until this time after a tap (double-click window)
        self._lock_palm     = None       # where the hand was when the lock started
        self._cursor        = None       # last cursor position we sent (screen px)
        self._pinch_vis     = None       # (thumb_px, finger_px, ratio) for the on-screen pinch meter
        self.gesture_label  = "Initialising..."
        self.gesture_id     = "none"     # lets the HUD highlight the active gesture
        self.hand_count     = 0
        self.pointer        = None       # (x, y) of the tracked hand position in frame pixels
        self.debug_text     = ""         # live numbers, shown when the user presses D
        self._events        = deque(maxlen=20)   # actions fired since the HUD last asked

    # ── Public API ────────────────────────────────────────────────────────────
    def process(self, frame: np.ndarray):
        """Detect landmarks, fire actions, draw skeleton. Returns (frame, label)."""
        # VIDEO mode needs monotonically increasing timestamps (ms)
        self._ts_ms = max(self._ts_ms + 1, int(time.monotonic() * 1000))

        mp_img = Image(image_format=ImageFormat.SRGB,
                       data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = self._landmarker.detect_for_video(mp_img, self._ts_ms)

        hands = []   # (pixel landmarks [21x2], world landmarks [21x3])
        for lm_list, world_list in zip(result.hand_landmarks or [], result.hand_world_landmarks or []):
            px = np.array([(lm.x * self.cam_w, lm.y * self.cam_h) for lm in lm_list])
            wd = np.array([(lm.x, lm.y, lm.z) for lm in world_list])
            hands.append((px, wd))
            self._draw_hand(frame, px)

        self._pinch_vis = None
        label = self._dispatch(hands)
        if self._pinch_vis is not None:
            self._draw_pinch_meter(frame, *self._pinch_vis)

        self.gesture_label = label
        self.gesture_id    = next((gid for prefix, gid in _LABEL_TO_ID if label.startswith(prefix)), "other")
        self.hand_count    = len(hands)
        self.pointer       = tuple(int(v) for v in self._palm_center(hands[0][0])) if hands else None
        if self._pinch_cd > 0:
            self._pinch_cd -= 1
        if self.zoom_cd > 0:
            self.zoom_cd -= 1
        return frame, label

    def pop_events(self):
        """Return (and clear) the names of actions fired since the last call, e.g. 'Left click'."""
        events = list(self._events)
        self._events.clear()
        return events

    def reset(self):
        """Release any held mouse button and clear transient state."""
        self._release_button()
        self.scroll_ref_y   = None
        self.prev_zoom_dist = None
        self._pinch_kind    = None
        self._lock_until    = 0.0
        self._cursor        = None
        self._buf.clear()

    def release(self):
        self.reset()
        self._landmarker.close()

    # ── Drawing ───────────────────────────────────────────────────────────────
    _CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
        (0, 5), (5, 6), (6, 7), (7, 8),          # index
        (0, 9), (9, 10), (10, 11), (11, 12),     # middle
        (0, 13), (13, 14), (14, 15), (15, 16),   # ring
        (0, 17), (17, 18), (18, 19), (19, 20),   # pinky
        (5, 9), (9, 13), (13, 17),               # palm
    ]

    def _draw_hand(self, frame, px):
        """Draw hand skeleton directly."""
        pts = [(int(x), int(y)) for x, y in px]
        for a, b in self._CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 180, 230), 2, cv2.LINE_AA)
        for i, pt in enumerate(pts):
            r = 6 if i in TIP_IDS else 4
            color = (0, 220, 255) if i in TIP_IDS else (200, 200, 255)
            cv2.circle(frame, pt, r, color, -1, cv2.LINE_AA)
            cv2.circle(frame, pt, r, (0, 0, 0), 1, cv2.LINE_AA)

    @staticmethod
    def _draw_pinch_meter(frame, a, b, ratio):
        """Line between thumb and finger tip that turns green when the pinch registers.

        Gives live feedback on how close you are to a click.
        """
        a, b = (int(a[0]), int(a[1])), (int(b[0]), int(b[1]))
        if ratio < PINCH_ON:
            cv2.line(frame, a, b, (110, 210, 90), 6, cv2.LINE_AA)
            cv2.circle(frame, ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2), 14, (110, 210, 90), 3, cv2.LINE_AA)
        elif ratio < PINCH_OFF:
            cv2.line(frame, a, b, (0, 190, 255), 3, cv2.LINE_AA)

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _palm_center(px):
        return tuple(np.mean(px[list(PALM_POINTS)], axis=0))

    @staticmethod
    def _dist(a, b):
        return float(np.linalg.norm(a - b))

    def _straightness(self, wd, finger):
        """1.0 = fully straight finger, ~0.4 = tightly curled. Works at any hand angle (3D)."""
        mcp, pip, dip, tip = FINGERS[finger]
        path = self._dist(wd[mcp], wd[pip]) + self._dist(wd[pip], wd[dip]) + self._dist(wd[dip], wd[tip])
        return self._dist(wd[mcp], wd[tip]) / max(path, 1e-6)

    def _pinch_ratio(self, wd, tip_id, hand_size):
        """Thumb-to-fingertip 3D distance / hand size, or None if that fingertip isn't held out."""
        if self._dist(wd[tip_id], wd[WRIST]) / hand_size < PINCH_REACH:
            return None                       # curled finger (e.g. a fist), not a pinch
        return self._dist(wd[THUMB_TIP], wd[tip_id]) / hand_size

    def _screen_point(self, px_point):
        """Map a camera-pixel point inside the active zone to a smoothed screen point."""
        x1, y1, x2, y2 = active_zone(self.cam_w, self.cam_h)
        mx, my = (x2 - x1) * EDGE_MARGIN, (y2 - y1) * EDGE_MARGIN
        nx = min(max((px_point[0] - x1 - mx) / max(x2 - x1 - 2 * mx, 1), 0.0), 1.0)
        ny = min(max((px_point[1] - y1 - my) / max(y2 - y1 - 2 * my, 1), 0.0), 1.0)
        self._buf.append((nx * (self.screen_w - 1), ny * (self.screen_h - 1)))
        xs, ys = zip(*self._buf)
        self._cursor = (int(np.mean(xs)), int(np.mean(ys)))
        return self._cursor

    def _inside_zone(self, p):
        x1, y1, x2, y2 = active_zone(self.cam_w, self.cam_h)
        return (x1 - ZONE_SLACK <= p[0] <= x2 + ZONE_SLACK) and (y1 - ZONE_SLACK <= p[1] <= y2 + ZONE_SLACK)

    def _release_button(self):
        """Let go of the left mouse button if we are holding it (end of a click or drag)."""
        if self._btn_down:
            pyautogui.mouseUp()
            self._btn_down = False
        self.dragging = False

    # ── Gesture dispatch ─────────────────────────────────────────────────────
    def _dispatch(self, hands):
        if not hands:
            self.reset()
            self.debug_text = ""
            return "No Hand Detected"

        # ── ZOOM: two hands ───────────────────────────────────────────────
        if len(hands) == 2:
            self._release_button()
            self._pinch_kind  = None
            self._lock_until  = 0.0
            self.scroll_ref_y = None
            self._buf.clear()
            c1, c2 = self._palm_center(hands[0][0]), self._palm_center(hands[1][0])
            dist = float(np.hypot(c1[0] - c2[0], c1[1] - c2[1]))
            label = "Zoom (two hands)"
            if self.prev_zoom_dist is not None:
                delta = dist - self.prev_zoom_dist
                if abs(delta) > ZOOM_STEP_RATIO * self.cam_w:
                    label = "Zoom In" if delta > 0 else "Zoom Out"
                    if self.zoom_cd == 0:
                        pyautogui.hotkey("ctrl", "+" if delta > 0 else "-")
                        self._events.append(label)
                        self.zoom_cd = ZOOM_COOLDOWN
                    self.prev_zoom_dist = dist
                # tiny jitter: keep the old reference so slow movement still accumulates
            else:
                self.prev_zoom_dist = dist
            return label

        self.prev_zoom_dist = None

        px, wd = hands[0]
        hand_size = max(self._dist(wd[WRIST], wd[9]), 1e-6)      # wrist → middle knuckle (metres)
        palm = self._palm_center(px)

        straight = {f: self._straightness(wd, f) for f in FINGERS}
        ratios   = {kind: self._pinch_ratio(wd, tip, hand_size) for kind, tip in _PINCH_TIPS}
        self.debug_text = "pinch idx %s mid %s ring %s (<%.2f)  fingers %s" % (
            *("%.2f" % r if r is not None else "--" for r in (ratios["click"], ratios["right"], ratios["double"])),
            PINCH_ON, " ".join("%.1f" % s for s in straight.values()))

        # Pinch meter shows the pinch being made (or the closest candidate)
        near = [(r, k) for k, r in ratios.items() if r is not None]
        if near:
            r, kind = min(near)
            tip = dict(_PINCH_TIPS)[kind]
            self._pinch_vis = (px[THUMB_TIP], px[tip], r)

        # Hand left the box → idle (let go of anything held)
        if not self._inside_zone(palm):
            self._release_button()
            self._pinch_kind = None
            self._lock_until = 0.0
            self.scroll_ref_y = None
            self._buf.clear()
            self._pinch_vis = None
            return "Idle - hand outside the box"

        # ── PINCH (thumb touching a fingertip) ────────────────────────────
        # Checked first so a pinch is never mistaken for another pose.
        now = time.monotonic()
        if self._pinch_kind is None:
            close = [(r, k) for r, k in near if r < PINCH_ON]
            if close and self._pinch_cd == 0:
                _, self._pinch_kind = min(close)
                self._pinch_t0 = now
                self._lock_until = 0.0
                self.scroll_ref_y = None
                if self._pinch_kind == "right":
                    pyautogui.rightClick()
                    self._events.append("Right click")
                    return "Right Click!"
                if self._pinch_kind == "double":
                    pyautogui.doubleClick()                    # opens files and folders
                    self._events.append("Double click")
                    return "Double Click!"
                # Thumb+index acts like the physical button: touching presses it ...
                pyautogui.mouseDown()
                self._btn_down = True
                return "Pinch - hold to drag"
        else:
            r = ratios[self._pinch_kind]
            if r is not None and r < PINCH_OFF:               # still pinching
                if self._pinch_kind == "right":
                    return "Right Click (release)"
                if self._pinch_kind == "double":
                    return "Double Click (release)"
                if not self.dragging and now - self._pinch_t0 >= DRAG_HOLD_S:
                    self.dragging = True                        # held long enough: the cursor now follows
                    self._events.append("Drag")
                if self.dragging:
                    pyautogui.moveTo(*self._screen_point(palm))
                    return "Dragging - release to drop"
                return "Pinch - hold to drag"                   # cursor stays frozen, so a tap is a clean click
            # ... and letting go releases it.
            kind, self._pinch_kind = self._pinch_kind, None
            self._pinch_cd = PINCH_COOLDOWN
            if kind == "click":
                was_drag = self.dragging
                self._release_button()
                if was_drag:
                    return "Dropped"
                self._events.append("Left click")
                self._lock_until = now + TAP_LOCK_S             # keep the cursor still for a 2nd tap
                self._lock_palm = palm
                return "Pinch - Left Click!"

        # ── FIST: scroll (all four fingers curled) ────────────────────────
        if all(s < FIST_STRAIGHT for s in straight.values()):
            self._buf.clear()
            y = palm[1]
            if self.scroll_ref_y is None:
                self.scroll_ref_y = y
            delta = self.scroll_ref_y - y
            if abs(delta) > SCROLL_DEADZONE:
                amount = int(delta / self.cam_h * SCROLL_SPEED * SCROLL_GAIN)
                if amount:
                    pyautogui.scroll(amount)
                self.scroll_ref_y = y
            return "Fist - Scroll (move up / down)"
        self.scroll_ref_y = None

        # ── POINTER: anything else - the cursor simply follows your hand ──
        # (the tracked point is the palm centre, which barely moves while you pinch)
        if self._lock_until:
            moved = float(np.linalg.norm(np.subtract(palm, self._lock_palm)))
            if now < self._lock_until and moved < TAP_LOCK_BREAK:
                return "Pointer - steady (tap again to double click)"
            self._lock_until = 0.0              # lock over, or you moved on purpose
            self._buf.clear()                   # jump straight to the hand, no stale smoothing
        pyautogui.moveTo(*self._screen_point(palm))
        return "Pointer - Move Cursor"
