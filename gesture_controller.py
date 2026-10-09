"""
gesture_controller.py
Hand landmark detection using MediaPipe Tasks API (v1.0+).
Maps hand gestures to OS-level cursor/keyboard actions.
"""

import cv2
import numpy as np
import pyautogui
import time
import os

from mediapipe import Image, ImageFormat
from mediapipe.tasks.python       import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
    RunningMode,
    drawing_utils,
    drawing_styles,
)

pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0

# ── Landmark indices ──────────────────────────────────────────────────────────
# Tip ids: thumb=4, index=8, middle=12, ring=16, pinky=20
TIP_IDS = [4, 8, 12, 16, 20]
PIP_IDS = [3, 6, 10, 14, 18]   # second joints (for curl detection)
MCP_IDS = [2, 5,  9, 13, 17]   # knuckles

# ── Tunables ─────────────────────────────────────────────────────────────────
SMOOTHING       = 6      # frames to average for cursor
CLICK_THRESHOLD = 0.06   # normalised pinch distance → click
SCROLL_SPEED    = 25     # pixels per scroll event
CURSOR_SPEED    = 1.4    # amplification

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")


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
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        self._landmarker = HandLandmarker.create_from_options(opts)
        self._ts_ms      = 0   # monotonic timestamp for VIDEO mode

        # Smoothing buffers
        self._sx = [0] * SMOOTHING
        self._sy = [0] * SMOOTHING
        self._si = 0

        # State
        self.prev_pinch     = False
        self.prev_right     = False
        self.prev_double    = False
        self.dragging       = False
        self.scroll_ref_y   = None
        self.prev_zoom_dist = None
        self.gesture_label  = "Initialising…"
        self.click_cd       = 0          # cooldown frames

    # ── Public API ────────────────────────────────────────────────────────────
    def process(self, frame: np.ndarray):
        """Detect landmarks, fire actions, draw skeleton. Returns (frame, label)."""
        self._ts_ms += 33                # ~30 fps tick

        mp_img = Image(image_format=ImageFormat.SRGB,
                       data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = self._landmarker.detect_for_video(mp_img, self._ts_ms)

        hands = []
        if result.hand_landmarks:
            for lm_list, wlm, handedness in zip(
                result.hand_landmarks,
                result.hand_world_landmarks,
                result.handedness,
            ):
                # Convert normalised to pixel coords
                pixels = [
                    (int(lm.x * self.cam_w), int(lm.y * self.cam_h), lm.z)
                    for lm in lm_list
                ]
                label = handedness[0].display_name  # "Left" / "Right"
                hands.append((pixels, label))

                # Draw skeleton manually
                self._draw_hand(frame, lm_list)

        label = self._dispatch(hands)
        self.gesture_label = label
        if self.click_cd > 0:
            self.click_cd -= 1
        return frame, label

    def release(self):
        self._landmarker.close()

    # ── Drawing ───────────────────────────────────────────────────────────────
    def _draw_hand(self, frame, lm_list):
        """Draw hand skeleton directly."""
        pts = [
            (int(lm.x * self.cam_w), int(lm.y * self.cam_h))
            for lm in lm_list
        ]
        # Connections
        CONNECTIONS = [
            (0,1),(1,2),(2,3),(3,4),      # thumb
            (0,5),(5,6),(6,7),(7,8),      # index
            (0,9),(9,10),(10,11),(11,12), # middle
            (0,13),(13,14),(14,15),(15,16),# ring
            (0,17),(17,18),(18,19),(19,20),# pinky
            (5,9),(9,13),(13,17),          # palm
        ]
        for a, b in CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 180, 230), 2)
        # Joints
        for i, pt in enumerate(pts):
            r = 6 if i in TIP_IDS else 4
            color = (0, 220, 255) if i in TIP_IDS else (200, 200, 255)
            cv2.circle(frame, pt, r, color, -1)
            cv2.circle(frame, pt, r, (0, 0, 0), 1)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _fingers_up(self, lms):
        up = []
        # Thumb: compare x
        up.append(1 if lms[TIP_IDS[0]][0] < lms[TIP_IDS[0]-1][0] else 0)
        # Others: tip y < pip y → extended
        for i in range(1, 5):
            up.append(1 if lms[TIP_IDS[i]][1] < lms[PIP_IDS[i]][1] else 0)
        return up

    def _ndist(self, p1, p2):
        """Normalised euclidean distance (relative to cam width)."""
        return np.hypot(p1[0]-p2[0], p1[1]-p2[1]) / self.cam_w

    def _smooth_move(self, nx, ny):
        sx = int(max(0, min(nx * self.screen_w * CURSOR_SPEED, self.screen_w-1)))
        sy = int(max(0, min(ny * self.screen_h * CURSOR_SPEED, self.screen_h-1)))
        self._sx[self._si] = sx
        self._sy[self._si] = sy
        self._si = (self._si + 1) % SMOOTHING
        return int(np.mean(self._sx)), int(np.mean(self._sy))

    # ── Gesture dispatch ─────────────────────────────────────────────────────
    def _dispatch(self, hands):
        if not hands:
            self.scroll_ref_y   = None
            self.prev_zoom_dist = None
            if self.dragging:
                pyautogui.mouseUp()
                self.dragging = False
            return "No Hand Detected"

        lms, hand_label = hands[0]
        fingers = self._fingers_up(lms)
        count   = sum(fingers)

        idx_tip   = lms[8]
        thu_tip   = lms[4]
        mid_tip   = lms[12]

        pinch_d = self._ndist(thu_tip, idx_tip)
        rclik_d = self._ndist(thu_tip, mid_tip)
        dclk_d  = self._ndist(idx_tip, mid_tip)

        # ── ZOOM: two hands ───────────────────────────────────────────────
        if len(hands) == 2:
            lms2, _ = hands[1]
            c1 = np.mean([[l[0] for l in lms],  [l[1] for l in lms ]], axis=1)
            c2 = np.mean([[l[0] for l in lms2], [l[1] for l in lms2]], axis=1)
            dist = np.hypot(c1[0]-c2[0], c1[1]-c2[1])
            if self.prev_zoom_dist is not None:
                delta = dist - self.prev_zoom_dist
                if abs(delta) > 3:
                    pyautogui.hotkey("ctrl", "+" if delta > 0 else "-")
            self.prev_zoom_dist = dist
            self.scroll_ref_y   = None
            return "Zoom " + ("In 🔍" if (self.prev_zoom_dist or 0) > 0 else "Out 🔎")

        self.prev_zoom_dist = None

        # ── FIST ─────────────────────────────────────────────────────────
        if count == 0:
            if self.dragging:
                pyautogui.mouseUp()
                self.dragging = False
            self.scroll_ref_y = None
            return "Fist ✊ — Cursor Paused"

        # ── OPEN PALM → drag ─────────────────────────────────────────────
        if count == 5:
            cx, cy = self._smooth_move(idx_tip[0]/self.cam_w, idx_tip[1]/self.cam_h)
            if not self.dragging:
                pyautogui.mouseDown(cx, cy)
                self.dragging = True
            else:
                pyautogui.moveTo(cx, cy)
            self.scroll_ref_y = None
            return "Open Palm 🖐 — Dragging"

        if self.dragging:
            pyautogui.mouseUp()
            self.dragging = False

        # ── CURSOR MOVE: index only ───────────────────────────────────────
        if fingers == [0, 1, 0, 0, 0]:
            cx, cy = self._smooth_move(idx_tip[0]/self.cam_w, idx_tip[1]/self.cam_h)
            pyautogui.moveTo(cx, cy, _pause=False)
            self.scroll_ref_y = None
            return "☝️ Pointer — Move Cursor"

        # ── LEFT CLICK: pinch thumb+index ─────────────────────────────────
        if pinch_d < CLICK_THRESHOLD and self.click_cd == 0:
            if not self.prev_pinch:
                pyautogui.click()
                self.click_cd   = 18
                self.prev_pinch = True
                return "👌 Pinch — Left Click!"
        else:
            self.prev_pinch = False

        # ── RIGHT CLICK: thumb+middle ─────────────────────────────────────
        if (fingers[0]==1 and fingers[2]==1 and fingers[1]==0
                and rclik_d < CLICK_THRESHOLD and self.click_cd == 0):
            if not self.prev_right:
                pyautogui.rightClick()
                self.click_cd   = 22
                self.prev_right = True
                return "🖱️ Right Click!"
        else:
            self.prev_right = False

        # ── DOUBLE CLICK: index+middle touching ───────────────────────────
        if (fingers[1]==1 and fingers[2]==1
                and dclk_d < CLICK_THRESHOLD and self.click_cd == 0):
            if not self.prev_double:
                pyautogui.doubleClick()
                self.click_cd    = 28
                self.prev_double = True
                return "✌️ Double Click!"
        else:
            self.prev_double = False

        # ── SCROLL: index+middle extended, slide up/down ──────────────────
        if fingers[1]==1 and fingers[2]==1 and fingers[3]==0 and fingers[0]==0:
            mid_y = (idx_tip[1] + mid_tip[1]) / 2
            if self.scroll_ref_y is None:
                self.scroll_ref_y = mid_y
            delta = self.scroll_ref_y - mid_y
            if abs(delta) > 8:
                direction = 1 if delta > 0 else -1
                pyautogui.scroll(int(direction * SCROLL_SPEED * abs(delta) / self.cam_h * 12))
                self.scroll_ref_y = mid_y
            return "✌️ Two Fingers — Scroll"

        self.scroll_ref_y = None
        return f"Fingers: {fingers} ({count} up)"
