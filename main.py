"""
main.py  –  Hand Gesture + Voice Command Computer Controller
Run with:  python main.py
Press  Q  in the camera window to quit.
Press  V  to toggle voice control on/off.
Press  G  to toggle gesture control on/off.
"""

import cv2
import numpy as np
import time
import threading
import sys

from gesture_controller import GestureController
from voice_controller    import VoiceController

# ─── CONFIG ─────────────────────────────────────────────────────────────────
CAM_INDEX   = 0
CAM_W, CAM_H = 1280, 720
FPS_CAP     = 30

# HUD colour palette (BGR)
CLR_BG      = (15,  15,  30)
CLR_ACCENT  = (0,  200, 255)    # cyan
CLR_GREEN   = (50, 220, 100)
CLR_ORANGE  = (30, 160, 255)
CLR_RED     = (50,  50, 230)
CLR_WHITE   = (240, 240, 255)
CLR_DIM     = (100, 100, 130)
CLR_VOICE   = (200, 100, 255)   # purple for voice

FONT        = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD   = cv2.FONT_HERSHEY_DUPLEX


# ─── Gesture reference card ──────────────────────────────────────────────────
GESTURE_GUIDE = [
    ("☝️  Index only",    "Move Cursor"),
    ("👌 Pinch Thumb+Idx","Left Click"),
    ("✌️  2-finger touch","Double Click"),
    ("✌️  2-finger slide","Scroll Up/Down"),
    ("🤟 Thumb+Mid",      "Right Click"),
    ("✊ Fist",           "Pause Control"),
    ("🖐 Open Palm",      "Drag"),
    ("🤲 Both Hands",     "Zoom In/Out"),
]

VOICE_GUIDE = [
    '"click"',        '"right click"',
    '"double click"', '"scroll up/down"',
    '"zoom in/out"',  '"screenshot"',
    '"minimize"',     '"maximize"',
    '"close"',        '"copy/paste"',
    '"volume up/down"','"mute"',
    '"switch window"', '"press enter"',
]


def draw_rounded_rect(img, x, y, w, h, r, color, thickness=-1, alpha=1.0):
    """Draw a filled or outlined rounded rectangle."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x+r, y), (x+w-r, y+h), color, thickness)
    cv2.rectangle(overlay, (x, y+r), (x+w, y+h-r), color, thickness)
    for cx, cy in [(x+r, y+r), (x+w-r, y+r), (x+r, y+h-r), (x+w-r, y+h-r)]:
        cv2.circle(overlay, (cx, cy), r, color, thickness)
    if alpha < 1.0:
        cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
    else:
        img[:] = overlay


def draw_hud(frame, gesture_label, voice_label, voice_listening,
             gesture_on, voice_on, fps, frame_count):
    h, w = frame.shape[:2]

    # ── Semi-transparent overlay panels ─────────────────────────────────────
    overlay = frame.copy()

    # Top status bar
    cv2.rectangle(overlay, (0, 0), (w, 56), (10, 10, 22), -1)

    # Left panel (gesture guide)
    cv2.rectangle(overlay, (0, 56), (260, h), (12, 12, 25), -1)

    # Right panel (voice guide)
    cv2.rectangle(overlay, (w-220, 56), (w, h), (12, 12, 25), -1)

    # Bottom status bar
    cv2.rectangle(overlay, (0, h-60), (w, h), (10, 10, 22), -1)

    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    # ── TOP BAR ─────────────────────────────────────────────────────────────
    # Logo / title
    cv2.putText(frame, "GESTURE", (12, 34), FONT_BOLD, 0.8, CLR_ACCENT, 2)
    cv2.putText(frame, " + VOICE", (110, 34), FONT_BOLD, 0.8, CLR_VOICE, 2)
    cv2.putText(frame, "CONTROLLER", (240, 34), FONT_BOLD, 0.8, CLR_WHITE, 2)

    # FPS
    fps_color = CLR_GREEN if fps > 20 else CLR_ORANGE if fps > 10 else CLR_RED
    cv2.putText(frame, f"FPS: {fps:2.0f}", (w-210, 36), FONT, 0.65, fps_color, 2)

    # Gesture / Voice toggle indicators
    g_col = CLR_GREEN if gesture_on else CLR_RED
    v_col = CLR_GREEN if voice_on   else CLR_RED
    g_txt = "GESTURE [G]: ON " if gesture_on else "GESTURE [G]: OFF"
    v_txt = "VOICE [V]: ON " if voice_on   else "VOICE [V]: OFF"
    cv2.putText(frame, g_txt, (w-540, 22), FONT, 0.52, g_col, 1)
    cv2.putText(frame, v_txt, (w-540, 44), FONT, 0.52, v_col, 1)

    # ── LEFT PANEL – gesture guide ───────────────────────────────────────────
    cv2.putText(frame, "GESTURE GUIDE", (8, 76), FONT_BOLD, 0.5, CLR_ACCENT, 1)
    cv2.line(frame, (8, 82), (252, 82), CLR_ACCENT, 1)
    for i, (gesture, action) in enumerate(GESTURE_GUIDE):
        y = 100 + i * 46
        cv2.putText(frame, gesture,  (8,  y),    FONT, 0.42, CLR_WHITE,  1)
        cv2.putText(frame, f"→ {action}", (8, y+17), FONT, 0.38, CLR_GREEN, 1)

    # ── RIGHT PANEL – voice command list ─────────────────────────────────────
    cv2.putText(frame, "VOICE COMMANDS", (w-215, 76), FONT_BOLD, 0.45, CLR_VOICE, 1)
    cv2.line(frame, (w-215, 82), (w-4, 82), CLR_VOICE, 1)
    for i, cmd in enumerate(VOICE_GUIDE):
        y = 100 + i * 30
        cv2.putText(frame, cmd, (w-212, y), FONT, 0.38, CLR_WHITE, 1)

    # ── BOTTOM STATUS BAR ───────────────────────────────────────────────────
    # Current gesture
    cv2.putText(frame, "GESTURE:", (8, h-35), FONT, 0.5, CLR_DIM, 1)
    cv2.putText(frame, gesture_label, (95, h-35), FONT_BOLD, 0.55, CLR_ACCENT, 1)

    # Divider
    cv2.line(frame, (w//2, h-58), (w//2, h-4), CLR_DIM, 1)

    # Voice status
    mic_color  = CLR_GREEN if voice_listening else CLR_DIM
    mic_symbol = "🎙 LISTENING..." if voice_listening else "🎙 Waiting..."
    cv2.putText(frame, "VOICE:", (w//2 + 8, h-35), FONT, 0.5, CLR_DIM, 1)
    cv2.putText(frame, voice_label or "—", (w//2 + 72, h-35), FONT_BOLD, 0.52, CLR_VOICE, 1)

    # Mic pulse indicator
    if voice_listening:
        pulse = int(abs(np.sin(frame_count * 0.15)) * 14) + 6
        cv2.circle(frame, (w-10, h-30), pulse, CLR_GREEN, -1)
    else:
        cv2.circle(frame, (w-10, h-30), 5, CLR_DIM, -1)

    # Quit hint
    cv2.putText(frame, "Q: Quit", (8, h-12), FONT, 0.38, CLR_DIM, 1)
    cv2.putText(frame, "V: Toggle Voice", (90, h-12), FONT, 0.38, CLR_DIM, 1)
    cv2.putText(frame, "G: Toggle Gesture", (230, h-12), FONT, 0.38, CLR_DIM, 1)

    return frame


def draw_active_zone(frame):
    """Draw the 'active' camera region where gesture tracking works."""
    h, w = frame.shape[:2]
    # Exclude left and right panels
    x1, y1 = 264, 60
    x2, y2 = w-224, h-64
    cv2.rectangle(frame, (x1, y1), (x2, y2), CLR_ACCENT, 1)
    cv2.putText(frame, "ACTIVE ZONE", (x1+4, y1+18), FONT, 0.45, CLR_ACCENT, 1)


def main():
    print("=" * 60)
    print("  HAND GESTURE + VOICE COMPUTER CONTROLLER")
    print("=" * 60)
    print("  Starting camera...")

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, FPS_CAP)

    if not cap.isOpened():
        print("  ERROR: Cannot open webcam. Check camera index.")
        sys.exit(1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Camera: {actual_w}x{actual_h}")

    gesture_ctrl = GestureController(actual_w, actual_h)
    print("  Gesture controller ready.")

    # Shared state for voice callbacks
    voice_state = {
        "last_cmd"  : "—",
        "listening" : False,
        "log"       : [],
    }
    voice_lock = threading.Lock()

    def on_cmd(cmd):
        with voice_lock:
            voice_state["last_cmd"] = cmd
            voice_state["log"].insert(0, f"▶ {cmd}")
            if len(voice_state["log"]) > 5:
                voice_state["log"].pop()

    def on_listen(state):
        with voice_lock:
            voice_state["listening"] = state

    voice_ctrl = VoiceController(on_command=on_cmd, on_listening=on_listen)
    voice_ctrl.start()
    print("  Voice controller started (listening in background).")
    print("  Press Q in window to quit, V to toggle voice, G to toggle gesture.")
    print("=" * 60)

    gesture_on  = True
    voice_on    = True
    frame_count = 0
    fps         = 0
    t_prev      = time.time()

    cv2.namedWindow("Gesture + Voice Controller", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Gesture + Voice Controller", actual_w, actual_h)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("  Camera read failed – retrying...")
            time.sleep(0.05)
            continue

        # Mirror for natural feel
        frame = cv2.flip(frame, 1)
        frame_count += 1

        # FPS calc
        t_now = time.time()
        fps   = 0.9 * fps + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
        t_prev = t_now

        # Gesture processing
        gesture_label = "— (paused)"
        if gesture_on:
            frame, gesture_label = gesture_ctrl.process(frame)

        # Voice state read
        with voice_lock:
            voice_label     = voice_state["last_cmd"]
            voice_listening = voice_state["listening"] and voice_on

        # Toggle voice enable
        voice_ctrl.enabled = voice_on

        # Draw HUD
        draw_active_zone(frame)
        frame = draw_hud(
            frame,
            gesture_label,
            voice_label,
            voice_listening,
            gesture_on,
            voice_on,
            fps,
            frame_count,
        )

        cv2.imshow("Gesture + Voice Controller", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:
            break
        elif key == ord('v') or key == ord('V'):
            voice_on = not voice_on
            print(f"  Voice: {'ON' if voice_on else 'OFF'}")
        elif key == ord('g') or key == ord('G'):
            gesture_on = not gesture_on
            print(f"  Gesture: {'ON' if gesture_on else 'OFF'}")

    print("\n  Shutting down...")
    gesture_ctrl.release()
    voice_ctrl.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("  Done. Goodbye!")


if __name__ == "__main__":
    main()
