"""
main.py  -  QUARD: control your computer with your hand and your voice.

Run:   python main.py

How the program is put together
  main.py                the window loop: reads the camera, asks the two
                         controllers what happened, hands the result to the HUD
  gesture_controller.py  MediaPipe hand tracking  -> mouse / scroll / zoom
  voice_controller.py    microphone + speech recognition (background thread)
  ui.py                  everything that is drawn on screen

Keys (in the camera window)
  G      hand control on/off        V      voice control on/off
  SPACE  pause / resume everything  H      show / hide the help screen
  D      show live tuning numbers   Q      quit (Esc closes help, or quits)

The app starts on the help screen with everything paused, so nothing moves your
mouse until you press H or SPACE.
"""

import sys
import time

import cv2

from gesture_controller import GestureController
from voice_controller    import VoiceController
from ui import Hud, W, H, ACCENT, GREEN, RED, VOICE

# ─── CONFIG ─────────────────────────────────────────────────────────────────
CAM_INDEX = 0          # try 1, 2 ... if you have several cameras
CAM_W, CAM_H = 1280, 720
FPS_CAP   = 30
WINDOW    = "QUARD - Hand + Voice Control"

KEY_ESC, KEY_ENTER, KEY_SPACE = 27, 13, 32


def open_camera():
    """Open the webcam, using DirectShow on Windows (faster start-up, honours resolution)."""
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(CAM_INDEX, backend)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, FPS_CAP)
    return cap


def fit_frame(frame):
    """Centre-crop the camera image to 16:9 and scale it to the HUD size (W x H).

    The HUD layout is fixed, so every camera (640x480, 1080p ...) is made to fit it.
    """
    h, w = frame.shape[:2]
    if w * H > h * W:                       # too wide -> trim left/right
        new_w = h * W // H
        x0 = (w - new_w) // 2
        frame = frame[:, x0:x0 + new_w]
    elif w * H < h * W:                     # too tall -> trim top/bottom
        new_h = w * H // W
        y0 = (h - new_h) // 2
        frame = frame[y0:y0 + new_h]
    if frame.shape[1] != W or frame.shape[0] != H:
        frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
    return frame


def voice_status(voice_ctrl, active):
    """Translate the voice controller's state into (state, text) for the HUD."""
    if not active:
        return "off", "Voice is off"
    if voice_ctrl.error:
        return "error", "Voice problem (see below)"
    if voice_ctrl.listening:
        return "listening", "Listening - say a command"
    return "busy", "Understanding..."


def main():
    print("=" * 60)
    print("  QUARD - hand gesture + voice computer control")
    print("=" * 60)
    print("  Starting camera...")

    cap = open_camera()
    if cap is None:
        print("  ERROR: Cannot open the webcam. Close other apps using it, "
              "or change CAM_INDEX at the top of main.py.")
        sys.exit(1)

    gesture_ctrl = None
    voice_ctrl   = None
    try:
        gesture_ctrl = GestureController(W, H)          # works on the resized frame
        print("  Hand tracking ready.")

        hud = Hud()
        voice_ctrl = VoiceController(on_command=hud.voice_hit)
        voice_ctrl.start()
        print("  Voice control started (needs a microphone and internet).")
        print("  The window opens on the help screen - press H or SPACE to begin.")
        print("=" * 60)

        # ── state ──────────────────────────────────────────────────────────────
        gesture_on = True        # G toggle
        voice_on   = True        # V toggle
        paused     = False       # SPACE toggle
        help_open  = True        # shown at start; controls stay off until it is closed
        first_run  = True
        show_debug = False       # D toggle: live pinch distances in the bottom bar
        fps, t_prev, read_fails = 0.0, time.time(), 0
        was_gesture_active = False

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, W, H)

        while True:
            ok, frame = cap.read()
            if not ok:
                read_fails += 1
                if read_fails > 100:
                    print("  ERROR: the camera stopped delivering frames.")
                    break
                time.sleep(0.05)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    break
                continue
            read_fails = 0

            frame = fit_frame(cv2.flip(frame, 1))       # mirror: moving right moves the cursor right

            t_now = time.time()
            fps   = 0.9 * fps + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
            t_prev = t_now

            # What is actually allowed to act right now?
            gesture_active = gesture_on and not paused and not help_open
            voice_active   = voice_on   and not paused and not help_open
            voice_ctrl.enabled = voice_active

            # ── hands ────────────────────────────────────────────────────────────
            if gesture_active:
                frame, gesture_label = gesture_ctrl.process(frame)
                for event in gesture_ctrl.pop_events():
                    hud.toast(event, ACCENT)
            else:
                gesture_label = "Off"
                if was_gesture_active:
                    gesture_ctrl.reset()                # never leave the mouse button held down
            was_gesture_active = gesture_active

            # ── draw ─────────────────────────────────────────────────────────────
            v_state, v_text = voice_status(voice_ctrl, voice_active)
            hud.draw(
                frame,
                gesture_on=gesture_on, voice_on=voice_on, paused=paused or help_open, fps=fps,
                gesture_id=gesture_ctrl.gesture_id if gesture_active else "none",
                gesture_label=gesture_label,
                hand_count=gesture_ctrl.hand_count if gesture_active else 0,
                pointer=gesture_ctrl.pointer if gesture_active else None,
                voice_state=v_state,
                voice_text=voice_ctrl.error[:60] if v_state == "error" else v_text,
                heard=voice_ctrl.heard,
                debug=gesture_ctrl.debug_text if (show_debug and gesture_active) else "",
            )
            if help_open:
                hud.draw_help(frame, first_run)

            cv2.imshow(WINDOW, frame)

            # ── keys ─────────────────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            elif key == KEY_ESC:
                if help_open:
                    help_open = first_run = False
                else:
                    break
            elif key in (ord("h"), ord("H"), KEY_ENTER) or (key == KEY_SPACE and help_open):
                help_open = not help_open
                first_run = False
            elif key == KEY_SPACE:
                paused = not paused
                hud.toast("Paused" if paused else "Resumed", RED if paused else GREEN)
            elif key in (ord("v"), ord("V")):
                voice_on = not voice_on
                hud.toast("Voice " + ("ON" if voice_on else "OFF"), VOICE)
            elif key in (ord("g"), ord("G")):
                gesture_on = not gesture_on
                hud.toast("Hands " + ("ON" if gesture_on else "OFF"), ACCENT)
            elif key in (ord("d"), ord("D")):
                show_debug = not show_debug
                hud.toast("Tuning numbers " + ("ON" if show_debug else "OFF"), ACCENT)

            # Window closed with the title-bar X button
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        print("\n  Shutting down...")
        if gesture_ctrl is not None:
            gesture_ctrl.release()
        if voice_ctrl is not None:
            voice_ctrl.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("  Done. Goodbye!")


if __name__ == "__main__":
    main()
