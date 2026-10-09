# 🖐️ Hand Gesture + Voice Command Controller

A Python prototype that lets you **control your computer entirely hands-free** using a webcam and microphone.

---

## 📦 Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
python main.py
```

`hand_landmarker.task` (the MediaPipe model) is included in the repo. If it is
missing, download it from
<https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task>
and place it next to `main.py`.

---

## ✋ Gesture Controls

| Gesture | Action |
|---|---|
| ✋ Just move your hand (any hand shape) | **Move cursor** |
| 👌 Tap thumb to index finger | **Left click** (tap twice quickly = **double click**) |
| 👌 Same pinch, but hold it (about ⅓ s) and move | **Drag** – release to drop |
| 🤏 Tap thumb to middle finger | **Right click** |
| 🤏 Tap thumb to ring finger | **Double click** – opens files and folders |
| ✊ Make a fist, move it up/down | **Scroll** |
| 🤲 Two hands moving apart/together | **Zoom in/out** |

Keep your hand inside the amber box; a hand **outside the box does nothing**, so
that is how you rest. A green line appears between thumb and finger when a pinch
registers.

**Opening a file or folder** = point the cursor at it and double-click: tap your
thumb to your **ring finger** (most reliable), tap thumb+index twice quickly, or
say **"open"**. Windows only counts two clicks as a double click if the cursor does
not move at all between them, so after a tap QUARD holds the cursor still for
about half a second (`TAP_LOCK_S`); moving your hand on purpose releases it at once.

Detection uses MediaPipe's 3D hand coordinates, so it works at any hand angle and
distance. Press `D` in the window to see live "how close is this pinch" numbers;
if a gesture is too hard or too easy to trigger, adjust `PINCH_ON`, `PINCH_OFF`,
`FIST_STRAIGHT` and `DRAG_HOLD_S` at the top of `gesture_controller.py`.

---

## 🎙️ Voice Commands

| Say | Action |
|---|---|
| "click" / "left click" | Left mouse click |
| "right click" | Right mouse click |
| "double click" / "open" | Double click – opens the file or folder under the cursor |
| "scroll up" / "scroll down" | Scroll |
| "zoom in" / "zoom out" | Ctrl+/Ctrl- |
| "screenshot" | Save screenshot |
| "minimize" / "maximize" | Window control |
| "close" / "close window" | Alt+F4 |
| "volume up" / "volume down" | Volume control |
| "mute" | Toggle mute |
| "copy" / "paste" | Ctrl+C / Ctrl+V |
| "undo" | Ctrl+Z |
| "select all" | Ctrl+A |
| "new tab" / "close tab" | Browser tabs |
| "go back" / "go forward" | Navigation |
| "switch window" | Alt+Tab |
| "task manager" | Ctrl+Shift+Esc |
| "press enter/escape/space" | Key presses |

---

## ⌨️ Keyboard Shortcuts (In Window)

The app opens on a **help screen with everything paused**, so nothing moves your
mouse until you press `H` or `Space`.

| Key | Action |
|---|---|
| `H` / `Enter` | Show / hide the help screen |
| `Space` | Pause / resume everything (also starts the app from the help screen) |
| `G` | Toggle hand control ON/OFF |
| `V` | Toggle voice control ON/OFF |
| `Q` | Quit (`Esc` closes help, or quits) |

### Reading the window

- **Left panel** – every hand gesture; the one you are making lights up.
- **Right panel** – every voice command; the one it just heard lights up.
- **Centre** – your camera. Keep your hand inside the amber corner box; a ring marks the fingertip that moves the cursor.
- **Pop-ups** at the bottom of the box confirm each action (e.g. "Left click").
- **Bottom bar** – the current gesture, the microphone status and the last phrase heard.
- `ui.py` holds all the drawing code if you want to change colours or layout.

---

## 🛠️ Requirements

- Python 3.9+ (tested on 3.13)
- Webcam
- Microphone + internet (for Google Speech Recognition). `PyAudio` is installed by `requirements.txt`
- Windows (tested), should also work on Linux/macOS

---

## 💡 Tips

- Keep your hand **within the blue active zone** on the camera feed
- Good lighting improves gesture detection accuracy
- Speak clearly and naturally for voice commands
- Use **Fist** gesture to temporarily pause cursor if you need to scratch your nose 😄
- For Zoom, hold both hands in front of the camera and move them apart (zoom in) or together (zoom out)
