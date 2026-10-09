# 🖐️ Hand Gesture + Voice Command Controller

A Python prototype that lets you **control your computer entirely hands-free** using a webcam and microphone.

---

## 📦 Setup

```bash
pip install -r requirements.txt
python main.py
```

---

## ✋ Gesture Controls

| Gesture | Action |
|---|---|
| ☝️ Index finger only | **Move cursor** |
| 👌 Pinch (thumb + index) | **Left click** |
| ✌️ Two fingers touching | **Double click** |
| ✌️ Two fingers sliding up/down | **Scroll** |
| 🤟 Thumb + middle finger | **Right click** |
| ✊ Fist | **Pause cursor control** |
| 🖐 Open palm (all 5 fingers) | **Drag** |
| 🤲 Two hands moving apart/together | **Zoom in/out** |

---

## 🎙️ Voice Commands

| Say | Action |
|---|---|
| "click" / "left click" | Left mouse click |
| "right click" | Right mouse click |
| "double click" | Double click |
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

| Key | Action |
|---|---|
| `Q` or `Esc` | Quit |
| `V` | Toggle voice control ON/OFF |
| `G` | Toggle gesture control ON/OFF |

---

## 🛠️ Requirements

- Python 3.9+
- Webcam
- Microphone + internet (for Google Speech Recognition)
- Windows (tested), should also work on Linux/macOS

---

## 💡 Tips

- Keep your hand **within the blue active zone** on the camera feed
- Good lighting improves gesture detection accuracy
- Speak clearly and naturally for voice commands
- Use **Fist** gesture to temporarily pause cursor if you need to scratch your nose 😄
- For Zoom, hold both hands in front of the camera and move them apart (zoom in) or together (zoom out)
