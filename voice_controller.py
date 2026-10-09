"""
voice_controller.py
Background thread that listens for voice commands and fires OS-level actions.
"""

import os
import re
import threading
import time

import pyautogui
import speech_recognition as sr

SCREENSHOT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screenshot.png")

# Mapping of spoken phrases to actions
COMMANDS = {
    "click"        : lambda: pyautogui.click(),
    "left click"   : lambda: pyautogui.click(),
    "right click"  : lambda: pyautogui.rightClick(),
    "double click" : lambda: pyautogui.doubleClick(),
    "open"         : lambda: pyautogui.doubleClick(),     # opens the file / folder under the cursor
    "scroll up"    : lambda: pyautogui.scroll(300),
    "scroll down"  : lambda: pyautogui.scroll(-300),
    "zoom in"      : lambda: pyautogui.hotkey("ctrl", "+"),
    "zoom out"     : lambda: pyautogui.hotkey("ctrl", "-"),
    "screenshot"   : lambda: pyautogui.screenshot(SCREENSHOT_PATH),
    "minimize"     : lambda: pyautogui.hotkey("win", "down"),
    "maximize"     : lambda: pyautogui.hotkey("win", "up"),
    "close"        : lambda: pyautogui.hotkey("alt", "f4"),
    "close window" : lambda: pyautogui.hotkey("alt", "f4"),
    "volume up"    : lambda: [pyautogui.press("volumeup") for _ in range(3)],
    "volume down"  : lambda: [pyautogui.press("volumedown") for _ in range(3)],
    "mute"         : lambda: pyautogui.press("volumemute"),
    "copy"         : lambda: pyautogui.hotkey("ctrl", "c"),
    "paste"        : lambda: pyautogui.hotkey("ctrl", "v"),
    "undo"         : lambda: pyautogui.hotkey("ctrl", "z"),
    "select all"   : lambda: pyautogui.hotkey("ctrl", "a"),
    "new tab"      : lambda: pyautogui.hotkey("ctrl", "t"),
    "close tab"    : lambda: pyautogui.hotkey("ctrl", "w"),
    "go back"      : lambda: pyautogui.hotkey("alt", "left"),
    "go forward"   : lambda: pyautogui.hotkey("alt", "right"),
    "switch window": lambda: pyautogui.hotkey("alt", "tab"),
    "task manager" : lambda: pyautogui.hotkey("ctrl", "shift", "esc"),
    "press enter"  : lambda: pyautogui.press("enter"),
    "press escape" : lambda: pyautogui.press("escape"),
    "press space"  : lambda: pyautogui.press("space"),
}

# Longest phrase first so "close tab" wins over "close"; whole-word match so
# e.g. "disclose" never triggers "close".
_PATTERNS = [
    (phrase, re.compile(r"\b" + re.escape(phrase) + r"\b"))
    for phrase in sorted(COMMANDS, key=len, reverse=True)
]


class VoiceController:
    """Listens on a background thread and executes matched voice commands."""

    def __init__(self, on_command=None, on_listening=None):
        """
        on_command   : callback(cmd_text: str) called when a command fires
        on_listening : callback(state: bool) called on listen state change
        """
        self._on_command   = on_command   or (lambda cmd: None)
        self._on_listening = on_listening or (lambda s: None)

        self._recognizer  = sr.Recognizer()
        self._recognizer.energy_threshold        = 300
        self._recognizer.dynamic_energy_threshold = True
        self._recognizer.pause_threshold          = 0.6

        self._tts      = None          # created lazily; TTS is optional
        self._tts_lock = threading.Lock()

        self._running       = False
        self._thread        = None
        self.last_command   = "-"
        self.heard          = ""       # last phrase the recogniser understood (matched or not)
        self.error          = ""       # current problem (no mic / no internet ...), "" when healthy
        self.listening      = False
        self.enabled        = True

    # ── public API ────────────────────────────────────────────────────────
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def speak(self, text: str):
        """Non-blocking TTS (silently skipped if no speech engine is available)."""
        threading.Thread(target=self._say, args=(text,), daemon=True).start()

    # ── internals ─────────────────────────────────────────────────────────
    def _say(self, text):
        with self._tts_lock:
            try:
                if self._tts is None:
                    import pyttsx3
                    self._tts = pyttsx3.init()
                    self._tts.setProperty("rate", 170)
                    self._tts.setProperty("volume", 0.85)
                self._tts.say(text)
                self._tts.runAndWait()
            except Exception:
                pass

    def _set_listening(self, state: bool):
        self.listening = state
        self._on_listening(state)

    def _loop(self):
        try:
            mic = sr.Microphone()
        except Exception as e:      # missing PyAudio / no input device
            self.error = f"Microphone unavailable: {e}"
            return

        try:
            with mic as source:
                self._recognizer.adjust_for_ambient_noise(source, duration=1)
                while self._running:
                    if not self.enabled:
                        if self.listening:
                            self._set_listening(False)
                        time.sleep(0.2)
                        continue
                    try:
                        self._set_listening(True)
                        audio = self._recognizer.listen(source, timeout=3, phrase_time_limit=4)
                        self._set_listening(False)
                        text = self._recognizer.recognize_google(audio).lower().strip()
                        self._match(text)
                    except (sr.WaitTimeoutError, sr.UnknownValueError):
                        self._set_listening(False)
                    except sr.RequestError as e:
                        self.error = f"Speech service unreachable (internet?): {e}"
                        self._set_listening(False)
                        time.sleep(2)
                    except Exception as e:
                        self.error = f"Voice error: {e}"
                        self._set_listening(False)
                        time.sleep(0.5)
        except Exception as e:
            self.error = f"Microphone error: {e}"
        finally:
            self._set_listening(False)

    def _match(self, text: str):
        """Find best matching command from spoken text."""
        self.heard = text
        self.error = ""            # a successful recognition means the mic and service work
        matched = next((p for p, rx in _PATTERNS if rx.search(text)), None)

        if matched:
            self.last_command = matched
            self._on_command(matched)
            try:
                COMMANDS[matched]()
            except Exception as e:
                self.last_command = f"[Error: {e}]"
        else:
            self.last_command = f'["{text}" - no match]'
