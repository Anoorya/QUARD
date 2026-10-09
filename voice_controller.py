"""
voice_controller.py
Background thread that listens for voice commands and fires OS-level actions.
"""

import threading
import time
import pyautogui
import speech_recognition as sr
import pyttsx3

# Mapping of spoken phrases to actions
COMMANDS = {
    "click"        : lambda: pyautogui.click(),
    "left click"   : lambda: pyautogui.click(),
    "right click"  : lambda: pyautogui.rightClick(),
    "double click" : lambda: pyautogui.doubleClick(),
    "scroll up"    : lambda: pyautogui.scroll(300),
    "scroll down"  : lambda: pyautogui.scroll(-300),
    "zoom in"      : lambda: pyautogui.hotkey("ctrl", "+"),
    "zoom out"     : lambda: pyautogui.hotkey("ctrl", "-"),
    "screenshot"   : lambda: pyautogui.screenshot("screenshot.png"),
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

        self._tts = pyttsx3.init()
        self._tts.setProperty("rate", 170)
        self._tts.setProperty("volume", 0.85)

        self._running       = False
        self._thread        = None
        self.last_command   = "—"
        self.listening      = False
        self.enabled        = True

    # ── public API ────────────────────────────────────────────────────────
    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def speak(self, text: str):
        """Non-blocking TTS."""
        threading.Thread(target=self._say, args=(text,), daemon=True).start()

    # ── internals ─────────────────────────────────────────────────────────
    def _say(self, text):
        try:
            self._tts.say(text)
            self._tts.runAndWait()
        except Exception:
            pass

    def _loop(self):
        with sr.Microphone() as mic:
            self._recognizer.adjust_for_ambient_noise(mic, duration=1)
            while self._running:
                if not self.enabled:
                    time.sleep(0.2)
                    continue
                try:
                    self.listening = True
                    self._on_listening(True)
                    audio = self._recognizer.listen(mic, timeout=3, phrase_time_limit=4)
                    self.listening = False
                    self._on_listening(False)
                    text = self._recognizer.recognize_google(audio).lower().strip()
                    self._match(text)
                except sr.WaitTimeoutError:
                    self.listening = False
                    self._on_listening(False)
                except sr.UnknownValueError:
                    self.listening = False
                    self._on_listening(False)
                except sr.RequestError as e:
                    self.last_command = f"[API Error: {e}]"
                    self.listening = False
                    self._on_listening(False)
                    time.sleep(2)
                except Exception as e:
                    self.listening = False
                    time.sleep(0.5)

    def _match(self, text: str):
        """Find best matching command from spoken text."""
        # Longest match wins
        matched = None
        for phrase in sorted(COMMANDS.keys(), key=len, reverse=True):
            if phrase in text:
                matched = phrase
                break

        if matched:
            self.last_command = matched
            self._on_command(matched)
            try:
                COMMANDS[matched]()
            except Exception as e:
                self.last_command = f"[Error: {e}]"
        else:
            self.last_command = f'["{text}" – no match]'
