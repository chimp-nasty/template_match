import ctypes as C
from ctypes import wintypes as W
from time import perf_counter, sleep

user32 = C.windll.user32
user32.GetAsyncKeyState.argtypes = [W.INT]
user32.GetAsyncKeyState.restype = W.SHORT

# Common key name -> VK code
_VK = {
    "ESC": 0x1B,
    "ESCAPE": 0x1B,
    "SPACE": 0x20,
    "P": 0x50,
    "F1": 0x70, "F2": 0x71, "F3": 0x72, "F4": 0x73,
    "F5": 0x74, "F6": 0x75, "F7": 0x76, "F8": 0x77,
    "F9": 0x78, "F10": 0x79, "F11": 0x7A, "F12": 0x7B,
}

def _to_vk(key):
    """Accept 'SPACE', 'ESC', 'P', 'F10' or an int VK code."""
    if isinstance(key, int):
        return key
    if not isinstance(key, str):
        raise ValueError("Hotkey must be str or int")
    key = key.strip().upper()
    if key in _VK:
        return _VK[key]
    if len(key) == 1 and "A" <= key <= "Z":
        return ord(key)
    if key.startswith("F"):
        try:
            n = int(key[1:])
            if 1 <= n <= 24:
                return 0x6F + n  # F1 = 0x70
        except ValueError:
            pass
    raise ValueError(f"Unknown hotkey: {key}")

class Controller:
    """
    Simple main-loop controller: pause/terminate via hotkeys.
    - pause_key toggles paused state (default: SPACE)
    - terminate_key sets terminated=True (default: ESC)
    No threads; call controller.poll() each loop.
    """
    def __init__(self, pause_key="SPACE", terminate_key="ESC"):
        self.vk_pause = _to_vk(pause_key)
        self.vk_term  = _to_vk(terminate_key)
        self.paused = False
        self.terminated = False
        self._down_prev = {}  # edge detection

    def _is_down(self, vk):
        # 0x8000 bit == key currently down
        return (user32.GetAsyncKeyState(vk) & 0x8000) != 0

    def _edge(self, vk):
        now = self._is_down(vk)
        was = self._down_prev.get(vk, False)
        self._down_prev[vk] = now
        return now and not was

    def poll(self):
        """Update paused/terminated states based on hotkeys."""
        if self._edge(self.vk_pause):
            self.paused = not self.paused
        if self._edge(self.vk_term):
            self.terminated = True

    # Optional convenience pacing (you can ignore this if you already pace elsewhere)
    def pace(self, next_t, dt):
        next_t += dt
        remaining = next_t - perf_counter()
        if remaining > 0:
            sleep(remaining)
        else:
            next_t = perf_counter()
        return next_t
