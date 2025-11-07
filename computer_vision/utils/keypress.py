import win32con, win32api
import time


class KeyPress:
    KEY_MAP = {
        "backspace": win32con.VK_BACK,
        "tab": win32con.VK_TAB,
        "enter": win32con.VK_RETURN,
        "shift": win32con.VK_SHIFT,
        "ctrl": win32con.VK_CONTROL,
        "alt": win32con.VK_MENU,
        "pause": win32con.VK_PAUSE,
        "capslock": win32con.VK_CAPITAL,
        "escape": win32con.VK_ESCAPE,
        "space": win32con.VK_SPACE,
        "pageup": win32con.VK_PRIOR,
        "pagedown": win32con.VK_NEXT,
        "end": win32con.VK_END,
        "home": win32con.VK_HOME,
        "left": win32con.VK_LEFT,
        "up": win32con.VK_UP,
        "right": win32con.VK_RIGHT,
        "down": win32con.VK_DOWN,
        "insert": win32con.VK_INSERT,
        "delete": win32con.VK_DELETE,
        
        "0": 0x30,  # '0' key
        "1": 0x31,  # '1' key
        "2": 0x32,  # '2' key
        "3": 0x33,  # '3' key
        "4": 0x34,  # '4' key
        "5": 0x35,  # '5' key
        "6": 0x36,  # '6' key
        "7": 0x37,  # '7' key
        "8": 0x38,  # '8' key
        "9": 0x39,  # '9' key

        "a": 0x41,  # 'A' key
        "b": 0x42,  # 'B' key
        "c": 0x43,  # 'C' key
        "d": 0x44,  # 'D' key
        "e": 0x45,  # 'E' key
        "f": 0x46,  # 'F' key
        "g": 0x47,  # 'G' key
        "h": 0x48,  # 'H' key
        "i": 0x49,  # 'I' key
        "j": 0x4A,  # 'J' key
        "k": 0x4B,  # 'K' key
        "l": 0x4C,  # 'L' key
        "m": 0x4D,  # 'M' key
        "n": 0x4E,  # 'N' key
        "o": 0x4F,  # 'O' key
        "p": 0x50,  # 'P' key
        "q": 0x51,  # 'Q' key
        "r": 0x52,  # 'R' key
        "s": 0x53,  # 'S' key
        "t": 0x54,  # 'T' key
        "u": 0x55,  # 'U' key
        "v": 0x56,  # 'V' key
        "w": 0x57,  # 'W' key
        "x": 0x58,  # 'X' key
        "y": 0x59,  # 'Y' key
        "z": 0x5A,  # 'Z' key

        "numpad0": win32con.VK_NUMPAD0,
        "numpad1": win32con.VK_NUMPAD1,
        "numpad2": win32con.VK_NUMPAD2,
        "numpad3": win32con.VK_NUMPAD3,
        "numpad4": win32con.VK_NUMPAD4,
        "numpad5": win32con.VK_NUMPAD5,
        "numpad6": win32con.VK_NUMPAD6,
        "numpad7": win32con.VK_NUMPAD7,
        "numpad8": win32con.VK_NUMPAD8,
        "numpad9": win32con.VK_NUMPAD9,
        "*": win32con.VK_MULTIPLY,
        "+": win32con.VK_ADD,
        "-": win32con.VK_SUBTRACT,
        ".": win32con.VK_DECIMAL,
        "/": win32con.VK_DIVIDE,

        "f1": win32con.VK_F1,
        "f2": win32con.VK_F2,
        "f3": win32con.VK_F3,
        "f4": win32con.VK_F4,
        "f5": win32con.VK_F5,
        "f6": win32con.VK_F6,
        "f7": win32con.VK_F7,
        "f8": win32con.VK_F8,
        "f9": win32con.VK_F9,
        "f10": win32con.VK_F10,
        "f11": win32con.VK_F11,
        "f12": win32con.VK_F12,

        "numlock": win32con.VK_NUMLOCK,
        "scrolllock": win32con.VK_SCROLL,
        
        ";": 0xBA,
        "=": 0xBB,
        ",": 0xBC,
        "-": 0xBD,
        ".": 0xBE,
        "/": 0xBF,
        "`": 0xC0,

        "[": 0xDB,  # '[' key
        "]": 0xDD, # ']' key
        "'": 0xDE         # "'" key
    }
    
    # DELAY_LIST = [
    #     "Auto Shot",
    #     "Shoot",
    #     "Pick Pocket",
    #     "Pick Pocket Backstab",
    # ]
    
    def __init__(self, hwnd, name):
        self.hwnd = hwnd
        self.name = name
    
    def use_keybind(self, keybind):
        """ 
        can add some logic here like delays etc
        separates from the raw windows api calls
        """
        self.post_hotkey(keybind)
    
    def post_hotkey(self, keybind) -> None:
        """ Send inputs with PostMessage to client window
        """
        key_parts = keybind.split('-')
        
        # Convert key names to virtual key codes
        key_codes = []
        for key in key_parts:
            key_lower = key.lower()
            if key_lower in self.KEY_MAP:
                key_codes.append(self.KEY_MAP[key_lower])
            else:
                key_codes.append(ord(key.upper()))
        # Press
        for key in key_codes:
            win32api.PostMessage(self.hwnd, win32con.WM_KEYDOWN, key, 0)
            time.sleep(0.01)
        
        # Release
        for key in reversed(key_codes):
            win32api.PostMessage(self.hwnd, win32con.WM_KEYUP, key, 0)
            time.sleep(0.01)

            
    