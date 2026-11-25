"""System-wide hotkeys through Carbon RegisterEventHotKey (no Accessibility permission needed)."""

import ctypes

_carbon = ctypes.CDLL("/System/Library/Frameworks/Carbon.framework/Carbon")


class _EventTypeSpec(ctypes.Structure):
    _fields_ = [("eventClass", ctypes.c_uint32), ("eventKind", ctypes.c_uint32)]


class _HotKeyID(ctypes.Structure):
    _fields_ = [("signature", ctypes.c_uint32), ("id", ctypes.c_uint32)]


_HANDLER = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)

_carbon.GetEventDispatcherTarget.restype = ctypes.c_void_p
_carbon.InstallEventHandler.argtypes = [ctypes.c_void_p, _HANDLER, ctypes.c_ulong, ctypes.POINTER(_EventTypeSpec), ctypes.c_void_p, ctypes.c_void_p]
_carbon.InstallEventHandler.restype = ctypes.c_int32
_carbon.RegisterEventHotKey.argtypes = [ctypes.c_uint32, ctypes.c_uint32, _HotKeyID, ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)]
_carbon.RegisterEventHotKey.restype = ctypes.c_int32
_carbon.UnregisterEventHotKey.argtypes = [ctypes.c_void_p]
_carbon.GetEventParameter.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_void_p, ctypes.c_void_p]
_carbon.GetEventParameter.restype = ctypes.c_int32


def _fourcc(s):
    return int.from_bytes(s.encode("ascii"), "big")


CONTROL = 0x1000
OPTION = 0x0800
COMMAND = 0x0100
SHIFT = 0x0200
KEYCODES = {"1": 0x12, "2": 0x13, "3": 0x14, "4": 0x15, "5": 0x17, "6": 0x16, "7": 0x1A, "8": 0x1C, "9": 0x19, "0": 0x1D}


class Hotkeys:
    def __init__(self):
        self.callbacks = {}
        self.refs = []
        self._handler = _HANDLER(self._on_event)
        spec = _EventTypeSpec(_fourcc("keyb"), 5)
        self._spec = spec
        _carbon.InstallEventHandler(_carbon.GetEventDispatcherTarget(), self._handler, 1, ctypes.byref(spec), None, None)
        self._next = 1

    def register(self, keycode, modifiers, callback):
        hid = _HotKeyID(_fourcc("VNSH"), self._next)
        ref = ctypes.c_void_p()
        err = _carbon.RegisterEventHotKey(keycode, modifiers, hid, _carbon.GetEventDispatcherTarget(), 0, ctypes.byref(ref))
        if err == 0:
            self.callbacks[self._next] = callback
            self.refs.append(ref)
        self._next += 1
        return err == 0

    def unregister_all(self):
        for ref in self.refs:
            _carbon.UnregisterEventHotKey(ref)
        self.refs.clear()
        self.callbacks.clear()

    def _on_event(self, _call, event, _data):
        hid = _HotKeyID()
        _carbon.GetEventParameter(event, _fourcc("----"), _fourcc("hkid"), None, ctypes.sizeof(hid), None, ctypes.byref(hid))
        cb = self.callbacks.get(hid.id)
        if cb is not None:
            try:
                cb()
            except Exception:
                pass
        return 0
