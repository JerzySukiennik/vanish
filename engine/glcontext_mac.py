"""Off-screen OpenGL context on the integrated GPU, so the discrete Radeon can stay asleep."""

import ctypes

_gl = ctypes.CDLL("/System/Library/Frameworks/OpenGL.framework/OpenGL")
_gl.CGLChoosePixelFormat.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int)]
_gl.CGLCreateContext.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
_gl.CGLSetCurrentContext.argtypes = [ctypes.c_void_p]
_gl.CGLSetVirtualScreen.argtypes = [ctypes.c_void_p, ctypes.c_int]
_gl.glGetString.argtypes = [ctypes.c_uint]
_gl.glGetString.restype = ctypes.c_char_p

ALLOW_OFFLINE = 96
ACCELERATED = 73
PROFILE = 99
CORE_32 = 0x3200
AUTO_SWITCH = 101
GL_RENDERER = 0x1F01


class CGLContext:
    def __init__(self, prefer="intel"):
        attrs = (ctypes.c_int * 7)(PROFILE, CORE_32, ALLOW_OFFLINE, ACCELERATED, AUTO_SWITCH, 0, 0)
        pix = ctypes.c_void_p()
        n = ctypes.c_int()
        err = _gl.CGLChoosePixelFormat(attrs, ctypes.byref(pix), ctypes.byref(n))
        if err or not pix.value:
            raise RuntimeError(f"CGLChoosePixelFormat failed: {err}")
        ctx = ctypes.c_void_p()
        err = _gl.CGLCreateContext(pix, None, ctypes.byref(ctx))
        if err or not ctx.value:
            raise RuntimeError(f"CGLCreateContext failed: {err}")
        self.handle = ctx
        self.make_current()
        names = []
        for vs in range(n.value):
            _gl.CGLSetVirtualScreen(ctx, vs)
            names.append((_gl.glGetString(GL_RENDERER) or b"").decode())
        pick = next((i for i, name in enumerate(names) if prefer in name.lower()), 0)
        _gl.CGLSetVirtualScreen(ctx, pick)
        self.renderer = names[pick] if names else "?"
        self.renderers = names

    def make_current(self):
        _gl.CGLSetCurrentContext(self.handle)
