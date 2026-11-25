"""GPU compositor: holds camera, mask, snapshot and background-plate textures and renders effects off-screen."""

import cv2
import moderngl
import numpy as np

from . import glsl
from .glcontext_mac import CGLContext

W, H = 1280, 720
MW, MH = 640, 360

SLOTS = ["u_live", "u_plate", "u_hide", "u_hideTight", "u_liveMask", "u_snap", "u_snapMask", "u_snapHide", "u_plateFill"]
FW, FH = 320, 180


class Renderer:
    def __init__(self):
        try:
            self.cgl = CGLContext("intel")
            self.ctx = moderngl.create_context(require=330)
        except Exception:
            self.cgl = None
            self.ctx = moderngl.create_standalone_context(require=330)
        ctx = self.ctx
        self.gpu = ctx.info.get("GL_RENDERER", "?")
        self.live = self._tex((W, H), 3, mip=True)
        self.snap = self._tex((W, H), 3, mip=True)
        self.hide = self._tex((MW, MH), 1)
        self.hide_tight = self._tex((MW, MH), 1)
        self.live_mask = self._tex((MW, MH), 1, mip=True)
        self.snap_mask = self._tex((MW, MH), 1, mip=True)
        self.snap_hide = self._tex((MW, MH), 1, mip=True)
        self.plates = [self._tex((W, H), 4, mip=True, dtype="f2") for _ in range(2)]
        for p in self.plates:
            p.write(np.zeros((H, W, 4), np.float16))
            p.build_mipmaps()
        self.gain = (1.0, 1.0, 1.0)
        self.fill = self._tex((FW, FH), 3, dtype="f2")
        self.fill.write(np.zeros((FH, FW, 3), np.float16))
        self.plate_fbos = [ctx.framebuffer([p]) for p in self.plates]
        self.plate_i = 0
        self.out = ctx.texture((W, H), 4)
        self.fbo = ctx.framebuffer([self.out], ctx.depth_renderbuffer((W, H)))
        quad = np.array([0, 0, 1, 0, 0, 1, 1, 1], "f4")
        self.quad = ctx.buffer(quad.tobytes())
        self._progs = {}
        self.gone_prog = self.program(glsl.GONE_FS)
        self.blend_prog = self.program(glsl.BLEND_FS)
        self.plate_prog = ctx.program(vertex_shader=glsl.FULLSCREEN_VS, fragment_shader=glsl.PLATE_FS)
        self.plate_vao = ctx.vertex_array(self.plate_prog, [(self.quad, "2f", "in_pos")])
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    def _tex(self, size, comps, mip=False, dtype="f1"):
        t = self.ctx.texture(size, comps, dtype=dtype)
        t.repeat_x = False
        t.repeat_y = False
        t.filter = (moderngl.LINEAR_MIPMAP_LINEAR if mip else moderngl.LINEAR, moderngl.LINEAR)
        if mip:
            t.build_mipmaps()
        return t

    def activate(self):
        if self.cgl is not None:
            self.cgl.make_current()
        else:
            self.ctx.__enter__()

    @property
    def plate(self):
        return self.plates[self.plate_i]

    def program(self, fs, vs=None):
        prog = self.ctx.program(vertex_shader=vs or glsl.FULLSCREEN_VS, fragment_shader=fs)
        vao = self.ctx.vertex_array(prog, [(self.quad, "2f", "in_pos")], skip_errors=True) if vs is None else None
        self._progs[id(prog)] = vao
        return prog

    def bind(self, prog, t=0.0, time=0.0, dur=1.0, **uniforms):
        textures = [self.live, self.plate, self.hide, self.hide_tight, self.live_mask, self.snap, self.snap_mask, self.snap_hide, self.fill]
        for unit, (name, tex) in enumerate(zip(SLOTS, textures)):
            tex.use(location=unit)
            u = prog.get(name, None)
            if u is not None:
                u.value = unit
        for name, value in (("u_res", (W, H)), ("u_t", t), ("u_time", time), ("u_dur", dur), ("u_gain", self.gain)):
            u = prog.get(name, None)
            if u is not None:
                u.value = value
        for name, value in uniforms.items():
            u = prog.get(name, None)
            if u is not None:
                u.value = value

    def full(self, prog, **kw):
        self.bind(prog, **kw)
        self._progs[id(prog)].render(moderngl.TRIANGLE_STRIP)

    def begin_frame(self):
        self.fbo.use()
        self.ctx.disable(moderngl.BLEND | moderngl.DEPTH_TEST)
        self.fbo.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)

    def read(self):
        data = self.fbo.read(components=4, alignment=1)
        return cv2.cvtColor(np.frombuffer(data, np.uint8).reshape(H, W, 4), cv2.COLOR_RGBA2RGB)

    def upload_live(self, rgb, mip=False):
        self.live.write(rgb)
        if mip:
            self.live.build_mipmaps()

    def upload_masks(self, soft, tight, safe, mip=False):
        self.live_mask.write(soft)
        self.hide_tight.write(tight)
        self.hide.write(safe)
        if mip:
            self.live_mask.build_mipmaps()

    def set_snapshot(self, rgb, soft, safe):
        self.snap.write(rgb)
        self.snap.build_mipmaps()
        self.snap_mask.write(soft)
        self.snap_mask.build_mipmaps()
        self.snap_hide.write(safe)
        self.snap_hide.build_mipmaps()

    def update_plate(self, rate=0.25):
        src = self.plates[self.plate_i]
        dst_i = 1 - self.plate_i
        self.plate_fbos[dst_i].use()
        self.ctx.disable(moderngl.BLEND | moderngl.DEPTH_TEST)
        src.use(location=0)
        self.live.use(location=1)
        self.hide.use(location=2)
        self.plate_prog["u_prev"].value = 0
        self.plate_prog["u_live"].value = 1
        self.plate_prog["u_hide"].value = 2
        self.plate_prog["u_rate"].value = rate
        self.plate_prog["u_gain"].value = self.gain
        self.plate_vao.render(moderngl.TRIANGLE_STRIP)
        self.plate_i = dst_i
        self.plates[dst_i].build_mipmaps()

    def read_plate(self):
        data = self.plate_fbos[self.plate_i].read(components=4, alignment=1, dtype="f2")
        return np.frombuffer(data, np.float16).reshape(H, W, 4).astype(np.float32)

    def write_plate(self, rgba):
        self.plate.write(np.ascontiguousarray(rgba, np.float16))
        self.plate.build_mipmaps()

    def seen_fraction(self):
        a = self.read_plate()[::8, ::8, 3]
        return float((a > 0.5).mean())

    def fill_plate_holes(self):
        data = self.plate.read(level=2)
        rgba = np.frombuffer(data, np.float16).reshape(FH, FW, 4).astype(np.float32)
        seen = rgba[..., 3] > 0.5
        if seen.all() or not seen.any():
            return
        filled = pull_push(rgba[..., :3], seen.astype(np.float32))
        self.fill.write(np.ascontiguousarray(filled.astype(np.float16)))

    def draw_gone(self):
        self.begin_frame()
        self.full(self.gone_prog)

    def draw_blend(self, k):
        self.begin_frame()
        self.full(self.blend_prog, u_k=float(k))


def pull_push(img, valid):
    levels = []
    cur_img, cur_w = img * valid[..., None], valid.copy()
    while min(cur_w.shape) > 6:
        levels.append((cur_img, cur_w))
        h, w = cur_w.shape
        size = (max(w // 2, 1), max(h // 2, 1))
        cur_img = cv2.resize(cur_img, size, interpolation=cv2.INTER_AREA)
        cur_w = cv2.resize(cur_w, size, interpolation=cv2.INTER_AREA)
    coarse = cur_img / np.maximum(cur_w[..., None], 1e-4)
    for lvl_img, lvl_w in reversed(levels):
        h, w = lvl_w.shape
        up = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_LINEAR)
        up = cv2.GaussianBlur(up, (0, 0), 1.0)
        known = lvl_img / np.maximum(lvl_w[..., None], 1e-4)
        k = np.clip(lvl_w * 1.5, 0, 1)[..., None]
        coarse = known * k + up * (1 - k)
    return coarse
