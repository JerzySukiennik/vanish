"""Per-frame state machine: matte, plate, gestures and commands in, composited frame out."""

import math
import queue
import time as _time

import cv2
import numpy as np

from .effects import EFFECTS, GESTURE_TO_EFFECT
from .matte import Matte, analyze
from .renderer import MH, MW, H, Renderer, W

PRESENT, VANISH, GONE, RETURN, BLEND, SHRINK, GROW = "present", "vanish", "gone", "return", "blend", "shrink", "grow"


class Pipeline:
    PLATE_RATE = 0.06

    def __init__(self, gestures=None, shrink_minutes=10.0):
        self.matte = Matte()
        self.gest = gestures
        self.r = Renderer()
        self.r.activate()
        self.effects = {cls.key: cls(self.r) for cls in EFFECTS}
        self.order = [cls.key for cls in EFFECTS]
        self.state = PRESENT
        self.effect = None
        self.last_effect = "dust"
        self.t0 = 0.0
        self.cmds = queue.SimpleQueue()
        self.frame_i = 0
        self.masks = None
        self.plate_t = -1.0
        self.shrink_minutes = shrink_minutes
        self.scale = 1.0
        self.grow_from = 1.0
        self.gesture_status = (None, 0.0)
        self.gestures_on = True
        self.seen = 0.0
        self.seen_t = -10.0
        self.last_rgb = None
        self.last_out = None
        self.plate_lr = np.zeros((45, 80, 3), np.float32)
        self.seen_lr = np.zeros((45, 80), bool)
        self.gain = np.ones(3, np.float32)
        self.validate_in = 0

    def command(self, kind, key=None, **opts):
        self.cmds.put((kind, key, opts))

    def busy(self):
        return self.state in (VANISH, RETURN, BLEND, GROW)

    def step(self, rgb, now=None):
        now = _time.monotonic() if now is None else now
        if rgb.shape[0] != H or rgb.shape[1] != W:
            rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
        rgb = np.ascontiguousarray(rgb)
        self.frame_i += 1
        self.last_rgb = rgb
        small = cv2.resize(rgb, (MW, MH), interpolation=cv2.INTER_AREA)
        if self.state in (SHRINK, GROW) or self.masks is None:
            need_seg = True
        elif self.state == PRESENT:
            need_seg = self.frame_i % 3 == 0
        else:
            need_seg = self.frame_i % 2 == 0
        if need_seg:
            soft, tight, safe, _ = self.matte.process(small)
            self.masks = (soft, tight, safe)
        if self.gest is not None and self.gestures_on and self.frame_i % 3 == 1 and not self.busy():
            ev, self.gesture_status = self.gest.process(small, now)
            if ev is not None:
                name, info = ev
                key = GESTURE_TO_EFFECT.get(name)
                opts = {}
                if name == "thumb_side":
                    opts["dir"] = 1.0 if info.get("dx", 1.0) > 0 else -1.0
                elif "hand_x" in info:
                    opts["hand_x"] = info["hand_x"]
                self.command("trigger", key, **opts)
        self.r.activate()
        if need_seg:
            self._track_gain(small, now - self.plate_t > 0.2)
        if need_seg and now - self.plate_t > 0.2:
            self.r.upload_live(rgb)
            self.r.upload_masks(*self.masks)
            self.r.update_plate(self.PLATE_RATE)
            self.plate_t = now
            self.seen = float(self.seen_lr.mean())
        while True:
            try:
                kind, key, opts = self.cmds.get_nowait()
            except queue.Empty:
                break
            self._handle(kind, key, opts, rgb, now)
        if self.state == PRESENT:
            out = rgb
        else:
            out = self._render(rgb, now)
        self.last_out = out
        return out

    def _track_gain(self, small, learn):
        live = cv2.resize(small, (80, 45), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        hide = cv2.resize(self.masks[2], (80, 45), interpolation=cv2.INTER_AREA)
        vis = hide < 4
        valid = vis & self.seen_lr
        if valid.sum() > 60:
            ratio = live[valid] / np.maximum(self.plate_lr[valid], 0.03)
            g = np.clip(np.median(ratio, axis=0), 0.4, 2.5)
            self.gain = self.gain * 0.2 + g * 0.8
        self.r.gain = tuple(float(v) for v in self.gain)
        if not learn:
            return
        w = np.where(self.seen_lr, self.PLATE_RATE, 1.0) * vis
        self.plate_lr += (live / self.gain - self.plate_lr) * w[..., None]
        self.seen_lr |= vis
        if self.validate_in > 0:
            self.validate_in -= 1
            if self.validate_in == 0 and valid.sum() > 60:
                err = float(np.median(np.abs(live[valid] - self.plate_lr[valid] * self.gain)))
                if err > 0.08:
                    self._reset_plate()

    def _reset_plate(self):
        self.r.write_plate(np.zeros((H, W, 4), np.float32))
        self.plate_lr[:] = 0.0
        self.seen_lr[:] = False
        self.gain[:] = 1.0

    def save_state(self, path):
        self.r.activate()
        np.savez_compressed(path, plate=self.r.read_plate().astype(np.float16), plate_lr=self.plate_lr, seen_lr=self.seen_lr, gain=self.gain)

    def load_state(self, path):
        d = np.load(path)
        self.r.activate()
        self.r.write_plate(d["plate"].astype(np.float32))
        self.plate_lr = d["plate_lr"].astype(np.float32)
        self.seen_lr = d["seen_lr"].astype(bool)
        self.gain = d["gain"].astype(np.float32)
        self.r.gain = tuple(float(v) for v in self.gain)
        self.validate_in = 10

    def _info(self, opts):
        info = analyze(self.masks[0])
        if "dir" in opts:
            info["dir"] = opts["dir"]
        elif "hand_x" in opts:
            info["dir"] = 1.0 if info["center"][0] >= opts["hand_x"] else -1.0
        return info

    def _snapshot(self, rgb):
        soft, tight, safe = self.masks
        self.r.fill_plate_holes()
        self.r.set_snapshot(rgb, soft, safe)

    def _handle(self, kind, key, opts, rgb, now):
        if kind == "plate_reset":
            self._reset_plate()
            return
        if self.busy():
            return
        if kind == "return":
            key = None
        if self.state == PRESENT and kind == "trigger" and key:
            if key == "shrink":
                self.r.fill_plate_holes()
                self.effects["shrink"].begin(self._info(opts), self.masks[0])
                self.state, self.t0, self.scale = SHRINK, now, 1.0
                return
            self._snapshot(rgb)
            self.effect = self.effects[key]
            self.effect.begin(self._info(opts), self.masks[0])
            self.last_effect = key
            self.state, self.t0 = VANISH, now
            return
        if self.state == GONE and kind in ("trigger", "return"):
            key = key if key and key != "shrink" else self.last_effect
            self._snapshot(rgb)
            self.effect = self.effects[key]
            self.effect.begin(self._info(opts), self.masks[0])
            self.state, self.t0 = RETURN, now
            return
        if self.state == SHRINK and kind in ("trigger", "return"):
            self.grow_from = self.scale
            self.state, self.t0 = GROW, now

    def shrink_scale(self, elapsed):
        total = max(self.shrink_minutes * 60.0, 1.0)
        x = min(1.0, elapsed / total)
        return math.exp(math.log(self.effects["shrink"].S_MIN) * x ** 1.35)

    def _render(self, rgb, now):
        r = self.r
        soft, tight, safe = self.masks
        live_mip = self.state in (SHRINK, GROW)
        r.upload_live(rgb)
        r.upload_masks(soft, tight, safe, mip=live_mip)
        el = now - self.t0
        if self.state == VANISH:
            t = el / self.effect.duration
            if t >= 1.0:
                self.state = GONE
            else:
                self.effect.draw(t, el)
                return r.read()
        if self.state == GONE:
            r.draw_gone()
            return r.read()
        if self.state == RETURN:
            t = 1.0 - el / self.effect.duration
            if t <= 0.0:
                self.state, self.t0 = BLEND, now
                el = 0.0
            else:
                self.effect.draw(t, el)
                return r.read()
        if self.state == BLEND:
            k = el / 0.3
            if k >= 1.0:
                self.state = PRESENT
                return rgb
            r.draw_blend(k)
            return r.read()
        shrink = self.effects["shrink"]
        if self.state == SHRINK:
            self.scale = self.shrink_scale(el)
            safe_k = min(1.0, max(0.0, (1.0 - self.scale) / 0.02))
            shrink.draw_scale(self.scale, safe_k)
            return r.read()
        if self.state == GROW:
            k = el / shrink.duration
            if k >= 1.0:
                self.state, self.scale = PRESENT, 1.0
                return rgb
            x = k - 1.0
            e = x * x * (2.4 * x + 1.4) + 1.0
            self.scale = self.grow_from + (1.0 - self.grow_from) * e
            shrink.draw_scale(self.scale, 1.0)
            return r.read()
        return rgb
