"""Person matte: MediaPipe selfie segmentation refined with a guided filter, plus tight and safe hide masks."""

from collections import deque

import cv2
import mediapipe as mp
import numpy as np


class Matte:
    def __init__(self):
        self.seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
        self.prev = None
        self.hist = deque(maxlen=3)
        self.k_tight = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.k_safe = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))

    def reset(self):
        self.prev = None
        self.hist.clear()

    def process(self, small_rgb):
        raw = self.seg.process(small_rgb).segmentation_mask.astype(np.float32)
        guide = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        refined = fast_guided(guide, raw, 4, 4e-4)
        soft = np.clip((refined - 0.28) / 0.44, 0.0, 1.0)
        if self.prev is not None:
            soft = soft * 0.7 + self.prev * 0.3
        self.prev = soft
        core = (soft > 0.03).astype(np.uint8)
        tight = cv2.dilate(core, self.k_tight)
        tight = cv2.GaussianBlur(tight.astype(np.float32), (0, 0), 1.0)
        h, w = soft.shape
        half = cv2.resize(soft, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        self.hist.append((half > 0.03).astype(np.uint8))
        union = np.maximum.reduce(list(self.hist))
        safe = cv2.dilate(union, self.k_safe)
        safe = cv2.GaussianBlur(safe.astype(np.float32), (0, 0), 1.5)
        safe = cv2.resize(np.clip(safe * 1.6, 0.0, 1.0), (w, h), interpolation=cv2.INTER_LINEAR)
        to8 = lambda m: np.ascontiguousarray((m * 255.0 + 0.5).astype(np.uint8))
        return to8(soft), to8(np.clip(tight * 1.5, 0, 1)), to8(safe), float(soft.mean())


def fast_guided(guide, src, r, eps, s=2):
    h, w = guide.shape
    size = (w // s, h // s)
    gi = cv2.resize(guide, size, interpolation=cv2.INTER_AREA)
    pi = cv2.resize(src, size, interpolation=cv2.INTER_AREA)
    k = (2 * max(1, r // s) + 1,) * 2
    mean = lambda x: cv2.boxFilter(x, -1, k)
    mi = mean(gi)
    mp_ = mean(pi)
    var = mean(gi * gi) - mi * mi
    cov = mean(gi * pi) - mi * mp_
    a = cov / (var + eps)
    b = mp_ - a * mi
    A = cv2.resize(mean(a), (w, h), interpolation=cv2.INTER_LINEAR)
    B = cv2.resize(mean(b), (w, h), interpolation=cv2.INTER_LINEAR)
    return A * guide + B


def analyze(mask8):
    mask = mask8.astype(np.float32) / 255.0
    h, w = mask.shape
    m = mask > 0.5
    info = {
        "box": (0.3, 0.2, 0.7, 1.0),
        "neck": (0.5, 0.45),
        "head_r": 0.07,
        "chest": (0.5, 0.6),
        "center": (0.5, 0.6),
        "anchor": (0.5, 1.0),
        "exit": 1.0,
        "empty": True,
    }
    if m.sum() < 400:
        return info
    ys, xs = np.nonzero(m)
    x0, x1 = xs.min() / w, (xs.max() + 1) / w
    y0, y1 = ys.min() / h, (ys.max() + 1) / h
    widths = m.sum(1).astype(np.float32)
    top = int(ys.min())
    head_rows = ys < top + 0.12 * h
    head_x = float(xs[head_rows].mean()) if head_rows.any() else float(xs.mean())
    r0 = int(top + 0.04 * h)
    r1 = int(min(h - 1, top + 0.5 * h))
    mid = max(r0 + 2, (r0 + r1) // 2)
    widest = r0 + int(np.argmax(widths[r0:mid])) if mid > r0 else r0
    neck_row = widest + int(np.argmin(widths[widest:r1])) if r1 > widest else widest
    head_r = float(widths[widest]) / 2.0 / w
    neck = (head_x / w, neck_row / h)
    chest = (head_x / w, min(y1 - 0.02, neck_row / h + 0.14))
    bottom_rows = ys > ys.max() - 0.1 * h
    anchor = (float(xs[bottom_rows].mean()) / w, min(1.0, y1))
    cx = float(xs.mean()) / w
    info.update(
        box=(float(x0), float(y0), float(x1), float(y1)),
        neck=neck,
        head_r=max(head_r, 0.03),
        chest=chest,
        center=(cx, float(ys.mean()) / h),
        anchor=anchor,
        exit=1.0 if cx >= 0.5 else -1.0,
        empty=False,
    )
    return info
