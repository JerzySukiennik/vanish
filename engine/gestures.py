"""Hand-pose triggers from MediaPipe Hands: a pose must be held steadily before it fires."""

import mediapipe as mp
import numpy as np

FINGERS = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}

LABELS = {
    "palm": "otwarta dłoń",
    "horns": "rogi",
    "fist": "pięść",
    "gun": "pistolet z palców",
    "thumb_down": "kciuk w dół",
    "peace": "V z palców",
    "shaka": "shaka",
    "thumb_side": "kciuk w bok",
    "pinch": "szczypta",
}


def _cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))


def classify(pts):
    wrist = pts[0]
    palm = np.linalg.norm(pts[9, :2] - wrist[:2]) + 1e-6
    state = {}
    for name, (mcp, pip, dip, tip) in FINGERS.items():
        straight = _cos(pts[pip] - pts[mcp], pts[tip] - pts[pip])
        reach = np.linalg.norm(pts[tip, :2] - wrist[:2]) / (np.linalg.norm(pts[pip, :2] - wrist[:2]) + 1e-6)
        if straight > 0.72 and reach > 1.12:
            state[name] = "ext"
        elif straight < 0.35 or reach < 1.0:
            state[name] = "curl"
        else:
            state[name] = "mid"
    thumb_dir = pts[4, :2] - pts[2, :2]
    thumb_len = np.linalg.norm(thumb_dir) + 1e-6
    thumb_dir = thumb_dir / thumb_len
    thumb_straight = _cos(pts[3] - pts[2], pts[4] - pts[3]) > 0.6
    thumb_out = np.linalg.norm(pts[4, :2] - pts[5, :2]) > 0.55 * palm and np.linalg.norm(pts[4, :2] - pts[10, :2]) > 0.55 * palm
    thumb_ext = thumb_straight and thumb_out
    s = state
    four_curl = all(s[f] == "curl" for f in FINGERS)
    info = {"dx": float(thumb_dir[0])}
    pinch_gap = np.linalg.norm(pts[4, :2] - pts[8, :2]) / palm
    index_reach = np.linalg.norm(pts[8, :2] - wrist[:2]) / palm
    if s["middle"] == "curl" and s["ring"] == "curl" and s["pinky"] == "curl" and pinch_gap < 0.33 and index_reach > 0.85 and s["index"] != "ext":
        return "pinch", info
    if all(s[f] == "ext" for f in FINGERS) and thumb_ext:
        return "palm", info
    if four_curl:
        if thumb_ext:
            if thumb_dir[1] > 0.6:
                return "thumb_down", info
            if abs(thumb_dir[0]) > 0.75:
                return "thumb_side", info
            return None, info
        return "fist", info
    if s["index"] == "ext" and s["middle"] == "ext" and s["ring"] == "curl" and s["pinky"] == "curl":
        if np.linalg.norm(pts[8, :2] - pts[12, :2]) > 0.28 * palm:
            return "peace", info
        return None, info
    if s["index"] == "ext" and s["pinky"] == "ext" and s["middle"] == "curl" and s["ring"] == "curl":
        return "horns", info
    if s["index"] == "ext" and s["middle"] == "curl" and s["ring"] == "curl" and s["pinky"] == "curl":
        return "gun", info
    if s["pinky"] == "ext" and thumb_ext and s["index"] == "curl" and s["middle"] == "curl" and s["ring"] == "curl":
        return "shaka", info
    return None, info


class Gestures:
    HOLD = 0.6
    GRACE = 0.3
    COOLDOWN = 2.0

    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=1, model_complexity=0,
            min_detection_confidence=0.6, min_tracking_confidence=0.5,
        )
        self.current = None
        self.since = 0.0
        self.last_seen = 0.0
        self.armed = True
        self.cooldown_until = 0.0
        self.info = {}

    def process(self, small_rgb, now):
        res = self.hands.process(small_rgb)
        g = None
        info = {}
        if res.multi_hand_landmarks:
            h, w = small_rgb.shape[:2]
            lm = res.multi_hand_landmarks[0].landmark
            pts = np.array([[p.x * w, p.y * h, p.z * w] for p in lm], np.float32)
            if pts[0, 1] < h * 0.96:
                g, info = classify(pts)
                info["hand_x"] = float(pts[:, 0].mean() / w)
        if g is not None:
            self.last_seen = now
            if g != self.current:
                self.current = g
                self.since = now
            self.info = info
        elif self.current is not None and now - self.last_seen > self.GRACE:
            self.current = None
        if self.current is None:
            self.armed = True
            return None, (None, 0.0)
        progress = min(1.0, (now - self.since) / self.HOLD)
        if progress >= 1.0 and self.armed and now >= self.cooldown_until:
            self.armed = False
            self.cooldown_until = now + self.COOLDOWN
            return (self.current, dict(self.info)), (self.current, 1.0)
        return None, (self.current, progress if self.armed else 0.0)
