"""Offline check: runs the real pipeline on recorded footage and writes one mp4 plus a contact sheet per effect."""

import argparse
import pathlib
import subprocess
import sys

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.pipeline import PRESENT, Pipeline

FPS = 30


def decode(video, start, end, crop):
    x, y, w, h = crop
    vf = f"crop={w}:{h}:{x}:{y},scale=1280:720:flags=lanczos,fps={FPS}"
    cmd = ["ffmpeg", "-v", "error", "-threads", "2", "-ss", str(start), "-to", str(end), "-i", video, "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    data = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(data, np.uint8).reshape(-1, 720, 1280, 3)


def writer(path):
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x720", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-threads", "2", str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def sheet(frames, cols=4):
    thumbs = [cv2.resize(f, (320, 180), interpolation=cv2.INTER_AREA) for f in frames]
    while len(thumbs) % cols:
        thumbs.append(np.zeros_like(thumbs[0]))
    rows = [np.concatenate(thumbs[i:i + cols], 1) for i in range(0, len(thumbs), cols)]
    return np.concatenate(rows, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--crop", default="488,274,137,477")
    ap.add_argument("--effects", default="dust,glitch,freeze,glass,melt,plane,hook,dash,shrink")
    ap.add_argument("--out", default=str(ROOT / "Niepotrzebne" / "test"))
    ap.add_argument("--reverse", action="store_true")
    args = ap.parse_args()
    w, h, x, y = (int(v) for v in args.crop.split(","))
    crop = (x, y, w, h)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    empty = decode(args.video, 3.45, 5.4, crop)
    person = np.concatenate([decode(args.video, 0.0, 2.55, crop), decode(args.video, 6.1, 7.55, crop)])
    loop = np.concatenate([person, person[::-1]])
    pipe = Pipeline(gestures=None, shrink_minutes=0.12)
    for key in args.effects.split(","):
        eff = pipe.effects[key]
        clock = 0.0
        pipe.state = PRESENT
        pipe.matte.reset()
        for f in empty[:50]:
            pipe.step(f, clock)
            clock += 1.0 / FPS
        enc = writer(out / f"{key}.mp4")
        grabs = []
        if key == "shrink":
            plan = [(0.8, "trigger"), (8.5, "return"), (10.2, None)]
            sample = [0.8 + i * 0.7 for i in range(12)]
        else:
            d = eff.duration
            t_back = 0.8 + d + 1.0
            plan = [(0.8, "trigger"), (t_back, "return"), (t_back + d + 0.8, None)]
            base = t_back if args.reverse else 0.8
            sample = [base + d * i / 11.0 for i in range(12)]
        i = 0
        local = 0.0
        pending = list(plan)
        while pending:
            at, what = pending[0]
            if local >= at:
                pending.pop(0)
                if what == "trigger":
                    pipe.command("trigger", key)
                elif what == "return":
                    pipe.command("return")
                continue
            frame = loop[i % len(loop)]
            res = pipe.step(frame, clock)
            enc.stdin.write(np.ascontiguousarray(res).tobytes())
            if sample and local >= sample[0]:
                grabs.append(res.copy())
                sample.pop(0)
            i += 1
            local += 1.0 / FPS
            clock += 1.0 / FPS
        enc.stdin.close()
        enc.wait()
        cv2.imwrite(str(out / f"{key}.jpg"), cv2.cvtColor(sheet(grabs), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(key, "ok", pipe.state, flush=True)


if __name__ == "__main__":
    main()
