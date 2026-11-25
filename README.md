# Vanish

A macOS menu-bar app that cuts you out of your webcam feed and lets you disappear from a Teams (or Zoom, Meet) call with a visual effect, leaving a live view of your empty desk. The output goes to the OBS virtual camera, so any meeting app can use it.

## Effects

| # | Effect | Gesture (hold ~0.6 s) | Hotkey |
|---|--------|-----------------------|--------|
| 1 | Dust in the wind: you crumble into particles blown sideways | open palm | ⌃⌥1 |
| 2 | Digital glitch: RGB split, tearing, dropped frames | horns | ⌃⌥2 |
| 3 | Frozen feed: fake bad connection, freeze, compression blocks, gone | fist | ⌃⌥3 |
| 4 | Shattered glass: cracks from a shot, shards fall | finger gun | ⌃⌥4 |
| 5 | Melt: you drip down like wax | thumb down | ⌃⌥5 |
| 6 | Paper plane: you get printed on paper, folded and flown away | V sign | ⌃⌥6 |
| 7 | Stage hook: a cane yanks you out of frame | shaka | ⌃⌥7 |
| 8 | Dash: you sprint out of frame with afterimages and dust | thumb sideways (sets direction) | ⌃⌥8 |
| 9 | Shrink: you shrink live, too slowly to notice at first (5 to 40 min) | pinch | ⌃⌥9 |

While you are gone, any gesture or hotkey brings you back by playing that effect in reverse. ⌃⌥0 returns with the last effect. A shrink snaps back with a spring.

## How it works

- AVFoundation captures the built-in camera (virtual and printer cameras are skipped).
- MediaPipe selfie segmentation (refined with a guided filter) gives the person matte; MediaPipe Hands drives the gestures.
- A background plate is learned from every pixel that is not covered by you. It is stored exposure-normalised, and a per-frame gain estimate keeps it matched to the camera's auto exposure, so no silhouette is left behind. Areas never seen are filled with a pull-push fill.
- Effects are GLSL shaders rendered off-screen with moderngl on the integrated GPU (the discrete GPU stays asleep).
- pyvirtualcam sends frames to the OBS virtual camera.
- While nothing is happening the camera frame is passed through untouched, which keeps CPU use low.

## Setup

1. Install OBS (`brew install --cask obs`), open it once, click **Start Virtual Camera**, approve the camera extension in System Settings, then stop it and quit OBS.
2. Python 3.12 venv in `Niepotrzebne/venv` with `mediapipe==0.10.21 moderngl==5.12.0 pyvirtualcam==0.15.0` and `pyobjc-*==11.1` (Cocoa, AVFoundation, CoreMedia, Quartz, libdispatch). pyobjc 13 breaks CoreMedia on this macOS.
3. Open `Vanish.app`, allow camera access, step out of frame for 3 seconds so it learns your empty background.
4. In Teams: Settings, Devices, Camera, **OBS Virtual Camera**.

## Testing without a camera

`tools/render_test.py --video clip.mp4 --crop w,h,x,y` runs the real pipeline on recorded footage and writes one mp4 and a contact sheet per effect.

## Files

- `vanish.py`: menu-bar app, worker thread, preview window.
- `engine/pipeline.py`: state machine (present, vanish, gone, return, shrink).
- `engine/effects.py`: the nine effects.
- `engine/matte.py`, `engine/gestures.py`, `engine/camera.py`, `engine/hotkeys.py`, `engine/renderer.py`, `engine/glsl.py`, `engine/glcontext_mac.py`.

Logs: `~/Library/Logs/Vanish.log`. Settings and the learned background: `~/Library/Application Support/Vanish/`.
