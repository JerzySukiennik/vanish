"""One-time activation of the OBS virtual camera without the user ever opening OBS."""

import configparser
import pathlib
import subprocess

import AVFoundation as AVF
from AppKit import NSRunningApplication

OBS_APP = pathlib.Path("/Applications/OBS.app")
OBS_ID = "com.obsproject.obs-studio"
CFG = pathlib.Path.home() / "Library" / "Application Support" / "obs-studio"
LIBOBS_33 = (33 << 24) | 3
SETTINGS_URL = "x-apple.systempreferences:com.apple.LoginItems-Settings.extension"


def obs_installed():
    return OBS_APP.exists()


def camera_ready():
    s = AVF.AVCaptureDeviceDiscoverySession.discoverySessionWithDeviceTypes_mediaType_position_(
        [AVF.AVCaptureDeviceTypeExternal], AVF.AVMediaTypeVideo, AVF.AVCaptureDevicePositionUnspecified
    )
    return any("obs virtual camera" in str(d.localizedName()).lower() for d in s.devices())


def _ensure(path, values):
    cp = configparser.ConfigParser(interpolation=None, strict=False)
    cp.optionxform = str
    if path.exists():
        try:
            cp.read(path, encoding="utf-8-sig")
        except configparser.Error:
            cp = configparser.ConfigParser(interpolation=None, strict=False)
            cp.optionxform = str
    if not cp.has_section("General"):
        cp.add_section("General")
    changed = False
    for k, v in values.items():
        if not cp.has_option("General", k):
            cp.set("General", k, v)
            changed = True
    if changed or not path.exists():
        with path.open("w", encoding="utf-8") as f:
            cp.write(f, space_around_delimiters=False)


def prepare_config():
    CFG.mkdir(parents=True, exist_ok=True)
    _ensure(CFG / "global.ini", {"LastVersion": str(LIBOBS_33), "MacOSPermissionsDialogLastShown": "99"})
    _ensure(CFG / "user.ini", {"FirstRun": "true"})


def quarantined():
    try:
        r = subprocess.run(["xattr", "-p", "com.apple.quarantine", str(OBS_APP)], capture_output=True, text=True, timeout=3)
    except Exception:
        return False
    flags = r.stdout.strip().split(";")[0]
    if r.returncode != 0 or not flags:
        return False
    try:
        return not (int(flags, 16) & 0x40)
    except ValueError:
        return False


def obs_running():
    return len(NSRunningApplication.runningApplicationsWithBundleIdentifier_(OBS_ID)) > 0


def launch_obs():
    prepare_config()
    subprocess.Popen(
        ["open", "-g", "-a", str(OBS_APP), "--args", "--startvirtualcam", "--minimize-to-tray",
         "--disable-shutdown-check", "--disable-updater", "--disable-missing-files-check"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def quit_obs():
    for app in NSRunningApplication.runningApplicationsWithBundleIdentifier_(OBS_ID):
        app.terminate()


def open_settings():
    subprocess.Popen(["open", SETTINGS_URL])
