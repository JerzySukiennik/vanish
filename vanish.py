"""Vanish: menu-bar app that cuts you out of the webcam feed with a vanish effect and sends it to the OBS virtual camera for Teams."""

import json
import pathlib
import sys
import threading
import time
import traceback

import cv2
import objc
import Quartz
from AppKit import (
    NSApplication,
    NSApplicationActivationPolicyAccessory,
    NSBackingStoreBuffered,
    NSColor,
    NSEventModifierFlagControl,
    NSEventModifierFlagOption,
    NSFont,
    NSImage,
    NSImageScaleProportionallyUpOrDown,
    NSImageView,
    NSMenu,
    NSMenuItem,
    NSStatusBar,
    NSTextField,
    NSVariableStatusItemLength,
    NSWindow,
    NSWindowStyleMaskClosable,
    NSWindowStyleMaskMiniaturizable,
    NSWindowStyleMaskTitled,
)
from Foundation import NSData, NSMakeRect, NSMakeSize, NSObject, NSRunLoop, NSRunLoopCommonModes, NSTimer

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from engine import camera as cammod
from engine.effects import EFFECTS
from engine.gestures import LABELS, Gestures
from engine.hotkeys import CONTROL, KEYCODES, OPTION, Hotkeys

SUPPORT = pathlib.Path.home() / "Library" / "Application Support" / "Vanish"
CONFIG = SUPPORT / "config.json"
STATE = SUPPORT / "plate.npz"
DEFAULTS = {"shrink_minutes": 10, "gestures": True, "camera": None, "first_run": True}
STATE_TEXT = {
    "present": "Jesteś w kadrze",
    "vanish": "Znikasz",
    "gone": "Zniknąłeś, widać puste biurko",
    "return": "Wracasz",
    "blend": "Wracasz",
    "shrink": "Kurczysz się",
    "grow": "Wracasz do rozmiaru",
}
ICONS = {
    "present": "person.crop.square",
    "gone": "square.dashed",
    "shrink": "arrow.down.right.and.arrow.up.left",
    "busy": "sparkles",
    "hold": "hand.raised",
    "error": "exclamationmark.triangle",
}
SHRINK_CHOICES = [5, 10, 20, 40]


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def load_config():
    cfg = dict(DEFAULTS)
    try:
        cfg.update(json.loads(CONFIG.read_text()))
    except Exception:
        pass
    return cfg


def save_config(cfg):
    SUPPORT.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))


class Worker(threading.Thread):
    def __init__(self, cfg):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.pipe = None
        self.cam = None
        self.vcam = None
        self.vcam_error = "jeszcze nie startowała"
        self.error = None
        self.running = True
        self.done = threading.Event()
        self.latest = None
        self.fps = 0.0

    def open_vcam(self):
        try:
            import pyvirtualcam

            self.vcam = pyvirtualcam.Camera(width=1280, height=720, fps=30, fmt=pyvirtualcam.PixelFormat.RGB, backend="obs")
            self.vcam_error = None
        except Exception as e:
            self.vcam = None
            self.vcam_error = str(e)

    def run(self):
        try:
            from engine.pipeline import Pipeline

            gest = Gestures()
            self.pipe = Pipeline(gestures=gest, shrink_minutes=self.cfg["shrink_minutes"])
            self.pipe.gestures_on = self.cfg["gestures"]
            if STATE.exists():
                try:
                    self.pipe.load_state(STATE)
                except Exception:
                    pass
            self.cam = cammod.Camera(self.cfg.get("camera"))
            self.cam.start()
            self.open_vcam()
            log(f"start: kamera {self.cam.name}, GPU {self.pipe.r.gpu}, OBS {'ok' if self.vcam else self.vcam_error}")
            seq = 0
            beat = 0
            frames = 0
            t_fps = time.monotonic()
            t_vcam = t_fps
            t_save = t_fps
            while self.running:
                frame, new_seq = self.cam.read(seq, 1.0)
                if frame is None or new_seq == seq:
                    continue
                seq = new_seq
                out = self.pipe.step(frame)
                self.latest = out
                now = time.monotonic()
                if self.vcam is not None:
                    try:
                        self.vcam.send(out)
                    except Exception as e:
                        self.vcam_error = str(e)
                        self.vcam = None
                        t_vcam = now
                elif now - t_vcam > 8.0:
                    t_vcam = now
                    self.open_vcam()
                frames += 1
                if now - t_fps >= 1.0:
                    self.fps = frames / (now - t_fps)
                    frames = 0
                    t_fps = now
                    beat += 1
                    if beat % 10 == 0:
                        log(f"{self.fps:.1f} kl/s, stan {self.pipe.state}, tło {self.pipe.seen:.0%}, OBS {'ok' if self.vcam else 'brak'}")
                if now - t_save > 300 and self.pipe.state == "present":
                    self.save()
                    t_save = now
        except Exception:
            self.error = traceback.format_exc()
            print(self.error, file=sys.stderr, flush=True)
        finally:
            self.save()
            if self.cam is not None:
                self.cam.stop()
            if self.vcam is not None:
                try:
                    self.vcam.close()
                except Exception:
                    pass
            self.done.set()

    def save(self):
        if self.pipe is None:
            return
        try:
            SUPPORT.mkdir(parents=True, exist_ok=True)
            self.pipe.save_state(STATE)
        except Exception:
            pass


def frame_to_nsimage(rgb):
    small = cv2.resize(rgb, (640, 360), interpolation=cv2.INTER_AREA)
    small = cv2.flip(small, 1)
    data = NSData.dataWithBytes_length_(small.tobytes(), small.nbytes)
    provider = Quartz.CGDataProviderCreateWithCFData(data)
    cs = Quartz.CGColorSpaceCreateDeviceRGB()
    img = Quartz.CGImageCreate(640, 360, 8, 24, 640 * 3, cs, Quartz.kCGImageAlphaNone, provider, None, False, Quartz.kCGRenderingIntentDefault)
    return NSImage.alloc().initWithCGImage_size_(img, NSMakeSize(640, 360))


def symbol(name):
    img = NSImage.imageWithSystemSymbolName_accessibilityDescription_(name, "Vanish")
    if img is not None:
        img.setTemplate_(True)
    return img


class AppDelegate(NSObject):
    def applicationDidFinishLaunching_(self, note):
        self.cfg = load_config()
        self.worker = None
        self.window = None
        self.icon_name = None
        self.status_item = NSStatusBar.systemStatusBar().statusItemWithLength_(NSVariableStatusItemLength)
        self.set_icon("present")
        self.build_menu()
        self.hotkeys = Hotkeys()
        for i, cls in enumerate(EFFECTS):
            self.hotkeys.register(KEYCODES[str(i + 1)], CONTROL | OPTION, self.make_trigger(cls.key))
        self.hotkeys.register(KEYCODES["0"], CONTROL | OPTION, lambda: self.send("return"))
        self.timer = NSTimer.timerWithTimeInterval_target_selector_userInfo_repeats_(1.0 / 15.0, self, b"tick:", None, True)
        NSRunLoop.currentRunLoop().addTimer_forMode_(self.timer, NSRunLoopCommonModes)
        status = cammod.authorization()
        log(f"uprawnienie do kamery: {status}")
        if status == 3:
            self.start_worker()
        elif status == 0:
            cammod.request_access(self.access_result)
        if self.cfg.get("first_run", True):
            self.cfg["first_run"] = False
            save_config(self.cfg)
            self.showPreview_(None)

    @objc.python_method
    def access_result(self, granted):
        log(f"odpowiedź o kamerę: {bool(granted)}")
        if granted:
            self.start_worker()

    @objc.python_method
    def make_trigger(self, key):
        return lambda: self.send("trigger", key)

    @objc.python_method
    def start_worker(self):
        if self.worker is None:
            self.worker = Worker(self.cfg)
            self.worker.start()

    @objc.python_method
    def send(self, kind, key=None):
        w = self.worker
        if w is not None and w.pipe is not None:
            w.pipe.command(kind, key)

    @objc.python_method
    def item(self, title, action=None, key="", tag=0):
        it = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, action, key)
        it.setTarget_(self)
        it.setTag_(tag)
        if key:
            it.setKeyEquivalentModifierMask_(NSEventModifierFlagControl | NSEventModifierFlagOption)
        return it

    @objc.python_method
    def build_menu(self):
        menu = NSMenu.alloc().init()
        menu.setAutoenablesItems_(False)
        menu.setDelegate_(self)
        self.state_item = self.item("Startuję")
        self.state_item.setEnabled_(False)
        self.vcam_item = self.item("Teams: łączę z kamerą OBS")
        self.vcam_item.setEnabled_(False)
        self.plate_item = self.item("Tło: uczę się")
        self.plate_item.setEnabled_(False)
        for it in (self.state_item, self.vcam_item, self.plate_item):
            menu.addItem_(it)
        menu.addItem_(NSMenuItem.separatorItem())
        for i, cls in enumerate(EFFECTS):
            it = self.item(f"{cls.name}   ({cls.gesture})", b"pickEffect:", str(i + 1), i)
            menu.addItem_(it)
        menu.addItem_(self.item("Wróć do kadru", b"returnNow:", "0"))
        menu.addItem_(NSMenuItem.separatorItem())
        self.gest_item = self.item("Gesty", b"toggleGestures:")
        self.gest_item.setState_(1 if self.cfg["gestures"] else 0)
        menu.addItem_(self.gest_item)
        shrink_root = self.item("Tempo kurczenia")
        sub = NSMenu.alloc().init()
        self.shrink_items = []
        for m in SHRINK_CHOICES:
            it = self.item(f"{m} min", b"pickShrink:", "", m)
            it.setState_(1 if self.cfg["shrink_minutes"] == m else 0)
            sub.addItem_(it)
            self.shrink_items.append(it)
        shrink_root.setSubmenu_(sub)
        menu.addItem_(shrink_root)
        menu.addItem_(self.item("Zapomnij tło", b"resetPlate:"))
        menu.addItem_(self.item("Podgląd", b"showPreview:"))
        menu.addItem_(NSMenuItem.separatorItem())
        menu.addItem_(self.item("Zakończ Vanish", b"quit:"))
        self.status_item.setMenu_(menu)
        self.menu = menu

    def menuWillOpen_(self, menu):
        self.refresh_texts()

    @objc.python_method
    def status_texts(self):
        w = self.worker
        if w is None:
            st = cammod.authorization()
            if st in (1, 2):
                return "Brak dostępu do kamery (Ustawienia, Prywatność, Kamera)", "", ""
            return "Czekam na dostęp do kamery", "", ""
        if w.error:
            return "Błąd: " + w.error.strip().splitlines()[-1][:80], "", ""
        p = w.pipe
        if p is None:
            return "Startuję", "", ""
        state = STATE_TEXT.get(p.state, p.state)
        if p.state == "shrink":
            state += f" ({round(p.scale * 100)}% rozmiaru)"
        vcam = "Teams: kamera OBS Virtual Camera gotowa" if w.vcam is not None else "Teams: kamera OBS nieaktywna, włącz ją raz w OBS"
        seen = round(p.seen * 100)
        plate = f"Tło zapamiętane w {seen}%" + ("" if seen > 97 else ", wyjdź na 3 s z kadru, żeby uzupełnić")
        return state, vcam, plate

    @objc.python_method
    def refresh_texts(self):
        a, b, c = self.status_texts()
        self.state_item.setTitle_(a)
        self.vcam_item.setTitle_(b or "Teams: czekam")
        self.plate_item.setTitle_(c or "Tło: czekam")

    @objc.python_method
    def set_icon(self, name):
        if name == self.icon_name:
            return
        self.icon_name = name
        img = symbol(ICONS[name])
        button = self.status_item.button()
        if img is not None:
            button.setImage_(img)
            button.setTitle_("")
        else:
            button.setTitle_("V")

    def tick_(self, timer):
        w = self.worker
        p = w.pipe if w is not None else None
        if w is not None and w.error:
            self.set_icon("error")
        elif p is not None:
            g, prog = p.gesture_status
            if p.busy():
                self.set_icon("busy")
            elif g is not None and prog > 0 and p.gestures_on:
                self.set_icon("hold")
            elif p.state in ("gone", "shrink"):
                self.set_icon(p.state)
            else:
                self.set_icon("present")
        if self.window is not None and self.window.isVisible() and w is not None and w.latest is not None:
            self.image_view.setImage_(frame_to_nsimage(w.latest))
            a, b, c = self.status_texts()
            extra = ""
            if p is not None:
                g, prog = p.gesture_status
                if g is not None and p.gestures_on:
                    extra = f"   Gest: {LABELS.get(g, g)} {round(prog * 100)}%"
            self.label.setStringValue_(f"{a}   {round(w.fps)} kl/s{extra}\n{b}\n{c}")

    def pickEffect_(self, sender):
        self.send("trigger", EFFECTS[sender.tag()].key)

    def returnNow_(self, sender):
        self.send("return")

    def toggleGestures_(self, sender):
        self.cfg["gestures"] = not self.cfg["gestures"]
        save_config(self.cfg)
        self.gest_item.setState_(1 if self.cfg["gestures"] else 0)
        if self.worker is not None and self.worker.pipe is not None:
            self.worker.pipe.gestures_on = self.cfg["gestures"]

    def pickShrink_(self, sender):
        m = int(sender.tag())
        self.cfg["shrink_minutes"] = m
        save_config(self.cfg)
        for it in self.shrink_items:
            it.setState_(1 if it.tag() == m else 0)
        if self.worker is not None and self.worker.pipe is not None:
            self.worker.pipe.shrink_minutes = m

    def resetPlate_(self, sender):
        self.send("plate_reset")

    def showPreview_(self, sender):
        if self.window is None:
            style = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable
            win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(NSMakeRect(200, 200, 640, 430), style, NSBackingStoreBuffered, False)
            win.setTitle_("Vanish: podgląd (to samo widzą inni w Teams)")
            win.setReleasedWhenClosed_(False)
            win.setBackgroundColor_(NSColor.blackColor())
            view = NSImageView.alloc().initWithFrame_(NSMakeRect(0, 70, 640, 360))
            view.setImageScaling_(NSImageScaleProportionallyUpOrDown)
            label = NSTextField.alloc().initWithFrame_(NSMakeRect(12, 6, 616, 58))
            label.setEditable_(False)
            label.setBordered_(False)
            label.setDrawsBackground_(False)
            label.setTextColor_(NSColor.colorWithWhite_alpha_(0.85, 1.0))
            label.setFont_(NSFont.systemFontOfSize_(12))
            label.setStringValue_("Wyjdź z kadru na 3 sekundy, żebym zapamiętał puste tło.")
            win.contentView().addSubview_(view)
            win.contentView().addSubview_(label)
            self.window = win
            self.image_view = view
            self.label = label
        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
        self.window.makeKeyAndOrderFront_(None)

    def quit_(self, sender):
        if self.worker is not None:
            self.worker.running = False
            self.worker.done.wait(3.0)
        self.hotkeys.unregister_all()
        NSApplication.sharedApplication().terminate_(None)


def main():
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    delegate = AppDelegate.alloc().init()
    app.setDelegate_(delegate)
    app.run()


if __name__ == "__main__":
    main()
