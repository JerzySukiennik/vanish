"""Vanish: cuts you out of the webcam feed with a vanish effect and sends it to the OBS virtual camera for Teams."""

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
    NSAppearance,
    NSAppearanceNameDarkAqua,
    NSApplication,
    NSApplicationActivationPolicyRegular,
    NSBackingStoreBuffered,
    NSButton,
    NSColor,
    NSEventModifierFlagControl,
    NSEventModifierFlagOption,
    NSFont,
    NSFontWeightRegular,
    NSFontWeightSemibold,
    NSImage,
    NSImageScaleProportionallyUpOrDown,
    NSImageView,
    NSMenu,
    NSMenuItem,
    NSPopUpButton,
    NSStatusBar,
    NSTextField,
    NSVariableStatusItemLength,
    NSView,
    NSWindow,
    NSWindowStyleMaskClosable,
    NSWindowStyleMaskMiniaturizable,
    NSWindowStyleMaskTitled,
)
from Foundation import NSBundle, NSData, NSMakeRect, NSMakeSize, NSObject, NSRunLoop, NSRunLoopCommonModes, NSTimer

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from engine import camera as cammod
from engine import obs_setup
from engine.effects import EFFECTS
from engine.gestures import LABELS, Gestures
from engine.hotkeys import CONTROL, KEYCODES, OPTION, Hotkeys

SUPPORT = pathlib.Path.home() / "Library" / "Application Support" / "Vanish"
CONFIG = SUPPORT / "config.json"
STATE = SUPPORT / "plate.npz"
SNAP_REQUEST = SUPPORT / "snapshot-request"
ICON = ROOT / "Vanish.app" / "Contents" / "Resources" / "AppIcon.icns"
DEFAULTS = {"shrink_minutes": 10, "gestures": True, "camera": None}
SHRINK_CHOICES = [5, 10, 20, 40]
STATE_TEXT = {
    "present": "Jesteś w kadrze",
    "vanish": "Znikasz",
    "gone": "Zniknąłeś, widać puste biurko",
    "return": "Wracasz",
    "blend": "Wracasz",
    "shrink": "Kurczysz się",
    "grow": "Wracasz do rozmiaru",
}
OBS_TEXT = {
    None: "Kamera dla Teams: sprawdzam",
    "launched": "Włączam kamerę dla Teams (OBS działa w tle, nic nie klikaj)",
    "gatekeeper": "macOS pyta, czy otworzyć OBS: kliknij Otwórz w małym okienku",
    "approve": "macOS czeka na jedną zgodę: włącz rozszerzenie kamery OBS w Ustawieniach",
    "closing": "Kamera dla Teams prawie gotowa",
    "ready": "Kamera dla Teams gotowa: w Teams wybierz kamerę OBS Virtual Camera",
    "busy": "Zamknij OBS, Vanish sam wysyła obraz do Teams",
    "no_obs": "Brak OBS w Aplikacjach",
    "error": "Kamera dla Teams: błąd, szczegóły w logu",
}
CARD_ICONS = {
    "dust": "wind",
    "glitch": "waveform",
    "freeze": "wifi.exclamationmark",
    "glass": "burst",
    "melt": "drop",
    "plane": "paperplane",
    "hook": "arrow.uturn.left",
    "dash": "hare",
    "shrink": "arrow.down.right.and.arrow.up.left",
}
BAR_ICONS = {"present": "person.crop.square", "gone": "square.dashed", "shrink": "arrow.down.right.and.arrow.up.left", "busy": "sparkles", "hold": "hand.raised"}

WIN_W = 672
PAD = 16
PREV_H = 360
CARD_H = 64
GAP = 10
WIN_H = PAD + PREV_H + 12 + 40 + 12 + 3 * CARD_H + 2 * GAP + 14 + 30 + PAD


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
        self.vcam_error = None
        self.obs_phase = None
        self.obs_launch_t = 0.0
        self.next_vcam_try = 0.0
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

    def ensure_vcam(self, now):
        if self.vcam is not None or now < self.next_vcam_try:
            return
        self.next_vcam_try = now + 2.0
        if self.obs_phase in ("launched", "approve", "gatekeeper"):
            if obs_setup.camera_ready():
                log("rozszerzenie kamery OBS aktywne, zamykam OBS")
                obs_setup.quit_obs()
                self.obs_phase = "closing"
            elif obs_setup.quarantined():
                self.obs_phase = "gatekeeper"
                self.obs_launch_t = now
            elif now - self.obs_launch_t > 5.0:
                self.obs_phase = "approve"
            return
        if self.obs_phase == "closing" and obs_setup.obs_running():
            return
        self.open_vcam()
        if self.vcam is not None:
            if self.obs_phase != "ready":
                log("kamera dla Teams gotowa")
            self.obs_phase = "ready"
            return
        err = (self.vcam_error or "").lower()
        if "not installed" in err:
            if not obs_setup.obs_installed():
                self.obs_phase = "no_obs"
            elif not obs_setup.obs_running():
                log("uruchamiam OBS w tle, żeby włączyć wirtualną kamerę")
                obs_setup.launch_obs()
                self.obs_launch_t = now
                self.obs_phase = "launched"
            else:
                self.obs_launch_t = now
                self.obs_phase = "launched"
        elif obs_setup.obs_running():
            self.obs_phase = "busy"
        else:
            self.obs_phase = "error"
            log(f"kamera OBS: {self.vcam_error}")

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
            log(f"start: kamera {self.cam.name}, GPU {self.pipe.r.gpu}")
            seq = 0
            beat = 0
            frames = 0
            t_fps = time.monotonic()
            t_save = t_fps
            while self.running:
                frame, new_seq = self.cam.read(seq, 1.0)
                now = time.monotonic()
                self.ensure_vcam(now)
                if frame is None or new_seq == seq:
                    continue
                seq = new_seq
                out = self.pipe.step(frame)
                self.latest = out
                if self.vcam is not None:
                    try:
                        self.vcam.send(out)
                    except Exception as e:
                        self.vcam_error = str(e)
                        self.vcam = None
                        self.obs_phase = None
                frames += 1
                if now - t_fps >= 1.0:
                    self.fps = frames / (now - t_fps)
                    frames = 0
                    t_fps = now
                    beat += 1
                    if beat % 30 == 0:
                        log(f"{self.fps:.1f} kl/s, stan {self.pipe.state}, tło {self.pipe.seen:.0%}, Teams {self.obs_phase}")
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


def gray(w):
    return NSColor.colorWithWhite_alpha_(w, 1.0)


def label(text, size, weight, white):
    lab = NSTextField.labelWithString_(text)
    lab.setFont_(NSFont.systemFontOfSize_weight_(size, weight))
    lab.setTextColor_(gray(white))
    return lab


def top(y, h):
    return WIN_H - y - h


class AppDelegate(NSObject):
    def applicationDidFinishLaunching_(self, note):
        self.cfg = load_config()
        self.worker = None
        self.bar_icon = None
        self.active_key = "?"
        self.cards = []
        self.build_main_menu()
        self.build_window()
        self.build_status_item()
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
        self.showWindow_(None)

    def applicationShouldHandleReopen_hasVisibleWindows_(self, app, visible):
        self.showWindow_(None)
        return True

    def applicationShouldTerminateAfterLastWindowClosed_(self, app):
        return False

    def applicationWillTerminate_(self, note):
        if self.worker is not None:
            self.worker.running = False
            self.worker.done.wait(3.0)
        self.hotkeys.unregister_all()

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
    def build_main_menu(self):
        main = NSMenu.alloc().init()
        app_item = NSMenuItem.alloc().init()
        main.addItem_(app_item)
        app_menu = NSMenu.alloc().initWithTitle_("Vanish")
        app_menu.addItemWithTitle_action_keyEquivalent_("Zamknij okno", b"performClose:", "w")
        app_menu.addItemWithTitle_action_keyEquivalent_("Zakończ Vanish", b"terminate:", "q")
        app_item.setSubmenu_(app_menu)
        NSApplication.sharedApplication().setMainMenu_(main)

    @objc.python_method
    def build_window(self):
        style = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable
        win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(NSMakeRect(0, 0, WIN_W, WIN_H), style, NSBackingStoreBuffered, False)
        win.setTitle_("Vanish")
        win.setReleasedWhenClosed_(False)
        win.setAppearance_(NSAppearance.appearanceNamed_(NSAppearanceNameDarkAqua))
        win.setBackgroundColor_(gray(0.08))
        win.center()
        root = win.contentView()
        inner = WIN_W - 2 * PAD

        holder = NSView.alloc().initWithFrame_(NSMakeRect(PAD, top(PAD, PREV_H), inner, PREV_H))
        holder.setWantsLayer_(True)
        holder.layer().setCornerRadius_(12)
        holder.layer().setMasksToBounds_(True)
        holder.layer().setBackgroundColor_(gray(0.0).CGColor())
        self.preview = NSImageView.alloc().initWithFrame_(NSMakeRect(0, 0, inner, PREV_H))
        self.preview.setImageScaling_(NSImageScaleProportionallyUpOrDown)
        holder.addSubview_(self.preview)
        self.preview_hint = label("Uruchamiam kamerę", 13, NSFontWeightRegular, 0.55)
        self.preview_hint.setFrame_(NSMakeRect(0, PREV_H / 2 - 10, inner, 20))
        self.preview_hint.setAlignment_(2)
        holder.addSubview_(self.preview_hint)
        root.addSubview_(holder)

        y = PAD + PREV_H + 12
        self.state_label = label("Startuję", 14, NSFontWeightSemibold, 0.95)
        self.state_label.setFrame_(NSMakeRect(PAD, top(y, 20), inner - 170, 20))
        root.addSubview_(self.state_label)
        self.obs_label = label(OBS_TEXT[None], 11, NSFontWeightRegular, 0.6)
        self.obs_label.setFrame_(NSMakeRect(PAD, top(y + 22, 16), inner - 170, 16))
        root.addSubview_(self.obs_label)
        self.settings_btn = NSButton.buttonWithTitle_target_action_("Otwórz Ustawienia", self, b"openSettings:")
        self.settings_btn.setFrame_(NSMakeRect(WIN_W - PAD - 160, top(y + 4, 32), 160, 32))
        self.settings_btn.setHidden_(True)
        root.addSubview_(self.settings_btn)

        y += 40 + 12
        cw = (inner - 2 * GAP) / 3.0
        for i, cls in enumerate(EFFECTS):
            r, c = divmod(i, 3)
            frame = NSMakeRect(PAD + c * (cw + GAP), top(y + r * (CARD_H + GAP), CARD_H), cw, CARD_H)
            self.cards.append((cls.key, self.make_card(root, frame, i, cls)))

        y += 3 * CARD_H + 2 * GAP + 14
        back = NSButton.buttonWithTitle_target_action_("Wróć do kadru   ⌃⌥0", self, b"returnNow:")
        back.setFrame_(NSMakeRect(PAD - 6, top(y, 30), 190, 30))
        root.addSubview_(back)
        self.gest_box = NSButton.checkboxWithTitle_target_action_("Gesty", self, b"toggleGestures:")
        self.gest_box.setState_(1 if self.cfg["gestures"] else 0)
        self.gest_box.setFrame_(NSMakeRect(PAD + 196, top(y + 5, 20), 80, 20))
        root.addSubview_(self.gest_box)
        shrink_lab = label("Kurczenie", 12, NSFontWeightRegular, 0.7)
        shrink_lab.setFrame_(NSMakeRect(PAD + 280, top(y + 7, 16), 70, 16))
        root.addSubview_(shrink_lab)
        self.shrink_pop = NSPopUpButton.alloc().initWithFrame_pullsDown_(NSMakeRect(PAD + 348, top(y, 30), 100, 30), False)
        for m in SHRINK_CHOICES:
            self.shrink_pop.addItemWithTitle_(f"{m} min")
        if self.cfg["shrink_minutes"] in SHRINK_CHOICES:
            self.shrink_pop.selectItemAtIndex_(SHRINK_CHOICES.index(self.cfg["shrink_minutes"]))
        self.shrink_pop.setTarget_(self)
        self.shrink_pop.setAction_(b"pickShrink:")
        root.addSubview_(self.shrink_pop)
        forget = NSButton.buttonWithTitle_target_action_("Zapomnij tło", self, b"resetPlate:")
        forget.setFrame_(NSMakeRect(WIN_W - PAD - 124, top(y, 30), 130, 30))
        root.addSubview_(forget)
        self.window = win

    @objc.python_method
    def make_card(self, parent, frame, i, cls):
        w, h = frame.size.width, frame.size.height
        view = NSView.alloc().initWithFrame_(frame)
        view.setWantsLayer_(True)
        layer = view.layer()
        layer.setCornerRadius_(10)
        layer.setBackgroundColor_(gray(0.14).CGColor())
        layer.setBorderWidth_(1.0)
        layer.setBorderColor_(gray(0.22).CGColor())
        img = symbol(CARD_ICONS.get(cls.key, "sparkles"))
        if img is not None:
            icon = NSImageView.alloc().initWithFrame_(NSMakeRect(12, (h - 22) / 2, 22, 22))
            icon.setImage_(img)
            icon.setContentTintColor_(gray(0.92))
            view.addSubview_(icon)
        title = label(cls.name, 13, NSFontWeightSemibold, 0.95)
        title.setFrame_(NSMakeRect(44, h / 2 + 1, w - 50, 18))
        view.addSubview_(title)
        sub = label(f"{cls.gesture}   ⌃⌥{i + 1}", 11, NSFontWeightRegular, 0.55)
        sub.setFrame_(NSMakeRect(44, h / 2 - 17, w - 50, 15))
        view.addSubview_(sub)
        btn = NSButton.alloc().initWithFrame_(NSMakeRect(0, 0, w, h))
        btn.setTransparent_(True)
        btn.setTarget_(self)
        btn.setAction_(b"pickEffect:")
        btn.setTag_(i)
        view.addSubview_(btn)
        parent.addSubview_(view)
        return view

    @objc.python_method
    def build_status_item(self):
        self.status_item = NSStatusBar.systemStatusBar().statusItemWithLength_(NSVariableStatusItemLength)
        self.set_bar_icon("present")
        menu = NSMenu.alloc().init()
        menu.setAutoenablesItems_(False)
        show = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Pokaż Vanish", b"showWindow:", "")
        show.setTarget_(self)
        menu.addItem_(show)
        menu.addItem_(NSMenuItem.separatorItem())
        for i, cls in enumerate(EFFECTS):
            it = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(cls.name, b"pickEffect:", str(i + 1))
            it.setKeyEquivalentModifierMask_(NSEventModifierFlagControl | NSEventModifierFlagOption)
            it.setTarget_(self)
            it.setTag_(i)
            menu.addItem_(it)
        back = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Wróć do kadru", b"returnNow:", "0")
        back.setKeyEquivalentModifierMask_(NSEventModifierFlagControl | NSEventModifierFlagOption)
        back.setTarget_(self)
        menu.addItem_(back)
        menu.addItem_(NSMenuItem.separatorItem())
        menu.addItem_(NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Zakończ Vanish", b"terminate:", ""))
        self.status_item.setMenu_(menu)

    @objc.python_method
    def set_bar_icon(self, name):
        if name == self.bar_icon:
            return
        self.bar_icon = name
        img = symbol(BAR_ICONS[name])
        button = self.status_item.button()
        if img is not None:
            button.setImage_(img)
        else:
            button.setTitle_("V")

    @objc.python_method
    def texts(self):
        w = self.worker
        if w is None:
            st = cammod.authorization()
            if st in (1, 2):
                return "Brak dostępu do kamery", "Ustawienia systemowe, Prywatność i ochrona, Kamera: włącz Vanish"
            return "Czekam na dostęp do kamery", "Kliknij Pozwól w oknie macOS"
        if w.error:
            return "Błąd: " + w.error.strip().splitlines()[-1][:70], "Szczegóły w ~/Library/Logs/Vanish.log"
        p = w.pipe
        if p is None:
            return "Startuję", OBS_TEXT[None]
        state = STATE_TEXT.get(p.state, p.state)
        if p.state == "shrink":
            state += f", {round(p.scale * 100)}% rozmiaru"
        g, prog = p.gesture_status
        if g is not None and p.gestures_on and prog > 0 and not p.busy():
            state += f"   ·   gest: {LABELS.get(g, g)} {round(prog * 100)}%"
        if p.seen < 0.97 and p.state == "present":
            state += "   ·   wyjdź na 3 s z kadru, żebym poznał tło"
        return state, OBS_TEXT.get(w.obs_phase, OBS_TEXT[None])

    def tick_(self, timer):
        w = self.worker
        p = w.pipe if w is not None else None
        if p is not None:
            g, prog = p.gesture_status
            if p.busy():
                self.set_bar_icon("busy")
            elif g is not None and prog > 0 and p.gestures_on:
                self.set_bar_icon("hold")
            elif p.state in ("gone", "shrink"):
                self.set_bar_icon(p.state)
            else:
                self.set_bar_icon("present")
            active = None
            if p.state in ("vanish", "gone"):
                active = p.last_effect
            elif p.state == "return" and p.effect is not None:
                active = p.effect.key
            elif p.state in ("shrink", "grow"):
                active = "shrink"
            if active != self.active_key:
                self.active_key = active
                for key, view in self.cards:
                    on = key == active
                    view.layer().setBorderColor_(gray(0.92 if on else 0.22).CGColor())
                    view.layer().setBorderWidth_(1.5 if on else 1.0)
                    view.layer().setBackgroundColor_(gray(0.2 if on else 0.14).CGColor())
        if SNAP_REQUEST.exists():
            SNAP_REQUEST.unlink()
            self.save_snapshot()
        if not self.window.isVisible() or self.window.isMiniaturized():
            return
        if w is not None and w.latest is not None:
            self.preview.setImage_(frame_to_nsimage(w.latest))
            self.preview_hint.setHidden_(True)
        a, b = self.texts()
        self.state_label.setStringValue_(a)
        self.obs_label.setStringValue_(b)
        self.settings_btn.setHidden_(not (w is not None and w.obs_phase == "approve"))

    @objc.python_method
    def save_snapshot(self):
        view = self.window.contentView()
        rep = view.bitmapImageRepForCachingDisplayInRect_(view.bounds())
        view.cacheDisplayInRect_toBitmapImageRep_(view.bounds(), rep)
        rep.representationUsingType_properties_(4, {}).writeToFile_atomically_(str(SUPPORT / "window.png"), True)

    def showWindow_(self, sender):
        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
        self.window.makeKeyAndOrderFront_(None)

    def pickEffect_(self, sender):
        self.send("trigger", EFFECTS[sender.tag()].key)

    def returnNow_(self, sender):
        self.send("return")

    def openSettings_(self, sender):
        obs_setup.open_settings()

    def toggleGestures_(self, sender):
        self.cfg["gestures"] = bool(sender.state())
        save_config(self.cfg)
        if self.worker is not None and self.worker.pipe is not None:
            self.worker.pipe.gestures_on = self.cfg["gestures"]

    def pickShrink_(self, sender):
        m = SHRINK_CHOICES[sender.indexOfSelectedItem()]
        self.cfg["shrink_minutes"] = m
        save_config(self.cfg)
        if self.worker is not None and self.worker.pipe is not None:
            self.worker.pipe.shrink_minutes = m

    def resetPlate_(self, sender):
        self.send("plate_reset")


def main():
    info = NSBundle.mainBundle().infoDictionary()
    if info is not None:
        info["CFBundleName"] = "Vanish"
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyRegular)
    if ICON.exists():
        app.setApplicationIconImage_(NSImage.alloc().initWithContentsOfFile_(str(ICON)))
    delegate = AppDelegate.alloc().init()
    app.setDelegate_(delegate)
    app.run()


if __name__ == "__main__":
    main()
