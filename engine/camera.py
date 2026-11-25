"""Webcam capture through AVFoundation; skips virtual and printer cameras so OBS never loops back into itself."""

import threading
import time

import AVFoundation as AVF
import CoreMedia
import cv2
import dispatch
import numpy as np
import objc
import Quartz
from Foundation import NSNotificationCenter, NSObject

EXCLUDE = ("obs", "virtual", "printer", "laserjet", "officejet", "scanner")


def list_cameras():
    types = [AVF.AVCaptureDeviceTypeBuiltInWideAngleCamera, AVF.AVCaptureDeviceTypeExternal]
    cont = getattr(AVF, "AVCaptureDeviceTypeContinuityCamera", None)
    if cont is not None:
        types.append(cont)
    session = AVF.AVCaptureDeviceDiscoverySession.discoverySessionWithDeviceTypes_mediaType_position_(
        types, AVF.AVMediaTypeVideo, AVF.AVCaptureDevicePositionUnspecified
    )
    return [d for d in session.devices() if not any(x in d.localizedName().lower() for x in EXCLUDE)]


def pick_camera(preferred=None):
    devs = list_cameras()
    if preferred:
        for d in devs:
            if d.localizedName() == preferred:
                return d
    builtin = [d for d in devs if d.deviceType() == AVF.AVCaptureDeviceTypeBuiltInWideAngleCamera]
    return (builtin or devs or [None])[0]


def authorization():
    return int(AVF.AVCaptureDevice.authorizationStatusForMediaType_(AVF.AVMediaTypeVideo))


def request_access(callback):
    AVF.AVCaptureDevice.requestAccessForMediaType_completionHandler_(AVF.AVMediaTypeVideo, callback)


class _Delegate(NSObject, protocols=[objc.protocolNamed("AVCaptureVideoDataOutputSampleBufferDelegate")]):
    def initWithCamera_(self, cam):
        self = objc.super(_Delegate, self).init()
        if self is None:
            return None
        self.cam = cam
        return self

    def captureOutput_didOutputSampleBuffer_fromConnection_(self, output, sample, connection):
        self.cam._on_sample(sample)


class Camera:
    def __init__(self, preferred=None):
        self.device = pick_camera(preferred)
        if self.device is None:
            raise RuntimeError("Nie znaleziono kamery")
        self.name = str(self.device.localizedName())
        self.cond = threading.Condition()
        self.frame = None
        self.seq = 0
        self.session = None

    def start(self):
        session = AVF.AVCaptureSession.alloc().init()
        if session.canSetSessionPreset_(AVF.AVCaptureSessionPreset1280x720):
            session.setSessionPreset_(AVF.AVCaptureSessionPreset1280x720)
        inp, err = AVF.AVCaptureDeviceInput.deviceInputWithDevice_error_(self.device, None)
        if inp is None:
            raise RuntimeError(f"Kamera niedostępna: {err}")
        session.addInput_(inp)
        self.lock_frame_rate(30, 24)
        out = AVF.AVCaptureVideoDataOutput.alloc().init()
        out.setVideoSettings_({Quartz.kCVPixelBufferPixelFormatTypeKey: Quartz.kCVPixelFormatType_32BGRA})
        out.setAlwaysDiscardsLateVideoFrames_(True)
        self.queue = dispatch.dispatch_queue_create(b"fun.gzowo.vanish.camera", None)
        self.delegate = _Delegate.alloc().initWithCamera_(self)
        out.setSampleBufferDelegate_queue_(self.delegate, self.queue)
        session.addOutput_(out)
        center = NSNotificationCenter.defaultCenter()
        self.observers = []
        for name in ("AVCaptureSessionRuntimeErrorNotification", "AVCaptureSessionWasInterruptedNotification", "AVCaptureSessionInterruptionEndedNotification"):
            obs = center.addObserverForName_object_queue_usingBlock_(name, session, None, self._on_note)
            self.observers.append(obs)
        session.startRunning()
        self.session = session
        self.output = out

    def lock_frame_rate(self, fast, slow):
        dev = self.device
        try:
            ranges = dev.activeFormat().videoSupportedFrameRateRanges()
            if not any(r.minFrameRate() <= slow and r.maxFrameRate() >= fast for r in ranges):
                return False
            ok, _ = dev.lockForConfiguration_(None)
            if not ok:
                return False
            dev.setActiveVideoMinFrameDuration_(CoreMedia.CMTimeMake(1, fast))
            dev.setActiveVideoMaxFrameDuration_(CoreMedia.CMTimeMake(1, slow))
            dev.unlockForConfiguration()
            return True
        except Exception as e:
            print(time.strftime("%H:%M:%S"), "kamera: nie ustawię kl/s:", e, flush=True)
            return False

    def _on_note(self, note):
        info = note.userInfo()
        print(time.strftime("%H:%M:%S"), "kamera:", str(note.name()), dict(info) if info else "", flush=True)

    def _on_sample(self, sample):
        pb = CoreMedia.CMSampleBufferGetImageBuffer(sample)
        if pb is None:
            return
        Quartz.CVPixelBufferLockBaseAddress(pb, Quartz.kCVPixelBufferLock_ReadOnly)
        try:
            w = Quartz.CVPixelBufferGetWidth(pb)
            h = Quartz.CVPixelBufferGetHeight(pb)
            bpr = Quartz.CVPixelBufferGetBytesPerRow(pb)
            buf = Quartz.CVPixelBufferGetBaseAddress(pb).as_buffer(bpr * h)
            arr = np.frombuffer(buf, np.uint8).reshape(h, bpr)[:, : w * 4].reshape(h, w, 4)
            rgb = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
        finally:
            Quartz.CVPixelBufferUnlockBaseAddress(pb, Quartz.kCVPixelBufferLock_ReadOnly)
        with self.cond:
            self.frame = rgb
            self.seq += 1
            self.cond.notify_all()

    def read(self, last_seq, timeout=1.0):
        with self.cond:
            self.cond.wait_for(lambda: self.seq != last_seq, timeout)
            return self.frame, self.seq

    def stop(self):
        if self.session is not None:
            center = NSNotificationCenter.defaultCenter()
            for obs in getattr(self, "observers", []):
                center.removeObserver_(obs)
            self.session.stopRunning()
            self.session = None
