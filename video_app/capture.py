"""Capture locale et redimensionnement (OpenCV)."""

from __future__ import annotations

import cv2

# Ordre aire décroissante (meilleur effort V4L2 / backends OpenCV).
_RESOLUTION_CANDIDATES: list[tuple[int, int]] = [
    (3840, 2160),
    (2560, 1920),
    (2560, 1440),
    (1920, 1080),
    (1600, 1200),
    (1280, 720),
    (1024, 768),
    (800, 600),
    (640, 480),
]


def is_local_opencv_capture_device(device: int | str) -> bool:
    """Index ou chemin sans schéma URL (webcam locale, pas RTSP/http)."""
    if isinstance(device, int):
        return True
    s = str(device).strip().lower()
    return "://" not in s


def _read_frame_size(cap: cv2.VideoCapture) -> tuple[int, int]:
    ok, frame = cap.read()
    if not ok or frame is None:
        return 0, 0
    h, w = frame.shape[:2]
    return w, h


def _warmup_reads(cap: cv2.VideoCapture, n: int = 4) -> None:
    for _ in range(n):
        cap.read()


def configure_webcam_for_send_size(
    cap: cv2.VideoCapture,
    min_w: int,
    min_h: int,
    *,
    apply_fps: bool,
) -> tuple[int, int, float]:
    """
    Règle MJPEG au plus petit format courant couvrant min_w×min_h (aire croissante).
    Évite de capturer en 5 Mpx puis redimensionner : le goulot est souvent cap.read() USB.
    """
    min_w = max(1, int(min_w))
    min_h = max(1, int(min_h))
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    best_aw, best_ah = 0, 0
    for w, h in sorted(_RESOLUTION_CANDIDATES, key=lambda wh: wh[0] * wh[1]):
        if w < min_w or h < min_h:
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        _warmup_reads(cap, 3)
        aw, ah = _read_frame_size(cap)
        if aw >= min_w and ah >= min_h:
            best_aw, best_ah = aw, ah
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, aw)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ah)
            _warmup_reads(cap, 2)
            break
    if best_aw <= 0:
        return configure_webcam_best_effort(
            cap, apply_resolution=True, apply_fps=apply_fps
        )
    if apply_fps:
        cap.set(cv2.CAP_PROP_FPS, 240.0)
        _warmup_reads(cap, 3)
    ok, last = cap.read()
    if ok and last is not None:
        lh, lw = last.shape[:2]
        pw, ph = int(lw), int(lh)
    else:
        pw, ph = best_aw, best_ah
    pfps = float(cap.get(cv2.CAP_PROP_FPS))
    return (pw, ph, pfps)


def configure_webcam_best_effort(
    cap: cv2.VideoCapture,
    *,
    apply_resolution: bool,
    apply_fps: bool,
) -> tuple[int, int, float]:
    """
    Résolution / FPS au mieux. Beaucoup de webcams USB : le 1920×1080 n’est disponible
    qu’en MJPEG ; sans cela le pilote reste souvent en YUYV 640×480. On se fie à la taille
    réelle du dernier read(), pas seulement à cap.get().
    """
    pw = ph = 0
    pfps = 0.0

    if not apply_resolution and not apply_fps:
        if cap.isOpened():
            pw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            ph = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            pfps = float(cap.get(cv2.CAP_PROP_FPS))
        return (pw, ph, pfps)

    if apply_resolution:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        best_w, best_h = 0, 0
        for w, h in _RESOLUTION_CANDIDATES:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            _warmup_reads(cap)
            aw, ah = _read_frame_size(cap)
            if aw > 0 and ah > 0 and aw * ah > best_w * best_h:
                best_w, best_h = aw, ah
        if best_w > 0 and best_h > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, best_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, best_h)
            _warmup_reads(cap)

    if apply_fps:
        cap.set(cv2.CAP_PROP_FPS, 240.0)
        _warmup_reads(cap, 3)

    ok, last = cap.read()
    if ok and last is not None:
        lh, lw = last.shape[:2]
        pw, ph = int(lw), int(lh)
    else:
        pw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ph = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pfps = float(cap.get(cv2.CAP_PROP_FPS))
    return (pw, ph, pfps)


def capture_resized(cap, target_width: int, target_height: int):
    ret, frame = cap.read()
    if not ret:
        return None
    h, w = frame.shape[:2]
    aspect = w / h
    if target_width / target_height > aspect:
        nw = int(target_height * aspect)
        nh = target_height
    else:
        nw = target_width
        nh = int(target_width / aspect)
    return cv2.resize(frame, (nw, nh))
