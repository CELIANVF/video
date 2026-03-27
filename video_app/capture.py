"""Capture locale et redimensionnement."""

import cv2


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
