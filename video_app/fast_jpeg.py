"""JPEG BGR rapide (libjpeg-turbo / PyTurboJPEG) avec repli OpenCV."""

from __future__ import annotations

import cv2
import numpy as np

_TJ_ENC = None
_TJ_DEC = None
_TJPF_BGR = None


def _ensure_turbo() -> tuple[object, object] | None:
    """Retourne (TurboJPEG, TJPF_BGR) ou None."""
    global _TJ_ENC, _TJ_DEC, _TJPF_BGR
    if _TJ_ENC is False:
        return None
    if _TJ_ENC is not None and _TJPF_BGR is not None:
        return (_TJ_ENC, _TJPF_BGR)
    try:
        from turbojpeg import TJPF_BGR, TurboJPEG

        _TJ_ENC = TurboJPEG()
        _TJ_DEC = _TJ_ENC
        _TJPF_BGR = TJPF_BGR
        return (_TJ_ENC, _TJPF_BGR)
    except Exception:
        _TJ_ENC = False
        _TJ_DEC = False
        _TJPF_BGR = False
        return None


def turbojpeg_available() -> bool:
    return _ensure_turbo() is not None


def encode_bgr_jpeg(frame: np.ndarray, quality: int) -> bytes | None:
    """BGR uint8 HWC → JPEG. None → utiliser encode_bgr_jpeg_cv2."""
    t = _ensure_turbo()
    if t is None:
        return None
    tj, pix = t
    q = max(1, min(100, int(quality)))
    try:
        return tj.encode(
            np.ascontiguousarray(frame, dtype=np.uint8),
            quality=q,
            pixel_format=pix,
        )
    except Exception:
        return None


def encode_bgr_jpeg_cv2(frame: np.ndarray, quality: int) -> bytes | None:
    params = [int(cv2.IMWRITE_JPEG_QUALITY), max(1, min(100, quality))]
    ok, jpeg = cv2.imencode(".jpg", frame, params)
    if not ok:
        return None
    return jpeg.tobytes()


def encode_bgr_jpeg_best(frame: np.ndarray, quality: int) -> bytes | None:
    b = encode_bgr_jpeg(frame, quality)
    if b is not None:
        return b
    return encode_bgr_jpeg_cv2(frame, quality)


def decode_jpeg_bgr(jpeg_bytes: bytes) -> np.ndarray | None:
    t = _ensure_turbo()
    if t is not None:
        tj, pix = t
        try:
            arr = tj.decode(jpeg_bytes, pixel_format=pix)
            if arr is not None and arr.size > 0:
                return arr
        except Exception:
            pass
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)
