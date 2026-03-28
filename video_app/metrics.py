"""Métriques Prometheus (optionnel)."""

from __future__ import annotations

import logging
import threading
from typing import Any

_metrics_enabled = False
_lock = threading.Lock()
M_DECODE_ERRORS: Any = None
M_FRAMES_APPENDED: Any = None
M_STREAMS_ACTIVE: Any = None


def _ensure_metrics() -> None:
    global M_DECODE_ERRORS, M_FRAMES_APPENDED, M_STREAMS_ACTIVE
    if M_STREAMS_ACTIVE is not None:
        return
    from prometheus_client import Counter, Gauge

    M_STREAMS_ACTIVE = Gauge(
        "video_streams_active",
        "Nombre de flux connectés",
    )
    M_FRAMES_APPENDED = Counter(
        "video_frames_appended_total",
        "Images ajoutées au tampon après décodage",
        ["stream_id"],
    )
    M_DECODE_ERRORS = Counter(
        "video_jpeg_decode_errors_total",
        "Échecs de décodage JPEG",
        ["stream_id"],
    )


def start_metrics_server(host: str, port: int) -> None:
    global _metrics_enabled
    with _lock:
        if _metrics_enabled:
            return
        try:
            from prometheus_client import start_http_server
        except ImportError:
            logging.getLogger("video_app.metrics").warning(
                "prometheus_client absent : pip install prometheus_client"
            )
            return
        _ensure_metrics()
        try:
            start_http_server(port, addr=host)
        except OSError as e:
            logging.getLogger("video_app.metrics").error(
                "Impossible de démarrer /metrics sur %s:%s : %s", host, port, e
            )
            return
        _metrics_enabled = True


def bump_decode_error(stream_id: str) -> None:
    if not _metrics_enabled or M_DECODE_ERRORS is None:
        return
    M_DECODE_ERRORS.labels(stream_id=stream_id).inc()


def bump_frame_appended(stream_id: str) -> None:
    if not _metrics_enabled or M_FRAMES_APPENDED is None:
        return
    M_FRAMES_APPENDED.labels(stream_id=stream_id).inc()


def set_streams_active(n: int) -> None:
    if not _metrics_enabled or M_STREAMS_ACTIVE is None:
        return
    M_STREAMS_ACTIVE.set(n)


def metrics_loop_thread(registry: Any, stop_event: threading.Event) -> threading.Thread:
    def _loop() -> None:
        while not stop_event.wait(2.0):
            try:
                set_streams_active(len(registry.ids()))
            except Exception:
                pass

    th = threading.Thread(target=_loop, daemon=True)
    th.start()
    return th
