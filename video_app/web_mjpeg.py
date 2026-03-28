"""Serveur HTTP minimal : liste des flux, MJPEG par flux, page d’accueil."""

from __future__ import annotations

import html
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np


def _jpeg_bytes(bgr: np.ndarray, quality: int = 85) -> bytes | None:
    ok, enc = cv2.imencode(
        ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), max(1, min(100, quality))]
    )
    if not ok:
        return None
    return enc.tobytes()


class MJPEGHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        registry: Any,
        path_prefix: str = "",
    ) -> None:
        self.registry = registry
        self.path_prefix = (path_prefix or "").rstrip("/")
        super().__init__(server_address, MJPEGRequestHandler)


class MJPEGRequestHandler(BaseHTTPRequestHandler):
    server: MJPEGHTTPServer

    def log_message(self, fmt: str, *args: Any) -> None:
        logging.getLogger("video_app.web").info(
            "%s %s", self.address_string(), fmt % args
        )

    def _full_path(self) -> str:
        p = urlparse(self.path).path
        pre = self.server.path_prefix
        if pre and (p == pre or p.startswith(pre + "/")):
            p = p[len(pre) :] or "/"
        return p if p.startswith("/") else "/" + p

    def do_GET(self) -> None:  # noqa: N802
        path = self._full_path()
        if path == "/api/streams":
            self._send_streams_json()
            return
        if path.startswith("/mjpeg/"):
            sid = path[len("/mjpeg/") :].strip("/")
            if sid:
                self._stream_mjpeg(sid)
                return
        if path in ("/", ""):
            self._send_index_html()
            return
        if path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        self.send_error(404, "Not Found")

    def _send_streams_json(self) -> None:
        ids = self.server.registry.ids()
        body = json.dumps({"streams": ids}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_index_html(self) -> None:
        ids = self.server.registry.ids()
        pre = self.server.path_prefix or ""
        parts = [
            "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Flux vidéo</title>",
            "<style>body{font-family:sans-serif;background:#111;color:#eee;padding:16px;}",
            "img{max-width:100%;border:1px solid #444;margin:8px 0;background:#000}</style></head><body>",
            "<h1>Flux</h1>",
        ]
        if not ids:
            parts.append("<p>Aucun flux connecté.</p>")
        for sid in ids:
            esc = html.escape(sid, quote=True)
            src = html.escape(f"{pre}/mjpeg/{sid}", quote=True)
            parts.append(f"<h2>{esc}</h2><img src='{src}' alt='{esc}' />")
        parts.append("</body></html>")
        body = "".join(parts).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _stream_mjpeg(self, stream_id: str) -> None:
        boundary = b"frame"
        self.send_response(200)
        self.send_header(
            "Content-Type",
            f"multipart/x-mixed-replace; boundary={boundary.decode('ascii')}",
        )
        self.send_header("Cache-Control", "no-cache, no-store")
        self.end_headers()
        buf = self.server.registry.get(stream_id)
        try:
            while buf is not None:
                fr = buf.latest()
                if fr is not None:
                    # Copie : latest() ne garde pas le verrou ; imencode en parallèle
                    # des append réseau pourrait sinon toucher un buffer réutilisé.
                    fr = np.ascontiguousarray(fr, dtype=np.uint8).copy()
                    jb = _jpeg_bytes(fr)
                    if jb:
                        try:
                            self.wfile.write(b"--" + boundary + b"\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n")
                            self.wfile.write(f"Content-Length: {len(jb)}\r\n\r\n".encode())
                            self.wfile.write(jb)
                            self.wfile.write(b"\r\n")
                        except (BrokenPipeError, ConnectionResetError, OSError):
                            break
                time.sleep(max(1.0 / 30.0, 0.02))
                buf = self.server.registry.get(stream_id)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass


def start_web_server(
    host: str,
    port: int,
    registry: Any,
    path_prefix: str = "",
) -> tuple[threading.Thread, MJPEGHTTPServer]:
    srv = MJPEGHTTPServer((host, port), registry, path_prefix=path_prefix)

    def _run() -> None:
        try:
            srv.serve_forever(poll_interval=0.5)
        except Exception:
            logging.getLogger("video_app.web").exception("serveur web arrêté")

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    return th, srv
