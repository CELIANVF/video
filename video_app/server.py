"""Serveur TCP : plusieurs flux (réseau + caméras locales optionnelles)."""

from __future__ import annotations

import math
import socket
import threading
import time
from collections import OrderedDict
from typing import Callable

import cv2
import numpy as np
import screeninfo

from video_app.buffer import StreamBuffer
from video_app.capture import capture_resized
from video_app.protocol import read_line, recv_jpeg_frame


class StreamRegistry:
    def __init__(self, frame_rate: int, buffer_duration: int):
        self._lock = threading.Lock()
        self._streams: OrderedDict[str, StreamBuffer] = OrderedDict()
        self.frame_rate = frame_rate
        self.buffer_duration = buffer_duration

    def get_or_create(self, stream_id: str) -> StreamBuffer:
        with self._lock:
            if stream_id not in self._streams:
                self._streams[stream_id] = StreamBuffer(
                    stream_id, self.frame_rate, self.buffer_duration
                )
            return self._streams[stream_id]

    def remove(self, stream_id: str) -> None:
        with self._lock:
            self._streams.pop(stream_id, None)

    def ids(self) -> list[str]:
        with self._lock:
            return list(self._streams.keys())

    def get(self, stream_id: str) -> StreamBuffer | None:
        with self._lock:
            return self._streams.get(stream_id)

    def all_buffers(self) -> list[StreamBuffer]:
        with self._lock:
            return list(self._streams.values())

    def set_all_buffer_duration(self, seconds: int) -> None:
        with self._lock:
            for b in self._streams.values():
                b.set_buffer_duration(seconds)
            self.buffer_duration = seconds


def _client_loop(
    conn: socket.socket,
    addr,
    registry: StreamRegistry,
    stop_event: threading.Event,
    log: Callable[[str], None],
) -> None:
    stream_id = f"net_{addr[0]}_{addr[1]}"
    try:
        line = read_line(conn)
        if line.upper().startswith("CAMERA "):
            name = line[7:].strip()
            if name:
                stream_id = name
    except (ConnectionError, ValueError, OSError) as e:
        log(f"client {addr}: en-tête invalide ({e})")
        return

    buf = registry.get_or_create(stream_id)
    log(f"flux connecté: {stream_id} depuis {addr}")
    try:
        while not stop_event.is_set():
            try:
                jpeg = recv_jpeg_frame(conn)
            except (ConnectionError, ValueError, OSError):
                break
            arr = np.frombuffer(jpeg, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                buf.append(frame)
    finally:
        registry.remove(stream_id)
        log(f"flux déconnecté: {stream_id}")
        conn.close()


def _local_camera_loop(
    device: int | str,
    stream_id: str,
    registry: StreamRegistry,
    target_w: int,
    target_h: int,
    stop_event: threading.Event,
    log: Callable[[str], None],
) -> None:
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        log(f"caméra locale {stream_id}: impossible d’ouvrir {device!r}")
        return
    buf = registry.get_or_create(stream_id)
    log(f"caméra locale démarrée: {stream_id} ({device})")
    try:
        while not stop_event.is_set():
            frame = capture_resized(cap, target_w, target_h)
            if frame is not None:
                buf.append(frame)
            time.sleep(0.001)
    finally:
        cap.release()
        registry.remove(stream_id)
        log(f"caméra locale arrêtée: {stream_id}")


def _make_grid(
    frames: list[tuple[str, np.ndarray]],
    cell_w: int,
    cell_h: int,
) -> np.ndarray | None:
    if not frames:
        return None
    n = len(frames)
    cols = max(1, math.ceil(math.sqrt(n)))
    rows = math.ceil(n / cols)
    grid_h = rows * cell_h
    grid_w = cols * cell_w
    canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    for i, (name, frame) in enumerate(frames):
        r, c = divmod(i, cols)
        if frame is None:
            continue
        h, w = frame.shape[:2]
        scale = min(cell_w / w, cell_h / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        small = cv2.resize(frame, (nw, nh))
        y0 = r * cell_h + (cell_h - nh) // 2
        x0 = c * cell_w + (cell_w - nw) // 2
        canvas[y0 : y0 + nh, x0 : x0 + nw] = small
        cv2.putText(
            canvas,
            name[:32],
            (x0 + 4, y0 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def run_server(
    host: str = "0.0.0.0",
    port: int = 8765,
    frame_rate: int = 30,
    buffer_duration: int = 5,
    local_devices: list[tuple[str, int | str]] | None = None,
) -> None:
    """
    local_devices: liste de (stream_id, device) ex. [("local_0", 0)].
    """
    local_devices = local_devices or []
    stop_event = threading.Event()
    registry = StreamRegistry(frame_rate, buffer_duration)

    try:
        mon = screeninfo.get_monitors()[0]
        screen_w, screen_h = mon.width, mon.height
    except Exception:
        screen_w, screen_h = 1280, 720

    # Grille : utiliser une portion de l’écran pour laisser la marge HUD
    grid_max_w = min(screen_w - 80, 1600)
    grid_max_h = min(screen_h - 120, 900)

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((host, port))
    server_sock.listen(16)
    server_sock.settimeout(0.5)

    def log(msg: str) -> None:
        print(f"[serveur] {msg}")

    threads: list[threading.Thread] = []

    for sid, dev in local_devices:
        t = threading.Thread(
            target=_local_camera_loop,
            args=(dev, sid, registry, grid_max_w, grid_max_h, stop_event, log),
            daemon=True,
        )
        t.start()
        threads.append(t)

    def accept_loop() -> None:
        while not stop_event.is_set():
            try:
                conn, addr = server_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            t = threading.Thread(
                target=_client_loop,
                args=(conn, addr, registry, stop_event, log),
                daemon=True,
            )
            t.start()

    accept_thread = threading.Thread(target=accept_loop, daemon=True)
    accept_thread.start()
    log(f"écoute sur {host}:{port} — lancez camera.py sur les machines caméra")

    buffer_duration_live = buffer_duration
    selected_index = 0
    real_fps_count = 0
    last_fps_time = time.time()
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)

    try:
        window_sized = False
        while True:
            loop_start = time.time()
            ids = registry.ids()
            frames_data: list[tuple[str, np.ndarray]] = []
            for sid in ids:
                b = registry.get(sid)
                if b is None:
                    continue
                fr = b.latest()
                if fr is not None:
                    frames_data.append((sid, fr))

            n = max(1, len(frames_data))
            cols = max(1, math.ceil(math.sqrt(n)))
            rows = math.ceil(n / cols)
            cell_w = max(160, grid_max_w // cols)
            cell_h = max(120, grid_max_h // rows)

            grid = _make_grid(frames_data, cell_w, cell_h)
            if grid is None:
                grid = np.zeros((grid_max_h // 2, grid_max_w // 2, 3), dtype=np.uint8)
                cv2.putText(
                    grid,
                    "En attente de flux (camera.py ou --local)",
                    (20, grid.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA,
                )

            hud_w = max(280, int(grid.shape[1] * 0.22))
            hud = np.zeros((grid.shape[0], hud_w, 3), dtype=np.uint8)
            wider = np.zeros((grid.shape[0], grid.shape[1] + hud_w, 3), dtype=np.uint8)
            wider[:, : grid.shape[1]] = grid
            wider[:, grid.shape[1] :] = hud

            sel_label = ""
            if ids:
                selected_index = min(selected_index, len(ids) - 1)
                sel_label = ids[selected_index]
            lines = [
                f"FPS cible: {frame_rate}",
                f"Tampon: {buffer_duration_live} s",
                f"Sélection: {sel_label or '-'}",
                "",
                "Touches:",
                "1-9: choisir flux",
                "+/-: durée tampon",
                "s: enregistrer flux sélectionné",
                "q: quitter",
            ]
            hx = grid.shape[1] + 10
            for i, line in enumerate(lines):
                cv2.putText(
                    wider,
                    line,
                    (hx, 28 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (240, 240, 240),
                    1,
                    cv2.LINE_AA,
                )

            real_fps_count += 1
            if time.time() - last_fps_time >= 1.0:
                print(f"FPS affichage: {real_fps_count}")
                real_fps_count = 0
                last_fps_time = time.time()

            cv2.imshow("Frame", wider)
            if not window_sized:
                cv2.resizeWindow("Frame", wider.shape[1], wider.shape[0])
                window_sized = True

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s") and sel_label:
                b = registry.get(sel_label)
                if b is not None:

                    def _save() -> None:
                        b.save_last_seconds()

                    threading.Thread(target=_save, daemon=True).start()
            elif key == ord("+"):
                buffer_duration_live += 1
                registry.set_all_buffer_duration(buffer_duration_live)
            elif key == ord("-") and buffer_duration_live > 1:
                buffer_duration_live -= 1
                registry.set_all_buffer_duration(buffer_duration_live)
            elif ord("1") <= key <= ord("9"):
                idx = key - ord("1")
                if idx < len(ids):
                    selected_index = idx

            elapsed = time.time() - loop_start
            time.sleep(max(1 / frame_rate - elapsed, 0))

    finally:
        stop_event.set()
        try:
            server_sock.close()
        except OSError:
            pass
        cv2.destroyAllWindows()
