"""Serveur TCP : plusieurs flux (réseau + caméras locales optionnelles)."""

from __future__ import annotations

import math
import os
import queue
import socket
import threading
import time
from collections import OrderedDict
from typing import Callable

import cv2
import numpy as np
import screeninfo

from video_app import ffmpeg_io
from video_app.buffer import StreamBuffer
from video_app.capture import capture_resized, configure_webcam_best_effort
from video_app.fast_jpeg import decode_jpeg_bgr, turbojpeg_available
from video_app.display_core import (
    close_continuous_stack_state,
    gather_display_frames,
    tick_continuous_stack_recording,
)
from video_app.export_video import save_per_stream_and_stack
from video_app.protocol import (
    PT_AUDIO,
    PT_VIDEO,
    peel_transport,
    read_line,
    recv_jpeg_frame,
    recv_v2_packet,
)


class StreamRegistry:
    def __init__(self, frame_rate: int, buffer_duration: int):
        self._lock = threading.Lock()
        self._streams: OrderedDict[str, StreamBuffer] = OrderedDict()
        self.frame_rate = frame_rate
        self.buffer_duration = buffer_duration
        self._continuous_active = False
        self._continuous_session_ts = 0

    def get_or_create(self, stream_id: str) -> StreamBuffer:
        with self._lock:
            if stream_id not in self._streams:
                b = StreamBuffer(
                    stream_id, self.frame_rate, self.buffer_duration
                )
                if self._continuous_active:
                    b.start_continuous(self._continuous_session_ts)
                self._streams[stream_id] = b
            return self._streams[stream_id]

    def remove(self, stream_id: str) -> None:
        with self._lock:
            b = self._streams.pop(stream_id, None)
        if b is not None:
            b.stop_continuous()

    def is_continuous_recording(self) -> bool:
        with self._lock:
            return self._continuous_active

    def get_continuous_session_ts(self) -> int:
        with self._lock:
            return self._continuous_session_ts

    def set_continuous_recording(self, active: bool) -> None:
        with self._lock:
            self._continuous_active = active
            if active:
                self._continuous_session_ts = int(time.time())
            buffers = list(self._streams.values())
            session_ts = self._continuous_session_ts
        for buf in buffers:
            if active:
                buf.start_continuous(session_ts)
            else:
                buf.stop_continuous()
        if active:
            print(
                f"[serveur] Enregistrement continu ACTIF "
                f"(session {session_ts}) — touche r pour arrêter"
            )
        else:
            print("[serveur] Enregistrement continu désactivé")

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


def _net_decode_worker_count() -> int:
    n = os.cpu_count() or 4
    return min(8, max(2, n))


class _StreamPacketMerger:
    """Réordonne vidéo (décodage JPEG parallèle) et audio pour des append séquentiels."""

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._audio: dict[int, bytes] = {}
        self._video: dict[int, np.ndarray | None] = {}
        self._end_seq: int | None = None

    def add_audio(self, seq: int, pcm: bytes) -> None:
        with self._cv:
            self._audio[seq] = pcm
            self._cv.notify_all()

    def add_video(self, seq: int, frame: np.ndarray | None) -> None:
        with self._cv:
            self._video[seq] = frame
            self._cv.notify_all()

    def set_stream_end(self, max_seq: int) -> None:
        with self._cv:
            self._end_seq = max_seq
            self._cv.notify_all()

    def run(
        self,
        buf: StreamBuffer,
        fps_diag: _NetFpsDiag | None = None,
    ) -> None:
        next_seq = 1
        while True:
            kind = ""
            pcm_out: bytes | None = None
            video_out: np.ndarray | None = None
            with self._cv:
                while True:
                    if self._end_seq is not None and next_seq > self._end_seq:
                        return
                    if next_seq in self._audio:
                        pcm_out = self._audio.pop(next_seq)
                        next_seq += 1
                        kind = "a"
                        break
                    if next_seq in self._video:
                        video_out = self._video.pop(next_seq)
                        next_seq += 1
                        kind = "v"
                        break
                    self._cv.wait()
            if kind == "a" and pcm_out is not None:
                if fps_diag:
                    fps_diag.bump("append_a")
                buf.append_audio(pcm_out)
            elif kind == "v" and video_out is not None:
                if fps_diag:
                    fps_diag.bump("append_v")
                buf.append(video_out, copy_frame=False)


class _NetFpsDiag:
    """Compteurs thread-safe + impression 1 Hz des étapes réseau → tampon."""

    def __init__(self, label: str) -> None:
        self._label = label
        self._lock = threading.Lock()
        self._c = {
            "recv_pkt": 0,
            "recv_vid": 0,
            "recv_aud": 0,
            "dispatch_vid": 0,
            "decode": 0,
            "append_v": 0,
            "append_a": 0,
        }
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def bump(self, key: str, n: int = 1) -> None:
        with self._lock:
            self._c[key] = self._c.get(key, 0) + n

    def _loop(self) -> None:
        while not self._stop.wait(1.0):
            with self._lock:
                s = self._c.copy()
                for k in self._c:
                    self._c[k] = 0
            print(
                f"[debug-fps serveur {self._label}] "
                f"recv_pkt={s['recv_pkt']}/s vid={s['recv_vid']}/s aud={s['recv_aud']}/s | "
                f"dispatch_vid={s['dispatch_vid']}/s decode={s['decode']}/s | "
                f"append_vid={s['append_v']}/s append_aud={s['append_a']}/s"
            )

    def close(self) -> None:
        self._stop.set()


def _client_loop(
    conn: socket.socket,
    addr,
    registry: StreamRegistry,
    stop_event: threading.Event,
    log: Callable[[str], None],
    debug_fps: bool = False,
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
    recv_thread: threading.Thread | None = None
    dispatch_thread: threading.Thread | None = None
    worker_threads: list[threading.Thread] = []
    fps_diag: _NetFpsDiag | None = None
    try:
        mode, extra = peel_transport(conn)
        if debug_fps:
            fps_diag = _NetFpsDiag(stream_id)
        if not turbojpeg_available():
            log(
                f"flux {stream_id}: JPEG sans TurboJPEG (plus lent) — "
                "PyTurboJPEG + libturbojpeg recommandés"
            )
        # Réception TCP, dispatch ordonné, décodage JPEG sur plusieurs fils, fusion par n° de séquence.
        ordered_q: queue.Queue[object] = queue.Queue(maxsize=128)
        work_q: queue.Queue[object] = queue.Queue(maxsize=64)
        merger = _StreamPacketMerger()
        n_workers = _net_decode_worker_count()

        def _dispatch_ordered() -> None:
            while True:
                item = ordered_q.get()
                if isinstance(item, tuple) and len(item) == 2 and item[0] is None:
                    _, max_seq = item
                    merger.set_stream_end(int(max_seq))
                    for _ in range(n_workers):
                        work_q.put(None)
                    break
                if not isinstance(item, tuple) or len(item) != 3:
                    continue
                seq_i, typ_i, data_i = item
                seq_i = int(seq_i)
                if typ_i == PT_VIDEO:
                    if fps_diag:
                        fps_diag.bump("dispatch_vid")
                    work_q.put((seq_i, data_i))
                elif typ_i == PT_AUDIO:
                    merger.add_audio(seq_i, data_i)

        def _decode_worker() -> None:
            while True:
                job = work_q.get()
                if job is None:
                    break
                seq_j, jpeg_b = job  # type: ignore[misc]
                fr = decode_jpeg_bgr(jpeg_b)
                if fps_diag:
                    fps_diag.bump("decode")
                merger.add_video(int(seq_j), fr)

        def _recv_legacy(jpeg0: bytes) -> None:
            last_seq = 1
            try:
                if fps_diag:
                    fps_diag.bump("recv_pkt")
                    fps_diag.bump("recv_vid")
                ordered_q.put((1, PT_VIDEO, jpeg0))
                seq = 2
                while not stop_event.is_set():
                    jpeg = recv_jpeg_frame(conn)
                    if fps_diag:
                        fps_diag.bump("recv_pkt")
                        fps_diag.bump("recv_vid")
                    ordered_q.put((seq, PT_VIDEO, jpeg))
                    last_seq = seq
                    seq += 1
            except (ConnectionError, ValueError, OSError):
                pass
            finally:
                ordered_q.put((None, last_seq))

        def _recv_v2() -> None:
            last_seq = 0
            try:
                s = 1
                while not stop_event.is_set():
                    typ, data = recv_v2_packet(conn)
                    if fps_diag:
                        fps_diag.bump("recv_pkt")
                        if typ == PT_VIDEO:
                            fps_diag.bump("recv_vid")
                        elif typ == PT_AUDIO:
                            fps_diag.bump("recv_aud")
                    ordered_q.put((s, typ, data))
                    last_seq = s
                    s += 1
            except (ConnectionError, ValueError, OSError):
                pass
            finally:
                ordered_q.put((None, last_seq))

        dispatch_thread = threading.Thread(target=_dispatch_ordered, daemon=True)
        dispatch_thread.start()
        for _ in range(n_workers):
            wt = threading.Thread(target=_decode_worker, daemon=True)
            worker_threads.append(wt)
            wt.start()

        if mode == "legacy":
            jpeg0: bytes = extra
            recv_thread = threading.Thread(
                target=_recv_legacy, args=(jpeg0,), daemon=True
            )
            recv_thread.start()
            merger.run(buf, fps_diag)
        else:
            audio_params = extra
            if audio_params is not None:
                buf.set_audio_params(audio_params[0], audio_params[1])
                log(
                    f"flux {stream_id}: audio {audio_params[0]} Hz, "
                    f"{audio_params[1]} canal(aux)"
                )
            recv_thread = threading.Thread(target=_recv_v2, daemon=True)
            recv_thread.start()
            merger.run(buf, fps_diag)
    finally:
        if fps_diag is not None:
            fps_diag.close()
        try:
            conn.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        if recv_thread is not None:
            recv_thread.join(timeout=3.0)
        if dispatch_thread is not None:
            dispatch_thread.join(timeout=3.0)
        for wt in worker_threads:
            wt.join(timeout=3.0)
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
    ew, eh, _ = configure_webcam_best_effort(
        cap, apply_resolution=True, apply_fps=True
    )
    buf = registry.get_or_create(stream_id)
    dim = f"{ew}×{eh} px" if ew > 0 and eh > 0 else "résolution inconnue"
    log(f"caméra locale {stream_id} ({device}) — {dim}")
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
    gui: bool = False,
    debug_fps: bool = False,
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
                args=(conn, addr, registry, stop_event, log, debug_fps),
                daemon=True,
            )
            t.start()

    accept_thread = threading.Thread(target=accept_loop, daemon=True)
    accept_thread.start()
    log(f"écoute sur {host}:{port} — lancez camera.py sur les machines caméra")
    if not ffmpeg_io.ffmpeg_available():
        log(
            "ffmpeg introuvable (PATH ou « pip install imageio-ffmpeg ») : "
            "REC en AVI OpenCV avec FPS mesuré sur le flux."
        )

    buffer_duration_live = buffer_duration
    live_display = True
    display_delay_sec = float(max(1, min(5, buffer_duration)))
    real_fps_count = 0
    last_fps_time = time.time()

    stack_state: dict = {"writer": None, "path": None, "bound_session": None}

    def close_continuous_stack_writer() -> None:
        close_continuous_stack_state(stack_state)

    if gui:
        try:
            from video_app.qt_gui import run_qt_application
        except ImportError as e:
            print(
                "Interface graphique indisponible. Installez PyQt6 : pip install PyQt6\n"
                f"({e})"
            )
        else:

            def _shutdown_socket() -> None:
                try:
                    server_sock.close()
                except OSError:
                    pass

            try:
                run_qt_application(
                    registry=registry,
                    frame_rate=frame_rate,
                    buffer_duration=buffer_duration,
                    grid_max_w=grid_max_w,
                    grid_max_h=grid_max_h,
                    stop_event=stop_event,
                    stack_state=stack_state,
                    on_shutdown=_shutdown_socket,
                )
            finally:
                close_continuous_stack_writer()
                registry.set_continuous_recording(False)
                stop_event.set()
                try:
                    server_sock.close()
                except OSError:
                    pass
            return

    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)

    try:
        window_sized = False
        while True:
            loop_start = time.time()
            frames_data = gather_display_frames(
                registry, live_display, display_delay_sec
            )
            tick_continuous_stack_recording(
                registry, frames_data, frame_rate, stack_state
            )
            ids = registry.ids()

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

            mode_label = "LIVE" if live_display else f"DÉLAI {display_delay_sec:.0f}s"
            mode_color = (80, 255, 80) if live_display else (80, 180, 255)
            cv2.putText(
                wider,
                mode_label,
                (12, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                mode_color,
                2,
                cv2.LINE_AA,
            )
            if registry.is_continuous_recording():
                cv2.putText(
                    wider,
                    "REC",
                    (12, 72),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            lines = [
                f"FPS cible: {frame_rate}",
                f"Tampon: {buffer_duration_live} s",
                f"Flux actifs: {len(ids)}",
                f"Affichage: {'direct' if live_display else 'retardé'}",
                "",
                "Touches:",
                "m: live / délai",
                "[ ] ou , .: délai ±1s",
                "s: AVI par flux + empilée",
                "r: REC continu (par flux + empilé si 2+)",
                "+/-: durée tampon",
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
            if key == ord("r"):
                was_rec = registry.is_continuous_recording()
                registry.set_continuous_recording(not was_rec)
                if was_rec:
                    close_continuous_stack_writer()
            if key == ord("s"):
                # Ordre registry.ids() = bandeau 0 en haut, 1 en dessous, etc. dans stack_*.avi
                to_save = []
                for sid in registry.ids():
                    b = registry.get(sid)
                    if b is not None:
                        to_save.append(b)
                if to_save:
                    fps = frame_rate

                    def _save_all() -> None:
                        save_per_stream_and_stack(to_save, fps)

                    threading.Thread(target=_save_all, daemon=True).start()
            elif key == ord("m"):
                live_display = not live_display
            elif key in (ord("["), ord(",")):
                display_delay_sec = max(1.0, display_delay_sec - 1.0)
            elif key in (ord("]"), ord(".")):
                display_delay_sec = min(
                    float(buffer_duration_live), display_delay_sec + 1.0
                )
            elif key == ord("+"):
                buffer_duration_live += 1
                registry.set_all_buffer_duration(buffer_duration_live)
                display_delay_sec = min(display_delay_sec, float(buffer_duration_live))
            elif key == ord("-") and buffer_duration_live > 1:
                buffer_duration_live -= 1
                registry.set_all_buffer_duration(buffer_duration_live)
                display_delay_sec = min(display_delay_sec, float(buffer_duration_live))
                display_delay_sec = max(1.0, display_delay_sec)

            elapsed = time.time() - loop_start
            time.sleep(max(1 / frame_rate - elapsed, 0))

    finally:
        close_continuous_stack_writer()
        registry.set_continuous_recording(False)
        stop_event.set()
        try:
            server_sock.close()
        except OSError:
            pass
        cv2.destroyAllWindows()
