#!/usr/bin/env python3
"""
Client caméra : envoie un flux vidéo (webcam, RTSP, etc.) vers le serveur (main.py).
Avec --audio : protocole V2 + micro (PCM 16 bits), muxé côté serveur dans le MP4 en REC.
Débit audio par défaut : natif du micro (--audio-rate 0).
"""

from __future__ import annotations

import argparse
import logging
import queue
import select
import socket
import sys
import threading
import time
from datetime import datetime
from typing import Any, Callable

import cv2
import numpy as np

from video_app.capture import (
    configure_webcam_best_effort,
    configure_webcam_for_send_size,
    is_local_opencv_capture_device,
)
from video_app.config import ClientSettings, load_client_toml
from video_app.fast_jpeg import encode_bgr_jpeg_best, turbojpeg_available
from video_app.logutil import setup_logging
from video_app.protocol import (
    PT_AUDIO,
    PT_VIDEO,
    send_camera_header,
    send_jpeg_frame,
    send_v2_header,
    send_v2_packet,
)

try:
    import sounddevice as sd
except (ImportError, OSError):
    sd = None


class _ClientFpsDiag:
    """Impression 1 Hz : lecture caméra, encodage JPEG, envoi TCP (vidéo + paquets audio)."""

    def __init__(self, label: str) -> None:
        self._label = label
        self._lock = threading.Lock()
        self._c = {"read": 0, "encode": 0, "send_v": 0, "audio_pkt": 0}
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def bump(self, key: str, n: int = 1) -> None:
        with self._lock:
            self._c[key] = self._c.get(key, 0) + n

    def _loop(self) -> None:
        log = logging.getLogger("video_app.camera")
        while not self._stop.wait(1.0):
            with self._lock:
                s = self._c.copy()
                for k in self._c:
                    self._c[k] = 0
            log.info(
                "[debug-fps client %s] read=%s img/s | encode=%s/s | send_video=%s/s | audio_pkt=%s/s",
                self._label,
                s["read"],
                s["encode"],
                s["send_v"],
                s["audio_pkt"],
            )

    def close(self) -> None:
        self._stop.set()


def _output_dimensions(
    native_w: int, native_h: int, req_w: int, req_h: int
) -> tuple[int, int]:
    """Taille des images envoyées après redimensionnement éventuel."""
    if native_w <= 0 or native_h <= 0:
        return max(0, req_w), max(0, req_h)
    if req_w > 0 and req_h > 0:
        return req_w, req_h
    if req_w > 0:
        scale = req_w / native_w
        return req_w, max(1, int(native_h * scale))
    if req_h > 0:
        scale = req_h / native_h
        return max(1, int(native_w * scale)), req_h
    return native_w, native_h


def _overlay_time_bgr(frame: np.ndarray, stream_name: str, enabled: bool) -> np.ndarray:
    if not enabled:
        return frame
    out = frame.copy()
    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{t}  {stream_name}"
    cv2.putText(
        out,
        line,
        (8, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return out


def _open_audio_input_stream(
    preferred_sr: int,
    channels: int,
    callback: Callable[..., None],
) -> tuple[Any, int]:
    """Ouvre la capture ; essaie plusieurs débits car ALSA refuse souvent 16 kHz natif."""
    if sd is None:
        raise RuntimeError("sounddevice indisponible")
    try:
        di = sd.query_devices(kind="input")
        default_sr = int(float(di["default_samplerate"]))
    except Exception:
        default_sr = 48000
    seen: set[int] = set()
    candidates: list[int] = []
    order: list[int] = []
    if preferred_sr > 0:
        order.append(preferred_sr)
    order.extend(
        [
            default_sr,
            48000,
            44100,
            96000,
            32000,
            24000,
            22050,
            16000,
            8000,
        ]
    )
    for sr in order:
        sr = int(sr)
        if sr > 0 and sr not in seen:
            seen.add(sr)
            candidates.append(sr)
    last_err: BaseException | None = None
    for sr in candidates:
        blocksize = max(160, sr // 50)
        try:
            stream = sd.RawInputStream(
                samplerate=sr,
                channels=channels,
                dtype="int16",
                blocksize=blocksize,
                callback=callback,
            )
        except sd.PortAudioError as e:
            last_err = e
            continue
        stream.start()
        return stream, sr
    raise RuntimeError(
        "Aucun débit d’échantillonnage accepté par le micro. "
        "Dernière erreur PortAudio : " + str(last_err)
    ) from last_err


def _server_rejected_duplicate_name(sock: socket.socket, log: logging.Logger) -> bool:
    """
    Après CAMERA, le serveur peut répondre ERROR duplicate_stream_id si le nom est déjà pris.
    Attente courte puis lecture non bloquante d’une ligne.
    """
    r, _, _ = select.select([sock], [], [], 0.25)
    if not r:
        return False
    buf = bytearray()
    try:
        while len(buf) < 512:
            chunk = sock.recv(1)
            if not chunk:
                break
            buf.extend(chunk)
            if buf.endswith(b"\n"):
                break
    except OSError:
        return False
    line = bytes(buf).decode("utf-8", errors="replace").strip()
    if line.startswith("ERROR"):
        log.error(
            "%s — un autre flux utilise déjà ce nom ; choisissez un --name différent.",
            line,
        )
        return True
    return False


def _run_one_socket_session(
    sock: socket.socket,
    cap: cv2.VideoCapture,
    args: argparse.Namespace,
    audio_stream: Any,
    audio_q: queue.Queue[bytes],
    fps_diag: _ClientFpsDiag | None,
    log: logging.Logger,
) -> str:
    """
    Envoie jusqu’à erreur réseau ou fin de flux vidéo.
    Retour : 'stop' (ne pas reconnect), 'reconnect' (perte connexion).
    """
    send_camera_header(sock, args.name)
    if _server_rejected_duplicate_name(sock, log):
        return "stop"
    if args.audio and audio_stream is not None:
        sr = int(getattr(audio_stream, "samplerate", 48000))
        send_v2_header(sock, True, sample_rate=sr, channels=args.audio_channels)

    min_interval = 1.0 / args.fps if args.fps and args.fps > 0 else 0.0
    last_send = 0.0
    q = max(1, min(100, args.jpeg_quality))

    def _process_frame(raw: np.ndarray) -> np.ndarray:
        h, w = raw.shape[:2]
        if args.width > 0 and args.height > 0:
            out = cv2.resize(raw, (args.width, args.height))
        elif args.width > 0:
            scale = args.width / w
            out = cv2.resize(raw, (args.width, max(1, int(h * scale))))
        elif args.height > 0:
            scale = args.height / h
            out = cv2.resize(raw, (max(1, int(w * scale)), args.height))
        else:
            out = raw
        return _overlay_time_bgr(out, args.name, args.overlay_time)

    def _send_jpeg(jpeg_b: bytes) -> bool:
        try:
            if args.audio:
                send_v2_packet(sock, PT_VIDEO, jpeg_b)
                for _ in range(12):
                    try:
                        pcm = audio_q.get_nowait()
                        send_v2_packet(sock, PT_AUDIO, pcm)
                        if fps_diag:
                            fps_diag.bump("audio_pkt")
                    except queue.Empty:
                        break
            else:
                send_jpeg_frame(sock, jpeg_b)
        except OSError as e:
            log.warning("Envoi interrompu : %s", e)
            return False
        return True

    use_pipeline = not args.sequential_encode

    if use_pipeline:
        in_q: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=2)
        out_q: queue.Queue[bytes] = queue.Queue(maxsize=2)

        def _encode_worker() -> None:
            while True:
                fr = in_q.get()
                if fr is None:
                    break
                jpeg_b = encode_bgr_jpeg_best(fr, q)
                if jpeg_b:
                    if fps_diag:
                        fps_diag.bump("encode")
                    out_q.put(jpeg_b)

        enc_th = threading.Thread(target=_encode_worker, daemon=True)
        enc_th.start()

        def _read_one() -> np.ndarray | None:
            nonlocal last_send
            if min_interval:
                now = time.time()
                wait = min_interval - (now - last_send)
                if wait > 0:
                    time.sleep(wait)
            ok, frame = cap.read()
            if not ok or frame is None:
                return None
            last_send = time.time()
            out = _process_frame(frame)
            if fps_diag:
                fps_diag.bump("read")
            return out

        try:
            f0 = _read_one()
            if f0 is None:
                log.info("Fin de flux ou erreur de lecture.")
                return "stop"
            in_q.put(f0)
            f1 = _read_one()
            if f1 is None:
                log.info("Fin de flux ou erreur de lecture.")
                try:
                    jpeg_b = out_q.get(timeout=30.0)
                except queue.Empty:
                    jpeg_b = b""
                if jpeg_b:
                    if _send_jpeg(jpeg_b) and fps_diag:
                        fps_diag.bump("send_v")
                return "stop"
            in_q.put(f1)
            while True:
                jpeg_b = out_q.get()
                ok_send = _send_jpeg(jpeg_b)
                if ok_send and fps_diag:
                    fps_diag.bump("send_v")
                if not ok_send:
                    return "reconnect"
                f_next = _read_one()
                if f_next is None:
                    log.info("Fin de flux ou erreur de lecture.")
                    return "stop"
                in_q.put(f_next)
        finally:
            in_q.put(None)
            enc_th.join(timeout=3.0)
    else:
        while True:
            if min_interval:
                now = time.time()
                wait = min_interval - (now - last_send)
                if wait > 0:
                    time.sleep(wait)

            ok, frame = cap.read()
            if not ok or frame is None:
                log.info("Fin de flux ou erreur de lecture.")
                return "stop"

            frame = _process_frame(frame)
            if fps_diag:
                fps_diag.bump("read")
            jpeg_b = encode_bgr_jpeg_best(frame, q)
            if not jpeg_b:
                continue
            if fps_diag:
                fps_diag.bump("encode")
            if not _send_jpeg(jpeg_b):
                return "reconnect"
            if fps_diag:
                fps_diag.bump("send_v")
            last_send = time.time()
    return "stop"


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_ns, argv_rest = pre.parse_known_args(sys.argv[1:])
    cset = load_client_toml(pre_ns.config) if pre_ns.config else ClientSettings()

    p = argparse.ArgumentParser(
        description="Envoie un flux vidéo vers le serveur video.",
        parents=[pre],
    )
    p.add_argument("--host", default=cset.host, help="Adresse du serveur")
    p.add_argument("--port", type=int, default=cset.port, help="Port TCP du serveur")
    p.add_argument(
        "--name",
        default="camera",
        help="Identifiant affiché côté serveur (unique par flux)",
    )
    p.add_argument(
        "--device",
        default="0",
        help="Source OpenCV : 0,1,... pour webcam, ou chemin/URL (RTSP, http, fichier)",
    )
    p.add_argument("--jpeg-quality", type=int, default=85, help="Qualité JPEG 1-100")
    p.add_argument(
        "--sequential-encode",
        action="store_true",
        help="Désactive le chevauchement lecture caméra / encodage JPEG (un seul fil)",
    )
    p.add_argument(
        "--width",
        type=int,
        default=cset.width,
        help="Largeur d’envoi (0 = native). La webcam locale est quand même ouverte en HD/MJPEG.",
    )
    p.add_argument(
        "--height",
        type=int,
        default=cset.height,
        help="Hauteur d’envoi (0 = native). Combiné à --width pour forcer la taille JPEG.",
    )
    p.add_argument("--fps", type=float, default=0, help="Limite FPS (0 = max possible)")
    p.add_argument(
        "--audio",
        action="store_true",
        help="Capturer le micro et l’envoyer (REC continu sur le serveur → MP4 avec son)",
    )
    p.add_argument(
        "--audio-rate",
        type=int,
        default=0,
        help="Hz PCM 16 bits (0 = débit natif du micro d’abord, moins d’erreurs ALSA) ; sinon ex. 16000",
    )
    p.add_argument("--audio-channels", type=int, default=1, help="1 = mono")
    p.add_argument(
        "--debug-fps",
        action="store_true",
        help="Chaque seconde : débit read caméra / encodage JPEG / envoi vidéo + paquets audio",
    )
    p.add_argument(
        "--overlay-time",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Superposer date/heure locale + nom du flux sur l’image avant JPEG",
    )
    p.add_argument(
        "--reconnect",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Réessayer la connexion TCP avec backoff en cas de perte",
    )
    p.add_argument(
        "--reconnect-initial-delay",
        type=float,
        default=1.0,
        help="Première attente avant reconnexion (secondes)",
    )
    p.add_argument(
        "--reconnect-max-delay",
        type=float,
        default=30.0,
        help="Plafond du backoff (secondes)",
    )
    p.add_argument("--log-level", default="INFO", help="Niveau logging")
    p.add_argument(
        "--log-json",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Logs en JSON ligne par ligne",
    )
    args = p.parse_args(argv_rest)

    setup_logging(args.log_level, args.log_json)
    log = logging.getLogger("video_app.camera")

    if args.audio and sd is None:
        log.error(
            "Audio indisponible : installez sounddevice (pip) et PortAudio "
            "(ex. Debian: sudo apt install libportaudio2)"
        )
        return

    try:
        dev = int(args.device)
    except ValueError:
        dev = args.device

    cap = cv2.VideoCapture(dev)
    if not cap.isOpened():
        log.error("Impossible d’ouvrir la source : %r", args.device)
        return

    probe_local = is_local_opencv_capture_device(dev)
    apply_fps_cap = args.fps == 0 and probe_local
    if probe_local and args.width > 0 and args.height > 0:
        eff_w, eff_h, eff_fps = configure_webcam_for_send_size(
            cap,
            args.width,
            args.height,
            apply_fps=apply_fps_cap,
        )
        apply_native_configure = True
    else:
        apply_native_configure = probe_local
        eff_w, eff_h, eff_fps = configure_webcam_best_effort(
            cap,
            apply_resolution=apply_native_configure,
            apply_fps=apply_fps_cap,
        )
    out_w, out_h = _output_dimensions(eff_w, eff_h, args.width, args.height)
    if probe_local and (apply_native_configure or apply_fps_cap):
        fps_str = f"{eff_fps:g}" if eff_fps > 0 else "?"
        if eff_w > 0 and eff_h > 0:
            if out_w != eff_w or out_h != eff_h:
                log.info(
                    "Capture : %s×%s px (native) → envoi %s×%s px, FPS signalé ≈ %s",
                    eff_w,
                    eff_h,
                    out_w,
                    out_h,
                    fps_str,
                )
            else:
                log.info(
                    "Capture : %s×%s px, FPS signalé ≈ %s", eff_w, eff_h, fps_str
                )
        else:
            log.info("Capture : taille native inconnue, FPS signalé ≈ %s", fps_str)

    audio_stream = None
    audio_q: queue.Queue[bytes] = queue.Queue(maxsize=300)

    if args.audio:

        def _audio_cb(indata, frames, _t, status) -> None:
            if status:
                log.debug("%s", status)
            try:
                audio_q.put_nowait(bytes(indata))
            except queue.Full:
                pass

        try:
            audio_stream, audio_sr = _open_audio_input_stream(
                args.audio_rate, args.audio_channels, _audio_cb
            )
        except (RuntimeError, sd.PortAudioError) as e:
            log.error("%s", e)
            cap.release()
            return
        if args.audio_rate > 0 and audio_sr != args.audio_rate:
            log.info(
                "Micro : %s Hz utilisés (%s Hz refusés par le périphérique).",
                audio_sr,
                args.audio_rate,
            )

    log.info(
        "Flux « %s » → %s:%s%s",
        args.name,
        args.host,
        args.port,
        " (+ audio)" if args.audio else "",
    )
    if turbojpeg_available():
        log.info("JPEG : TurboJPEG (libjpeg-turbo)")
    else:
        log.info(
            "JPEG : OpenCV seul — PyTurboJPEG + libturbojpeg recommandés pour la vitesse"
        )

    fps_diag: _ClientFpsDiag | None = None
    if args.debug_fps:
        fps_diag = _ClientFpsDiag(args.name)
        log.debug(
            "debug-fps : read = images lues ; encode = JPEG ; "
            "send_video / audio_pkt = paquets TCP"
        )

    backoff = max(0.1, float(args.reconnect_initial_delay))
    backoff_max = max(backoff, float(args.reconnect_max_delay))

    try:
        while True:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                sock.connect((args.host, args.port))
            except OSError as e:
                log.warning("Connexion à %s:%s impossible : %s", args.host, args.port, e)
                sock.close()
                if not args.reconnect:
                    break
                time.sleep(backoff)
                backoff = min(backoff * 2, backoff_max)
                continue

            backoff = max(0.1, float(args.reconnect_initial_delay))

            reason = _run_one_socket_session(
                sock, cap, args, audio_stream, audio_q, fps_diag, log
            )
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()

            if reason == "stop" or not args.reconnect:
                break
            log.info("Reconnexion dans %.1f s…", backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, backoff_max)
    finally:
        if fps_diag is not None:
            fps_diag.close()
        if audio_stream is not None:
            audio_stream.stop()
            audio_stream.close()
        cap.release()


if __name__ == "__main__":
    main()
