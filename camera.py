#!/usr/bin/env python3
"""
Client caméra : envoie un flux vidéo (webcam, RTSP, etc.) vers le serveur (main.py).
Avec --audio : protocole V2 + micro (PCM 16 bits), muxé côté serveur dans le MP4 en REC.
"""

from __future__ import annotations

import argparse
import queue
import socket
import time

import cv2

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
except ImportError:
    sd = None


def main() -> None:
    p = argparse.ArgumentParser(description="Envoie un flux vidéo vers le serveur video.")
    p.add_argument("--host", default="127.0.0.1", help="Adresse du serveur")
    p.add_argument("--port", type=int, default=8765, help="Port TCP du serveur")
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
        "--width",
        type=int,
        default=0,
        help="Largeur max (0 = taille source)",
    )
    p.add_argument(
        "--height",
        type=int,
        default=0,
        help="Hauteur max (0 = taille source)",
    )
    p.add_argument("--fps", type=float, default=0, help="Limite FPS (0 = max possible)")
    p.add_argument(
        "--audio",
        action="store_true",
        help="Capturer le micro et l’envoyer (REC continu sur le serveur → MP4 avec son)",
    )
    p.add_argument("--audio-rate", type=int, default=16000, help="Hz (PCM 16 bits)")
    p.add_argument("--audio-channels", type=int, default=1, help="1 = mono")
    args = p.parse_args()

    if args.audio and sd is None:
        print("Le module sounddevice est requis pour --audio : pip install sounddevice")
        return

    try:
        dev = int(args.device)
    except ValueError:
        dev = args.device

    cap = cv2.VideoCapture(dev)
    if not cap.isOpened():
        print(f"Impossible d’ouvrir la source: {args.device!r}")
        return

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        sock.connect((args.host, args.port))
    except OSError as e:
        print(f"Connexion à {args.host}:{args.port} impossible: {e}")
        cap.release()
        return

    send_camera_header(sock, args.name)
    audio_stream = None
    audio_q: queue.Queue[bytes] = queue.Queue(maxsize=300)

    if args.audio:

        def _audio_cb(indata, frames, _t, status) -> None:
            if status:
                print(status)
            try:
                audio_q.put_nowait(indata.copy().tobytes())
            except queue.Full:
                pass

        audio_stream = sd.RawInputStream(
            samplerate=args.audio_rate,
            channels=args.audio_channels,
            dtype="int16",
            blocksize=max(160, args.audio_rate // 50),
            callback=_audio_cb,
        )
        audio_stream.start()
        send_v2_header(
            sock, True, sample_rate=args.audio_rate, channels=args.audio_channels
        )
    print(f"Flux « {args.name} » → {args.host}:{args.port}" + (" (+ audio)" if args.audio else ""))

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), max(1, min(100, args.jpeg_quality))]
    min_interval = 1.0 / args.fps if args.fps and args.fps > 0 else 0.0
    last_send = 0.0

    try:
        while True:
            if min_interval:
                now = time.time()
                wait = min_interval - (now - last_send)
                if wait > 0:
                    time.sleep(wait)

            ok, frame = cap.read()
            if not ok or frame is None:
                print("Fin de flux ou erreur de lecture.")
                break

            h, w = frame.shape[:2]
            if args.width > 0 and args.height > 0:
                frame = cv2.resize(frame, (args.width, args.height))
            elif args.width > 0:
                scale = args.width / w
                frame = cv2.resize(frame, (args.width, max(1, int(h * scale))))
            elif args.height > 0:
                scale = args.height / h
                frame = cv2.resize(frame, (max(1, int(w * scale)), args.height))

            ok, jpeg = cv2.imencode(".jpg", frame, encode_params)
            if not ok:
                continue
            jpeg_b = jpeg.tobytes()
            try:
                if args.audio:
                    send_v2_packet(sock, PT_VIDEO, jpeg_b)
                    for _ in range(12):
                        try:
                            pcm = audio_q.get_nowait()
                            send_v2_packet(sock, PT_AUDIO, pcm)
                        except queue.Empty:
                            break
                else:
                    send_jpeg_frame(sock, jpeg_b)
            except OSError as e:
                print(f"Envoi interrompu: {e}")
                break
            last_send = time.time()
    finally:
        if audio_stream is not None:
            audio_stream.stop()
            audio_stream.close()
        sock.close()
        cap.release()


if __name__ == "__main__":
    main()
