#!/usr/bin/env python3
"""
Client caméra : envoie un flux vidéo (webcam locale, index, chemin fichier ou URL RTSP)
vers le serveur (main.py).
"""

import argparse
import socket
import time

import cv2

from video_app.protocol import send_camera_header, send_jpeg_frame


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
    args = p.parse_args()

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
    print(f"Flux « {args.name} » → {args.host}:{args.port}")

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
            try:
                send_jpeg_frame(sock, jpeg.tobytes())
            except OSError as e:
                print(f"Envoi interrompu: {e}")
                break
            last_send = time.time()
    finally:
        sock.close()
        cap.release()


if __name__ == "__main__":
    main()
