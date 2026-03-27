#!/usr/bin/env python3
"""
Point d’entrée serveur : reçoit plusieurs flux vidéo (réseau via camera.py
et/ou caméras locales avec --local).
"""

import argparse

from video_app.server import run_server


def _parse_local(spec: str) -> tuple[str, int | str]:
    """
    « id:0 » ou « id:/dev/video0 » ou « 0 » (id auto local_0).
    """
    if ":" in spec:
        sid, dev = spec.split(":", 1)
        sid = sid.strip() or "local"
        dev = dev.strip()
        try:
            return sid, int(dev)
        except ValueError:
            return sid, dev
    try:
        idx = int(spec)
        return f"local_{idx}", idx
    except ValueError:
        return "local", spec


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serveur multi-flux : camera.py envoie les flux réseau ; "
        "--local ajoute une webcam sur cette machine."
    )
    parser.add_argument("--host", default="0.0.0.0", help="Adresse d’écoute")
    parser.add_argument("--port", type=int, default=8765, help="Port TCP")
    parser.add_argument("--frame-rate", type=int, default=30, help="FPS cible affichage/tampon")
    parser.add_argument(
        "--buffer-duration",
        type=int,
        default=5,
        help="Durée du tampon (secondes) pour l’enregistrement",
    )
    parser.add_argument(
        "--local",
        action="append",
        default=[],
        metavar="SPEC",
        help="Caméra locale sur le serveur. Ex: --local 0 --local cam_b:1",
    )
    args = parser.parse_args()

    local_devices = [_parse_local(s) for s in args.local]
    run_server(
        host=args.host,
        port=args.port,
        frame_rate=args.frame_rate,
        buffer_duration=args.buffer_duration,
        local_devices=local_devices,
    )


if __name__ == "__main__":
    main()
