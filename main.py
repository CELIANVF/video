#!/usr/bin/env python3
"""
Point d’entrée serveur : reçoit plusieurs flux vidéo (réseau via camera.py
et/ou caméras locales avec --local).
"""

import argparse
import os
import sys

# Avant tout import qui charge cv2 (via video_app.server).
if sys.platform.startswith("linux") and "QT_QPA_PLATFORM" not in os.environ:
    os.environ["QT_QPA_PLATFORM"] = "xcb"

from video_app.config import AppSettings, load_toml_file
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
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_ns, argv_rest = pre.parse_known_args(sys.argv[1:])
    settings = load_toml_file(pre_ns.config) if pre_ns.config else AppSettings()

    parser = argparse.ArgumentParser(
        description="Serveur multi-flux : camera.py envoie les flux réseau ; "
        "--local ajoute une webcam sur cette machine.",
        parents=[pre],
    )
    parser.add_argument(
        "--host",
        default=settings.host,
        help="Adresse d’écoute (défaut : config ou 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.port,
        help="Port TCP",
    )
    parser.add_argument(
        "--frame-rate",
        type=int,
        default=settings.frame_rate,
        help="FPS cible affichage/tampon",
    )
    parser.add_argument(
        "--buffer-duration",
        type=int,
        default=settings.buffer_duration,
        help="Durée du tampon (secondes) pour l’enregistrement",
    )
    parser.add_argument(
        "--export-dir",
        default=settings.export_dir,
        help="Répertoire des exports vidéo / snapshots",
    )
    parser.add_argument(
        "--local",
        action="append",
        default=[],
        metavar="SPEC",
        help="Caméra locale sur le serveur. Ex: --local 0 --local cam_b:1",
    )
    parser.add_argument(
        "--gui",
        action=argparse.BooleanOptionalAction,
        default=settings.gui_enabled,
        help="Interface PyQt6 (boutons, F1, …). Défaut : valeur [gui] enabled du TOML, sinon OpenCV + raccourcis.",
    )
    parser.add_argument(
        "--debug-fps",
        action="store_true",
        help="Journal chaque seconde les débits par étape (réception/décodage/append) par flux réseau",
    )
    parser.add_argument(
        "--log-level",
        default=settings.log_level,
        help="Niveau logging (DEBUG, INFO, WARNING, …)",
    )
    parser.add_argument(
        "--log-json",
        action=argparse.BooleanOptionalAction,
        default=settings.log_json,
        help="Journal JSON ligne par ligne (sinon texte)",
    )
    parser.add_argument(
        "--metrics",
        action=argparse.BooleanOptionalAction,
        default=settings.metrics_enabled,
        help="Exposer les métriques Prometheus",
    )
    parser.add_argument(
        "--metrics-host",
        default=settings.metrics_host,
        help="Adresse d’écoute du endpoint /metrics",
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=settings.metrics_port,
        help="Port du endpoint /metrics",
    )
    parser.add_argument(
        "--web",
        action=argparse.BooleanOptionalAction,
        default=settings.web_enabled,
        help="Serveur HTTP MJPEG + page de contrôle",
    )
    parser.add_argument(
        "--web-host",
        default=settings.web_host,
        help="Adresse d’écoute du serveur web",
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=settings.web_port,
        help="Port du serveur web",
    )
    parser.add_argument(
        "--web-path-prefix",
        default=settings.web_path_prefix,
        help="Préfixe URL (ex. /view) ; laisser vide pour la racine",
    )
    args = parser.parse_args(argv_rest)

    local_devices = [_parse_local(s) for s in args.local]
    run_server(
        host=args.host,
        port=args.port,
        frame_rate=args.frame_rate,
        buffer_duration=args.buffer_duration,
        local_devices=local_devices,
        gui=args.gui,
        debug_fps=args.debug_fps,
        export_dir=args.export_dir,
        stream_labels=settings.stream_labels,
        stream_order=settings.stream_order,
        log_level=args.log_level,
        log_json=args.log_json,
        metrics_enabled=args.metrics,
        metrics_host=args.metrics_host,
        metrics_port=args.metrics_port,
        web_enabled=args.web,
        web_host=args.web_host,
        web_port=args.web_port,
        web_path_prefix=args.web_path_prefix,
    )


if __name__ == "__main__":
    main()
