"""Chargement TOML et paramètres fusionnés (fichier + CLI)."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AppSettings:
    host: str = "0.0.0.0"
    port: int = 8765
    frame_rate: int = 30
    buffer_duration: int = 5
    export_dir: str = "./video"
    gui_enabled: bool = False
    stream_labels: dict[str, str] = field(default_factory=dict)
    stream_order: list[str] = field(default_factory=list)
    log_level: str = "INFO"
    log_json: bool = False
    metrics_enabled: bool = False
    metrics_host: str = "127.0.0.1"
    metrics_port: int = 9090
    web_enabled: bool = False
    web_host: str = "127.0.0.1"
    web_port: int = 8080
    web_path_prefix: str = ""


def _table(d: dict[str, Any], key: str) -> dict[str, Any]:
    v = d.get(key)
    return v if isinstance(v, dict) else {}


def load_toml_file(path: str | Path) -> AppSettings:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"fichier de config introuvable : {path}")
    import tomllib

    with path.open("rb") as f:
        raw = tomllib.load(f)
    s = AppSettings()
    srv = _table(raw, "server")
    if "host" in srv:
        s.host = str(srv["host"])
    if "port" in srv:
        s.port = int(srv["port"])
    if "frame_rate" in srv:
        s.frame_rate = int(srv["frame_rate"])
    if "buffer_duration" in srv:
        s.buffer_duration = int(srv["buffer_duration"])
    paths = _table(raw, "paths")
    if "export_dir" in paths:
        s.export_dir = str(paths["export_dir"])
    gui = _table(raw, "gui")
    if "enabled" in gui:
        s.gui_enabled = bool(gui["enabled"])
    if "stream_labels" in gui and isinstance(gui["stream_labels"], dict):
        s.stream_labels = {str(k): str(v) for k, v in gui["stream_labels"].items()}
    if "stream_order" in gui and isinstance(gui["stream_order"], list):
        s.stream_order = [str(x) for x in gui["stream_order"]]
    log = _table(raw, "logging")
    if "level" in log:
        s.log_level = str(log["level"])
    if "json" in log:
        s.log_json = bool(log["json"])
    met = _table(raw, "metrics")
    if "enabled" in met:
        s.metrics_enabled = bool(met["enabled"])
    if "host" in met:
        s.metrics_host = str(met["host"])
    if "port" in met:
        s.metrics_port = int(met["port"])
    web = _table(raw, "web")
    if "enabled" in web:
        s.web_enabled = bool(web["enabled"])
    if "host" in web:
        s.web_host = str(web["host"])
    if "port" in web:
        s.web_port = int(web["port"])
    if "path_prefix" in web:
        p = str(web["path_prefix"]).strip()
        s.web_path_prefix = p.rstrip("/") if p else ""
    return s


@dataclass
class ClientSettings:
    host: str = "127.0.0.1"
    port: int = 8765
    width: int = 0
    height: int = 0


def load_client_toml(path: str | Path) -> ClientSettings:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"fichier de config introuvable : {path}")
    import tomllib

    with path.open("rb") as f:
        raw = tomllib.load(f)
    c = ClientSettings()
    cl = _table(raw, "client")
    srv = _table(raw, "server")
    if "host" in cl:
        c.host = str(cl["host"])
    elif "host" in srv:
        c.host = str(srv["host"])
    if "port" in cl:
        c.port = int(cl["port"])
    elif "port" in srv:
        c.port = int(srv["port"])
    if "width" in cl:
        c.width = int(cl["width"])
    if "height" in cl:
        c.height = int(cl["height"])
    elif "hight" in cl:
        c.height = int(cl["hight"])
    return c


def merge_settings(base: AppSettings, overrides: dict[str, Any]) -> AppSettings:
    """Copie profonde + champs non-None depuis overrides (clés plates)."""
    s = copy.deepcopy(base)
    for k, v in overrides.items():
        if v is None:
            continue
        if hasattr(s, k):
            setattr(s, k, v)
    return s
