"""Configuration du journal (texte ou JSON ligne)."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        for k in ("stream_id", "peer", "component"):
            v = getattr(record, k, None)
            if v is not None:
                payload[k] = v
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: str = "INFO", json_format: bool = False) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    lvl = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(lvl)
    h = logging.StreamHandler(sys.stdout)
    h.setLevel(lvl)
    if json_format:
        h.setFormatter(JsonLineFormatter())
    else:
        h.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
    root.addHandler(h)
    # Réduire le bruit des librairies
    logging.getLogger("urllib3").setLevel(logging.WARNING)
