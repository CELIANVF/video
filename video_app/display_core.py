"""Logique d’affichage / REC empilé partagée entre OpenCV et l’interface Qt."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import cv2

from video_app import ffmpeg_io
from video_app.export_video import build_vertical_stack

logger = logging.getLogger(__name__)


def ordered_stream_ids_from_list(
    raw_ids: list[str], stream_order: list[str] | None
) -> list[str]:
    raw = list(raw_ids)
    if not stream_order:
        return raw
    seen: set[str] = set()
    out: list[str] = []
    for sid in stream_order:
        if sid in raw and sid not in seen:
            out.append(sid)
            seen.add(sid)
    for sid in raw:
        if sid not in seen:
            out.append(sid)
            seen.add(sid)
    return out


def ordered_stream_ids(registry: Any, stream_order: list[str] | None) -> list[str]:
    return ordered_stream_ids_from_list(list(registry.ids()), stream_order)


def gather_display_frames(
    registry: Any,
    live_display: bool,
    display_delay_sec: float,
    stream_order: list[str] | None = None,
    *,
    active_stream_ids: list[str] | None = None,
) -> list[tuple[str, Any]]:
    ids = (
        ordered_stream_ids_from_list(active_stream_ids, stream_order)
        if active_stream_ids is not None
        else ordered_stream_ids(registry, stream_order)
    )
    out: list[tuple[str, Any]] = []
    for sid in ids:
        b = registry.get(sid)
        if b is None:
            continue
        fr = (
            b.latest()
            if live_display
            else b.frame_at_delay(display_delay_sec)
        )
        if fr is not None:
            out.append((sid, fr))
    return out


def gather_display_frames_with_ts(
    registry: Any,
    live_display: bool,
    display_delay_sec: float,
    stream_order: list[str] | None = None,
    *,
    active_stream_ids: list[str] | None = None,
) -> list[tuple[str, Any, float | None]]:
    ids = (
        ordered_stream_ids_from_list(active_stream_ids, stream_order)
        if active_stream_ids is not None
        else ordered_stream_ids(registry, stream_order)
    )
    out: list[tuple[str, Any, float | None]] = []
    for sid in ids:
        b = registry.get(sid)
        if b is None:
            continue
        if live_display:
            fr, ts = b.latest_with_ts()
        else:
            fr, ts = b.frame_at_delay_with_ts(display_delay_sec)
        if fr is not None:
            out.append((sid, fr, ts))
    return out


def close_continuous_stack_state(state: dict[str, Any]) -> None:
    w = state.get("writer")
    path_b = state.get("path")
    n = int(state.get("stack_frame_count") or 0)
    t0 = state.get("stack_wall_start")
    t1 = state.get("stack_wall_last")
    enc_fps = int(state.get("stack_enc_fps") or 30)

    if w is not None:
        if hasattr(w, "release"):
            w.release()
        else:
            code, err = w.close()
            if code != 0 and err:
                logger.warning(
                    "[cont_stack] ffmpeg : %s",
                    err.decode(errors="replace")[:200],
                )
    state["writer"] = None

    final = path_b
    if (
        path_b
        and os.path.isfile(path_b)
        and ffmpeg_io.ffmpeg_available()
        and n >= 2
        and t0 is not None
        and t1 is not None
    ):
        final = ffmpeg_io.retime_continuous_video_file(
            path_b,
            frame_count=n,
            nominal_fps=enc_fps,
            wall_start=float(t0),
            wall_end=float(t1),
        )

    if final and os.path.isfile(final):
        logger.info("[cont_stack] Fichier terminé : %s", final)

    state["path"] = None
    state["bound_session"] = None
    state["stack_frame_count"] = 0
    state["stack_wall_start"] = None
    state["stack_wall_last"] = None
    state["stack_enc_fps"] = None


def tick_continuous_stack_recording(
    registry: Any,
    frames_data: list[tuple[str, Any]],
    frame_rate: int,
    state: dict,
    export_dir: str = "./video",
) -> None:
    if not registry.is_continuous_recording():
        return
    row_stack = [fr for _sid, fr in frames_data]
    stacked = build_vertical_stack(row_stack)
    sess = registry.get_continuous_session_ts()
    if state.get("bound_session") is not None and sess != state["bound_session"]:
        close_continuous_stack_state(state)
    if stacked is None:
        return
    sh, sw = stacked.shape[:2]
    if state.get("writer") is None:
        ed = export_dir.rstrip("/") or "."
        os.makedirs(ed, exist_ok=True)
        if ffmpeg_io.ffmpeg_available():
            path = os.path.join(ed, f"cont_stack_{sess}.mp4")
            state["writer"] = ffmpeg_io.FfmpegBGRWriter(
                path, sw, sh, frame_rate
            )
            state["path"] = path
            state["bound_session"] = sess
            state["stack_frame_count"] = 0
            state["stack_wall_start"] = None
            state["stack_wall_last"] = None
            logger.info("[cont_stack] REC empilé (H.264) → %s", path)
        else:
            path = os.path.join(ed, f"cont_stack_{sess}.avi")
            writer = cv2.VideoWriter(
                path,
                cv2.VideoWriter_fourcc(*"XVID"),
                max(1, frame_rate),
                (sw, sh),
            )
            if not writer.isOpened():
                logger.warning("[cont_stack] Impossible d’ouvrir %s", path)
                state["writer"] = None
                state["path"] = None
                return
            state["writer"] = writer
            state["path"] = path
            state["bound_session"] = sess
            state["stack_frame_count"] = 0
            state["stack_wall_start"] = None
            state["stack_wall_last"] = None
            logger.info("[cont_stack] REC empilé (AVI) → %s", path)
    writer = state.get("writer")
    if writer is not None:
        now = time.time()
        if state.get("stack_wall_start") is None:
            state["stack_wall_start"] = now
        state["stack_wall_last"] = now
        state["stack_frame_count"] = int(state.get("stack_frame_count") or 0) + 1
        state["stack_enc_fps"] = int(frame_rate)
        writer.write(stacked)
