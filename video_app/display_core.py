"""Logique d’affichage / REC empilé partagée entre OpenCV et l’interface Qt."""

from __future__ import annotations

import os
import time
from typing import Any

import cv2

from video_app import ffmpeg_io
from video_app.export_video import build_vertical_stack


def gather_display_frames(
    registry: Any,
    live_display: bool,
    display_delay_sec: float,
) -> list[tuple[str, Any]]:
    ids = registry.ids()
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
                print(
                    "[cont_stack] ffmpeg :",
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
        print(f"[cont_stack] Fichier terminé : {final}")

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
        os.makedirs("./video", exist_ok=True)
        if ffmpeg_io.ffmpeg_available():
            path = f"./video/cont_stack_{sess}.mp4"
            state["writer"] = ffmpeg_io.FfmpegBGRWriter(
                path, sw, sh, frame_rate
            )
            state["path"] = path
            state["bound_session"] = sess
            state["stack_frame_count"] = 0
            state["stack_wall_start"] = None
            state["stack_wall_last"] = None
            print(f"[cont_stack] REC empilé (H.264) → {path}")
        else:
            path = f"./video/cont_stack_{sess}.avi"
            writer = cv2.VideoWriter(
                path,
                cv2.VideoWriter_fourcc(*"XVID"),
                max(1, frame_rate),
                (sw, sh),
            )
            if not writer.isOpened():
                print(f"[cont_stack] Impossible d’ouvrir {path}")
                state["writer"] = None
                state["path"] = None
                return
            state["writer"] = writer
            state["path"] = path
            state["bound_session"] = sess
            state["stack_frame_count"] = 0
            state["stack_wall_start"] = None
            state["stack_wall_last"] = None
            print(f"[cont_stack] REC empilé (AVI) → {path}")
    writer = state.get("writer")
    if writer is not None:
        now = time.time()
        if state.get("stack_wall_start") is None:
            state["stack_wall_start"] = now
        state["stack_wall_last"] = now
        state["stack_frame_count"] = int(state.get("stack_frame_count") or 0) + 1
        state["stack_enc_fps"] = int(frame_rate)
        writer.write(stacked)
