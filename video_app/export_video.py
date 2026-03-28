"""Export : MP4 H.264 (Android) par flux + empilée ; repli AVI si pas de ffmpeg."""

from __future__ import annotations

import os
import time
from typing import Any

import cv2
import numpy as np

from video_app import ffmpeg_io
from video_app.buffer import StreamBuffer, effective_fps_from_timestamps


def _write_video_file(path: str, frames: list, fps: float, label: str) -> bool:
    if not frames:
        print(f"[{label}] Aucune image à enregistrer.")
        return False
    h, w = frames[0].shape[:2]
    if w == 0 or h == 0:
        print(f"[{label}] Dimensions invalides.")
        return False
    os.makedirs("./video", exist_ok=True)
    if path.endswith(".mp4") and ffmpeg_io.ffmpeg_available():
        if ffmpeg_io.write_frames_bgr_to_mp4(path, frames, fps, label):
            return True
        print(f"[{label}] MP4 ffmpeg a échoué, repli AVI")
        path = path[:-4] + ".avi"
    elif path.endswith(".mp4"):
        path = path[:-4] + ".avi"
    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*"XVID"), float(max(1.0, fps)), (w, h)
    )
    if not writer.isOpened():
        print(f"[{label}] Impossible d’ouvrir le writer pour {path}.")
        return False
    try:
        for f in frames:
            writer.write(f)
    finally:
        writer.release()
    print(f"[{label}] Vidéo enregistrée : {path}")
    return True


def _resize_to_width(frame: np.ndarray, target_w: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= 0:
        return frame
    nh = max(1, int(h * target_w / w))
    return cv2.resize(frame, (target_w, nh))


def _stack_frames_row(frames: list[np.ndarray], target_w: int) -> np.ndarray:
    parts = [_resize_to_width(f, target_w) for f in frames]
    return np.vstack(parts)


def build_vertical_stack(frames: list[np.ndarray]) -> np.ndarray | None:
    """Empile verticalement (flux 0 en haut). Au moins 2 images requises."""
    if len(frames) < 2:
        return None
    max_w = max(f.shape[1] for f in frames)
    return _stack_frames_row(frames, max_w)


def save_per_stream_and_stack(
    buffers: list[StreamBuffer],
    frame_rate: int,
    ts: int | None = None,
) -> None:
    """
    Ordre des `buffers` = ordre d’affichage : premier = bandeau du haut (flux 0), etc.
    """
    ts = ts if ts is not None else int(time.time())
    ext = ".mp4" if ffmpeg_io.ffmpeg_available() else ".avi"
    fb = float(frame_rate)
    snapshots: list[tuple[str, list[tuple[float, Any]]]] = [
        (b.stream_id, b.snapshot_timed()) for b in buffers
    ]

    for stream_id, timed in snapshots:
        if not timed:
            print(f"[{stream_id}] Aucune image à enregistrer.")
            continue
        frames = [fr for _, fr in timed]
        times = [t for t, _ in timed]
        fps = effective_fps_from_timestamps(times, fb)
        path = f"./video/{stream_id}_{ts}{ext}"
        _write_video_file(path, frames, fps, stream_id)

    non_empty = [(sid, t) for sid, t in snapshots if t]
    if len(non_empty) < 2:
        if len(non_empty) == 1:
            print(
                "[stack] Un seul flux actif : pas de vidéo empilée "
                "(inutile, la vidéo par flux suffit)."
            )
        return

    min_len = min(len(t) for _, t in non_empty)

    stacked_frames: list[np.ndarray] = []
    for i in range(min_len):
        row = [t[i][1] for _, t in non_empty]
        stacked = build_vertical_stack(row)
        if stacked is not None:
            stacked_frames.append(stacked)

    fps_vals: list[float] = []
    for _, timed in non_empty:
        head = timed[:min_len]
        if len(head) >= 2:
            fps_vals.append(
                effective_fps_from_timestamps([x[0] for x in head], fb)
            )
    fps_stack = sum(fps_vals) / len(fps_vals) if fps_vals else fb

    out = f"./video/stack_{ts}{ext}"
    _write_video_file(out, stacked_frames, fps_stack, "stack")
