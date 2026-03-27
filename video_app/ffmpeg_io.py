"""Encodage MP4 (H.264 + AAC) via ffmpeg — lisible sur Android."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import BinaryIO

import numpy as np


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


class FfmpegBGRWriter:
    """Vidéo seule (sans piste audio), BGR 8 bits → MP4 H.264."""

    def __init__(self, path: str, w: int, h: int, fps: int):
        self._path = path
        self._proc = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{w}x{h}",
                "-r",
                str(max(1, int(fps))),
                "-i",
                "-",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                path,
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._stdin: BinaryIO | None = self._proc.stdin

    def write(self, frame: np.ndarray) -> None:
        if self._stdin is None:
            return
        self._stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self) -> tuple[int, bytes]:
        err = b""
        if self._stdin is not None:
            self._stdin.close()
            self._stdin = None
        try:
            _, err = self._proc.communicate(timeout=600)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            _, err = self._proc.communicate()
        return self._proc.returncode, err or b""


def write_frames_bgr_to_mp4(
    path: str, frames: list, fps: int, label: str = "export"
) -> bool:
    if not frames or not ffmpeg_available():
        return False
    h, w = frames[0].shape[:2]
    if w <= 0 or h <= 0:
        return False
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    wobj = FfmpegBGRWriter(path, w, h, fps)
    try:
        for f in frames:
            wobj.write(f)
    finally:
        code, err = wobj.close()
    if code != 0:
        print(f"[{label}] ffmpeg a échoué ({code}): {err.decode(errors='replace')[:500]}")
        return False
    print(f"[{label}] Vidéo enregistrée : {path}")
    return True


def mux_video_pcm_to_mp4(
    video_mp4: str,
    pcm_s16le_path: str,
    sample_rate: int,
    channels: int,
    out_mp4: str,
) -> bool:
    if not ffmpeg_available():
        return False
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                video_mp4,
                "-f",
                "s16le",
                "-ar",
                str(sample_rate),
                "-ac",
                str(channels),
                "-i",
                pcm_s16le_path,
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-movflags",
                "+faststart",
                "-shortest",
                out_mp4,
            ],
            check=True,
            capture_output=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(
            "[mux] échec ffmpeg :",
            (e.stderr or b"").decode(errors="replace")[:400],
        )
        return False
