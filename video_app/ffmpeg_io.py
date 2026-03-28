"""Encodage MP4 (H.264 + AAC) via ffmpeg — lisible sur Android."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import BinaryIO

import numpy as np


def ffmpeg_executable() -> str | None:
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def ffprobe_executable() -> str | None:
    p = shutil.which("ffprobe")
    if p:
        return p
    fe = ffmpeg_executable()
    if fe:
        d = os.path.dirname(fe)
        for name in ("ffprobe", "ffprobe.exe"):
            sibling = os.path.join(d, name)
            if os.path.isfile(sibling) and os.access(sibling, os.X_OK):
                return sibling
    return None


def ffmpeg_available() -> bool:
    return ffmpeg_executable() is not None


def probe_video_duration(path: str) -> float | None:
    """Durée vidéo (s) : piste v:0 d’abord, sinon format."""
    probe = ffprobe_executable()
    if not probe:
        return None

    def _parse_one(cmd: list[str]) -> float | None:
        r = subprocess.run(cmd, capture_output=True, text=True)
        try:
            d = float((r.stdout or "").strip().splitlines()[0].strip())
            if d <= 0 or d != d:
                return None
            return d
        except (ValueError, IndexError):
            return None

    d = _parse_one(
        [
            probe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ]
    )
    if d is not None:
        return d
    return _parse_one(
        [
            probe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ]
    )


def retime_continuous_video_file(
    path: str,
    *,
    frame_count: int,
    nominal_fps: int,
    wall_start: float | None,
    wall_end: float | None,
) -> str:
    """
    Corrige une vidéo enregistrée avec un débit d’entrée ffmpeg fixe (ex. 30 Hz)
    alors que les images arrivent plus lentement : la durée fichier est trop
    courte → lecture accélérée. On ré-encode avec un étirement PTS calé sur le
    temps réel entre 1re et dernière image (ffprobe + marge dernière frame).
    """
    ff = ffmpeg_executable()
    if not ff or frame_count < 2:
        return path
    if wall_start is None or wall_end is None:
        return path
    wall = float(wall_end - wall_start)
    if wall < 0.001:
        return path
    wall_playback = wall

    dur_file = probe_video_duration(path)
    if dur_file is None or dur_file <= 0.001:
        dur_file = frame_count / float(max(1, int(nominal_fps)))

    factor = wall_playback / dur_file
    if abs(factor - 1.0) < 1e-6:
        return path

    out_fps = frame_count / wall_playback

    base, _ext = os.path.splitext(path)
    tmp = base + ".retime.tmp.mp4"
    out_final = base + ".mp4"
    print(
        f"[retime] {os.path.basename(path)}: fichier {dur_file:.2f}s → "
        f"cible {wall_playback:.2f}s (×{factor:.3f}, ~{out_fps:.2f} img/s)"
    )
    try:
        subprocess.run(
            [
                ff,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-fflags",
                "+genpts",
                "-i",
                path,
                "-vf",
                f"setpts=PTS*{factor}",
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
                tmp,
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        err = e.stderr
        if isinstance(err, bytes):
            err_s = err.decode(errors="replace")[:400]
        else:
            err_s = (err or "")[:400]
        print("[retime] échec ffmpeg :", err_s)
        try:
            os.remove(tmp)
        except OSError:
            pass
        return path

    try:
        os.remove(path)
    except OSError:
        pass
    try:
        os.replace(tmp, out_final)
    except OSError:
        return path
    return out_final


class FfmpegBGRWriter:
    """Vidéo seule (sans piste audio), BGR 8 bits → MP4 H.264."""

    def __init__(self, path: str, w: int, h: int, fps: float):
        self._path = path
        ff = ffmpeg_executable()
        if not ff:
            raise RuntimeError("ffmpeg introuvable")
        rf = max(1.0, min(240.0, float(fps)))
        self._proc = subprocess.Popen(
            [
                ff,
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
                f"{rf:.6g}",
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
        # Ne pas appeler communicate() après stdin.close() : en Python 3.13+ cela
        # tente un flush sur un flux déjà fermé (ValueError). stderr est DEVNULL.
        if self._stdin is not None:
            try:
                self._stdin.flush()
            except (ValueError, OSError, BrokenPipeError):
                pass
            try:
                self._stdin.close()
            except (ValueError, OSError):
                pass
            self._stdin = None
        try:
            self._proc.wait(timeout=600)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            try:
                self._proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                pass
        return self._proc.returncode, b""


def write_frames_bgr_to_mp4(
    path: str, frames: list, fps: float, label: str = "export"
) -> bool:
    if not frames or not ffmpeg_available():
        return False
    h, w = frames[0].shape[:2]
    if w <= 0 or h <= 0:
        return False
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    try:
        wobj = FfmpegBGRWriter(path, w, h, fps)
    except RuntimeError:
        return False
    try:
        for f in frames:
            wobj.write(f)
    finally:
        code, err = wobj.close()
    if code != 0:
        msg = err.decode(errors="replace")[:500] if err else ""
        print(f"[{label}] ffmpeg a échoué ({code}): {msg}")
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
    ff = ffmpeg_executable()
    if not ff:
        return False
    try:
        subprocess.run(
            [
                ff,
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
        err = e.stderr
        if isinstance(err, bytes):
            err_s = err.decode(errors="replace")[:400]
        else:
            err_s = (err or "")[:400]
        print("[mux] échec ffmpeg :", err_s)
        return False
