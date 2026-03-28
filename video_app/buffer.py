"""Tampon circulaire par flux et export vidéo."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from typing import Any

import cv2

from video_app import ffmpeg_io

logger = logging.getLogger(__name__)


def _max_deque_len(buffer_seconds: int) -> int:
    """Plafond de sécurité (images) : doit couvrir durée × débit élevé sans éjecter l’historique utile."""
    s = max(1, buffer_seconds)
    return max(1024, min(350_000, s * 2048))


def effective_fps_from_timestamps(
    times: list[float], fallback: float
) -> float:
    """FPS moyen à partir des dates de réception (n−1 / Δt), pour encoder la bonne durée."""
    if len(times) < 2:
        return max(1.0, min(240.0, fallback))
    span = times[-1] - times[0]
    if span < 1e-6:
        return max(1.0, min(240.0, fallback))
    v = (len(times) - 1) / span
    return max(1.0, min(240.0, v))


class StreamBuffer:
    def __init__(
        self,
        stream_id: str,
        frame_rate: int,
        buffer_duration: int,
        *,
        export_dir: str = "./video",
    ):
        self.stream_id = stream_id
        self.frame_rate = max(1, frame_rate)
        self._buffer_duration = max(1, buffer_duration)
        self.export_dir = export_dir.rstrip("/") or "."
        self._lock = threading.Lock()
        self._frames: deque = deque(maxlen=_max_deque_len(self._buffer_duration))
        self._continuous = False
        self._cont_session_ts = 0
        self._cont_stem: str | None = None
        self._cont_writer = None  # FfmpegBGRWriter ou cv2.VideoWriter
        self._cont_path: str | None = None
        self._cont_video_temp: str | None = None
        self._audio_sr: int | None = None
        self._audio_ch: int | None = None
        self._pcm_file = None
        self._cont_frame_count = 0
        self._cont_wall_t0: float | None = None
        self._cont_wall_t1: float | None = None
        self._opencv_warmup_frames: list[tuple] = []

    @property
    def buffer_duration(self) -> int:
        return self._buffer_duration

    def _trim_to_duration_unlocked(self, now: float) -> None:
        cutoff = now - self._buffer_duration
        while self._frames and self._frames[0][0] < cutoff:
            self._frames.popleft()

    def set_buffer_duration(self, seconds: int) -> None:
        seconds = max(1, seconds)
        with self._lock:
            self._buffer_duration = seconds
            self._trim_to_duration_unlocked(time.time())
            kept = list(self._frames)
            self._frames = deque(kept, maxlen=_max_deque_len(seconds))

    def set_audio_params(self, sample_rate: int, channels: int) -> None:
        with self._lock:
            self._audio_sr = int(sample_rate)
            self._audio_ch = int(channels)
            if (
                self._continuous
                and self._cont_stem
                and self._pcm_file is None
            ):
                os.makedirs(self.export_dir, exist_ok=True)
                self._pcm_file = open(f"{self._cont_stem}._a.pcm", "wb")

    def _open_pcm_if_needed_unlocked(self) -> None:
        if self._pcm_file is not None or not self._cont_stem:
            return
        if self._audio_sr is None:
            return
        os.makedirs(self.export_dir, exist_ok=True)
        self._pcm_file = open(f"{self._cont_stem}._a.pcm", "wb")

    def _ensure_opencv_warmup_flushed_unlocked(self) -> None:
        """Sans ffmpeg : ouvre VideoWriter et vide le tampon (FPS mesuré, pas --frame-rate)."""
        if ffmpeg_io.ffmpeg_available():
            self._opencv_warmup_frames.clear()
            return
        if self._cont_writer is not None or not self._opencv_warmup_frames:
            return
        if not self._cont_stem:
            self._opencv_warmup_frames.clear()
            return
        last_f = self._opencv_warmup_frames[-1][0]
        h, w = last_f.shape[:2]
        n = len(self._opencv_warmup_frames)
        span = self._opencv_warmup_frames[-1][1] - self._opencv_warmup_frames[0][1]
        if n >= 2 and span > 1e-6:
            fps_eff = (n - 1) / span
        else:
            fps_eff = float(self.frame_rate)
        fps_eff = min(75.0, max(1.0, fps_eff))
        self._cont_video_temp = f"{self._cont_stem}._v.avi"
        self._cont_writer = cv2.VideoWriter(
            self._cont_video_temp,
            cv2.VideoWriter_fourcc(*"XVID"),
            fps_eff,
            (w, h),
        )
        if not self._cont_writer.isOpened():
            logger.warning(
                "[%s] REC OpenCV : impossible d’ouvrir %s",
                self.stream_id,
                self._cont_video_temp,
            )
            self._cont_writer = None
            self._opencv_warmup_frames.clear()
            return
        logger.info(
            "[%s] REC OpenCV AVI (~%.1f img/s réelles, pas %s Hz serveur)",
            self.stream_id,
            fps_eff,
            self.frame_rate,
        )
        for f, ts in self._opencv_warmup_frames:
            if self._cont_frame_count == 0:
                self._cont_wall_t0 = ts
            self._cont_wall_t1 = ts
            self._cont_frame_count += 1
            self._cont_writer.write(f)
        self._opencv_warmup_frames.clear()

    def _release_cont_writer_unlocked(self) -> None:
        if self._cont_writer is not None:
            if hasattr(self._cont_writer, "release"):
                self._cont_writer.release()
            else:
                code, err = self._cont_writer.close()
                if code != 0 and err:
                    logger.warning(
                        "[%s] ffmpeg fin anormale: %s",
                        self.stream_id,
                        err.decode(errors="replace")[:300],
                    )
            self._cont_writer = None
        self._cont_path = None
        self._cont_video_temp = None

    def _finalize_continuous_unlocked(self) -> None:
        stem = self._cont_stem
        self._ensure_opencv_warmup_flushed_unlocked()
        n_frames = self._cont_frame_count
        w0, w1 = self._cont_wall_t0, self._cont_wall_t1
        self._release_cont_writer_unlocked()
        self._cont_frame_count = 0
        self._cont_wall_t0 = None
        self._cont_wall_t1 = None
        if self._pcm_file is not None:
            try:
                self._pcm_file.close()
            except OSError:
                pass
            self._pcm_file = None
        if not stem:
            self._cont_stem = None
            return
        v_mp4 = f"{stem}._v.mp4"
        v_avi = f"{stem}._v.avi"
        pcm = f"{stem}._a.pcm"
        out_mp4 = f"{stem}.mp4"
        out_avi = f"{stem}.avi"
        video_src = v_mp4 if os.path.isfile(v_mp4) else (v_avi if os.path.isfile(v_avi) else None)
        if not video_src:
            self._cont_stem = None
            return
        if ffmpeg_io.ffmpeg_available():
            video_src = ffmpeg_io.retime_continuous_video_file(
                video_src,
                frame_count=n_frames,
                nominal_fps=self.frame_rate,
                wall_start=w0,
                wall_end=w1,
            )
        if (
            self._audio_sr is not None
            and os.path.isfile(pcm)
            and os.path.getsize(pcm) > 0
            and ffmpeg_io.ffmpeg_available()
        ):
            if ffmpeg_io.mux_video_pcm_to_mp4(
                video_src,
                pcm,
                self._audio_sr,
                self._audio_ch or 1,
                out_mp4,
            ):
                try:
                    os.remove(video_src)
                    os.remove(pcm)
                except OSError:
                    pass
                logger.info(
                    "[%s] Enregistrement continu terminé (vidéo+audio) : %s",
                    self.stream_id,
                    out_mp4,
                )
                self._cont_stem = None
                return
        try:
            if video_src.endswith(".mp4"):
                os.replace(video_src, out_mp4)
                final = out_mp4
            else:
                os.replace(video_src, out_avi)
                final = out_avi
            if os.path.isfile(pcm):
                try:
                    os.remove(pcm)
                except OSError:
                    pass
            logger.info(
                "[%s] Enregistrement continu terminé : %s", self.stream_id, final
            )
        except OSError as e:
            logger.warning(
                "[%s] Impossible de finaliser l’enregistrement : %s",
                self.stream_id,
                e,
            )
        self._cont_stem = None

    def _write_continuous_frame_unlocked(self, fr) -> None:
        if not self._continuous:
            return
        h, w = fr.shape[:2]
        if w <= 0 or h <= 0:
            return
        if self._cont_writer is None:
            if ffmpeg_io.ffmpeg_available():
                self._cont_video_temp = f"{self._cont_stem}._v.mp4"
                try:
                    self._cont_writer = ffmpeg_io.FfmpegBGRWriter(
                        self._cont_video_temp, w, h, self.frame_rate
                    )
                except (RuntimeError, OSError) as e:
                    logger.warning(
                        "[%s] ffmpeg pipe indisponible (%s), repli OpenCV",
                        self.stream_id,
                        e,
                    )
                    self._cont_writer = None
                else:
                    self._cont_path = self._cont_video_temp
                    logger.info(
                        "[%s] REC continu → %s.mp4 (H.264, Android)",
                        self.stream_id,
                        self._cont_stem,
                    )
            if self._cont_writer is None:
                self._opencv_warmup_frames.append((fr.copy(), time.time()))
                span = (
                    self._opencv_warmup_frames[-1][1]
                    - self._opencv_warmup_frames[0][1]
                )
                nw = len(self._opencv_warmup_frames)
                if nw < 2 or (span < 0.2 and nw < 25):
                    return
                fps_eff = (nw - 1) / max(0.05, span)
                fps_eff = min(75.0, max(1.0, fps_eff))
                self._cont_video_temp = f"{self._cont_stem}._v.avi"
                self._cont_writer = cv2.VideoWriter(
                    self._cont_video_temp,
                    cv2.VideoWriter_fourcc(*"XVID"),
                    fps_eff,
                    (w, h),
                )
                self._cont_path = self._cont_video_temp
                if not self._cont_writer.isOpened():
                    logger.warning(
                        "[%s] REC continu : impossible d’ouvrir %s",
                        self.stream_id,
                        self._cont_video_temp,
                    )
                    self._continuous = False
                    self._cont_writer = None
                    self._opencv_warmup_frames.clear()
                    return
                logger.info(
                    "[%s] REC continu → %s.avi (OpenCV ~%.1f img/s)",
                    self.stream_id,
                    self._cont_stem,
                    fps_eff,
                )
                for f, ts in self._opencv_warmup_frames:
                    if self._cont_frame_count == 0:
                        self._cont_wall_t0 = ts
                    self._cont_wall_t1 = ts
                    self._cont_frame_count += 1
                    self._cont_writer.write(f)
                self._opencv_warmup_frames.clear()
                return

        if self._cont_writer is not None and hasattr(self._cont_writer, "write"):
            now = time.time()
            if self._cont_frame_count == 0:
                self._cont_wall_t0 = now
            self._cont_wall_t1 = now
            self._cont_frame_count += 1
            self._cont_writer.write(fr)

    def start_continuous(self, session_ts: int) -> None:
        with self._lock:
            self._finalize_continuous_unlocked()
            self._continuous = True
            self._cont_session_ts = session_ts
            os.makedirs(self.export_dir, exist_ok=True)
            self._cont_stem = os.path.join(
                self.export_dir, f"cont_{self.stream_id}_{session_ts}"
            )
            self._cont_writer = None
            self._cont_path = None
            self._cont_video_temp = None
            self._cont_frame_count = 0
            self._cont_wall_t0 = None
            self._cont_wall_t1 = None
            self._opencv_warmup_frames.clear()
            if self._pcm_file is not None:
                try:
                    self._pcm_file.close()
                except OSError:
                    pass
                self._pcm_file = None
            self._open_pcm_if_needed_unlocked()

    def stop_continuous(self) -> None:
        with self._lock:
            was = self._continuous
            self._continuous = False
            if was:
                self._finalize_continuous_unlocked()

    def append_audio(self, pcm: bytes) -> None:
        if not pcm:
            return
        with self._lock:
            if not self._continuous:
                return
            self._open_pcm_if_needed_unlocked()
            if self._pcm_file is not None:
                self._pcm_file.write(pcm)

    def append(self, frame, *, copy_frame: bool = True) -> None:
        with self._lock:
            now = time.time()
            fr = frame.copy() if copy_frame else frame
            self._frames.append((now, fr))
            self._trim_to_duration_unlocked(now)
            self._write_continuous_frame_unlocked(fr)

    def snapshot_timed(self) -> list[tuple[float, Any]]:
        with self._lock:
            self._trim_to_duration_unlocked(time.time())
            return list(self._frames)

    def snapshot_frames(self):
        return [fr for _, fr in self.snapshot_timed()]

    def latest(self):
        with self._lock:
            self._trim_to_duration_unlocked(time.time())
            return self._frames[-1][1] if self._frames else None

    def latest_with_ts(self) -> tuple[Any, float | None]:
        with self._lock:
            self._trim_to_duration_unlocked(time.time())
            if not self._frames:
                return None, None
            ts, fr = self._frames[-1]
            return fr, ts

    def frame_at_delay_with_ts(self, delay_sec: float) -> tuple[Any, float | None]:
        if delay_sec <= 0:
            return self.latest_with_ts()
        target = time.time() - delay_sec
        with self._lock:
            self._trim_to_duration_unlocked(time.time())
            if not self._frames:
                return None, None
            chosen_fr = None
            chosen_ts: float | None = None
            for ts, fr in self._frames:
                if ts <= target:
                    chosen_fr = fr
                    chosen_ts = ts
                else:
                    break
            if chosen_fr is not None:
                return chosen_fr, chosen_ts
            ts0, fr0 = self._frames[0]
            return fr0, ts0

    def measured_input_fps(self) -> float:
        """Débit moyen d’images reçues (sur la fenêtre du tampon temporel)."""
        with self._lock:
            self._trim_to_duration_unlocked(time.time())
            n = len(self._frames)
            if n < 2:
                return 0.0
            t0 = self._frames[0][0]
            t1 = self._frames[-1][0]
            dt = t1 - t0
            if dt < 1e-6:
                return 0.0
            return (n - 1) / dt

    def frame_at_delay(self, delay_sec: float):
        if delay_sec <= 0:
            return self.latest()
        target = time.time() - delay_sec
        with self._lock:
            self._trim_to_duration_unlocked(time.time())
            if not self._frames:
                return None
            chosen = None
            for ts, fr in self._frames:
                if ts <= target:
                    chosen = fr
                else:
                    break
            if chosen is not None:
                return chosen
            return self._frames[0][1]

    def save_last_seconds(self) -> str | None:
        timed = self.snapshot_timed()
        if not timed:
            logger.info("[%s] Aucune image à enregistrer.", self.stream_id)
            return None
        frames = [fr for _, fr in timed]
        times = [t for t, _ in timed]
        fps = effective_fps_from_timestamps(times, float(self.frame_rate))
        h, w = frames[0].shape[:2]
        if w == 0 or h == 0:
            logger.warning("[%s] Dimensions invalides.", self.stream_id)
            return None
        os.makedirs(self.export_dir, exist_ok=True)
        ts = int(time.time())
        if ffmpeg_io.ffmpeg_available():
            path = os.path.join(self.export_dir, f"{self.stream_id}_{ts}.mp4")
            if ffmpeg_io.write_frames_bgr_to_mp4(
                path, frames, fps, self.stream_id
            ):
                return path
        path = os.path.join(self.export_dir, f"{self.stream_id}_{ts}.avi")
        writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h)
        )
        if not writer.isOpened():
            logger.warning(
                "[%s] Impossible d’ouvrir le writer pour %s.", self.stream_id, path
            )
            return None
        try:
            for f in frames:
                writer.write(f)
        finally:
            writer.release()
        logger.info("[%s] Vidéo enregistrée : %s", self.stream_id, path)
        return path

    def timed_frames_since(self, seconds: float) -> list[tuple[float, Any]]:
        now = time.time()
        cutoff = now - max(0.01, float(seconds))
        with self._lock:
            self._trim_to_duration_unlocked(now)
            return [(t, f) for t, f in self._frames if t >= cutoff]

    def save_clip_last_seconds(self, seconds: float) -> str | None:
        timed = self.timed_frames_since(seconds)
        if not timed:
            logger.info(
                "[%s] Aucune image pour les %g dernières secondes.",
                self.stream_id,
                seconds,
            )
            return None
        frames = [fr for _, fr in timed]
        times = [t for t, _ in timed]
        fps = effective_fps_from_timestamps(times, float(self.frame_rate))
        h, w = frames[0].shape[:2]
        if w == 0 or h == 0:
            logger.warning("[%s] Dimensions invalides.", self.stream_id)
            return None
        snap = os.path.join(self.export_dir, "clips")
        os.makedirs(snap, exist_ok=True)
        ts = int(time.time())
        if ffmpeg_io.ffmpeg_available():
            path = os.path.join(snap, f"{self.stream_id}_clip{int(seconds)}s_{ts}.mp4")
            if ffmpeg_io.write_frames_bgr_to_mp4(
                path, frames, fps, self.stream_id
            ):
                logger.info("[%s] Clip enregistré : %s", self.stream_id, path)
                return path
        path = os.path.join(snap, f"{self.stream_id}_clip{int(seconds)}s_{ts}.avi")
        writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h)
        )
        if not writer.isOpened():
            logger.warning(
                "[%s] Impossible d’ouvrir le writer pour %s.", self.stream_id, path
            )
            return None
        try:
            for f in frames:
                writer.write(f)
        finally:
            writer.release()
        logger.info("[%s] Clip enregistré : %s", self.stream_id, path)
        return path

    def save_latest_png(self) -> str | None:
        with self._lock:
            self._trim_to_duration_unlocked(time.time())
            if not self._frames:
                logger.info("[%s] Aucune image pour snapshot PNG.", self.stream_id)
                return None
            _, fr = self._frames[-1]
            img = fr.copy()
        snap = os.path.join(self.export_dir, "snapshots")
        os.makedirs(snap, exist_ok=True)
        ts = int(time.time())
        path = os.path.join(snap, f"{self.stream_id}_{ts}.png")
        if not cv2.imwrite(path, img):
            logger.warning("[%s] Échec écriture %s", self.stream_id, path)
            return None
        logger.info("[%s] PNG : %s", self.stream_id, path)
        return path
