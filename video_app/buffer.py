"""Tampon circulaire par flux et export vidéo."""

from __future__ import annotations

import os
import threading
import time
from collections import deque

import cv2

from video_app import ffmpeg_io


class StreamBuffer:
    def __init__(self, stream_id: str, frame_rate: int, buffer_duration: int):
        self.stream_id = stream_id
        self.frame_rate = max(1, frame_rate)
        self._buffer_duration = max(1, buffer_duration)
        self._lock = threading.Lock()
        self._frames: deque = deque(maxlen=self.frame_rate * self._buffer_duration)
        self._continuous = False
        self._cont_session_ts = 0
        self._cont_stem: str | None = None
        self._cont_writer = None  # FfmpegBGRWriter ou cv2.VideoWriter
        self._cont_path: str | None = None
        self._cont_video_temp: str | None = None
        self._audio_sr: int | None = None
        self._audio_ch: int | None = None
        self._pcm_file = None

    @property
    def buffer_duration(self) -> int:
        return self._buffer_duration

    def set_buffer_duration(self, seconds: int) -> None:
        seconds = max(1, seconds)
        with self._lock:
            self._buffer_duration = seconds
            kept = list(self._frames)[-self.frame_rate * seconds :]
            self._frames = deque(kept, maxlen=self.frame_rate * seconds)

    def set_audio_params(self, sample_rate: int, channels: int) -> None:
        with self._lock:
            self._audio_sr = int(sample_rate)
            self._audio_ch = int(channels)
            if (
                self._continuous
                and self._cont_stem
                and self._pcm_file is None
            ):
                os.makedirs("./video", exist_ok=True)
                self._pcm_file = open(f"{self._cont_stem}._a.pcm", "wb")

    def _open_pcm_if_needed_unlocked(self) -> None:
        if self._pcm_file is not None or not self._cont_stem:
            return
        if self._audio_sr is None:
            return
        os.makedirs("./video", exist_ok=True)
        self._pcm_file = open(f"{self._cont_stem}._a.pcm", "wb")

    def _release_cont_writer_unlocked(self) -> None:
        if self._cont_writer is not None:
            if hasattr(self._cont_writer, "release"):
                self._cont_writer.release()
            else:
                code, err = self._cont_writer.close()
                if code != 0 and err:
                    print(
                        f"[{self.stream_id}] ffmpeg fin anormale: "
                        f"{err.decode(errors='replace')[:300]}"
                    )
            self._cont_writer = None
        self._cont_path = None
        self._cont_video_temp = None

    def _finalize_continuous_unlocked(self) -> None:
        stem = self._cont_stem
        self._release_cont_writer_unlocked()
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
                print(f"[{self.stream_id}] Enregistrement continu terminé (vidéo+audio) : {out_mp4}")
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
            print(f"[{self.stream_id}] Enregistrement continu terminé : {final}")
        except OSError as e:
            print(f"[{self.stream_id}] Impossible de finaliser l’enregistrement : {e}")
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
                self._cont_writer = ffmpeg_io.FfmpegBGRWriter(
                    self._cont_video_temp, w, h, self.frame_rate
                )
                self._cont_path = self._cont_video_temp
                print(
                    f"[{self.stream_id}] REC continu → {self._cont_stem}.mp4 "
                    "(H.264, Android)"
                )
            else:
                self._cont_video_temp = f"{self._cont_stem}._v.avi"
                self._cont_writer = cv2.VideoWriter(
                    self._cont_video_temp,
                    cv2.VideoWriter_fourcc(*"XVID"),
                    self.frame_rate,
                    (w, h),
                )
                self._cont_path = self._cont_video_temp
                if not self._cont_writer.isOpened():
                    print(
                        f"[{self.stream_id}] REC continu : impossible d’ouvrir "
                        f"{self._cont_video_temp} (installez ffmpeg pour du MP4)."
                    )
                    self._continuous = False
                    self._cont_writer = None
                    return
                print(
                    f"[{self.stream_id}] REC continu → {self._cont_stem}.avi "
                    "(ffmpeg absent — format AVI)"
                )
        if hasattr(self._cont_writer, "write"):
            self._cont_writer.write(fr)

    def start_continuous(self, session_ts: int) -> None:
        with self._lock:
            self._finalize_continuous_unlocked()
            self._continuous = True
            self._cont_session_ts = session_ts
            os.makedirs("./video", exist_ok=True)
            self._cont_stem = f"./video/cont_{self.stream_id}_{session_ts}"
            self._cont_writer = None
            self._cont_path = None
            self._cont_video_temp = None
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

    def append(self, frame) -> None:
        with self._lock:
            fr = frame.copy()
            self._frames.append((time.time(), fr))
            self._write_continuous_frame_unlocked(fr)

    def snapshot_frames(self):
        with self._lock:
            return [fr for _, fr in self._frames]

    def latest(self):
        with self._lock:
            return self._frames[-1][1] if self._frames else None

    def frame_at_delay(self, delay_sec: float):
        if delay_sec <= 0:
            return self.latest()
        target = time.time() - delay_sec
        with self._lock:
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
        frames = self.snapshot_frames()
        if not frames:
            print(f"[{self.stream_id}] Aucune image à enregistrer.")
            return None
        h, w = frames[0].shape[:2]
        if w == 0 or h == 0:
            print(f"[{self.stream_id}] Dimensions invalides.")
            return None
        os.makedirs("./video", exist_ok=True)
        ts = int(time.time())
        if ffmpeg_io.ffmpeg_available():
            path = f"./video/{self.stream_id}_{ts}.mp4"
            if ffmpeg_io.write_frames_bgr_to_mp4(
                path, frames, self.frame_rate, self.stream_id
            ):
                return path
        path = f"./video/{self.stream_id}_{ts}.avi"
        writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"XVID"), self.frame_rate, (w, h)
        )
        if not writer.isOpened():
            print(f"[{self.stream_id}] Impossible d’ouvrir le writer pour {path}.")
            return None
        try:
            for f in frames:
                writer.write(f)
        finally:
            writer.release()
        print(f"[{self.stream_id}] Vidéo enregistrée : {path}")
        return path
