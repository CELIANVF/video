"""Tampon circulaire par flux et export vidéo."""

from __future__ import annotations

import os
import threading
import time
from collections import deque

import cv2


class StreamBuffer:
    def __init__(self, stream_id: str, frame_rate: int, buffer_duration: int):
        self.stream_id = stream_id
        self.frame_rate = max(1, frame_rate)
        self._buffer_duration = max(1, buffer_duration)
        self._lock = threading.Lock()
        self._frames: deque = deque(maxlen=self.frame_rate * self._buffer_duration)

    @property
    def buffer_duration(self) -> int:
        return self._buffer_duration

    def set_buffer_duration(self, seconds: int) -> None:
        seconds = max(1, seconds)
        with self._lock:
            self._buffer_duration = seconds
            self._frames = deque(
                list(self._frames)[-self.frame_rate * seconds :],
                maxlen=self.frame_rate * seconds,
            )

    def append(self, frame) -> None:
        with self._lock:
            self._frames.append(frame.copy())

    def snapshot_frames(self):
        with self._lock:
            return list(self._frames)

    def latest(self):
        with self._lock:
            return self._frames[-1] if self._frames else None

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
        path = f"./video/{self.stream_id}_{int(time.time())}.avi"
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
