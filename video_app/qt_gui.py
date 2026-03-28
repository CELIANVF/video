"""Interface graphique PyQt6 : grille, flux détachables, contrôles."""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Callable

import numpy as np

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QCloseEvent, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from video_app.display_core import (
    close_continuous_stack_state,
    gather_display_frames,
    tick_continuous_stack_recording,
)
from video_app.export_video import save_per_stream_and_stack


def _numpy_bgr_to_pixmap(arr: np.ndarray) -> QPixmap:
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    h, w, c = arr.shape
    if c != 3:
        raise ValueError("BGR 3 canaux attendu")
    qimg = QImage(arr.data, w, h, w * 3, QImage.Format.Format_BGR888)
    return QPixmap.fromImage(qimg.copy())


class VideoLabel(QLabel):
    def __init__(self) -> None:
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(280, 200)
        self.setStyleSheet("background-color: #1e1e1e; color: #888;")
        self.setScaledContents(False)
        self._pix: QPixmap | None = None

    def set_frame(self, bgr: np.ndarray) -> None:
        self._pix = _numpy_bgr_to_pixmap(bgr.copy())
        self._apply_scale()

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._apply_scale()

    def _apply_scale(self) -> None:
        if self._pix is None or self._pix.isNull():
            return
        scaled = self._pix.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        super().setPixmap(scaled)


class StreamTile(QFrame):
    def __init__(self, stream_id: str, main: "ServerMainWindow") -> None:
        super().__init__()
        self.stream_id = stream_id
        self._main = main
        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        lay = QVBoxLayout(self)
        head = QHBoxLayout()
        head.addWidget(QLabel(f"<b>{stream_id}</b>"))
        self._lbl_fps = QLabel("—")
        self._lbl_fps.setStyleSheet("color: #9ad; font-weight: bold; min-width: 72px;")
        self._lbl_fps.setToolTip("Images reçues par seconde (moyenne sur le tampon)")
        head.addWidget(self._lbl_fps)
        head.addStretch()
        btn = QPushButton("Fenêtre séparée")
        btn.setToolTip("Détacher ce flux (redimensionnable, déplaçable)")
        btn.clicked.connect(lambda: main.detach_stream(stream_id))
        head.addWidget(btn)
        lay.addLayout(head)
        self.video = VideoLabel()
        lay.addWidget(self.video, 1)
        self._res_lbl = QLabel("")
        self._res_lbl.setStyleSheet("color: #888; font-size: 11px;")
        lay.addWidget(self._res_lbl)

    def set_frame(self, bgr: np.ndarray) -> None:
        h, w = bgr.shape[:2]
        self._res_lbl.setText(f"{w}×{h} px")
        self.video.setToolTip(
            f"Résolution du flux : {w}×{h} px (mise à l’échelle à l’écran, sans déformation)"
        )
        self.video.set_frame(bgr)

    def set_stream_fps(self, fps: float) -> None:
        if fps < 0.05:
            self._lbl_fps.setText("—")
        else:
            self._lbl_fps.setText(f"{fps:.1f} img/s")


class DetachedStreamWindow(QWidget):
    def __init__(self, stream_id: str, main: "ServerMainWindow") -> None:
        super().__init__(None, Qt.WindowType.Window)
        self.stream_id = stream_id
        self._main = main
        self.setWindowTitle(f"Flux — {stream_id}")
        self.resize(720, 520)
        lay = QVBoxLayout(self)
        head = QHBoxLayout()
        head.addWidget(QLabel(f"<b>{stream_id}</b>"))
        self._lbl_fps = QLabel("—")
        self._lbl_fps.setStyleSheet("color: #9ad; font-weight: bold; min-width: 72px;")
        self._lbl_fps.setToolTip("Images reçues par seconde (moyenne sur le tampon)")
        head.addWidget(self._lbl_fps)
        head.addStretch()
        btn = QPushButton("Réintégrer à la grille")
        btn.clicked.connect(lambda: main.attach_stream(stream_id))
        head.addWidget(btn)
        lay.addLayout(head)
        self.video = VideoLabel()
        lay.addWidget(self.video, 1)
        self._res_lbl = QLabel("")
        self._res_lbl.setStyleSheet("color: #888; font-size: 11px;")
        lay.addWidget(self._res_lbl)

    def set_frame(self, bgr: np.ndarray) -> None:
        h, w = bgr.shape[:2]
        self._res_lbl.setText(f"{w}×{h} px")
        self.video.setToolTip(
            f"Résolution du flux : {w}×{h} px (mise à l’échelle à l’écran, sans déformation)"
        )
        self.video.set_frame(bgr)

    def set_stream_fps(self, fps: float) -> None:
        if fps < 0.05:
            self._lbl_fps.setText("—")
            self.setWindowTitle(f"Flux — {self.stream_id}")
        else:
            self._lbl_fps.setText(f"{fps:.1f} img/s")
            self.setWindowTitle(f"Flux — {self.stream_id} ({fps:.1f} img/s)")

    def closeEvent(self, e: QCloseEvent) -> None:
        self._main.on_detached_window_closed(self.stream_id)
        e.accept()


class ServerMainWindow(QMainWindow):
    def __init__(
        self,
        *,
        registry: Any,
        frame_rate: int,
        buffer_duration: int,
        grid_max_w: int,
        grid_max_h: int,
        stop_event: threading.Event,
        stack_state: dict[str, Any],
        on_shutdown: Callable[[], None],
    ) -> None:
        super().__init__()
        self.registry = registry
        self.frame_rate = max(1, frame_rate)
        self.buffer_duration_live = buffer_duration
        self.grid_max_w = grid_max_w
        self.grid_max_h = grid_max_h
        self.stop_event = stop_event
        self._stack_state = stack_state
        self._on_shutdown = on_shutdown

        self.live_display = True
        self.display_delay_sec = float(max(1, min(5, buffer_duration)))

        self._tiles: dict[str, StreamTile] = {}
        self._detached: dict[str, DetachedStreamWindow] = {}
        self._fps_count = 0
        self._fps_last = time.time()

        self.setWindowTitle("Serveur multi-flux")
        self.resize(min(1500, grid_max_w + 400), min(920, grid_max_h + 140))

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._grid_host = QWidget()
        self._grid_layout = QGridLayout(self._grid_host)
        self._grid_layout.setSpacing(10)
        scroll.setWidget(self._grid_host)
        root.addWidget(scroll, 1)

        self._placeholder = QLabel(
            "En attente de flux…\n(lancez camera.py ou --local)"
        )
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet("color: #666; font-size: 16px; padding: 48px;")
        self._grid_layout.addWidget(self._placeholder, 0, 0)

        side = self._build_side_panel()
        root.addWidget(side)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(max(8, int(1000 / self.frame_rate)))

    def _build_side_panel(self) -> QWidget:
        w = QWidget()
        w.setMinimumWidth(300)
        w.setMaximumWidth(400)
        lay = QVBoxLayout(w)

        g = QGroupBox("Enregistrement")
        gl = QVBoxLayout(g)
        self.chk_rec = QCheckBox("REC continu")
        self.chk_rec.toggled.connect(self._on_rec_toggled)
        gl.addWidget(self.chk_rec)
        b_snap = QPushButton("Snapshot (s) — MP4 par flux + empilé")
        b_snap.clicked.connect(self._snapshot)
        gl.addWidget(b_snap)
        lay.addWidget(g)

        g2 = QGroupBox("Affichage")
        g2l = QVBoxLayout(g2)
        self.chk_live = QCheckBox("Mode direct (live)")
        self.chk_live.setChecked(True)
        self.chk_live.toggled.connect(lambda v: setattr(self, "live_display", v))
        g2l.addWidget(self.chk_live)
        g2l.addWidget(QLabel("Délai (s) :"))
        self.spin_delay = QSpinBox()
        self.spin_delay.setRange(1, 999)
        self.spin_delay.setValue(int(self.display_delay_sec))
        self.spin_delay.valueChanged.connect(self._on_delay_changed)
        g2l.addWidget(self.spin_delay)
        lay.addWidget(g2)

        g3 = QGroupBox("Tampon")
        g3l = QVBoxLayout(g3)
        g3l.addWidget(QLabel("Durée tampon (s) :"))
        self.spin_buffer = QSpinBox()
        self.spin_buffer.setRange(1, 600)
        self.spin_buffer.setValue(self.buffer_duration_live)
        self.spin_buffer.valueChanged.connect(self._on_buffer_changed)
        g3l.addWidget(self.spin_buffer)
        lay.addWidget(g3)

        self.lbl_status = QLabel("—")
        self.lbl_status.setWordWrap(True)
        lay.addWidget(self.lbl_status)

        lay.addStretch()
        b_quit = QPushButton("Quitter")
        b_quit.clicked.connect(self.close)
        lay.addWidget(b_quit)

        tip = QLabel(
            "• Détacher : bouton sur chaque vignette\n"
            "• Fenêtres séparées : redimensionnez librement\n"
            "• Fermer une fenêtre détachée = réintégration"
        )
        tip.setStyleSheet("color: #777; font-size: 11px;")
        lay.addWidget(tip)

        return w

    def _on_rec_toggled(self, on: bool) -> None:
        was = self.registry.is_continuous_recording()
        self.registry.set_continuous_recording(on)
        if was and not on:
            close_continuous_stack_state(self._stack_state)

    def _on_delay_changed(self, v: int) -> None:
        self.display_delay_sec = float(v)

    def _on_buffer_changed(self, v: int) -> None:
        self.buffer_duration_live = v
        self.registry.set_all_buffer_duration(v)
        self.spin_delay.setMaximum(max(1, v))
        if self.spin_delay.value() > v:
            self.spin_delay.setValue(v)

    def _snapshot(self) -> None:
        to_save = []
        for sid in self.registry.ids():
            b = self.registry.get(sid)
            if b is not None:
                to_save.append(b)
        if not to_save:
            return
        fps = self.frame_rate

        def _run() -> None:
            save_per_stream_and_stack(to_save, fps)

        threading.Thread(target=_run, daemon=True).start()

    def _relayout_grid(self) -> None:
        while self._grid_layout.count():
            it = self._grid_layout.takeAt(0)
            w = it.widget()
            if w is not None:
                w.setParent(None)
        ids = [x for x in self.registry.ids() if x in self._tiles]
        if not ids:
            if self._detached:
                self._placeholder.setText(
                    "Tous les flux sont dans des fenêtres séparées."
                )
            else:
                self._placeholder.setText(
                    "En attente de flux…\n(lancez camera.py ou --local)"
                )
            self._grid_layout.addWidget(self._placeholder, 0, 0)
            return
        n = len(ids)
        cols = max(1, math.ceil(math.sqrt(n)))
        for i, sid in enumerate(ids):
            r, c = divmod(i, cols)
            self._grid_layout.addWidget(self._tiles[sid], r, c)

    def detach_stream(self, sid: str) -> None:
        if sid in self._detached:
            return
        tile = self._tiles.pop(sid, None)
        if tile:
            self._grid_layout.removeWidget(tile)
            tile.deleteLater()
            self._relayout_grid()
        win = DetachedStreamWindow(sid, self)
        self._detached[sid] = win
        win.show()

    def on_detached_window_closed(self, sid: str) -> None:
        self._detached.pop(sid, None)
        self._ensure_tile(sid)

    def _ensure_tile(self, sid: str) -> None:
        if sid not in self.registry.ids():
            return
        if sid not in self._tiles:
            self._tiles[sid] = StreamTile(sid, self)
            self._relayout_grid()

    def attach_stream(self, sid: str) -> None:
        win = self._detached.pop(sid, None)
        if win:
            win.hide()
            win.deleteLater()
        self._ensure_tile(sid)

    def _tick(self) -> None:
        if self.stop_event.is_set():
            self.close()
            return

        delay = float(self.spin_delay.value())
        frames = gather_display_frames(
            self.registry, self.live_display, delay
        )
        tick_continuous_stack_recording(
            self.registry, frames, self.frame_rate, self._stack_state
        )

        ids = set(self.registry.ids())

        for sid in list(self._tiles.keys()):
            if sid not in ids:
                t = self._tiles.pop(sid)
                self._grid_layout.removeWidget(t)
                t.deleteLater()
                self._relayout_grid()
        for sid in list(self._detached.keys()):
            if sid not in ids:
                w = self._detached.pop(sid)
                w.close()
                w.deleteLater()

        for sid in self.registry.ids():
            if sid in self._detached or sid in self._tiles:
                continue
            self._tiles[sid] = StreamTile(sid, self)
            self._relayout_grid()

        fd = {sid: fr for sid, fr in frames}
        for sid, fr in fd.items():
            if sid in self._detached:
                self._detached[sid].set_frame(fr)
            elif sid in self._tiles:
                self._tiles[sid].set_frame(fr)

        for sid in ids:
            buf = self.registry.get(sid)
            if buf is None:
                continue
            fp = buf.measured_input_fps()
            if sid in self._tiles:
                self._tiles[sid].set_stream_fps(fp)
            if sid in self._detached:
                self._detached[sid].set_stream_fps(fp)

        self._fps_count += 1
        if time.time() - self._fps_last >= 1.0:
            self.lbl_status.setText(
                f"FPS ≈ {self._fps_count}\n"
                f"Flux actifs : {len(ids)}\n"
                f"Mode : {'LIVE' if self.live_display else f'délai {delay:.0f}s'}\n"
                f"REC : {'oui' if self.registry.is_continuous_recording() else 'non'}"
            )
            self._fps_count = 0
            self._fps_last = time.time()

        self.chk_rec.blockSignals(True)
        self.chk_rec.setChecked(self.registry.is_continuous_recording())
        self.chk_rec.blockSignals(False)
        self.chk_live.blockSignals(True)
        self.chk_live.setChecked(self.live_display)
        self.chk_live.blockSignals(False)

    def closeEvent(self, e: QCloseEvent) -> None:
        self._timer.stop()
        close_continuous_stack_state(self._stack_state)
        self.registry.set_continuous_recording(False)
        self.stop_event.set()
        self._on_shutdown()
        e.accept()
        QApplication.quit()


def run_qt_application(
    *,
    registry: Any,
    frame_rate: int,
    buffer_duration: int,
    grid_max_w: int,
    grid_max_h: int,
    stop_event: threading.Event,
    stack_state: dict[str, Any],
    on_shutdown: Callable[[], None],
) -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    win = ServerMainWindow(
        registry=registry,
        frame_rate=frame_rate,
        buffer_duration=buffer_duration,
        grid_max_w=grid_max_w,
        grid_max_h=grid_max_h,
        stop_event=stop_event,
        stack_state=stack_state,
        on_shutdown=on_shutdown,
    )
    win.show()
    app.exec()
