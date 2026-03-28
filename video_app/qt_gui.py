"""Interface graphique PyQt6 : grille, flux détachables, contrôles."""

from __future__ import annotations

import math
import queue
import threading
import time
from typing import Any, Callable

import numpy as np

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QCloseEvent, QImage, QKeySequence, QPixmap, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from video_app.display_core import (
    close_continuous_stack_state,
    gather_display_frames_with_ts,
    ordered_stream_ids,
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
    def __init__(self, on_double_click: Callable[[], None] | None = None) -> None:
        super().__init__()
        self._on_double_click = on_double_click
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(280, 200)
        self.setStyleSheet("background-color: #1e1e1e; color: #888;")
        self.setScaledContents(False)
        self._pix: QPixmap | None = None

    def mouseDoubleClickEvent(self, e) -> None:
        if self._on_double_click is not None:
            self._on_double_click()
        super().mouseDoubleClickEvent(e)

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
        self._fullscreen_dlg: QDialog | None = None
        self._last_bgr: np.ndarray | None = None
        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        lay = QVBoxLayout(self)
        head = QHBoxLayout()
        self._title = QLabel(f"<b>{main.stream_display_name(stream_id)}</b>")
        head.addWidget(self._title)
        self._lbl_fps = QLabel("—")
        self._lbl_fps.setStyleSheet("color: #9ad; font-weight: bold; min-width: 72px;")
        self._lbl_fps.setToolTip("Images reçues par seconde (moyenne sur le tampon)")
        head.addWidget(self._lbl_fps)
        self._lbl_lat = QLabel("—")
        self._lbl_lat.setStyleSheet("color: #c9a; font-weight: bold; min-width: 56px;")
        self._lbl_lat.setToolTip(
            "Âge de l’image affichée côté serveur (réception → affichage), en ms"
        )
        head.addWidget(self._lbl_lat)
        head.addStretch()
        btn = QPushButton("Fenêtre séparée")
        btn.setToolTip("Détacher ce flux (redimensionnable, déplaçable)")
        btn.clicked.connect(lambda: main.detach_stream(stream_id))
        head.addWidget(btn)
        lay.addLayout(head)
        self.video = VideoLabel(on_double_click=self._toggle_fullscreen)
        lay.addWidget(self.video, 1)
        self._res_lbl = QLabel("")
        self._res_lbl.setStyleSheet("color: #888; font-size: 11px;")
        lay.addWidget(self._res_lbl)

        sc = QShortcut(QKeySequence("F11"), self)
        sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        sc.activated.connect(self._toggle_fullscreen)

    def _toggle_fullscreen(self) -> None:
        if self._fullscreen_dlg is not None:
            self._fullscreen_dlg.close()
            self._fullscreen_dlg = None
            return
        if self._last_bgr is None:
            return
        dlg = QDialog(self.window())
        dlg.setWindowTitle(self._main.stream_display_name(self.stream_id))
        dlg.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(0, 0, 0, 0)
        big = VideoLabel(on_double_click=dlg.close)
        big.set_frame(self._last_bgr)
        lay.addWidget(big)
        dlg.showFullScreen()
        self._fullscreen_dlg = dlg
        dlg.finished.connect(lambda *_: setattr(self, "_fullscreen_dlg", None))

    def refresh_title(self) -> None:
        self._title.setText(f"<b>{self._main.stream_display_name(self.stream_id)}</b>")

    def set_frame(self, bgr: np.ndarray) -> None:
        self._last_bgr = bgr.copy()
        h, w = bgr.shape[:2]
        self._res_lbl.setText(f"{w}×{h} px")
        self.video.setToolTip(
            f"Résolution du flux : {w}×{h} px (double-clic ou F11 : plein écran)"
        )
        self.video.set_frame(bgr)
        if self._fullscreen_dlg is not None:
            for wdg in self._fullscreen_dlg.findChildren(VideoLabel):
                if wdg is not self.video:
                    wdg.set_frame(self._last_bgr)
                    break

    def set_stream_fps(self, fps: float) -> None:
        if fps < 0.05:
            self._lbl_fps.setText("—")
        else:
            self._lbl_fps.setText(f"{fps:.1f} img/s")

    def set_latency_ms(self, ms: float | None) -> None:
        if ms is None or ms < 0 or ms > 3_600_000:
            self._lbl_lat.setText("—")
        else:
            self._lbl_lat.setText(f"{ms:.0f} ms")


class DetachedStreamWindow(QWidget):
    def __init__(self, stream_id: str, main: "ServerMainWindow") -> None:
        super().__init__(None, Qt.WindowType.Window)
        self.stream_id = stream_id
        self._main = main
        self._fullscreen_dlg: QDialog | None = None
        self._last_bgr: np.ndarray | None = None
        self.setWindowTitle(f"Flux — {main.stream_display_name(stream_id)}")
        self.resize(720, 520)
        lay = QVBoxLayout(self)
        head = QHBoxLayout()
        self._title = QLabel(f"<b>{main.stream_display_name(stream_id)}</b>")
        head.addWidget(self._title)
        self._lbl_fps = QLabel("—")
        self._lbl_fps.setStyleSheet("color: #9ad; font-weight: bold; min-width: 72px;")
        self._lbl_fps.setToolTip("Images reçues par seconde (moyenne sur le tampon)")
        head.addWidget(self._lbl_fps)
        self._lbl_lat = QLabel("—")
        self._lbl_lat.setStyleSheet("color: #c9a; font-weight: bold; min-width: 56px;")
        self._lbl_lat.setToolTip("Âge de l’image affichée (ms)")
        head.addWidget(self._lbl_lat)
        head.addStretch()
        btn = QPushButton("Réintégrer à la grille")
        btn.clicked.connect(lambda: main.attach_stream(stream_id))
        head.addWidget(btn)
        lay.addLayout(head)
        self.video = VideoLabel(on_double_click=self._toggle_fullscreen)
        lay.addWidget(self.video, 1)
        self._res_lbl = QLabel("")
        self._res_lbl.setStyleSheet("color: #888; font-size: 11px;")
        lay.addWidget(self._res_lbl)

        sc = QShortcut(QKeySequence("F11"), self)
        sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        sc.activated.connect(self._toggle_fullscreen)

    def _toggle_fullscreen(self) -> None:
        if self._fullscreen_dlg is not None:
            self._fullscreen_dlg.close()
            self._fullscreen_dlg = None
            return
        if self._last_bgr is None:
            return
        dlg = QDialog(self)
        dlg.setWindowTitle(self._main.stream_display_name(self.stream_id))
        dlg.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(0, 0, 0, 0)
        big = VideoLabel(on_double_click=dlg.close)
        big.set_frame(self._last_bgr)
        lay.addWidget(big)
        dlg.showFullScreen()
        self._fullscreen_dlg = dlg
        dlg.finished.connect(lambda *_: setattr(self, "_fullscreen_dlg", None))

    def refresh_title(self) -> None:
        name = self._main.stream_display_name(self.stream_id)
        self._title.setText(f"<b>{name}</b>")

    def set_frame(self, bgr: np.ndarray) -> None:
        self._last_bgr = bgr.copy()
        h, w = bgr.shape[:2]
        self._res_lbl.setText(f"{w}×{h} px")
        self.video.setToolTip(
            f"Résolution du flux : {w}×{h} px (double-clic ou F11 : plein écran)"
        )
        self.video.set_frame(bgr)
        if self._fullscreen_dlg is not None:
            for wdg in self._fullscreen_dlg.findChildren(VideoLabel):
                if wdg is not self.video:
                    wdg.set_frame(self._last_bgr)
                    break

    def set_stream_fps(self, fps: float) -> None:
        name = self._main.stream_display_name(self.stream_id)
        if fps < 0.05:
            self._lbl_fps.setText("—")
            self.setWindowTitle(f"Flux — {name}")
        else:
            self._lbl_fps.setText(f"{fps:.1f} img/s")
            self.setWindowTitle(f"Flux — {name} ({fps:.1f} img/s)")

    def set_latency_ms(self, ms: float | None) -> None:
        if ms is None or ms < 0 or ms > 3_600_000:
            self._lbl_lat.setText("—")
        else:
            self._lbl_lat.setText(f"{ms:.0f} ms")

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
        export_dir: str,
        stream_labels: dict[str, str],
        stream_order: list[str],
        disconnect_notice: queue.SimpleQueue | None,
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
        self._export_dir = export_dir.rstrip("/") or "."
        self._stream_labels = dict(stream_labels)
        self._stream_order = list(stream_order)
        self._disconnect_notice = disconnect_notice
        self._disconnect_banner = ""

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

        h = QShortcut(QKeySequence("F1"), self)
        h.setContext(Qt.ShortcutContext.WindowShortcut)
        h.activated.connect(self._show_help)

    def stream_display_name(self, stream_id: str) -> str:
        return self._stream_labels.get(stream_id, stream_id)

    def _order_for_display(self) -> list[str] | None:
        return self._stream_order if self._stream_order else None

    def _grid_ordered_ids(self) -> list[str]:
        return ordered_stream_ids(self.registry, self._order_for_display())

    def _show_help(self) -> None:
        QMessageBox.information(
            self,
            "Aide — raccourcis",
            "<p><b>Fenêtre principale</b></p>"
            "<ul>"
            "<li><b>F1</b> : cette aide</li>"
            "<li><b>REC continu</b> : enregistrement jusqu’à désactivation</li>"
            "<li><b>Snapshot MP4</b> : export tampon complet par flux + empilé</li>"
            "<li><b>PNG</b> : image actuelle par flux</li>"
            "<li><b>Clip</b> : vidéo des N dernières secondes (réglage à côté)</li>"
            "</ul>"
            "<p><b>Par vignette / fenêtre détachée</b></p>"
            "<ul>"
            "<li><b>Double-clic</b> ou <b>F11</b> : plein écran (Échap via fermeture du cadre)</li>"
            "<li><b>Fenêtre séparée</b> : détacher la vue</li>"
            "</ul>"
            "<p>Exports dans le répertoire configuré (export_dir), sous-dossiers "
            "<code>snapshots/</code> et <code>clips/</code>.</p>",
        )

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
        b_snap = QPushButton("Snapshot tampon → MP4 (par flux + empilé)")
        b_snap.setToolTip("Enregistre tout le tampon courant comme avec la touche s (OpenCV)")
        b_snap.clicked.connect(self._snapshot_mp4)
        gl.addWidget(b_snap)
        b_png = QPushButton("Snapshot image → PNG (par flux)")
        b_png.clicked.connect(self._snapshot_png)
        gl.addWidget(b_png)
        clip_row = QHBoxLayout()
        clip_row.addWidget(QLabel("Clip (s) :"))
        self.spin_clip_sec = QSpinBox()
        self.spin_clip_sec.setRange(1, 600)
        self.spin_clip_sec.setValue(10)
        clip_row.addWidget(self.spin_clip_sec)
        b_clip = QPushButton("Exporter clip")
        b_clip.setToolTip("Dernières N secondes par flux (fichiers dans clips/)")
        b_clip.clicked.connect(self._export_clip)
        clip_row.addWidget(b_clip)
        gl.addLayout(clip_row)
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
            "• F1 : aide\n"
            "• Détacher : bouton sur chaque vignette\n"
            "• Double-clic / F11 : plein écran sur la vidéo"
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

    def _snapshot_mp4(self) -> None:
        to_save = []
        for sid in self.registry.ids():
            b = self.registry.get(sid)
            if b is not None:
                to_save.append(b)
        if not to_save:
            return
        fps = self.frame_rate

        def _run() -> None:
            save_per_stream_and_stack(
                to_save, fps, export_dir=self._export_dir
            )

        threading.Thread(target=_run, daemon=True).start()

    def _snapshot_png(self) -> None:
        for sid in self.registry.ids():
            b = self.registry.get(sid)
            if b is not None:

                def _run(buf: Any = b) -> None:
                    buf.save_latest_png()

                threading.Thread(target=_run, daemon=True).start()

    def _export_clip(self) -> None:
        sec = int(self.spin_clip_sec.value())
        for sid in self.registry.ids():
            b = self.registry.get(sid)
            if b is not None:

                def _run(buf: Any = b, s: int = sec) -> None:
                    buf.save_clip_last_seconds(float(s))

                threading.Thread(target=_run, daemon=True).start()

    def _relayout_grid(self) -> None:
        while self._grid_layout.count():
            it = self._grid_layout.takeAt(0)
            w = it.widget()
            if w is not None:
                w.setParent(None)
        ids = [x for x in self._grid_ordered_ids() if x in self._tiles]
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

    def _drain_disconnect_toasts(self) -> None:
        if self._disconnect_notice is None:
            return
        parts: list[str] = []
        while True:
            try:
                sid = self._disconnect_notice.get_nowait()
            except queue.Empty:
                break
            parts.append(self.stream_display_name(sid))
        if parts:
            self._disconnect_banner = (
                "Déconnecté : " + ", ".join(parts) + "\n\n"
            )

    def _tick(self) -> None:
        if self.stop_event.is_set():
            self.close()
            return

        self._drain_disconnect_toasts()

        delay = float(self.spin_delay.value())
        raw_ids = list(self.registry.ids())
        frames = gather_display_frames_with_ts(
            self.registry,
            self.live_display,
            delay,
            stream_order=self._order_for_display(),
            active_stream_ids=raw_ids,
        )
        tick_continuous_stack_recording(
            self.registry,
            [(sid, fr) for sid, fr, _ts in frames],
            self.frame_rate,
            self._stack_state,
            export_dir=self._export_dir,
        )

        ids = set(raw_ids)

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

        for sid in raw_ids:
            if sid in self._detached or sid in self._tiles:
                continue
            self._tiles[sid] = StreamTile(sid, self)
            self._relayout_grid()

        for t in self._tiles.values():
            t.refresh_title()
        for w in self._detached.values():
            w.refresh_title()

        now = time.time()
        fd = {sid: (fr, ts) for sid, fr, ts in frames}
        for sid, (fr, ts) in fd.items():
            if sid in self._detached:
                self._detached[sid].set_frame(fr)
            elif sid in self._tiles:
                self._tiles[sid].set_frame(fr)
            lat_ms = None
            if ts is not None:
                lat_ms = (now - ts) * 1000.0
            if sid in self._detached:
                self._detached[sid].set_latency_ms(lat_ms)
            elif sid in self._tiles:
                self._tiles[sid].set_latency_ms(lat_ms)

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
            banner = self._disconnect_banner
            self._disconnect_banner = ""
            self.lbl_status.setText(
                banner
                + f"FPS UI ≈ {self._fps_count}\n"
                f"Flux actifs : {len(ids)}\n"
                f"Mode : {'LIVE' if self.live_display else f'délai {delay:.0f}s'}\n"
                f"REC : {'oui' if self.registry.is_continuous_recording() else 'non'}\n"
                f"Exports : {self._export_dir}"
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
    export_dir: str = "./video",
    stream_labels: dict[str, str] | None = None,
    stream_order: list[str] | None = None,
    disconnect_notice: queue.SimpleQueue | None = None,
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
        export_dir=export_dir,
        stream_labels=stream_labels or {},
        stream_order=stream_order or [],
        disconnect_notice=disconnect_notice,
    )
    win.show()
    app.exec()
