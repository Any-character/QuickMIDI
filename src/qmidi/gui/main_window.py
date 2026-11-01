"""The window: three panels, a transport, and nothing clever.

All state lives in :class:`qmidi.gui.model.Session`; all audio lives in
:class:`qmidi.gui.player.PreviewPlayer`.  This module wires them to actions and
decides *when* to re-render -- the one interesting rule being that editing a
segment while it is looping restarts that loop, because "change the number and
hear it again" is the whole workflow.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..exporters import (
    BITRATES,
    DEFAULT_BITRATE,
    default_midi_output,
    export_midi,
    export_mp3,
)
from ..i18n import tr
from ..keymap import KeyMap, pretty
from ..settings import Settings
from .model import Session
from .overview import OverviewBar
from .piano_roll import PianoRoll
from .player import PREVIEW_GAIN, PreviewPlayer
from .render_plan import PreviewMidi, play_request
from .segment_panel import SegmentPanel
from .settings_dialog import SettingsDialog
from .track_panel import TrackPanel

APP_NAME = "QuickMIDI"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.session: Optional[Session] = None
        self.keymap = KeyMap.load()
        self.settings = Settings.load()
        self.preview = PreviewMidi()
        self.player = PreviewPlayer(self)
        #: A seek (ruler click, overview drag) restarts playback at most this
        #: often.  Restarting on every mouse-move would stutter instead of
        #: scrub, and the render backend already costs ~0.3 s to spin up.
        self._seek_timer = QTimer(self)
        self._seek_timer.setSingleShot(True)
        self._seek_timer.setInterval(120)
        self._seek_timer.timeout.connect(self._apply_seek)
        #: Where the cursor wants to be while the renderer catches up.  Kept until
        #: the reported position reaches it, so the playhead does not snap back.
        self._pending_seek_tick: Optional[int] = None
        self._seek_deadline = 0.0
        #: Guards the spin box while it is being programmatically updated,
        #: otherwise refreshing the window would look like the user typed a tempo.
        self._syncing_bpm = False
        self._last_position_paint = 0.0

        self.setWindowTitle(APP_NAME)
        self.resize(1280, 820)
        self.setAcceptDrops(True)

        self.roll = PianoRoll(self)
        self.roll.set_keymap(self.keymap)
        self.roll.set_show_all_tracks(self.settings.show_all_tracks)
        self.overview = OverviewBar(self)
        self.tracks = TrackPanel(self)
        self.segments = SegmentPanel(self)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(0)
        top.addWidget(self.tracks)
        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(0)
        right.addWidget(self.roll, 1)
        right.addWidget(self.overview)
        top.addLayout(right, 1)
        body = QWidget(self)
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)
        body_layout.addLayout(top, 1)
        body_layout.addWidget(self.segments)
        self.setCentralWidget(body)

        self._position_label = QLabel("", self)
        self._position_label.setObjectName("positionLabel")
        self._status_message = QLabel("", self)
        self._gain_hint = QLabel(
            tr("预览音量 ×{gain:.2f}（导出不衰减）").format(gain=PREVIEW_GAIN), self)
        self._gain_hint.setObjectName("gainHint")
        self.statusBar().addWidget(self._position_label)
        self.statusBar().addWidget(self._status_message, 1)
        self.statusBar().addPermanentWidget(self._gain_hint)

        self._build_actions()
        self.roll.segmentPicked.connect(self._on_segment_picked)
        self.roll.cursorMoved.connect(self._on_cursor_moved)
        self.roll.nudgeRequested.connect(self._on_delta)
        self.roll.stepRequested.connect(self._step_selection)
        self.roll.clearRequested.connect(self._on_clear)
        self.roll.pedalToggled.connect(self._on_pedal_toggled)
        self.roll.pedalErased.connect(self._on_pedal_erased)
        self.roll.velocityDragStarted.connect(self._on_velocity_drag_started)
        self.roll.velocityDragged.connect(self._on_velocity_dragged)
        self.roll.velocityDragFinished.connect(self._on_velocity_drag_finished)
        self.roll.viewChanged.connect(self._on_view_changed)
        self.overview.seekRequested.connect(self._on_overview_seek)
        self.overview.viewDragged.connect(self.roll.set_view_left)
        self.segments.pedalCleared.connect(self._on_pedal_cleared)
        self.segments.deltaRequested.connect(self._on_delta)
        self.segments.targetRequested.connect(self._on_target)
        self.segments.clearRequested.connect(self._on_clear)
        self.segments.mergeRequested.connect(self._on_merge)
        self.segments.splitRequested.connect(self._on_split)
        self.segments.auditionRequested.connect(self._loop_selected)
        self.tracks.trackActivated.connect(self._activate_track)
        self.tracks.muteToggled.connect(self._toggle_mute)
        self.tracks.soloToggled.connect(self._toggle_solo)
        self.tracks.renameRequested.connect(self._on_rename_track)
        self.player.positionChanged.connect(self._on_position)
        self.player.started.connect(
            lambda label: self._say(tr("播放：{label}").format(label=label)))
        self.player.finished.connect(self._on_play_finished)
        self.player.failed.connect(self._on_play_failed)
        self._apply_keymap()

    # -- construction ------------------------------------------------------ #
    def _build_actions(self) -> None:
        bar = self.addToolBar(tr("主"))
        bar.setMovable(False)

        def act(action_id: str, slot: Callable, tip: str = "",
                checkable: bool = False, in_bar: bool = True) -> QAction:
            binding = self.keymap.binding(action_id)
            action = QAction(binding.title, self)
            action.setObjectName(action_id)
            action.setCheckable(checkable)
            action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
            # Remember the un-annotated text: rebinding rebuilds the tooltip
            # from it instead of appending a second "（L）".
            action.setProperty("baseTip", tip or binding.label)
            # Checkable actions must report *state*, not only clicks: code that
            # arms loop mode programmatically (the "循环本段" button, a restored
            # session) calls setChecked(), which fires toggled but never
            # triggered.
            if checkable:
                action.toggled.connect(slot)
            else:
                action.triggered.connect(slot)
            if in_bar:
                bar.addAction(action)
            self.addAction(action)  # keep shortcuts alive even if the bar hides
            self._bind_action(action, action_id)
            return action

        self.act_open = act("open", self.open_file, tr("打开 MIDI 文件"))
        self.act_save = act("save", self.save_project,
                            tr("保存力度段到 sidecar JSON"))
        bar.addSeparator()
        self.act_play = act("play", self._toggle_play,
                            tr("从播放头播放；再按一次停止"))
        self.act_loop = act("loop", self._on_loop_toggled,
                            tr("循环光标所在的那一段"), checkable=True)
        self.act_home = act("home", self._go_home, tr("播放头回到开头"), in_bar=False)

        self.bpm_label = QLabel("BPM", self)
        self.bpm_label.setObjectName("bpmLabel")
        bar.addWidget(self.bpm_label)
        self.bpm_box = QDoubleSpinBox(self)
        self.bpm_box.setObjectName("bpmBox")
        self.bpm_box.setRange(20.0, 300.0)
        self.bpm_box.setDecimals(0)
        self.bpm_box.setSingleStep(1.0)
        self.bpm_box.setFixedWidth(58)
        self.bpm_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.bpm_box.setButtonSymbols(
            QDoubleSpinBox.ButtonSymbols.NoButtons)  # flat, like a DAW tempo field
        self.bpm_box.setToolTip(tr("播放速度：可直接输入，也可以滚轮微调；改完从播放头重新开始"))
        # Without this the box fires on every keystroke ("1", "12", "120") and
        # each one restarts playback.
        self.bpm_box.setKeyboardTracking(False)
        self.bpm_box.valueChanged.connect(self._on_bpm_changed)
        bar.addWidget(self.bpm_box)
        bar.addSeparator()

        self.act_zoom_out = act("zoom_out", lambda: self.roll.zoom_time(1 / 1.25),
                                tr("时间轴缩小"))
        self.zoom_button = QToolButton(self)
        self.zoom_button.setText("100%")
        self.zoom_button.setFixedWidth(56)
        self.zoom_button.setToolTip(tr("当前缩放；点击恢复全曲适配"))
        self.zoom_button.setAutoRaise(True)
        self.zoom_button.clicked.connect(self.roll.zoom_fit)
        bar.addWidget(self.zoom_button)
        self.act_zoom_in = act("zoom_in", lambda: self.roll.zoom_time(1.25),
                               tr("时间轴放大"))

        bar.addSeparator()
        self.act_undo = act("undo", self._undo)
        self.act_redo = act("redo", self._redo)
        bar.addSeparator()
        # One button, two formats.  The actions stay window-level so Ctrl+E /
        # Ctrl+Shift+E still fire without opening the menu -- the menu is the
        # discoverable path, the shortcuts are the fast one.
        self.act_export_midi = act("export_midi", self.export_midi_file, in_bar=False)
        self.act_export_mp3 = act("export_mp3", self.export_mp3_file, in_bar=False)
        self.export_button = QToolButton(self)
        self.export_button.setText(tr("导出"))
        self.export_button.setToolTip(tr("导出为 MIDI 或 MP3"))
        self.export_button.setAutoRaise(True)
        self.export_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.export_menu = QMenu(self.export_button)
        self.export_menu.addAction(self.act_export_midi)
        self.export_menu.addAction(self.act_export_mp3)
        self.export_button.setMenu(self.export_menu)
        bar.addWidget(self.export_button)
        self.act_resegment = act("resegment", self.resegment_tracks,
                                 tr("按当前规则重新切分所有轨道（会丢弃段调整）"))
        self.act_auto_pedal = act("auto_pedal", self.auto_pedal,
                                  tr("按低音与和声变化重做踏板（可撤销）"))
        self.act_stereo = act("stereo_split", self.set_stereo_spread,
                              tr("两台钢琴分左右（写进导出的 MIDI 和 MP3）"))
        self.act_master = act("master", self._on_master_toggled,
                              tr("母带：混响 + 温和压缩 + 响度标准化（只进 MP3）"),
                              checkable=True)
        bar.addSeparator()
        self.act_fit = act("fit", self.roll.zoom_fit, tr("缩放到全曲"), in_bar=False)
        self.act_settings = act("settings", self.open_settings,
                                tr("音源、显示方式和快捷键"))

        self._update_enabled()

    def _bind_action(self, action: QAction, action_id: str) -> None:
        """Apply one key from the current map, and restate it in the tooltip."""
        sequence = self.keymap.sequence(action_id)
        action.setShortcut(QKeySequence(sequence))
        base = action.property("baseTip") or action.text()
        action.setToolTip(f"{base}（{pretty(sequence)}）")

    def _apply_keymap(self) -> None:
        """Push the current map everywhere it is read: actions, roll, hint line."""
        for action in self.actions():
            name = action.objectName()
            if name:
                self._bind_action(action, name)
        self._sync_play_action()
        self.roll.set_keymap(self.keymap)
        self.segments.set_hint(self.keymap.hint_text())

    def open_settings(self) -> None:
        previous_instrument = self.settings.sfz_path
        dialog = SettingsDialog(self.settings, self.keymap, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.settings = dialog.result_settings
        self.keymap = dialog.result_keymap
        if self.session is not None:
            self.session.snap_division = self.settings.snap_division
        self._apply_keymap()
        self.roll.set_show_all_tracks(self.settings.show_all_tracks)
        self.preview.invalidate()
        if self.session is not None:
            self._refresh_all()
        if self.settings.sfz_path != previous_instrument:
            self._reload_instrument()
            self._say(tr("音源已切换：{label}").format(label=self.settings.instrument_label))
        else:
            self._say(tr("设置已保存"))

    def _reload_instrument(self) -> None:
        """Swap the instrument and play a moment of it -- that is the point."""
        self._seek_timer.stop()
        self.player.stop()
        if self.session is None:
            return
        self._play(self.session.loop_enabled)

    # -- files ------------------------------------------------------------- #
    def open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, tr("打开 MIDI"), "",
                                              tr("MIDI 文件 (*.mid *.midi)"))
        if path:
            self.load(Path(path))

    def load(self, path: Path) -> None:
        try:
            session = Session.open(path)
        except Exception as exc:
            QMessageBox.critical(self, tr("打不开这个文件"),
                                 f"{path.name}\n\n{type(exc).__name__}: {exc}")
            return
        self.player.stop()
        session.snap_division = self.settings.snap_division
        self.session = session
        self.preview = PreviewMidi()
        self.roll.set_session(session, session.resolved_velocity_map())
        self.overview.set_session(session)
        self.overview.set_view(self.roll.view_left_tick, self.roll.visible_ticks)
        self.overview.set_playhead(0)
        self.act_loop.blockSignals(True)
        self.act_loop.setChecked(bool(session.loop_enabled))
        self.act_loop.blockSignals(False)
        # The sidecar remembers the mastering switch; the checkbox must agree
        # with it or the next toggle writes the opposite of what the user sees.
        self.act_master.blockSignals(True)
        self.act_master.setChecked(bool(session.project.master))
        self.act_master.blockSignals(False)
        self._position_label.setText(session.position_label(0.0))
        self.tracks.set_session(session)
        self._refresh_all()
        self._say(tr("{name} · {tracks} 轨 · {segments} 段 · {tpb} ticks/beat").format(
            name=path.name, tracks=session.track_count,
            segments=session.project.segment_count,
            tpb=session.project.ticks_per_beat))
        if session.project.warnings:
            warnings = session.project.warnings
            QMessageBox.warning(
                self, tr("工程文件与 MIDI 不一致"),
                tr("sidecar 工程文件描述的段与当前 MIDI 对不上，已按能对上的部分载入：\n\n")
                + "\n".join(f"· {w}" for w in warnings[:6])
                + ("\n…" if len(warnings) > 6 else "")
            )

    def save_project(self) -> None:
        if self.session is None:
            return
        try:
            written = self.session.save()
        except Exception as exc:
            QMessageBox.critical(self, tr("保存失败"), str(exc))
            return
        self._say(tr("已保存：{name}").format(name=written.name))
        self._refresh_title()

    def resegment_tracks(self) -> None:
        """Re-cut everything with the current rules -- the only way to see them.

        Segmentation runs once, when a file is first opened.  Without this button
        a user who already saved a sidecar would never see an improved rule.
        """
        if self.session is None:
            return
        answer = QMessageBox.question(
            self, tr("重新自动切段"),
            tr("按当前规则重新切分所有轨道的力度段。\n\n"
               "你做过的合并、拆分、力度调整都会丢失（可以按 Ctrl+Z 撤销）。\n\n继续吗？"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.player.stop()
        if not self.session.resegment():
            self._say(tr("这个文件没有可切的音符"))
            return
        self._after_edit()
        self._say(tr("已重新切段：{n} 段").format(n=self.session.project.segment_count))

    # -- pedal / pan / mastering ------------------------------------------- #
    def auto_pedal(self) -> None:
        """Re-take the pedal from the harmony (one undo step, file untouched)."""
        if self.session is None:
            return
        try:
            count = self.session.auto_pedal()
        except Exception as exc:
            QMessageBox.critical(self, tr("生成踏板失败"),
                                 f"{type(exc).__name__}: {exc}")
            return
        if not count:
            self._say(tr("没有可用的音符，没生成踏板"))
            return
        self._after_edit()
        self._say(tr("已按和声重做踏板：{count} 轨（Ctrl+Z 撤销；原 MIDI 文件不动）")
                  .format(count=count))

    def set_stereo_spread(self) -> None:
        """Two clicks to a two-piano image: presets, not a pan slider."""
        if self.session is None:
            return
        channels = self.session.channels()
        if len(channels) < 2:
            self._say(tr("这个文件只有一个通道，分不了声像"))
            return
        presets = [tr("关闭（都放中间）"), tr("轻微（52 / 76）"),
                   tr("标准（40 / 88）"), tr("宽阔（28 / 100）")]
        values = [(None, None), (52, 76), (40, 88), (28, 100)]
        current = 2
        left_now = self.session.project.pan(channels[0])
        for i, (left, _right) in enumerate(values):
            if left == left_now:
                current = i
                break
        choice, ok = QInputDialog.getItem(
            self, tr("双钢琴声像"),
            tr("钢琴 1 = 通道 {a}，钢琴 2 = 通道 {b}\n分开多少：").format(
                a=channels[0], b=channels[1]),
            presets, current, False)
        if not ok:
            return
        left, right = values[presets.index(choice)]
        changed = self.session.set_channel_pans({channels[0]: left, channels[1]: right})
        self._after_edit()
        if left is None:
            self._say(tr("声像已关闭（两台琴回到正中间）"))
        else:
            self._say(tr("声像：通道 {a} → {left}，通道 {b} → {right}"
                         "（导出的 MIDI/MP3 生效，预览是两架混在中间的）").format(
                a=channels[0], left=left, b=channels[1], right=right))
        if not changed:
            self._say(tr("声像没变"))

    def _on_master_toggled(self, checked: bool) -> None:
        """The checkbox *is* the setting: on = the default chain, off = untouched."""
        if self.session is None:
            return
        if checked:
            from dataclasses import asdict

            from ..mastering import MasterSettings

            self.session.set_master(asdict(MasterSettings()))
            self._say(tr("母带已开启：混响 14% + 2:1 温和压缩 + 响度 -14 LUFS"
                         "（只作用于导出的 MP3）"))
        else:
            self.session.set_master(None)
            self._say(tr("母带已关闭：导出的 MP3 回到纯归一化"))
        self._after_edit()

    # -- export ------------------------------------------------------------ #
    def export_midi_file(self) -> None:
        if self.session is None:
            return
        suggested = default_midi_output(self.session.project)
        path, _ = QFileDialog.getSaveFileName(self, tr("导出 MIDI"), str(suggested),
                                              tr("MIDI 文件 (*.mid)"))
        if not path:
            return
        project = self.session.project

        def job(_report):
            return export_midi(project, path, verify=True)

        self._run_background(tr("正在写出 MIDI…"), job, self._on_midi_exported)

    def _on_midi_exported(self, result) -> None:
        self._say(tr("MIDI 已导出：{name}（改力度 {notes} 个，重触发 {retrig} 处，"
                     "涉及 {tracks} 轨）").format(
            name=result.path.name, notes=result.notes_rewritten,
            retrig=result.retriggers, tracks=result.tracks_touched))
        if result.warnings:
            QMessageBox.warning(self, tr("导出完成，但有提醒"),
                                "\n".join(f"· {w}" for w in result.warnings[:8]))

    def export_mp3_file(self) -> None:
        if self.session is None:
            return
        labels = [f"{rate} kbps" for rate in BITRATES]
        choice, ok = QInputDialog.getItem(self, tr("MP3 码率"), tr("选择码率："), labels,
                                          BITRATES.index(DEFAULT_BITRATE), False)
        if not ok:
            return
        bitrate = BITRATES[labels.index(choice)]
        source = Path(self.session.project.source)
        suggested = source.with_name(f"{source.stem}_qmidi.mp3")
        path, _ = QFileDialog.getSaveFileName(self, tr("导出 MP3"), str(suggested),
                                              tr("MP3 文件 (*.mp3)"))
        if not path:
            return
        project = self.session.project
        preview = self.preview
        # Captured here, not read inside the thread: the user may change the
        # instrument while a long render is running.
        sfz = Path(self.settings.sfz_path) if self.settings.sfz_path else None
        quality = int(self.settings.export_quality)
        pans = dict(project.pans) or None
        master_settings = None
        if project.master:
            from ..mastering import MasterSettings

            master_settings = MasterSettings(**project.master)

        def job(report):
            # Render *from an exported MIDI*, so what is heard is what is sent.
            midi = preview.ensure(project)
            return export_mp3(midi, path, bitrate=bitrate, sfz_path=sfz,
                              pans=pans, master=master_settings,
                              on_progress=report, sample_quality=quality)

        self._run_background(tr("正在渲染 MP3（整曲，可能要几分钟）…"), job,
                             self._on_mp3_exported)

    def _on_mp3_exported(self, result) -> None:
        loudness = ""
        if result.lufs_after > float("-inf"):
            loudness = tr(" · 响度 {value} LUFS").format(value=f"{result.lufs_after:.1f}")
        self._say(tr("MP3 已导出：{name} · {bitrate} kbps · {sec}s · 峰值 {peak}"
                     "{loudness}").format(
            name=result.path.name, bitrate=result.bitrate,
            sec=f"{result.frames / result.sample_rate:.1f}",
            peak=f"{result.peak_after:.2f}", loudness=loudness))

    def _run_background(self, label: str, job: Callable, on_done: Callable) -> None:
        """Run ``job`` off the GUI thread behind a progress dialog.

        ``job`` receives a ``report(fraction)`` callback.  A job that never calls
        it just leaves the bar at zero -- the dialog still says what is happening,
        so a spinner is never *needed*, only nice to have.
        """
        box: dict = {"result": None, "error": None, "done": False, "progress": 0.0}
        started = time.monotonic()

        def report(fraction: float) -> None:
            box["progress"] = min(1.0, max(0.0, float(fraction)))

        def work() -> None:
            try:
                box["result"] = job(report)
            except Exception as exc:  # reported on the GUI thread below
                box["error"] = exc
            finally:
                box["done"] = True

        threading.Thread(target=work, daemon=True).start()
        dialog = QProgressDialog(label, "", 0, 100, self)
        dialog.setWindowTitle(tr("请稍候"))
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        dialog.setCancelButton(None)
        dialog.setMinimumDuration(0)
        dialog.show()
        timer = QTimer(self)
        timer.setInterval(120)

        def poll() -> None:
            if not box["done"]:
                percent = int(box["progress"] * 100)
                dialog.setValue(percent)
                elapsed = time.monotonic() - started
                # Only estimate once there is something to extrapolate from, and
                # never below a second -- "剩余约 0 秒" reads as a glitch.
                if 3 <= percent < 100 and elapsed > 2.0:
                    remaining = elapsed / (percent / 100.0) - elapsed
                    dialog.setLabelText(tr("{label}\n剩余约 {n} 秒").format(
                        label=label, n=max(1, round(remaining))))
                return
            timer.stop()
            timer.deleteLater()
            dialog.close()
            dialog.deleteLater()
            if box["error"] is not None:
                QMessageBox.critical(self, tr("导出失败"),
                                     f"{type(box['error']).__name__}: {box['error']}")
            else:
                on_done(box["result"])

        timer.timeout.connect(poll)
        timer.start()

    # -- track panel ------------------------------------------------------- #
    def _on_rename_track(self, track: int) -> None:
        """Right-click a track: give it a name you can tell apart.

        The name is stored in the sidecar, so it survives reopening.  Clearing it
        falls back to the ``track_name`` inside the MIDI file rather than leaving
        a nameless row.
        """
        if self.session is None:
            return
        try:
            current = self.session.project.track(track).name
        except (IndexError, KeyError):
            return
        name, ok = QInputDialog.getText(
            self, tr("重命名音轨"),
            tr("轨道 {track} 的新名字（留空则用文件里的名字）：").format(track=track),
            text=current,
        )
        if not ok:
            return
        if self.session.rename_track(track, name):
            self._refresh_all()
            self._say(tr("轨道 {track} 已改名").format(track=track) if name.strip()
                      else tr("轨道 {track} 用回文件里的名字").format(track=track))

    def _activate_track(self, track: int) -> None:
        if self.session is None:
            return
        self.session.active_track = track
        if self.session.selected_index(track) is None:
            tick = self.session.cursor_tick
            index = self.session.project.track(track).index_of_tick(tick)
            self.session.select(track, 0 if index is None else index)
        self._refresh_all()
        # With only one track drawn, a switch to a track an octave away would
        # otherwise leave the user staring at empty rows.
        self.roll.fit_pitch()

    def _toggle_mute(self, track: int) -> None:
        if self.session is None:
            return
        self.session.toggle_mute(track)
        self._refresh_all()
        self._restart_if_playing()

    def _toggle_solo(self, track: int) -> None:
        if self.session is None:
            return
        self.session.toggle_solo(track)
        self._refresh_all()
        self._restart_if_playing()

    # -- selection / edits ------------------------------------------------- #
    def _on_segment_picked(self, track: int, index: int) -> None:
        if self.session is None:
            return
        self.session.select(track, index)
        self._refresh_all()

    def _on_cursor_moved(self, tick: int) -> None:
        # A ruler click is one deliberate jump: seek now rather than waiting out
        # the drag debounce, or it reads as "the click did nothing".
        self._move_cursor(tick, immediate=True)

    def _on_overview_seek(self, tick: int) -> None:
        self._move_cursor(tick)

    def _move_cursor(self, tick: int, *, immediate: bool = False) -> None:
        """Move the playhead; if audio is running, jump there.

        The debounce exists for the overview bar, which emits one event per pixel
        of drag.  A ruler click passes ``immediate`` because a single click should
        not feel like it was ignored for 120 ms.
        """
        if self.session is None:
            return
        # Placing the cursor snaps to the playhead grid; the playhead following
        # audio (``_on_position``) deliberately does not.
        tick = self.session.snap_tick(tick)
        self.session.cursor_tick = tick
        # Moving the cursor also selects the segment under it.  In loop mode the
        # thing that repeats is the selected segment; without this, clicking the
        # ruler while looping would keep repeating whatever was selected before
        # and give no hint why.  Only repaint when the segment actually changes:
        # the overview bar emits one of these per pixel of drag.
        before = self.session.selected_index()
        selected = self.session.select_at_tick(tick)
        if selected is not None and selected != before:
            self.segments.refresh(self.session)
            self.roll.set_session(self.session, self.session.resolved_velocity_map())
            self.roll.set_loop(self.session.loop_ticks() if self.session.loop_enabled else None)
        self.roll.set_playhead_tick(tick, follow=False)
        self.overview.set_playhead(tick)
        self._refresh_position(self.session.clock.seconds(tick))
        if self.player.is_playing:
            self._pending_seek_tick = self.session.cursor_tick
            self._seek_deadline = time.monotonic() + 1.2
            if immediate:
                self._seek_timer.stop()
                self._apply_seek()
            else:
                self._seek_timer.start()

    def _apply_seek(self) -> None:
        """Hand the target to the renderer; the device keeps running."""
        if self.session is None or not self.player.is_playing:
            self._pending_seek_tick = None
            return
        tick = self._pending_seek_tick
        if tick is None:
            tick = self.session.cursor_tick
        self._seek_deadline = time.monotonic() + 1.2
        self.player.seek(self.session.clock.seconds(max(0, int(tick))))

    def _go_home(self) -> None:
        self._move_cursor(0, immediate=True)

    def _step_selection(self, delta: int) -> None:
        if self.session is None:
            return
        self.session.step_selection(delta)
        self.roll.set_playhead_tick(self.session.cursor_tick)
        self.overview.set_playhead(self.session.cursor_tick)
        self._refresh_position(self.session.clock.seconds(self.session.cursor_tick))
        self._refresh_all()

    def _on_delta(self, delta: int) -> None:
        if self.session is None:
            return
        if not self.session.set_delta(delta):
            return
        self._after_edit()

    def _on_target(self, velocity: int) -> None:
        if self.session is None:
            return
        if not self.session.set_target(velocity):
            return
        self._after_edit()

    def _on_clear(self) -> None:
        if self.session is None:
            return
        if self.session.clear_adjustment():
            self._after_edit()

    def _on_merge(self) -> None:
        if self.session is None:
            return
        try:
            if self.session.merge_with_next():
                self._after_edit()
        except Exception as exc:
            self._say(tr("不能合并：{reason}").format(reason=exc))

    def _on_split(self) -> None:
        if self.session is None:
            return
        try:
            if self.session.split_at_cursor():
                self._after_edit()
        except Exception as exc:
            self._say(tr("这里不能拆分：{reason}").format(reason=exc))

    def _on_pedal_toggled(self, tick: int) -> None:
        if self.session is None:
            return
        if self.session.toggle_pedal_at(tick):
            self._after_edit()

    # -- velocity lane ----------------------------------------------------- #
    def _on_velocity_drag_started(self, reference: int) -> None:
        if self.session is not None:
            self.session.begin_velocity_drag(reference)

    def _on_velocity_dragged(self, steps: int) -> None:
        if self.session is not None and self.session.drag_velocity(steps):
            self._after_edit()

    def _on_velocity_drag_finished(self) -> None:
        if self.session is not None:
            self.session.end_velocity_drag()
            self._refresh_all()

    def _on_pedal_erased(self, tick: int) -> None:
        if self.session is None:
            return
        if self.session.erase_pedal(tick):
            self._after_edit()

    def _on_pedal_cleared(self) -> None:
        if self.session is None:
            return
        if self.session.clear_pedal():
            self._after_edit()

    def _undo(self) -> None:
        if self.session is not None and self.session.undo():
            self._after_edit()

    def _redo(self) -> None:
        if self.session is not None and self.session.redo():
            self._after_edit()

    def _after_edit(self) -> None:
        self.preview.invalidate()
        self._refresh_all()
        self._restart_if_playing()

    # -- playback ---------------------------------------------------------- #
    def _toggle_play(self) -> None:
        if self.player.is_playing:
            self.player.stop()
            self._say(tr("停止"))
            self._sync_play_action()
            return
        if self.session is None:
            self._say(tr("先打开一个 MIDI 文件"))
            return
        self._play(self.session.loop_enabled)

    def _loop_selected(self) -> None:
        """The segment panel's button: loop this segment, no questions asked."""
        if self.session is None:
            return
        if self.player.is_playing:
            self.player.stop()
        self.session.loop_enabled = True
        self.act_loop.blockSignals(True)
        self.act_loop.setChecked(True)
        self.act_loop.blockSignals(False)
        self.roll.set_loop(self.session.loop_ticks())
        segment = self.session.selected_segment()
        if segment is not None:
            self.roll.scroll_to_tick(segment.start_tick)
        self._play(True)

    def _on_loop_toggled(self, checked: bool) -> None:
        if self.session is None:
            return
        self.session.loop_enabled = bool(checked)
        self.roll.set_loop(self.session.loop_ticks() if checked else None)
        self._say(tr("循环：当前段") if checked else tr("循环关闭：从播放头播到结尾"))
        if self.player.is_playing:
            self.player.stop()
            self._play(bool(checked))

    def _play(self, loop: bool) -> None:
        if self.session is None:
            self._say(tr("先打开一个 MIDI 文件"))
            return
        try:
            midi = self.preview.ensure(self.session.project)
        except Exception as exc:
            QMessageBox.critical(self, tr("无法生成预览 MIDI"),
                                 f"{type(exc).__name__}: {exc}")
            return
        sfz = Path(self.settings.sfz_path) if self.settings.sfz_path else None
        request = play_request(self.session, midi, loop=loop, sfz_path=sfz)
        if request is None:
            self._say(tr("没有可播放的内容（这一段没有音符，或者所有轨道都被静音了）"))
            return
        self._last_position_paint = 0.0
        self._pending_seek_tick = None
        tick = self.session.clock.tick(request.start_sec)
        self.roll.set_playhead_tick(tick)
        self.overview.set_playhead(tick)
        self.player.play(request)
        self._sync_play_action()

    def _sync_play_action(self) -> None:
        title = self.keymap.binding("play").title
        self.act_play.setText(tr("■ 停止") if self.player.is_playing else title)

    def _restart_if_playing(self) -> None:
        """Loop mode only: replay the segment so an edit is heard immediately.

        Outside loop mode an edit deliberately restarts nothing.  The current
        pass keeps playing the audio it already rendered, and the next press of
        play picks the change up -- the "don't stop me, I'll listen in a second"
        behaviour the user chose.
        """
        if self.session is None or not self.player.is_playing:
            return
        if not self.session.loop_enabled:
            return
        self.player.stop()
        self._play(True)

    def _on_position(self, seconds: float) -> None:
        if self.session is None:
            return
        pending = self._pending_seek_tick
        if pending is not None:
            target = self.session.clock.seconds(pending)
            # The device is still playing the audio it buffered *before* the
            # seek.  Holding the cursor on the target is what makes the jump look
            # instant; the deadline stops a failed seek from freezing it forever.
            if abs(seconds - target) > 0.35 and time.monotonic() < self._seek_deadline:
                return
            self._pending_seek_tick = None
        now = time.monotonic()
        # 20 fps: a playhead is one vertical line, and every extra frame here is
        # time the renderer thread does not get.  33 ms was measurably eating
        # into the ring's headroom (measured 2026-10-25).
        if now - self._last_position_paint < 0.05:
            return
        self._last_position_paint = now
        # The audio keeps ringing for the release tail after the last note, but
        # the playhead should stop at the end of the music rather than crawl into
        # a region that has no notes in it.
        content_end = self.session.total_end_tick()
        tick = min(self.session.clock.tick(seconds), content_end)
        self.session.cursor_tick = tick
        self.roll.set_playhead_tick(tick)
        self.overview.set_playhead(tick)
        self._refresh_position(min(seconds, self.session.clock.seconds(content_end)))

    def _on_play_finished(self) -> None:
        self._say(tr("播放结束（空格从当前位置再播）"))
        self._sync_play_action()

    def _on_play_failed(self, message: str) -> None:
        self._say(tr("播放失败：{message}").format(message=message))

    # -- shared refresh ---------------------------------------------------- #
    def _refresh_all(self, *, repaint_roll: bool = True) -> None:
        session = self.session
        self.segments.refresh(session)
        self.tracks.refresh()
        if session is not None:
            if repaint_roll:
                self.roll.set_session(session, session.resolved_velocity_map())
            # The loop band on the ruler only means something when looping is
            # actually on; drawing it always would advertise a repeat that is
            # not armed.
            self.roll.set_loop(session.loop_ticks() if session.loop_enabled else None)
            self._sync_bpm(session.bpm)
            # A tempo change alters every second value in the readout, including
            # the total duration on the right of the slash.
            self._refresh_position(session.clock.seconds(session.cursor_tick))
        self._update_enabled()
        self._refresh_title()

    def _sync_bpm(self, bpm: float) -> None:
        self._syncing_bpm = True
        try:
            self.bpm_box.setValue(float(bpm))
        finally:
            self._syncing_bpm = False

    def _on_bpm_changed(self, value: float) -> None:
        """Apply a tempo edit; restart playback so both speeds never overlap."""
        if self.session is None or self._syncing_bpm:
            return
        session = self.session
        if not session.set_bpm(value):
            return
        was_playing = self.player.is_playing
        if was_playing:
            self.player.stop()
        self._refresh_all()
        if was_playing:
            self._play(session.loop_enabled)
        self._say(tr("速度 {value} BPM（改动可撤销）").format(value=f"{value:.0f}"))

    def _refresh_position(self, seconds: float) -> None:
        if self.session is None:
            self._position_label.setText("")
            return
        self._position_label.setText(self.session.position_label(seconds))

    def _on_view_changed(self) -> None:
        self.zoom_button.setText(f"{self.roll.zoom_percent}%")
        self.overview.set_view(self.roll.view_left_tick, self.roll.visible_ticks)

    def _refresh_title(self) -> None:
        if self.session is None:
            self.setWindowTitle(APP_NAME)
            return
        mark = " •" if self.session.dirty else ""
        self.setWindowTitle(f"{APP_NAME} — {Path(self.session.project.source).name}{mark}")

    def _update_enabled(self) -> None:
        has = self.session is not None
        self.act_play.setEnabled(has)
        self.act_loop.setEnabled(has)
        self.act_zoom_in.setEnabled(has)
        self.act_zoom_out.setEnabled(has)
        self.act_save.setEnabled(has)
        self.act_export_midi.setEnabled(has)
        self.act_export_mp3.setEnabled(has)
        self.export_button.setEnabled(has)
        self.act_resegment.setEnabled(has)
        self.act_auto_pedal.setEnabled(has)
        self.act_stereo.setEnabled(has)
        self.act_master.setEnabled(has)
        self.bpm_box.setEnabled(has)
        self.act_undo.setEnabled(has and self.session.can_undo())
        self.act_redo.setEnabled(has and self.session.can_redo())

    def _say(self, message: str) -> None:
        self._status_message.setText(message)

    # -- window events ------------------------------------------------------ #
    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.toLocalFile().lower().endswith((".mid", ".midi")):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event) -> None:  # noqa: N802
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.suffix.lower() in (".mid", ".midi"):
                self.load(path)
                return

    def closeEvent(self, event) -> None:  # noqa: N802
        self.player.stop()
        if self.session is None or not self.session.dirty:
            event.accept()
            return
        answer = QMessageBox.question(
            self, tr("还有未保存的力度调整"),
            tr("关闭前要保存工程吗？\n（不保存的话，这次的段和力度都会丢失）"),
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Cancel:
            event.ignore()
            return
        if answer == QMessageBox.StandardButton.Save:
            self.save_project()
        event.accept()
