"""Smoke test for the window itself -- construction, paint, one edit.

Run offscreen so it works in CI and on a machine with no display.  It is
deliberately shallow: the window has no logic worth asserting beyond "it
builds, it loads, it paints, an edit reaches the model".  The rules live in
``tests/test_gui_model.py``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import mido
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def small_midi(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.mid"
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack([mido.MetaMessage("track_name", name="Piano")])
    last = 0
    for i in range(8):
        tick = i * 480
        track.append(mido.Message("note_on", note=60 + i, velocity=60 + i * 5,
                                  time=tick - last))
        last = tick
    for i in range(8):
        track.append(mido.Message("note_on", note=60 + i, velocity=0, time=120))
        last += 120
    track.append(mido.MetaMessage("end_of_track", time=0))
    midi.tracks.append(track)
    midi.save(str(path))
    return path


@pytest.fixture()
def long_midi() -> Path:
    """The real sample piece: long enough for the overview box to have room to scroll.

    The tiny fixture above fits entirely on screen, so its window box covers the
    whole strip and "drag the box" has nowhere to go -- which would make these
    tests pass by accident instead of by behaviour.
    """
    path = ROOT / "example" / "midi" / "Unchained.mid"
    if not path.is_file():
        pytest.skip("example MIDI not present")
    return path


def test_window_builds_loads_and_paints(qt_app, small_midi: Path):
    from qmidi.gui.main_window import MainWindow

    window = MainWindow()
    window.show()
    qt_app.processEvents()
    window.load(small_midi)
    qt_app.processEvents()

    assert window.session is not None
    assert window.session.project.segment_count >= 1
    assert window.roll.grab().width() > 0

    before = window.session.selected_segment().delta
    window._on_delta(-5)
    qt_app.processEvents()
    assert window.session.selected_segment().delta == before - 5
    assert window.session.dirty

    window._undo()
    qt_app.processEvents()
    assert window.session.selected_segment().delta == before

    window._toggle_mute(0)
    assert 0 in window.session.muted
    window._toggle_solo(0)
    assert window.session.visible_tracks() == frozenset({0})

    image = window.roll.grab().toImage()
    assert not image.isNull()
    window.deleteLater()
    qt_app.processEvents()


def test_transport_overview_and_zoom_are_wired(qt_app, small_midi: Path):
    from qmidi.gui.main_window import MainWindow

    window = MainWindow()
    window.show()
    qt_app.processEvents()
    window.load(small_midi)
    qt_app.processEvents()

    assert window.act_loop.isCheckable()
    assert not window.act_loop.isChecked()
    assert not window.session.loop_enabled
    assert window.zoom_button.text().endswith("%")
    assert not window.overview.grab().toImage().isNull()

    before = window.roll.zoom_percent
    window.roll.zoom_time(1.25)
    qt_app.processEvents()
    assert window.roll.zoom_percent > before
    assert window.zoom_button.text() == f"{window.roll.zoom_percent}%"

    # Arming loop programmatically must reach the model, not just the checkbox.
    window.act_loop.setChecked(True)
    qt_app.processEvents()
    assert window.session.loop_enabled
    assert window.roll._loop is not None
    window.act_loop.setChecked(False)
    qt_app.processEvents()
    assert not window.session.loop_enabled
    assert window.roll._loop is None

    window.deleteLater()
    qt_app.processEvents()


def test_clicking_the_pedal_strip_presses_it_down(qt_app, small_midi: Path):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest

    from qmidi.gui.main_window import MainWindow

    window = MainWindow()
    window.show()
    qt_app.processEvents()
    window.load(small_midi)
    qt_app.processEvents()

    roll = window.roll
    lane = roll._pedal_area()
    QTest.mouseClick(roll, Qt.MouseButton.LeftButton,
                     pos=QPoint(roll.width() // 2, lane.center().y()))
    qt_app.processEvents()

    spans = window.session.pedal_spans()
    assert len(spans) == 1
    assert spans[0].end_tick == window.session.total_end_tick()

    window.deleteLater()
    qt_app.processEvents()


def test_dragging_a_velocity_bar_moves_the_selected_segment(qt_app, small_midi: Path):
    from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtTest import QTest

    from qmidi.gui.main_window import MainWindow

    window = MainWindow()
    window.show()
    qt_app.processEvents()
    window.load(small_midi)
    qt_app.processEvents()

    roll = window.roll
    lane = roll._velocity_area()
    x = int(roll.x_of_tick(0)) + 2
    y = lane.center().y()
    before = window.session.selected_segment().scale

    QTest.mousePress(roll, Qt.MouseButton.LeftButton, pos=QPoint(x, y))
    move = QMouseEvent(QEvent.Type.MouseMove, QPointF(x, y - 8), QPointF(x, y - 8),
                       Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton,
                       Qt.KeyboardModifier.NoModifier)
    roll.mouseMoveEvent(move)
    QTest.mouseRelease(roll, Qt.MouseButton.LeftButton, pos=QPoint(x, y - 8))
    qt_app.processEvents()

    # dragging up is louder, and the whole segment is rescaled
    assert window.session.selected_segment().scale > before
    assert window.session.can_undo()

    window._undo()
    assert window.session.selected_segment().scale == pytest.approx(before)

    window.deleteLater()
    qt_app.processEvents()


def test_shortcuts_page_lists_every_action(qt_app):
    from qmidi.gui.shortcut_page import ShortcutPage
    from qmidi.keymap import DEFAULTS, KeyMap

    page = ShortcutPage(KeyMap())
    groups = page.tree.topLevelItemCount()
    listed = sum(page.tree.topLevelItem(i).childCount() for i in range(groups))
    assert groups == 4  # 文件 / 播放 / 编辑 / 视图
    assert listed == len(DEFAULTS)
    page.deleteLater()
    qt_app.processEvents()


def test_settings_dialog_has_the_three_pages(qt_app, tmp_path: Path, monkeypatch):
    from PySide6.QtWidgets import QTabWidget

    from qmidi.gui.settings_dialog import SettingsDialog
    from qmidi.keymap import KeyMap
    from qmidi.settings import Settings

    monkeypatch.setenv("QMIDI_CONFIG_DIR", str(tmp_path))
    dialog = SettingsDialog(Settings(), KeyMap())
    tabs = dialog.findChild(QTabWidget)
    assert tabs is not None
    assert [tabs.tabText(i) for i in range(tabs.count())] == [
        "音源", "显示", "功能说明", "快捷键",
    ]

    # the display page round-trips into the result
    dialog.show_all.setChecked(True)
    assert dialog.result_settings.show_all_tracks
    assert dialog.result_settings.sfz_path == ""

    dialog.deleteLater()
    qt_app.processEvents()


def test_settings_round_trip_through_the_config_file(tmp_path: Path,
                                                     monkeypatch):
    from qmidi.settings import Settings, settings_path

    monkeypatch.setenv("QMIDI_CONFIG_DIR", str(tmp_path / "cfg"))
    instrument = str(tmp_path / "Piano" / "Concert.sfz")
    assert Settings.load() == Settings()          # no file -> defaults
    saved = Settings(sfz_path=instrument, show_all_tracks=True,
                     snap_division=8, language="en").save()
    assert saved == settings_path()
    assert Settings.load() == Settings(sfz_path=instrument, show_all_tracks=True,
                                       snap_division=8, language="en")

    settings_path().write_text("{not json", encoding="utf-8")
    assert Settings.load() == Settings()          # a broken file is ignored


def test_rebinding_reaches_the_window_the_roll_and_the_hint(qt_app, small_midi: Path,
                                                            tmp_path: Path, monkeypatch):
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent

    from qmidi.gui.main_window import MainWindow
    from qmidi.keymap import KeyMap

    monkeypatch.setenv("QMIDI_CONFIG_DIR", str(tmp_path))
    window = MainWindow()
    window.show()
    qt_app.processEvents()
    window.load(small_midi)
    qt_app.processEvents()

    table = KeyMap()
    table.bind("nudge_up", "W")
    table.bind("loop", "Ctrl+L")
    window.keymap = table
    window._apply_keymap()

    assert window.act_loop.shortcut().toString() == "Ctrl+L"
    assert "Ctrl+L" in window.act_loop.toolTip()
    assert window.roll._keymap is table
    assert "W 力度+3" in window.segments.hint.text()

    nudges: list[int] = []
    window.roll.nudgeRequested.connect(nudges.append)
    window.roll.keyPressEvent(
        QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_W, Qt.KeyboardModifier.NoModifier))
    assert nudges == [3]
    # the old key is free again: Up must no longer nudge
    window.roll.keyPressEvent(
        QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Up, Qt.KeyboardModifier.NoModifier))
    assert nudges == [3]

    window.deleteLater()
    qt_app.processEvents()


def _mouse(kind, x, y, button=None):
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    held = Qt.MouseButton.LeftButton if button is None else button
    return QMouseEvent(kind, QPointF(x, y), QPointF(x, y), held, held,
                       Qt.KeyboardModifier.NoModifier)


def test_dragging_the_overview_box_scrolls_without_moving_the_playhead(
        qt_app, long_midi: Path, tmp_path: Path, monkeypatch):
    from PySide6.QtCore import QEvent

    from qmidi.gui.main_window import MainWindow

    monkeypatch.setenv("QMIDI_CONFIG_DIR", str(tmp_path))
    window = MainWindow()
    window.show()
    qt_app.processEvents()
    window.load(long_midi)
    qt_app.processEvents()
    for _ in range(3):
        window.roll.zoom_time(1.25)
    qt_app.processEvents()

    bar = window.overview
    rect = bar._view_rect()
    left_before = window.roll.view_left_tick
    cursor_before = window.session.cursor_tick

    centre = rect.center()
    bar.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress, centre.x(), centre.y()))
    bar.mouseMoveEvent(_mouse(QEvent.Type.MouseMove, centre.x() + 60, centre.y()))
    bar.mouseReleaseEvent(_mouse(QEvent.Type.MouseButtonRelease,
                                 centre.x() + 60, centre.y()))
    qt_app.processEvents()

    assert window.roll.view_left_tick > left_before, "the box should scroll the view"
    assert window.session.cursor_tick == cursor_before, "scrolling must not seek"

    window.deleteLater()
    qt_app.processEvents()


def test_clicking_outside_the_overview_box_seeks(qt_app, long_midi: Path,
                                                 tmp_path: Path, monkeypatch):
    from PySide6.QtCore import QEvent, QPoint

    from qmidi.gui.main_window import MainWindow

    monkeypatch.setenv("QMIDI_CONFIG_DIR", str(tmp_path))
    window = MainWindow()
    window.show()
    qt_app.processEvents()
    window.load(long_midi)
    qt_app.processEvents()
    for _ in range(3):
        window.roll.zoom_time(1.25)
    qt_app.processEvents()

    bar = window.overview
    rect = bar._view_rect()
    plot = bar._plot()
    y = rect.center().y()
    x = next(candidate for candidate in (plot.left() + 3, plot.right() - 3)
             if not rect.contains(QPoint(candidate, y)))
    cursor_before = window.session.cursor_tick

    bar.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress, x, y))
    bar.mouseReleaseEvent(_mouse(QEvent.Type.MouseButtonRelease, x, y))
    qt_app.processEvents()

    assert window.session.cursor_tick != cursor_before

    window.deleteLater()
    qt_app.processEvents()


def test_the_velocity_gutter_reads_out_the_hovered_note(qt_app, small_midi: Path,
                                                        tmp_path: Path, monkeypatch):
    from PySide6.QtCore import QEvent, Qt

    from qmidi.gui.main_window import MainWindow

    monkeypatch.setenv("QMIDI_CONFIG_DIR", str(tmp_path))
    window = MainWindow()
    window.show()
    qt_app.processEvents()
    window.load(small_midi)
    qt_app.processEvents()
    window._activate_track(0)
    qt_app.processEvents()
    window.roll.zoom_time(4.0)
    qt_app.processEvents()

    roll = window.roll
    lane = roll._velocity_area()
    notes = roll._notes_by_track.get(0) or []
    note = notes[len(notes) // 2]
    x = int(roll.x_of_tick(note[2]))
    if x <= lane.left() or x > lane.right():
        pytest.skip("the note landed outside the visible lane")
    roll.mouseMoveEvent(_mouse(QEvent.Type.MouseMove, x, lane.center().y(),
                               button=Qt.MouseButton.NoButton))
    assert roll._hover_velocity == note[5]

    roll.leaveEvent(QEvent(QEvent.Type.Leave))  # clearing the readout must not raise
    assert roll._hover_velocity is None
    assert roll._velocity_text  # falls back to the segment range

    window.deleteLater()
    qt_app.processEvents()


def test_only_the_active_track_is_drawn_by_default(qt_app, long_midi: Path,
                                                   tmp_path: Path, monkeypatch):
    from qmidi.gui.main_window import MainWindow

    monkeypatch.setenv("QMIDI_CONFIG_DIR", str(tmp_path))
    window = MainWindow()
    window.show()
    qt_app.processEvents()
    window.load(long_midi)
    qt_app.processEvents()

    drawn = {note[0] for note in window.roll._notes}
    assert drawn == {window.session.active_track}, "only the active track by default"

    window.roll.set_show_all_tracks(True)
    qt_app.processEvents()
    assert len({note[0] for note in window.roll._notes}) > 1, "all tracks when asked"

    window.roll.set_show_all_tracks(False)
    qt_app.processEvents()
    assert {note[0] for note in window.roll._notes} == {window.session.active_track}

    window.deleteLater()
    qt_app.processEvents()


def test_switching_tracks_recentres_the_rows(qt_app, long_midi: Path,
                                             tmp_path: Path, monkeypatch):
    from qmidi.gui.main_window import MainWindow

    monkeypatch.setenv("QMIDI_CONFIG_DIR", str(tmp_path))
    window = MainWindow()
    window.show()
    qt_app.processEvents()
    window.load(long_midi)
    qt_app.processEvents()

    window._activate_track(3)  # the other piano part
    qt_app.processEvents()
    assert window.session.active_track == 3
    assert {note[0] for note in window.roll._notes} == {3}
    # the rows were re-fitted, so the new track's register is on screen
    assert 21 <= window.roll._top_pitch <= 108

    window.deleteLater()
    qt_app.processEvents()


def test_clicking_the_ruler_snaps_the_playhead(qt_app, long_midi: Path,
                                               tmp_path: Path, monkeypatch):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest

    from qmidi.gui.main_window import MainWindow

    monkeypatch.setenv("QMIDI_CONFIG_DIR", str(tmp_path))
    window = MainWindow()
    window.show()
    qt_app.processEvents()
    window.load(long_midi)
    qt_app.processEvents()
    for _ in range(4):
        window.roll.zoom_time(1.25)
    qt_app.processEvents()

    step = window.session.project.ticks_per_beat // 4
    roll = window.roll
    grid = step * 200
    target = grid + step // 3  # a third of a cell past the line
    x = int(roll.x_of_tick(target))
    QTest.mouseClick(roll, Qt.MouseButton.LeftButton, pos=QPoint(x, 10))
    qt_app.processEvents()

    assert window.session.cursor_tick % step == 0, "the playhead must land on a grid line"
    assert window.session.cursor_tick == grid

    # and the setting really turns it off
    window.session.snap_division = 0
    x = int(roll.x_of_tick(target))
    QTest.mouseClick(roll, Qt.MouseButton.LeftButton, pos=QPoint(x, 10))
    qt_app.processEvents()
    assert window.session.cursor_tick % step != 0

    window.deleteLater()
    qt_app.processEvents()


def test_export_is_one_button_with_two_formats(qt_app, tmp_path: Path, monkeypatch):
    from qmidi.gui.main_window import MainWindow

    monkeypatch.setenv("QMIDI_CONFIG_DIR", str(tmp_path))
    window = MainWindow()
    window.show()
    qt_app.processEvents()

    assert window.export_button.text() == "导出"
    entries = window.export_menu.actions()
    assert entries == [window.act_export_midi, window.act_export_mp3]
    assert window.act_export_midi.text() == "导出 MIDI"
    assert window.act_export_mp3.text() == "导出音频"

    # The menu is the discoverable path; the shortcuts must still work without
    # opening it, or merging the buttons would have cost the fast path.
    assert window.act_export_midi.shortcut().toString() == "Ctrl+E"
    assert window.act_export_mp3.shortcut().toString() == "Ctrl+Shift+E"
    assert not window.export_button.isEnabled(), "nothing open, nothing to export"

    window.deleteLater()
    qt_app.processEvents()


def test_english_language_translates_the_toolbar(qt_app, long_midi: Path,
                                                 tmp_path: Path, monkeypatch):
    """End to end: an en settings file gives an English toolbar and tooltips."""
    from qmidi.i18n import set_language
    from qmidi.gui.main_window import MainWindow

    cfg = tmp_path / "cfg"
    cfg.mkdir()
    (cfg / "settings.json").write_text('{"language": "en"}', encoding="utf-8")
    monkeypatch.setenv("QMIDI_CONFIG_DIR", str(cfg))
    # Binding.text is translated on *read*, so this must be set before the
    # window builds its actions -- the same order app.main uses.
    set_language("en")
    window = MainWindow()
    window.show()
    qt_app.processEvents()
    window.load(long_midi)
    qt_app.processEvents()

    assert window.act_open.text() == "Open"
    assert window.act_undo.text() == "Undo"
    assert "Open MIDI file" in window.act_open.toolTip()

    set_language("zh")  # other tests in this process expect Chinese
    window.deleteLater()
    qt_app.processEvents()


def test_english_language_translates_the_settings_tabs(qt_app, tmp_path: Path,
                                                       monkeypatch):
    """In English mode the settings dialog's four tabs must be English,
    and the About page must serve the English text, not the Chinese fallback."""
    from PySide6.QtWidgets import QTabWidget

    from qmidi.gui.settings_dialog import SettingsDialog
    from qmidi.i18n import set_language
    from qmidi.keymap import KeyMap
    from qmidi.settings import Settings

    set_language("en")
    dialog = SettingsDialog(Settings(), KeyMap.load())
    tabs = dialog.findChildren(QTabWidget)[0]
    titles = [tabs.tabText(i) for i in range(tabs.count())]
    assert titles == ["Instrument", "Display",
                      "About the features", "Shortcuts"]
    dialog.deleteLater()
    set_language("zh")  # other tests in this process expect Chinese
    qt_app.processEvents()


def test_display_page_keeps_every_control_in_one_layout(qt_app, tmp_path: Path,
                                                        monkeypatch):
    """Two QVBoxLayout(page) calls in one page = an orphan layout that hides
    everything added to it.  This caught the bug where only the language combo
    was visible and snapping/export quality had vanished (2026-11-01)."""
    from PySide6.QtWidgets import QComboBox

    from qmidi.gui.settings_dialog import SettingsDialog
    from qmidi.keymap import KeyMap
    from qmidi.settings import Settings

    monkeypatch.setenv("QMIDI_CONFIG_DIR", str(tmp_path))
    dialog = SettingsDialog(Settings(), KeyMap())
    page = dialog.show_all.parentWidget()
    inside = set()
    for i in range(page.layout().count()):
        widget = page.layout().itemAt(i).widget()
        if widget is not None:
            inside.add(widget)

    assert dialog.show_all in inside
    assert dialog.snap in inside
    assert dialog.export_quality in inside
    assert dialog.language in inside

    dialog.deleteLater()
    qt_app.processEvents()


def test_export_dialog_switches_its_settings_with_the_format(qt_app, tmp_path: Path):
    """MP3 shows a bitrate, WAV/FLAC a bit depth, MIDI neither -- and the
    chosen path carries exactly one extension, whatever the user typed."""
    from qmidi.gui.export_dialog import ExportDialog

    dialog = ExportDialog(None, source=tmp_path / "song.mid", fmt="mp3")
    qt_app.processEvents()

    mp3 = dialog.choice()
    assert mp3.fmt == "mp3"
    assert mp3.extension == ".mp3"
    assert mp3.bitrate in (128, 192, 320)
    assert mp3.bit_depth == 0

    dialog._format.setCurrentIndex(dialog._format.findData("wav"))
    wav = dialog.choice()
    assert wav.extension == ".wav"
    assert wav.bit_depth == 24
    assert wav.sample_rate == 48000

    dialog._format.setCurrentIndex(dialog._format.findData("flac"))
    assert dialog.choice().bit_depth in (16, 24)

    dialog._format.setCurrentIndex(dialog._format.findData("midi"))
    qt_app.processEvents()
    midi = dialog.choice()
    assert midi.is_midi
    assert midi.path.name == "song_qmidi.mid"
    assert midi.sample_rate == 0 and midi.bit_depth == 0

    dialog._format.setCurrentIndex(dialog._format.findData("flac"))
    dialog._name.setText("take 3.mid")
    assert dialog.choice().path.name == "take 3.flac"

    dialog.deleteLater()
    qt_app.processEvents()


def test_window_survives_an_empty_project(qt_app):
    """A file with no notes must not crash the roll or the panels."""
    from qmidi.gui.main_window import MainWindow

    window = MainWindow()
    qt_app.processEvents()
    assert window.session is None
    assert window.roll.grab().isNull() is False
    window.deleteLater()
    qt_app.processEvents()
