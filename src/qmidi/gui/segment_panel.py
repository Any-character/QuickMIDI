"""Bottom inspector: what the selected segment is, and how to retune it.

The whole point of the program is this panel: pick a segment, nudge its
velocity, hear it immediately.  So the controls are big, the numbers are black
on a dark panel, and every action also has a keyboard shortcut on the window --
a mouse trip between the roll and a spin box is exactly the friction the tool
is meant to remove.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..i18n import tr
from .model import Session


def _button(text: str, tooltip: str, width: int = 44) -> QPushButton:
    button = QPushButton(text)
    button.setFixedWidth(width)
    button.setToolTip(tooltip)
    return button


class SegmentPanel(QWidget):
    """Shows the selected segment and emits edit intents (never edits directly)."""

    deltaRequested = Signal(int)
    targetRequested = Signal(int)
    clearRequested = Signal()
    mergeRequested = Signal()
    splitRequested = Signal()
    auditionRequested = Signal()
    pedalCleared = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("segmentPanel")
        self.setFixedHeight(96)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 6, 10, 6)
        outer.setSpacing(4)

        top = QHBoxLayout()
        top.setSpacing(10)
        self.title = QLabel(tr("未选择力度段"), self)
        self.title.setObjectName("segmentTitle")
        self.info = QLabel("", self)
        self.info.setObjectName("segmentInfo")
        top.addWidget(self.title)
        top.addWidget(self.info, 1)
        #: Filled by the window from the live key map: a hint that hard-codes
        #: the keys is a hint that lies the moment somebody rebinds them.
        self.hint = QLabel("", self)
        self.hint.setObjectName("panelHint")
        top.addWidget(self.hint)
        outer.addLayout(top)

        row = QHBoxLayout()
        row.setSpacing(6)
        self.minus9 = _button("−9", tr("力度 −9"))
        self.minus3 = _button("−3", tr("力度 −3"))
        self.spin = QSpinBox(self)
        self.spin.setRange(1, 127)
        self.spin.setFixedWidth(64)
        self.spin.setToolTip(tr("把这一段所有音符钉在这个绝对力度上"))
        self.set_button = _button(tr("设为"), tr("整段设为这个力度（绝对）"), 52)
        self.plus3 = _button("＋3", tr("力度 +3"))
        self.plus9 = _button("＋9", tr("力度 +9"))
        self.clear = _button(tr("清零"), tr("撤销这一段的调整"), 52)
        self.merge = _button(tr("合并→"), tr("与下一段合并"), 64)
        self.split = _button(tr("拆分"), tr("在光标处拆开这一段"), 52)
        self.pedal_clear = _button(tr("清踏板"), tr("清空当前轨的所有踏板"), 62)
        self.audition = _button(tr("循环本段"), tr("打开循环并从这一段开始播"), 78)

        for widget in (self.minus9, self.minus3):
            row.addWidget(widget)
        row.addWidget(self.spin)
        row.addWidget(self.set_button)
        for widget in (self.plus3, self.plus9):
            row.addWidget(widget)
        row.addSpacing(12)
        row.addWidget(self.clear)
        row.addWidget(self.merge)
        row.addWidget(self.split)
        row.addSpacing(12)
        row.addWidget(self.pedal_clear)
        row.addSpacing(12)
        row.addWidget(self.audition)
        row.addStretch(1)
        outer.addLayout(row)

        self.minus9.clicked.connect(lambda: self.deltaRequested.emit(-9))
        self.minus3.clicked.connect(lambda: self.deltaRequested.emit(-3))
        self.plus3.clicked.connect(lambda: self.deltaRequested.emit(3))
        self.plus9.clicked.connect(lambda: self.deltaRequested.emit(9))
        self.set_button.clicked.connect(lambda: self.targetRequested.emit(self.spin.value()))
        self.clear.clicked.connect(self.clearRequested)
        self.merge.clicked.connect(self.mergeRequested)
        self.split.clicked.connect(self.splitRequested)
        self.audition.clicked.connect(self.auditionRequested)
        self.pedal_clear.clicked.connect(self.pedalCleared)
        self.refresh(None)

    # ------------------------------------------------------------------ #
    def set_hint(self, text: str) -> None:
        """Show the keys that are actually bound right now."""
        self.hint.setText(text)

    def refresh(self, session: Optional[Session]) -> None:
        segment = None if session is None else session.selected_segment()
        enabled = segment is not None
        for widget in (self.minus9, self.minus3, self.plus3, self.plus9,
                       self.set_button, self.clear, self.merge, self.split,
                       self.spin):
            widget.setEnabled(enabled)
        self.audition.setEnabled(session is not None)
        self.pedal_clear.setEnabled(session is not None and bool(session.pedal_spans()))
        if segment is None or session is None:
            self.title.setText(tr("未选择力度段"))
            self.info.setText("")
            return

        track = session.active()
        index = session.selected_index()
        notes = track.notes_in(segment)
        values = sorted({n.velocity for n in notes})
        resolved = segment.velocity_for(notes[0]) if notes else 0
        low = min((segment.velocity_for(n) for n in notes), default=0)
        high = max((segment.velocity_for(n) for n in notes), default=0)

        self.title.setText(
            tr("轨道 {track} · 第 {index}/{total} 段 · {label}").format(
                track=track.track, index=index + 1,
                total=len(track.segments), label=segment.label))
        origin = "/".join(str(v) for v in values[:6]) + ("…" if len(values) > 6 else "")
        span = f"{segment.start_tick / max(1, session.project.ticks_per_beat):g}–" \
               f"{segment.end_tick / max(1, session.project.ticks_per_beat):g} " \
               + tr("拍")
        self.info.setText(
            "  " + tr("原力度 {origin}  →  现在 {low}–{high}  ·  {notes} 音符  ·  {span}").format(
                origin=origin, low=low, high=high, notes=len(notes), span=span)
            + "  ·  " + (tr("偏移 %+d") % segment.delta if segment.target is None
                         else tr("固定 %d") % segment.target)
        )
        self.spin.blockSignals(True)
        self.spin.setValue(resolved if notes else 96)
        self.spin.blockSignals(False)
