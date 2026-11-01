"""Left-hand track list with mute / solo.

Mute and solo are the feature the user asked for by name ("支持灵活静音或独奏音轨"),
and they are per *mido track* -- in the sample file the pedal lives on tracks of
its own, so muting is not the same as muting a channel.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..i18n import tr
from .model import Session, TrackStats


class TrackRow(QFrame):
    """One track: mute, solo, name, and a one-line summary."""

    activated = Signal(int)
    muteClicked = Signal(int)
    soloClicked = Signal(int)
    renameRequested = Signal(int)

    def __init__(self, stats: TrackStats, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._track = stats.track
        self.setObjectName("trackRow")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        row = QHBoxLayout(self)
        row.setContentsMargins(6, 5, 6, 5)
        row.setSpacing(4)

        self.mute = QToolButton(self)
        self.mute.setText("M")
        self.mute.setCheckable(True)
        self.mute.setFixedSize(22, 20)
        self.mute.setToolTip(tr("静音这一轨"))
        self.mute.setObjectName("muteButton")
        self.mute.clicked.connect(lambda: self.muteClicked.emit(self._track))

        self.solo = QToolButton(self)
        self.solo.setText("S")
        self.solo.setCheckable(True)
        self.solo.setFixedSize(22, 20)
        self.solo.setToolTip(tr("独奏：只听（选中的）这几轨"))
        self.solo.setObjectName("soloButton")
        self.solo.clicked.connect(lambda: self.soloClicked.emit(self._track))

        texts = QVBoxLayout()
        texts.setContentsMargins(0, 0, 0, 0)
        texts.setSpacing(0)
        self.name = QLabel(self)
        self.detail = QLabel(self)
        self.detail.setObjectName("trackDetail")
        texts.addWidget(self.name)
        texts.addWidget(self.detail)

        row.addWidget(self.mute)
        row.addWidget(self.solo)
        row.addLayout(texts, 1)
        self.update_stats(stats, active=False, muted=False, soloed=False)

    def update_stats(self, stats: TrackStats, *, active: bool, muted: bool,
                     soloed: bool) -> None:
        # The index is part of the label because two pianos often share a name
        # ("Piano", "Piano"), and the track number is what the file's own
        # structure uses.
        self.name.setText(f"{stats.track}   {stats.name}")
        self.detail.setText(
            tr("{notes} notes · {segments} segments · velocity {low}–{high}").format(
                notes=stats.notes, segments=stats.segments,
                low=stats.velocity_low, high=stats.velocity_high)
        )
        self.mute.setChecked(muted)
        self.solo.setChecked(soloed)
        self.setProperty("active", "true" if active else "false")
        self.setProperty("silent", "true" if (muted and not soloed) else "false")
        self.style().unpolish(self)
        self.style().polish(self)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.activated.emit(self._track)
        super().mousePressEvent(event)

    def contextMenuEvent(self, event) -> None:  # noqa: N802
        menu = QMenu(self)
        rename = menu.addAction(tr("重命名音轨…"))
        if menu.exec(event.globalPos()) is rename:
            self.renameRequested.emit(self._track)


class TrackPanel(QWidget):
    """Vertical list of :class:`TrackRow`, rebuilt whenever the track count changes."""

    trackActivated = Signal(int)
    muteToggled = Signal(int)
    soloToggled = Signal(int)
    renameRequested = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("trackPanel")
        self.setFixedWidth(232)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        title = QLabel(tr("轨道"), self)
        title.setObjectName("panelTitle")
        layout.addWidget(title)
        self._hint = QLabel(tr("M 静音 · S 独奏（可多选）· 右键改名"), self)
        self._hint.setObjectName("panelHint")
        self._hint.setWordWrap(True)
        layout.addWidget(self._hint)
        self._rows: list[TrackRow] = []
        self._layout = layout
        layout.addStretch(1)
        self._session: Optional[Session] = None

    def set_session(self, session: Optional[Session]) -> None:
        self._session = session
        self.refresh()

    def refresh(self) -> None:
        if self._session is None:
            return
        stats = self._session.stats()
        if len(stats) != len(self._rows):
            for row in self._rows:
                row.deleteLater()
            self._rows = []
            for item in stats:
                row = TrackRow(item, self)
                row.activated.connect(self.trackActivated)
                row.muteClicked.connect(self.muteToggled)
                row.soloClicked.connect(self.soloToggled)
                row.renameRequested.connect(self.renameRequested)
                self._layout.insertWidget(self._layout.count() - 1, row)
                self._rows.append(row)
        for row, item in zip(self._rows, stats):
            row.update_stats(
                item,
                active=item.track == self._session.active_track,
                muted=item.track in self._session.muted,
                soloed=item.track in self._session.solo,
            )
