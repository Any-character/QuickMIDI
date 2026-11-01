"""The shortcuts editor, as a page inside the settings window.

One gesture: click the key cell, press the key you want.  No modifier dropdown
to assemble, no "Ctrl+" prefix to type -- the same interaction every DAW uses,
and the one that teaches the shortcut while you set it.

Conflicts are refused rather than warned about: a key that fires two actions is
a bug that shows up a week later as "undo sometimes closes the file".
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..i18n import tr
from ..keymap import DEFAULTS, KeyMap, KeyMapError, pretty

# Chinese originals; translated at the read boundary like the keymap table.
_IDLE_HINT = "点右边任意一行，然后按下你想用的键。Esc 取消，Backspace 清除这一项。"
_RECORDING_HINT = "正在设置「{label}」：按下新键，Esc 取消，Backspace 清除。"


class _ShortcutTree(QTreeWidget):
    """A tree that can swallow the next key press as a new binding."""

    captured = Signal(str)  # QKeySequence text; "" means "clear this one"
    cancelled = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._recording = False
        self.setColumnCount(2)
        self.setHeaderLabels([tr("动作"), tr("快捷键")])
        self.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.setRootIsDecorated(True)
        # Alternating rows paint with the *palette's* alternate base colour, not
        # the stylesheet's: on this dark theme that produced white rows with
        # light text.  Group headers already separate the sections.
        self.setAlternatingRowColors(False)
        self.setSelectionMode(QTreeWidget.SelectionMode.SingleSelection)

    def start_recording(self) -> None:
        self._recording = True

    def stop_recording(self) -> None:
        self._recording = False

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if not self._recording:
            super().keyPressEvent(event)
            return
        key = event.key()
        # A modifier on its own is not a shortcut; wait for the real key.
        if key in (Qt.Key.Key_Shift, Qt.Key.Key_Control,
                   Qt.Key.Key_Alt, Qt.Key.Key_Meta):
            return
        if key == Qt.Key.Key_Escape:
            self._recording = False
            self.cancelled.emit()
            return
        if key == Qt.Key.Key_Backspace:
            self._recording = False
            self.captured.emit("")
            return
        text = QKeySequence(event.keyCombination()).toString()
        if not text:
            return
        self._recording = False
        self.captured.emit(text)


class ShortcutPage(QWidget):
    """Edit a copy of the key map; the dialog decides when to save it."""

    def __init__(self, keymap: KeyMap, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._draft = keymap.copy()
        self._items: dict[str, QTreeWidgetItem] = {}
        self._recording_item: Optional[QTreeWidgetItem] = None
        self._recording_action: Optional[str] = None
        self._previous = ""

        self.tree = _ShortcutTree(self)
        self._populate()

        self.status = QLabel(tr(_IDLE_HINT), self)
        self.status.setObjectName("shortcutStatus")
        self.status.setWordWrap(True)

        reset = QPushButton(tr("全部恢复默认"), self)
        reset.clicked.connect(self._reset_all)
        bottom = QHBoxLayout()
        bottom.addWidget(reset)
        bottom.addStretch(1)

        intro = QLabel(tr("这些键全局生效（所有工程），点一行再按新键即可修改。"), self)
        intro.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addWidget(self.tree, 1)
        layout.addWidget(self.status)
        layout.addLayout(bottom)

        self.tree.itemClicked.connect(self._on_clicked)
        self.tree.captured.connect(self._on_captured)
        self.tree.cancelled.connect(self._on_cancelled)

    # -- construction ------------------------------------------------------ #
    def _populate(self) -> None:
        self.tree.clear()
        self._items.clear()
        groups: dict[str, QTreeWidgetItem] = {}
        for binding in DEFAULTS:
            # binding() translates group/label on read; DEFAULTS stays Chinese.
            shown = self._draft.binding(binding.action)
            group = groups.get(shown.group)
            if group is None:
                group = QTreeWidgetItem(self.tree, [shown.group, ""])
                font = group.font(0)
                font.setBold(True)
                group.setFont(0, font)
                group.setFlags(Qt.ItemFlag.ItemIsEnabled)
                groups[shown.group] = group
            item = QTreeWidgetItem(
                group,
                [shown.label, pretty(self._draft.sequence(binding.action))],
            )
            item.setData(0, Qt.ItemDataRole.UserRole, binding.action)
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self._items[binding.action] = item
        self.tree.expandAll()

    # -- editing ----------------------------------------------------------- #
    def _on_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        action = item.data(0, Qt.ItemDataRole.UserRole)
        if not action or self._recording_item is not None:
            return
        self._recording_item = item
        self._recording_action = action
        self._previous = item.text(1)
        item.setText(1, tr("按下新键…"))
        self.status.setText(
            tr(_RECORDING_HINT).format(label=self._draft.binding(action).label))
        self.tree.start_recording()
        self.tree.setFocus()

    def _finish_recording(self) -> None:
        self._recording_item = None
        self._recording_action = None
        self.tree.stop_recording()

    def _on_captured(self, text: str) -> None:
        item, action = self._recording_item, self._recording_action
        self._finish_recording()
        if item is None or action is None:
            return
        try:
            self._draft.bind(action, text)
        except KeyMapError as exc:
            item.setText(1, self._previous)
            self.status.setText(tr("{reason}。换一个键，或先改那一项。").format(reason=exc))
            return
        item.setText(1, pretty(self._draft.sequence(action)))
        self.status.setText(
            tr("已是默认设置。") if self._draft.is_default()
            else tr("已修改；点「确定」保存。")
        )

    def _on_cancelled(self) -> None:
        if self._recording_item is not None:
            self._recording_item.setText(1, self._previous)
        self._finish_recording()
        self.status.setText(tr("已取消这一项。"))

    def _reset_all(self) -> None:
        if self._recording_item is not None:
            self._on_cancelled()
        self._draft.reset()
        for action, item in self._items.items():
            item.setText(1, pretty(self._draft.sequence(action)))
        self.status.setText(tr("已恢复全部默认；点「确定」保存。"))

    # -- result ------------------------------------------------------------ #
    @property
    def keymap(self) -> KeyMap:
        return self._draft

    def stop_recording(self) -> None:
        """Called before the dialog closes, so a half-finished rebind is dropped."""
        if self._recording_item is not None:
            self._on_cancelled()
