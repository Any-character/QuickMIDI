"""The export dialog: one window for the name, the folder and the format.

Why it replaced two ``QFileDialog`` calls
----------------------------------------
The old flow asked for a file name and nothing else: the container came from
whichever menu item was clicked, and the audio settings were hard-coded (16 bit,
48 kHz, 192 kbps).  A user who wanted 24-bit FLAC had no way to ask for it, and
renaming meant retyping a whole path in a system dialog.

The layout follows a DAW's export screen: a bare file name, a folder, a format,
and only the settings that format can actually use.  Changing the format hides
the rows that stop applying (a bit depth on MP3, everything on MIDI) instead of
leaving greyed-out controls that imply the choice still matters.

Everything format-related lives in :mod:`qmidi.exporters` -- Qt-free and tested
there; this module only moves widgets and reads them back.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..exporters import (
    AUDIO_FORMATS,
    BITRATES,
    DEFAULT_BITRATE,
    DEFAULT_SAMPLE_RATE,
    audio_format,
    compose_export_path,
)
from ..i18n import tr

__all__ = ["ExportChoice", "ExportDialog"]

#: MIDI is not an :class:`AudioFormat` (it carries no audio settings at all),
#: so its extension lives here.
_MIDI_EXTENSION = ".mid"

#: Format key -> combo box text.  MIDI leads: handing the file to notation
#: software is the other half of what this program is for.
_FORMAT_LABELS = {
    "midi": "MIDI 文件（保留原始结构，不渲染音频）",
    "wav": "WAV 文件（无损，体积最大）",
    "flac": "FLAC 文件（无损，体积小于 WAV）",
    "mp3": "MP3 文件（有损，最通用）",
}


@dataclass(frozen=True)
class ExportChoice:
    """What the user asked for, resolved to one path and its settings."""

    path: Path
    fmt: str
    bitrate: int = 0
    sample_rate: int = 0
    bit_depth: int = 0

    @property
    def extension(self) -> str:
        if self.fmt == "midi":
            return _MIDI_EXTENSION
        return audio_format(self.fmt).extension

    @property
    def is_midi(self) -> bool:
        return self.fmt == "midi"


class ExportDialog(QDialog):
    """Name, folder, format, and the settings that apply to that format."""

    def __init__(self, parent=None, *, source: Path, fmt: str = "mp3",
                 directory: Optional[Path] = None,
                 name: Optional[str] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("导出"))
        self.setModal(True)
        self.setMinimumWidth(560)

        source = Path(source)
        self._name = QLineEdit(name if name is not None else f"{source.stem}_qmidi")
        self._name.setPlaceholderText(tr("文件名（不用写扩展名）"))
        self._name.selectAll()

        self._directory = QLineEdit(str(directory if directory is not None else source.parent))
        browse = QToolButton(self)
        browse.setText("…")
        browse.setToolTip(tr("选择导出目录"))
        browse.clicked.connect(self._pick_directory)
        directory_row = QWidget(self)
        directory_layout = QHBoxLayout(directory_row)
        directory_layout.setContentsMargins(0, 0, 0, 0)
        directory_layout.setSpacing(6)
        directory_layout.addWidget(self._directory, 1)
        directory_layout.addWidget(browse)

        self._format = QComboBox(self)
        for key in ("midi",) + tuple(info.key for info in AUDIO_FORMATS):
            self._format.addItem(tr(_FORMAT_LABELS[key]), key)
        index = self._format.findData(fmt)
        self._format.setCurrentIndex(max(0, index))

        self._quality = QComboBox(self)
        for rate in BITRATES:
            self._quality.addItem(f"{rate} kbps", rate)
        self._quality.setCurrentIndex(BITRATES.index(DEFAULT_BITRATE))

        self._rate = QComboBox(self)
        self._depth = QComboBox(self)

        self._form = QFormLayout()
        self._form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self._form.addRow(tr("文件名："), self._name)
        self._form.addRow(tr("目录："), directory_row)
        self._form.addRow(tr("格式："), self._format)
        self._form.addRow(tr("品质："), self._quality)
        self._form.addRow(tr("采样率："), self._rate)
        self._form.addRow(tr("位深度："), self._depth)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel, self)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(tr("导出"))
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(tr("取消"))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(self._form)
        layout.addSpacing(6)
        layout.addWidget(buttons)

        self._format.currentIndexChanged.connect(self._sync_format)
        self._sync_format()

    # -- format-dependent rows --------------------------------------------- #
    def _sync_format(self) -> None:
        """Rebuild the rate/depth lists for the chosen container."""
        key = str(self._format.currentData())
        info = None if key == "midi" else audio_format(key)

        self._form.setRowVisible(self._quality, key == "mp3")
        self._form.setRowVisible(self._rate, info is not None)
        self._form.setRowVisible(self._depth, bool(info is not None and info.bit_depths))

        if info is None:
            return

        wanted_rate = int(self._rate.currentData() or DEFAULT_SAMPLE_RATE)
        if wanted_rate not in info.sample_rates:
            wanted_rate = DEFAULT_SAMPLE_RATE
        self._rate.blockSignals(True)
        self._rate.clear()
        for rate in info.sample_rates:
            self._rate.addItem(f"{rate} Hz", rate)
        self._rate.setCurrentIndex(max(0, self._rate.findData(wanted_rate)))
        self._rate.blockSignals(False)

        if not info.bit_depths:
            return
        wanted_depth = int(self._depth.currentData() or info.default_bit_depth)
        if wanted_depth not in info.bit_depths:
            wanted_depth = info.default_bit_depth
        self._depth.blockSignals(True)
        self._depth.clear()
        for depth in info.bit_depths:
            label = tr("32 位浮点") if depth == 32 else tr("{n} 位").format(n=depth)
            self._depth.addItem(label, depth)
        self._depth.setCurrentIndex(max(0, self._depth.findData(wanted_depth)))
        self._depth.blockSignals(False)

    # -- result ------------------------------------------------------------ #
    def choice(self) -> ExportChoice:
        """The current selection, with the extension composed onto the name."""
        key = str(self._format.currentData())
        info = None if key == "midi" else audio_format(key)
        extension = info.extension if info is not None else _MIDI_EXTENSION
        path = compose_export_path(
            self._directory.text().strip() or ".", self._name.text(), extension)
        return ExportChoice(
            path=path,
            fmt=key,
            bitrate=int(self._quality.currentData() or 0) if key == "mp3" else 0,
            sample_rate=int(self._rate.currentData() or 0) if info is not None else 0,
            bit_depth=(int(self._depth.currentData() or 0)
                       if info is not None and info.bit_depths else 0),
        )

    # -- events ------------------------------------------------------------ #
    def _pick_directory(self) -> None:
        start = self._directory.text().strip() or str(Path.home())
        chosen = QFileDialog.getExistingDirectory(self, tr("选择导出目录"), start)
        if chosen:
            self._directory.setText(chosen)

    def accept(self) -> None:  # noqa: D102 - QDialog hook
        if not self._name.text().strip():
            QMessageBox.warning(self, tr("还差一个文件名"), tr("请填写文件名。"))
            return
        choice = self.choice()
        if choice.path.exists():
            answer = QMessageBox.question(
                self, tr("文件已存在"),
                tr("{name} 已存在，要覆盖吗？").format(name=choice.path.name),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                return
        super().accept()
