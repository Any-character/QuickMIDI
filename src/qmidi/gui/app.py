"""Application bootstrap: palette, style sheet, and the one window."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import QApplication

_STYLE = """
QWidget { background: #15171c; color: #c9d1dc; font-size: 12px; }
QMainWindow::separator { background: #2e333d; width: 1px; }
QToolBar { background: #1c1f26; border: 0; border-bottom: 1px solid #2e333d; padding: 4px; spacing: 4px; }
QToolBar QToolButton { padding: 5px 10px; border-radius: 4px; }
QToolBar QToolButton:hover { background: #2a2f3a; }
QToolBar QToolButton:pressed { background: #343b49; }
QToolBar QToolButton:disabled { color: #5a6270; }
QToolBar QToolButton:checked { background: #2f5f8f; border: 1px solid #4fa3ff; color: #fff; }

/* Tempo field: reads as a value in the transport, not as a form control. */
QLabel#bpmLabel { color: #6f7887; padding: 0 4px 0 8px; font-size: 9pt; }
QToolBar QDoubleSpinBox#bpmBox {
    background: #1a1d23;
    border: 1px solid #2e333d;
    border-radius: 4px;
    color: #e2e7ef;
    font-weight: 600;
    padding: 2px 4px;
}
QToolBar QDoubleSpinBox#bpmBox:focus { border-color: #4fa3ff; background: #1f242c; }
QToolBar QDoubleSpinBox#bpmBox:disabled { color: #5a6270; border-color: #262a32; }

QWidget#trackPanel { background: #1c1f26; border-right: 1px solid #2e333d; }
QFrame#trackRow { background: #20242c; border-radius: 5px; }
QFrame#trackRow:hover { background: #262b35; }
QFrame#trackRow[active="true"] { background: #2a3140; border: 1px solid #4fa3ff; }
QFrame#trackRow[silent="true"] QLabel { color: #6b7280; }
QLabel#trackDetail { color: #79828f; font-size: 10px; }
QLabel#panelTitle { font-weight: 600; color: #e2e7ef; }
QLabel#panelHint { color: #6f7887; font-size: 10px; }
QLabel#gainHint { color: #6f7887; padding-right: 8px; }
QLabel#positionLabel { color: #e2e7ef; font-weight: 600; padding-right: 14px; }

QToolButton#muteButton { background: #262b35; border: 1px solid #343b49; border-radius: 3px; font-weight: 600; }
QToolButton#muteButton:checked { background: #a8442f; border-color: #d4614a; color: #fff; }
QToolButton#soloButton { background: #262b35; border: 1px solid #343b49; border-radius: 3px; font-weight: 600; }
QToolButton#soloButton:checked { background: #b58a2c; border-color: #e0b146; color: #101318; }

QWidget#segmentPanel { background: #1c1f26; border-top: 1px solid #2e333d; }
QLabel#segmentTitle { font-weight: 600; color: #ffd9a3; }
QLabel#segmentInfo { color: #9aa4b2; }

QPushButton { background: #262b35; border: 1px solid #343b49; border-radius: 4px; padding: 4px 8px; }
QPushButton:hover { background: #2f3542; }
QPushButton:pressed { background: #3a4251; }
QPushButton:disabled { color: #5a6270; border-color: #2a2f3a; }

QSpinBox { background: #20242c; border: 1px solid #343b49; border-radius: 4px; padding: 2px 4px; }

QDialog { background: #15171c; }
QTabWidget::pane { border: 1px solid #2e333d; border-radius: 4px; top: -1px; }
QTabBar::tab { background: #1a1d23; color: #9aa4b2; padding: 6px 16px; border: 1px solid #2e333d; border-bottom: 0; }
QTabBar::tab:selected { background: #262c36; color: #e2e7ef; }
QTabBar::tab:hover { color: #c9d1dc; }
QTreeWidget { background: #1c1f26; alternate-background-color: #20242c; border: 1px solid #2e333d; border-radius: 4px; }
QTreeWidget::item { padding: 3px 4px; }
QTreeWidget::item:selected { background: #2f5f8f; color: #ffffff; }
QHeaderView::section { background: #20242c; color: #9aa4b2; border: 0; border-bottom: 1px solid #2e333d; padding: 4px; }
QLabel#shortcutStatus { color: #9aa4b2; }
QTextBrowser#aboutPage { background: #1c1f26; border: 1px solid #2e333d; border-radius: 4px; padding: 6px; }

QStatusBar { background: #1c1f26; border-top: 1px solid #2e333d; color: #9aa4b2; }
QScrollBar:vertical { background: #15171c; width: 10px; }
QScrollBar::handle:vertical { background: #3a4251; border-radius: 5px; }
QToolTip { background: #2a2f3a; color: #dfe5ee; border: 1px solid #3c4453; }
"""


def _dark_palette(app: QApplication) -> None:
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#15171c"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#c9d1dc"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#1c1f26"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#c9d1dc"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#262b35"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#c9d1dc"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#4fa3ff"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#0d1014"))
    app.setPalette(palette)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv if argv is None else argv)
    # The language must be known *before* MainWindow is imported:
    # ``keymap.DEFAULTS`` builds its translated titles at import time.
    from ..i18n import set_language
    from ..settings import Settings

    set_language(Settings.load().language)

    from .main_window import APP_NAME, MainWindow

    app = QApplication(args)
    app.setApplicationName(APP_NAME)
    _dark_palette(app)
    app.setStyleSheet(_STYLE)

    window = MainWindow()
    window.show()
    for candidate in args[1:]:
        path = Path(candidate)
        if path.suffix.lower() in (".mid", ".midi") and path.is_file():
            window.load(path)
            break
    return app.exec()
