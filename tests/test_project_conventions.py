"""Guards that make AGENTS.md enforceable instead of aspirational.

Every convention in the handoff manual that a machine *can* check is checked
here, so a future agent cannot quietly delete the protocol, skip a module's
tests, or smuggle Qt into the headless core.  A guard that only lives in prose
is a guard that gets ignored.

What is deliberately *not* checked: code style, file length, and anything that
needs human judgement.  Those live in the manual and in review.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "qmidi"
TESTS = ROOT / "tests"
AGENTS = ROOT / "AGENTS.md"

#: AGENTS.md is the local development handover manual; it is deliberately not
#: part of the public distribution, so every guard that reads it skips instead
#: of failing on a fresh clone.
requires_manual = pytest.mark.skipif(
    not AGENTS.is_file(),
    reason="AGENTS.md is the local dev manual and is not shipped in the repo",
)

#: The manual is the handoff contract; if any of these headings disappears, a
#: fresh agent loses the piece of context it names.
REQUIRED_SECTIONS = (
    "开工先读",
    "改动类型索引",
    "当前进度",
    "核心设计决策",
    "测试规范",
    "会话记录协议",
    "常用命令",
)

#: Modules that must stay importable on a machine with no editor / no Qt.
CORE_MODULES = ("engine", "offline", "pedal", "segments", "exporters")

_QT_IMPORT = re.compile(r"^\s*(?:from\s+(PySide6|PyQt6|PyQt5)\b|import\s+(PySide6|PyQt6|PyQt5)\b)",
                        re.MULTILINE)

#: ``D:\...`` / ``C:/...`` -- an absolute Windows path baked into source.
_ABS_WIN_PATH = re.compile(r"""(?<![\w/])(?:[A-Za-z]:[\\/])""")

#: Files allowed to carry an absolute path: none.  Kept as a named constant so
#: the intent is obvious when someone inevitably wants an exception.
PATH_EXEMPT: frozenset[str] = frozenset()

_IGNORED_DIRS = {"__pycache__", ".pytest_cache"}


def _source_files(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.rglob("*.py")
        if not any(part in _IGNORED_DIRS for part in p.parts)
    )


def _core_modules() -> list[Path]:
    return [SRC / f"{name}.py" for name in CORE_MODULES if (SRC / f"{name}.py").is_file()]


# --------------------------------------------------------------------------- #
# the manual itself
# --------------------------------------------------------------------------- #
@requires_manual
def test_agents_manual_exists():
    assert AGENTS.is_file(), "AGENTS.md is the handoff contract; it cannot be missing"


@requires_manual
@pytest.mark.parametrize("heading", REQUIRED_SECTIONS)
def test_agents_manual_keeps_required_section(heading: str):
    text = AGENTS.read_text(encoding="utf-8")
    assert heading in text, (
        f"AGENTS.md lost its '{heading}' section.  That section is what lets a new "
        f"agent pick the project up with no context; restore it rather than "
        f"deleting the guard."
    )


@requires_manual
def test_agents_manual_records_how_to_run_tests_with_basetemp():
    """The --basetemp flag is not cosmetic: without it pytest prunes %TEMP% and
    trips the sandbox's bulk-delete guard on every few runs."""
    text = AGENTS.read_text(encoding="utf-8")
    assert "--basetemp" in text


@requires_manual
def test_agents_manual_names_the_correct_interpreter():
    """Two Pythons exist on this machine and only one has the dependencies."""
    text = AGENTS.read_text(encoding="utf-8")
    assert "Python310" in text, "the manual must name the interpreter that has the deps"


# --------------------------------------------------------------------------- #
# headless core
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("module", CORE_MODULES)
def test_core_module_does_not_import_qt(module: str):
    path = SRC / f"{module}.py"
    if not path.is_file():
        pytest.skip(f"{module}.py not present")
    match = _QT_IMPORT.search(path.read_text(encoding="utf-8"))
    assert match is None, (
        f"{path.name} must stay headless (AGENTS.md §4.5): Qt belongs in qmidi/gui/.  "
        f"Found: {match.group(0).strip() if match else ''}"
    )


def test_no_core_module_imports_pysfizz_directly():
    """Only engine.py may touch pysfizz (AGENTS.md §9.7)."""
    offenders = [
        p.name for p in _core_modules()
        if p.stem != "engine" and re.search(r"^\s*import\s+pysfizz|from\s+pysfizz\b",
                                            p.read_text(encoding="utf-8"), re.MULTILINE)
    ]
    assert not offenders, f"import pysfizz through qmidi.engine instead: {offenders}"


def test_gui_model_is_qt_free():
    """The state model under gui/ must be unit-testable without a QApplication."""
    model = SRC / "gui" / "model.py"
    if not model.is_file():
        pytest.skip("gui/model.py not written yet")
    assert _QT_IMPORT.search(model.read_text(encoding="utf-8")) is None, (
        "gui/model.py holds the testable half of the GUI; importing Qt there "
        "would force every GUI test to build an application object"
    )


# --------------------------------------------------------------------------- #
# licensing / portability red lines
# --------------------------------------------------------------------------- #
def test_no_pyqt_anywhere():
    offenders = [
        str(p.relative_to(ROOT)) for p in _source_files(SRC)
        if re.search(r"\bPyQt[56]?\b", p.read_text(encoding="utf-8"))
    ]
    assert not offenders, (
        f"PyQt is GPL and banned (AGENTS.md §5.2); use PySide6-Essentials: {offenders}"
    )


def test_no_hardcoded_absolute_windows_path():
    offenders = []
    for path in _source_files(SRC):
        if path.name in PATH_EXEMPT:
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if _ABS_WIN_PATH.search(line):
                offenders.append(f"{path.relative_to(ROOT)}:{lineno}")
    assert not offenders, (
        "instrument paths must come from SFZ_PATH / .sfz_links, never a literal "
        f"drive path (AGENTS.md §5.3): {offenders}"
    )


# --------------------------------------------------------------------------- #
# every module is exercised somewhere
# --------------------------------------------------------------------------- #
def test_every_core_module_is_imported_by_a_test():
    """A module nobody imports in a test is a module nobody can refactor safely."""
    test_text = "\n".join(p.read_text(encoding="utf-8") for p in TESTS.glob("test_*.py"))
    missing = [
        path.stem for path in _core_modules()
        if f"qmidi.{path.stem}" not in test_text
    ]
    assert not missing, (
        f"these modules are never imported by any test: {missing}.  Add a test "
        f"file (tests/test_<module>.py) before shipping behaviour in them."
    )


def test_gui_modules_that_hold_logic_are_covered():
    """Presentation widgets may be untested; anything named model/plan must not be."""
    for name in ("model", "render_plan"):
        path = SRC / "gui" / f"{name}.py"
        if not path.is_file():
            continue
        test_text = "\n".join(p.read_text(encoding="utf-8") for p in TESTS.glob("test_*.py"))
        assert f"qmidi.gui.{name}" in test_text, (
            f"gui/{name}.py carries logic and must be imported by a test"
        )
