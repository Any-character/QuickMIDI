"""Thin adapter over the pysfizz sfz engine -- the SFZ seam of the whole app.

Why this module exists
----------------------
pysfizz 0.1.3's public wrapper (``pysfizz.Synth``) only exposes
``load_sfz_file`` / ``render_note`` / ``get_note_info`` / ``playable_keys``.
Every real-time action we need (``note_on`` / ``note_off`` / ``cc`` /
``render_block`` / ``all_sound_off``) lives on the nanobind binding object
reached through the wrapper's ``_synth`` attribute.  Upstream has published no
release that promotes those to the public wrapper (PyPI tops out at 0.1.3).

So: rather than let the rest of the codebase reach into ``_synth`` (which
would be GPL-free but still a scatter-shot dependency on a private name), we
funnel *all* of it through exactly six actions:

    load_sfz(path)   note_on(pitch, vel)   note_off(pitch)
    cc(num, val)     render_block()        all_sound_off()

Business code outside this file must not import pysfizz or touch ``_synth``.

Forward compatibility
---------------------
``_bind()`` resolves each action against the public wrapper first and falls
back to the inner binding.  If a future pysfizz release promotes ``note_on``
etc. to the public wrapper, this file keeps working with zero changes.

Freewheeling: the one setting that is not optional
-------------------------------------------------
Measured on this machine (Concert Grand, 48 kHz):

    freewheeling=False, blocks pushed as fast as possible -> every voice dies
        after ~0.21 s, regardless of pitch or pedal state
    freewheeling=False, blocks paced to wall-clock real time -> note sustains
        correctly for the full duration
    freewheeling=True, blocks pushed as fast as possible -> correct, full length

sfizz's real-time ``renderBlock`` advances its sample readers against *actual
elapsed time*, so an offline render that outruns the clock starves every voice.
Freewheeling is therefore not an optimisation or a determinism knob -- it is
the only mode in which rendering may run faster than real time.

Hence ``freewheeling=True`` is the default here.  Pass ``False`` only when the
caller is genuinely paced by a real audio device (live preview), where the
wall-clock behaviour is exactly what you want.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pysfizz

__all__ = [
    "EngineCache",
    "EngineError",
    "EngineNotLoaded",
    "SfzEngine",
    "SFZ_PATH_ENV",
    "resolve_sfz_path",
    "find_base_sfz",
]

#: Environment variable holding an explicit instrument location (.sfz file or
#: a directory of .sfz files).  Checked before the repo-relative default.
SFZ_PATH_ENV = "SFZ_PATH"

#: Repo-relative fallback, kept in sync with .gitignore/AGENTS.md.  The entry
#: itself is a directory junction -- never hardcode its target.
_DEFAULT_SFZ_DIR = Path(".sfz_links") / "Concert_Grand__Reverb"

#: The base preset's stem ends here; every other candidate in that folder is a
#: named variant ("(Realistic)", "(2 seconds release)", ...) that has tweaks
#: baked in.  Match the stem, never the file size -- all four are 5626 B.
_BASE_STEM_SUFFIX = "concert grand"

_MIDI_MIN, _MIDI_MAX = 0, 127


class EngineError(RuntimeError):
    """Engine could not be constructed or driven."""


class EngineNotLoaded(EngineError):
    """An action was called before a successful ``load_sfz()``."""


def _midi_byte(value, what: str) -> int:
    """Coerce to a MIDI byte 0..127, rejecting NaN/float junk explicitly.

    ``isinstance(x, int)`` would let ``True`` through and would not catch a
    float that is merely integral, so validate by value instead.  ``bool`` is
    rejected outright -- ``note_on(60, some_flag)`` is always a bug, never an
    intent.  numpy integer scalars are accepted.
    """
    if isinstance(value, bool):
        raise ValueError(f"{what} must be an integer in 0..127, got bool {value!r}")
    try:
        ivalue = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        # OverflowError: int(np.inf) / int(float('inf')) raise it, and float
        # 'inf' is not caught by the range check that follows.
        raise ValueError(f"{what} must be an integer in 0..127, got {value!r}") from exc
    if ivalue != value or not _MIDI_MIN <= ivalue <= _MIDI_MAX:
        raise ValueError(f"{what} must be an integer in 0..127, got {value!r}")
    return ivalue


def _sample_count(value, what: str) -> int:
    """Coerce to a non-negative integer sample count (no 127 cap).

    Same shape as :func:`_midi_byte` -- ``bool`` is rejected, ``int()`` is
    guarded against ``OverflowError`` (inf), and the round-trip ``ivalue !=
    value`` check rejects floats with a fractional part instead of truncating.
    """
    if isinstance(value, bool):
        raise ValueError(f"{what} must be an integer >= 0, got bool {value!r}")
    try:
        ivalue = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{what} must be an integer >= 0, got {value!r}") from exc
    if ivalue != value or ivalue < 0:
        raise ValueError(f"{what} must be an integer >= 0, got {value!r}")
    return ivalue


def _repo_root() -> Optional[Path]:
    """Nearest ancestor (from cwd, then from this file) holding .sfz_links."""
    starts = [Path.cwd(), *Path(__file__).resolve().parents]
    for base in starts:
        try:
            if (base / _DEFAULT_SFZ_DIR.parent.name).is_dir():
                return base
        except OSError:
            continue
    return None


def find_base_sfz(directory: Path) -> Path:
    """Pick the unparenthesised base .sfz from ``directory``.

    Picks the shortest stem ending in "concert grand" -- that is the base
    preset; the parenthetical siblings are deliberately tweaked variants.
    """
    candidates = sorted(directory.glob("*.sfz"))
    if not candidates:
        raise EngineError(f"no .sfz file under {directory}")
    base = [c for c in candidates if c.stem.strip().lower().endswith(_BASE_STEM_SUFFIX)]
    pool = base or candidates
    # Shortest stem == the unparenthesised one when both exist.
    return min(pool, key=lambda p: (len(p.stem), p.name))


def resolve_sfz_path(path: Optional[os.PathLike] = None) -> Path:
    """Resolve an instrument location to a concrete .sfz file.

    Order: explicit argument > ``SFZ_PATH`` env > repo-relative default.
    An explicit file is honoured as-is; a directory goes through
    :func:`find_base_sfz`.

    An empty ``SFZ_PATH`` counts as *unset* (``"" or None``), matching the
    usual environment convention -- an exported-but-blank variable is a config
    mistake that silently hides itself otherwise.  An explicit ``path=""`` is
    a real path argument and raises, since only the env branch is ambiguous.
    """
    raw = path if path is not None else os.environ.get(SFZ_PATH_ENV) or None
    if raw is None:
        root = _repo_root()
        if root is None:
            raise EngineError(
                f"no instrument location: pass a path, set {SFZ_PATH_ENV}, "
                f"or run from a tree containing {_DEFAULT_SFZ_DIR.parent.name}/"
            )
        raw = root / _DEFAULT_SFZ_DIR
    candidate = Path(raw)
    if candidate.suffix.lower() == ".sfz":
        if not candidate.is_file():
            raise EngineError(f"sfz file not found: {candidate}")
        return candidate
    if not candidate.is_dir():
        raise EngineError(f"sfz location does not exist: {candidate}")
    return find_base_sfz(candidate)


class SfzEngine:
    """Block-rendering sfz engine with a deliberately small surface.

    Usage is always load-then-drive::

        eng = SfzEngine()
        eng.load_sfz()                      # SFZ_PATH or repo default
        eng.note_on(60, 80)
        block = eng.render_block()          # (2, block_size) float32
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        block_size: int = 1024,
        freewheeling: bool = True,
        *,
        sample_quality: Optional[int] = None,
        num_voices: Optional[int] = None,
    ) -> None:
        """Create the engine.

        ``freewheeling`` must stay ``True`` for any offline render; see the
        module docstring for the measurement behind that.  ``pysfizz.Synth``
        hardcodes ``enable_freewheeling()``, so both states are asserted
        rather than assumed.

        ``sample_quality`` (0..10) and ``num_voices`` are sfizz's two knobs that
        actually move the needle, measured on this machine: dropping quality from
        the default 10 to 6 made a dense passage render **2.2x** faster with the
        voice count unchanged, and the preview's dense passages genuinely need
        ~74 voices while the default cap is 64 -- sfizz was silently cutting
        notes off, which is audible as clicks and costs nothing to fix by
        raising the cap.  ``None`` keeps sfizz's own default; offline export
        leaves both alone because it can afford the maximum quality.
        """
        if sample_rate <= 0 or block_size <= 0:
            raise ValueError("sample_rate and block_size must be positive")
        if sample_quality is not None and not 0 <= int(sample_quality) <= 10:
            raise ValueError(f"sample_quality must be 0..10, got {sample_quality}")
        if num_voices is not None and int(num_voices) <= 0:
            raise ValueError(f"num_voices must be positive, got {num_voices}")
        self._sample_rate = int(sample_rate)
        self._block_size = int(block_size)
        self._sample_quality = None if sample_quality is None else int(sample_quality)
        self._num_voices = None if num_voices is None else int(num_voices)
        self._synth = pysfizz.Synth(sample_rate=sample_rate, block_size=block_size)
        if not freewheeling:
            self._synth._synth.disable_freewheeling()
        self._freewheeling = bool(self._synth._synth.is_freewheeling())
        if self._freewheeling != freewheeling:
            raise EngineError("could not set freewheeling state as requested")
        self._sfz_path: Optional[Path] = None
        self._playable_keys: tuple = ()
        self._bound: dict[str, Callable] = {}

    # ------------------------------------------------------------------ #
    # the six actions
    # ------------------------------------------------------------------ #

    def load_sfz(self, path: Optional[os.PathLike] = None) -> Path:
        """Load an instrument; returns the resolved .sfz actually loaded."""
        sfz = resolve_sfz_path(path)
        if not self._synth.load_sfz_file(str(sfz)):
            raise EngineError(f"pysfizz refused to load {sfz} (0 regions)")
        self._sfz_path = sfz
        self._playable_keys = tuple(self._synth.playable_keys)
        self._bound = {}
        # After load: pysfizz's Synth constructor creates the C++ object, and
        # loading a file may reset these, so they are applied last.
        if self._num_voices is not None:
            self._synth._synth.set_num_voices(self._num_voices)
        if self._sample_quality is not None:
            self._synth._synth.set_sample_quality(self._sample_quality)
        return sfz

    def note_on(self, pitch: int, vel: int) -> None:
        """Start a voice. ``vel`` is the raw MIDI attack velocity."""
        pitch, vel = _midi_byte(pitch, "pitch"), _midi_byte(vel, "vel")
        self._live("note_on")(0, pitch, vel)

    def note_off(self, pitch: int, vel: int = 0, delay_samples: int = 0) -> None:
        """Release a voice; ``delay_samples`` schedules it inside the next block.

        ``vel`` is the release velocity.  Defaults keep the common
        ``note_off(pitch)`` call working; the extra kwargs exist because
        block-accurate release is what a velocity-segment editor needs.

        ``delay_samples`` is a sample count, not a MIDI byte, so it is only
        range-checked for integrality and ``>= 0`` -- validated the same way
        ``_midi_byte`` validates, so a stray ``None``/``str``/float/``inf``
        raises the module's own ``ValueError`` instead of a native ``TypeError``
        or a silent truncation of ``3.7`` to ``3``.
        """
        delay = _sample_count(delay_samples, "delay_samples")
        pitch, vel = _midi_byte(pitch, "pitch"), _midi_byte(vel, "vel")
        self._live("note_off")(delay, pitch, vel)

    def cc(self, num: int, val: int) -> None:
        """Controller change; ``num`` 64 is the sustain pedal in sfz."""
        num, val = _midi_byte(num, "cc number"), _midi_byte(val, "cc value")
        self._live("cc")(0, num, val)

    def render_block(self) -> np.ndarray:
        """Render one block -> ``(2, block_size)`` float32, channels first.

        Shape matches pysfizz's own ``render_note`` so downstream code has a
        single layout convention.
        """
        left, right = self._live("render_block")()
        return np.stack(
            [np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32)]
        )

    def all_sound_off(self) -> None:
        """Panic: kill every voice immediately."""
        self._live("all_sound_off")()

    # ------------------------------------------------------------------ #
    # read-only extras (not actions -- pure observation)
    # ------------------------------------------------------------------ #

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def is_freewheeling(self) -> bool:
        """False here means every voice dies ~0.21 s into a fast offline render."""
        return self._freewheeling

    @property
    def sfz_path(self) -> Optional[Path]:
        return self._sfz_path

    @property
    def playable_keys(self) -> tuple:
        return self._playable_keys

    # ------------------------------------------------------------------ #

    def _live(self, action: str) -> Callable:
        """Resolve an action, preferring the public wrapper over ``_synth``."""
        fn = self._bound.get(action)
        if fn is not None:
            return fn
        if self._sfz_path is None:
            raise EngineNotLoaded(f"{action}() before a successful load_sfz()")
        for holder in (self._synth, getattr(self._synth, "_synth", None)):
            fn = getattr(holder, action, None)
            if callable(fn):
                self._bound[action] = fn
                return fn
        raise EngineError(
            f"pysfizz {getattr(pysfizz, '__version__', '?')} exposes no {action}(); "
            "the adapter needs updating for this release"
        )


class EngineCache:
    """Keep one loaded instrument alive between playbacks.

    Every press of play used to build a fresh :class:`SfzEngine` and call
    :meth:`SfzEngine.load_sfz`, re-parsing the SFZ and re-mapping its samples:
    measured on this machine with the Concert Grand, 51 ms warm and 450 ms when
    the samples are not yet in the OS cache -- the largest single startup cost
    after the audio device buffer.

    Reuse is safe rather than clever.  ``offline.BlockStreamer.seek`` begins with
    ``all_sound_off()`` and then restores the pedal, the state controllers and
    every still-sounding note at its original attack velocity, so by the time the
    first block renders, a reused engine is in the state a freshly loaded one
    would be in.

    The pool hands out *ownership*, not a shared object: :meth:`acquire` removes
    an engine and the caller keeps it until :meth:`release`.  That is what keeps
    two renderer threads from driving one engine at once -- if the previous
    worker has not exited yet, its engine is not in the pool, so the next
    playback builds a new one instead of colliding with it.
    """

    def __init__(self, capacity: int = 1) -> None:
        self._lock = threading.Lock()
        self._free: list[tuple[tuple, SfzEngine]] = []
        self._capacity = max(1, int(capacity))

    def acquire(self, key: tuple, build: Callable[[], SfzEngine]) -> SfzEngine:
        """An engine for ``key``; ``build()`` runs only on a miss."""
        with self._lock:
            for index in range(len(self._free) - 1, -1, -1):
                free_key, engine = self._free[index]
                if free_key == key:
                    del self._free[index]
                    return engine
        return build()

    def release(self, key: tuple, engine: SfzEngine) -> None:
        """Return a finished engine; past ``capacity`` the oldest one is dropped."""
        with self._lock:
            self._free.append((key, engine))
            while len(self._free) > self._capacity:
                self._free.pop(0)

    @property
    def idle(self) -> int:
        """Engines currently in the pool (tests and diagnostics)."""
        with self._lock:
            return len(self._free)