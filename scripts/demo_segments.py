"""Headless sanity check for the auto velocity-segment cut.

Run::

    python scripts/demo_segments.py

Prints per-track segment counts and the first segments of the project so the
cut can be judged by eye: a good cut has few, long, musically meaningful
segments; a bad one has one segment per note.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qmidi.segments import (  # noqa: E402
    AutoSegmentOptions,
    SegmentProject,
    build_tracks,
)

MIDI = ROOT / "example" / "midi" / "Unchained.mid"
if not MIDI.is_file():
    sys.exit(
        "example/midi/Unchained.mid not found -- the example folder is not part "
        "of the public distribution; edit MIDI in this script to point at your "
        "own .mid file."
    )
TPB = 480


def _vels(track, seg) -> str:
    note = {n.index: n for n in track.notes}
    values = sorted({note[i].velocity for i in seg.note_indices if i in note})
    return ",".join(str(v) for v in values) or "-"


def main(argv: list[str]) -> int:
    # Reason labels are Chinese; a GBK console would otherwise mangle them.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:  # pragma: no cover - very old interpreters
        pass
    limit = int(argv[1]) if len(argv) > 1 else 15
    opts = AutoSegmentOptions()
    tracks = build_tracks(MIDI, ticks_per_beat=TPB, options=opts)

    print(f"file    : {MIDI.name}  ({TPB} ticks/beat)")
    print(f"options : tolerance={opts.velocity_tolerance} "
          f"period_beats={opts.period_beats} "
          f"max_cycle_len={opts.max_cycle_len} min_cycles={opts.min_cycles}")
    print()
    print(f"{'track':>5} {'notes':>6} {'segments':>8} {'notes/seg':>9} {'cut ticks/beat':>16}")
    for t in tracks:
        notes = len(t.notes)
        cuts = [s.start_tick for s in t.segments[1:]]
        beats = ", ".join(f"{c / TPB:g}" for c in cuts[:6])
        more = " ..." if len(cuts) > 6 else ""
        ratio = notes / len(t.segments) if t.segments else 0
        print(f"{t.track:>5} {notes:>6} {len(t.segments):>8} {ratio:>9.1f} {beats}{more}")

    project = SegmentProject(source=MIDI, ticks_per_beat=TPB, tracks=tracks)
    for t in tracks:
        t.validate()
    print(f"\ntotal segments: {project.segment_count}  (validate() passed)")

    print(f"\nfirst {limit} segments:")
    print(f"{'#':>3} {'trk':>3} {'start':>8} {'end':>8} {'beats':>13} {'n':>4} "
          f"{'orig vel':>12}  reason")
    flat = [(i, t, s) for t in tracks for i, s in enumerate(t.segments)]
    for n, (i, t, s) in enumerate(flat[:limit]):
        span = f"{s.start_tick / TPB:g}-{s.end_tick / TPB:g}"
        print(f"{n:>3} {s.track:>3} {s.start_tick:>8} {s.end_tick:>8} {span:>13} "
              f"{len(s.note_indices):>4} {_vels(t, s):>12}  {s.label}")

    periodic = [(t.track, i, s) for t in tracks for i, s in enumerate(t.segments)
                if s.is_periodic]
    print(f"\nperiodic segments: {len(periodic)}")
    for trk, i, s in periodic[:8]:
        print(f"  track {trk} seg {i}: {s.label} "
              f"beats {s.start_tick / TPB:g}-{s.end_tick / TPB:g} "
              f"notes={len(s.note_indices)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))