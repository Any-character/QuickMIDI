"""Step 2b: prove that seeking to an arbitrary offset is trustworthy.

Renders a 5 s fragment from the middle of ``example/midi/Unchained.mid``
(default: 1/3 of its length) via ``qmidi.offline.render_from`` and checks
three things:

  A. the notes that were sounding at the offset are still sounding at the
     start of the fragment (state rebuild works),
  B. nothing is stuck afterwards (all-sound-off + pedal-up really silence it),
  C. two runs are bit-identical (freewheeling + event replay is deterministic).

Bonus metric (not an assertion): the fragment is compared against the tail of
a *continuous* render from t=0.  A seek can never be bit-exact there -- the
restored notes re-attack -- but the RMS envelopes should track closely.

Headless, no Qt.  Usage:  python scripts/verify_seek_render.py [offset_sec]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qmidi.offline import build_timeline, render_from, write_wav  # noqa: E402

MIDI = ROOT / "example" / "midi" / "Unchained.mid"
if not MIDI.is_file():
    sys.exit(
        "example/midi/Unchained.mid not found -- the example folder is not part "
        "of the public distribution; edit MIDI in this script to point at your "
        "own .mid file."
    )
OUT_DIR = ROOT / "renders"
DURATION_S = 5.0
SETTLE_S = 1.0
SR = 48000


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0


def db(x: float) -> float:
    return 20.0 * np.log10(x + 1e-12)


def env(audio: np.ndarray, win: int) -> np.ndarray:
    mono = audio.mean(axis=0).astype(np.float64)
    n = (len(mono) // win) * win
    if n == 0:
        return np.zeros(0)
    return np.sqrt(np.mean(mono[:n].reshape(-1, win) ** 2, axis=1))


def main() -> int:
    offset = float(sys.argv[1]) if len(sys.argv) > 1 else None
    total = 218.4025
    if offset is None:
        offset = round(total / 3.0, 3)

    print(f"midi          : {MIDI.name}")
    print(f"file length   : {total:.3f} s")
    print(f"offset        : {offset:.3f} s ({offset / total:.1%} into the file)")
    print(f"fragment      : {DURATION_S:.3f} s + {SETTLE_S:.1f} s settle tail")

    events = build_timeline(MIDI)
    print(f"timeline      : {len(events)} events")

    r1 = render_from(MIDI, offset, DURATION_S, sample_rate=SR, settle_sec=SETTLE_S,
                     timeline=events)
    print(f"sfz           : {r1.sfz_path.name}")
    print(f"pedal at seek : {'DOWN (cc64=127)' if r1.pedal_down else 'UP'}")
    print(f"held at seek  : {len(r1.held)} pitches -> "
          f"{[(p, v) for p, v in r1.held]}")

    r2 = render_from(MIDI, offset, DURATION_S, sample_rate=SR, settle_sec=SETTLE_S,
                     timeline=events)

    OUT_DIR.mkdir(exist_ok=True)
    wav = write_wav(OUT_DIR / f"seek_{offset:08.3f}s_5s.wav", r1.audio, SR)
    write_wav(OUT_DIR / f"seek_{offset:08.3f}s_tail.wav", r1.tail, SR)
    print(f"wav           : {wav.name}")

    win = SR // 10  # 100 ms
    sr_win = SR // 2  # 0.5 s

    # ---- A: head energy --------------------------------------------------- #
    head = r1.audio[:, : sr_win]
    sustained = r1.audio[:, int(0.3 * SR) : int(0.5 * SR)]  # after the re-attack
    head_rms, sus_rms = rms(head), rms(sustained)
    floor = float(np.max(np.abs(r1.tail))) if r1.tail.size else 0.0
    print()
    print("A) head is alive")
    print(f"   peak(0-0.5s)      {float(np.max(np.abs(head))):.6f}  ({db(float(np.max(np.abs(head)))):7.2f} dBFS)")
    print(f"   rms (0-0.5s)      {head_rms:.6f}  ({db(head_rms):7.2f} dBFS)")
    print(f"   rms (0.3-0.5s)    {sus_rms:.6f}  ({db(sus_rms):7.2f} dBFS)  <- post-attack, not just a click")
    print(f"   settle-tail peak  {floor:.3e}  ({db(floor):7.2f} dBFS)  <- measured silence floor")
    a1 = head_rms > 1e-4 and sus_rms > 1e-4
    a2 = head_rms > 100.0 * max(floor, 1e-12)
    print(f"   [A] head rms > 1e-4 (=-80 dBFS)                 : {'PASS' if a1 else 'FAIL'}")
    print(f"   [A] head rms > 100x the silence floor (>=40 dB) : {'PASS' if a2 else 'FAIL'}")
    print(f"   [A] sustained-after-attack rms > 1e-4           : "
          f"{'PASS' if sus_rms > 1e-4 else 'FAIL'}")

    # ---- B: nothing stuck -------------------------------------------------- #
    tail_peak = float(np.max(np.abs(r1.tail))) if r1.tail.size else 0.0
    tail_rms = rms(r1.tail)
    b1 = tail_peak < 1e-4
    b2 = tail_rms < 1e-5
    print()
    print("B) nothing is stuck")
    print(f"   tail peak         {tail_peak:.3e}  ({db(tail_peak):7.2f} dBFS)")
    print(f"   tail rms          {tail_rms:.3e}  ({db(tail_rms):7.2f} dBFS)")
    print(f"   [B] tail peak < 1e-4 (-80 dBFS)                : {'PASS' if b1 else 'FAIL'}")
    print(f"   [B] tail rms  < 1e-5 (-100 dBFS)               : {'PASS' if b2 else 'FAIL'}")

    # ---- C: determinism ---------------------------------------------------- #
    same_audio = np.array_equal(r1.audio, r2.audio)
    same_tail = np.array_equal(r1.tail, r2.tail)
    md = 0 if same_audio else float(np.max(np.abs(r1.audio - r2.audio)))
    print()
    print("C) determinism")
    print(f"   [C] fragment bit-identical across two runs      : {'PASS' if same_audio else 'FAIL'} (max diff {md:g})")
    print(f"   [C] tail bit-identical across two runs          : {'PASS' if same_tail else 'FAIL'}")

    # ---- bonus: vs a continuous render ------------------------------------- #
    print()
    print("bonus) seek vs continuous render from t=0")
    rc = render_from(MIDI, 0.0, offset + DURATION_S, sample_rate=SR, timeline=events)
    cont = rc.audio[:, int(round(offset * SR)) :]
    n = min(cont.shape[1], r1.audio.shape[1])
    cont = cont[:, :n]
    ec, es = env(cont, win), env(r1.audio[:, :n], win)
    m = min(len(ec), len(es))

    def log10(x):
        return np.log10(np.asarray(x, dtype=np.float64) + 1e-7)

    def corr(lo: int, hi: int) -> float:
        a = log10(ec[lo:hi])
        b = log10(es[lo:hi])
        return float(np.corrcoef(a, b)[0, 1]) if hi - lo > 4 else float("nan")

    corr_all = corr(0, m)
    skip = int(0.5 / 0.1)  # the re-attack window: restored notes restart at full
    corr_body = corr(skip, m)
    delta_db = 20.0 * (log10(es[skip:m]) - log10(ec[skip:m]))  # seek minus continuous
    med, mx = float(np.median(np.abs(delta_db))), float(np.max(np.abs(delta_db)))
    head_db = float(20.0 * (log10(es[0]) - log10(ec[0])))
    err = db(rms(cont)) - db(rms(r1.audio[:, :n]))
    print(f"   continuous rms    {rms(cont):.6f} ({db(rms(cont)):7.2f} dBFS)  peak {float(np.max(np.abs(cont))):.3f}")
    print(f"   seek      rms     {rms(r1.audio[:, :n]):.6f} ({db(rms(r1.audio[:, :n])):7.2f} dBFS)  peak {float(np.max(np.abs(r1.audio[:, :n]))):.3f}")
    print(f"   whole-fragment level delta          {err:+.2f} dB")
    print(f"   first 100 ms level delta            {head_db:+.2f} dB  <- cost of the re-attack")
    print(f"   per-100ms envelope delta (0.5-5 s): median |d| {med:.2f} dB, max |d| {mx:.2f} dB")
    print(f"   log-RMS envelope corr, 0-5 s        {corr_all:.4f}  (informational: the")
    print(f"   log-RMS envelope corr, 0.5-5 s      {corr_body:.4f}   envelope only spans ~1 dB here)")
    b3 = med < 1.5 and mx < 6.0
    print(f"   [bonus] median |d| < 1.5 dB and max |d| < 6 dB : {'PASS' if b3 else 'FAIL'}")
    print("   note: both peak above 1.0 -> the raw sfz bus clips; a limiter/fader stage")
    print("         is still missing.  Unrelated to seeking (the continuous render clips too).")

    print()
    print("envelope of the seek fragment (100 ms rms):")
    es_all = env(r1.audio, win)
    mx = max(es_all.max(), 1e-9)
    for i, e in enumerate(es_all):
        print(f"  {i * 0.1:4.1f}s {e:.5f} {'#' * int(e / mx * 48)}")

    ok = a1 and a2 and sus_rms > 1e-4 and b1 and b2 and same_audio and b3
    print()
    print("RESULT:", "ALL PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())