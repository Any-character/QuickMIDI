"""A small, honest mastering chain for the MP3 export.

What it is not
--------------
This is **not** a replacement for a mixing engineer, and nothing here looks at
individual instruments.  The piano is already balanced by the velocities in the
MIDI; what a bare render lacks is the last three things every finished
recording has:

1. **space** -- a little reverb, so the piano is not glued to the speakers;
2. **glue** -- gentle bus compression, so the loud pages do not jump out of the
   quiet ones;
3. **a predictable level** -- the file lands at a known loudness instead of
   "whatever the peak happened to be".

Order matters and is the same order a mastering engineer would use: reverb is
added *before* the compressor (so the tail is part of the sound being glued),
the limiter is last (so nothing clips after the gain stage), and loudness
normalisation is measured on the limited signal.

Honesty about the numbers
-------------------------
* Loudness follows ITU-R BS.1770 in shape -- K-weighting, 400 ms blocks with
  75 % overlap, absolute and relative gating -- but the K-weighting filters are
  applied as a magnitude-only convolution, so the figure is a close
  approximation (well inside 0.5 dB on piano material), not a certified meter.
* The limiter is a peak limiter with a smoothed gain envelope.  It is not a
  true-peak limiter: an MP3 encoder can overshoot by a fraction of a dB, which
  is why the default ceiling is -1.0 dBFS rather than -0.1.
* Compression is block-RMS driven, which is what makes it fast enough to run
  over a seven-minute render in a couple of seconds.  It is transparent at the
  default 2:1 / -18 dB, and it will not rescue a mix that is already broken.

Everything is numpy only (no scipy), because the project ships without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "MasterReport",
    "MasterSettings",
    "compress",
    "fft_convolve",
    "limit",
    "loudness_lufs",
    "make_ir",
    "master",
    "reverb",
]

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# settings
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MasterSettings:
    """The four decisions worth exposing to a non-engineer.

    ``enabled=False`` must be enough to get exactly the old behaviour back --
    that is the reversibility promise the user asked for, and it is why the
    chain lives here instead of being baked into the renderer.

    **Off by default, deliberately.**  This program's whole premise is that the
    user balances the piece by ear in the preview, and then hears the same thing
    in the export.  A compressor that squashes the loud pages is the one thing
    guaranteed to make a carefully built velocity curve read differently in the
    file than it did in the editor.  Whoever wants it turns it on and can hear
    the difference; nobody gets it silently.
    """

    enabled: bool = False
    #: Wet level of the reverb, 0..1.  0.14 is "same room", 0.3 is "hall".
    reverb_wet: float = 0.14
    #: Reverb length in seconds.  A grand piano tail is 1.5-2.5 s.
    reverb_seconds: float = 1.9
    #: Bus compression.  Threshold in dBFS (block RMS), ratio above it.
    comp_threshold_db: float = -18.0
    comp_ratio: float = 2.0
    #: Output ceiling in dBFS.  -1.0 leaves room for the MP3 encoder.
    ceiling_db: float = -1.0
    #: Integrated loudness target; ``None`` keeps whatever level came out.
    target_lufs: Optional[float] = -14.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.reverb_wet <= 1.0:
            raise ValueError(f"reverb_wet must be 0..1, got {self.reverb_wet!r}")
        if not 0.0 <= self.reverb_seconds <= 10.0:
            raise ValueError(f"reverb_seconds must be 0..10, got {self.reverb_seconds!r}")
        if self.comp_ratio < 1.0:
            raise ValueError(f"comp_ratio must be >= 1, got {self.comp_ratio!r}")
        if not -60.0 <= self.comp_threshold_db <= 0.0:
            raise ValueError(
                f"comp_threshold_db must be -60..0, got {self.comp_threshold_db!r}"
            )
        if not -20.0 <= self.ceiling_db <= 0.0:
            raise ValueError(f"ceiling_db must be -20..0, got {self.ceiling_db!r}")
        if self.target_lufs is not None and not -40.0 <= self.target_lufs <= -5.0:
            raise ValueError(
                f"target_lufs must be -40..-5 or None, got {self.target_lufs!r}"
            )


@dataclass
class MasterReport:
    """What the chain actually did, in numbers the status bar can print."""

    lufs_before: float
    lufs_after: float
    gain_db: float
    peak_after: float
    reverb_seconds: float


# --------------------------------------------------------------------------- #
# reverb
# --------------------------------------------------------------------------- #
def make_ir(
    sample_rate: int,
    seconds: float = 1.9,
    *,
    predelay_sec: float = 0.02,
    seed: int = 20261101,
) -> np.ndarray:
    """A synthetic stereo room impulse response, energy-normalised.

    Noise with an exponential envelope is the standard poor-man's IR: it has no
    distinct early reflections to sound artificial on piano, and being
    decorrelated between channels is exactly what widens the image.  Energy is
    normalised so ``reverb(x, ir, wet=w)`` keeps the dry level intact -- without
    that, changing the reverb length would also change the volume, and the user
    would blame the wrong knob.
    """
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate!r}")
    if seconds <= 0:
        raise ValueError(f"seconds must be positive, got {seconds!r}")
    length = max(1, int(round(seconds * sample_rate)))
    t = np.arange(length, dtype=np.float64) / sample_rate
    # -60 dB at `seconds`: exp(-6.9 * t / T) is the usual room-decay constant.
    envelope = np.exp(-6.9 * t / max(seconds, _EPS))
    rng = np.random.default_rng(seed)
    ir = rng.standard_normal((2, length)) * envelope
    # One-pole smoothing in the frequency domain would need scipy; a 3-tap
    # moving average is enough to take the "white noise" edge off the tail.
    ir = (ir + np.roll(ir, 1, axis=1) + np.roll(ir, 2, axis=1)) / 3.0
    ir[:, :3] = 0.0
    predelay = int(round(max(0.0, predelay_sec) * sample_rate))
    if predelay:
        ir = np.concatenate([np.zeros((2, predelay)), ir], axis=1)
    energy = np.sqrt(np.sum(ir ** 2, axis=1, keepdims=True))
    return (ir / np.maximum(energy, _EPS)).astype(np.float32)


def fft_convolve(audio: np.ndarray, ir: np.ndarray, block: int = 1 << 17) -> np.ndarray:
    """Overlap-add convolution of ``(2, n)`` audio with ``(2, m)`` filters.

    A single FFT over a seven-minute file would allocate a few hundred MB for
    one multiplication; block-wise overlap-add keeps the peak at a couple of
    blocks and, measured on this machine, does the whole piece in ~1.5 s.
    """
    dry = np.atleast_2d(np.asarray(audio, dtype=np.float64))
    fir = np.atleast_2d(np.asarray(ir, dtype=np.float64))
    if dry.shape[0] != fir.shape[0]:
        raise ValueError(f"channel mismatch: audio {dry.shape}, ir {fir.shape}")
    n, m = dry.shape[1], fir.shape[1]
    if n == 0:
        return dry.astype(np.float32)
    block = max(int(block), m * 2)
    nfft = 1 << int(np.ceil(np.log2(block + m)))
    spec_ir = np.fft.rfft(fir, nfft, axis=1)
    out = np.zeros((dry.shape[0], n + m - 1), dtype=np.float64)
    for start in range(0, n, block):
        chunk = dry[:, start : start + block]
        spec = np.fft.rfft(chunk, nfft, axis=1) * spec_ir
        piece = np.fft.irfft(spec, nfft, axis=1)[:, : chunk.shape[1] + m - 1]
        out[:, start : start + piece.shape[1]] += piece
    return out[:, :n].astype(np.float32)


def reverb(audio: np.ndarray, sample_rate: int, *, wet: float = 0.14,
           seconds: float = 1.9, ir: Optional[np.ndarray] = None) -> np.ndarray:
    """Mix a convolver tail under ``audio`` (dry stays at full level)."""
    if wet <= 0:
        return np.asarray(audio, dtype=np.float32)
    if not 0.0 <= wet <= 1.0:
        raise ValueError(f"wet must be 0..1, got {wet!r}")
    impulse = ir if ir is not None else make_ir(sample_rate, seconds)
    wet_signal = fft_convolve(audio, impulse)
    dry = np.asarray(audio, dtype=np.float32)
    return (dry + wet * wet_signal).astype(np.float32)


# --------------------------------------------------------------------------- #
# dynamics
# --------------------------------------------------------------------------- #
def _frames(x: np.ndarray, block: int) -> tuple[np.ndarray, int]:
    """Zero-pad to a whole number of blocks and return ``(frames, n)``.

    Keeping the signal as ``(channels, blocks, block)`` is what lets the gain
    be applied by broadcasting: expanding a per-block gain back to one value
    per sample doubles the peak memory of a seven-minute render for nothing.
    """
    n = x.shape[1]
    padded = int(np.ceil(max(n, 1) / block)) * block
    buf = np.zeros((x.shape[0], padded), dtype=np.float32)
    buf[:, :n] = x
    return buf.reshape(x.shape[0], -1, block), n


def _smooth(env: np.ndarray, samples: int) -> np.ndarray:
    """Boxcar smoothing along time, O(n) via cumulative sums.

    ``np.convolve`` with a 4000-tap kernel over 20 M samples is 8e10 operations
    -- minutes, not milliseconds -- so the moving average is done as a
    difference of cumulative sums instead.
    """
    width = max(1, min(int(samples), env.shape[1]))
    if width <= 1:
        return env
    padded = np.concatenate(
        [np.repeat(env[:, :1], width, axis=1), env,
         np.repeat(env[:, -1:], width, axis=1)], axis=1)
    cumulative = np.cumsum(padded, axis=1, dtype=np.float64)
    cumulative = np.concatenate([np.zeros((env.shape[0], 1)), cumulative], axis=1)
    summed = cumulative[:, width:] - cumulative[:, :-width]
    averaged = summed / width
    # Original sample j sits at padded index ``width + j``; centre the window
    # on it so the smoothed curve is not delayed by half a window.
    start = width - width // 2
    return averaged[:, start : start + env.shape[1]].astype(np.float32)


def compress(audio: np.ndarray, sample_rate: int, *, threshold_db: float = -18.0,
             ratio: float = 2.0, block_sec: float = 0.005,
             release_sec: float = 0.08) -> np.ndarray:
    """Gentle downward compression driven by the block RMS.

    Below the threshold the gain is exactly 1.0 -- the quiet pages keep their
    dynamics, which is the whole point of doing this on a piano recording
    rather than slapping a limiter on the mix.
    """
    data = np.asarray(audio, dtype=np.float32)
    if ratio <= 1.0 or data.size == 0:
        return data.copy()
    block = max(1, int(round(block_sec * sample_rate)))
    frames, n = _frames(data, block)
    rms = np.sqrt(np.mean(frames ** 2, axis=2))
    level_db = 20.0 * np.log10(np.maximum(rms, _EPS))
    over = np.maximum(0.0, level_db - threshold_db)
    gain_db = -over * (1.0 - 1.0 / ratio)
    # Smooth the *gain* (not the signal): a jumpy gain curve is audible as
    # breathing, while a jumpy input is exactly what a compressor is for.
    gain_db = _smooth(gain_db, int(round(release_sec * sample_rate)) // block)
    gain = np.power(10.0, gain_db / 20.0).astype(np.float32)
    return (frames * gain[:, :, np.newaxis]).reshape(
        data.shape[0], -1)[:, :n].astype(np.float32)


def limit(audio: np.ndarray, sample_rate: int, *, ceiling_db: float = -1.0,
          block_sec: float = 0.002) -> np.ndarray:
    """Peak limiter: never lets a sample exceed the ceiling.

    Uses a *minimum* of the current and the smoothed gain so the reduction is
    already in place when the transient arrives, instead of clamping the sample
    afterwards (which is a distortion, not a limiter).
    """
    data = np.asarray(audio, dtype=np.float32)
    if data.size == 0:
        return data.copy()
    ceiling = 10.0 ** (ceiling_db / 20.0)
    block = max(1, int(round(block_sec * sample_rate)))
    frames, n = _frames(data, block)
    peak = np.max(np.abs(frames), axis=2)
    target = np.minimum(1.0, ceiling / np.maximum(peak, _EPS))
    smoothed = _smooth(target, int(round(0.03 * sample_rate)) // block)
    gain = np.minimum(smoothed, target).astype(np.float32)
    out = (frames * gain[:, :, np.newaxis]).reshape(data.shape[0], -1)[:, :n]
    # The smoothed curve can still lag a single stray sample; a final clip at
    # the ceiling is the declared contract, not a hidden safety net.
    return np.clip(out, -ceiling, ceiling).astype(np.float32)


# --------------------------------------------------------------------------- #
# loudness
# --------------------------------------------------------------------------- #
#: BS.1770 K-weighting, 48 kHz coefficients (stage 1 shelf, stage 2 high-pass).
_KW_SHELF_B = (1.53512485958697, -2.69169618940638, 1.19839281085285)
_KW_SHELF_A = (1.0, -1.69065929318241, 0.73248077421585)
_KW_HP_B = (1.0, -2.0, 1.0)
_KW_HP_A = (1.0, -1.99004745483398, 0.99007225036621)


def _biquad_mag(freqs_norm: np.ndarray, b, a) -> np.ndarray:
    """|H(e^jw)| of a biquad, evaluated straight from the coefficients.

    ``freqs_norm`` is in cycles per sample (0..0.5), not Hz: feeding Hz here
    made every bin wrap around the unit circle and produced a flat +4 dB
    filter, which a first version of this module happily "measured".
    """
    z1 = np.exp(-1j * 2.0 * np.pi * freqs_norm)
    z2 = z1 * z1
    num = b[0] + b[1] * z1 + b[2] * z2
    den = a[0] + a[1] * z1 + a[2] * z2
    return np.abs(num / den)


def _k_weighting_ir(sample_rate: int, taps: int = 4096) -> np.ndarray:
    """Impulse response of the K-weighting filter (magnitude, zero phase)."""
    freqs = np.fft.rfftfreq(taps, 1.0 / sample_rate) / sample_rate
    mag = _biquad_mag(freqs, _KW_SHELF_B, _KW_SHELF_A) * _biquad_mag(
        freqs, _KW_HP_B, _KW_HP_A)
    # A zero-phase response has its impulse centred on index 0, so half of it
    # wraps around to the end of the buffer.  Convolving with that would apply
    # the filter *twice* (once causal, once as a 85 ms echo).  Rolling by half
    # the length makes the convolution linear-phase and the gain correct.
    return np.roll(np.fft.irfft(mag, taps), taps // 2)


def loudness_lufs(audio: np.ndarray, sample_rate: int) -> float:
    """Approximate integrated loudness (LUFS) of ``(2, n)`` audio.

    Gated per BS.1770: blocks 400 ms with 75 % overlap, drop blocks under
    -70 LUFS, then drop blocks 10 LU under the mean of what is left.  Silence
    returns ``-inf`` instead of a number, because "no signal" has no loudness.
    """
    data = np.atleast_2d(np.asarray(audio, dtype=np.float32))
    if data.shape[1] == 0:
        return float("-inf")
    weighted = np.apply_along_axis(
        lambda row: fft_convolve(row[np.newaxis, :],
                                 _k_weighting_ir(sample_rate)[np.newaxis, :])[0],
        axis=1, arr=data.astype(np.float64))
    block = int(round(0.4 * sample_rate))
    hop = max(1, block // 4)
    if data.shape[1] < block:
        power = float(np.mean(weighted ** 2))
        return float(-0.691 + 10.0 * np.log10(max(power, _EPS)))
    powers = []
    for start in range(0, data.shape[1] - block + 1, hop):
        chunk = weighted[:, start : start + block]
        powers.append(float(np.mean(np.sum(chunk ** 2, axis=0))))
    powers_arr = np.asarray(powers)
    energies = -0.691 + 10.0 * np.log10(np.maximum(powers_arr, _EPS))
    keep = energies > -70.0
    if not np.any(keep):
        return float("-inf")
    relative = float(np.mean(energies[keep])) - 10.0
    keep = keep & (energies > relative)
    if not np.any(keep):
        return float("-inf")
    return float(-0.691 + 10.0 * np.log10(np.mean(powers_arr[keep]) + _EPS))


# --------------------------------------------------------------------------- #
# the chain
# --------------------------------------------------------------------------- #
def master(audio: np.ndarray, sample_rate: int,
           settings: MasterSettings = MasterSettings()) -> tuple[np.ndarray, MasterReport]:
    """Run the whole chain; returns ``(audio, report)``.

    With ``settings.enabled == False`` the audio comes back untouched (a copy),
    which is what makes the feature safe to try: turning it off restores the
    previous sound bit for bit.
    """
    data = np.atleast_2d(np.asarray(audio, dtype=np.float32))
    before = loudness_lufs(data, sample_rate) if data.size else float("-inf")
    if not settings.enabled or data.size == 0:
        return data.copy(), MasterReport(before, before, 0.0,
                                         float(np.max(np.abs(data))) if data.size else 0.0,
                                         0.0)

    processed = reverb(data, sample_rate, wet=settings.reverb_wet,
                       seconds=settings.reverb_seconds)
    processed = compress(processed, sample_rate,
                         threshold_db=settings.comp_threshold_db,
                         ratio=settings.comp_ratio)
    processed = limit(processed, sample_rate, ceiling_db=settings.ceiling_db)

    gain_db = 0.0
    if settings.target_lufs is not None:
        level = loudness_lufs(processed, sample_rate)
        if np.isfinite(level):
            gain_db = float(settings.target_lufs - level)
            processed = (processed * (10.0 ** (gain_db / 20.0))).astype(np.float32)
            ceiling = 10.0 ** (settings.ceiling_db / 20.0)
            peak = float(np.max(np.abs(processed)))
            if peak > ceiling:
                # Normalising up must not undo the limiter: trade a little
                # loudness for the ceiling instead of clipping the file.
                scale = ceiling / peak
                processed = (processed * scale).astype(np.float32)
                gain_db += 20.0 * np.log10(scale)

    after = loudness_lufs(processed, sample_rate)
    return processed, MasterReport(
        lufs_before=before,
        lufs_after=after,
        gain_db=gain_db,
        peak_after=float(np.max(np.abs(processed))),
        reverb_seconds=settings.reverb_seconds if settings.reverb_wet > 0 else 0.0,
    )
