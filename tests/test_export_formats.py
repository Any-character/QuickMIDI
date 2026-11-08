"""Audio format plumbing: path composition, depth validation, container writing.

These tests need neither an instrument nor ``example/`` -- they cover the parts
of the export dialog's contract that must hold on a fresh clone: a typed suffix
never stacks, a depth the container cannot store is refused instead of silently
substituted, and WAV/FLAC come back with the subtype the user picked.
"""
from __future__ import annotations

import sys
from pathlib import Path

import mido
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qmidi import exporters  # noqa: E402
from qmidi.exporters import (  # noqa: E402
    AUDIO_FORMATS,
    DEFAULT_SAMPLE_RATE,
    ExportError,
    audio_format,
    compose_export_path,
    default_audio_output,
    export_audio,
    write_lossless,
)

pytest.importorskip("soundfile")


def _tiny_midi(path: Path) -> Path:
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    track.append(mido.Message("note_on", note=60, velocity=64, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, time=480))
    midi.tracks.append(track)
    midi.save(str(path))
    return path


# -- the format table -------------------------------------------------------- #
def test_every_offered_format_can_write_its_default_depth():
    for info in AUDIO_FORMATS:
        if info.lossless:
            assert info.default_bit_depth in info.bit_depths
        else:
            assert info.bit_depths == ()
            assert info.default_bit_depth == 0


def test_mp3_does_not_advertise_rates_lame_cannot_encode():
    assert 96000 not in audio_format("mp3").sample_rates
    assert DEFAULT_SAMPLE_RATE in audio_format("mp3").sample_rates


def test_an_unknown_format_is_rejected():
    with pytest.raises(ExportError):
        audio_format("ogg")


# -- path composition -------------------------------------------------------- #
@pytest.mark.parametrize("typed,expected", [
    ("song", "song.wav"),
    ("song.wav", "song.wav"),
    ("song.mid", "song.wav"),
    ("song.midi", "song.wav"),
    ("  song  ", "song.wav"),
    ("", "untitled.wav"),
])
def test_compose_export_path_never_stacks_suffixes(tmp_path, typed, expected):
    assert compose_export_path(tmp_path, typed, ".wav") == tmp_path / expected


def test_default_audio_output_follows_the_format_extension():
    source = Path("music/song.mid")
    assert default_audio_output(source, "flac").name == "song_qmidi.flac"
    assert default_audio_output(source, "wav").name == "song_qmidi.wav"
    assert default_audio_output(source, "mp3").name == "song_qmidi.mp3"


# -- writing containers ------------------------------------------------------ #
def _stereo(frames: int = 1000) -> np.ndarray:
    ramp = np.linspace(-0.5, 0.5, frames, dtype=np.float32)
    return np.stack([ramp, -ramp])


@pytest.mark.parametrize("depth,subtype", [
    (16, "PCM_16"),
    (24, "PCM_24"),
    (32, "FLOAT"),
])
def test_wav_keeps_the_requested_depth(tmp_path, depth, subtype):
    import soundfile as sf

    target = write_lossless(tmp_path / "a.wav", _stereo(), 48000,
                            fmt="wav", bit_depth=depth)
    info = sf.info(str(target))
    assert info.subtype == subtype
    assert info.samplerate == 48000
    assert info.channels == 2


@pytest.mark.parametrize("depth", [16, 24])
def test_flac_keeps_the_requested_depth(tmp_path, depth):
    import soundfile as sf

    target = write_lossless(tmp_path / "a.flac", _stereo(), 48000,
                            fmt="flac", bit_depth=depth)
    assert sf.info(str(target)).subtype == f"PCM_{depth}"


def test_a_depth_the_container_cannot_store_is_refused(tmp_path):
    audio = _stereo()
    with pytest.raises(ExportError):
        write_lossless(tmp_path / "a.flac", audio, 48000, fmt="flac", bit_depth=32)
    with pytest.raises(ExportError):
        write_lossless(tmp_path / "a.mp3", audio, 48000, fmt="mp3", bit_depth=16)


# -- export_audio ------------------------------------------------------------ #
def test_export_audio_validates_before_it_renders(tmp_path):
    midi = _tiny_midi(tmp_path / "ok.mid")
    with pytest.raises(ExportError, match="cannot write"):
        export_audio(midi, tmp_path / "x.wav", fmt="wav", sample_rate=22050)
    with pytest.raises(ExportError, match="cannot write"):
        export_audio(midi, tmp_path / "x.mp3", fmt="mp3", sample_rate=96000)
    with pytest.raises(ExportError, match="supports"):
        export_audio(midi, tmp_path / "x.flac", fmt="flac", bit_depth=32)
    with pytest.raises(ExportError, match="bitrate"):
        export_audio(midi, tmp_path / "x.mp3", fmt="mp3", bitrate=999)


def test_export_audio_writes_flac_through_the_real_encoder(tmp_path, monkeypatch):
    """The render is stubbed; the container path is what is under test."""
    midi = _tiny_midi(tmp_path / "ok.mid")
    audio = np.stack([np.sin(np.linspace(0.0, 20.0, 4800)).astype(np.float32)] * 2)

    monkeypatch.setattr(exporters, "_render_with_pans",
                        lambda *args, **kwargs: (audio, Path("fake.sfz"), 48000))

    result = export_audio(midi, tmp_path / "out.flac", fmt="flac", bit_depth=24)

    import soundfile as sf

    assert result.format == "flac"
    assert result.bit_depth == 24
    assert result.bitrate == 0
    assert result.frames == 4800
    assert result.sample_rate == 48000
    assert sf.info(str(result.path)).subtype == "PCM_24"
    assert result.path.name == "out.flac"


def test_export_audio_reports_no_depth_for_mp3(tmp_path, monkeypatch):
    midi = _tiny_midi(tmp_path / "ok.mid")
    audio = np.stack([np.sin(np.linspace(0.0, 20.0, 4800)).astype(np.float32)] * 2)
    monkeypatch.setattr(exporters, "_render_with_pans",
                        lambda *args, **kwargs: (audio, Path("fake.sfz"), 48000))

    result = export_audio(midi, tmp_path / "out.mp3", fmt="mp3", bitrate=320)

    assert result.format == "mp3"
    assert result.bit_depth == 0
    assert result.bitrate == 320
    assert result.path.stat().st_size > 0
