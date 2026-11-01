"""Live preview: one renderer thread feeding one QAudioSink.

Why not just play a pre-rendered buffer?  Measured on this machine, dense
passages render at 0.8-1.7x real time (AGENTS.md §4.1) -- a five second loop
would need four seconds of silence first.  Streaming lets playback start after
~0.3 s of audio exists, which is the difference between "adjust and listen" and
"adjust and wait".

Threading contract
------------------
The engine is owned by :class:`_RenderWorker` and touched by nobody else.  The
worker pushes finished float blocks into a lock-protected ring; the GUI thread
pulls int16 bytes out of it on a timer.  ``SfzEngine`` never leaves the worker,
so there is no engine locking anywhere -- the one thing that would be painful
to debug.
"""

from __future__ import annotations

import gc
import threading
import time
from collections import deque
from typing import Optional

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtMultimedia import QAudioFormat, QAudioSink, QMediaDevices
from PySide6.QtWidgets import QApplication

from ..engine import SfzEngine
from ..i18n import tr
from ..offline import BlockStreamer, build_timeline
from .render_plan import (
    PREVIEW_BLOCK_SIZE,
    PREVIEW_NUM_VOICES,
    PREVIEW_SAMPLE_QUALITY,
    PREVIEW_SAMPLE_RATE,
    PreviewRequest,
)

#: Static preview gain.  sfizz sums voices without limiting, so the sample file
#: peaks around 2.2 in its loudest bars.  A *fixed* 0.45 (about -7 dB) keeps
#: that in range while never touching the dynamics -- a limiter would hide
#: exactly the velocity differences this program exists to expose.  The status
#: bar says the preview is attenuated so nobody mistakes it for the export.
PREVIEW_GAIN = 0.45

#: How long the renderer steps aside after each block so the GUI thread can run.
#:
#: **Left at zero on purpose.**  ``Event.wait`` is not a sub-millisecond tool on
#: Windows: asking for 0.1 ms dropped the render from 1.08x to 0.73x real time
#: and starved 72% of the pump calls, because the wait rounds up to a scheduler
#: tick.  It turned out to be unnecessary anyway -- once the preview renders at
#: 2x real time (see ``PREVIEW_SAMPLE_QUALITY``) each block holds the interpreter
#: lock briefly enough that the GUI thread gets in on its own: measured 78 pump
#: ticks a second against a timer asking for 83, with only 4% of gaps over 25 ms.
#: Keep the constant so a future machine can be re-measured, but do not raise it
#: without measuring the starvation rate.
YIELD_SEC = 0.0

#: Seconds of audio kept ready ahead of the device.  1.5 s absorbs a UI hiccup
#: (one 80 ms repaint) without the pump ever reaching the bottom of the ring;
#: it does not delay the start, because playback begins as soon as the device
#: buffer is full, not when the ring is.
BUFFER_SEC = 1.5

#: Stereo int16: 2 channels * 2 bytes.
FRAME_BYTES = 4

#: GUI timer period.  Short enough to top the device up smoothly, long enough
#: to cost nothing while painting the piano roll.
PUMP_MS = 12


class _Ring:
    """Thread-safe FIFO of ready-to-write int16 bytes.

    The float -> int16 conversion happens in :meth:`push`, i.e. on the renderer
    thread.  Doing it in the GUI thread instead (the original design) meant the
    pump -- already a 12 ms timer competing with painting -- also allocated and
    converted ~1000 frames every tick.  Measured 2026-10-25: moving it here is
    free (the renderer has headroom) and takes the last numpy work out of the
    audio-feeding path.
    """

    def __init__(self, capacity_frames: int) -> None:
        self._blocks: deque[np.ndarray] = deque()
        self._frames = 0
        self._capacity = int(capacity_frames)
        self._lock = threading.Lock()

    def push(self, block: np.ndarray) -> None:
        # Store floats and convert on the way out.  Converting here instead
        # (the previous design) cost the renderer 46% of its throughput --
        # measured 2.09x -> 1.12x -- because the numpy pass runs on the same
        # thread that has to keep the ring fed.
        with self._lock:
            self._blocks.append(np.asarray(block, dtype=np.float32))
            self._frames += block.shape[1]

    def clear(self) -> None:
        with self._lock:
            self._blocks.clear()
            self._frames = 0

    @property
    def frames(self) -> int:
        with self._lock:
            return self._frames

    def full(self) -> bool:
        with self._lock:
            return self._frames >= self._capacity

    def empty(self) -> bool:
        with self._lock:
            return self._frames == 0

    def take(self, frames: int) -> bytes:
        """Up to ``frames`` frames as interleaved int16 bytes (may be empty)."""
        wanted = max(0, int(frames))
        if wanted <= 0:
            return b""
        with self._lock:
            if not self._blocks:
                return b""
            pieces: list[np.ndarray] = []
            while wanted > 0 and self._blocks:
                block = self._blocks[0]
                have = block.shape[1]
                if have <= wanted:
                    pieces.append(self._blocks.popleft())
                    self._frames -= have
                    wanted -= have
                else:
                    pieces.append(block[:, :wanted])
                    self._blocks[0] = np.ascontiguousarray(block[:, wanted:])
                    self._frames -= wanted
                    wanted = 0
        if not pieces:
            return b""
        audio = pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=1)
        scaled = np.clip(audio * PREVIEW_GAIN, -1.0, 1.0)
        return (scaled.T * 32767.0).astype("<i2").tobytes()


class _RenderWorker(threading.Thread):
    """Renders one request into the ring until stopped (or the span ends)."""

    def __init__(self, request: PreviewRequest, ring: _Ring, on_error) -> None:
        super().__init__(daemon=True, name="qmidi-preview-render")
        self.request = request
        self._ring = ring
        self._stop_event = threading.Event()
        self._on_error = on_error
        self._seek_lock = threading.Lock()
        self._seek_to: Optional[float] = None
        self.error: Optional[str] = None
        self.underruns = 0

    def stop(self) -> None:
        self._stop_event.set()

    def seek_to(self, seconds: float) -> None:
        """Ask the renderer to jump; honoured before the next block is produced."""
        with self._seek_lock:
            self._seek_to = float(seconds)

    def _take_seek(self) -> Optional[float]:
        with self._seek_lock:
            target = self._seek_to
            self._seek_to = None
        return target

    def run(self) -> None:  # pragma: no cover - needs an instrument and a thread
        try:
            timeline = build_timeline(self.request.midi_path,
                                      tracks=sorted(self.request.tracks))
            if not timeline:
                self.error = tr("所选轨道没有可播放的事件（可能全部静音或文件为空）")
                self._on_error(self.error)
                return
            engine = SfzEngine(sample_rate=PREVIEW_SAMPLE_RATE,
                               block_size=PREVIEW_BLOCK_SIZE,
                               freewheeling=True,
                               sample_quality=PREVIEW_SAMPLE_QUALITY,
                               num_voices=PREVIEW_NUM_VOICES)
            engine.load_sfz(self.request.sfz_path)
            streamer = BlockStreamer(engine, timeline)
            start, end = self.request.start_sec, self.request.end_sec
            streamer.seek(start)
            while not self._stop_event.is_set():
                target = self._take_seek()
                if target is not None:
                    # Jumping here -- not restarting the device in the GUI thread
                    # -- is what makes scrubbing during playback cost ~50 ms of
                    # silence instead of the ~0.3 s engine start-up.
                    streamer.seek(target)
                    self._ring.clear()
                    continue
                if self._ring.full():
                    # Backpressure is the whole flow-control story: never render
                    # more than a buffer ahead of what the device has consumed.
                    self._stop_event.wait(0.004)
                    continue
                if streamer.position >= end:
                    if not self.request.loop:
                        break
                    streamer.seek(start)
                    continue
                self._ring.push(streamer.next_block())
                # Hand the interpreter lock over after every block.  pysfizz holds
                # it for the whole render, so without this the GUI thread waits
                # until the ring is full -- measured 280 ms stalls and a pump that
                # ran 10 times a second instead of 83.  A real wait (not sleep(0),
                # which on Windows is barely a yield) costs ~0.5 ms per 2.7 ms
                # block and buys a stable window.
                if YIELD_SEC:
                    self._stop_event.wait(YIELD_SEC)
        except Exception as exc:  # surface instrumentation problems, don't hang
            self.error = f"{type(exc).__name__}: {exc}"
            self._on_error(self.error)


class PreviewPlayer(QObject):
    """Owns the audio device and the renderer; one playback at a time."""

    positionChanged = Signal(float)  # seconds in the file
    started = Signal(str)  # human label
    finished = Signal()
    failed = Signal(str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._sink: Optional[QAudioSink] = None
        self._io = None
        self._ring: Optional[_Ring] = None
        self._worker: Optional[_RenderWorker] = None
        self._request: Optional[PreviewRequest] = None
        self._written_frames = 0
        #: Position follows the device's own progress counter *relative to a
        #: baseline*, not the wall clock.  ``processedUSecs`` starts counting only
        #: when the device begins making sound, so it is the one number that knows
        #: how far the ears have got; the wall clock ran ~0.7 s ahead of it
        #: (measured 2026-11-01).  The baseline moves on seek, so the audio still
        #: buffered from before the jump does not teleport the playhead.
        self._device_base: Optional[float] = None
        self._clock_position = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(PUMP_MS)
        self._timer.timeout.connect(self._pump)

    # -- public ------------------------------------------------------------ #
    @property
    def is_playing(self) -> bool:
        return self._sink is not None

    def play(self, request: PreviewRequest) -> bool:
        """Start ``request``; returns False when no audio device is available."""
        self.stop()
        device = QMediaDevices.defaultAudioOutput()
        if device.isNull():
            self.failed.emit(tr("找不到音频输出设备"))
            return False

        ring = _Ring(int(PREVIEW_SAMPLE_RATE * BUFFER_SEC))
        self._ring = ring
        self._request = request
        self._written_frames = 0
        self._clock_position = request.start_sec
        self._device_base = None  # set by the first pump that hears audio
        self._worker = _RenderWorker(request, ring, self._worker_failed)

        fmt = QAudioFormat()
        fmt.setSampleRate(PREVIEW_SAMPLE_RATE)
        fmt.setChannelCount(2)
        fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        sink = QAudioSink(device, fmt, self)
        # ~250 ms of device buffer: enough to ride out a thread hiccup without
        # making "stop" feel laggy.
        sink.setBufferSize(PREVIEW_SAMPLE_RATE // 4 * FRAME_BYTES)
        self._sink = sink
        self._worker.start()
        # Let the ring fill before the device starts asking for data.  Starting
        # both at once left the first ~0.7 s of every play with an empty ring --
        # measured 9-12 starved pumps, which is the little stutter you hear at
        # the very beginning of playback.  Costs 250 ms of extra latency once,
        # and the event loop keeps running so the window does not freeze.
        self._prime(ring, PREVIEW_SAMPLE_RATE // 4)
        self._io = sink.start()
        if self._io is None:
            self.stop()
            self.failed.emit(tr("音频设备启动失败（格式不被支持？）"))
            return False
        self._timer.start()
        # Playback churns through a few thousand short-lived bytes objects every
        # second.  A generation-2 collection landing mid-buffer is a 100-200 ms
        # stall -- measured 2026-10-25, and long enough to drain the ring.  The
        # ring holds plain bytes (no reference cycles), so the cycle collector
        # buys nothing here; it is re-enabled in ``stop``.
        gc.disable()
        self.started.emit(request.label)
        return True

    def seek(self, seconds: float) -> None:
        """Jump while playing; the audio device is never stopped.

        Stopping it would cost the engine start-up (~0.3 s) on every click, which
        is exactly what makes scrubbing during playback feel broken.  The clock
        moves immediately; the renderer catches up within one block, and the
        few tens of milliseconds of stale buffered audio are what you hear as
        the jump itself.
        """
        worker = self._worker
        if worker is None:
            return
        target = max(0.0, float(seconds))
        self._clock_position = target
        sink = self._sink
        if sink is not None and self._device_base is not None:
            # The device is still playing what it buffered before the jump.  Skip
            # exactly that much device progress, so the playhead waits for the new
            # audio instead of jumping ahead of it.
            latency = sink.bufferSize() / float(PREVIEW_SAMPLE_RATE * FRAME_BYTES)
            self._device_base = sink.processedUSecs() / 1_000_000.0 + latency
        worker.seek_to(target)

    def stop(self) -> None:
        self._timer.stop()
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.stop()
        io, self._io = self._io, None
        if io is not None:
            try:
                io.close()
            except RuntimeError:
                pass
        sink, self._sink = self._sink, None
        if sink is not None:
            sink.stop()
            sink.deleteLater()
        if self._ring is not None:
            self._ring.clear()
        self._request = None
        self._device_base = None
        self._clock_position = 0.0
        gc.enable()  # paired with play(); harmless when it was never disabled
        if worker is not None and worker is not threading.current_thread():
            # One block of render is ~15 ms; joining briefly is cheaper than
            # leaving a thread that could still touch a deleted engine.
            worker.join(timeout=0.5)

    # -- internals --------------------------------------------------------- #
    @staticmethod
    def _prime(ring: "_Ring", frames: int, timeout: float = 1.5) -> None:
        """Wait for ``ring`` to hold ``frames`` frames, without freezing the UI.

        ``processEvents`` matters: a plain sleep here would freeze the window for
        the whole priming period, which is exactly the kind of stall this method
        exists to remove.
        """
        app = QApplication.instance()
        deadline = time.monotonic() + timeout
        while ring.frames < frames:
            if time.monotonic() > deadline:
                return  # a slow renderer: start anyway, a hang is worse
            if app is not None:
                app.processEvents()
            time.sleep(0.002)

    def _worker_failed(self, message: str) -> None:
        # Called from the renderer thread.  The signal is queued to the GUI
        # thread; stop() must be deferred too, because joining the worker from
        # inside the worker raises "cannot join current thread".
        self.failed.emit(message)
        QTimer.singleShot(0, self.stop)

    def _pump(self) -> None:
        sink, ring, request = self._sink, self._ring, self._request
        if sink is None or ring is None or request is None:
            return
        free = sink.bytesFree()
        if free >= 4096:
            data = ring.take(free // FRAME_BYTES)
            if data and self._io is not None:
                self._io.write(data)
                self._written_frames += len(data) // FRAME_BYTES

        # "Drained" means the *device* has nothing left queued -- not that we
        # stopped writing.  Comparing the lifetime ``_written_frames`` counter
        # with the buffer size made this false forever, so a finished page never
        # reported itself finished (found 2026-11-01 while adding the release
        # tail: the audio stopped but the transport still said "playing").
        queued = max(0, sink.bufferSize() - sink.bytesFree())
        drained = queued <= FRAME_BYTES
        page_done = self._worker is not None and not self._worker.is_alive()
        if drained and page_done and ring.empty():
            if not request.loop:
                # Park the playhead on the last sample: that is where the next
                # press of play resumes from, and a playhead drifting past the
                # end of the file reads as a bug.
                self.positionChanged.emit(request.end_sec)
            self.stop()
            self.finished.emit()
            return

        device = sink.processedUSecs() / 1_000_000.0
        if self._device_base is None:
            # ``processedUSecs`` stays at zero until the device is actually making
            # sound -- about 0.4 s after start(), once the ring has been filled and
            # the backend wakes up.  Until then the playhead holds at the start
            # position instead of running ahead of silence.
            if device <= 0:
                return
            self._device_base = 0.0

        played = self._clock_position + max(0.0, device - self._device_base)
        if request.loop and request.duration_sec > 0:
            position = request.start_sec + (
                (played - request.start_sec) % request.duration_sec)
        else:
            position = min(request.end_sec, played)
        self.positionChanged.emit(position)
