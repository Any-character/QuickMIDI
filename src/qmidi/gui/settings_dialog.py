"""The settings window: instrument, display mode, shortcuts.

Three pages because they are the three things the user can actually change
without breaking the app's premise.  Numbers that were tuned against measurements
(preview gain, playback buffer) are deliberately absent: exposing them invites
tuning that quietly ruins either the dynamics or the "change it and hear it"
latency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ..i18n import LANGUAGES, current_language, tr
from ..keymap import KeyMap
from ..settings import Settings
from .shortcut_page import ShortcutPage

_ABOUT_HTML = """
<style>
  body { font-size: 10pt; }
  h3 { margin: 16px 0 4px 0; }
  p  { margin: 6px 0; }
  li { margin: 3px 0; }
  .note { color: #c9a227; }
  .soft { color: #9aa4b2; }
</style>

<h3>这个软件是什么</h3>

<p>从打谱软件导出的钢琴 MIDI 通常缺少演奏表现：所有音符力度相同、延音踏板按小节
机械地踩放、结尾没有自然衰减。这类问题在宿主软件（Cubase、FL Studio 等）中当然可以
修正，但需要先掌握建轨、挂载音源、绘制力度、混音等技能——学习成本远高于问题本身。</p>

<p><b>QuickMIDI 专注于一件事：把这份 MIDI 调整为接近真实演奏的表现，
并导出可分享的音频成品。</b></p>

<p>它与宿主软件的分工：</p>

<ul>
  <li><b>无需搭建工程</b>——打开 MIDI 即可试听，钢琴音色已经内置，
      增益余量、踏板与结尾余音都已预先处理；</li>
  <li><b>调整力度无需逐个修改音符</b>——程序自动按强弱将乐曲分段，
      调整单条滑杆即可整段变化，段内的强弱比例保持不变；</li>
  <li><b>所见即所听</b>——预览与导出的 MIDI 和音频遵循同一套规则，
      包括结尾余音的长度；</li>
  <li><b>不依赖混音经验的润色</b>——以下三个功能各一键完成，均可撤销。</li>
</ul>

<p class="soft">另有一些容易出错的细节已由程序自动处理：原始 .mid 文件不被修改、
预览音量固定预留防削波余量、导出响度统一到平台标准。各自的小节中有具体说明。</p>

<p>工具栏右侧有三个按键：<b>自动踏板</b>、<b>声像</b>、<b>母带</b>。
它们的功能不同，但有三条共同的规则：</p>

<ul>
  <li><b>原始 .mid 文件不会被修改。</b>所有改动都写入乐曲旁的
      <code>.qmidi.json</code>；删除该文件即可完全恢复原状。</li>
  <li><b>三个功能均可通过 Ctrl+Z 撤销</b>，随时恢复到修改前的状态。</li>
  <li>所有改动仅保存在当前工程中，不会应用到其他乐曲。</li>
</ul>

<h3>自动踏板（快捷键 P）</h3>

<p><b>它自动决定延音踏板的切换时机。</b></p>

<p>这类 MIDI 通常每小节固定踩放一次踏板，与音乐内容无关。其结果是不同小节的声音
相互混叠，且无法通过力度调整弥补——混叠的成因不在力度。</p>

<p><b>按下后会发生什么：</b>程序分析全曲音符，每半拍记录一次当前发声的音级集合。
当当前和弦与踏板踩下时的和弦有 <b>2 个以上</b>的音级不同（例如 G 换为 A），
即在该处切换踏板。另设有两重限制：
两次切换间隔至少 <b>3 拍</b>，避免颤音与快速分解和弦导致踏板频繁动作；
至多 <b>6 拍</b>，避免慢速段落过度延音。</p>

<p><b>会影响什么：</b>文件里那条<b>已经带踏板</b>的轨道，它的踏板会被整套替换。
你手画过的踏板不会保留。如果文件里根本没有踏板，这个按键什么也不会做。</p>

<p><b>能听到吗：</b>能。预览和导出用的是同一份踏板。</p>

<h3>声像</h3>

<p><b>它将两架钢琴分别置于左右声道。</b></p>

<p>两架钢琴同置中央时，听觉上难以区分；分置左右后，两个声部的层次会清晰可辨。</p>

<p><b>按下后会发生什么：</b>为每架钢琴（扒谱文件中通常对应一条 MIDI 通道）
指定一个声像位置：0 为最左，127 为最右，64 为正中。该值会写入导出的 MIDI 文件，
其他播放软件同样识别。</p>

<p class="note"><b>需要提前说明：预览中无法听到声像效果。</b>
预览将所有轨道混为一路输出，不包含声道分离，因此声像只能在导出后听到。
此外，声道分离需要将每架钢琴单独渲染后再合并，<b>导出耗时约增加一倍</b>
（七分钟的乐曲约需十一分钟）。两架钢琴都置于中央时不产生此开销。</p>

<h3>母带</h3>

<p><b>它对导出的音频做最后一道处理</b>，使其听感接近发行成品。</p>

<p><b>具体包含四个步骤，按以下顺序：</b></p>
<ol>
  <li><b>混响</b>——增加少量空间感，避免声音过于贴近扬声器。</li>
  <li><b>压缩</b>——适当降低过响段落的电平，使整曲音量更均匀。</li>
  <li><b>限幅</b>——将峰值限制在 -1 分贝以内，避免削波失真。</li>
  <li><b>响度标准化</b>——将整曲响度统一到 -14 LUFS（流媒体平台通行的标准），
      使导出文件在不同设备上音量一致。</li>
</ol>

<p><b>会影响什么：</b>仅导出的音频（WAV、FLAC 与 MP3 都包含）。导出的 MIDI 与预览
均不受影响——预览始终是未经母带处理的声音。</p>

<p class="note"><b>需要特别注意：压缩会削弱力度层次的差异。</b>
这正与本软件的核心用途相关——预览中细致调整的段落强弱对比，
会被压缩部分抹平。<b>因此母带默认关闭。</b>
需要成品听感时开启，需要如实体现力度调整时保持关闭。</p>

<h3>为什么预览与导出可能不同</h3>

<p>以上三项中，只有踏板在预览中可以听到；声像与母带仅作用于导出文件。
这是有意的设计：预览的职责是帮助判断<b>力度与踏板</b>，
若预览也加入混响和压缩，你将基于被处理过的声音调整力度，判断依据就被破坏了。</p>

<p class="soft">若确需在预览中听到声像或母带，需要另一套架构（每个通道独立的音源实例
加实时混响），实现成本与风险显著更高，当前版本未提供。</p>
"""

_ABOUT_HTML_EN = """
<style>
  body { font-size: 10pt; }
  h3 { margin: 16px 0 4px 0; }
  p  { margin: 6px 0; }
  li { margin: 3px 0; }
  .note { color: #c9a227; }
  .soft { color: #9aa4b2; }
</style>

<h3>What this is</h3>

<p>Piano MIDI exported from notation software usually lacks performance nuance:
uniform velocities, a sustain pedal applied mechanically once per bar, and no
natural decay at the ending.  These are certainly fixable in a DAW (Cubase, FL
Studio and the like), but doing so requires tracks, instrument routing, velocity
editing and mixing -- a learning cost far above the problem itself.</p>

<p><b>QuickMIDI does one thing: shape that MIDI toward a convincing performance,
and export audio you can share.</b></p>

<p>How it divides the work with a DAW:</p>

<ul>
  <li><b>No project setup</b> -- open the MIDI and it plays; a piano sound is
      built in, with gain headroom, pedal and the release tail already
      handled;</li>
  <li><b>Shape dynamics without editing notes</b> -- the piece is divided into
      dynamic segments automatically; moving one slider reshapes the whole
      segment while its internal balance stays intact;</li>
  <li><b>What you hear is what you get</b> -- the preview and the exported
      MIDI/audio follow the same rules, down to the length of the ending;</li>
  <li><b>Polish without mixing experience</b> -- the three features below are
      one click each, and all of them are undoable.</li>
</ul>

<p class="soft">A few error-prone details are handled automatically: the original
.mid is never modified, the preview volume is fixed to leave headroom, and
exported loudness is normalised to what streaming platforms expect.  Each has
its own section below.</p>

<p>The toolbar has three buttons on its right: <b>Auto Pedal</b>, <b>Pan</b> and
<b>Master</b>.  They do different things but share three rules:</p>

<ul>
  <li><b>Your original .mid file is never modified.</b>  Every edit goes into the
      <code>.qmidi.json</code> next to the song; deleting that file restores the
      original state completely.</li>
  <li><b>All three features are undoable with Ctrl+Z.</b></li>
  <li>Their edits live in this project only -- they never carry over to another song.</li>
</ul>

<h3>Auto Pedal (key P)</h3>

<p><b>It decides for you when to change the sustain pedal.</b></p>

<p>MIDI exported from notation software tends to apply the pedal once per bar,
regardless of the music.  The result is that passages bleed into each other, and
no velocity edit can fix that -- the blending does not come from velocity.</p>

<p><b>What happens when you press it:</b> the program walks the whole piece and,
every half beat, records which pitch classes are sounding.  When the current
chord differs from the chord the pedal was pressed on by <b>more than 2 pitch
classes</b> (G moved to A, say), the pedal changes there.  Two limits apply: at
least <b>3 beats</b> between changes, so that trills and fast arpeggios cannot
flurry the pedal; at most <b>6 beats</b>, so that slow passages do not over-ring.</p>

<p><b>What it touches:</b> the pedal on the track that <b>already carries one</b>
is replaced wholesale -- pedal spans drawn by hand are not kept.  A file with no
pedal at all leaves this button without effect.</p>

<p><b>Is it audible:</b> yes.  Preview and export share the same pedal.</p>

<h3>Pan</h3>

<p><b>It places the two pianos left and right.</b>  Both centred, they are hard
to tell apart; separated, the two parts become clearly distinguishable.</p>

<p><b>What happens when you press it:</b> each piano (usually one MIDI channel
per piano in transcription files) gets a position: 0 far left, 127 far right,
64 centre.  The value is written into the exported MIDI file, so other players
honour it too.</p>

<p class="note"><b>To state the cost up front: panning is not audible in the
preview.</b>  The preview mixes all tracks into one stream, so panning is only
audible after export.  Separating the pianos also means rendering each one
separately and combining them, <b>which about doubles the export time</b> (a
seven-minute piece takes around eleven minutes).  Both centred, there is no cost.</p>

<h3>Master</h3>

<p><b>It applies a final pass to the exported audio</b>, bringing its sound closer
to a released recording.  Four steps, in order:</p>
<ol>
  <li><b>Reverb</b> -- a modest amount of space, so the piano does not sound
      glued to the speakers.</li>
  <li><b>Compression</b> -- the loudest passages are eased back for a more even
      level.</li>
  <li><b>Limiter</b> -- peaks are capped at -1 dB, preventing clipping
      distortion.</li>
  <li><b>Loudness normalisation</b> -- the whole piece lands at -14 LUFS (the
      common streaming target), so it plays at a consistent volume across
      devices.</li>
</ol>

<p><b>What it touches:</b> only the exported audio (WAV, FLAC and MP3 alike).
The exported MIDI is unchanged, and so is the preview -- the preview is always
the un-mastered sound.</p>

<p class="note"><b>One thing must be stated plainly: compression reduces the
differences between dynamics.</b>  That bears directly on this software's
purpose -- the passage-to-passage contrasts you worked out in the preview get
narrowed somewhat.  <b>It is therefore off by default.</b>  Turn it on for a
finished sound; leave it off to export the dynamics exactly as you set them.</p>

<h3>Why preview and export can differ</h3>

<p>Of the three, only the pedal is audible in the preview; pan and mastering act
on exported files only.  That is deliberate: the preview exists so you can judge
<b>velocity and pedal</b>, and judging velocity against reverb-and-compression
would be judging the wrong sound.</p>

<p class="soft">Hearing pan or mastering in the preview would need a different
architecture (one instrument instance per channel plus real-time reverb) --
substantially more cost and risk, so it is not built in the current version.</p>
"""


class SettingsDialog(QDialog):
    """Edit a copy of the settings; ``accept`` writes both files."""

    def __init__(self, settings: Settings, keymap: KeyMap,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("设置"))
        self.resize(640, 720)
        self._settings = settings
        self._instrument = settings.sfz_path

        tabs = QTabWidget(self)
        tabs.addTab(self._instrument_page(), tr("音源"))
        tabs.addTab(self._display_page(), tr("显示"))
        tabs.addTab(self._about_page(), tr("功能说明"))
        self.shortcuts = ShortcutPage(keymap, self)
        tabs.addTab(self.shortcuts, tr("快捷键"))

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(tr("确定"))
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(tr("取消"))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(tabs, 1)
        layout.addWidget(buttons)

    # -- pages ------------------------------------------------------------- #
    def _instrument_page(self) -> QWidget:
        page = QWidget(self)
        self.instrument_edit = QLineEdit(self._instrument, page)
        self.instrument_edit.setReadOnly(True)
        self.instrument_edit.setPlaceholderText(tr("默认音源（Concert Grand）"))

        choose = QPushButton(tr("选择 .sfz 文件…"), page)
        choose.clicked.connect(self._choose_file)
        folder = QPushButton(tr("选择文件夹…"), page)
        folder.clicked.connect(self._choose_folder)
        reset = QPushButton(tr("恢复默认"), page)
        reset.clicked.connect(self._reset_instrument)

        row = QHBoxLayout()
        row.addWidget(choose)
        row.addWidget(folder)
        row.addWidget(reset)
        row.addStretch(1)

        hint = QLabel(
            tr("可以选一个 .sfz 文件，也可以选装着 .sfz 的文件夹（会自动挑基础预设，\n"
               "跳过带括号的变体）。换音源后，预览和音频导出都会用它。"),
            page,
        )
        hint.setWordWrap(True)

        layout = QVBoxLayout(page)
        layout.addWidget(QLabel(tr("当前音源："), page))
        layout.addWidget(self.instrument_edit)
        layout.addLayout(row)
        layout.addWidget(hint)
        layout.addStretch(1)
        return page

    def _display_page(self) -> QWidget:
        page = QWidget(self)
        self.language = QComboBox(page)
        for value, display in LANGUAGES:
            self.language.addItem(display, value)
        index = self.language.findData(self._settings.language)
        self.language.setCurrentIndex(index if index >= 0 else 0)
        language_hint = QLabel(
            tr("切换语言后需要重启软件才能完全生效。"), page)
        language_hint.setWordWrap(True)

        layout = QVBoxLayout(page)
        layout.addWidget(QLabel(tr("界面语言："), page))
        layout.addWidget(self.language)
        layout.addWidget(language_hint)
        layout.addSpacing(14)

        self.show_all = QCheckBox(tr("显示所有轨道的音符"), page)
        self.show_all.setChecked(self._settings.show_all_tracks)
        layout.addWidget(self.show_all)

        hint = QLabel(
            tr("关闭时，钢琴卷帘只画当前选中轨道的音符。其他轨道仍然会播放，"
               "只是不再挡视线；切换轨道时，卷帘的纵向位置会自动对准新轨道。"),
            page,
        )
        hint.setWordWrap(True)

        self.snap = QComboBox(page)
        for label, division in ((tr("十六分音符（1/4 拍）"), 4),
                                (tr("三十二分音符（1/8 拍）"), 8),
                                (tr("八分音符（1/2 拍）"), 2),
                                (tr("不吸附"), 0)):
            self.snap.addItem(label, division)
        index = self.snap.findData(self._settings.snap_division)
        self.snap.setCurrentIndex(index if index >= 0 else 0)

        self.export_quality = QComboBox(page)
        for label, quality in ((tr("最高（渲染最慢）"), 10),
                               (tr("均衡"), 8),
                               (tr("快速（约为最高档的四倍速）"), 6)):
            self.export_quality.addItem(label, quality)
        index = self.export_quality.findData(self._settings.export_quality)
        self.export_quality.setCurrentIndex(index if index >= 0 else 0)

        snap_hint = QLabel(
            tr("点击标尺或总览带定位播放头时，会落在最近的格子上。只影响“把它放到哪”，"
               "播放时跟着音频走的播放头不会跳格。"),
            page,
        )
        snap_hint.setWordWrap(True)

        layout.addWidget(self.show_all)
        layout.addWidget(hint)
        layout.addSpacing(14)
        layout.addWidget(QLabel(tr("播放头吸附："), page))
        layout.addWidget(self.snap)
        layout.addWidget(snap_hint)
        layout.addSpacing(14)
        layout.addWidget(QLabel(tr("导出质量（只影响音频导出）："), page))
        layout.addWidget(self.export_quality)
        quality_hint = QLabel(
            tr("「快速」把导出速度提高约四倍（七分钟曲子约两分钟），代价是极高频细节"
               "略微变软；「最高」和现在一样。预览不受这个设置影响。"),
            page,
        )
        quality_hint.setWordWrap(True)
        layout.addWidget(quality_hint)
        layout.addStretch(1)
        return page

    def _about_page(self) -> QWidget:
        """What the three polish buttons do, in words a musician can act on.

        Deliberately not a marketing page: it says what is added, what it costs,
        what it does *not* touch, and how to get back.  The user asked for plain
        language and detail, not adjectives.  Both languages live in this file
        because the page is one continuous argument, not a list of labels.
        """
        view = QTextBrowser(self)
        view.setObjectName("aboutPage")
        view.setOpenExternalLinks(False)
        view.setHtml(_ABOUT_HTML_EN if current_language() == "en" else _ABOUT_HTML)
        return view

    # -- instrument -------------------------------------------------------- #
    def _choose_file(self) -> None:
        start = str(Path(self._instrument).parent) if self._instrument else ""
        path, _ = QFileDialog.getOpenFileName(self, tr("选择音源"), start,
                                              tr("SFZ 音源 (*.sfz)"))
        if path:
            self._instrument = path
            self.instrument_edit.setText(path)

    def _choose_folder(self) -> None:
        start = self._instrument if self._instrument else str(Path.home())
        path = QFileDialog.getExistingDirectory(self, tr("选择装有 .sfz 的文件夹"), start)
        if path:
            self._instrument = path
            self.instrument_edit.setText(path)

    def _reset_instrument(self) -> None:
        self._instrument = ""
        self.instrument_edit.setText("")

    def _instrument_ok(self) -> bool:
        if not self._instrument:
            return True
        path = Path(self._instrument)
        valid = path.is_file() if path.suffix.lower() == ".sfz" else path.is_dir()
        if not valid:
            QMessageBox.warning(self, tr("音源不可用"),
                                tr("找不到这个音源：\n{path}").format(path=path))
            return False
        return True

    # -- result ------------------------------------------------------------ #
    @property
    def result_settings(self) -> Settings:
        return Settings(
            sfz_path=self._instrument,
            show_all_tracks=self.show_all.isChecked(),
            snap_division=int(self.snap.currentData()),
            export_quality=int(self.export_quality.currentData()),
            language=str(self.language.currentData()),
        )

    @property
    def result_keymap(self) -> KeyMap:
        return self.shortcuts.keymap

    def accept(self) -> None:  # noqa: D102 (Qt override)
        if not self._instrument_ok():
            return
        self.shortcuts.stop_recording()
        try:
            self.result_settings.save()
            self.result_keymap.save()
        except OSError as exc:
            QMessageBox.warning(self, tr("保存失败"),
                                tr("写不进配置文件：\n{reason}").format(reason=exc))
            return
        super().accept()
