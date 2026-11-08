"""UI language selection and translation lookup (Qt-free).

How it works
------------
Chinese is the source language: every call site writes its Chinese string and
wraps it in :func:`tr`.  With ``zh`` active (the default) ``tr`` returns the
string untouched, so a missed wrap is invisible.  With ``en`` active it looks
the string up in :data:`_EN`; an entry that was never translated falls back to
the Chinese original rather than crashing or showing a key code.

Why not ``gettext`` or Qt ``tr()``: the whole UI vocabulary is ~150 short
strings owned by this project, and a table we can read in one file beats a
toolchain we have to run.  Keys are the Chinese originals, which means the
call site *is* the default translation and cannot drift from it.

When the language is applied
----------------------------
:meth:`Settings.load` runs before any GUI module is imported (see
``gui/app.py:main``), because ``keymap.DEFAULTS`` is built at import time and
its titles must already be translated.  Changing the language in the settings
dialog therefore asks for a restart; live re-translation would need every
widget rebuilt, which is not worth the risk for a setting that changes rarely.
"""

from __future__ import annotations

__all__ = ["LANGUAGES", "current_language", "set_language", "tr"]

#: (value, display name) pairs for the settings dropdown.  The display names
#: are written in each language itself -- that is how every application lists
#: languages, and the only way a user who cannot read the current one can find
#: the switch.
LANGUAGES: tuple[tuple[str, str], ...] = (
    ("zh", "中文"),
    ("en", "English"),
)

_LANGUAGE = "zh"

#: Chinese original -> English.  Only entries that differ need to be here.
_EN: dict[str, str] = {
    # -- keymap: groups ------------------------------------------------------ #
    "文件": "File",
    "播放": "Playback",
    "编辑": "Edit",
    "视图": "View",
    # -- keymap: actions ----------------------------------------------------- #
    "打开 MIDI 文件": "Open MIDI file",
    "打开": "Open",
    "保存工程": "Save Project",
    "保存工程（力度段）": "Save project (velocity segments)",
    "导出 MIDI": "Export MIDI",
    "导出 MP3": "Export MP3",
    "导出音频": "Export audio",
    "重新自动切段": "Re-segment automatically",
    "重切": "Re-cut",
    "按和声自动重做踏板": "Re-do sustain pedal from the harmony",
    "自动踏板": "Auto Pedal",
    "双钢琴分声像": "Pan two pianos apart",
    "声像": "Pan",
    "母带处理（混响/压缩/响度）": "Mastering (reverb / compression / loudness)",
    "母带": "Master",
    "播放 / 停止": "Play / Stop",
    "▶ 播放": "▶ Play",
    "循环当前段": "Loop current segment",
    "↺ 循环": "↺ Loop",
    "回到开头": "Back to start",
    "力度 +3": "Velocity +3",
    "力度 −3": "Velocity −3",
    "力度 +9": "Velocity +9",
    "力度 −9": "Velocity −9",
    "上一段": "Previous segment",
    "下一段": "Next segment",
    "清除本段调整": "Clear this segment",
    "撤销": "Undo",
    "重做": "Redo",
    "放大时间轴": "Zoom in",
    "缩小时间轴": "Zoom out",
    "全曲适配": "Fit whole piece",
    "设置（音源 / 显示 / 快捷键）": "Settings (instrument / display / keys)",
    "设置": "Settings",
    # -- toolbar tooltips ----------------------------------------------------- #
    "保存力度段到 sidecar JSON": "Save velocity segments to the sidecar JSON",
    "从播放头播放；再按一次停止": "Play from the playhead; press again to stop",
    "循环光标所在的那一段": "Loop the segment under the playhead",
    "播放头回到开头": "Playhead back to the start",
    "播放速度：可直接输入，也可以滚轮微调；改完从播放头重新开始":
        "Tempo: type a value or scroll to nudge; changing it restarts playback "
        "from the playhead",
    "当前缩放；点击恢复全曲适配": "Current zoom; click to fit the whole piece",
    "导出": "Export",
    "导出 MIDI，或音频（WAV / FLAC / MP3）":
        "Export MIDI, or audio (WAV / FLAC / MP3)",
    "按当前规则重新切分所有轨道（会丢弃段调整）":
        "Re-cut every track with the current rules (drops segment edits)",
    "按低音与和声变化重做踏板（可撤销）":
        "Re-do the sustain pedal from bass and harmony changes (undoable)",
    "两台钢琴分左右（写进导出的 MIDI 和音频）":
        "Place the two pianos left and right (written into the exported MIDI and audio)",
    "母带：混响 + 温和压缩 + 响度标准化（只进导出的音频）":
        "Mastering: reverb + gentle compression + loudness normalisation "
        "(exported audio only)",
    "缩放到全曲": "Zoom to fit",
    "音源、显示方式和快捷键": "Instrument, display and keyboard shortcuts",
    # -- settings dialog ------------------------------------------------------ #
    "音源": "Instrument",
    "显示": "Display",
    "功能说明": "About the features",
    "快捷键": "Shortcuts",
    "确定": "OK",
    "取消": "Cancel",
    "默认音源（Concert Grand）": "Default instrument (Concert Grand)",
    "选择 .sfz 文件…": "Choose an .sfz file…",
    "选择文件夹…": "Choose a folder…",
    "恢复默认": "Reset to default",
    "当前音源：": "Current instrument:",
    "可以选一个 .sfz 文件，也可以选装着 .sfz 的文件夹（会自动挑基础预设，\n"
    "跳过带括号的变体）。换音源后，预览和音频导出都会用它。":
        "Pick one .sfz file, or a folder of them (the base preset is chosen "
        "automatically, bracketed variants skipped). Both the preview and the "
        "audio export will use it.",
    "界面语言：": "Interface language:",
    "切换语言后需要重启软件才能完全生效。":
        "Restart the application after switching languages.",
    "显示所有轨道的音符": "Show notes from all tracks",
    "关闭时，钢琴卷帘只画当前选中轨道的音符。其他轨道仍然会播放，"
    "只是不再挡视线；切换轨道时，卷帘的纵向位置会自动对准新轨道。":
        "When off, the piano roll draws only the active track's notes.  The "
        "other tracks still play -- they just stay out of the way -- and "
        "switching tracks re-centres the rows on the new one.",
    "十六分音符（1/4 拍）": "Sixteenth notes (1/4 beat)",
    "三十二分音符（1/8 拍）": "Thirty-second notes (1/8 beat)",
    "八分音符（1/2 拍）": "Eighth notes (1/2 beat)",
    "不吸附": "No snapping",
    "播放头吸附：": "Playhead snapping:",
    "点击标尺或总览带定位播放头时，会落在最近的格子上。只影响“把它放到哪”，"
    "播放时跟着音频走的播放头不会跳格。":
        "Clicking the ruler or the overview bar lands the playhead on the "
        "nearest grid line.  This only affects *placing* it; the playhead that "
        "follows audio never jumps between lines.",
    "导出质量（只影响音频导出）：": "Export quality (audio exports only):",
    "最高（渲染最慢）": "Maximum (slowest render)",
    "均衡": "Balanced",
    "快速（约为最高档的四倍速）": "Fast (about 4x the maximum setting)",
    "「快速」把导出速度提高约四倍（七分钟曲子约两分钟），代价是极高频细节"
    "略微变软；「最高」和现在一样。预览不受这个设置影响。":
        "Fast renders about 4x quicker (a seven-minute piece in about two "
        "minutes) at the cost of slightly softer extreme highs; Maximum sounds "
        "as it does now.  The preview is unaffected.",
    "选择音源": "Choose an instrument",
    "选择装有 .sfz 的文件夹": "Choose a folder of .sfz files",
    # -- segment panel -------------------------------------------------------- #
    "未选择力度段": "No segment selected",
    "把这一段所有音符钉在这个绝对力度上":
        "Pin every note of this segment to this absolute velocity",
    "设为": "Set",
    "整段设为这个力度（绝对）": "Set the whole segment to this velocity (absolute)",
    "清零": "Reset",
    "撤销这一段的调整": "Undo this segment's edits",
    "合并→": "Merge →",
    "与下一段合并": "Merge into the next segment",
    "拆分": "Split",
    "在光标处拆开这一段": "Split this segment at the playhead",
    "清踏板": "Clear Pedal",
    "清空当前轨的所有踏板": "Clear every pedal span on this track",
    "循环本段": "Loop This Segment",
    "打开循环并从这一段开始播": "Turn on looping and play from this segment",
    "轨道 {track} · 第 {index}/{total} 段 · {label}":
        "Track {track} · segment {index}/{total} · {label}",
    "拍": "beats",
    "原力度 {origin}  →  现在 {low}–{high}  ·  {notes} 音符  ·  {span}":
        "original {origin}  →  now {low}–{high}  ·  {notes} notes  ·  {span}",
    "偏移 %+d": "offset %+d",
    "固定 %d": "pinned to %d",
    # -- track panel ---------------------------------------------------------- #
    "静音这一轨": "Mute this track",
    "独奏：只听（选中的）这几轨": "Solo: hear only these tracks",
    "重命名音轨…": "Rename Track…",
    "{notes} 音符 · {segments} 段 · 力度 {low}–{high}":
        "{notes} notes · {segments} segments · velocity {low}–{high}",
    # -- main window: status bar ---------------------------------------------- #
    "预览音量 ×{gain:.2f}（导出不衰减）":
        "Preview volume ×{gain:.2f} (export is not attenuated)",
    "播放：{label}": "Playing: {label}",
    "主": "Main",
    "停止": "Stop",
    "■ 停止": "■ Stop",
    "播放结束（空格从当前位置再播）":
        "Playback finished (Space plays again from here)",
    "播放失败：{message}": "Playback failed: {message}",
    "循环：当前段": "Loop: current segment",
    "循环关闭：从播放头播到结尾":
        "Loop off: play from the playhead to the end",
    "先打开一个 MIDI 文件": "Open a MIDI file first",
    "没有可播放的内容（这一段没有音符，或者所有轨道都被静音了）":
        "Nothing to play (this segment has no notes, or every track is muted)",
    "速度 {value} BPM（改动可撤销）": "Tempo {value} BPM (edit is undoable)",
    "时间轴缩小": "Zoom out the time axis",
    "时间轴放大": "Zoom in the time axis",
    # -- main window: dialogs ------------------------------------------------- #
    "打开 MIDI": "Open MIDI",
    "MIDI 文件 (*.mid *.midi)": "MIDI files (*.mid *.midi)",
    "MIDI 文件 (*.mid)": "MIDI files (*.mid)",
    "MP3 文件 (*.mp3)": "MP3 files (*.mp3)",
    "打不开这个文件": "Cannot open this file",
    "{name} · {tracks} 轨 · {segments} 段 · {tpb} ticks/beat":
        "{name} · {tracks} tracks · {segments} segments · {tpb} ticks/beat",
    "工程文件与 MIDI 不一致":
        "The project file disagrees with the MIDI",
    "sidecar 工程文件描述的段与当前 MIDI 对不上，已按能对上的部分载入：\n\n":
        "The segments described by the sidecar project file do not match the "
        "current MIDI; loaded what lined up:\n\n",
    "还有未保存的力度调整": "Unsaved velocity edits",
    "关闭前要保存工程吗？\n（不保存的话，这次的段和力度都会丢失）":
        "Save the project before closing?\n"
        "(If not, this session's segments and velocities are lost)",
    "保存失败": "Save failed",
    "已保存：{name}": "Saved: {name}",
    "音源已切换：{label}": "Instrument switched: {label}",
    "设置已保存": "Settings saved",
    "重命名音轨": "Rename Track",
    "轨道 {track} 的新名字（留空则用文件里的名字）：":
        "New name for track {track} (leave empty to use the name from the file):",
    "轨道 {track} 已改名": "Track {track} renamed",
    "轨道 {track} 用回文件里的名字":
        "Track {track} back to the name from the file",
    # -- main window: re-segment / pedal -------------------------------------- #
    "按当前规则重新切分所有轨道的力度段。\n\n"
    "你做过的合并、拆分、力度调整都会丢失（可以按 Ctrl+Z 撤销）。\n\n继续吗？":
        "Re-cut every track's velocity segments with the current rules.\n\n"
        "Merges, splits and velocity edits you made will be lost "
        "(Ctrl+Z can undo).\n\nContinue?",
    "这个文件没有可切的音符": "This file has no notes to segment",
    "已重新切段：{n} 段": "Re-segmented: {n} segments",
    "生成踏板失败": "Generating the pedal failed",
    "没有可用的音符，没生成踏板": "No usable notes; no pedal was generated",
    "已按和声重做踏板：{count} 轨（Ctrl+Z 撤销；原 MIDI 文件不动）":
        "Re-did the pedal from the harmony on {count} track(s) "
        "(Ctrl+Z to undo; the original MIDI file is untouched)",
    "不能合并：{reason}": "Cannot merge: {reason}",
    "这里不能拆分：{reason}": "Cannot split here: {reason}",
    # -- main window: pan / mastering ------------------------------------------ #
    "这个文件只有一个通道，分不了声像":
        "This file has only one channel; there is nothing to pan apart",
    "关闭（都放中间）": "Off (both centred)",
    "轻微（52 / 76）": "Slight (52 / 76)",
    "标准（40 / 88）": "Standard (40 / 88)",
    "宽阔（28 / 100）": "Wide (28 / 100)",
    "双钢琴声像": "Two-piano panning",
    "钢琴 1 = 通道 {a}，钢琴 2 = 通道 {b}\n分开多少：":
        "Piano 1 = channel {a}, piano 2 = channel {b}\nHow far apart:",
    "声像已关闭（两台琴回到正中间）":
        "Panning off (both pianos back in the centre)",
    "声像：通道 {a} → {left}，通道 {b} → {right}"
    "（导出的 MIDI/音频生效，预览是两架混在中间的）":
        "Pan: channel {a} → {left}, channel {b} → {right} "
        "(applies to the exported MIDI/audio; the preview mixes both in the centre)",
    "声像没变": "Panning unchanged",
    "母带已开启：混响 14% + 2:1 温和压缩 + 响度 -14 LUFS（只作用于导出的音频）":
        "Mastering on: 14% reverb + gentle 2:1 compression + -14 LUFS loudness "
        "(applies only to the exported audio)",
    "母带已关闭：导出的音频回到纯归一化":
        "Mastering off: the exported audio goes back to plain normalisation",
    # -- main window: export ---------------------------------------------------- #
    "正在写出 MIDI…": "Writing the MIDI…",
    "MIDI 已导出：{name}（改力度 {notes} 个，重触发 {retrig} 处，涉及 {tracks} 轨）":
        "MIDI exported: {name} ({notes} velocities rewritten, {retrig} "
        "re-triggers, {tracks} track(s) touched)",
    "导出完成，但有提醒": "Export finished, with warnings",
    "正在渲染音频（整曲，可能要几分钟）…":
        "Rendering the audio (whole piece, may take minutes)…",
    " · 响度 {value} LUFS": " · loudness {value} LUFS",
    "已导出：{name} · {fmt} · {quality} · {rate} Hz · {sec}s · 峰值 {peak}{loudness}":
        "Exported: {name} · {fmt} · {quality} · {rate} Hz · {sec}s · "
        "peak {peak}{loudness}",
    # -- export dialog ---------------------------------------------------------- #
    "文件名：": "File name:",
    "目录：": "Folder:",
    "格式：": "Format:",
    "品质：": "Quality:",
    "采样率：": "Sample rate:",
    "位深度：": "Bit depth:",
    "文件名（不用写扩展名）": "File name (no extension needed)",
    "选择导出目录": "Choose the export folder",
    "还差一个文件名": "A file name is needed",
    "请填写文件名。": "Please enter a file name.",
    "文件已存在": "File already exists",
    "{name} 已存在，要覆盖吗？": "{name} already exists. Overwrite it?",
    "32 位浮点": "32-bit float",
    "{n} 位": "{n}-bit",
    "MIDI 文件（保留原始结构，不渲染音频）":
        "MIDI file (original structure kept, no audio rendered)",
    "WAV 文件（无损，体积最大）": "WAV file (lossless, largest)",
    "FLAC 文件（无损，体积小于 WAV）": "FLAC file (lossless, smaller than WAV)",
    "MP3 文件（有损，最通用）": "MP3 file (lossy, most compatible)",
    "请稍候": "Please wait",
    "{label}\n剩余约 {n} 秒": "{label}\nAbout {n} s remaining",
    "导出失败": "Export failed",
    "无法生成预览 MIDI": "Cannot build the preview MIDI",
    # -- model ------------------------------------------------------------------ #
    "轨道 {track}": "Track {track}",
    "第 {bar} 小节 · {beat} 拍": "Bar {bar} · beat {beat}",
    # -- player ----------------------------------------------------------------- #
    "所选轨道没有可播放的事件（可能全部静音或文件为空）":
        "The selected tracks have no playable events "
        "(all muted, or the file is empty)",
    "找不到音频输出设备": "No audio output device found",
    "音频设备启动失败（格式不被支持？）":
        "The audio device failed to start (unsupported format?)",
    # -- render_plan ------------------------------------------------------------ #
    "循环 {a}–{b}": "Loop {a}–{b}",
    "播放 {a}–{b}": "Play {a}–{b}",
    "整曲 {a}–{b}": "Whole piece {a}–{b}",
    # -- piano_roll ------------------------------------------------------------- #
    "打开一个 MIDI 文件开始（Ctrl+O）": "Open a MIDI file to start (Ctrl+O)",
    # -- settings dialog: leftovers ---------------------------------------------- #
    "SFZ 音源 (*.sfz)": "SFZ instruments (*.sfz)",
    "音源不可用": "Instrument not available",
    "找不到这个音源：\n{path}": "Cannot find this instrument:\n{path}",
    "写不进配置文件：\n{reason}": "Cannot write the configuration file:\n{reason}",
    # -- keymap: display-time pieces ------------------------------------------- #
    "未设置": "unset",
    "空格": "Space",
    "回车": "Enter",
    "退格": "Bksp",
    "未知的快捷键动作：{action}": "Unknown shortcut action: {action}",
    "{key} 已经分配给「{label}」": "{key} is already assigned to \"{label}\"",
    "{up} 力度+3 · {up_big} 力度+9 · {prev} {next} 换段 · 拖力度条整段升降 · "
    "踏板点一下踩、再点一下松":
        "{up} velocity+3 · {up_big} velocity+9 · {prev} {next} switch segment · "
        "drag a velocity bar to move the whole segment · click the pedal to "
        "press it, click again to release",
    # -- segments: labels -------------------------------------------------------- #
    "周期{values}": "cycle {values}",
    "同值{value}": "uniform {value}",
    "同段{values}": "uniform {values}",
    "合并": "merged",
    "手动": "manual",
    # -- panels / lanes ----------------------------------------------------------- #
    "轨道": "Tracks",
    "M 静音 · S 独奏（可多选）· 右键改名":
        "M mute · S solo (multi-select) · right-click to rename",
    "全曲总览：点击或拖动跳转播放位置":
        "Whole-piece overview: click or drag to move the playhead",
    "总览": "Overview",
    "力度": "Velocity",
    "踏板": "Pedal",
    # -- shortcuts page ------------------------------------------------------------ #
    "动作": "Action",
    "点右边任意一行，然后按下你想用的键。Esc 取消，Backspace 清除这一项。":
        "Click any row on the right, then press the key you want.  Esc cancels; "
        "Backspace clears the entry.",
    "正在设置「{label}」：按下新键，Esc 取消，Backspace 清除。":
        "Setting \"{label}\": press the new key.  Esc cancels; Backspace clears.",
    "按下新键…": "Press a new key…",
    "全部恢复默认": "Reset all to defaults",
    "这些键全局生效（所有工程），点一行再按新键即可修改。":
        "These shortcuts work globally (all projects); click a row, then press "
        "the new key.",
    "{reason}。换一个键，或先改那一项。":
        "{reason}  Pick another key, or change that one first.",
    "已是默认设置。": "Everything is at its default.",
    "已修改；点「确定」保存。": "Modified; press OK to save.",
    "已取消这一项。": "That entry was cancelled.",
    "已恢复全部默认；点「确定」保存。": "All defaults restored; press OK to save.",
}


def set_language(language: str) -> None:
    """Select the UI language; anything unknown keeps Chinese."""
    global _LANGUAGE
    _LANGUAGE = language if language in ("zh", "en") else "zh"


def current_language() -> str:
    return _LANGUAGE


def tr(zh: str) -> str:
    """Translate ``zh`` into the active language (identity for Chinese)."""
    if _LANGUAGE == "zh":
        return zh
    return _EN.get(zh, zh)
