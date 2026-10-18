"""Pure ASS (Advanced SubStation Alpha) karaoke caption builder for burned-in reel captions.

No file I/O, no ffmpeg calls — this is a pure string builder.

Invariants (CLAUDE.md §"Nicht verhandelbare Invarianten"):
- All timing state is kept in integer frames.
- ASS centiseconds are produced *only* at the format edge (``_frame_to_ass_time``).
- ASS is an export artefact; it is never the source of truth.
"""

from __future__ import annotations

# Word = (text, start_frame, end_frame)  — end_frame is end-exclusive per project invariant.
Word = tuple[str, int, int]
Line = list[Word]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_STYLE_FORMAT = (
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
    "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
    "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding"
)

_EVENTS_FORMAT = "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"


def _frame_to_ass_time(frame: int, rate_num: int, rate_den: int) -> str:
    """Convert an integer frame index to ASS time ``h:mm:ss.cc``.

    ASS uses centiseconds (1/100 s).  Hours have *no* leading zero by
    convention (``0:00:01.00``, not ``00:00:01.00``).

    The conversion is done as pure integer arithmetic to avoid float drift,
    then a single division at the end produces the centisecond component.
    """
    # total centiseconds = frame * rate_den * 100 / rate_num  (rounded)
    # Use integer rounding: add rate_num//2 before dividing for HALF_UP.
    total_cs_numerator = frame * rate_den * 100
    total_cs = (total_cs_numerator + rate_num // 2) // rate_num

    # Split into components.
    h, rem = divmod(total_cs, 360000)   # 100 * 3600
    m, rem = divmod(rem, 6000)          # 100 * 60
    s, cc = divmod(rem, 100)

    # cc is already in [0, 99] after the divmod; rounding in integer division
    # can produce total_cs values that are exact multiples of 6000/360000 so
    # cc never reaches 100 via this path — but guard anyway.
    if cc >= 100:  # pragma: no cover
        cc -= 100
        s += 1
        if s >= 60:
            s -= 60
            m += 1
            if m >= 60:
                m -= 60
                h += 1

    return f"{h}:{m:02d}:{s:02d}.{cc:02d}"


def _escape_ass_text(text: str) -> str:
    """Escape *word text* so it cannot break ASS override-block syntax.

    Rules:
    - ``{`` / ``}`` open/close override blocks → replace with ``(`` / ``)``.
    - Backslash starts tag sequences inside override blocks; outside they are
      passed through by most renderers but can form spurious ``\\N`` (hard
      newline) or ``\\h`` (non-breaking space) → drop stray backslashes.
    - Literal newlines in a word are rare but normalised to ASS hard-newline.
    """
    text = text.replace("\\", "")      # drop backslashes first (before \n check)
    text = text.replace("\n", r"\N")   # soft-wrap → hard ASS newline
    text = text.replace("{", "(")
    text = text.replace("}", ")")
    return text


def _kf_cs(start_frame: int, end_frame: int, rate_num: int, rate_den: int) -> int:
    """Duration of a karaoke syllable in centiseconds (rounded to nearest int)."""
    duration_cs_num = (end_frame - start_frame) * rate_den * 100
    return (duration_cs_num + rate_num // 2) // rate_num


def _build_dialogue_text(
    words: Line,
    rate_num: int,
    rate_den: int,
    *,
    mode: str = "karaoke",
) -> str:
    """Build the ASS ``Text`` field with ``{\\kf<cs>}`` karaoke tags."""
    parts: list[str] = []
    for text, start_frame, end_frame in words:
        escaped = _escape_ass_text(text)
        if mode == "normal":
            parts.append(escaped)
        else:
            cs = _kf_cs(start_frame, end_frame, rate_num, rate_den)
            parts.append(f"{{\\kf{cs}}}{escaped}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _satzende(vorher: str, naechstes: str) -> bool:
    ende = vorher.rstrip("\"'»«)”“")
    return bool(ende) and ende[-1] in ".!?…" and bool(naechstes) and (
        naechstes.lstrip("\"'»«(„“")[:1].isupper() or naechstes[:1].isdigit())


def group_caption_lines(
    words: list[tuple[str, int, int]],
    *,
    max_words_per_line: int = 4,
    max_gap_frames: int = 15,
    max_chars_per_line: int = 30,
) -> list[list[tuple[str, int, int]]]:
    """Group *words* into caption lines by word-count, width and silence-gap thresholds.

    A new line is started when:
    - the current line already contains ``max_words_per_line`` words, **or**
    - adding the word would push the joined line text past
      ``max_chars_per_line`` characters (at fontsize 72 on a 1080px frame,
      ~31 rendered chars fit — longer lines spill past both edges), **or**
    - the gap between the previous word's ``seq_end`` and the next word's
      ``seq_start`` exceeds ``max_gap_frames``, **or**
    - the previous word ends a sentence (``.``/``!``/``?``/``…``) and the next word starts
      with a capital letter or digit (18.10.2026: without this, lines like "Stimme. Die
      Untertitel" ran across sentences; "z.B. ein" stays together).

    A single word longer than the budget still gets its own line.  Words are
    never reordered or dropped.  Empty input returns ``[]``.
    """
    if not words:
        return []

    lines: list[list[tuple[str, int, int]]] = []
    current: list[tuple[str, int, int]] = []
    current_chars = 0  # joined text length incl. separating spaces

    for word in words:
        if current:
            prev_end = current[-1][2]  # seq_end of the last word in the line
            gap = word[1] - prev_end   # word seq_start − prev seq_end
            if (
                len(current) >= max_words_per_line
                or gap > max_gap_frames
                or current_chars + 1 + len(word[0]) > max_chars_per_line
                or _satzende(current[-1][0], word[0])
            ):
                lines.append(current)
                current = []
                current_chars = 0
        current_chars += len(word[0]) + (1 if current else 0)
        current.append(word)

    if current:
        lines.append(current)

    return lines


def build_ass(
    lines: list[list[tuple[str, int, int]]],
    *,
    rate_num: int,
    rate_den: int,
    play_w: int = 1080,
    play_h: int = 1920,
    fontsize: int = 72,
    margin_v: int = 250,
    mode: str = "karaoke",
    position: str = "bottom",
) -> str:
    """Build a complete ASS document with karaoke word-by-word highlighting.

    Parameters
    ----------
    lines:
        Each element is a *line* = list of ``(text, start_frame, end_frame)``
        tuples.  ``end_frame`` is end-exclusive (project invariant).
    rate_num, rate_den:
        Frame rate as a rational ``rate_num / rate_den`` (e.g. 30/1, 24000/1001).
    play_w, play_h:
        ASS PlayRes dimensions.
    fontsize:
        Point size of the Reel style.
    margin_v:
        Vertical margin (pixels from bottom edge for Alignment 2).
    mode:
        ``"karaoke"`` emits per-word ``\\kf`` tags; ``"normal"`` emits plain lines.
    position:
        ``"top"``, ``"middle"`` or ``"bottom"`` mapped to ASS centre alignments.

    Returns
    -------
    str
        A complete ``.ass`` document as a Unicode string (LF line endings).
    """
    sections: list[str] = []

    # ------------------------------------------------------------------
    # [Script Info]
    # ------------------------------------------------------------------
    sections.append(
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {play_w}\n"
        f"PlayResY: {play_h}\n"
        "WrapStyle: 2\n"
    )

    # ------------------------------------------------------------------
    # [V4+ Styles]
    # ------------------------------------------------------------------
    alignment_by_position = {"top": 8, "middle": 5, "bottom": 2}
    alignment = alignment_by_position.get(position, 2)
    caption_mode = "normal" if mode == "normal" else "karaoke"

    style_fields = (
        "Reel"           # Name
        ",Arial"         # Fontname
        f",{fontsize}"   # Fontsize
        ",&H00FFFFFF"    # PrimaryColour   — white
        ",&H0000FFFF"    # SecondaryColour — karaoke pre-highlight (cyan)
        ",&H00000000"    # OutlineColour   — black
        ",&H80000000"    # BackColour      — semi-transparent black
        ",1"             # Bold
        ",0"             # Italic
        ",0"             # Underline
        ",0"             # StrikeOut
        ",100"           # ScaleX
        ",100"           # ScaleY
        ",0"             # Spacing
        ",0"             # Angle
        ",1"             # BorderStyle
        ",3"             # Outline
        ",1"             # Shadow
        f",{alignment}"  # Alignment — centre by requested vertical position
        ",80"            # MarginL
        ",80"            # MarginR
        f",{margin_v}"   # MarginV
        ",1"             # Encoding
    )
    sections.append(
        "[V4+ Styles]\n"
        f"{_STYLE_FORMAT}\n"
        f"Style: {style_fields}\n"
    )

    # ------------------------------------------------------------------
    # [Events]
    # ------------------------------------------------------------------
    dialogue_rows: list[str] = []
    for line in lines:
        non_empty = [w for w in line if w[0]]
        if not non_empty:
            continue
        start_frame = non_empty[0][1]
        end_frame = non_empty[-1][2]
        start_t = _frame_to_ass_time(start_frame, rate_num, rate_den)
        end_t = _frame_to_ass_time(end_frame, rate_num, rate_den)
        text = _build_dialogue_text(non_empty, rate_num, rate_den, mode=caption_mode)
        dialogue_rows.append(
            f"Dialogue: 0,{start_t},{end_t},Reel,,0,0,0,,{text}"
        )

    events_body = "\n".join(dialogue_rows)
    if events_body:
        events_body += "\n"
    sections.append(f"[Events]\n{_EVENTS_FORMAT}\n{events_body}")

    return "\n".join(sections)
