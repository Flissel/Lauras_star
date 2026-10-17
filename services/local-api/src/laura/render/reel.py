"""Pure ffmpeg video-filter chain builder for social-media reel exports.

No I/O, no ffmpeg calls — only string manipulation. The returned string is
ready to be passed as the value of an ffmpeg ``-vf`` / ``-filter_complex`` option.

Overlay text is supplied via drawtext ``textfile=`` (a *basename*, resolved
against ffmpeg's working directory), NOT inline ``text='…'``. This is a
deliberate correctness choice: arbitrary user text (apostrophes, colons,
commas, ``%``) cannot be escaped reliably for inline drawtext on Windows, and
inline text is also a filtergraph-injection vector. A file read verbatim by
ffmpeg removes both problems. The caller writes the file and passes its
basename (see ``render_clips_mp4``). The basename must be free of ffmpeg filter
metacharacters — callers control it, so we keep this builder pure.
"""
from __future__ import annotations

from math import gcd


def reel_video_chain(
    *,
    vertical: bool,
    hook_textfile: str | None = None,
    disclosure_textfile: str | None = None,
    font: str,
    reel_fit: bool = False,
    out_w: int = 1080,
    out_h: int = 1920,
) -> str:
    """Build a comma-joined ffmpeg video-filter string for a reel export.

    Returns an empty string when no filters are requested.

    Filter order:
    1. Reframe to 9:16 (1080 × 1920)              — when *vertical* is True.
       * Default (``reel_fit=False``): center-crop, clamped to the source
         (``crop='min(iw,ih*9/16)':'min(ih,iw*16/9)', scale=1080:1920``).  Fills the frame
         but may cut off content that is off-center.  The clamp keeps a source already
         narrower/taller than 9:16 from requesting a crop larger than the frame.
       * Fit mode (``reel_fit=True``): scale-to-fit with letterbox padding
         (``scale=1080:1920:force_original_aspect_ratio=decrease:force_divisible_by=2,
         pad=1080:1920:(1080-iw)/2:(1920-ih)/2:black``).  The whole source frame is kept
         visible; dead space is filled with black bars and the frame is centred on both
         axes.  Use for screencasts or any content where center-crop would slice off
         readable text or UI.
    2. Centered top ``drawtext``                   — when *hook_textfile* is set.
    3. Bottom-right ``drawtext``                   — when *disclosure_textfile* is set.

    Note: this function does NOT handle ``reel_blur_fill`` mode.  That mode requires a
    split/overlay sub-graph that cannot be expressed as a simple comma chain.  See
    :func:`reel_blur_fill_graph` for the blur-fill filtergraph fragment and the caller
    (:func:`laura.render.mp4.render_clips_mp4`) for how it is wired in.

    Args:
        vertical:             Reframe to the target canvas (default 9:16, 1080 × 1920).
        hook_textfile:        Basename of a UTF-8 file with the top-centre hook text.
        disclosure_textfile:  Basename of a UTF-8 file with the bottom-right disclosure.
        font:                 Resolved fontfile path passed verbatim to drawtext.
        reel_fit:             When True (and *vertical* is True) use letterbox fit instead
                              of center-crop.  Default ``False`` — existing behavior.
        out_w / out_h:        Target canvas when *vertical* is True. Defaults keep the
                              classic 1080×1920 reel byte-identical; 1080×1080 gives the
                              square (LinkedIn) preset through the same clamp/fit logic.
    """
    parts: list[str] = []

    if vertical:
        if reel_fit:
            # Scale the source to fit *inside* the full out_w×out_h box (decrease on BOTH
            # dimensions), then pad to exactly out_w×out_h, centring on both axes.
            # Fitting against the whole box (not just width) is what keeps a source that
            # is already taller/narrower than the target from overshooting in height — the
            # earlier ``scale=1080:-2`` only constrained width, so a portrait source
            # (e.g. 464×832) scaled to height 1936 and the subsequent ``pad`` to 1920
            # failed ("Padded dimensions cannot be smaller than input dimensions").
            # ``force_divisible_by=2`` keeps both dimensions even (required by libx264).
            parts.append(
                f"scale={out_w}:{out_h}"
                ":force_original_aspect_ratio=decrease:force_divisible_by=2"
            )
            parts.append(f"pad={out_w}:{out_h}:({out_w}-iw)/2:({out_h}-ih)/2:black")
        else:
            # Center-crop to an out_w:out_h window, clamped to the source so a source that
            # is already narrower than the target (e.g. 464×832, taller than 9:16) does not
            # request a crop wider/taller than the frame — that made ``crop=ih*9/16:ih``
            # ask for 468px of width from a 464px source and the crop filter aborted (-22).
            # For landscape/standard sources the min() clamp resolves to the exact ratio,
            # so behavior is unchanged. The ratio is gcd-reduced so the default canvas
            # emits the classic ``ih*9/16`` form byte-identically.
            g = gcd(out_w, out_h) or 1
            rw, rh = out_w // g, out_h // g
            parts.append(f"crop='min(iw,ih*{rw}/{rh})':'min(ih,iw*{rh}/{rw})'")
            parts.append(f"scale={out_w}:{out_h}")

    if hook_textfile:
        parts.append(
            f"drawtext=fontfile={font}:textfile={hook_textfile}"
            ":x=(w-text_w)/2:y=120:fontsize=64:fontcolor=white"
            ":box=1:boxcolor=black@0.5"
        )

    if disclosure_textfile:
        parts.append(
            f"drawtext=fontfile={font}:textfile={disclosure_textfile}"
            ":x=w-text_w-24:y=h-text_h-24:fontsize=30:fontcolor=white"
            ":box=1:boxcolor=black@0.45"
        )

    return ",".join(parts)


def reel_blur_fill_graph(
    in_label: str, out_label: str, *, out_w: int = 1080, out_h: int = 1920, tag: str = "_rb"
) -> str:
    """Return the semicolon-joined ffmpeg filtergraph fragment for blurred-background fill.

    Produces a 1080 × 1920 (9:16) output from an arbitrary-aspect-ratio input by:
      1. Splitting the input into two copies (``split``).
      2. Background copy: scale-to-cover 1080×1920 (``force_original_aspect_ratio=increase``)
         + exact ``crop=1080:1920`` + heavy Gaussian blur (``boxblur=20:2``).
      3. Foreground copy: scale-to-fit 1080×1920 (``force_original_aspect_ratio=decrease``),
         preserving every pixel of the source frame.
      4. Overlay foreground centred on blurred background (``overlay=(W-w)/2:(H-h)/2``).

    This is the industry-standard "blurred-background fill" used by Instagram Reels /
    TikTok for landscape-to-vertical conversion: no black bars, no cropped content.

    The fragment is a *sub-graph* with internal labels — it must be embedded inside a
    larger ``-filter_complex`` string (separated by ``;``).  The caller connects
    ``in_label`` from the preceding stage and reads ``out_label`` for the next stage
    (e.g. drawtext / ass captions).

    Drawtext/ASS caption filters must be applied AFTER this graph (on the composited
    1080×1920 stream) — the caller is responsible for chaining them.

    Example (single-clip hard-cut path):
        ``[vcat]split=2[_bbg][_bfg];[_bbg]scale=...,boxblur=20:2[_bbl];
        [_bfg]scale=...[_bfl];[_bbl][_bfl]overlay=(W-w)/2:(H-h)/2[out]``

    Args:
        in_label:  The labeled input stream (e.g. ``"[vcat]"``).
        out_label: The labeled output stream (e.g. ``"[out]"``).
        tag:       Tag prefix for internal labels (default "_rb" creates ``[_rbbg]``, etc).
                   Custom tags keep multiple sub-graph instances collision-free.
    """
    # Internal scratch labels — the tag keeps multiple instances of this
    # sub-graph (one per zoom_hybrid segment) collision-free in one graph.
    bg_split = f"[{tag}bg]"
    fg_split = f"[{tag}fg]"
    bg_blurred = f"[{tag}bl]"
    fg_scaled = f"[{tag}fl]"

    parts = [
        # 1. Split input into background and foreground copies.
        f"{in_label}split=2{bg_split}{fg_split}",
        # 2. Background: scale to COVER the canvas (may overshoot), crop exactly to canvas,
        #    then apply a strong Gaussian blur so it reads as a soft colour fill.
        (
            f"{bg_split}scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
            f"crop={out_w}:{out_h},"
            f"boxblur=20:2{bg_blurred}"
        ),
        # 3. Foreground: scale to FIT within the canvas (never overflows).
        (
            f"{fg_split}scale={out_w}:{out_h}:force_original_aspect_ratio=decrease{fg_scaled}"
        ),
        # 4. Overlay foreground centred on blurred background.
        f"{bg_blurred}{fg_scaled}overlay=(W-w)/2:(H-h)/2{out_label}",
    ]
    return ";".join(parts)


def kennzeichnung_text(arten: set[str]) -> str | None:
    """Hinweistext aus den KI-Arten eines Videos (``kennzeichnung="auto"``).

    ``arten`` kommt aus ``render.handlers._synthetic_kinds``: ``stimme`` (KI-Stimme),
    ``gesicht`` (Reenact/Lipsync), ``synthetisch`` (alles andere Kuenstliche). Alles
    Unbestimmte gewinnt den allgemeinen Hinweis; ohne KI-Anteil gibt es keinen.
    """
    if not arten:
        return None
    if "synthetisch" in arten:
        return "KI · synthetisch"
    if arten == {"stimme"}:
        return "Stimme: KI"
    if arten == {"gesicht"}:
        return "Gesicht: KI"
    return "Gesicht und Stimme: KI"


#: Fallback-Schriften in dieser Reihenfolge, wenn LAURA_FONT fehlt. Die erste VORHANDENE
#: gewinnt. Vorher stand hier nur die Windows-Arial - auf der Linux-VM (keine LAURA_FONT
#: gesetzt, 17.10.2026 gemessen) liess das jeden drawtext (Hook, KI-Hinweis) scheitern.
_FONT_KANDIDATEN: tuple[str, ...] = (
    r"C:/Windows/Fonts/arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
)


def _drawtext_pfad(pfad: str) -> str:
    # Windows-Laufwerksdoppelpunkt als DOPPELTER Backslash (bewiesen: ``C\\:/Windows/...``
    # geht, einfacher Backslash nicht); Pfade ohne Laufwerk bleiben unveraendert.
    return pfad.replace(":", r"\\:", 1) if len(pfad) > 1 and pfad[1] == ":" else pfad


def resolve_font() -> str:
    """ffmpeg-drawtext-safe font path. LAURA_FONT env overrides; sonst die erste vorhandene
    Schrift aus ``_FONT_KANDIDATEN`` (Windows-Arial, dann DejaVu unter Linux)."""
    import os

    gesetzt = os.environ.get("LAURA_FONT")
    if gesetzt:
        return _drawtext_pfad(gesetzt)
    for kandidat in _FONT_KANDIDATEN:
        if os.path.isfile(kandidat):
            return _drawtext_pfad(kandidat)
    return _drawtext_pfad(_FONT_KANDIDATEN[0])
