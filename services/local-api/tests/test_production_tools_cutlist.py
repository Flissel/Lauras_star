"""production_tools: script voice synthesis (hash cache) + deterministic cutlist builder
(Slice 3, Task 5).

DB fixture mirrors ``tests/test_production_tools_review.py``'s ``_seed_scene`` but with TWO
scenes (each ``SCENE_FRAMES`` long) so the cutlist tests have more than one segment to order.
No analysis run / transcript is seeded — ``build_cutlist`` only needs each scene's SOURCE frame
range (via ``_resolve_scene`` -> ``context._scene_src_ranges``), which degrades to an empty
transcript text without a run; that text is never read by these tools.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from laura.config import Settings
from laura.db import repos
from laura.db.database import Database, SqliteDatabase
from laura.short_creator.board import Board
from laura.short_creator.board_models import (
    BestWindow,
    BoardMeta,
    Chapter,
    Cutlist,
    Roi,
    SceneReview,
    SceneWindowRef,
    Script,
    ScriptLine,
    Storyline,
    VoiceArtifact,
)
from laura.short_creator.production_tools import (
    ProductionDeps,
    _scale_chapter_durations,
    _segment_duration_s,
    build_production_tool_specs,
    chapter_audio_windows,
    line_starts,
    script_hash,
    script_text,
)

FPS = 30
SCENE_FRAMES = 300  # 300 frames @ 30fps = 10.0s per scene
N_SCENES = 2


def _seed_two_scenes(tmp_path: Path) -> tuple[Database, str]:
    """Project + asset + a TWO-scene rough cut. One lane-0 clip spans both scenes 1:1
    (SOURCE == SEQUENCE, speed 1): scene 1 is source frames [0, SCENE_FRAMES), scene 2 is
    [SCENE_FRAMES, 2*SCENE_FRAMES)."""
    settings = Settings(workspace_root=tmp_path / "ws", start_runner=False)
    db: Database = SqliteDatabase(settings.db_path)
    db.migrate()
    project = repos.create_project(
        db,
        name="p",
        rate_num=FPS,
        rate_den=1,
        drop_frame=False,
        workspace_root=str(tmp_path / "ws" / "proj"),
    )
    asset = repos.create_asset(
        db,
        project_id=project["id"],
        type="video",
        display_name="a.mp4",
        source_path=str(tmp_path / "a.mp4"),
    )
    timeline = repos.create_timeline(
        db,
        project_id=project["id"],
        name="Rough Cut",
        kind="rough_cut",
        created_from=asset["id"],
    )
    repos.add_timeline_clip(
        db,
        timeline_id=timeline["id"],
        asset_id=asset["id"],
        src_in_frame=0,
        src_out_frame_exclusive=SCENE_FRAMES * N_SCENES,
        seq_in_frame=0,
        seq_out_frame_exclusive=SCENE_FRAMES * N_SCENES,
        lane=0,
        role="base",
    )
    repos.replace_scenes(
        db,
        project["id"],
        timeline["id"],
        [(i * SCENE_FRAMES, (i + 1) * SCENE_FRAMES) for i in range(N_SCENES)],
    )
    return db, str(asset["id"])


def _board(tmp_path: Path, asset_id: str) -> Board:
    meta = BoardMeta(
        session_id="s1",
        asset_id=asset_id,
        created_utc="2026-07-13T00:00:00Z",
        task="overview short",
        target_seconds=20.0,
    )
    return Board.create(tmp_path / "board", meta)


def _review(
    board: Board,
    scene_number: int,
    *,
    best_window: BestWindow | None = None,
    roi: Roi | None = None,
    windows: list[BestWindow] | None = None,
) -> None:
    """Write a minimal valid SceneReview straight to the board (build_cutlist reads
    windows/roi from these, not from the tool's own re-derivation)."""
    board.save_scene_review(
        SceneReview(
            scene_number=scene_number,
            src_start_frame=(scene_number - 1) * SCENE_FRAMES,
            src_end_frame_exclusive=scene_number * SCENE_FRAMES,
            description="d",
            whats_happening="h",
            hook_score=5,
            best_window=best_window
            if best_window is not None
            else BestWindow(offset_s=0.0, duration_s=2.0),
            windows=windows if windows is not None else [],
            roi=roi,
        )
    )


def _storyline(
    *, scene_numbers: list[int | SceneWindowRef] | None = None, target_seconds: float = 4.0
) -> Storyline:
    return Storyline(
        red_thread="stop scrolling",
        arc=[
            Chapter(
                chapter=1,
                role="hook",
                message="stop scrolling",
                scene_numbers=scene_numbers if scene_numbers is not None else [1, 2],
                target_seconds=target_seconds,
            )
        ],
    )


def _script() -> Script:
    return Script(
        language="de",
        lines=[
            ScriptLine(chapter=1, scene_number=1, text="Stopp dein Team"),
            ScriptLine(chapter=1, scene_number=2, text="Ein Klick genügt"),
        ],
    )


def _words() -> list[dict[str, Any]]:
    """Word timings for ``_script()``'s six whitespace tokens (same numbers as
    ``test_synthesize_uses_cache_on_same_hash``) — reused by the provenance stamp tests."""
    return [
        {"text": "Stopp", "start_s": 0.0, "end_s": 0.3},
        {"text": "dein", "start_s": 0.3, "end_s": 0.6},
        {"text": "Team", "start_s": 0.6, "end_s": 1.0},
        {"text": "Ein", "start_s": 1.2, "end_s": 1.4},
        {"text": "Klick", "start_s": 1.4, "end_s": 1.7},
        {"text": "genügt", "start_s": 1.7, "end_s": 2.1},
    ]


class _FakeVoiceBackend:
    """Fake VoiceBackend: writes a dummy mp3 + a timings sidecar with caller-supplied word
    times (so the cutlist zoom test can hand-compute expected values), and counts calls."""

    def __init__(
        self, *, words: list[dict[str, Any]] | None = None, ok: bool = True, reason: str = "boom"
    ) -> None:
        self.calls = 0
        self.last_text: str | None = None
        self._words = words if words is not None else []
        self._ok = ok
        self._reason = reason

    def synthesize(self, text: str, out_path: Path) -> dict[str, Any]:
        self.calls += 1
        self.last_text = text
        if not self._ok:
            return {"ok": False, "reason": self._reason}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"id3-fake-mp3")
        timings_path = Path(str(out_path) + ".timings.json")
        timings_path.write_text(json.dumps({"words": self._words}), encoding="utf-8")
        return {"ok": True, "path": str(out_path), "timings_path": str(timings_path)}


def _save_voice(board: Board, tmp_path: Path, words: list[dict[str, Any]]) -> None:
    """Seed the board's voice artifact directly with a real sidecar file on disk (as
    build_cutlist reads it from ``timings_path``, not from the fake backend).

    The hash is computed from the script actually on the board — build_cutlist now refuses a
    voice whose script_hash disagrees with the current script (a reverted voice would poison
    the downstream provenance to stale=False), so the seed must satisfy the real contract.
    """
    from laura.short_creator.board_models import Storyline, lines_in_storyline_order

    timings_path = tmp_path / "voice.mp3.timings.json"
    timings_path.write_text(json.dumps({"words": words}), encoding="utf-8")
    script = board.load("script")
    storyline = board.load("storyline")
    assert isinstance(script, Script), "seed the script before the voice"
    lines = (
        lines_in_storyline_order(script, storyline)
        if isinstance(storyline, Storyline)
        else script.lines
    )
    board.save(
        "voice",
        VoiceArtifact(
            script_hash=script_hash(lines),
            mp3_path=str(tmp_path / "voice.mp3"),
            timings_path=str(timings_path),
        ),
    )


def test_script_text_and_hash_stable() -> None:
    script = Script(
        language="de",
        lines=[
            ScriptLine(chapter=1, scene_number=1, text="Stopp dein Team"),
            ScriptLine(chapter=1, scene_number=2, text="Ein Klick genügt"),
            ScriptLine(chapter=2, scene_number=3, text="Und fertig"),
        ],
    )

    text = script_text(script.lines)

    assert text == "Stopp dein Team Ein Klick genügt Und fertig"

    h1 = script_hash(script.lines)
    h2 = script_hash(script.model_copy(deep=True).lines)
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex digest

    changed = script.model_copy(
        update={"lines": [*script.lines, ScriptLine(chapter=2, scene_number=4, text="!")]}
    )
    assert script_hash(changed.lines) != h1


def test_line_starts_maps_tokens_to_word_times() -> None:
    script = Script(
        language="de",
        lines=[
            ScriptLine(chapter=1, scene_number=1, text="Stopp dein Team"),
            ScriptLine(chapter=1, scene_number=2, text="Ein Klick"),
        ],
    )
    words = [
        {"text": "Stopp", "start_s": 0.0, "end_s": 0.3},
        {"text": "dein", "start_s": 0.4, "end_s": 0.7},
        {"text": "Team", "start_s": 0.8, "end_s": 1.3},
        {"text": "Ein", "start_s": 1.5, "end_s": 1.7},
        {"text": "Klick", "start_s": 1.9, "end_s": 2.3},
    ]

    starts = line_starts(script.lines, words)

    assert starts == {(1, 1): 0.0, (1, 2): 1.5}


def test_synthesize_uses_cache_on_same_hash(tmp_path: Path) -> None:
    db, asset_id = _seed_two_scenes(tmp_path)
    board = _board(tmp_path, asset_id)
    board.save("storyline", _storyline())
    board.save("script", _script())
    backend = _FakeVoiceBackend(
        words=[
            {"text": "Stopp", "start_s": 0.0, "end_s": 0.3},
            {"text": "dein", "start_s": 0.3, "end_s": 0.6},
            {"text": "Team", "start_s": 0.6, "end_s": 1.0},
            {"text": "Ein", "start_s": 1.2, "end_s": 1.4},
            {"text": "Klick", "start_s": 1.4, "end_s": 1.7},
            {"text": "genügt", "start_s": 1.7, "end_s": 2.1},
        ]
    )
    deps = ProductionDeps(voice_backend=backend)
    specs = {
        s.name: s for s in build_production_tool_specs(db, board, asset_id=asset_id, deps=deps)
    }

    first = specs["synthesize_script_voice"].func()
    assert first["ok"] is True
    assert first.get("cached") is False
    assert first["voice_s"] == pytest.approx(2.1)
    assert backend.calls == 1

    second = specs["synthesize_script_voice"].func()
    assert second["ok"] is True
    assert second["cached"] is True
    assert backend.calls == 1  # no second synthesis — same script hash

    saved = board.load("voice")
    assert isinstance(saved, VoiceArtifact)
    # default _storyline() scene_numbers=[1, 2] matches _script()'s natural line order, so the
    # storyline-ordered hash equals the raw line-order hash here.
    assert saved.script_hash == script_hash(_script().lines)


def test_synthesize_reports_backend_failure(tmp_path: Path) -> None:
    db, asset_id = _seed_two_scenes(tmp_path)
    board = _board(tmp_path, asset_id)
    board.save("storyline", _storyline())
    board.save("script", _script())
    backend = _FakeVoiceBackend(ok=False, reason="quota exceeded")
    deps = ProductionDeps(voice_backend=backend)
    specs = {
        s.name: s for s in build_production_tool_specs(db, board, asset_id=asset_id, deps=deps)
    }

    out = specs["synthesize_script_voice"].func()

    assert out == {"ok": False, "reason": "quota exceeded"}
    assert board.load("voice") is None


def test_build_cutlist_requires_prereqs(tmp_path: Path) -> None:
    db, asset_id = _seed_two_scenes(tmp_path)
    board = _board(tmp_path, asset_id)
    specs = {s.name: s for s in build_production_tool_specs(db, board, asset_id=asset_id)}

    out = specs["build_cutlist"].func()
    assert out["ok"] is False
    assert "storyline" in out["reason"]

    board.save("storyline", _storyline())
    out = specs["build_cutlist"].func()
    assert out["ok"] is False
    assert "script" in out["reason"]

    board.save("script", _script())
    out = specs["build_cutlist"].func()
    assert out["ok"] is False
    assert "synthesize" in out["reason"]


def test_build_cutlist_deterministic_segments_and_zoom(tmp_path: Path) -> None:
    db, asset_id = _seed_two_scenes(tmp_path)
    board = _board(tmp_path, asset_id)
    _review(
        board,
        1,
        best_window=BestWindow(offset_s=1.0, duration_s=3.0),
        roi=Roi(x=0.1, y=0.1, w=0.2, h=0.2),
    )
    _review(
        board,
        2,
        best_window=BestWindow(offset_s=0.0, duration_s=3.0),
        roi=Roi(x=0.3, y=0.3, w=0.2, h=0.2),
    )
    board.save("storyline", _storyline(scene_numbers=[1, 2], target_seconds=4.0))
    board.save("script", _script())
    # 6 words for the script's 6 whitespace tokens ("Stopp dein Team" + "Ein Klick genügt");
    # first word of line 1 starts at 0.2s, first word of line 2 starts at 2.5s.
    _save_voice(
        board,
        tmp_path,
        words=[
            {"text": "Stopp", "start_s": 0.2, "end_s": 0.45},
            {"text": "dein", "start_s": 0.5, "end_s": 0.75},
            {"text": "Team", "start_s": 0.9, "end_s": 1.2},
            {"text": "Ein", "start_s": 2.5, "end_s": 2.75},
            {"text": "Klick", "start_s": 2.8, "end_s": 3.0},
            {"text": "genügt", "start_s": 3.1, "end_s": 3.4},
        ],
    )
    specs = {s.name: s for s in build_production_tool_specs(db, board, asset_id=asset_id)}

    out = specs["build_cutlist"].func()

    # target_seconds(4.0)/2 scenes = 2.0s budget/scene; both best_windows are 3.0s (>2.0s) so
    # the upper bound is 3.0s -> seg_dur clamps to the 2.0s floor -> 60 frames @ 30fps each.
    assert out == {"ok": True, "segments": 2, "total_seconds": 4.0, "with_zoom": 2}

    cutlist = board.load("cutlist")
    assert isinstance(cutlist, Cutlist)
    seg0, seg1 = cutlist.segments

    assert (seg0.order, seg0.scene_number) == (0, 1)
    # start = src_start(0) + offset_s(1.0)*30fps = 30; end = 30 + 60 frames = 90.
    assert (seg0.start_frame, seg0.end_frame_exclusive) == (30, 90)
    # video_start for the FIRST segment is 0.0 -> zoom = max(0, 0.2 - 0.0 + 0.4) = 0.6.
    assert seg0.zoom_start_s == pytest.approx(0.6)

    assert (seg1.order, seg1.scene_number) == (1, 2)
    # start = src_start(300) + offset_s(0.0)*30fps = 300; end = 300 + 60 frames = 360.
    assert (seg1.start_frame, seg1.end_frame_exclusive) == (300, 360)
    seg1_dauer = (seg0.end_frame_exclusive - seg0.start_frame) / FPS  # segment 1's own duration
    assert seg1_dauer == pytest.approx(2.0)
    # zoom_start_s of segment 2 = line_start - seg1_dauer + transition_lead_s (hand-computed).
    assert seg1.zoom_start_s == pytest.approx(2.5 - seg1_dauer + 0.4)
    assert seg1.zoom_start_s == pytest.approx(0.9)


def test_build_cutlist_clamps_inside_scene(tmp_path: Path) -> None:
    db, asset_id = _seed_two_scenes(tmp_path)
    board = _board(tmp_path, asset_id)
    # best_window sits right at the scene's end (10.0s scene, offset 9.0s + 2.0s segment would
    # run 1.0s past src_end without clamping).
    _review(
        board,
        1,
        best_window=BestWindow(offset_s=9.0, duration_s=3.0),
        roi=None,
    )
    board.save("storyline", _storyline(scene_numbers=[1], target_seconds=2.0))
    board.save("script", _script())
    _save_voice(board, tmp_path, words=[])
    specs = {s.name: s for s in build_production_tool_specs(db, board, asset_id=asset_id)}

    out = specs["build_cutlist"].func()

    assert out["ok"] is True
    assert out["segments"] == 1
    cutlist = board.load("cutlist")
    assert isinstance(cutlist, Cutlist)
    seg = cutlist.segments[0]
    # raw start would be 0 + 9.0*30 = 270; 270 + 60 frames = 330 > src_end(300) -> clamped so
    # start_frame + dur_frames == src_end exactly.
    assert seg.start_frame == 240
    assert seg.end_frame_exclusive == 300
    assert seg.end_frame_exclusive <= SCENE_FRAMES
    assert seg.zoom_start_s is None  # roi=None -> no zoom regardless of timing


def test_storyline_order_drives_voice_text_and_zoom_timing(tmp_path: Path) -> None:
    """Regression test for the A/V-sync bug: the script's lines are WRITTEN scene 1 first, then
    scene 2 (``_script()``'s natural order), but the storyline PLAYS scene 2 first, then scene 1
    (``scene_numbers=[2, 1]``) — a scrambled chapter. Voice text, the word-start map and the
    cutlist's per-segment zoom must all follow the STORYLINE's order, not the script's raw list
    order, or narration/captions/zoom desync from the picture.

    Also covers the voice cache: it must invalidate when ONLY the storyline's scene order
    changes (a re-save of the storyline is downstream-invalidating for script/voice/cutlist per
    board.py's ``_CHAIN``, so the script has to be re-saved too before synthesizing again)."""
    db, asset_id = _seed_two_scenes(tmp_path)
    board = _board(tmp_path, asset_id)
    _review(
        board,
        1,
        best_window=BestWindow(offset_s=1.0, duration_s=3.0),
        roi=Roi(x=0.1, y=0.1, w=0.2, h=0.2),
    )
    _review(
        board,
        2,
        best_window=BestWindow(offset_s=0.0, duration_s=3.0),
        roi=Roi(x=0.3, y=0.3, w=0.2, h=0.2),
    )
    # Storyline plays scene 2 BEFORE scene 1 ...
    board.save("storyline", _storyline(scene_numbers=[2, 1], target_seconds=4.0))
    # ... but the script's lines were written scene 1 first, scene 2 second (scrambled vs the
    # storyline — _script() is unchanged from every other test in this file).
    board.save("script", _script())

    # Word timings as a real TTS backend would produce them for the CORRECTLY (storyline-)
    # ordered text "Ein Klick genügt Stopp dein Team" (scene 2's line, then scene 1's line).
    backend = _FakeVoiceBackend(
        words=[
            {"text": "Ein", "start_s": 0.2, "end_s": 0.45},
            {"text": "Klick", "start_s": 0.5, "end_s": 0.75},
            {"text": "genügt", "start_s": 0.8, "end_s": 1.1},
            {"text": "Stopp", "start_s": 2.5, "end_s": 2.75},
            {"text": "dein", "start_s": 2.8, "end_s": 3.0},
            {"text": "Team", "start_s": 3.1, "end_s": 3.4},
        ]
    )
    deps = ProductionDeps(voice_backend=backend)
    specs = {
        s.name: s for s in build_production_tool_specs(db, board, asset_id=asset_id, deps=deps)
    }

    synth = specs["synthesize_script_voice"].func()
    assert synth["ok"] is True
    assert synth["cached"] is False
    assert backend.calls == 1
    assert backend.last_text == "Ein Klick genügt Stopp dein Team"

    built = specs["build_cutlist"].func()
    assert built["ok"] is True

    cutlist = board.load("cutlist")
    assert isinstance(cutlist, Cutlist)
    seg0, seg1 = cutlist.segments

    # Video order follows the STORYLINE (scene 2 first, scene 1 second) ...
    assert (seg0.order, seg0.scene_number) == (0, 2)
    assert (seg1.order, seg1.scene_number) == (1, 1)
    # ... and the zoom timing follows that SAME order: scene 2's zoom uses the EARLY word start
    # (its line is spoken first), scene 1's zoom uses the LATE one (its line is spoken second) —
    # the exact opposite of what the naive (chapter, list-position) order would have produced
    # (which would have paired scene 1 with the early words and scene 2 with the late ones).
    assert seg0.zoom_start_s == pytest.approx(0.6)
    assert seg1.zoom_start_s == pytest.approx(0.9)

    # The voice cache is keyed on the ORDERED text: re-saving the storyline with a DIFFERENT
    # scene order changes that text (even though the script itself is untouched) and must
    # invalidate the cache. save_storyline invalidates script downstream (board.py's _CHAIN), so
    # the script has to be re-saved before synthesizing again.
    board.save("storyline", _storyline(scene_numbers=[1, 2], target_seconds=4.0))
    assert board.load("script") is None  # confirms the downstream invalidation actually fired
    board.save("script", _script())

    second = specs["synthesize_script_voice"].func()
    assert second["ok"] is True
    assert second["cached"] is False  # order-only change still busts the cache
    assert backend.calls == 2
    assert backend.last_text == "Stopp dein Team Ein Klick genügt"


def test_chapters_narrated_in_numeric_order_not_save_order(tmp_path: Path) -> None:
    """Regression test: when a storyline arc is saved out of numeric order (e.g., [chapter 2,
    chapter 1]), voice synthesis must follow numeric chapter order (1, 2), not save order. This
    ensures audio/video/zoom stay in sync during playback."""
    db, asset_id = _seed_two_scenes(tmp_path)
    board = _board(tmp_path, asset_id)

    # Reviews for both scenes
    _review(
        board,
        1,
        best_window=BestWindow(offset_s=0.0, duration_s=3.0),
        roi=Roi(x=0.1, y=0.1, w=0.2, h=0.2),
    )
    _review(
        board,
        2,
        best_window=BestWindow(offset_s=0.0, duration_s=3.0),
        roi=Roi(x=0.3, y=0.3, w=0.2, h=0.2),
    )

    # Storyline with TWO chapters saved OUT OF NUMERIC ORDER: arc=[chapter 2, chapter 1]
    # Voice synthesis must still follow numeric order: chapter 1 first, then chapter 2.
    board.save(
        "storyline",
        Storyline(
            red_thread="multi-chapter test",
            arc=[
                Chapter(
                    chapter=2,
                    role="payoff_cta",
                    message="chapter 2",
                    scene_numbers=[2],
                    target_seconds=2.0,
                ),
                Chapter(
                    chapter=1,
                    role="hook",
                    message="chapter 1",
                    scene_numbers=[1],
                    target_seconds=2.0,
                ),
            ],
        ),
    )

    # Script with lines for both chapters (in natural scene order)
    board.save(
        "script",
        Script(
            language="de",
            lines=[
                ScriptLine(chapter=1, scene_number=1, text="First chapter"),
                ScriptLine(chapter=2, scene_number=2, text="Second chapter"),
            ],
        ),
    )

    # Voice backend with words for CORRECTLY ORDERED text: "First chapter Second chapter"
    backend = _FakeVoiceBackend(
        words=[
            {"text": "First", "start_s": 0.0, "end_s": 0.3},
            {"text": "chapter", "start_s": 0.3, "end_s": 0.6},
            {"text": "Second", "start_s": 1.0, "end_s": 1.3},
            {"text": "chapter", "start_s": 1.3, "end_s": 1.6},
        ]
    )
    deps = ProductionDeps(voice_backend=backend)
    specs = {
        s.name: s for s in build_production_tool_specs(db, board, asset_id=asset_id, deps=deps)
    }

    # Synthesize voice: backend must receive text in NUMERIC chapter order, not save order
    synth = specs["synthesize_script_voice"].func()
    assert synth["ok"] is True
    assert backend.last_text == "First chapter Second chapter"

    # Build cutlist: must follow numeric order, not save order
    built = specs["build_cutlist"].func()
    assert built["ok"] is True

    cutlist = board.load("cutlist")
    assert isinstance(cutlist, Cutlist)
    assert len(cutlist.segments) == 2

    # First segment in cutlist is chapter 1 (scene 1), not chapter 2 (despite save order)
    seg0 = cutlist.segments[0]
    assert seg0.scene_number == 1

    # Second segment is chapter 2 (scene 2)
    seg1 = cutlist.segments[1]
    assert seg1.scene_number == 2


# --- chapter audio windows (pure) --------------------------------------------------------------


def test_chapter_audio_windows_midpoint_boundary_and_tail() -> None:
    """Two chapters: chapter 1 spans [0, midpoint), the boundary is the middle between chapter
    1's last word end and chapter 2's first word start, and the LAST chapter runs to the voice
    end plus the ~0.6s tail."""
    lines = [
        ScriptLine(chapter=1, scene_number=1, text="a b c"),
        ScriptLine(chapter=1, scene_number=2, text="d e"),
        ScriptLine(chapter=2, scene_number=3, text="f g"),
    ]
    words = [
        {"text": "a", "start_s": 0.0, "end_s": 0.4},
        {"text": "b", "start_s": 0.5, "end_s": 0.9},
        {"text": "c", "start_s": 1.0, "end_s": 1.4},
        {"text": "d", "start_s": 1.6, "end_s": 2.0},
        {"text": "e", "start_s": 2.2, "end_s": 3.0},
        {"text": "f", "start_s": 3.6, "end_s": 4.0},
        {"text": "g", "start_s": 4.2, "end_s": 5.0},
    ]

    windows = chapter_audio_windows(lines, words)

    assert set(windows) == {1, 2}
    # boundary = mid(3.0, 3.6) = 3.3; last chapter ends at voice_end(5.0) + 0.6 tail.
    assert windows[1] == pytest.approx((0.0, 3.3))
    assert windows[2] == pytest.approx((3.3, 5.6))


def test_chapter_audio_windows_empty_words() -> None:
    lines = [ScriptLine(chapter=1, scene_number=1, text="a b")]
    assert chapter_audio_windows(lines, []) == {}


def test_chapter_audio_windows_sidecar_runs_short() -> None:
    """A sidecar shorter than the script: chapters whose lines got no words have NO window (the
    caller falls back to the target_seconds budget for them); the covered chapter still ends at
    the stream's voice end + tail."""
    lines = [
        ScriptLine(chapter=1, scene_number=1, text="a b c"),
        ScriptLine(chapter=2, scene_number=2, text="d e"),
    ]
    words = [
        {"text": "a", "start_s": 0.0, "end_s": 0.4},
        {"text": "b", "start_s": 0.5, "end_s": 0.9},
        {"text": "c", "start_s": 1.0, "end_s": 1.4},
    ]

    windows = chapter_audio_windows(lines, words)

    assert set(windows) == {1}
    assert windows[1] == pytest.approx((0.0, 2.0))


# --- proportional duration scaling (pure) -------------------------------------------------------


def test_scale_chapter_durations_proportional() -> None:
    assert _scale_chapter_durations([2.0, 2.0], [10.0, 10.0], 8.0) == [
        pytest.approx(4.0),
        pytest.approx(4.0),
    ]
    # weights follow the base durations: 2:4 -> 3:6 for a 9s window.
    assert _scale_chapter_durations([2.0, 4.0], [10.0, 10.0], 9.0) == [
        pytest.approx(3.0),
        pytest.approx(6.0),
    ]


def test_scale_chapter_durations_cap_redistributes() -> None:
    # item 0 hits its 3.0s cap; the released time goes to item 1 so the sum stays 8.0.
    assert _scale_chapter_durations([2.0, 2.0], [3.0, 10.0], 8.0) == [
        pytest.approx(3.0),
        pytest.approx(5.0),
    ]


def test_scale_chapter_durations_saturates_at_floors_and_caps() -> None:
    # chapter audio shorter than n*2s floor -> stay at the floors, accept the overhang.
    assert _scale_chapter_durations([2.0, 2.0], [10.0, 10.0], 3.0) == [
        pytest.approx(2.0),
        pytest.approx(2.0),
    ]
    # chapter audio longer than the summed caps -> saturate at the caps, accept the shortfall.
    assert _scale_chapter_durations([3.0], [8.0], 9.5) == [pytest.approx(8.0)]
    assert _scale_chapter_durations([], [], 5.0) == []


# --- build_cutlist coupled to chapter audio windows ---------------------------------------------


def test_build_cutlist_couples_segment_durations_to_chapter_audio(tmp_path: Path) -> None:
    """The A/V drift fix: chapter word-shares != chapter time-shares. Both chapters have the
    same target_seconds (2.0), but chapter 1's audio runs to ~5s while chapter 2's is short —
    the video's chapter boundary must land exactly on the AUDIO boundary, not on the
    target_seconds split, and a segment may stretch past its best_window.duration_s to do so."""
    db, asset_id = _seed_two_scenes(tmp_path)
    board = _board(tmp_path, asset_id)
    _review(
        board,
        1,
        best_window=BestWindow(offset_s=0.0, duration_s=3.0),
        roi=Roi(x=0.1, y=0.1, w=0.2, h=0.2),
    )
    _review(
        board,
        2,
        best_window=BestWindow(offset_s=0.0, duration_s=3.0),
        roi=Roi(x=0.3, y=0.3, w=0.2, h=0.2),
    )
    board.save(
        "storyline",
        Storyline(
            red_thread="drift fix",
            arc=[
                Chapter(
                    chapter=1,
                    role="hook",
                    message="c1",
                    scene_numbers=[1],
                    target_seconds=2.0,
                ),
                Chapter(
                    chapter=2,
                    role="payoff_cta",
                    message="c2",
                    scene_numbers=[2],
                    target_seconds=2.0,
                ),
            ],
        ),
    )
    board.save(
        "script",
        Script(
            language="de",
            lines=[
                ScriptLine(chapter=1, scene_number=1, text="Stopp dein Team"),
                ScriptLine(chapter=2, scene_number=2, text="Ein Klick genügt"),
            ],
        ),
    )
    # chapter 1's words run to 4.9s, chapter 2's start at 5.5s -> audio boundary at 5.2s;
    # voice ends 7.4s -> chapter 2's window ends 8.0s (voice end + 0.6 tail).
    _save_voice(
        board,
        tmp_path,
        words=[
            {"text": "Stopp", "start_s": 0.2, "end_s": 0.45},
            {"text": "dein", "start_s": 0.5, "end_s": 0.75},
            {"text": "Team", "start_s": 0.9, "end_s": 4.9},
            {"text": "Ein", "start_s": 5.5, "end_s": 5.75},
            {"text": "Klick", "start_s": 5.8, "end_s": 6.0},
            {"text": "genügt", "start_s": 6.1, "end_s": 7.4},
        ],
    )
    specs = {s.name: s for s in build_production_tool_specs(db, board, asset_id=asset_id)}

    out = specs["build_cutlist"].func()

    assert out == {"ok": True, "segments": 2, "total_seconds": 8.0, "with_zoom": 2}

    cutlist = board.load("cutlist")
    assert isinstance(cutlist, Cutlist)
    seg0, seg1 = cutlist.segments

    # chapter 1's segment fills its 5.2s audio window (156 frames @ 30fps) — stretched well past
    # its best_window.duration_s of 3.0s, from the best_window offset.
    assert (seg0.start_frame, seg0.end_frame_exclusive) == (0, 156)
    # chapter 2's segment starts in the video exactly at the audio boundary ...
    video_boundary_s = (seg0.end_frame_exclusive - seg0.start_frame) / FPS
    assert video_boundary_s == pytest.approx(5.2)
    # ... and fills the remaining 2.8s window (84 frames).
    assert (seg1.start_frame, seg1.end_frame_exclusive) == (300, 384)

    # the zoom still lands where the words land: chapter 2's line starts 0.3s into its segment.
    assert seg0.zoom_start_s == pytest.approx(0.6)
    assert seg1.zoom_start_s == pytest.approx(0.7)


def test_build_cutlist_floors_when_chapter_audio_shorter_than_scenes(tmp_path: Path) -> None:
    """Chapter audio (3.0s incl. tail) shorter than n_scenes * 2.0s floor: both segments stay at
    the 2s floor and the video simply runs 1s past the chapter's audio window."""
    db, asset_id = _seed_two_scenes(tmp_path)
    board = _board(tmp_path, asset_id)
    _review(board, 1, best_window=BestWindow(offset_s=1.0, duration_s=3.0))
    _review(board, 2, best_window=BestWindow(offset_s=0.0, duration_s=3.0))
    board.save("storyline", _storyline(scene_numbers=[1, 2], target_seconds=4.0))
    board.save("script", _script())
    # voice ends at 2.4s -> single chapter window [0, 3.0) < 2 scenes * 2s floor.
    _save_voice(
        board,
        tmp_path,
        words=[
            {"text": "Stopp", "start_s": 0.1, "end_s": 0.3},
            {"text": "dein", "start_s": 0.35, "end_s": 0.6},
            {"text": "Team", "start_s": 0.7, "end_s": 1.0},
            {"text": "Ein", "start_s": 1.2, "end_s": 1.5},
            {"text": "Klick", "start_s": 1.6, "end_s": 1.9},
            {"text": "genügt", "start_s": 2.0, "end_s": 2.4},
        ],
    )
    specs = {s.name: s for s in build_production_tool_specs(db, board, asset_id=asset_id)}

    out = specs["build_cutlist"].func()

    assert out["ok"] is True
    assert out["total_seconds"] == pytest.approx(4.0)  # 2 x 2s floor, NOT squeezed to 3.0
    cutlist = board.load("cutlist")
    assert isinstance(cutlist, Cutlist)
    seg0, seg1 = cutlist.segments
    assert (seg0.start_frame, seg0.end_frame_exclusive) == (30, 90)
    assert (seg1.start_frame, seg1.end_frame_exclusive) == (300, 360)


def test_build_cutlist_stretch_stops_at_scene_end_keeping_offset(tmp_path: Path) -> None:
    """A chapter audio window (9.5s) longer than the scene can host from its best_window offset
    (10s scene, offset 2s -> at most 8s): the segment keeps the offset as its start, stretches
    to the scene's end-exclusive boundary and NOT past it, and the shortfall is accepted."""
    db, asset_id = _seed_two_scenes(tmp_path)
    board = _board(tmp_path, asset_id)
    _review(board, 1, best_window=BestWindow(offset_s=2.0, duration_s=3.0))
    board.save("storyline", _storyline(scene_numbers=[1], target_seconds=4.0))
    board.save(
        "script",
        Script(
            language="de",
            lines=[ScriptLine(chapter=1, scene_number=1, text="Stopp dein Team")],
        ),
    )
    _save_voice(
        board,
        tmp_path,
        words=[
            {"text": "Stopp", "start_s": 0.2, "end_s": 0.5},
            {"text": "dein", "start_s": 0.6, "end_s": 1.0},
            {"text": "Team", "start_s": 1.2, "end_s": 8.9},
        ],
    )
    specs = {s.name: s for s in build_production_tool_specs(db, board, asset_id=asset_id)}

    out = specs["build_cutlist"].func()

    assert out["ok"] is True
    cutlist = board.load("cutlist")
    assert isinstance(cutlist, Cutlist)
    seg = cutlist.segments[0]
    # start stays at the best_window offset (frame 60), the end is the scene's boundary — the
    # segment is 8.0s, past its 3.0s best_window but never past the scene (frame invariant).
    assert (seg.start_frame, seg.end_frame_exclusive) == (60, 300)
    assert seg.end_frame_exclusive <= SCENE_FRAMES
    assert out["total_seconds"] == pytest.approx(8.0)


# --- storyline window references -----------------------------------------------------------------


def test_build_cutlist_uses_referenced_window(tmp_path: Path) -> None:
    """Storyline references window 1 of scene 1: the segment starts at THAT window's offset,
    its stretch cap runs from that offset to the scene end, and the segment's roi is the
    window's own roi (the review-level roi is only the fallback for window roi = None)."""
    db, asset_id = _seed_two_scenes(tmp_path)
    board = _board(tmp_path, asset_id)
    w0 = BestWindow(offset_s=0.0, duration_s=2.0)
    w1 = BestWindow(offset_s=5.0, duration_s=3.0, roi=Roi(x=0.3, y=0.3, w=0.2, h=0.2))
    _review(board, 1, best_window=w0, windows=[w0, w1], roi=Roi(x=0.1, y=0.1, w=0.2, h=0.2))
    board.save(
        "storyline",
        Storyline(
            red_thread="rt",
            arc=[
                Chapter(
                    chapter=1,
                    role="hook",
                    message="m",
                    scene_numbers=[SceneWindowRef(scene=1, window=1)],
                    target_seconds=4.0,
                )
            ],
        ),
    )
    board.save(
        "script",
        Script(
            language="de", lines=[ScriptLine(chapter=1, scene_number=1, text="Stopp dein Team")]
        ),
    )
    # chapter audio window [0, 9.5): longer than the 5s the scene can host from offset 5.0 ->
    # the segment stretches from the WINDOW's offset to the scene end (frames 150..300).
    _save_voice(
        board,
        tmp_path,
        words=[
            {"text": "Stopp", "start_s": 0.2, "end_s": 0.5},
            {"text": "dein", "start_s": 0.6, "end_s": 1.0},
            {"text": "Team", "start_s": 1.2, "end_s": 8.9},
        ],
    )
    specs = {s.name: s for s in build_production_tool_specs(db, board, asset_id=asset_id)}

    out = specs["build_cutlist"].func()

    assert out["ok"] is True
    cutlist = board.load("cutlist")
    assert isinstance(cutlist, Cutlist)
    seg = cutlist.segments[0]
    assert (seg.start_frame, seg.end_frame_exclusive) == (150, 300)
    assert seg.roi is not None and seg.roi.x == 0.3  # window roi wins over review roi


def test_build_cutlist_same_scene_twice_with_different_windows(tmp_path: Path) -> None:
    """One chapter plays scene 1 twice — window 0 (plain int entry) then window 1 — as two
    separate segments, each cut from its own window's offset."""
    db, asset_id = _seed_two_scenes(tmp_path)
    board = _board(tmp_path, asset_id)
    w0 = BestWindow(offset_s=1.0, duration_s=2.0)
    w1 = BestWindow(offset_s=6.0, duration_s=2.0)
    _review(board, 1, best_window=w0, windows=[w0, w1])
    board.save(
        "storyline",
        Storyline(
            red_thread="rt",
            arc=[
                Chapter(
                    chapter=1,
                    role="hook",
                    message="m",
                    scene_numbers=[1, SceneWindowRef(scene=1, window=1)],
                    target_seconds=4.0,
                )
            ],
        ),
    )
    board.save(
        "script",
        Script(
            language="de", lines=[ScriptLine(chapter=1, scene_number=1, text="Stopp dein Team")]
        ),
    )
    _save_voice(board, tmp_path, words=[])  # no sidecar -> plain target_seconds budget
    specs = {s.name: s for s in build_production_tool_specs(db, board, asset_id=asset_id)}

    out = specs["build_cutlist"].func()

    # 4.0s / 2 refs = 2.0s each -> 60 frames from each window's own offset.
    assert out == {"ok": True, "segments": 2, "total_seconds": 4.0, "with_zoom": 0}
    cutlist = board.load("cutlist")
    assert isinstance(cutlist, Cutlist)
    seg0, seg1 = cutlist.segments
    assert (seg0.scene_number, seg0.start_frame, seg0.end_frame_exclusive) == (1, 30, 90)
    assert (seg1.scene_number, seg1.start_frame, seg1.end_frame_exclusive) == (1, 180, 240)


def test_build_cutlist_rejects_out_of_range_window_ref(tmp_path: Path) -> None:
    """A storyline saved straight to the board (bypassing save_storyline's gate — e.g. written
    before a scene was re-reviewed down to fewer windows) must fail loud and correctable
    instead of silently cutting a different moment."""
    db, asset_id = _seed_two_scenes(tmp_path)
    board = _board(tmp_path, asset_id)
    _review(board, 1)  # a single window (0)
    board.save(
        "storyline",
        Storyline(
            red_thread="rt",
            arc=[
                Chapter(
                    chapter=1,
                    role="hook",
                    message="m",
                    scene_numbers=[SceneWindowRef(scene=1, window=2)],
                    target_seconds=4.0,
                )
            ],
        ),
    )
    board.save(
        "script",
        Script(
            language="de", lines=[ScriptLine(chapter=1, scene_number=1, text="Stopp dein Team")]
        ),
    )
    _save_voice(board, tmp_path, words=[])
    specs = {s.name: s for s in build_production_tool_specs(db, board, asset_id=asset_id)}

    out = specs["build_cutlist"].func()

    assert out["ok"] is False
    assert "window 2" in out["reason"] and "scene 1" in out["reason"]
    assert board.load("cutlist") is None


def test_build_cutlist_stamps_storyline_script_and_voice_parents(tmp_path: Path) -> None:
    from laura.short_creator.board_models import content_hash

    db, asset_id = _seed_two_scenes(tmp_path)
    board = _board(tmp_path, asset_id)
    board.save("storyline", _storyline())
    board.save("script", _script())
    _save_voice(board, tmp_path, _words())
    storyline = board.load("storyline")
    script = board.load("script")
    voice = board.load("voice")
    assert storyline is not None and script is not None and voice is not None
    specs = {s.name: s for s in build_production_tool_specs(db, board, asset_id=asset_id)}

    out = specs["build_cutlist"].func()
    assert out["ok"] is True, out

    cutlist = board.load("cutlist")
    assert isinstance(cutlist, Cutlist)
    assert cutlist.parents == {
        "storyline": content_hash(storyline),
        "script": content_hash(script),
        "voice": content_hash(voice),
    }


def test_synthesize_stamps_storyline_and_script_parents(tmp_path: Path) -> None:
    from laura.short_creator.board_models import content_hash

    db, asset_id = _seed_two_scenes(tmp_path)
    board = _board(tmp_path, asset_id)
    board.save("storyline", _storyline())
    board.save("script", _script())
    storyline = board.load("storyline")
    script = board.load("script")
    assert storyline is not None and script is not None
    deps = ProductionDeps(voice_backend=_FakeVoiceBackend(words=_words()))
    specs = {
        s.name: s for s in build_production_tool_specs(db, board, asset_id=asset_id, deps=deps)
    }

    out = specs["synthesize_script_voice"].func()
    assert out["ok"] is True, out

    voice = board.load("voice")
    assert isinstance(voice, VoiceArtifact)
    assert voice.parents == {
        "storyline": content_hash(storyline),
        "script": content_hash(script),
    }


def test_build_cutlist_refuses_a_voice_from_a_different_script(tmp_path: Path) -> None:
    """Review finding: build_cutlist presence-checked the voice and never asked WHOSE voice.

    The cut's audio windows and zoom timings come from that voice's timings sidecar, and the
    render muxes its mp3 as the film's audio. revert_artifact("voice", vOld) is a documented
    follow-up flow — after it, the current script and the voice disagree, and a cutlist built
    from the pair would carry the current script's hash while cutting to a different take:
    stale=False on a film whose narration is not the script. The mismatch must refuse loudly
    with the correcting tool named, exactly like every other order guard.
    """
    db, asset_id = _seed_two_scenes(tmp_path)
    board = _board(tmp_path, asset_id)
    board.save("storyline", _storyline())
    board.save("script", _script())
    timings_path = tmp_path / "old-take.mp3.timings.json"
    timings_path.write_text(json.dumps({"words": []}), encoding="utf-8")
    board.save(
        "voice",
        VoiceArtifact(
            script_hash="a-different-scripts-hash",
            mp3_path=str(tmp_path / "old-take.mp3"),
            timings_path=str(timings_path),
        ),
    )
    specs = {s.name: s for s in build_production_tool_specs(db, board, asset_id=asset_id)}

    out = specs["build_cutlist"].func()

    assert out["ok"] is False
    assert "synthesize_script_voice" in out["reason"]
    assert board.load("cutlist") is None


def test_segment_duration_ignores_window_length() -> None:
    """The decoupling (spec 2026-07-20-window-bias-design.md §2): a reel-trained reviewer's
    0.5s window must not shrink a held screen's weight below the chapter budget. Before the
    fix the window's duration capped the base (0.5s window -> 2.0s weight vs 8.0s for a long
    window in the same chapter)."""
    assert (
        _segment_duration_s(target_seconds=16.0, n_scenes=2, scene_duration_s=10.0) == 8.0
    )


def test_segment_duration_keeps_floor_and_scene_clamp() -> None:
    assert _segment_duration_s(target_seconds=1.0, n_scenes=2, scene_duration_s=10.0) == 2.0
    assert _segment_duration_s(target_seconds=60.0, n_scenes=2, scene_duration_s=4.5) == 4.5


def test_build_cutlist_gives_short_and_long_windows_equal_time(tmp_path: Path) -> None:
    """One chapter, two scenes: a 0.5s window (held screen, reel-scored) and an 8.0s window
    (motion). The chapter's 6.0s audio window must split EQUALLY (3.0s each) — before the
    decoupling the bases were [2.0, 8.0] and the split came out [2.0, 4.0]."""
    db, asset_id = _seed_two_scenes(tmp_path)
    board = _board(tmp_path, asset_id)
    _review(board, 1, best_window=BestWindow(offset_s=0.0, duration_s=0.5))
    _review(board, 2, best_window=BestWindow(offset_s=0.0, duration_s=8.0))
    board.save("storyline", _storyline(scene_numbers=[1, 2], target_seconds=16.0))
    board.save("script", _script())
    # All six words belong to chapter 1; the voice ends at 5.4s -> one chapter audio window
    # [0.0, 6.0) incl. the 0.6s tail. Bases [8.0, 8.0] scale by one factor to [3.0, 3.0].
    _save_voice(
        board,
        tmp_path,
        words=[
            {"text": "Stopp", "start_s": 0.0, "end_s": 0.3},
            {"text": "dein", "start_s": 0.3, "end_s": 0.6},
            {"text": "Team", "start_s": 0.6, "end_s": 1.0},
            {"text": "Ein", "start_s": 1.2, "end_s": 1.4},
            {"text": "Klick", "start_s": 1.4, "end_s": 1.7},
            {"text": "genügt", "start_s": 1.7, "end_s": 5.4},
        ],
    )
    specs = {s.name: s for s in build_production_tool_specs(db, board, asset_id=asset_id)}

    out = specs["build_cutlist"].func()

    assert out == {"ok": True, "segments": 2, "total_seconds": 6.0, "with_zoom": 0}
    cutlist = board.load("cutlist")
    assert isinstance(cutlist, Cutlist)
    seg0, seg1 = cutlist.segments
    # 3.0s each (90 frames @ 30fps), starting at each window's offset.
    assert (seg0.start_frame, seg0.end_frame_exclusive) == (0, 90)
    assert (seg1.start_frame, seg1.end_frame_exclusive) == (300, 390)


def test_build_cutlist_tool_description_teaches_offset_only_window_contract(
    tmp_path: Path,
) -> None:
    """The agent team reads ``ToolSpec.description`` (harvested from ``build_cutlist``'s
    docstring), not the source. It must not teach the removed semantics — a review window
    capping a segment's length — or the agent will keep believing the window IS the weight
    (spec 2026-07-20-window-bias-design.md §2)."""
    db, asset_id = _seed_two_scenes(tmp_path)
    board = _board(tmp_path, asset_id)
    specs = {s.name: s for s in build_production_tool_specs(db, board, asset_id=asset_id)}

    description = specs["build_cutlist"].description

    assert "base-duration cap" not in description
    assert "never from the window's duration" in description


# --- the zoom lever (live finding 2026-08-04) ----------------------------------------------------
# The user asked for the full frame ("zeig das volle Bild, kein enger Zoom"); rois derive from
# the STORYLINE's window references, and three follow-up runs failed to re-save the storyline —
# twice rebuilding the cutlist from the unchanged storyline. The workaround was stripping rois
# out of cutlist.json by hand. zoom="off" is that workaround as a first-class, one-call lever.


def test_build_cutlist_zoom_off_drops_all_rois_and_zoom(tmp_path: Path) -> None:
    """zoom="off": EVERY segment comes out roi-less and unzoomed, regardless of the storyline's
    window references — while the cut itself (window offsets, durations) stays exactly the one
    zoom="auto" builds. Same fixture numbers as
    ``test_build_cutlist_deterministic_segments_and_zoom`` (which pins the auto behaviour)."""
    db, asset_id = _seed_two_scenes(tmp_path)
    board = _board(tmp_path, asset_id)
    _review(
        board,
        1,
        best_window=BestWindow(offset_s=1.0, duration_s=3.0),
        roi=Roi(x=0.1, y=0.1, w=0.2, h=0.2),
    )
    _review(
        board,
        2,
        best_window=BestWindow(offset_s=0.0, duration_s=3.0),
        roi=Roi(x=0.3, y=0.3, w=0.2, h=0.2),
    )
    board.save("storyline", _storyline(scene_numbers=[1, 2], target_seconds=4.0))
    board.save("script", _script())
    _save_voice(
        board,
        tmp_path,
        words=[
            {"text": "Stopp", "start_s": 0.2, "end_s": 0.45},
            {"text": "dein", "start_s": 0.5, "end_s": 0.75},
            {"text": "Team", "start_s": 0.9, "end_s": 1.2},
            {"text": "Ein", "start_s": 2.5, "end_s": 2.75},
            {"text": "Klick", "start_s": 2.8, "end_s": 3.0},
            {"text": "genügt", "start_s": 3.1, "end_s": 3.4},
        ],
    )
    specs = {s.name: s for s in build_production_tool_specs(db, board, asset_id=asset_id)}

    out = specs["build_cutlist"].func(zoom="off")

    assert out["ok"] is True
    assert out["segments"] == 2
    assert out["with_zoom"] == 0
    assert "zoom" in str(out.get("note", "")).lower()  # the override is named in the reply

    cutlist = board.load("cutlist")
    assert isinstance(cutlist, Cutlist)
    seg0, seg1 = cutlist.segments
    assert seg0.roi is None and seg0.zoom_start_s is None
    assert seg1.roi is None and seg1.zoom_start_s is None
    # The CUT is untouched — same frames as the zoom="auto" run over these exact numbers.
    assert (seg0.start_frame, seg0.end_frame_exclusive) == (30, 90)
    assert (seg1.start_frame, seg1.end_frame_exclusive) == (300, 360)


def test_build_cutlist_zoom_off_overrides_window_refs(tmp_path: Path) -> None:
    """The live failure mode exactly: the storyline references a review window WITH a roi, and
    zoom="off" must still drop it — the whole point is that the storyline does not need to be
    re-saved without its window refs."""
    db, asset_id = _seed_two_scenes(tmp_path)
    board = _board(tmp_path, asset_id)
    w0 = BestWindow(offset_s=0.0, duration_s=2.0)
    w1 = BestWindow(offset_s=5.0, duration_s=3.0, roi=Roi(x=0.3, y=0.3, w=0.2, h=0.2))
    _review(board, 1, best_window=w0, windows=[w0, w1], roi=Roi(x=0.1, y=0.1, w=0.2, h=0.2))
    board.save(
        "storyline",
        Storyline(
            red_thread="rt",
            arc=[
                Chapter(
                    chapter=1,
                    role="hook",
                    message="m",
                    scene_numbers=[SceneWindowRef(scene=1, window=1)],
                    target_seconds=4.0,
                )
            ],
        ),
    )
    board.save(
        "script",
        Script(
            language="de", lines=[ScriptLine(chapter=1, scene_number=1, text="Stopp dein Team")]
        ),
    )
    _save_voice(board, tmp_path, words=[])
    specs = {s.name: s for s in build_production_tool_specs(db, board, asset_id=asset_id)}

    out = specs["build_cutlist"].func(zoom="off")

    assert out["ok"] is True
    cutlist = board.load("cutlist")
    assert isinstance(cutlist, Cutlist)
    seg = cutlist.segments[0]
    assert seg.roi is None and seg.zoom_start_s is None
    # The referenced window still decides WHERE the segment starts — only the crop is dropped.
    assert seg.start_frame == 150


def test_build_cutlist_rejects_unknown_zoom_value(tmp_path: Path) -> None:
    db, asset_id = _seed_two_scenes(tmp_path)
    board = _board(tmp_path, asset_id)
    specs = {s.name: s for s in build_production_tool_specs(db, board, asset_id=asset_id)}

    out = specs["build_cutlist"].func(zoom="wide")

    assert out["ok"] is False
    assert '"auto"' in out["reason"] and '"off"' in out["reason"]
    assert board.load("cutlist") is None


def test_build_cutlist_tool_description_documents_the_zoom_off_lever(tmp_path: Path) -> None:
    """The team can only comply with 'zeig das volle Bild' in one call if the docstring (the
    LLM-facing tool description) says so — three live follow-up runs failed without it."""
    db, asset_id = _seed_two_scenes(tmp_path)
    board = _board(tmp_path, asset_id)
    specs = {s.name: s for s in build_production_tool_specs(db, board, asset_id=asset_id)}

    description = specs["build_cutlist"].description

    assert 'zoom="off"' in description
    assert "regardless of the storyline" in description
    assert "full frame" in description
