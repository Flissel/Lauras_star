"""production_tools: the length + grounding levers from the 2026-08-04 live production.

Live findings that night, three of a kind: (1) 45-60s targets shipped as 20-34s films because
the team wrote ~50-word scripts although script_budget had computed a 93-word allocation —
its docstring says "call it ONCE before writing" and nothing enforced it; (2) the scripts were
generic marketing copy instead of claims the source supports; (3) the team never saw the
per-scene transcript. The code answers: ``save_script_chapter`` soft-gates a chapter far below
its budget share (reject once with the numbers, accept a deliberate re-save), and
``get_scene_transcript`` hands the writer the verbatim source words to quote from.

DB fixture mirrors ``tests/test_production_tools_write.py``'s ``_seed_scene`` but with
parameterizable scene lengths and word rows, since the budget gate needs a scene long enough
to carry a real allocation and the transcript tool needs word-level rows to join.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from laura.config import Settings
from laura.db import repos
from laura.db.database import Database, SqliteDatabase
from laura.short_creator.board import Board
from laura.short_creator.board_models import BestWindow, BoardMeta, SceneReview
from laura.short_creator.production_tools import build_production_tool_specs

FPS = 30


def _seed(
    tmp_path: Path,
    *,
    scenes: list[tuple[int, int]],
    segments: list[dict[str, Any]],
) -> tuple[Database, str]:
    """Project + asset + succeeded analysis run + rough cut with the given scenes/segments.

    Each ``segments`` entry is ``{"start_frame", "end_frame", "text", "words"?}``; words are
    passed through to ``insert_segment_with_words``. One lane-0 clip spans all scenes with
    ``src == seq`` so source and sequence frames coincide (same shape as the sibling files).
    """
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
    run = repos.create_analysis_run(db, asset_id=asset["id"], pipeline_version="t", config={})
    repos.start_analysis_run(db, run["id"])
    for seg in segments:
        repos.insert_segment_with_words(
            db,
            asset_id=asset["id"],
            run_id=run["id"],
            speaker_id=None,
            segment={
                "start_sample": seg["start_frame"] * 1600,
                "end_sample": seg["end_frame"] * 1600,
                "start_frame": seg["start_frame"],
                "end_frame": seg["end_frame"],
                "text": seg["text"],
                "confidence": 1.0,
            },
            words=seg.get("words", []),
        )
    repos.finish_analysis_run(db, run["id"], status="succeeded", diagnostics={})
    timeline = repos.create_timeline(
        db,
        project_id=project["id"],
        name="Rough Cut",
        kind="rough_cut",
        created_from=asset["id"],
    )
    total_frames = max(end for _start, end in scenes)
    repos.add_timeline_clip(
        db,
        timeline_id=timeline["id"],
        asset_id=asset["id"],
        src_in_frame=0,
        src_out_frame_exclusive=total_frames,
        seq_in_frame=0,
        seq_out_frame_exclusive=total_frames,
        lane=0,
        role="base",
    )
    repos.replace_scenes(db, project["id"], timeline["id"], scenes)
    return db, str(asset["id"])


def _board(tmp_path: Path, asset_id: str, *, target_seconds: float) -> Board:
    meta = BoardMeta(
        session_id="s1",
        asset_id=asset_id,
        created_utc="2026-08-04T00:00:00Z",
        task="overview short",
        target_seconds=target_seconds,
    )
    return Board.create(tmp_path / "board", meta)


def _review(board: Board, scene_number: int, *, scene_frames: int) -> None:
    board.save_scene_review(
        SceneReview(
            scene_number=scene_number,
            src_start_frame=0,
            src_end_frame_exclusive=scene_frames,
            description="d",
            whats_happening="h",
            hook_score=5,
            best_window=BestWindow(offset_s=0.0, duration_s=3.0),
        )
    )


def _word(idx: int, start_frame: int, end_frame: int, text: str) -> dict[str, Any]:
    return {
        "idx": idx,
        "start_sample": start_frame * 1600,
        "end_sample": end_frame * 1600,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "text": text,
        "confidence": 1.0,
    }


# --- the budget gate: a chapter far below its allocation must hear it BEFORE it ships -------
# Live 2026-08-04: three films in a row came out at 20-34s against 45-60s targets. The
# per-chapter budget existed (93 words for the 60s target), script_budget's docstring said
# "call it ONCE before writing" — and the team saved ~50-word scripts anyway. Prompts do not
# bind; the contract moves into the write path, same as the storyline-order guard.


def _sixty_second_fixture(tmp_path: Path) -> tuple[Database, str, Board, dict[str, Any]]:
    """One 60s scene, 60s target — the live geometry: a 93-word chapter allocation."""
    db, asset_id = _seed(
        tmp_path,
        scenes=[(0, 1800)],
        segments=[{"start_frame": 0, "end_frame": 1800, "text": "echte rede " * 90}],
    )
    board = _board(tmp_path, asset_id, target_seconds=60.0)
    _review(board, 1, scene_frames=1800)
    specs = {s.name: s for s in build_production_tool_specs(db, board, asset_id=asset_id)}
    saved = specs["save_storyline"].func(
        red_thread="r",
        chapters=[
            {
                "chapter": 1,
                "role": "hook",
                "message": "m",
                "scene_numbers": [1],
                "target_seconds": 60.0,
            }
        ],
    )
    assert saved["ok"] is True, saved
    budget = specs["script_budget"].func()
    assert budget["per_chapter"][0]["words"] == 93, "the live geometry must reproduce"
    return db, asset_id, board, specs


def test_a_far_under_budget_chapter_is_rejected_once_with_the_numbers(tmp_path: Path) -> None:
    """The first save at 50/93 words is refused, and the reason carries everything the author
    needs to act: both numbers, the source to write from, and the deliberate-shorter escape."""
    _db, _asset_id, board, specs = _sixty_second_fixture(tmp_path)

    out = specs["save_script_chapter"].func(
        chapter=1, lines=[{"scene_number": 1, "text": " ".join(["wort"] * 50)}]
    )

    assert out["ok"] is False
    assert "50" in out["reason"] and "93" in out["reason"]
    assert "script_budget" in out["reason"]
    assert "get_scene_transcript" in out["reason"]
    assert board.load("script") is None, "a rejected save must not reach the board"


def test_the_second_undershoot_save_is_accepted_with_a_warning(tmp_path: Path) -> None:
    """The gate is soft: saving the chapter again accepts the deliberate shorter film, and
    the reply still names the gap so nobody can claim they were not told."""
    _db, _asset_id, board, specs = _sixty_second_fixture(tmp_path)
    lines = [{"scene_number": 1, "text": " ".join(["wort"] * 50)}]
    first = specs["save_script_chapter"].func(chapter=1, lines=lines)
    assert first["ok"] is False

    second = specs["save_script_chapter"].func(chapter=1, lines=lines)

    assert second["ok"] is True
    assert "93" in second["budget_warning"]
    assert board.load("script") is not None


def test_a_chapter_near_its_budget_saves_clean(tmp_path: Path) -> None:
    _db, _asset_id, _board_, specs = _sixty_second_fixture(tmp_path)

    out = specs["save_script_chapter"].func(
        chapter=1, lines=[{"scene_number": 1, "text": " ".join(["wort"] * 85)}]
    )

    assert out["ok"] is True, out
    assert "budget_warning" not in out


def test_a_small_absolute_shortfall_stays_quiet(tmp_path: Path) -> None:
    """A 5s scene allocates ~7 words; writing 1 misses six words (~3.5s) — under the gate's
    missing-seconds floor. Tiny chapters must not nag: the gate exists for films that come
    out MATERIALLY short, not for single-digit word counts."""
    db, asset_id = _seed(
        tmp_path,
        scenes=[(0, 150)],
        segments=[{"start_frame": 0, "end_frame": 150, "text": "hallo welt"}],
    )
    board = _board(tmp_path, asset_id, target_seconds=20.0)
    _review(board, 1, scene_frames=150)
    specs = {s.name: s for s in build_production_tool_specs(db, board, asset_id=asset_id)}
    saved = specs["save_storyline"].func(
        red_thread="r",
        chapters=[
            {
                "chapter": 1,
                "role": "hook",
                "message": "m",
                "scene_numbers": [1],
                "target_seconds": 3.0,
            }
        ],
    )
    assert saved["ok"] is True, saved

    out = specs["save_script_chapter"].func(
        chapter=1, lines=[{"scene_number": 1, "text": "hi"}]
    )

    assert out["ok"] is True, out
    assert "budget_warning" not in out


# --- get_scene_transcript: the verbatim source the script must be able to point at ----------
# Live 2026-08-04: scripts were generic marketing copy ("maximale Effizienz", an invented
# drag-and-drop) because the team had no way to read what is actually SAID in a scene. The
# fix that satisfied the operator: script lines sourced only from the transcript + reviews —
# which needs a tool that joins transcript_segments/transcript_words for a scene's frames.


def _two_scene_fixture(tmp_path: Path) -> tuple[Database, str]:
    """Scenes 1 (frames 0-300) and 2 (300-600); one segment straddling the boundary with
    word rows on both sides, one segment entirely inside scene 2."""
    return _seed(
        tmp_path,
        scenes=[(0, 300), (300, 600)],
        segments=[
            {
                "start_frame": 0,
                "end_frame": 320,
                "text": "hallo welt danach",
                "words": [
                    _word(0, 0, 150, "hallo"),
                    _word(1, 150, 290, "welt"),
                    _word(2, 300, 320, "danach"),
                ],
            },
            {"start_frame": 330, "end_frame": 590, "text": "zweite szene inhalt"},
        ],
    )


def test_get_scene_transcript_returns_verbatim_segments_and_words(tmp_path: Path) -> None:
    db, asset_id = _two_scene_fixture(tmp_path)
    board = _board(tmp_path, asset_id, target_seconds=20.0)
    specs = {s.name: s for s in build_production_tool_specs(db, board, asset_id=asset_id)}

    out = specs["get_scene_transcript"].func(scene_number=1)

    assert out["ok"] is True
    assert out["scene_number"] == 1
    assert out["src_start_frame"] == 0 and out["src_end_frame_exclusive"] == 300
    assert out["duration_s"] == 10.0
    assert [seg["text"] for seg in out["segments"]] == ["hallo welt danach"]
    assert out["segments"][0]["start_s"] == 0.0
    # The word join is frame-range exact: "danach" is spoken past frame 300 and belongs to
    # scene 2, even though its segment overlaps scene 1.
    assert out["verbatim_words"] == "hallo welt"


def test_get_scene_transcript_maps_seconds_scene_relative(tmp_path: Path) -> None:
    db, asset_id = _two_scene_fixture(tmp_path)
    board = _board(tmp_path, asset_id, target_seconds=20.0)
    specs = {s.name: s for s in build_production_tool_specs(db, board, asset_id=asset_id)}

    out = specs["get_scene_transcript"].func(scene_number=2)

    assert out["ok"] is True
    texts = [seg["text"] for seg in out["segments"]]
    assert "zweite szene inhalt" in texts
    inside = next(seg for seg in out["segments"] if seg["text"] == "zweite szene inhalt")
    assert inside["start_s"] == 1.0  # frame 330 relative to scene start 300, at 30fps
    assert inside["end_s"] == 9.67
    # Scene 2's exact words: only "danach" has word rows in this range.
    assert out["verbatim_words"] == "danach"


def test_get_scene_transcript_unknown_scene(tmp_path: Path) -> None:
    db, asset_id = _two_scene_fixture(tmp_path)
    board = _board(tmp_path, asset_id, target_seconds=20.0)
    specs = {s.name: s for s in build_production_tool_specs(db, board, asset_id=asset_id)}

    out = specs["get_scene_transcript"].func(scene_number=7)

    assert out == {"ok": False, "reason": "unknown scene"}
