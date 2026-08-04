"""production_tools: render_production (zoom passthrough + polled export + checks),
review_export (VLM QA notes over real export frames), save_qa_report (Slice 3, Task 6).

The render_production tests need a REAL, deterministically-built Cutlist on the board, so
``_build_board_to_cutlist`` reuses ``tests/test_production_tools_cutlist.py``'s two-scene fixture
+ word-timing setup verbatim (same numbers, so segment math is already hand-verified there) and
drives it through the actual ``build_cutlist`` tool rather than hand-writing one. review_export
and save_qa_report never touch scenes or the cutlist chain at all, so they use a minimal
one-asset fixture with no rough cut.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from laura.config import Settings
from laura.db import repos
from laura.db.database import Database, SqliteDatabase
from laura.short_creator import production_tools
from laura.short_creator.board import Board
from laura.short_creator.board_models import (
    BestWindow,
    BoardMeta,
    Chapter,
    Cutlist,
    CutSegment,
    Format,
    QaReport,
    RenderCheck,
    RenderReport,
    Roi,
    SceneReview,
    SceneWindowRef,
    Script,
    ScriptLine,
    Storyline,
    VoiceArtifact,
)
from laura.short_creator.production_tools import (
    _MAX_RENDER_CYCLES,
    ProductionDeps,
    build_production_tool_specs,
    script_text,
)

FPS = 30
SCENE_FRAMES = 300  # 300 frames @ 30fps = 10.0s per scene
N_SCENES = 2


def _seed_two_scenes(tmp_path: Path) -> tuple[Database, str]:
    """Project + asset + a TWO-scene rough cut (copy of test_production_tools_cutlist.py's
    fixture — kept self-contained per this file's own convention)."""
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


def _seed_asset(tmp_path: Path) -> tuple[Database, str]:
    """Minimal project + asset, no rough cut — enough for review_export/save_qa_report, which
    never resolve scenes."""
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
    return db, str(asset["id"])


def _board(tmp_path: Path, asset_id: str, board_format: Format = "insta") -> Board:
    meta = BoardMeta(
        session_id="s1",
        asset_id=asset_id,
        created_utc="2026-07-13T00:00:00Z",
        task="overview short",
        format=board_format,
        target_seconds=20.0,
    )
    return Board.create(tmp_path / "board", meta)


def _review(
    board: Board,
    scene_number: int,
    *,
    best_window: BestWindow | None = None,
    roi: Roi | None = None,
) -> None:
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


def _save_voice(
    board: Board, tmp_path: Path, words: list[dict[str, Any]], *, voice_s: float | None = None
) -> VoiceArtifact:
    """Seed the board's voice artifact directly with a real sidecar file on disk (as
    build_cutlist reads it from timings_path, not from a fake backend) — mirrors
    test_production_tools_cutlist.py's helper, plus an explicit voice_s override so the render
    tests can force a voice_fits pass/fail deterministically.

    The hash is the CURRENT script's (storyline-ordered): build_cutlist refuses a voice whose
    script_hash disagrees with the script it is cutting for, so the seed must satisfy the
    real contract.
    """
    from laura.short_creator.board_models import lines_in_storyline_order, script_hash

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
    artifact = VoiceArtifact(
        script_hash=script_hash(lines),
        mp3_path=str(tmp_path / "voice.mp3"),
        timings_path=str(timings_path),
        voice_s=voice_s,
    )
    board.save("voice", artifact)
    return artifact


class _FakeRenderSegments:
    """Fake render_segments: records every call's kwargs and creates a REAL export row directly
    via repos (like the real tool_render_segments's job would eventually produce) so
    render_production's poll loop sees an already-terminal status — no sleep needed."""

    def __init__(self, *, status: str = "ready") -> None:
        self.calls: list[dict[str, Any]] = []
        self.status = status

    def __call__(
        self, db: Database, asset_id: str, segments: list[tuple[int, int]], **kwargs: Any
    ) -> dict[str, Any]:
        self.calls.append({"asset_id": asset_id, "segments": segments, **kwargs})
        asset = repos.get_asset(db, asset_id)
        assert asset is not None
        exp = repos.create_export(
            db, project_id=str(asset["project_id"]), timeline_id=None, format="mp4"
        )
        if self.status == "ready":
            repos.set_export_done(db, exp["id"], path=str(Path("out") / "short.mp4"), size_bytes=1)
        elif self.status == "error":
            repos.set_export_error(db, exp["id"], "render boom")
        return {"ok": True, "export_id": exp["id"], "job_id": "job-1", "segments": len(segments)}


def _build_board_to_cutlist(
    tmp_path: Path,
    *,
    scene2_roi: Roi | None,
    voice_s: float | None,
    board_format: Format = "insta",
) -> tuple[Database, str, Board]:
    """A full board up to (and including) a real cutlist — reviews, storyline, script, voice,
    then the actual build_cutlist tool. Same word timings as
    test_production_tools_cutlist.py::test_build_cutlist_deterministic_segments_and_zoom (already
    hand-verified there), so segment 0 (scene 1, which always has a roi) comes out to
    start/end=(30, 90) with zoom_start_s~=0.6; segment 1 (scene 2) is (300, 360) and its
    zoom_start_s depends on scene2_roi (None -> no zoom regardless of timing — the "segment
    without roi -> zoom entry None" case render_production must pass through unchanged).
    """
    db, asset_id = _seed_two_scenes(tmp_path)
    board = _board(tmp_path, asset_id, board_format)
    _review(
        board,
        1,
        best_window=BestWindow(offset_s=1.0, duration_s=3.0),
        roi=Roi(x=0.1, y=0.1, w=0.2, h=0.2),
    )
    _review(board, 2, best_window=BestWindow(offset_s=0.0, duration_s=3.0), roi=scene2_roi)
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
        voice_s=voice_s,
    )
    specs = {s.name: s for s in build_production_tool_specs(db, board, asset_id=asset_id)}
    built = specs["build_cutlist"].func()
    assert built["ok"] is True, built
    return db, asset_id, board


def test_render_production_passes_zoom_and_reports(tmp_path: Path) -> None:
    db, asset_id, board = _build_board_to_cutlist(tmp_path, scene2_roi=None, voice_s=3.4)
    fake = _FakeRenderSegments(status="ready")
    deps = ProductionDeps(render_segments=fake)
    specs = {
        s.name: s for s in build_production_tool_specs(db, board, asset_id=asset_id, deps=deps)
    }

    out = specs["render_production"].func()

    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["asset_id"] == asset_id
    assert call["segments"] == [(30, 90), (300, 360)]
    zoom = call["zoom"]
    assert len(zoom) == 2
    assert zoom[0]["roi"] == {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}
    assert zoom[0]["zoom_start_s"] == pytest.approx(0.6)
    assert zoom[1] is None  # scene 2 has no roi -> index-aligned None, not dropped
    assert call["captions"] is True
    assert call["fit"] == "blur"
    assert call["vertical"] is True
    assert call["out_size"] == (1080, 1920)
    assert call["voiceover_path"] == str(tmp_path / "voice.mp3")
    # _build_board_to_cutlist's storyline scene_numbers=[1, 2] matches _script()'s natural line
    # order, so the storyline-ordered text equals the raw line-order text here.
    assert call["voiceover_text"] == script_text(_script().lines)

    assert out["ok"] is True
    checks_by_name = {c["name"]: c for c in out["checks"]}
    assert checks_by_name["voice_fits"]["ok"] is True
    assert checks_by_name["export_ready"]["ok"] is True
    assert checks_by_name["has_voice_timings"]["ok"] is True

    report = board.load("render_report")
    assert isinstance(report, RenderReport)
    assert report.export_id == out["export_id"]
    assert report.video_s == pytest.approx(4.0)
    assert report.voice_s == pytest.approx(3.4)
    assert report.width == 1080
    assert report.height == 1920
    assert len(report.checks) == 3

    row = repos.get_export(db, report.export_id)
    assert row is not None
    assert row["status"] == "ready"


def test_render_production_renders_landscape_for_a_format_x_board(tmp_path: Path) -> None:
    """The board's format reaches the renderer — the reason a 16:9 demo is possible at all.

    Everything above this seam (reviews, storyline, script, cutlist, rois) is format-agnostic;
    only the canvas changes. A 9:16 crop of a screen recording throws away the half of the
    frame that carries the content, so this must not silently fall back to the reel.
    """
    db, asset_id, board = _build_board_to_cutlist(
        tmp_path, scene2_roi=None, voice_s=3.4, board_format="x"
    )
    fake = _FakeRenderSegments(status="ready")
    deps = ProductionDeps(render_segments=fake)
    specs = {
        s.name: s for s in build_production_tool_specs(db, board, asset_id=asset_id, deps=deps)
    }

    out = specs["render_production"].func()

    call = fake.calls[0]
    assert call["vertical"] is False, "landscape must not go through the vertical fit path"
    assert call["out_size"] == (1920, 1080)
    # The segments themselves are unchanged — the format decides the canvas, nothing else.
    assert call["segments"] == [(30, 90), (300, 360)]

    assert out["ok"] is True
    report = board.load("render_report")
    assert isinstance(report, RenderReport)
    assert (report.width, report.height) == (1920, 1080), "the report must record what shipped"


def test_render_production_voice_fit_check_fails(tmp_path: Path) -> None:
    db, asset_id, board = _build_board_to_cutlist(tmp_path, scene2_roi=None, voice_s=100.0)
    fake = _FakeRenderSegments(status="ready")
    deps = ProductionDeps(render_segments=fake)
    specs = {
        s.name: s for s in build_production_tool_specs(db, board, asset_id=asset_id, deps=deps)
    }

    out = specs["render_production"].func()

    assert out["ok"] is False  # video (4.0s) is nowhere near the (deliberately inflated) voice
    checks_by_name = {c["name"]: c for c in out["checks"]}
    assert checks_by_name["voice_fits"]["ok"] is False
    assert checks_by_name["export_ready"]["ok"] is True  # the export itself still rendered fine

    # The report is still persisted (a coding agent needs to SEE the failing check to react).
    report = board.load("render_report")
    assert isinstance(report, RenderReport)
    assert report.voice_s == pytest.approx(100.0)
    assert any(c.name == "voice_fits" and not c.ok for c in report.checks)


def test_render_production_requires_cutlist(tmp_path: Path) -> None:
    db, asset_id = _seed_two_scenes(tmp_path)
    board = _board(tmp_path, asset_id)
    specs = {s.name: s for s in build_production_tool_specs(db, board, asset_id=asset_id)}

    out = specs["render_production"].func()
    assert out["ok"] is False
    assert "cutlist" in out["reason"]

    # A cutlist can exist on the board without voice ever having been saved (only reachable by
    # writing straight to the board, as here — the real build_cutlist tool itself requires voice
    # first); render_production must still catch this instead of crashing on a None voice.
    board.save(
        "cutlist",
        Cutlist(
            segments=[CutSegment(order=0, scene_number=1, start_frame=0, end_frame_exclusive=60)]
        ),
    )
    out = specs["render_production"].func()
    assert out["ok"] is False
    assert "voice" in out["reason"]


def test_render_production_stops_after_the_revision_cap(tmp_path: Path) -> None:
    """A hard cap on real renders — the net for a loop that revises for some reason the
    budget fix cannot remove.

    Live finding: even with voice_fits satisfiable, gpt-5-mini rendered four times and
    gpt-5.5 rendered repeatedly, each render rebuilding the whole chain. The prompt's "one
    revise round" was ignored — a prompt does not enforce a limit. So render_production
    refuses to spend another render once the cap is reached: it returns the last one as final
    instead of calling the (expensive) renderer again. The turn budget stops the rest.
    """
    db, asset_id, board = _build_board_to_cutlist(tmp_path, scene2_roi=None, voice_s=3.4)
    fake = _FakeRenderSegments(status="ready")
    deps = ProductionDeps(render_segments=fake)
    specs = {
        s.name: s for s in build_production_tool_specs(db, board, asset_id=asset_id, deps=deps)
    }

    for _ in range(_MAX_RENDER_CYCLES):
        out = specs["render_production"].func()
        assert out.get("ok") is True, out
    assert len(fake.calls) == _MAX_RENDER_CYCLES, "every render up to the cap really rendered"

    capped = specs["render_production"].func()

    assert len(fake.calls) == _MAX_RENDER_CYCLES, "the cap must not spend another render"
    assert capped["ok"] is True
    assert capped["final"] is True
    assert capped["stale"] is False, "the script has not moved, so the cut still speaks it"
    assert "limit" in capped["note"].lower()
    assert capped["export_id"], "the last successful render is still shipped"


def test_the_cap_says_so_when_the_shipped_cut_no_longer_speaks_the_script(
    tmp_path: Path,
) -> None:
    """The live failure: a v14-era render shipped as final onto a v39 board.

    Hitting the cap restores the newest archived render so the finished export is still
    reported. If the script moved on in the meantime, that restored cut speaks words the board
    no longer contains — and its own voice_fits check still reads OK, because it was true when
    it was written. Shipping it remains right at the cap; presenting it as a finished film of
    the current script is what made a broken board look verified.
    """
    db, asset_id, board = _build_board_to_cutlist(tmp_path, scene2_roi=None, voice_s=3.4)
    fake = _FakeRenderSegments(status="ready")
    deps = ProductionDeps(render_segments=fake)
    specs = {
        s.name: s for s in build_production_tool_specs(db, board, asset_id=asset_id, deps=deps)
    }

    for _ in range(_MAX_RENDER_CYCLES):
        specs["render_production"].func()

    # The author rewrites the script after the last render. That save invalidates voice and
    # cutlist too, and the run rebuilds both — which is exactly what the live timeline shows
    # (script -> voice -> cutlist -> render). Only the render stays the restored old one.
    rendered = board.load("script")
    assert isinstance(rendered, Script)
    voice, cutlist = board.load("voice"), board.load("cutlist")
    board.save(
        "script",
        Script(
            language=rendered.language,
            lines=[
                line.model_copy(update={"text": f"{line.text} and something else entirely"})
                for line in rendered.lines
            ],
        ),
    )
    assert voice is not None and cutlist is not None
    board.save("voice", voice)
    board.save("cutlist", cutlist)
    board.revert("render_report", max(board.versions("render_report")))

    capped = specs["render_production"].func()

    assert capped["final"] is True, "the cap still ships the last cut"
    assert capped["stale"] is True
    assert capped["ok"] is False, "a cut that speaks a different script is not a good outcome"
    assert "earlier script" in capped["note"]


# --- the follow-up render allowance (live finding 2026-08-04) ------------------------------------
# The user explicitly asked for a reframe ("zeig das volle Bild") after the cap was already
# reached, and _MAX_RENDER_CYCLES silently shipped the old cut instead — an operator-requested
# change must not be eaten by a runaway-loop backstop. run_production raises the cap by ONE
# render per explicit user follow-up (deps.max_render_cycles = follow_up_render_cap(board)).


def test_follow_up_render_cap_values(tmp_path: Path) -> None:
    """Pure cap arithmetic: never below _MAX_RENDER_CYCLES (a follow-up on a board with budget
    left grants nothing extra), and exactly one render above what has already been spent."""
    from laura.short_creator.production_tools import follow_up_render_cap

    db, asset_id = _seed_asset(tmp_path)
    board = _board(tmp_path, asset_id)

    assert follow_up_render_cap(board) == _MAX_RENDER_CYCLES  # no renders yet -> plain cap

    for i in range(_MAX_RENDER_CYCLES):
        # Distinct export ids: board.save short-circuits an IDENTICAL re-save into a no-op,
        # which would leave the version count at 1 instead of really spending the cap.
        board.save(
            "render_report",
            RenderReport(export_id=f"e{i}", video_s=4.0, width=1080, height=1920, checks=[]),
        )
    assert follow_up_render_cap(board) == _MAX_RENDER_CYCLES + 1  # spent cap -> one more


def test_follow_up_cap_grants_exactly_one_more_render(tmp_path: Path) -> None:
    """With deps.max_render_cycles raised, render_production really renders ONE more time past
    the exhausted default cap — and the raised cap binds again right after."""
    from laura.short_creator.production_tools import follow_up_render_cap

    db, asset_id, board = _build_board_to_cutlist(tmp_path, scene2_roi=None, voice_s=3.4)
    fake = _FakeRenderSegments(status="ready")
    deps = ProductionDeps(render_segments=fake)
    specs = {
        s.name: s for s in build_production_tool_specs(db, board, asset_id=asset_id, deps=deps)
    }
    for _ in range(_MAX_RENDER_CYCLES):
        assert specs["render_production"].func()["ok"] is True
    assert specs["render_production"].func()["final"] is True  # the default cap is exhausted
    assert len(fake.calls) == _MAX_RENDER_CYCLES

    raised = ProductionDeps(
        render_segments=fake, max_render_cycles=follow_up_render_cap(board)
    )
    raised_specs = {
        s.name: s
        for s in build_production_tool_specs(db, board, asset_id=asset_id, deps=raised)
    }

    out = raised_specs["render_production"].func()

    assert out["ok"] is True
    assert out.get("final") is not True, "the operator-requested render really happened"
    assert len(fake.calls) == _MAX_RENDER_CYCLES + 1

    again = raised_specs["render_production"].func()
    assert again["final"] is True, "one extra render per follow-up — then the cap holds again"
    assert len(fake.calls) == _MAX_RENDER_CYCLES + 1
    assert str(_MAX_RENDER_CYCLES + 1) in again["note"]  # the note names the RAISED cap


def test_render_and_voice_and_sheet_stamp_their_parents(tmp_path: Path) -> None:
    from laura.short_creator.board_models import content_hash

    db, asset_id, board = _build_board_to_cutlist(tmp_path, scene2_roi=None, voice_s=3.4)
    storyline = board.load("storyline")
    script = board.load("script")
    voice = board.load("voice")
    cutlist = board.load("cutlist")
    assert storyline and script and voice and cutlist
    fake = _FakeRenderSegments(status="ready")
    deps = ProductionDeps(render_segments=fake)
    specs = {
        s.name: s for s in build_production_tool_specs(db, board, asset_id=asset_id, deps=deps)
    }

    out = specs["render_production"].func()
    assert out["ok"] is True, out

    render = board.load("render_report")
    assert isinstance(render, RenderReport)
    assert render.parents == {
        "storyline": content_hash(storyline),
        "script": content_hash(script),
        "voice": content_hash(voice),
        "cutlist": content_hash(cutlist),
    }
    # The voice artifact in this builder is seeded directly; synthesize_script_voice's own
    # stamp is asserted in test_production_tools_cutlist.py's synthesis test below.


def test_render_report_carries_target_ratio_and_note(tmp_path: Path) -> None:
    """Live finding 2026-07-20: a 31.6s film against a 174s target shipped with QA 'ship' —
    voice_fits checks A/V cover, nothing reported target adherence. Reporting only, never a
    fourth check (a failing length check would provoke render thrashing)."""
    db, asset_id, board = _build_board_to_cutlist(tmp_path, scene2_roi=None, voice_s=3.4)
    fake = _FakeRenderSegments(status="ready")
    deps = ProductionDeps(render_segments=fake)
    specs = {
        s.name: s for s in build_production_tool_specs(db, board, asset_id=asset_id, deps=deps)
    }

    out = specs["render_production"].func()

    assert out["ok"] is True
    assert "target_note" in out
    assert "vs target" in out["target_note"]
    render = board.load("render_report")
    assert isinstance(render, RenderReport)
    assert render.target_ratio is not None
    # ratio == video_s / meta.target_seconds, rounded to 3 places
    assert render.target_ratio == pytest.approx(
        round(render.video_s / board.meta().target_seconds, 3)
    )
    # the checks list stays exactly three — target adherence must never gate
    assert [c.name for c in render.checks] == [
        "voice_fits", "export_ready", "has_voice_timings"
    ]


def test_old_render_report_json_without_target_ratio_still_loads() -> None:
    report = RenderReport.model_validate_json(
        '{"export_id": "e1", "video_s": 10.0, "width": 1920, "height": 1080}'
    )
    assert report.target_ratio is None


def test_review_export_collects_notes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db, asset_id = _seed_asset(tmp_path)
    board = _board(tmp_path, asset_id)
    asset = repos.get_asset(db, asset_id)
    assert asset is not None
    exp = repos.create_export(
        db, project_id=str(asset["project_id"]), timeline_id=None, format="mp4"
    )
    export_path = str(tmp_path / "short.mp4")
    repos.set_export_done(db, exp["id"], path=export_path, size_bytes=42)
    board.save(
        "render_report",
        RenderReport(
            export_id=exp["id"],
            video_s=6.0,
            voice_s=5.0,
            width=1080,
            height=1920,
            checks=[RenderCheck(name="export_ready", ok=True)],
        ),
    )

    grab_calls: list[tuple[Path, list[float]]] = []

    def _fake_grab(path: Path, at_seconds: list[float]) -> list[bytes]:
        grab_calls.append((path, list(at_seconds)))
        return [f"frame@{t}".encode() for t in at_seconds]

    monkeypatch.setattr(production_tools, "_grab_video_frames", _fake_grab)

    class _Vlm:
        def available(self) -> bool:
            return True

        def describe(self, frames: list[bytes], prompt: str) -> str:
            assert len(frames) == 1
            return f"note for {frames[0].decode()}"

    deps = ProductionDeps(describe_backend=_Vlm())
    specs = {
        s.name: s for s in build_production_tool_specs(db, board, asset_id=asset_id, deps=deps)
    }

    out = specs["review_export"].func()

    assert out["ok"] is True
    assert out["notes"] == [
        {"at_s": 1.0, "note": "note for frame@1.0"},
        {"at_s": 3.0, "note": "note for frame@3.0"},
        {"at_s": 4.5, "note": "note for frame@4.5"},
    ]
    # Per-timestamp grabs: 3 separate calls
    assert grab_calls == [
        (Path(export_path), [1.0]),
        (Path(export_path), [3.0]),
        (Path(export_path), [4.5]),
    ]

    grab_calls.clear()
    out2 = specs["review_export"].func(at_seconds=[2.0])
    assert out2["ok"] is True
    assert out2["notes"] == [{"at_s": 2.0, "note": "note for frame@2.0"}]
    assert grab_calls == [(Path(export_path), [2.0])]

    # No configured describe backend -> degrade instead of blocking the QA reviewer. Guard
    # against a host env that has a VLM configured (mirrors test_production_tools_review.py's
    # test_review_scene_degrades_without_backend).
    monkeypatch.delenv("LAURA_VLM_MODEL", raising=False)
    monkeypatch.delenv("LAURA_VLM", raising=False)
    monkeypatch.delenv("LAURA_VLM_PROVIDER", raising=False)
    deps_no_backend = ProductionDeps(describe_backend=None)
    specs_no_backend = {
        s.name: s
        for s in build_production_tool_specs(db, board, asset_id=asset_id, deps=deps_no_backend)
    }

    degraded = specs_no_backend["review_export"].func()

    assert degraded == {"ok": True, "notes": [], "degraded": True}


def test_review_export_skips_failed_frame_grabs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify per-timestamp grabs: a failed grab in the middle doesn't mislabel remaining
    frames (regression test for the zip-pairing bug)."""
    db, asset_id = _seed_asset(tmp_path)
    board = _board(tmp_path, asset_id)
    asset = repos.get_asset(db, asset_id)
    assert asset is not None
    exp = repos.create_export(
        db, project_id=str(asset["project_id"]), timeline_id=None, format="mp4"
    )
    export_path = str(tmp_path / "short.mp4")
    repos.set_export_done(db, exp["id"], path=export_path, size_bytes=42)
    board.save(
        "render_report",
        RenderReport(
            export_id=exp["id"],
            video_s=6.0,
            voice_s=5.0,
            width=1080,
            height=1920,
            checks=[RenderCheck(name="export_ready", ok=True)],
        ),
    )

    grab_calls: list[tuple[Path, list[float]]] = []

    def _fake_grab_selective(path: Path, at_seconds: list[float]) -> list[bytes]:
        """Return [] for the middle timestamp (3.0), frame for others."""
        grab_calls.append((path, list(at_seconds)))
        result = []
        for t in at_seconds:
            if abs(t - 3.0) < 0.001:  # Skip middle timestamp
                pass
            else:
                result.append(f"frame@{t}".encode())
        return result

    monkeypatch.setattr(production_tools, "_grab_video_frames", _fake_grab_selective)

    call_count = [0]

    class _VlmWithCounter:
        def available(self) -> bool:
            return True

        def describe(self, frames: list[bytes], prompt: str) -> str:
            call_count[0] += 1
            assert len(frames) == 1
            return f"note#{call_count[0]} for {frames[0].decode()}"

    deps = ProductionDeps(describe_backend=_VlmWithCounter())
    specs = {
        s.name: s for s in build_production_tool_specs(db, board, asset_id=asset_id, deps=deps)
    }

    out = specs["review_export"].func(at_seconds=[1.0, 3.0, 4.5])

    assert out["ok"] is True
    # Only 2 notes: first and last timestamps (middle failed grab is skipped, never mislabeled)
    assert len(out["notes"]) == 2
    assert out["notes"][0]["at_s"] == 1.0
    assert out["notes"][0]["note"] == "note#1 for frame@1.0"
    assert out["notes"][1]["at_s"] == 4.5
    assert out["notes"][1]["note"] == "note#2 for frame@4.5"
    # Verify per-timestamp grabs: 3 separate calls
    assert len(grab_calls) == 3
    assert grab_calls[0] == (Path(export_path), [1.0])
    assert grab_calls[1] == (Path(export_path), [3.0])
    assert grab_calls[2] == (Path(export_path), [4.5])


def test_save_qa_report_validates(tmp_path: Path) -> None:
    db, asset_id = _seed_asset(tmp_path)
    board = _board(tmp_path, asset_id)
    # QA judges a render, so one must exist (order guard) before any verdict is accepted.
    board.save(
        "render_report",
        RenderReport(export_id="e1", video_s=10.0, width=1920, height=1080),
    )
    specs = {s.name: s for s in build_production_tool_specs(db, board, asset_id=asset_id)}

    bad = specs["save_qa_report"].func(verdict="maybe", findings=[])
    assert bad["ok"] is False
    assert "errors" in bad
    assert any("verdict" in err for err in bad["errors"])
    assert board.load("qa_report") is None

    ok = specs["save_qa_report"].func(
        verdict="ship",
        findings=[{"severity": "minor", "where": "scene 2 caption", "note": "syncs a beat late"}],
    )
    assert ok == {"ok": True, "version": 1}

    saved = board.load("qa_report")
    assert isinstance(saved, QaReport)
    assert saved.verdict == "ship"
    assert len(saved.findings) == 1
    assert saved.findings[0].severity == "minor"
    assert saved.findings[0].where == "scene 2 caption"

    bad_finding = specs["save_qa_report"].func(
        verdict="revise", findings=[{"severity": "minor", "note": "missing the where field"}]
    )
    assert bad_finding["ok"] is False
    assert any("where" in err for err in bad_finding["errors"])
