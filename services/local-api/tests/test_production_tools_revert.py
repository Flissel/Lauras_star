"""production_tools: revert_artifact tool + coding-agent charter (Slice 4, Task 2).

``revert_artifact`` is the coding agent's one content-editing tool: it restores an archived
version of a singleton board artifact as current and reports which downstream artifacts it
invalidates in the process. The tool itself only touches ``board`` — ``db``/``asset_id`` are
just the closure's usual construction arguments (mirrors ``tests/test_production_board.py``'s
minimal fixture, not ``tests/test_production_tools_write.py``'s full scene seed, since
``revert_artifact`` never reads the db).

The roster/prompt half of this file cross-checks ``production_agents.production_agent_specs()``
directly (no autogen, no db) — same style as ``tests/test_production_agents.py``.
"""

from __future__ import annotations

from pathlib import Path

from laura.config import Settings
from laura.db.database import Database, SqliteDatabase
from laura.short_creator import production_agents
from laura.short_creator.board import Board
from laura.short_creator.board_models import BoardMeta, Chapter, Script, ScriptLine, Storyline
from laura.short_creator.production_tools import build_production_tool_specs

ASSET_ID = "a1"


def _db(tmp_path: Path) -> Database:
    settings = Settings(workspace_root=tmp_path / "ws", start_runner=False)
    database: Database = SqliteDatabase(settings.db_path)
    database.migrate()
    return database


def _board(tmp_path: Path) -> Board:
    meta = BoardMeta(
        session_id="s1",
        asset_id=ASSET_ID,
        created_utc="2026-07-13T00:00:00Z",
        task="overview short",
        target_seconds=20.0,
    )
    return Board.create(tmp_path / "board", meta)


def _storyline(thread: str = "one app") -> Storyline:
    return Storyline(
        red_thread=thread,
        arc=[
            Chapter(
                chapter=1, role="hook", message="stop", scene_numbers=[1], target_seconds=3.0
            )
        ],
    )


def _script() -> Script:
    return Script(language="de", lines=[ScriptLine(chapter=1, scene_number=1, text="Stopp!")])


# --- revert_artifact tool ----------------------------------------------------------------------


def test_revert_artifact_happy_restores_content_and_reports_invalidated(tmp_path: Path) -> None:
    db = _db(tmp_path)
    board = _board(tmp_path)
    board.save("storyline", _storyline("v1 thread"))
    board.save("storyline", _storyline("v2 thread"))
    board.save("script", _script())
    specs = {s.name: s for s in build_production_tool_specs(db, board, asset_id=ASSET_ID)}

    out = specs["revert_artifact"].func(name="storyline", version=1)

    assert out == {
        "ok": True,
        "name": "storyline",
        "restored_version": 1,
        "invalidated": ["script"],
    }
    restored = board.load("storyline")
    assert isinstance(restored, Storyline)
    assert restored.red_thread == "v1 thread"
    assert board.load("script") is None  # actually gone, not just reported


def test_revert_artifact_unknown_version(tmp_path: Path) -> None:
    db = _db(tmp_path)
    board = _board(tmp_path)
    board.save("storyline", _storyline())
    specs = {s.name: s for s in build_production_tool_specs(db, board, asset_id=ASSET_ID)}

    out = specs["revert_artifact"].func(name="storyline", version=99)

    assert out == {"ok": False, "reason": "no archived storyline v99"}
    # nothing was touched
    loaded = board.load("storyline")
    assert isinstance(loaded, Storyline) and loaded.version == 1


def test_revert_artifact_unknown_name_lists_valid_names(tmp_path: Path) -> None:
    db = _db(tmp_path)
    board = _board(tmp_path)
    specs = {s.name: s for s in build_production_tool_specs(db, board, asset_id=ASSET_ID)}

    out = specs["revert_artifact"].func(name="nonsense", version=1)

    assert out == {
        "ok": False,
        "reason": (
            "unknown artifact 'nonsense'; valid: "
            "storyline, script, voice, cutlist, contact_sheet, render_report, qa_report"
        ),
    }


# --- roster / prompt charter --------------------------------------------------------------------


def test_only_coding_agent_gets_revert_artifact() -> None:
    by_name = {s.name: s for s in production_agents.production_agent_specs()}

    assert "revert_artifact" in by_name["coding_agent"].tool_names
    for name, spec in by_name.items():
        if name != "coding_agent":
            assert "revert_artifact" not in spec.tool_names, name


def test_coding_agent_prompt_carries_revert_charter() -> None:
    by_name = {s.name: s for s in production_agents.production_agent_specs()}

    assert "revert" in by_name["coding_agent"].system_message.lower()
