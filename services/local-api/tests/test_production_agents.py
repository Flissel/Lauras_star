"""production_agents: v2 production-team roster + Magentic-One team builder (Slice 3, Task 7).

``production_agent_specs`` is pure (no autogen, no db) — the roster, system prompts and tool
assignment are tested directly, mirroring ``tests/test_short_creator_agents.py``. A cross-check
guards that every assigned tool name really exists in ``production_tools.
build_production_tool_specs``'s in-process toolset.

``build_production_team`` is the only autogen-touching function here. autogen_agentchat happens
to be a real installed dependency in this repo, but — like every other autogen-touching test in
this suite (``test_short_creator_agents.py``, ``test_short_creator_magentic.py``, ...) —
``test_build_team_constructs`` fakes the whole autogen import surface rather than using the real
library: tests then pass whether or not the optional extra is installed, and stay hermetic (a
real import would permanently cache ``autogen_core.tools`` etc. in ``sys.modules`` for the rest
of the pytest process, breaking sibling files' "missing extra" tests — see
``_install_fake_autogen``'s docstring).

DB/board fixture is copied from ``tests/test_production_tools_review.py`` (project + asset +
succeeded analysis run + transcript + a hand-built one-scene rough cut via
``created_from=asset_id``) so this file stays self-contained.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from laura.config import Settings
from laura.db import repos
from laura.db.database import Database, SqliteDatabase
from laura.short_creator import production_agents, providers
from laura.short_creator.board import Board
from laura.short_creator.board_models import BoardMeta
from laura.short_creator.production_tools import ProductionDeps, build_production_tool_specs

FPS = 30
SCENE_FRAMES = 150  # 150 frames @ 30fps = 5.0s

EXPECTED_ROSTER = [
    "vision_reviewer",
    "story_architect",
    "scene_author",
    "coding_agent",
    "qa_reviewer",
]

# name -> (tool_names, max_tool_iterations), per the Task-7 brief's roster table.
EXPECTED_ASSIGNMENTS: dict[str, tuple[tuple[str, ...], int]] = {
    "vision_reviewer": (
        ("board_status", "get_scene_context", "review_scene", "get_reviews"),
        10,
    ),
    "story_architect": (
        ("get_reviews", "get_scene_context", "save_storyline", "get_storyline", "board_status"),
        4,
    ),
    "scene_author": (
        ("get_storyline", "get_reviews", "get_scene_context", "save_script_chapter", "get_script"),
        6,
    ),
    "coding_agent": (
        (
            "board_status",
            "get_storyline",
            "get_script",
            "get_reviews",
            "synthesize_script_voice",
            "build_cutlist",
            "save_contact_sheet",
            "render_production",
            "revert_artifact",
        ),
        10,
    ),
    "qa_reviewer": (
        (
            "board_status",
            "get_storyline",
            "get_script",
            "review_export",
            "save_contact_sheet",
            "save_qa_report",
        ),
        6,
    ),
}


def _seed_scene(tmp_path: Path) -> tuple[Database, str]:
    """Project + asset + succeeded analysis run w/ transcript + a ONE-scene rough cut.

    Returns ``(db, asset_id)``. Mirrors ``test_production_tools_review.py``'s ``_seed_scene``.
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
    repos.insert_segment_with_words(
        db,
        asset_id=asset["id"],
        run_id=run["id"],
        speaker_id=None,
        segment={
            "start_sample": 0,
            "end_sample": 96_000,
            "start_frame": 0,
            "end_frame": SCENE_FRAMES,
            "text": "hallo welt schauen wir uns das dashboard an",
            "confidence": 1.0,
        },
        words=[],
    )
    repos.finish_analysis_run(db, run["id"], status="succeeded", diagnostics={})
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
        src_out_frame_exclusive=SCENE_FRAMES,
        seq_in_frame=0,
        seq_out_frame_exclusive=SCENE_FRAMES,
        lane=0,
        role="base",
    )
    repos.replace_scenes(db, project["id"], timeline["id"], [(0, SCENE_FRAMES)])
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


# --- pure roster tests -----------------------------------------------------------------------


def test_roster_shape_and_tool_names_resolve(tmp_path: Path) -> None:
    db, asset_id = _seed_scene(tmp_path)
    board = _board(tmp_path, asset_id)
    deps = ProductionDeps()

    specs = production_agents.production_agent_specs()

    assert [s.name for s in specs] == EXPECTED_ROSTER
    tool_names = {
        t.name
        for t in build_production_tool_specs(db, board, asset_id=asset_id, deps=deps)
    }
    for spec in specs:
        missing = set(spec.tool_names) - tool_names
        assert not missing, f"{spec.name} references unknown tools {missing}"


def test_roster_tool_and_iteration_assignments() -> None:
    by_name = {s.name: s for s in production_agents.production_agent_specs()}
    for name, (tool_names, max_iters) in EXPECTED_ASSIGNMENTS.items():
        assert by_name[name].tool_names == tool_names, name
        assert by_name[name].max_tool_iterations == max_iters, name


def test_roster_specs_have_nonempty_description_and_prompt() -> None:
    for spec in production_agents.production_agent_specs():
        assert spec.description.strip(), spec.name
        assert spec.system_message.strip(), spec.name


def test_specs_are_pure_no_autogen(monkeypatch: pytest.MonkeyPatch) -> None:
    # production_agent_specs() must work even if the autogen import would fail entirely — no
    # import at module level (mirrors agents.py's agent_specs / toolset.py's build_tool_specs).
    monkeypatch.setitem(sys.modules, "autogen_agentchat", None)
    monkeypatch.setitem(sys.modules, "autogen_core", None)
    monkeypatch.setitem(sys.modules, "autogen_ext", None)

    specs = production_agents.production_agent_specs()

    assert [s.name for s in specs] == EXPECTED_ROSTER


def test_prompts_carry_contracts() -> None:
    by_name = {s.name: s for s in production_agents.production_agent_specs()}
    assert "review_scene" in by_name["vision_reviewer"].system_message
    assert "viral arc" in by_name["story_architect"].system_message.lower()
    assert "german" in by_name["scene_author"].system_message.lower()
    assert "never cut the voice" in by_name["coding_agent"].system_message.lower()
    assert "save_contact_sheet" in by_name["coding_agent"].system_message
    assert "ship or revise" in by_name["qa_reviewer"].system_message.lower()
    assert "save_contact_sheet" in by_name["qa_reviewer"].system_message


# --- build_production_team ---------------------------------------------------------------------


def test_build_team_missing_extra_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db, asset_id = _seed_scene(tmp_path)
    board = _board(tmp_path, asset_id)
    monkeypatch.setitem(sys.modules, "autogen_agentchat", None)

    with pytest.raises(RuntimeError, match="autoshort"):
        production_agents.build_production_team(
            db, board, providers.resolve_from_env({}), asset_id=asset_id
        )


def _install_fake_autogen(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Fake the whole autogen surface ``build_production_team`` touches.

    Mirrors ``tests/test_short_creator_magentic.py``'s ``_install_fake_autogen`` exactly.
    autogen_agentchat happens to be a real installed dependency in this repo, but every
    autogen-touching test in this suite fakes the import surface rather than using the real
    library — so tests pass whether or not the optional extra is installed, AND stay hermetic: a
    genuine import would permanently cache ``autogen_core.tools`` / ``autogen_agentchat.agents`` /
    ``autogen_agentchat.teams`` in ``sys.modules`` for the rest of the pytest process (imports are
    process-global, not per-test), which would break every other test file's "missing extra" test
    that simulates the missing optional dependency by nulling only the top-level package name
    (verified empirically: a real-import variant of this test made
    test_short_creator_{agents,magentic,providers,toolset}.py's missing-extra tests fail when the
    full suite runs together, because the submodule stays resolvable in ``sys.modules`` even after
    the parent package name is monkeypatched to ``None``).
    """
    created: dict[str, object] = {}

    class FakeClient:
        def __init__(self, **kw: object) -> None:
            pass

    class FakeFunctionTool:
        def __init__(self, func: object, *, name: str = "", description: str = "") -> None:
            self.name = name

    class FakeAssistantAgent:
        def __init__(
            self,
            *,
            name: str,
            model_client: object,
            tools: tuple[object, ...] = (),
            description: str = "",
            system_message: str = "",
            max_tool_iterations: int = 1,
        ) -> None:
            self.name = name
            self.model_client = model_client
            self.tools = list(tools)
            self.description = description
            self.system_message = system_message
            self.max_tool_iterations = max_tool_iterations

    class FakeMagenticOneGroupChat:
        def __init__(
            self, *, participants: tuple[object, ...], model_client: object, max_turns: int = 0
        ) -> None:
            # Mirrors the real autogen_agentchat MagenticOneGroupChat, which only exposes
            # these as private attributes (set by BaseGroupChat.__init__ / this __init__).
            self._participants = list(participants)
            self._model_client = model_client
            self._max_turns = max_turns
            created["team"] = self

    ollama = types.ModuleType("autogen_ext.models.ollama")
    ollama.OllamaChatCompletionClient = FakeClient  # type: ignore[attr-defined]
    core_tools = types.ModuleType("autogen_core.tools")
    core_tools.FunctionTool = FakeFunctionTool  # type: ignore[attr-defined]
    ac_agents = types.ModuleType("autogen_agentchat.agents")
    ac_agents.AssistantAgent = FakeAssistantAgent  # type: ignore[attr-defined]
    ac_teams = types.ModuleType("autogen_agentchat.teams")
    ac_teams.MagenticOneGroupChat = FakeMagenticOneGroupChat  # type: ignore[attr-defined]
    for name, mod in {
        "autogen_ext": types.ModuleType("autogen_ext"),
        "autogen_ext.models": types.ModuleType("autogen_ext.models"),
        "autogen_ext.models.ollama": ollama,
        "autogen_core": types.ModuleType("autogen_core"),
        "autogen_core.tools": core_tools,
        "autogen_agentchat": types.ModuleType("autogen_agentchat"),
        "autogen_agentchat.agents": ac_agents,
        "autogen_agentchat.teams": ac_teams,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return created


def test_build_team_constructs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db, asset_id = _seed_scene(tmp_path)
    board = _board(tmp_path, asset_id)
    _install_fake_autogen(monkeypatch)

    team = production_agents.build_production_team(
        db, board, providers.resolve_from_env({}), asset_id=asset_id
    )

    # autogen's MagenticOneGroupChat only exposes these privately (no public accessor) — same
    # access pattern as tests/test_short_creator_magentic.py::test_build_magentic_team_wires_roster.
    participants: list[Any] = team._participants
    assert {p.name for p in participants} == set(EXPECTED_ROSTER)
    assert len(participants) == len(EXPECTED_ROSTER)
    assert team._model_client is not None
    assert team._max_turns == production_agents.MAX_TURNS

    # Every agent got the shared agent-role model client and its own tool set (by name).
    by_name = {p.name: p for p in participants}
    for spec_name, (tool_names, max_iters) in EXPECTED_ASSIGNMENTS.items():
        agent = by_name[spec_name]
        assert agent.model_client is not None
        assert {t.name for t in agent.tools} == set(tool_names)
        assert agent.max_tool_iterations == max_iters
    # "ein geteilter Agent-Client" — one shared client instance across all agents, not one each.
    assert len({id(a.model_client) for a in participants}) == 1
    # The orchestrator gets its own client instance (role="orchestrator"), distinct from agents'.
    assert team._model_client is not by_name["vision_reviewer"].model_client
