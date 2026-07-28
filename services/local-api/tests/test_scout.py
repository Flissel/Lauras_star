"""Scout choice validation uses a real rough-cut scene universe and fake LLM replies."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from laura.config import Settings
from laura.db import repos
from laura.db.database import Database, SqliteDatabase
from laura.short_creator import scout
from laura.short_creator.providers import resolve_from_env

FPS = 30


def _db(tmp_path: Path) -> Database:
    settings = Settings(workspace_root=tmp_path / "ws", start_runner=False)
    db: Database = SqliteDatabase(settings.db_path)
    db.migrate()
    return db


def _seed_asset_with_scenes(db: Database, project_id: str, name: str) -> str:
    """Create one transcript-bearing asset with two rough-cut scenes [0,300)/[300,600)."""
    asset = repos.create_asset(
        db, project_id=project_id, type="video", display_name=name, source_path=f"/tmp/{name}"
    )
    run = repos.create_analysis_run(db, asset_id=asset["id"], pipeline_version="t", config={})
    repos.start_analysis_run(db, run["id"])
    repos.insert_segment_with_words(
        db,
        asset_id=asset["id"],
        run_id=run["id"],
        speaker_id=None,
        segment={
            "start_sample": 16_000,
            "end_sample": 96_000,
            "start_frame": 10,
            "end_frame": 60,
            "text": "mission briefing",
            "confidence": 1.0,
        },
        words=[],
    )
    repos.finish_analysis_run(db, run["id"], status="succeeded", diagnostics={})
    timeline = repos.create_timeline(
        db,
        project_id=project_id,
        name="Rough Cut",
        kind="rough_cut",
        created_from=asset["id"],
    )
    repos.add_timeline_clip(
        db,
        timeline_id=timeline["id"],
        asset_id=asset["id"],
        src_in_frame=0,
        src_out_frame_exclusive=600,
        seq_in_frame=0,
        seq_out_frame_exclusive=600,
    )
    repos.replace_scenes(db, project_id, timeline["id"], [(0, 300), (300, 600)])
    return str(asset["id"])


def _material(asset_id: str) -> dict[str, Any]:
    return {
        "source": "lexical",
        "ranking": [
            {
                "asset_id": asset_id,
                "display_name": "mission.mp4",
                "score": 2.0,
                "scene_hits": [
                    {"scene_number": 1, "snippet": "mission briefing", "score": 2.0}
                ],
            }
        ],
        "skipped": [],
    }


def _project(db: Database) -> str:
    return str(
        repos.create_project(
            db, name="p", rate_num=FPS, rate_den=1, drop_frame=False, workspace_root="/tmp/p"
        )["id"]
    )


def _call_real_scene_context_tool(
    db: Database,
    project_id: str,
    asset_id: str,
    scene_number: int,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    class FakeFunctionTool:
        def __init__(self, func: Any, *, name: str = "", description: str = "") -> None:
            self.func = func
            self.name = name
            self.description = description

    class FakeAssistantAgent:
        def __init__(
            self,
            *,
            name: str,
            model_client: object,
            tools: list[FakeFunctionTool],
            system_message: str,
            max_tool_iterations: int,
        ) -> None:
            del name, model_client, system_message, max_tool_iterations
            self.tools = tools

        async def run(self, *, task: str) -> Any:
            del task
            tool = next(tool for tool in self.tools if tool.name == "get_scene_context")
            captured["result"] = tool.func(asset_id=asset_id, scene_number=scene_number)
            return types.SimpleNamespace(messages=[types.SimpleNamespace(content="tool called")])

    core_tools = types.ModuleType("autogen_core.tools")
    core_tools.FunctionTool = FakeFunctionTool  # type: ignore[attr-defined]
    agentchat_agents = types.ModuleType("autogen_agentchat.agents")
    agentchat_agents.AssistantAgent = FakeAssistantAgent  # type: ignore[attr-defined]
    for name, module in {
        "autogen_core": types.ModuleType("autogen_core"),
        "autogen_core.tools": core_tools,
        "autogen_agentchat": types.ModuleType("autogen_agentchat"),
        "autogen_agentchat.agents": agentchat_agents,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(scout, "build_model_client", lambda config, role: object())

    scout._default_runner(db, resolve_from_env({}), project_id, "inspect scene")

    result = captured["result"]
    assert isinstance(result, dict)
    return result


def test_scene_context_tool_rejects_asset_from_another_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = _db(tmp_path)
    requested_project_id = _project(db)
    other_project_id = _project(db)
    other_asset_id = _seed_asset_with_scenes(db, other_project_id, "other.mp4")

    result = _call_real_scene_context_tool(
        db, requested_project_id, other_asset_id, 1, monkeypatch
    )

    assert result == {"ok": False, "reason": "asset is not in project"}


def test_scene_context_tool_does_not_create_missing_rough_cut(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = _db(tmp_path)
    project_id = _project(db)
    asset = repos.create_asset(
        db,
        project_id=project_id,
        type="video",
        display_name="raw.mp4",
        source_path="/tmp/raw.mp4",
    )

    result = _call_real_scene_context_tool(
        db, project_id, str(asset["id"]), 1, monkeypatch
    )

    assert result == {"ok": False, "reason": "no rough cut"}
    assert repos.get_asset_rough_cut(db, project_id, str(asset["id"])) is None


def test_scene_context_tool_preserves_same_project_scene_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = _db(tmp_path)
    project_id = _project(db)
    asset_id = _seed_asset_with_scenes(db, project_id, "mission.mp4")

    result = _call_real_scene_context_tool(db, project_id, asset_id, 1, monkeypatch)

    assert result == {
        "ok": True,
        "asset_id": asset_id,
        "scene_number": 1,
        "src_start_frame": 0,
        "src_end_frame_exclusive": 300,
        "text": "mission briefing",
    }


def test_valid_reply_is_adopted_and_task_embeds_ranked_material(tmp_path: Path) -> None:
    db = _db(tmp_path)
    project_id = _project(db)
    asset_id = _seed_asset_with_scenes(db, project_id, "mission.mp4")
    tasks: list[str] = []

    def runner(task: str) -> str:
        tasks.append(task)
        return '{"asset_id": "' + asset_id + '", "scene_numbers": [2], "rationale": "strong close"}'

    decision = scout.run_scout(
        db,
        resolve_from_env({}),
        project_id=project_id,
        topic="agent mission",
        material=_material(asset_id),
        runner=runner,
    )

    assert decision == {
        "asset_id": asset_id,
        "scene_numbers": [2],
        "rationale": "strong close",
        "fallback": False,
    }
    assert "agent mission" in tasks[0]
    assert asset_id in tasks[0]
    assert "mission.mp4" in tasks[0]
    assert "2.0" in tasks[0]
    assert "mission briefing" in tasks[0]


def test_valid_top_level_decision_with_nested_object_is_adopted(tmp_path: Path) -> None:
    db = _db(tmp_path)
    project_id = _project(db)
    asset_id = _seed_asset_with_scenes(db, project_id, "mission.mp4")

    decision = scout.run_scout(
        db,
        resolve_from_env({}),
        project_id=project_id,
        topic="mission",
        material=_material(asset_id),
        runner=lambda task: (
            '{"asset_id": "'
            + asset_id
            + '", "scene_numbers": [1], "rationale": "strong", "extra": {"note": "detail"}}'
        ),
    )

    assert decision == {
        "asset_id": asset_id,
        "scene_numbers": [1],
        "rationale": "strong",
        "fallback": False,
    }


def test_unknown_asset_retries_once_with_validation_error(tmp_path: Path) -> None:
    db = _db(tmp_path)
    project_id = _project(db)
    asset_id = _seed_asset_with_scenes(db, project_id, "mission.mp4")
    tasks: list[str] = []

    def runner(task: str) -> str:
        tasks.append(task)
        if len(tasks) == 1:
            return '{"asset_id": "unknown", "scene_numbers": [1], "rationale": "no"}'
        return '{"asset_id": "' + asset_id + '", "scene_numbers": [1], "rationale": "fixed"}'

    decision = scout.run_scout(
        db,
        resolve_from_env({}),
        project_id=project_id,
        topic="mission",
        material=_material(asset_id),
        runner=runner,
    )

    assert decision["fallback"] is False
    assert decision["rationale"] == "fixed"
    assert len(tasks) == 2
    assert "Validation error:" in tasks[1]
    assert "asset_id is not in the ranking" in tasks[1]


def test_whitespace_rationale_retries_once_then_uses_deterministic_fallback(
    tmp_path: Path,
) -> None:
    db = _db(tmp_path)
    project_id = _project(db)
    asset_id = _seed_asset_with_scenes(db, project_id, "mission.mp4")
    tasks: list[str] = []

    def runner(task: str) -> str:
        tasks.append(task)
        return (
            '{"asset_id": "'
            + asset_id
            + '", "scene_numbers": [1], "rationale": "  \\t  "}'
        )

    decision = scout.run_scout(
        db,
        resolve_from_env({}),
        project_id=project_id,
        topic="mission",
        material=_material(asset_id),
        runner=runner,
    )

    assert decision == {
        "asset_id": asset_id,
        "scene_numbers": [1],
        "rationale": "automatic fallback: top search score",
        "fallback": True,
    }
    assert len(tasks) == 2
    assert "rationale must not be empty" in tasks[1]


def test_valid_rationale_is_stripped(tmp_path: Path) -> None:
    db = _db(tmp_path)
    project_id = _project(db)
    asset_id = _seed_asset_with_scenes(db, project_id, "mission.mp4")

    decision = scout.run_scout(
        db,
        resolve_from_env({}),
        project_id=project_id,
        topic="mission",
        material=_material(asset_id),
        runner=lambda task: (
            '{"asset_id": "'
            + asset_id
            + '", "scene_numbers": [1], "rationale": "  strongest opening  "}'
        ),
    )

    assert decision["rationale"] == "strongest opening"
    assert decision["fallback"] is False


def test_invalid_replies_twice_use_deterministic_fallback(tmp_path: Path) -> None:
    db = _db(tmp_path)
    project_id = _project(db)
    asset_id = _seed_asset_with_scenes(db, project_id, "mission.mp4")

    decision = scout.run_scout(
        db,
        resolve_from_env({}),
        project_id=project_id,
        topic="mission",
        material=_material(asset_id),
        runner=lambda task: "not valid JSON",
    )

    assert decision == {
        "asset_id": asset_id,
        "scene_numbers": [1],
        "rationale": "automatic fallback: top search score",
        "fallback": True,
    }


def test_runner_exception_uses_fallback_without_retry(tmp_path: Path) -> None:
    db = _db(tmp_path)
    project_id = _project(db)
    asset_id = _seed_asset_with_scenes(db, project_id, "mission.mp4")
    calls = 0

    def runner(task: str) -> str:
        nonlocal calls
        del task
        calls += 1
        raise RuntimeError("provider unavailable")

    decision = scout.run_scout(
        db,
        resolve_from_env({}),
        project_id=project_id,
        topic="mission",
        material=_material(asset_id),
        runner=runner,
    )

    assert decision["fallback"] is True
    assert calls == 1


def test_runner_timeout_uses_exact_fallback_without_retry(tmp_path: Path) -> None:
    db = _db(tmp_path)
    project_id = _project(db)
    asset_id = _seed_asset_with_scenes(db, project_id, "mission.mp4")
    calls = 0

    def runner(task: str) -> str:
        nonlocal calls
        del task
        calls += 1
        raise TimeoutError("scout exceeded deadline")

    decision = scout.run_scout(
        db,
        resolve_from_env({}),
        project_id=project_id,
        topic="mission",
        material=_material(asset_id),
        runner=runner,
    )

    assert decision == {
        "asset_id": asset_id,
        "scene_numbers": [1],
        "rationale": "automatic fallback: top search score",
        "fallback": True,
    }
    assert calls == 1


def test_scene_universe_lookup_failure_uses_exact_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = _db(tmp_path)
    project_id = _project(db)
    asset_id = _seed_asset_with_scenes(db, project_id, "mission.mp4")

    def unavailable(db: Database, project_id: str, asset_id: str) -> None:
        del db, project_id, asset_id
        raise RuntimeError("scene index unavailable")

    monkeypatch.setattr(scout.discovery, "_scene_ranges", unavailable)

    decision = scout.run_scout(
        db,
        resolve_from_env({}),
        project_id=project_id,
        topic="mission",
        material=_material(asset_id),
        runner=lambda task: '{"asset_id": "unexpected", "scene_numbers": [1], "rationale": "x"}',
    )

    assert decision == {
        "asset_id": asset_id,
        "scene_numbers": [1],
        "rationale": "automatic fallback: top search score",
        "fallback": True,
    }


def test_empty_ranking_is_a_programming_error(tmp_path: Path) -> None:
    db = _db(tmp_path)

    with pytest.raises(ValueError, match="ranking"):
        scout.run_scout(
            db,
            resolve_from_env({}),
            project_id="project",
            topic="mission",
            material={"ranking": []},
            runner=lambda task: task,
        )
