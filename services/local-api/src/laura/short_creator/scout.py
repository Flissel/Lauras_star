"""A bounded single-agent scout with validated, deterministic fallback selection."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from typing import Any, TypedDict

from ..db import repos
from ..db.database import Database
from . import discovery, production_tools
from .providers import AgentConfig, build_model_client

logger = logging.getLogger(__name__)

_SCOUT_TIMEOUT_S = 60.0


class ScoutDecision(TypedDict):
    """One asset and its rough-cut scenes chosen for the proposed short."""

    asset_id: str
    scene_numbers: list[int]
    rationale: str
    fallback: bool


def _task_text(topic: str, ranking: list[dict[str, Any]], error: str | None = None) -> str:
    task = (
        "Choose the strongest material for a short about this topic:\n"
        f"{topic}\n\n"
        "The ranked material is below. Use the tools only for more depth; the ranking is enough "
        "to decide. Return only a JSON object with this exact shape: "
        '{"asset_id": str, "scene_numbers": [int], "rationale": str}.\n\n'
        f"Ranking:\n{json.dumps(ranking, ensure_ascii=False)}"
    )
    return task if error is None else f"{task}\n\nValidation error: {error}. Correct the JSON."


def _fallback(ranking: list[dict[str, Any]]) -> ScoutDecision:
    top = ranking[0]
    return {
        "asset_id": str(top["asset_id"]),
        "scene_numbers": [int(hit["scene_number"]) for hit in top["scene_hits"]],
        "rationale": "automatic fallback: top search score",
        "fallback": True,
    }


def _last_json_object(reply: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    last: dict[str, Any] | None = None
    for position, character in enumerate(reply):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(reply[position:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            last = value
    return last


def _known_scene_numbers(
    db: Database, project_id: str, ranking: list[dict[str, Any]]
) -> dict[str, set[int]]:
    known: dict[str, set[int]] = {}
    for entry in ranking:
        asset_id = str(entry["asset_id"])
        scene_numbers = {int(hit["scene_number"]) for hit in entry["scene_hits"]}
        ranges, _ = discovery._scene_ranges(db, project_id, asset_id)
        scene_numbers.update(number for number, _, _ in ranges)
        known[asset_id] = scene_numbers
    return known


def _validate_reply(
    reply: str, known_scenes: dict[str, set[int]]
) -> tuple[ScoutDecision | None, str | None]:
    payload = _last_json_object(reply)
    if payload is None:
        return None, "final reply has no JSON object"
    asset_id = payload.get("asset_id")
    if not isinstance(asset_id, str) or asset_id not in known_scenes:
        return None, "asset_id is not in the ranking"
    scene_numbers = payload.get("scene_numbers")
    if (
        not isinstance(scene_numbers, list)
        or not scene_numbers
        or any(not isinstance(number, int) or isinstance(number, bool) for number in scene_numbers)
    ):
        return None, "scene_numbers must be a non-empty list of integers"
    if not set(scene_numbers).issubset(known_scenes[asset_id]):
        return None, "scene_numbers include an unknown scene for asset_id"
    rationale = payload.get("rationale")
    if not isinstance(rationale, str):
        return None, "rationale must be a string"
    return {
        "asset_id": asset_id,
        "scene_numbers": scene_numbers,
        "rationale": rationale,
        "fallback": False,
    }, None


async def _run_agent(agent: Any, task: str) -> str:
    result = await agent.run(task=task)
    messages = getattr(result, "messages", None) or []
    if not messages:
        raise RuntimeError("scout agent returned no messages")
    content = getattr(messages[-1], "content", "")
    return content if isinstance(content, str) else str(content)


def _default_runner(db: Database, config: AgentConfig, project_id: str, task: str) -> str:
    """Build one in-process, bounded AutoGen scout and return its final reply."""
    try:
        from autogen_agentchat.agents import AssistantAgent
        from autogen_core.tools import FunctionTool
    except ImportError as exc:
        raise RuntimeError(
            "The short-creator needs the optional 'autoshort' extra. "
            "Install it with: uv sync --extra autoshort"
        ) from exc

    def search_material(topic: str) -> dict[str, Any]:
        """Search and rank project material for a topic."""
        return discovery.search_material(db, project_id, topic)

    def list_project_assets() -> list[dict[str, Any]]:
        """List the project's available assets."""
        return repos.list_assets(db, project_id)

    def get_scene_context(asset_id: str, scene_number: int) -> dict[str, Any]:
        """Get one rough-cut scene's source range and transcript context."""
        resolved = production_tools._resolve_scene(db, asset_id, scene_number)
        if resolved is None:
            return {"ok": False, "reason": "unknown scene"}
        src_start, src_end_exclusive, text = resolved
        return {
            "ok": True,
            "asset_id": asset_id,
            "scene_number": scene_number,
            "src_start_frame": src_start,
            "src_end_frame_exclusive": src_end_exclusive,
            "text": text,
        }

    agent = AssistantAgent(
        name="scout",
        model_client=build_model_client(config, role="agent"),
        tools=[
            FunctionTool(
                search_material, name="search_material", description=search_material.__doc__ or ""
            ),
            FunctionTool(
                list_project_assets,
                name="list_project_assets",
                description=list_project_assets.__doc__ or "",
            ),
            FunctionTool(
                get_scene_context,
                name="get_scene_context",
                description=get_scene_context.__doc__ or "",
            ),
        ],
        system_message=(
            "You are Laura's scout. Pick only ranked assets and valid rough-cut scene numbers. "
            "Return the requested JSON as your final answer."
        ),
        max_tool_iterations=3,
    )
    return asyncio.run(asyncio.wait_for(_run_agent(agent, task), timeout=_SCOUT_TIMEOUT_S))


def run_scout(
    db: Database,
    config: AgentConfig,
    *,
    project_id: str,
    topic: str,
    material: dict[str, Any],
    runner: Callable[[str], str] | None = None,
) -> ScoutDecision:
    """Ask the scout once, retry one validation failure, then always return a safe choice."""
    ranking_value = material.get("ranking")
    if not isinstance(ranking_value, list) or not ranking_value:
        raise ValueError("material ranking must be non-empty")
    ranking = [entry for entry in ranking_value if isinstance(entry, dict)]
    if len(ranking) != len(ranking_value):
        raise ValueError("material ranking entries must be objects")

    known_scenes = _known_scene_numbers(db, project_id, ranking)
    task = _task_text(topic, ranking)
    invoke = runner or (lambda prompt: _default_runner(db, config, project_id, prompt))
    try:
        reply = invoke(task)
    except (Exception, TimeoutError) as exc:
        logger.warning("scout failed; using deterministic fallback: %s", exc)
        return _fallback(ranking)
    decision, error = _validate_reply(reply, known_scenes)
    if decision is not None:
        return decision

    assert error is not None
    try:
        reply = invoke(_task_text(topic, ranking, error))
    except (Exception, TimeoutError) as exc:
        logger.warning("scout retry failed; using deterministic fallback: %s", exc)
        return _fallback(ranking)
    decision, _ = _validate_reply(reply, known_scenes)
    return decision if decision is not None else _fallback(ranking)
