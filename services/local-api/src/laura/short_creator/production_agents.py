"""The v2 production-team agent roster (AutoGen ``AssistantAgent``s) — Slice 3, Task 7.

Five specialists turn a reviewed rough cut into a shipped short by writing to, and reading from,
the Production Board (:mod:`.board`) — never talking directly to each other about content, only
through board artifacts (scene reviews -> storyline -> script -> voice/cutlist/render -> QA). The
pipeline is a fixed judgment chain: ``vision_reviewer`` (what is actually on screen) ->
``story_architect`` (the arc) -> ``scene_author`` (the words) -> ``coding_agent`` (voice, cutlist,
contact sheet, render — execution only, plus reverting the board to an earlier artifact version
when explicitly asked to) -> ``qa_reviewer`` (ship or revise on the rendered result). Each tool
name in the roster
is cross-checked in tests against :func:`production_tools.build_production_tool_specs`'s real
tool names.

Mirrors :mod:`.agents`/:mod:`.magentic`'s split: :func:`production_agent_specs` is pure data (no
autogen, no db — testable standalone) and :func:`build_production_team` is the only
autogen-touching constructor here, lazily importing the optional ``autoshort`` extra and wiring
one shared agent-role model client plus a separate orchestrator-role client into a
``MagenticOneGroupChat``, exactly like :func:`laura.short_creator.magentic.build_magentic_team`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..db.database import Database
from .agents import AgentSpec
from .board import Board
from .production_tools import ProductionDeps, build_production_tool_specs
from .providers import AgentConfig, Stage, build_model_client

if TYPE_CHECKING:  # annotation only — never imported at runtime
    from autogen_agentchat.teams import MagenticOneGroupChat

# Hard cap on the orchestrator's turn budget — a runaway-loop backstop (mirrors magentic.MAX_TURNS).
MAX_TURNS = 30


def production_agent_specs() -> list[AgentSpec]:
    """The fixed v2 production-team roster. Pure — no autogen, no db.

    Tool names must exist in :func:`production_tools.build_production_tool_specs` (cross-checked
    in tests). The five specialists judge in a fixed pipeline, each reading and writing the
    Production Board rather than talking to each other directly: ``vision_reviewer`` reviews every
    scene with the VLM, ``story_architect`` structures reviewed scenes into the viral arc,
    ``scene_author`` writes the spoken script from that arc, ``coding_agent`` executes voice ->
    cutlist -> contact sheet -> render without changing content, and ``qa_reviewer`` judges the
    rendered result.
    """
    return [
        AgentSpec(
            name="vision_reviewer",
            description="Reviews every rough-cut scene with the VLM before anything else runs.",
            system_message=(
                "You are the Vision Reviewer. You judge only what you SEE — never invent visual "
                "details or guess from a scene number alone; the tool's review is the record, not "
                "your assumption. Call board_status first to see which scenes are expected and "
                "which already have a review. For every expected scene without a review, call "
                "get_scene_context to orient yourself on its transcript and frame range, then call "
                "review_scene so the VLM looks at its actual frames and the review is written to "
                "the board. Keep going until every expected scene has been reviewed — a scene "
                "with no review blocks the whole team downstream, including yours. When every "
                "scene is covered, call get_reviews and report which scenes are strongest "
                "(highest hook_score) and which are weak, so the Story Architect knows what "
                "material the arc can actually be built from."
            ),
            tool_names=("board_status", "get_scene_context", "review_scene", "get_reviews"),
            max_tool_iterations=10,
        ),
        AgentSpec(
            name="story_architect",
            description="Structures the reviewed scenes into the fixed viral short arc.",
            system_message=(
                "You are the Story Architect. Fill the FIXED viral arc — chapter roles are "
                "exactly hook, problem, feature, payoff_cta: hook (2-3s) -> problem -> 3-4 "
                "feature chapters (2-3 scenes each, building on each other) -> payoff_cta. "
                "First call get_reviews to see what material is actually "
                "available, and get_scene_context for detail on any scene you are unsure about — "
                "use ONLY reviewed scenes, never a scene number that has no review yet. A "
                "first-time viewer who has never seen the source video must be able to follow "
                "every step of the arc without confusion or a missing link. Check board_status if "
                "you need the current resume point. Save your storyline via save_storyline; if it "
                "returns validation errors — an unreviewed scene, a malformed chapter — fix "
                "exactly what it names and save again. Confirm with get_storyline once accepted."
            ),
            tool_names=(
                "get_reviews",
                "get_scene_context",
                "save_storyline",
                "get_storyline",
                "board_status",
            ),
            max_tool_iterations=4,
        ),
        AgentSpec(
            name="scene_author",
            description="Writes the spoken script line by line from the storyline's scenes.",
            system_message=(
                "You are the Scene Author. First call get_storyline to see the arc and its "
                "chapters, then get_scene_context (and get_reviews for visual detail) for each "
                "scene you write for. Write 1-2 sentences per scene, per chapter, in the video's "
                "language (German) — never switch languages mid-script. Write each chapter's "
                "lines in the SAME scene order the storyline lists that chapter's scenes in — "
                "voice and captions play back in that order, not the order you type them in. "
                "Ground every sentence in what the review says is VISIBLE; never invent a claim "
                "the scene does not support. Keep the tone energetic with concrete value, no "
                "marketing fog, no sleepy phrasing — this has to hook a cold, scrolling viewer "
                "in the first seconds. Save chapter by chapter via save_script_chapter (it "
                "merges — other chapters' lines stay untouched); fix and resave on validation "
                "errors. Verify with get_script once every chapter in the storyline has its "
                "lines written."
            ),
            tool_names=(
                "get_storyline",
                "get_reviews",
                "get_scene_context",
                "save_script_chapter",
                "get_script",
            ),
            max_tool_iterations=6,
        ),
        AgentSpec(
            name="coding_agent",
            description="Executes voice, cutlist and render — never changes content decisions.",
            system_message=(
                "You are the Coding Agent. You execute; you do not change content decisions made "
                "by the Story Architect or Scene Author. Check board_status, then read the "
                "storyline (get_storyline), script (get_script) and reviews (get_reviews) so you "
                "know what you are rendering. Pipeline, strictly in order: "
                "synthesize_script_voice -> build_cutlist -> save_contact_sheet -> "
                "render_production. save_contact_sheet is the user's visual pre-render "
                "checkpoint — NEVER skip it, and re-run it after EVERY build_cutlist call (a "
                "cutlist save archives the current sheet). If the task or a user message says "
                "to stop at the contact sheet, end after save_contact_sheet and report its "
                "tiles instead of rendering. Read the checks "
                "render_production returns: if voice does not fit (voice_fits is false), do not "
                "shorten it — build_cutlist already sizes segments to the voice's chapter audio "
                "windows, so rebuild the cutlist (then a fresh contact sheet) and render again; "
                "if it STILL fails, the scenes "
                "ran out of material — report that the storyline needs more/longer scenes. NEVER "
                "cut the voice — it is the script the team already agreed on, and the video must "
                "fit it, not the other way round. "
                "Report the export_id and the checks verbatim so the QA Reviewer can verify them "
                "independently instead of trusting your summary. Use revert_artifact ONLY when "
                "the task or user message explicitly asks to go back to an earlier version — "
                "never as a routine step; afterwards rebuild whatever it invalidated through the "
                "normal pipeline, not by hand."
            ),
            tool_names=(
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
            max_tool_iterations=10,
        ),
        AgentSpec(
            name="qa_reviewer",
            description="Judges the rendered export before it ships.",
            system_message=(
                "You are the QA Reviewer, the last gate before anything ships. Judge the "
                "RENDERED result, not the plan — start from board_status, then read the storyline "
                "(get_storyline) and script (get_script) so you know what a cold viewer should "
                "experience. Call review_export to look at real frames of the finished video and "
                "check story flow for a cold viewer, caption sync, zoom legibility, and that the "
                "full voice is present with nothing clipped mid-word. Your verdict is ship or "
                "revise, saved via save_qa_report together with concrete findings — every finding "
                "must name WHERE (which scene or timestamp) and WHAT is wrong, no vague words "
                "like 'looks off'. Use the board's contact sheet as your segment map — its "
                "tiles are labeled '<order> S<scene_number>' — and if it is missing or stale "
                "after a cutlist change, call save_contact_sheet to restore the user's visual "
                "checkpoint. If save_qa_report returns validation errors, fix exactly what "
                "it names and save again."
            ),
            tool_names=(
                "board_status",
                "get_storyline",
                "get_script",
                "review_export",
                "save_contact_sheet",
                "save_qa_report",
            ),
            max_tool_iterations=6,
        ),
    ]


def build_production_team(
    db: Database,
    board: Board,
    config: AgentConfig,
    *,
    asset_id: str,
    stage: Stage = "A",
    deps: ProductionDeps | None = None,
) -> MagenticOneGroupChat:
    """Assemble the v2 production Magentic-One team (roster + orchestrator model client).

    Mirrors :func:`laura.short_creator.magentic.build_magentic_team`: lazy autogen import, one
    shared agent-role model client for *stage*, and a separate orchestrator-role client (also
    *stage*-scoped). Tools are the in-process ``FunctionTool``s built from :func:`production_tools.
    build_production_tool_specs`'s ``ToolSpec``s for *asset_id* (the same lazy-FunctionTool-wrap
    pattern as :func:`laura.short_creator.toolset.build_function_tools`, inlined here since that
    builder needs ``board``/``asset_id``/``deps`` that the plain short-creator toolset does not).
    Raises a clear :class:`RuntimeError` if the optional ``autoshort`` extra is missing.
    """
    try:
        from autogen_agentchat.agents import AssistantAgent
        from autogen_agentchat.teams import MagenticOneGroupChat
        from autogen_core.tools import FunctionTool
    except ImportError as exc:
        raise RuntimeError(
            "The short-creator needs the optional 'autoshort' extra. "
            "Install it with: uv sync --extra autoshort"
        ) from exc

    tool_specs = build_production_tool_specs(db, board, asset_id=asset_id, deps=deps)
    tools_by_name = {
        spec.name: FunctionTool(spec.func, name=spec.name, description=spec.description)
        for spec in tool_specs
    }
    model_client = build_model_client(config, role="agent", stage=stage)
    # list[Any]: participants wants list[ChatAgent]; AssistantAgent subclasses it, but list is
    # invariant (and ChatAgent is only importable with the extra installed).
    agents: list[Any] = [
        AssistantAgent(
            name=spec.name,
            model_client=model_client,
            tools=[tools_by_name[n] for n in spec.tool_names if n in tools_by_name],
            description=spec.description,
            system_message=spec.system_message,
            max_tool_iterations=spec.max_tool_iterations,
        )
        for spec in production_agent_specs()
    ]
    orchestrator = build_model_client(config, role="orchestrator", stage=stage)
    return MagenticOneGroupChat(participants=agents, model_client=orchestrator, max_turns=MAX_TURNS)
