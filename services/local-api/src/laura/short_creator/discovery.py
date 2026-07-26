"""Topic -> ranked material across the whole project (auto-short discovery layer).

Semantic when the [semantic] extra answers (the same index the search endpoint uses),
lexical fallback otherwise — one row shape either way. Hits are mapped onto each asset's
rough-cut scenes READ-ONLY: the ranking must never create timelines as a side effect
(spec 2026-07-21-auto-short-design.md §1).
"""

from __future__ import annotations

import logging
from typing import Any

from ..db import repos
from ..db.database import Database
from ..semantic import get_index
from . import context

logger = logging.getLogger(__name__)

_SNIPPET_CHARS = 160
_MAX_SCENE_SNIPPETS = 3


def _segment_hits(
    db: Database, project_id: str, topic: str, limit: int
) -> tuple[list[dict[str, Any]], str]:
    """(hits, source): semantic when the index exists AND answers, else lexical.
    Mirrors api/search.py's fallback stance — a broken index degrades, never raises."""
    index = get_index()
    if index is not None:
        try:
            hits = index.query(topic, project_id=project_id, limit=limit)
        except Exception:  # noqa: BLE001 - semantic search is best-effort
            logger.warning("semantic query failed; falling back to lexical", exc_info=True)
            hits = []
        if hits:
            return hits, "semantic"
    return (
        repos.search_transcript(db, project_id=project_id, query=topic, limit=limit),
        "lexical",
    )


def _scene_ranges(
    db: Database, project_id: str, asset_id: str
) -> list[tuple[int, int, int]] | None:
    """[(scene_number, src_start, src_end_exclusive)] for the asset's rough cut, or None
    when there is no rough cut / no scenes. Mirrors production_tools._resolve_scene's
    composition (list_scenes order_index+1, clips, context._scene_src_ranges) but strictly
    read-only."""
    timeline = repos.get_asset_rough_cut(db, project_id, asset_id)
    if timeline is None:
        return None
    scenes = repos.list_scenes(db, str(timeline["id"]))
    if not scenes:
        return None
    clips = repos.list_timeline_clips(db, str(timeline["id"]))
    resolved: list[tuple[int, int, int]] = []
    for scene in scenes:
        ranges = context._scene_src_ranges(
            clips,
            seq_in=int(scene["seq_in_frame"]),
            seq_out_exclusive=int(scene["seq_out_frame_exclusive"]),
        )
        if ranges:
            resolved.append((int(scene["order_index"]) + 1, ranges[0][0], ranges[-1][1]))
    return resolved or None


def search_material(
    db: Database, project_id: str, topic: str, *, limit: int = 40
) -> dict[str, Any]:
    """Rank project transcript hits by asset and map them to rough-cut scene numbers."""
    hits, source = _segment_hits(db, project_id, topic, limit)
    per_asset: dict[str, dict[str, Any]] = {}
    skipped: list[dict[str, str]] = []
    ranges_cache: dict[str, list[tuple[int, int, int]] | None] = {}
    for hit in hits:
        asset_id = str(hit["asset_id"])
        if asset_id not in ranges_cache:
            ranges_cache[asset_id] = _scene_ranges(db, project_id, asset_id)
            if ranges_cache[asset_id] is None:
                skipped.append({"asset_id": asset_id, "reason": "no rough cut"})
        ranges = ranges_cache[asset_id]
        if ranges is None:
            continue
        start = int(hit.get("start_frame", 0))
        scene_number = next((number for number, low, high in ranges if low <= start < high), None)
        if scene_number is None:
            continue
        score = float(hit.get("score") or 1.0)
        entry = per_asset.setdefault(
            asset_id,
            {
                "asset_id": asset_id,
                "display_name": str(hit.get("asset_name", "")),
                "score": 0.0,
                "scene_hits": [],
            },
        )
        entry["score"] += score
        entry["scene_hits"].append(
            {
                "scene_number": scene_number,
                "snippet": str(hit.get("text", ""))[:_SNIPPET_CHARS],
                "score": score,
            }
        )
    ranking = sorted(per_asset.values(), key=lambda entry: entry["score"], reverse=True)
    for entry in ranking:
        entry["scene_hits"] = sorted(
            entry["scene_hits"], key=lambda hit: (-hit["score"], hit["scene_number"])
        )[:_MAX_SCENE_SNIPPETS]
        entry["scene_hits"].sort(key=lambda hit: hit["scene_number"])
    return {"source": source, "ranking": ranking, "skipped": skipped}
