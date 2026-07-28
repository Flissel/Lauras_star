"""search_material: topic -> ranked assets + rough-cut scene hits across the whole project.

The discovery layer under auto-short (spec 2026-07-21-auto-short-design.md §1): semantic
when the [semantic] extra answers, lexical fallback otherwise, hits mapped onto each
asset's rough-cut scenes READ-ONLY — ranking must never create timelines as a side effect.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from laura.config import Settings
from laura.db import repos
from laura.db.database import Database, SqliteDatabase
from laura.short_creator import discovery
from laura.util import new_id

FPS = 30


def _db(tmp_path: Path) -> Database:
    settings = Settings(workspace_root=tmp_path / "ws", start_runner=False)
    db: Database = SqliteDatabase(settings.db_path)
    db.migrate()
    return db


def _seed_asset_with_scenes(
    db: Database, project_id: str, name: str, *, segments: list[tuple[int, int, str]]
) -> str:
    """Asset + succeeded analysis run with *segments* (start_frame, end_frame, text) + a
    rough-cut timeline with one 1:1 clip over [0, 600) and two scenes [0,300)/[300,600)."""
    asset = repos.create_asset(
        db, project_id=project_id, type="video", display_name=name, source_path=f"/tmp/{name}"
    )
    run = repos.create_analysis_run(db, asset_id=asset["id"], pipeline_version="t", config={})
    repos.start_analysis_run(db, run["id"])
    for start, end, text in segments:
        repos.insert_segment_with_words(
            db,
            asset_id=asset["id"],
            run_id=run["id"],
            speaker_id=None,
            segment={
                "start_sample": start * 1600,
                "end_sample": end * 1600,
                "start_frame": start,
                "end_frame": end,
                "text": text,
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


def test_lexical_ranking_maps_hits_to_scenes_and_ranks_assets(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setattr(discovery, "get_index", lambda: None)  # force lexical
    db = _db(tmp_path)
    project = repos.create_project(
        db, name="p", rate_num=FPS, rate_den=1, drop_frame=False, workspace_root="/tmp/p"
    )
    strong = _seed_asset_with_scenes(
        db,
        project["id"],
        "strong.mp4",
        segments=[
            (10, 60, "the agent farm plans the mission"),
            (320, 380, "agents execute the mission end to end"),
        ],
    )
    weak = _seed_asset_with_scenes(
        db,
        project["id"],
        "weak.mp4",
        segments=[(10, 60, "unrelated dashboard talk"), (320, 380, "one mission mention only")],
    )

    out = discovery.search_material(db, project["id"], "mission")

    assert out["source"] == "lexical"
    assert [r["asset_id"] for r in out["ranking"]] == [strong, weak]
    top = out["ranking"][0]
    # scene 1 covers src [0,300), scene 2 covers [300,600)
    assert [h["scene_number"] for h in top["scene_hits"]] == [1, 2]
    assert "mission" in top["scene_hits"][0]["snippet"]
    assert out["skipped"] == []


def test_scene_mapping_excludes_transcript_in_gap_between_disjoint_source_ranges(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setattr(discovery, "get_index", lambda: None)
    db = _db(tmp_path)
    project = repos.create_project(
        db, name="p", rate_num=FPS, rate_den=1, drop_frame=False, workspace_root="/tmp/p"
    )
    asset = repos.create_asset(
        db,
        project_id=project["id"],
        type="video",
        display_name="disjoint.mp4",
        source_path="/tmp/disjoint.mp4",
    )
    run = repos.create_analysis_run(
        db, asset_id=asset["id"], pipeline_version="t", config={}
    )
    repos.start_analysis_run(db, run["id"])
    for start, end, text in (
        (10, 20, "mission valid opening"),
        (150, 160, "mission gap must be excluded"),
        (310, 320, "mission valid closing"),
    ):
        repos.insert_segment_with_words(
            db,
            asset_id=asset["id"],
            run_id=run["id"],
            speaker_id=None,
            segment={
                "start_sample": start * 1600,
                "end_sample": end * 1600,
                "start_frame": start,
                "end_frame": end,
                "text": text,
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
    for src_in, src_out, seq_in, seq_out in (
        (0, 100, 0, 100),
        (300, 400, 100, 200),
    ):
        repos.add_timeline_clip(
            db,
            timeline_id=timeline["id"],
            asset_id=asset["id"],
            src_in_frame=src_in,
            src_out_frame_exclusive=src_out,
            seq_in_frame=seq_in,
            seq_out_frame_exclusive=seq_out,
        )
    repos.replace_scenes(db, project["id"], timeline["id"], [(0, 200)])

    out = discovery.search_material(db, project["id"], "mission")

    assert len(out["ranking"]) == 1
    ranked = out["ranking"][0]
    assert ranked["score"] == 2.0
    assert [hit["snippet"] for hit in ranked["scene_hits"]] == [
        "mission valid opening",
        "mission valid closing",
    ]


def test_asset_without_rough_cut_is_skipped_not_created(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(discovery, "get_index", lambda: None)
    db = _db(tmp_path)
    project = repos.create_project(
        db, name="p", rate_num=FPS, rate_den=1, drop_frame=False, workspace_root="/tmp/p"
    )
    # Asset with a transcript hit but NO rough-cut timeline at all.
    asset = repos.create_asset(
        db,
        project_id=project["id"],
        type="video",
        display_name="raw.mp4",
        source_path="/tmp/raw.mp4",
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
            "end_sample": 16000,
            "start_frame": 0,
            "end_frame": 50,
            "text": "mission talk",
            "confidence": 1.0,
        },
        words=[],
    )
    repos.finish_analysis_run(db, run["id"], status="succeeded", diagnostics={})

    out = discovery.search_material(db, project["id"], "mission")

    assert out["ranking"] == []
    assert out["skipped"] == [{"asset_id": str(asset["id"]), "reason": "no rough cut"}]
    # READ-ONLY invariant: discovery must not have created a timeline as a side effect.
    assert repos.get_asset_rough_cut(db, project["id"], str(asset["id"])) is None


def test_no_hits_is_an_empty_ranking_not_an_error(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(discovery, "get_index", lambda: None)
    db = _db(tmp_path)
    project = repos.create_project(
        db, name="p", rate_num=FPS, rate_den=1, drop_frame=False, workspace_root="/tmp/p"
    )
    _seed_asset_with_scenes(db, project["id"], "a.mp4", segments=[(10, 60, "hello world")])

    out = discovery.search_material(db, project["id"], "quantum chromodynamics")

    assert out == {"source": "lexical", "ranking": [], "skipped": []}


def test_index_acquisition_failure_falls_back_to_lexical(tmp_path: Path, monkeypatch: Any) -> None:
    def raise_on_get_index() -> None:
        raise RuntimeError("semantic index is unavailable")

    monkeypatch.setattr(discovery, "get_index", raise_on_get_index)
    db = _db(tmp_path)
    project = repos.create_project(
        db, name="p", rate_num=FPS, rate_den=1, drop_frame=False, workspace_root="/tmp/p"
    )
    asset_id = _seed_asset_with_scenes(
        db, project["id"], "fallback.mp4", segments=[(10, 60, "mission talk")]
    )

    out = discovery.search_material(db, project["id"], "mission")

    assert out["source"] == "lexical"
    assert [entry["asset_id"] for entry in out["ranking"]] == [asset_id]


def test_semantic_zero_score_is_not_replaced_with_lexical_default(
    tmp_path: Path, monkeypatch: Any
) -> None:
    class ZeroScoreIndex:
        def query(self, text: str, *, project_id: str, limit: int) -> list[dict[str, Any]]:
            del text, limit
            return [
                {
                    "score": 0.0,
                    "project_id": project_id,
                    "asset_id": asset_id,
                    "segment_id": "segment",
                    "asset_name": "zero.mp4",
                    "text": "mission talk",
                    "start_frame": 10,
                    "end_frame": 60,
                    "speaker_label": None,
                }
            ]

    db = _db(tmp_path)
    project = repos.create_project(
        db, name="p", rate_num=FPS, rate_den=1, drop_frame=False, workspace_root="/tmp/p"
    )
    asset_id = _seed_asset_with_scenes(
        db, project["id"], "zero.mp4", segments=[(10, 60, "mission talk")]
    )
    monkeypatch.setattr(discovery, "get_index", lambda: ZeroScoreIndex())

    out = discovery.search_material(db, project["id"], "mission")

    assert out["source"] == "semantic"
    assert out["ranking"][0]["score"] == 0.0
    assert out["ranking"][0]["scene_hits"][0]["score"] == 0.0


def test_rough_cut_without_scenes_is_skipped_with_its_own_reason(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setattr(discovery, "get_index", lambda: None)
    db = _db(tmp_path)
    project = repos.create_project(
        db, name="p", rate_num=FPS, rate_den=1, drop_frame=False, workspace_root="/tmp/p"
    )
    asset = repos.create_asset(
        db,
        project_id=project["id"],
        type="video",
        display_name="unscened.mp4",
        source_path="/tmp/u",
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
            "end_sample": 16000,
            "start_frame": 0,
            "end_frame": 50,
            "text": "mission talk",
            "confidence": 1.0,
        },
        words=[],
    )
    repos.finish_analysis_run(db, run["id"], status="succeeded", diagnostics={})
    repos.create_timeline(
        db, project_id=project["id"], name="Rough Cut", kind="rough_cut", created_from=asset["id"]
    )

    out = discovery.search_material(db, project["id"], "mission")

    assert out["ranking"] == []
    assert out["skipped"] == [{"asset_id": str(asset["id"]), "reason": "no scenes"}]


def test_semantic_ranking_uses_index_when_semantic_extra_is_available(
    tmp_path: Path, monkeypatch: Any
) -> None:
    pytest.importorskip("fastembed")
    pytest.importorskip("qdrant_client")
    from onnxruntime.capi.onnxruntime_pybind11_state import (  # type: ignore[import-untyped]
        NoSuchFile,
    )

    from laura.semantic import SemanticIndex

    db = _db(tmp_path)
    project = repos.create_project(
        db, name="p", rate_num=FPS, rate_den=1, drop_frame=False, workspace_root="/tmp/p"
    )
    asset_id = _seed_asset_with_scenes(
        db,
        project["id"],
        "semantic.mp4",
        segments=[
            (10, 60, "the agent farm plans the mission"),
            (320, 380, "agents execute the mission end to end"),
        ],
    )
    index = SemanticIndex(":memory:")
    try:
        index.index(
            [
                {
                    "id": new_id(),
                    "text": "the agent farm plans the mission",
                    "payload": {
                        "project_id": project["id"],
                        "asset_id": asset_id,
                        "segment_id": new_id(),
                        "asset_name": "semantic.mp4",
                        "text": "the agent farm plans the mission",
                        "start_frame": 10,
                        "end_frame": 60,
                        "speaker_label": None,
                    },
                },
                {
                    "id": new_id(),
                    "text": "agents execute the mission end to end",
                    "payload": {
                        "project_id": project["id"],
                        "asset_id": asset_id,
                        "segment_id": new_id(),
                        "asset_name": "semantic.mp4",
                        "text": "agents execute the mission end to end",
                        "start_frame": 320,
                        "end_frame": 380,
                        "speaker_label": None,
                    },
                },
            ]
        )
    except NoSuchFile as exc:  # known optional FastEmbed model/cache unavailability
        pytest.skip(f"semantic model unavailable: {exc}")
    monkeypatch.setattr(discovery, "get_index", lambda: index)

    out = discovery.search_material(db, project["id"], "agent mission")

    assert out["source"] == "semantic"
    assert out["ranking"]
