"""production_tools: the ``save_contact_sheet`` board tool (Kontaktbogen checkpoint).

Real-ffmpeg tests (skipped without ffmpeg/ffprobe on PATH, mirroring
``tests/test_zoom_e2e.py``): a synthetic mini-proxy is rendered with ``testsrc2`` and
registered as the asset's proxy file, the cutlist is seeded straight onto the board
(``save_contact_sheet`` reads segments from the CUTLIST — no rough-cut scenes needed), and the
tool's output PNG is verified with ffprobe (one grid image whose geometry matches the tile
count). PNG (not mjpeg) is the required frame codec — mjpeg breaks on non-full-range YUV.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from laura.config import Settings
from laura.db import repos
from laura.db.database import Database, SqliteDatabase
from laura.short_creator.board import Board
from laura.short_creator.board_models import (
    BoardMeta,
    ContactSheet,
    Cutlist,
    CutSegment,
)
from laura.short_creator.production_tools import build_production_tool_specs

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not on PATH",
)

FPS = 30
PROXY_W, PROXY_H = 320, 180
PROXY_SECONDS = 6  # 180 frames @ 30fps


def _make_proxy(path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size={PROXY_W}x{PROXY_H}:rate={FPS}:duration={PROXY_SECONDS}",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )


def _probe_size(path: Path) -> tuple[int, int]:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    stream = json.loads(out)["streams"][0]
    return int(stream["width"]), int(stream["height"])


def _seed_asset(tmp_path: Path, *, with_proxy: bool = True) -> tuple[Database, str]:
    """Project + asset (+ a real synthetic proxy registered via ``add_asset_file``)."""
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
    if with_proxy:
        proxy = tmp_path / "proxy.mp4"
        _make_proxy(proxy)
        repos.add_asset_file(db, asset_id=asset["id"], kind="proxy", path=str(proxy), is_proxy=True)
    return db, str(asset["id"])


def _board(tmp_path: Path, asset_id: str) -> Board:
    meta = BoardMeta(
        session_id="s1",
        asset_id=asset_id,
        created_utc="2026-07-15T00:00:00Z",
        task="overview short",
        target_seconds=20.0,
    )
    # Mirror the real layout (<...>/agent-runs/<sid>/board) so the tool's PNG output
    # directory (a SIBLING of the board dir) lands inside tmp_path's session dir.
    return Board.create(tmp_path / "agent-runs" / "s1" / "board", meta)


def _three_segment_cutlist() -> Cutlist:
    return Cutlist(
        segments=[
            CutSegment(order=0, scene_number=1, start_frame=0, end_frame_exclusive=60),
            CutSegment(order=1, scene_number=2, start_frame=60, end_frame_exclusive=120),
            CutSegment(order=2, scene_number=3, start_frame=120, end_frame_exclusive=180),
        ]
    )


def _specs(db: Database, board: Board, asset_id: str) -> dict[str, Any]:
    return {s.name: s for s in build_production_tool_specs(db, board, asset_id=asset_id)}


def test_save_contact_sheet_builds_grid_png_and_tile_list(tmp_path: Path) -> None:
    db, asset_id = _seed_asset(tmp_path)
    board = _board(tmp_path, asset_id)
    board.save("cutlist", _three_segment_cutlist())
    specs = _specs(db, board, asset_id)

    out = specs["save_contact_sheet"].func()

    assert out["ok"] is True, out
    assert out["version"] == 1
    # 3 tiles -> 2x2 grid (cols = ceil(sqrt(3)), rows = ceil(3/2)); last cell stays padding.
    assert (out["cols"], out["rows"]) == (2, 2)
    assert out["tiles"] == [
        {"order": 0, "scene_number": 1, "frame": 30, "label": "0 S1"},
        {"order": 1, "scene_number": 2, "frame": 90, "label": "1 S2"},
        {"order": 2, "scene_number": 3, "frame": 150, "label": "2 S3"},
    ]

    png = Path(out["png_path"])
    assert png.is_file() and png.stat().st_size > 0
    assert png.parent == board.root.parent / "contact_sheets"
    # Grid geometry: every tile is scaled to the fixed tile width (aspect preserved).
    width, height = _probe_size(png)
    assert width % out["cols"] == 0 and height % out["rows"] == 0
    tile_w, tile_h = width // out["cols"], height // out["rows"]
    assert tile_w / tile_h == pytest.approx(PROXY_W / PROXY_H, abs=0.02)

    saved = board.load("contact_sheet")
    assert isinstance(saved, ContactSheet)
    assert saved.png_path == out["png_path"]
    assert saved.version == 1
    assert [t.scene_number for t in saved.tiles] == [1, 2, 3]


def test_save_contact_sheet_versions_are_separate_files(tmp_path: Path) -> None:
    """A second save archives v1 on the board and writes a NEW png — the archived version's
    file keeps existing, so a revert's artifact still points at a real image."""
    db, asset_id = _seed_asset(tmp_path)
    board = _board(tmp_path, asset_id)
    board.save("cutlist", _three_segment_cutlist())
    specs = _specs(db, board, asset_id)

    first = specs["save_contact_sheet"].func()
    second = specs["save_contact_sheet"].func()

    assert first["ok"] and second["ok"]
    assert second["version"] == 2
    assert first["png_path"] != second["png_path"]
    assert Path(first["png_path"]).is_file()  # old version's image survives
    assert board.versions("contact_sheet") == [1]


def test_save_contact_sheet_requires_cutlist(tmp_path: Path) -> None:
    db, asset_id = _seed_asset(tmp_path)
    board = _board(tmp_path, asset_id)
    specs = _specs(db, board, asset_id)

    out = specs["save_contact_sheet"].func()

    assert out["ok"] is False
    assert "build_cutlist" in out["reason"]


def test_save_contact_sheet_without_proxy_degrades(tmp_path: Path) -> None:
    db, asset_id = _seed_asset(tmp_path, with_proxy=False)
    board = _board(tmp_path, asset_id)
    board.save("cutlist", _three_segment_cutlist())
    specs = _specs(db, board, asset_id)

    out = specs["save_contact_sheet"].func()

    assert out["ok"] is False
    assert "proxy" in out["reason"]
    assert board.load("contact_sheet") is None
