"""Store tests: lifecycle, scene-review versioning."""

from pathlib import Path

import pytest

from laura.short_creator.board import Board, downstream_of
from laura.short_creator.board_models import (  # noqa: E402
    BestWindow,
    BoardMeta,
    Chapter,
    ContactSheet,
    ContactSheetTile,
    Cutlist,
    CutSegment,
    Roi,
    SceneReview,
    Script,
    ScriptLine,
    Storyline,
)


def _meta() -> BoardMeta:
    return BoardMeta(session_id="s1", asset_id="a1", created_utc="2026-07-13T12:00:00Z",
                     task="overview short", target_seconds=60.0)


def _review(n: int = 1, desc: str = "dashboard") -> SceneReview:
    return SceneReview(
        scene_number=n, src_start_frame=0, src_end_frame_exclusive=120,
        description=desc, whats_happening="scrolling", hook_score=5,
        best_window=BestWindow(offset_s=0.5, duration_s=3.0),
        roi=Roi(x=0.1, y=0.1, w=0.5, h=0.5),
    )


def test_create_open_meta_roundtrip(tmp_path: Path) -> None:
    Board.create(tmp_path / "board", _meta())
    again = Board.open(tmp_path / "board")
    assert again.meta().session_id == "s1"
    assert (tmp_path / "board" / "scene_reviews").is_dir()
    assert (tmp_path / "board" / "versions").is_dir()


def test_open_missing_board_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Board.open(tmp_path / "nope")


def test_scene_review_save_versions_and_archives(tmp_path: Path) -> None:
    board = Board.create(tmp_path / "board", _meta())
    assert board.save_scene_review(_review(1, "first look")) == 1
    assert board.save_scene_review(_review(1, "second look")) == 2
    reviews = board.scene_reviews()
    assert len(reviews) == 1
    assert reviews[0].description == "second look" and reviews[0].version == 2
    archived = tmp_path / "board" / "versions" / "scene_1.v1.json"
    assert archived.is_file() and "first look" in archived.read_text(encoding="utf-8")


def test_scene_reviews_sorted_by_number(tmp_path: Path) -> None:
    board = Board.create(tmp_path / "board", _meta())
    board.save_scene_review(_review(10))
    board.save_scene_review(_review(2))
    assert [r.scene_number for r in board.scene_reviews()] == [2, 10]


def test_downstream_of() -> None:
    assert downstream_of("scene_reviews") == (
        "storyline", "script", "voice", "cutlist", "contact_sheet", "render_report", "qa_report")
    assert downstream_of("cutlist") == ("contact_sheet", "render_report", "qa_report")
    assert downstream_of("contact_sheet") == ("render_report", "qa_report")
    assert downstream_of("qa_report") == ()


# -- singleton artifacts --------------------------------------------------------


def _storyline(thread: str = "one app") -> Storyline:
    return Storyline(red_thread=thread, arc=[
        Chapter(chapter=1, role="hook", message="stop", scene_numbers=[1], target_seconds=3.0)])


def _script() -> Script:
    return Script(language="de", lines=[ScriptLine(chapter=1, scene_number=1, text="Stopp!")])


def _cutlist() -> Cutlist:
    return Cutlist(segments=[CutSegment(order=0, scene_number=1, start_frame=0,
                                        end_frame_exclusive=120)])


def _contact_sheet() -> ContactSheet:
    return ContactSheet(png_path="/tmp/sheet.png", cols=1, rows=1,
                        tiles=[ContactSheetTile(order=0, scene_number=1, frame=60, label="0 S1")])


def test_singleton_save_load_and_version_stamp(tmp_path: Path) -> None:
    board = Board.create(tmp_path / "board", _meta())
    assert board.save("storyline", _storyline("v1 thread")) == 1
    assert board.save("storyline", _storyline("v2 thread")) == 2
    loaded = board.load("storyline")
    assert isinstance(loaded, Storyline)
    assert loaded.version == 2 and loaded.red_thread == "v2 thread"
    assert board.versions("storyline") == [1]
    assert board.load("qa_report") is None


def test_save_rejects_wrong_type_and_unknown_name(tmp_path: Path) -> None:
    board = Board.create(tmp_path / "board", _meta())
    with pytest.raises(TypeError):
        board.save("storyline", _script())
    with pytest.raises(KeyError):
        board.save("nonsense", _storyline())
    with pytest.raises(KeyError):
        board.load("nonsense")


def test_save_invalidates_downstream(tmp_path: Path) -> None:
    board = Board.create(tmp_path / "board", _meta())
    board.save("storyline", _storyline())
    board.save("script", _script())
    board.save("cutlist", _cutlist())
    removed = board.invalidate("script")
    assert removed == ["cutlist"]
    assert board.load("cutlist") is None
    assert board.versions("cutlist") == [1]  # archived, not lost
    # save() itself invalidates downstream too:
    board.save("cutlist", _cutlist())
    board.save("storyline", _storyline("changed"))
    assert board.load("script") is None and board.load("cutlist") is None


def test_cutlist_save_archives_contact_sheet(tmp_path: Path) -> None:
    """A cutlist change invalidates the contact sheet (and everything downstream of it): the
    current sheet is archived — never lost — and gone from the board until save_contact_sheet
    reruns against the new cutlist."""
    board = Board.create(tmp_path / "board", _meta())
    board.save("storyline", _storyline())
    board.save("script", _script())
    board.save("cutlist", _cutlist())
    assert board.save("contact_sheet", _contact_sheet()) == 1
    loaded = board.load("contact_sheet")
    assert isinstance(loaded, ContactSheet)
    assert loaded.tiles[0].scene_number == 1

    board.save("cutlist", _cutlist())  # -> cutlist v2, invalidates contact_sheet + downstream

    assert board.load("contact_sheet") is None
    assert board.versions("contact_sheet") == [1]  # archived, not lost
    archived = tmp_path / "board" / "versions" / "contact_sheet.v1.json"
    assert archived.is_file() and "sheet.png" in archived.read_text(encoding="utf-8")


def test_revert_restores_and_invalidates(tmp_path: Path) -> None:
    board = Board.create(tmp_path / "board", _meta())
    board.save("storyline", _storyline("first"))
    board.save("storyline", _storyline("second"))
    board.save("script", _script())
    board.revert("storyline", 1)
    loaded = board.load("storyline")
    assert isinstance(loaded, Storyline) and loaded.red_thread == "first"
    assert board.load("script") is None  # downstream invalidated
    # v2 stays in the archive; saving after revert continues past ALL known versions
    assert 2 in board.versions("storyline")
    assert board.save("storyline", _storyline("third")) == 3


def test_revert_unknown_version_raises(tmp_path: Path) -> None:
    board = Board.create(tmp_path / "board", _meta())
    board.save("storyline", _storyline())
    with pytest.raises(FileNotFoundError):
        board.revert("storyline", 7)


# -- progress & status -------------------------------------------------------


def test_resume_point_progression(tmp_path: Path) -> None:
    board = Board.create(tmp_path / "board", _meta())
    assert board.resume_point([1, 2]) == "scene_reviews:1"
    board.save_scene_review(_review(1))
    assert board.resume_point([1, 2]) == "scene_reviews:2"
    board.save_scene_review(_review(2))
    assert board.resume_point([1, 2]) == "storyline"
    board.save("storyline", _storyline())
    assert board.resume_point([1, 2]) == "script"
    board.save("script", _script())
    assert board.resume_point([1, 2]) == "voice"


def test_resume_point_reflects_invalidation(tmp_path: Path) -> None:
    board = Board.create(tmp_path / "board", _meta())
    board.save_scene_review(_review(1))
    board.save("storyline", _storyline())
    board.save("script", _script())
    board.save("storyline", _storyline("changed"))  # invalidates script
    assert board.resume_point([1]) == "script"


def test_status_shape(tmp_path: Path) -> None:
    board = Board.create(tmp_path / "board", _meta())
    board.save_scene_review(_review(1))
    board.save("storyline", _storyline("a"))
    board.save("storyline", _storyline("b"))
    status = board.status()
    assert status["meta"]["session_id"] == "s1"
    assert status["scene_reviews"] == {"count": 1, "scenes": [1]}
    assert status["artifacts"]["storyline"] == {"version": 2, "archived_versions": [1]}
    assert status["artifacts"]["qa_report"] == {"version": None, "archived_versions": []}
