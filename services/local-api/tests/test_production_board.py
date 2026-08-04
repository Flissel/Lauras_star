"""Store tests: lifecycle, scene-review versioning."""

import json
import threading
from pathlib import Path

import pytest
from pydantic import ValidationError

from laura.short_creator.board import Board, _write_atomic, downstream_of
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

    # A CHANGED cutlist -> v2, invalidating the contact sheet built against the old one. (An
    # identical re-save is a no-op now — see test_saving_identical_content_does_not_invalidate
    # — so the change has to be real to exercise the chain propagation this test is about.)
    changed_cutlist = Cutlist(segments=[CutSegment(
        order=0, scene_number=1, start_frame=0, end_frame_exclusive=240)])
    board.save("cutlist", changed_cutlist)

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
    assert status["scene_reviews"] == {
        "count": 1,
        "scenes": [1],
        # A count alone cannot tell a reviewed board from one whose VLM never ran.
        "degraded_count": 0,
        "degraded_scenes": [],
    }
    assert status["artifacts"]["storyline"] == {"version": 2, "archived_versions": [1]}
    assert status["artifacts"]["qa_report"] == {"version": None, "archived_versions": []}


def test_saving_identical_content_does_not_invalidate_downstream(tmp_path: Path) -> None:
    """Live finding: an agent re-saved upstream artifacts three times in one run. Each save
    wiped the chain below, it rebuilt, and the turn budget ran out with only voice on the
    board and no render. A save that changes nothing has made nothing stale."""
    board = Board.create(tmp_path / "board", _meta())
    board.save("storyline", _storyline("one app"))
    board.save("script", _script())
    board.save("cutlist", _cutlist())

    version = board.save("storyline", _storyline("one app"))  # identical content

    assert board.load("script") is not None, "an unchanged storyline must not drop the script"
    assert board.load("cutlist") is not None
    assert version == 1, "an unchanged save keeps the current version — it is not a new one"
    assert board.versions("storyline") == [], "and it archives nothing"


def test_saving_changed_content_still_invalidates_downstream(tmp_path: Path) -> None:
    """The contract that matters survives untouched: a real change makes downstream stale."""
    board = Board.create(tmp_path / "board", _meta())
    board.save("storyline", _storyline("one app"))
    board.save("script", _script())

    version = board.save("storyline", _storyline("a different thread"))

    assert board.load("script") is None, "a changed storyline must drop the script"
    assert version == 2


# -- corruption & concurrency (live incident 2026-08-04) -------------------------
#
# A production team turn fired five save_script_chapter tool calls IN PARALLEL (AutoGen
# executes a turn's tool calls concurrently). Every _write_atomic used the same tmp path
# (script.json.tmp), so two threads truncated/wrote the one tmp file at interleaved
# offsets and replace() published "[valid short JSON][tail of a longer revision]" —
# plus a live [WinError 32] sharing violation. The corrupt file then bricked every
# load on the board, including resume.


def _run_all(threads: list[threading.Thread]) -> None:
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not any(t.is_alive() for t in threads), "worker threads deadlocked"


def test_write_atomic_refuses_non_json(tmp_path: Path) -> None:
    """Garbage must never land on the board: the writer validates before publishing."""
    target = tmp_path / "artifact.json"
    _write_atomic(target, '{"ok": true}')
    with pytest.raises(ValueError):
        _write_atomic(target, '{"version": 3, "l')  # a torn write fragment
    assert json.loads(target.read_text(encoding="utf-8")) == {"ok": True}


def test_write_atomic_concurrent_writers_leave_one_intact_payload(tmp_path: Path) -> None:
    """Racing writers may pick any winner, but the file must always be ONE whole payload."""
    target = tmp_path / "artifact.json"
    payloads = [
        json.dumps({"writer": i, "pad": "x" * (10 + 400 * (i % 2))}) for i in range(8)
    ]
    errors: list[Exception] = []
    barrier = threading.Barrier(len(payloads))

    def write(text: str) -> None:
        try:
            barrier.wait(timeout=30)
            for _ in range(25):
                _write_atomic(target, text)
        except Exception as exc:  # noqa: BLE001 — collected for the assertion below
            errors.append(exc)

    _run_all([threading.Thread(target=write, args=(p,)) for p in payloads])
    assert errors == []
    assert target.read_text(encoding="utf-8") in payloads


def test_load_salvages_valid_prefix_and_heals_file(tmp_path: Path) -> None:
    """The observed corruption class — valid JSON + tails of older revisions — must load
    from the valid prefix and heal the file on disk instead of bricking the board."""
    board = Board.create(tmp_path / "board", _meta())
    board.save("script", _script())
    path = tmp_path / "board" / "script.json"
    intact = path.read_text(encoding="utf-8")
    path.write_text(intact + '005a210a2b6cc228a7"\n  }\n}', encoding="utf-8")

    salvaged = board.load("script")

    assert isinstance(salvaged, Script)
    assert [line.text for line in salvaged.lines] == ["Stopp!"]
    healed = Script.model_validate_json(path.read_text(encoding="utf-8"))
    assert healed.lines == salvaged.lines, "the file itself is clean again after the load"


def test_load_still_raises_when_no_valid_prefix(tmp_path: Path) -> None:
    """A file with its head torn off has nothing to salvage — that stays a loud error."""
    board = Board.create(tmp_path / "board", _meta())
    board.save("script", _script())
    path = tmp_path / "board" / "script.json"
    path.write_text('ersion": 3, "lines": []}', encoding="utf-8")
    with pytest.raises(ValidationError):
        board.load("script")


def test_load_does_not_mask_schema_mismatch(tmp_path: Path) -> None:
    """Salvage is for trailing garbage only: intact JSON of the wrong shape still raises."""
    board = Board.create(tmp_path / "board", _meta())
    path = tmp_path / "board" / "script.json"
    path.write_text('{"not_a_script": true}', encoding="utf-8")
    with pytest.raises(ValidationError):
        board.load("script")


def test_scene_reviews_salvage_trailing_garbage(tmp_path: Path) -> None:
    board = Board.create(tmp_path / "board", _meta())
    board.save_scene_review(_review(1))
    path = tmp_path / "board" / "scene_reviews" / "scene_1.json"
    path.write_text(path.read_text(encoding="utf-8") + "\n  }\n}", encoding="utf-8")
    assert [r.scene_number for r in board.scene_reviews()] == [1]


def test_concurrent_saves_serialize_without_corruption(tmp_path: Path) -> None:
    """Simultaneous saves through SEPARATE Board instances on one root must behave like
    sequential saves: every version assigned once, every predecessor archived, file valid."""
    root = tmp_path / "board"
    board = Board.create(root, _meta())
    board.save("storyline", _storyline("base"))
    n = 8
    barrier = threading.Barrier(n)
    errors: list[Exception] = []

    def save(i: int) -> None:
        try:
            barrier.wait(timeout=30)
            Board.open(root).save("storyline", _storyline(f"thread {i}"))
        except Exception as exc:  # noqa: BLE001 — collected for the assertion below
            errors.append(exc)

    _run_all([threading.Thread(target=save, args=(i,)) for i in range(n)])
    assert errors == []
    loaded = board.load("storyline")
    assert isinstance(loaded, Storyline)
    assert loaded.version == n + 1
    assert board.versions("storyline") == list(range(1, n + 1))


def test_transaction_makes_read_merge_write_atomic(tmp_path: Path) -> None:
    """save_script_chapter's pattern: load script, merge one chapter, save. Under a
    parallel batch every chapter must survive — no lost updates between load and save."""
    root = tmp_path / "board"
    board = Board.create(root, _meta())
    board.save(
        "script",
        Script(language="de", lines=[ScriptLine(chapter=1, scene_number=1, text="eins")]),
    )
    chapters = list(range(2, 8))
    barrier = threading.Barrier(len(chapters))
    errors: list[Exception] = []

    def add_chapter(chapter: int) -> None:
        try:
            barrier.wait(timeout=30)
            b = Board.open(root)
            with b.transaction():
                existing = b.load("script")
                assert isinstance(existing, Script)
                merged = [
                    *existing.lines,
                    ScriptLine(chapter=chapter, scene_number=1, text=f"kapitel {chapter}"),
                ]
                b.save("script", Script(language="de", lines=merged))
        except Exception as exc:  # noqa: BLE001 — collected for the assertion below
            errors.append(exc)

    _run_all([threading.Thread(target=add_chapter, args=(c,)) for c in chapters])
    assert errors == []
    final = board.load("script")
    assert isinstance(final, Script)
    assert sorted(line.chapter for line in final.lines) == list(range(1, 8))
