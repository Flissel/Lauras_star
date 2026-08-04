"""Versioned artifact store for v2 production sessions (the "Production Board").

Layout under ``<workspace>/agent-runs/<session_id>/board/``::

    meta.json                       BoardMeta
    scene_reviews/scene_<n>.json    one SceneReview per scene
    storyline.json .. qa_report.json  singleton artifacts (see _SINGLETONS)
    versions/<stem>.v<k>.json       append-only archive of every replaced version

Writes are atomic (uniquely named tmp + replace), JSON-guarded, and serialized by a
process-wide per-board lock; reads heal the valid-prefix corruption class in place
(see ``_read_model``).  Invalidation always runs *downstream* along ``_CHAIN`` —
never upstream — so cached upstream work (above all: scene reviews) survives every
adjust/revert.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast
from uuid import uuid4

from pydantic import BaseModel, ValidationError

from laura.short_creator.board_models import (
    BoardMeta,
    BoardStatus,
    ContactSheet,
    Cutlist,
    QaReport,
    RenderReport,
    SceneReview,
    Script,
    Storyline,
    VoiceArtifact,
    content_hash,
    lines_in_storyline_order,
    script_hash,
)

logger = logging.getLogger(__name__)

_M = TypeVar("_M", bound=BaseModel)


class _Versioned(Protocol):
    """Protocol for artifacts with a version field."""

    version: int


class _BoardLock:
    """Reentrant lock, one per board directory, shared process-wide.

    Board instances are constructed ad hoc — per request, per tool call — so mutual
    exclusion cannot live on the instance. The registry keys on the resolved root, and
    every writer AND reader of one board's files goes through the same lock: on Windows
    a replace() colliding with any open handle on the same file is a sharing violation,
    exactly the ``[WinError 32]`` a live five-way-parallel tool batch produced.
    """

    def __init__(self) -> None:
        self._inner = threading.RLock()

    def __enter__(self) -> None:
        self._inner.acquire()

    def __exit__(self, *exc: object) -> None:
        self._inner.release()


_BOARD_LOCKS: dict[str, _BoardLock] = {}
_BOARD_LOCKS_GUARD = threading.Lock()


def _lock_for(root: Path) -> _BoardLock:
    key = os.path.normcase(str(root.resolve()))
    with _BOARD_LOCKS_GUARD:
        lock = _BOARD_LOCKS.get(key)
        if lock is None:
            lock = _BOARD_LOCKS[key] = _BoardLock()
        return lock

_CHAIN: tuple[str, ...] = (
    "storyline",
    "script",
    "voice",
    "cutlist",
    "contact_sheet",
    "render_report",
    "qa_report",
)
def _is_stale(artifact: BaseModel | None, current_script_hash: str | None) -> bool | None:
    """Whether this artifact was built from a script the board has since moved past.

    Three answers, and the third one matters: True (proven mismatch), False (proven match), and
    None — the artifact records no provenance, or there is no script to compare against. Boards
    written before provenance existed fall in the third case, and calling those "current" would
    repeat the original bug in a new place: asserting a freshness nobody established.
    """
    recorded = getattr(artifact, "script_hash", None)
    if not isinstance(recorded, str) or not recorded or not current_script_hash:
        return None
    return recorded != current_script_hash


def _parents_stale(
    load: Callable[[str], BaseModel | None], artifact: BaseModel
) -> bool | None:
    """Staleness via the parents chain: any drifted parent means stale.

    True — at least one recorded parent is present and its content hash differs.
    False — every recorded parent is present and matches.
    None — at least one recorded parent is missing (nothing to compare against).
    Only meaningful for artifacts with non-empty ``parents``; callers gate on that.
    """
    parents = getattr(artifact, "parents", None)
    if not isinstance(parents, dict) or not parents:
        return None
    saw_missing = False
    for name, recorded in parents.items():
        if name not in _SINGLETONS:
            # A hand-edited/corrupt archive naming an artifact that does not exist at all --
            # ``Board.load`` raises KeyError for it. Degrade like any other missing parent
            # (unknown, never a match) instead of letting that escape and kill the resume.
            saw_missing = True
            continue
        current = load(name)
        if current is None:
            saw_missing = True
            continue
        if content_hash(current) != recorded:
            return True
    return None if saw_missing else False


def _failed_checks(artifact: BaseModel | None) -> list[str] | None:
    """Names of this artifact's failed checks, or None when it records no checks at all."""
    checks = getattr(artifact, "checks", None)
    if checks is None:
        return None
    return [c.name for c in checks if not c.ok]


_SINGLETONS: dict[str, type[BaseModel]] = {
    "storyline": Storyline,
    "script": Script,
    "voice": VoiceArtifact,
    "cutlist": Cutlist,
    "contact_sheet": ContactSheet,
    "render_report": RenderReport,
    "qa_report": QaReport,
}


def downstream_of(name: str) -> tuple[str, ...]:
    """Artifacts invalidated by a change to ``name`` (chain order preserved)."""
    if name == "scene_reviews":
        return _CHAIN
    return _CHAIN[_CHAIN.index(name) + 1 :]


def _write_atomic(path: Path, text: str) -> None:
    """Publish ``text`` at ``path`` via a uniquely named tmp file + atomic replace.

    The tmp name MUST be unique per write. Every write once shared one ``<name>.tmp``:
    a parallel tool batch truncated/wrote that single file through interleaved handles,
    and replace() published a valid short document with the tail of a longer one glued
    on — the 2026-08-04 board corruption. The JSON guard is the write-time gate: torn
    or garbled text never lands on the board, whatever produced it.
    """
    try:
        json.loads(text)
    except ValueError as exc:
        raise ValueError(f"refusing to write invalid JSON to {path.name}: {exc}") from exc
    tmp = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        # Windows transiently denies a replace whose destination is mid-replace from
        # another writer (or briefly held by a scanner). The replace itself stays atomic
        # either way — the file is always exactly ONE writer's payload — so a short
        # bounded retry is correctness, not papering-over.
        for attempt in range(20):
            try:
                tmp.replace(path)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.005 * (attempt + 1))
    finally:
        tmp.unlink(missing_ok=True)


def _salvage_valid_prefix(text: str, model_type: type[_M]) -> _M | None:
    """The observed corruption class: a valid JSON document with tails of older, longer
    revisions appended. Returns the validated prefix, or None when this is not that class
    — a schema mismatch or a torn-off head has nothing trustworthy to salvage."""
    start = len(text) - len(text.lstrip())
    try:
        prefix, end = json.JSONDecoder().raw_decode(text, start)
    except ValueError:
        return None
    if not text[end:].strip():
        return None  # whole-document JSON: the failure is schema-shaped, not trailing junk
    try:
        return model_type.model_validate(prefix)
    except ValidationError:
        return None


def _read_model(path: Path, model_type: type[_M]) -> _M:
    """Read + validate a board file; heal the trailing-garbage corruption class in place.

    A corrupt singleton used to brick the session permanently: every load raised,
    including the resume path's, so the one artifact poisoned the whole board. When the
    file is a valid document plus trailing garbage, that prefix IS a complete revision
    of the artifact — return it, rewrite the file clean, and say so. Anything else
    still raises: silence about a destroyed artifact would be worse than the brick.

    Callers hold the board lock.
    """
    text = path.read_text(encoding="utf-8")
    try:
        return model_type.model_validate_json(text)
    except ValidationError:
        salvaged = _salvage_valid_prefix(text, model_type)
        if salvaged is None:
            raise
        logger.warning(
            "board file %s was corrupt (trailing data after a valid %s document); "
            "healed from the valid prefix",
            path,
            model_type.__name__,
        )
        _write_atomic(path, salvaged.model_dump_json(indent=2))
        return salvaged


def _same_content(a: BaseModel, b: BaseModel) -> bool:
    """Do two artifacts carry the same content? ``version`` is bookkeeping, not content."""
    return a.model_dump(exclude={"version"}) == b.model_dump(exclude={"version"})


class Board:
    """One production session's artifact store."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._lock = _lock_for(root)

    @classmethod
    def create(cls, root: Path, meta: BoardMeta) -> Board:
        (root / "scene_reviews").mkdir(parents=True, exist_ok=True)
        (root / "versions").mkdir(parents=True, exist_ok=True)
        board = cls(root)
        with board._lock:
            _write_atomic(root / "meta.json", meta.model_dump_json(indent=2))
        return board

    @classmethod
    def open(cls, root: Path) -> Board:
        if not (root / "meta.json").is_file():
            raise FileNotFoundError(f"no board at {root}")
        return cls(root)

    def transaction(self) -> _BoardLock:
        """The board's lock, for holding across a read-modify-write sequence.

        Every Board method is atomic on its own, but a caller that loads an artifact,
        merges into it, and saves it back is three atomic steps — a parallel tool batch
        interleaving between them loses updates. ``with board.transaction():`` makes the
        whole sequence one step; nested Board calls re-enter the same lock."""
        return self._lock

    def meta(self) -> BoardMeta:
        with self._lock:
            return _read_model(self.root / "meta.json", BoardMeta)

    # -- scene reviews ---------------------------------------------------

    def save_scene_review(self, review: SceneReview) -> int:
        with self._lock:
            stem = f"scene_{review.scene_number}"
            path = self.root / "scene_reviews" / f"{stem}.json"
            current_version = 0
            if path.is_file():
                old = _read_model(path, SceneReview)
                current_version = old.version
                self._archive(stem, old.version, path)
            version = max([current_version, *self.versions(stem)]) + 1
            stamped = review.model_copy(update={"version": version})
            _write_atomic(path, stamped.model_dump_json(indent=2))
            return version

    def scene_reviews(self) -> list[SceneReview]:
        with self._lock:
            folder = self.root / "scene_reviews"
            reviews = [
                _read_model(p, SceneReview) for p in folder.glob("scene_*.json")
            ]
            return sorted(reviews, key=lambda r: r.scene_number)

    # -- singleton artifacts -----------------------------------------------

    def save(self, name: str, artifact: BaseModel) -> int:
        """Persist a singleton artifact; archives the old version and
        invalidates everything downstream.  Returns the new version."""
        model_type = _SINGLETONS.get(name)
        if model_type is None:
            raise KeyError(f"unknown artifact: {name}")
        if not isinstance(artifact, model_type):
            raise TypeError(
                f"{name} expects {model_type.__name__}, got {type(artifact).__name__}"
            )
        with self._lock:
            path = self.root / f"{name}.json"
            current_version = 0
            if path.is_file():
                old = _read_model(path, model_type)
                old_versioned = cast(_Versioned, old)
                current_version = int(old_versioned.version)
                # A save that changes nothing has made nothing downstream stale. Re-saving an
                # identical artifact used to wipe the whole chain below it and force a rebuild —
                # a live run burned its turn budget doing exactly that three times over.
                if _same_content(old, artifact):
                    return current_version
                self._archive(name, current_version, path)
            version = max([current_version, *self.versions(name)]) + 1
            stamped = artifact.model_copy(update={"version": version})
            _write_atomic(path, stamped.model_dump_json(indent=2))
            self.invalidate(name)
            return version

    def load(self, name: str) -> BaseModel | None:
        model_type = _SINGLETONS.get(name)
        if model_type is None:
            raise KeyError(f"unknown artifact: {name}")
        with self._lock:
            path = self.root / f"{name}.json"
            if not path.is_file():
                return None
            return _read_model(path, model_type)

    def invalidate(self, name: str) -> list[str]:
        """Archive + remove every present artifact downstream of ``name``."""
        with self._lock:
            removed: list[str] = []
            for dep in downstream_of(name):
                path = self.root / f"{dep}.json"
                if path.is_file():
                    cur = _read_model(path, _SINGLETONS[dep])
                    cur_versioned = cast(_Versioned, cur)
                    self._archive(dep, int(cur_versioned.version), path)
                    path.unlink()
                    removed.append(dep)
            return removed

    def revert(self, name: str, version: int) -> None:
        """Restore an archived version as current; downstream is invalidated."""
        if name not in _SINGLETONS:
            raise KeyError(f"unknown artifact: {name}")
        with self._lock:
            archived = self.root / "versions" / f"{name}.v{version}.json"
            if not archived.is_file():
                raise FileNotFoundError(f"no archived {name} v{version}")
            model_type = _SINGLETONS[name]
            # A corrupt archive must fail loudly HERE, not become the current artifact.
            archived_text = archived.read_text(encoding="utf-8")
            model_type.model_validate_json(archived_text)
            path = self.root / f"{name}.json"
            if path.is_file():
                cur = _read_model(path, model_type)
                cur_versioned = cast(_Versioned, cur)
                self._archive(name, int(cur_versioned.version), path)
            _write_atomic(path, archived_text)
            self.invalidate(name)

    # -- shared internals --------------------------------------------------

    def versions(self, stem: str) -> list[int]:
        """Archived version numbers for a singleton name or ``scene_<n>`` stem."""
        with self._lock:
            prefix = f"{stem}.v"
            out: list[int] = []
            for p in (self.root / "versions").glob(f"{stem}.v*.json"):
                digits = p.name[len(prefix) : -len(".json")]
                if digits.isdigit():
                    out.append(int(digits))
            return sorted(out)

    def _archive(self, stem: str, version: int, path: Path) -> None:
        dest = self.root / "versions" / f"{stem}.v{version}.json"
        _write_atomic(dest, path.read_text(encoding="utf-8"))

    def restore_coherent_suffix(self) -> list[str]:
        """Bring back the longest archived suffix whose parents match the board — in order.

        Walks the chain; present links are skipped, a missing link is restored from its
        newest archived version whose EVERY recorded parent is present on the board with a
        matching content hash. Empty ``parents`` (pre-provenance archive, or a root) never
        restores — unknown is not coherent. The first missing link with no matching archive
        ends the walk. Checking happens on the peeked archive file BEFORE any revert, so a
        non-match is never even momentarily current; upstream-first order means each revert's
        downstream invalidation only touches links that are already missing, and every child
        checked afterwards points at the exact instance just restored.

        Successor to the review-killed single-link restore (41ecc51): script text alone could
        not identify a render; the parent-instance hashes can.
        """
        with self._lock:
            restored: list[str] = []
            for name in _CHAIN:
                if self.load(name) is not None:
                    continue
                candidate_version = self._newest_matching_version(name)
                if candidate_version is None:
                    break
                self.revert(name, candidate_version)
                restored.append(name)
            return restored

    def _newest_matching_version(self, name: str) -> int | None:
        """The newest archived version of ``name`` whose parents all match the board."""
        model_type = _SINGLETONS[name]
        for version in sorted(self.versions(name), reverse=True):
            path = self.root / "versions" / f"{name}.v{version}.json"
            try:
                candidate = model_type.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue  # unreadable archive: skip, never fatal
            parents = getattr(candidate, "parents", None)
            if not isinstance(parents, dict) or not parents:
                continue  # pre-provenance or root: unknown is not coherent
            if _parents_stale(self.load, candidate) is False:
                return version
        return None

    # -- progress -----------------------------------------------------------

    def set_status(self, value: BoardStatus) -> None:
        """Record the run's lifecycle on the board itself.

        The board is the store an operator reads; the job result is a different store. When a run
        hard-failed, only the job result knew, so the session endpoint kept reporting a serene
        "active" for a run that had been dead for the better part of an hour.
        """
        with self._lock:
            meta = self.meta().model_copy(update={"status": value})
            _write_atomic(self.root / "meta.json", meta.model_dump_json(indent=2))

    def resume_point(self, expected_scenes: list[int]) -> str:
        """First missing artifact — where a (re)started session job continues."""
        have = {r.scene_number for r in self.scene_reviews()}
        missing = [n for n in expected_scenes if n not in have]
        if missing:
            return f"scene_reviews:{missing[0]}"
        # Presence decides the chain, deliberately. It is tempting to also refuse a render_report
        # whose export_ready check failed — the chain does advance to QA over a render that never
        # became watchable. But that check is a snapshot taken when the poll gave up, not a
        # property of the artifact: an export that finished a moment after the timeout would be
        # re-rendered forever on the strength of a stale False. It also collides with
        # _MAX_RENDER_CYCLES, which refuses the re-render such a rule demands, and with status()
        # below, which stays presence-based — the agent prompt would then read "render_report is
        # DONE, do not redo it" and "resume at render_report" in the same breath.
        # The honest repair is at the WRITE site (do not record a render that did not happen) plus
        # a re-poll rather than a re-render, and that is a design, not a guard. Until then the
        # failure is at least VISIBLE: status() reports checks_ok and failed_checks.
        with self._lock:
            for name in _CHAIN:
                if self.load(name) is None:
                    return name
            return "done"

    def status(self) -> dict[str, Any]:
        """Board summary for the session API (versions + presence + whether the work happened)."""
        reviews = self.scene_reviews()
        degraded = [r.scene_number for r in reviews if r.degraded]
        script = self.load("script")
        storyline = self.load("storyline")
        if isinstance(script, Script):
            # The identity every write site records is the PLAYED order (the storyline's), not
            # the stored one. Hashing the stored order made a just-rendered report read stale
            # whenever a chapter's storyline reordered its scenes — a provenance signal that
            # cries wolf teaches everyone to ignore the case it exists for.
            lines = (
                lines_in_storyline_order(script, storyline)
                if isinstance(storyline, Storyline)
                else script.lines
            )
            current_hash: str | None = script_hash(lines)
        else:
            current_hash = None
        artifacts: dict[str, Any] = {}
        for name in _CHAIN:
            cur = self.load(name)
            if cur is not None:
                cur_versioned = cast(_Versioned, cur)
                version = int(cur_versioned.version)
            else:
                version = None
            entry: dict[str, Any] = {
                "version": version,
                "archived_versions": self.versions(name),
            }
            # An artifact can be present and still record that its work did not come off. The
            # presence alone used to be the whole story, which is how a timed-out render read
            # as a finished one.
            failed = _failed_checks(cur)
            if failed is not None:
                entry["checks_ok"] = not failed
                entry["failed_checks"] = failed
            # target_ratio is reporting, not a check (see RenderReport) — it never affects
            # checks_ok. It is still the only read path the QA reviewer has for whether the
            # render hit its length target, so status() surfaces it when the report carries one.
            target_ratio = getattr(cur, "target_ratio", None)
            if target_ratio is not None:
                entry["target_ratio"] = target_ratio
            # Presence says an artifact exists; provenance says whether it still belongs to the
            # board it sits on. A render built from a script 25 versions old looked finished.
            if hasattr(cur, "script_hash") or bool(getattr(cur, "parents", None)):
                parents_verdict = _parents_stale(self.load, cur) if cur is not None else None
                if getattr(cur, "parents", None):
                    entry["stale"] = parents_verdict
                else:
                    entry["stale"] = _is_stale(cur, current_hash)
            artifacts[name] = entry
        return {
            "meta": json.loads(self.meta().model_dump_json()),
            "scene_reviews": {
                "count": len(reviews),
                "scenes": [r.scene_number for r in reviews],
                # A degraded review is one the VLM never actually produced: neutral score, one
                # default window. The count alone cannot tell a fully analysed board from one
                # with no visual analysis at all.
                "degraded_count": len(degraded),
                "degraded_scenes": degraded,
            },
            "artifacts": artifacts,
        }
