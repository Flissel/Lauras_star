"""Versioned artifact store for v2 production sessions (the "Production Board").

Layout under ``<workspace>/agent-runs/<session_id>/board/``::

    meta.json                       BoardMeta
    scene_reviews/scene_<n>.json    one SceneReview per scene
    storyline.json .. qa_report.json  singleton artifacts (see _SINGLETONS)
    versions/<stem>.v<k>.json       append-only archive of every replaced version

Writes are atomic (tmp + replace) and pydantic-validated.  Invalidation always
runs *downstream* along ``_CHAIN`` — never upstream — so cached upstream work
(above all: scene reviews) survives every adjust/revert.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, cast

from pydantic import BaseModel

from laura.short_creator.board_models import (
    BoardMeta,
    ContactSheet,
    Cutlist,
    QaReport,
    RenderReport,
    SceneReview,
    Script,
    Storyline,
    VoiceArtifact,
)


class _Versioned(Protocol):
    """Protocol for artifacts with a version field."""

    version: int

_CHAIN: tuple[str, ...] = (
    "storyline",
    "script",
    "voice",
    "cutlist",
    "contact_sheet",
    "render_report",
    "qa_report",
)
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
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


class Board:
    """One production session's artifact store."""

    def __init__(self, root: Path) -> None:
        self.root = root

    @classmethod
    def create(cls, root: Path, meta: BoardMeta) -> Board:
        (root / "scene_reviews").mkdir(parents=True, exist_ok=True)
        (root / "versions").mkdir(parents=True, exist_ok=True)
        _write_atomic(root / "meta.json", meta.model_dump_json(indent=2))
        return cls(root)

    @classmethod
    def open(cls, root: Path) -> Board:
        if not (root / "meta.json").is_file():
            raise FileNotFoundError(f"no board at {root}")
        return cls(root)

    def meta(self) -> BoardMeta:
        raw = (self.root / "meta.json").read_text(encoding="utf-8")
        return BoardMeta.model_validate_json(raw)

    # -- scene reviews ---------------------------------------------------

    def save_scene_review(self, review: SceneReview) -> int:
        stem = f"scene_{review.scene_number}"
        path = self.root / "scene_reviews" / f"{stem}.json"
        current_version = 0
        if path.is_file():
            old = SceneReview.model_validate_json(path.read_text(encoding="utf-8"))
            current_version = old.version
            self._archive(stem, old.version, path)
        version = max([current_version, *self.versions(stem)]) + 1
        stamped = review.model_copy(update={"version": version})
        _write_atomic(path, stamped.model_dump_json(indent=2))
        return version

    def scene_reviews(self) -> list[SceneReview]:
        folder = self.root / "scene_reviews"
        reviews = [
            SceneReview.model_validate_json(p.read_text(encoding="utf-8"))
            for p in folder.glob("scene_*.json")
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
        path = self.root / f"{name}.json"
        current_version = 0
        if path.is_file():
            old = model_type.model_validate_json(path.read_text(encoding="utf-8"))
            old_versioned = cast(_Versioned, old)
            current_version = int(old_versioned.version)
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
        path = self.root / f"{name}.json"
        if not path.is_file():
            return None
        return model_type.model_validate_json(path.read_text(encoding="utf-8"))

    def invalidate(self, name: str) -> list[str]:
        """Archive + remove every present artifact downstream of ``name``."""
        removed: list[str] = []
        for dep in downstream_of(name):
            path = self.root / f"{dep}.json"
            if path.is_file():
                model_type = _SINGLETONS[dep]
                cur = model_type.model_validate_json(path.read_text(encoding="utf-8"))
                cur_versioned = cast(_Versioned, cur)
                self._archive(dep, int(cur_versioned.version), path)
                path.unlink()
                removed.append(dep)
        return removed

    def revert(self, name: str, version: int) -> None:
        """Restore an archived version as current; downstream is invalidated."""
        if name not in _SINGLETONS:
            raise KeyError(f"unknown artifact: {name}")
        archived = self.root / "versions" / f"{name}.v{version}.json"
        if not archived.is_file():
            raise FileNotFoundError(f"no archived {name} v{version}")
        path = self.root / f"{name}.json"
        if path.is_file():
            model_type = _SINGLETONS[name]
            cur = model_type.model_validate_json(path.read_text(encoding="utf-8"))
            cur_versioned = cast(_Versioned, cur)
            self._archive(name, int(cur_versioned.version), path)
        _write_atomic(path, archived.read_text(encoding="utf-8"))
        self.invalidate(name)

    # -- shared internals --------------------------------------------------

    def versions(self, stem: str) -> list[int]:
        """Archived version numbers for a singleton name or ``scene_<n>`` stem."""
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

    # -- progress -----------------------------------------------------------

    def resume_point(self, expected_scenes: list[int]) -> str:
        """First missing artifact — where a (re)started session job continues."""
        have = {r.scene_number for r in self.scene_reviews()}
        missing = [n for n in expected_scenes if n not in have]
        if missing:
            return f"scene_reviews:{missing[0]}"
        for name in _CHAIN:
            if self.load(name) is None:
                return name
        return "done"

    def status(self) -> dict[str, Any]:
        """Board summary for the session API (versions + presence)."""
        reviews = self.scene_reviews()
        artifacts: dict[str, Any] = {}
        for name in _CHAIN:
            cur = self.load(name)
            if cur is not None:
                cur_versioned = cast(_Versioned, cur)
                version = int(cur_versioned.version)
            else:
                version = None
            artifacts[name] = {
                "version": version,
                "archived_versions": self.versions(name),
            }
        return {
            "meta": json.loads(self.meta().model_dump_json()),
            "scene_reviews": {
                "count": len(reviews),
                "scenes": [r.scene_number for r in reviews],
            },
            "artifacts": artifacts,
        }
