"""Pydantic schemas for the v2 production-board artifacts.

Every artifact the agent team exchanges lives on the board as one of these
models.  Validation is the contract: a malformed agent output is rejected at
the tool boundary (the agent sees the error and corrects itself) instead of
propagating silently.  Frame fields follow the project invariant: integer
frames, end-exclusive.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Roi(BaseModel):
    """Normalized region of interest (fractions of source width/height)."""

    model_config = ConfigDict(extra="forbid")

    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    w: float = Field(gt=0.0, le=1.0)
    h: float = Field(gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _inside_frame(self) -> Roi:
        if self.x + self.w > 1.0 + 1e-9 or self.y + self.h > 1.0 + 1e-9:
            raise ValueError("roi exceeds frame bounds")
        return self


class BestWindow(BaseModel):
    """Strongest moment inside a scene, relative to the scene start."""

    model_config = ConfigDict(extra="forbid")

    offset_s: float = Field(ge=0.0)
    duration_s: float = Field(gt=0.0)


class SceneReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_number: int = Field(ge=1)
    src_start_frame: int = Field(ge=0)
    src_end_frame_exclusive: int
    description: str
    whats_happening: str
    hook_score: int = Field(ge=0, le=10)
    best_window: BestWindow
    roi: Roi | None = None
    legibility_notes: str = ""
    degraded: bool = False
    model: str = ""
    version: int = Field(default=1, ge=1)
    created_utc: str = ""

    @model_validator(mode="after")
    def _frames_end_exclusive(self) -> SceneReview:
        if self.src_end_frame_exclusive <= self.src_start_frame:
            raise ValueError("src_end_frame_exclusive must be > src_start_frame")
        return self


class Chapter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chapter: int = Field(ge=1)
    role: Literal["hook", "problem", "feature", "payoff_cta"]
    message: str
    scene_numbers: list[int] = Field(min_length=1)
    target_seconds: float = Field(gt=0.0)


class Storyline(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = Field(default=1, ge=1)
    red_thread: str
    arc: list[Chapter] = Field(min_length=1)


class ScriptLine(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chapter: int = Field(ge=1)
    scene_number: int = Field(ge=1)
    text: str = Field(min_length=1)


class Script(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = Field(default=1, ge=1)
    language: str
    lines: list[ScriptLine] = Field(min_length=1)


class VoiceArtifact(BaseModel):
    """Synthesis result, cached by script hash (re-voice only on text change)."""

    model_config = ConfigDict(extra="forbid")

    version: int = Field(default=1, ge=1)
    script_hash: str
    mp3_path: str
    timings_path: str | None = None
    voice_s: float | None = None


class CutSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order: int = Field(ge=0)
    scene_number: int = Field(ge=1)
    start_frame: int = Field(ge=0)
    end_frame_exclusive: int
    roi: Roi | None = None
    zoom_start_s: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def _frames_end_exclusive(self) -> CutSegment:
        if self.end_frame_exclusive <= self.start_frame:
            raise ValueError("end_frame_exclusive must be > start_frame")
        return self


class Cutlist(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = Field(default=1, ge=1)
    segments: list[CutSegment] = Field(min_length=1)

    @model_validator(mode="after")
    def _orders_contiguous(self) -> Cutlist:
        orders = [s.order for s in self.segments]
        if sorted(orders) != list(range(len(orders))):
            raise ValueError("segment orders must be 0..n-1 without gaps")
        return self


class ContactSheetTile(BaseModel):
    """One cutlist segment's sampled frame inside the contact-sheet grid."""

    model_config = ConfigDict(extra="forbid")

    order: int = Field(ge=0)
    scene_number: int = Field(ge=1)
    frame: int = Field(ge=0)  # sampled SOURCE frame (the segment window's middle)
    label: str


class ContactSheet(BaseModel):
    """One grid PNG over the cutlist (each segment's middle frame, tiles in segment order) —
    the visual pre-render checkpoint a user signs off on before render_production runs."""

    model_config = ConfigDict(extra="forbid")

    version: int = Field(default=1, ge=1)
    png_path: str
    cols: int = Field(gt=0)
    rows: int = Field(gt=0)
    labeled: bool = True  # False when no usable font was found (tiles unlabeled, sheet still valid)
    tiles: list[ContactSheetTile] = Field(min_length=1)


class RenderCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    ok: bool
    note: str = ""


class RenderReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = Field(default=1, ge=1)
    export_id: str
    video_s: float = Field(gt=0.0)
    voice_s: float | None = None
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    checks: list[RenderCheck] = Field(default_factory=list)


class QaFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    severity: Literal["info", "minor", "major"]
    where: str
    note: str


class QaReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = Field(default=1, ge=1)
    verdict: Literal["ship", "revise"]
    findings: list[QaFinding] = Field(default_factory=list)


class BoardMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    asset_id: str
    created_utc: str
    task: str
    format: str = "insta"
    target_seconds: float = Field(gt=0.0)
    status: str = "active"
