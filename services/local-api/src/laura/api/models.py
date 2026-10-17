"""Pydantic request/response models for the API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    sequence_rate_num: int = Field(default=30, gt=0, description="e.g. 30000 for 29.97")
    sequence_rate_den: int = Field(default=1, gt=0, description="e.g. 1001 for 29.97")
    drop_frame: bool = False


class ProjectOut(BaseModel):
    id: str
    name: str
    sequence_rate_num: int
    sequence_rate_den: int
    drop_frame: bool
    workspace_root: str
    created_at: str


class HealthOut(BaseModel):
    status: str
    version: str
    pipeline_version: str
    schema_version: int


class AssetImport(BaseModel):
    source_path: str | None = Field(default=None, min_length=1)
    source_url: str | None = Field(default=None, min_length=1)
    display_name: str | None = None
    # URL-ingest comfort options (yt-dlp only; ignored for source_path imports):
    # quality vocabulary best|1080|720|audio, and a browser to read cookies from for
    # private/age-restricted/login-walled sources (chrome|edge|firefox|brave).
    format: str | None = None
    cookies_from_browser: str | None = None

    @model_validator(mode="after")
    def _exactly_one_source(self) -> AssetImport:
        if bool(self.source_path) == bool(self.source_url):
            raise ValueError("provide exactly one of source_path or source_url")
        return self


class ImportAccepted(BaseModel):
    asset_id: str
    job_id: str
    # Additional asset ids when a playlist/channel URL fanned out into multiple assets
    # (asset_id/job_id stay the first entry for backward compatibility).
    extra_asset_ids: list[str] = Field(default_factory=list)


class AssetFileOut(BaseModel):
    id: str
    asset_id: str
    kind: str
    path: str
    size_bytes: int | None = None
    is_proxy: bool = False
    is_waveform: bool = False
    is_audio_extract: bool = False
    checksum: str | None = None


class AssetOut(BaseModel):
    id: str
    project_id: str
    type: str
    display_name: str
    source_path: str
    sha256: str | None = None
    duration_frames: int | None = None
    rate_num: int | None = None
    rate_den: int | None = None
    audio_sample_rate: int | None = None
    start_timecode: str | None = None
    width: int | None = None
    height: int | None = None
    codec_video: str | None = None
    codec_audio: str | None = None
    is_vfr: bool = False
    online: bool = True
    synthetic: bool = False
    ai_effect: str | None = None
    created_at: str
    # Wo eine Kopie der Bytes ausserhalb dieser Maschine liegt ("laura/assets/<id>.mp4"),
    # oder None ohne Objektspeicher -- der Normalfall auf dem Desktop. `source_path`
    # daneben bleibt der lokale Arbeitsweg und wird NIE ersetzt.
    object_key: str | None = Field(
        default=None,
        description=(
            "Bucket-qualifizierter Schluessel der Kopie ausserhalb dieser Maschine, "
            "z.B. 'laura/assets/<id>.mp4'; None ohne Objektspeicher -- der Normalfall "
            "auf dem Desktop. `source_path` daneben ist ein Pfad, der nur auf Lauras "
            "Rechner etwas bedeutet; dieser Schluessel gilt ueberall. "
            "Er wird beim Probe gesetzt, also kurz NACH dem Import -- ein Asset, das "
            "gerade erst angenommen wurde, hat ihn noch nicht. "
            "Assets, die vor dem 16.09.2026 importiert wurden, haben ihn gar nicht: "
            "bis dahin lud nur der Download-Zweig hoch, nie der Import per lokalem Pfad."
        ),
    )
    files: list[AssetFileOut] = Field(default_factory=list)


class JobOut(BaseModel):
    id: str
    queue: str
    kind: str
    status: str
    attempt: int
    max_attempts: int
    result_json: str | None = None
    error_json: str | None = None
    created_at: str
    updated_at: str
    finished_at: str | None = None


class JobAccepted(BaseModel):
    job_id: str


class AnalysisStart(BaseModel):
    scene: bool = True
    asr: bool = True
    diarize: bool = False
    align: bool = False
    model: str = "base"
    language: str | None = None
    detector: str = "adaptive"  # shot detector: adaptive|content|histogram|transnet|hybrid


class AnalysisAccepted(BaseModel):
    analysis_run_id: str
    job_id: str


class AnalysisRunOut(BaseModel):
    id: str
    asset_id: str
    pipeline_version: str
    status: str
    started_at: str | None = None
    finished_at: str | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class ShotOut(BaseModel):
    id: str
    src_in_frame: int
    src_out_frame_exclusive: int
    confidence: float | None = None
    method: str | None = None
    thumbnail_path: str | None = None
    black_ratio: float | None = None
    static_score: float | None = None
    phash: str | None = None
    blur_score: float | None = None
    keep: bool = True
    drop_reason: str | None = None


class WordOut(BaseModel):
    id: str
    idx: int
    start_sample: int
    end_sample: int
    start_frame: int
    end_frame: int
    text: str
    confidence: float | None = None
    is_punctuation: bool = False


class SegmentOut(BaseModel):
    id: str
    speaker_id: str | None = None
    speaker_label: str | None = None
    start_sample: int
    end_sample: int
    start_frame: int
    end_frame: int
    text: str
    confidence: float | None = None
    alignment_status: str = "aligned"
    alignment_job_id: str | None = None
    alignment_language: str | None = None
    alignment_error: str | None = None
    alignment_updated_at: str | None = None
    words: list[WordOut] = Field(default_factory=list)


class TimelineCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    kind: str = "rough_cut"


class FromShotsRequest(BaseModel):
    """Build a rough cut with one contiguous clip per kept shot of an asset."""

    asset_id: str
    run_id: str | None = None        # default: the asset's latest analysis run
    timeline_id: str | None = None   # populate this timeline (must be empty); else create new
    name: str | None = Field(default=None, max_length=200)
    lane: int = Field(default=0, ge=0)
    quality: bool = True                # drop weak shots + merge micro by default
    drop_black: bool | None = None      # per-filter overrides (None = follow `quality`)
    drop_static: bool | None = None
    drop_duplicates: bool | None = None
    drop_blur: bool | None = None
    merge_min_frames: int = Field(default=0, ge=0)
    align_editorial: bool = True        # snap clip cuts to transcript word-gaps (Stage 2)
    editorial_window: int = Field(default=12, ge=0)  # max frames a cut may move (~0.4s@30fps)
    # Picture-vs-sound preference for joint visual+editorial placement: 0 = picture-first (keep
    # the frame-exact visual cut), 1 = sound-first (favour the clean word edge). None -> the
    # product default 0.6/0.4 visual/editorial weights. Only used when align_editorial is True.
    cut_bias: float | None = Field(default=None, ge=0.0, le=1.0)


class DroppedShot(BaseModel):
    src_in_frame: int
    src_out_frame_exclusive: int
    drop_reason: str


class SplitCutOut(BaseModel):
    """A per-cut split-edit (L/J) recommendation for an inter-clip cut.

    RECOMMENDATION ONLY: the stored clips are unchanged hard cuts. This surfaces, per cut, the
    independent optimal picture frame (visual peak) and sound frame (real silence / clean
    word-gap) so the UI/editor can SEE which cuts would benefit from an L- or J-cut and by how
    many frames. All frames are source-frame indices of the asset.
    """

    seq_cut: int          # the original (visual) cut frame on the source
    video_frame: int      # recommended picture cut = visual peak
    audio_frame: int      # recommended sound cut = real silence / clean word-gap
    offset: int           # audio_frame - video_frame (0 == hard cut)
    kind: str             # "hard" | "L" (audio after video) | "J" (audio before video)


class RoughCutQualityOut(BaseModel):
    """Headline quality of a built rough cut — one score blending picture and speech.

    Computed on the fly over the built clips (no persistence): ``visual_exactness`` is the share
    of inter-clip cuts landing within one frame of the true luma peak, ``editorial_cleanliness``
    the share not bisecting a spoken word, and ``overall`` their ``cut_bias``-weighted blend (all
    in ``[0, 1]``). ``n_cuts`` is the number of inter-clip cuts scored; ``n_split_cuts`` the count
    of recommended L/J split edits among them. All fields mirror
    :class:`laura.analysis.eval_quality.RoughCutQuality` plus the split-cut tally.
    """

    overall: float
    visual_exactness: float
    editorial_cleanliness: float
    n_cuts: int
    n_split_cuts: int


class TimelineQualityOut(BaseModel):
    """Persisted quality verdict for a timeline, returned by GET /timelines/{id}/quality.

    ``status`` is one of:
    - ``'computed'``: scores are available (scored at build time).
    - ``'no_video'``: the build had no readable video so the visual half could not be scored.
    - ``'error'``: an unexpected exception during persistence.
    - ``'pending'``: sentinel (NOT stored) — timeline exists but no quality row yet.

    All score fields are ``None`` when status is not ``'computed'``.
    """

    timeline_id: str
    status: str
    overall: float | None = None
    visual_exactness: float | None = None
    editorial_cleanliness: float | None = None
    n_cuts: int | None = None
    n_split_cuts: int | None = None
    created_at: str | None = None


class ClipOut(BaseModel):
    id: str
    asset_id: str
    src_in_frame: int
    src_out_frame_exclusive: int
    seq_in_frame: int
    seq_out_frame_exclusive: int
    lane: int
    role: str = "base"
    speaker_id: str | None = None
    origin_word_start_id: str | None = None
    origin_word_end_id: str | None = None
    speed_num: int = 1
    speed_den: int = 1
    # Signed per-clip LEADING-edge audio-vs-video offset in samples (invariant #3); 0 = hard cut.
    # Surfaced so undo/redo (GET clips -> PUT clips snapshot) round-trips the L/J split state.
    audio_offset_samples: int = 0


class TimelineOut(BaseModel):
    id: str
    project_id: str
    name: str
    kind: str
    created_at: str
    clips: list[ClipOut] = Field(default_factory=list)


class TimelineWithScenesOut(TimelineOut):
    """A timeline response that also carries the scene markers reconciled by the same edit
    (used by delete_words / cut-at-frame so the UI updates clips and markers in one round-trip)."""

    scenes: list[SceneOut] = Field(default_factory=list)


class TimelineAudioClipBase(BaseModel):
    asset_id: str = Field(min_length=1)
    seq_in_frame: int = Field(ge=0)
    seq_out_frame_exclusive: int = Field(gt=0)
    asset_in_frame: int = Field(default=0, ge=0)
    gain_percent: int = Field(default=100, ge=0, le=400)
    fade_in_frames: int = Field(default=0, ge=0)
    fade_out_frames: int = Field(default=0, ge=0)
    mix_mode: Literal["mix", "replace_original", "mute_original"] = "mix"
    ducking_percent: int = Field(default=100, ge=0, le=100)
    label: str | None = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def _valid_range_and_fades(self) -> TimelineAudioClipBase:
        duration = self.seq_out_frame_exclusive - self.seq_in_frame
        if duration <= 0:
            raise ValueError("seq_out_frame_exclusive must be greater than seq_in_frame")
        if self.fade_in_frames + self.fade_out_frames > duration:
            raise ValueError("fade frames cannot exceed clip duration")
        return self


class TimelineAudioClipCreate(TimelineAudioClipBase):
    pass


class TimelineAudioClipUpdate(BaseModel):
    seq_in_frame: int | None = Field(default=None, ge=0)
    seq_out_frame_exclusive: int | None = Field(default=None, gt=0)
    asset_in_frame: int | None = Field(default=None, ge=0)
    gain_percent: int | None = Field(default=None, ge=0, le=400)
    fade_in_frames: int | None = Field(default=None, ge=0)
    fade_out_frames: int | None = Field(default=None, ge=0)
    mix_mode: Literal["mix", "replace_original", "mute_original"] | None = None
    ducking_percent: int | None = Field(default=None, ge=0, le=100)
    label: str | None = Field(default=None, max_length=200)


class TimelineAudioClipOut(TimelineAudioClipBase):
    id: str
    timeline_id: str
    created_at: str


class FromShotsOut(BaseModel):
    timeline: TimelineOut
    dropped: list[DroppedShot] = Field(default_factory=list)
    # Per-cut L/J split-edit recommendations for the inter-clip cuts (recommendation only — the
    # stored clips remain hard cuts). Empty when there is no transcript/silence to plan against.
    split_cuts: list[SplitCutOut] = Field(default_factory=list)
    # On-the-fly quality summary of the built rough cut (overall + visual exactness + editorial
    # cleanliness, blended by the request's cut_bias). None when it can't be computed gracefully
    # (no editorial alignment, or no readable video to score visual exactness against).
    quality: RoughCutQualityOut | None = None


class ClipSourceOut(BaseModel):
    """Jump-back anchor for a timeline clip: where it came from in the source."""

    clip_id: str
    asset_id: str
    src_in_frame: int
    src_out_frame_exclusive: int
    origin_word_start_id: str | None = None
    origin_word_end_id: str | None = None
    segment_id: str | None = None        # transcript segment the origin words belong to
    word_start_frame: int | None = None  # source frame of the first origin word
    word_end_frame: int | None = None    # source frame (exclusive) of the last origin word


class TimelineImportRequest(BaseModel):
    content: str
    format: str = "otio"
    name: str | None = None


class TimelineImportOut(BaseModel):
    timeline: TimelineOut
    matched_media: int   # distinct source files relinked to existing project assets
    offline_media: int   # distinct source files turned into offline placeholders


class ExportRequest(BaseModel):
    format: str
    options: dict[str, Any] = Field(default_factory=dict)


class ExportOut(BaseModel):
    id: str
    timeline_id: str
    format: str
    status: str
    output_path: str | None = None
    lossy: bool = False
    drops: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    content: str | None = None


class ValidateRequest(BaseModel):
    timeline_id: str
    format: str


class ValidateOut(BaseModel):
    format: str
    ok: bool
    lossy: bool
    warnings: list[str] = Field(default_factory=list)
    drops: list[str] = Field(default_factory=list)


class OperationRequest(BaseModel):
    # append_from_words|append_clip|insert_clip|delete|lift|set_speed|split|trim|move|
    # delete_words|set_audio_offset|place_clip
    op: str
    asset_id: str | None = None
    src_in_frame: int | None = None
    src_out_frame_exclusive: int | None = None
    word_start_id: str | None = None
    word_end_id: str | None = None
    seq_in_frame: int | None = None
    seq_out_frame_exclusive: int | None = None
    at_seq_frame: int | None = None
    lane: int = 0  # place_clip: target (destination) lane; other ops: lane selector
    lane_src: int | None = None  # place_clip: source lane identifying the clip to move (§1.3)
    speed_num: int | None = None
    speed_den: int | None = None
    new_src_in_frame: int | None = None          # trim: new source in point
    new_src_out_frame_exclusive: int | None = None  # trim: new source out point
    to_seq_frame: int | None = None              # move/place_clip: target sequence position
    # set_audio_offset: the LEADING-edge L/J audio offset of the clip at at_seq_frame, expressed in
    # FRAMES (the UI's native drag unit). The backend projects it onto canonical samples via the
    # project sequence rate (invariant #3), mirroring the accept endpoint; > 0 = L-cut (audio
    # later), < 0 = J-cut (audio earlier), |offset| < 1 frame is clamped to a hard cut.
    audio_offset_frames: int | None = Field(
        default=None, ge=-100_000, le=100_000
    )


class ClipIn(BaseModel):
    """A timeline clip as accepted by the wholesale set-clips endpoint (undo/redo)."""

    asset_id: str
    src_in_frame: int
    src_out_frame_exclusive: int
    seq_in_frame: int
    seq_out_frame_exclusive: int
    lane: int = 0
    speaker_id: str | None = None
    origin_word_start_id: str | None = None
    origin_word_end_id: str | None = None
    speed_num: int = 1
    speed_den: int = 1
    # Signed per-clip LEADING-edge audio-vs-video offset in samples (invariant #3); 0 = hard cut.
    # Restored verbatim from a snapshot so undo/redo brings back the L/J split as the live column.
    audio_offset_samples: int = 0
    # Clip role restored verbatim from a snapshot so undo/redo preserves replace-overlay clips.
    role: str = "base"


class SetClipsRequest(BaseModel):
    clips: list[ClipIn] = Field(default_factory=list)


# --- enterprise ----------------------------------------------------------
class OrgCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class OrgOut(BaseModel):
    id: str
    name: str
    created_at: str


class UserCreate(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    display_name: str | None = None
    role: str = "editor"


class UserOut(BaseModel):
    id: str
    email: str
    display_name: str | None = None
    created_at: str
    role: str


class KeyCreate(BaseModel):
    name: str | None = None
    role: str = "editor"
    user_id: str | None = None


class KeyCreated(BaseModel):
    id: str
    prefix: str
    role: str
    key: str  # full key — shown ONCE
    created_at: str


class AuditEventOut(BaseModel):
    id: str
    org_id: str | None = None
    principal_kind: str
    principal_id: str | None = None
    action: str
    entity_type: str | None = None
    entity_id: str | None = None
    created_at: str


class SearchRequest(BaseModel):
    project_id: str
    query: str = Field(min_length=1)
    limit: int = Field(default=50, ge=1, le=500)
    mode: str = "lexical"  # "lexical" (LIKE) | "semantic" (Qdrant; falls back if absent)


class SearchResult(BaseModel):
    segment_id: str
    asset_id: str
    asset_name: str
    start_frame: int
    end_frame: int
    text: str
    speaker_label: str | None = None
    score: float | None = None  # semantic similarity (None for lexical)


class SegmentUpdate(BaseModel):
    text: str | None = None
    speaker_id: str | None = None


class TranscriptRealignRequest(BaseModel):
    segment_ids: list[str] | None = None
    language: str | None = None


class TranscriptRealignAccepted(BaseModel):
    job_id: str


class TranscriptConfirmOut(BaseModel):
    """Response for POST /assets/{asset_id}/transcript:confirm (Transkript-Gates, Gate A)."""

    asset_id: str
    transcript_confirmed_at: str


class VoiceoverRequest(BaseModel):
    segment_id: str | None = None
    text: str | None = Field(default=None, min_length=1)
    seq_in_frame: int = Field(ge=0)
    seq_out_frame_exclusive: int = Field(gt=0)
    language: str | None = None
    backend: str | None = None
    runtime_id: str | None = None
    # Explicit TTS voice name (e.g. a Windows SAPI voice like "Microsoft Katja"); None picks a
    # voice by ``language`` when the backend supports it, else the system default.
    voice_id: str | None = Field(default=None, max_length=200)
    gain_percent: int = Field(default=100, ge=0, le=400)
    fade_in_frames: int = Field(default=0, ge=0)
    fade_out_frames: int = Field(default=0, ge=0)
    # How the voice-over treats the original audio under its span: 'mix' keeps the original (ducked
    # to ``ducking_percent``), 'replace_original'/'mute_original' silence it. Defaults are
    # backward-compatible (full-volume mix); the UI picks an audible default (duck the original).
    mix_mode: Literal["mix", "replace_original", "mute_original"] = "mix"
    ducking_percent: int = Field(default=100, ge=0, le=100)
    # Clip-span strategy (narrated-reel spec §3): 'slot' (default) pads/trims the synthesized
    # speech to exactly [seq_in_frame, seq_out_frame_exclusive) as before. 'natural' derives the
    # span from the measured speech length instead; seq_out_frame_exclusive is then only an
    # UPPER BOUND (the clip can end earlier, never later), so undo/idempotency stay unchanged and
    # nothing can grow unboundedly. pad_frames is the trailing silence added after natural-length
    # speech before the upper bound is applied; it is ignored in slot mode.
    fit: Literal["slot", "natural"] = "slot"
    pad_frames: int = Field(default=12, ge=0, le=120)

    @model_validator(mode="after")
    def _valid_voiceover_range(self) -> VoiceoverRequest:
        duration = self.seq_out_frame_exclusive - self.seq_in_frame
        if duration <= 0:
            raise ValueError("seq_out_frame_exclusive must be greater than seq_in_frame")
        if self.fade_in_frames + self.fade_out_frames > duration:
            raise ValueError("fade frames cannot exceed voiceover duration")
        if self.text is None and self.segment_id is None:
            raise ValueError("provide either text or segment_id")
        return self


class VoiceoverAccepted(BaseModel):
    job_id: str


class DemoDraftItem(BaseModel):
    src_in_frame: int = Field(ge=0)
    src_out_frame_exclusive: int = Field(gt=0)
    label: str = Field(min_length=1, max_length=200)
    voiceover_text: str = Field(min_length=1, max_length=2000)
    thumb_frame: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
    enabled: bool = True

    @model_validator(mode="after")
    def _valid_range(self) -> DemoDraftItem:
        if self.src_out_frame_exclusive <= self.src_in_frame:
            raise ValueError("src_out_frame_exclusive must be greater than src_in_frame")
        return self


class DemoDraftAccepted(BaseModel):
    draft_id: str
    job_id: str


class DemoDraftOut(BaseModel):
    id: str
    project_id: str
    asset_id: str
    status: str
    items: list[DemoDraftItem] = Field(default_factory=list)
    result: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str
    applied_at: str | None = None


class DemoDraftUpdate(BaseModel):
    items: list[DemoDraftItem]


class DemoDraftApplyOut(BaseModel):
    draft: DemoDraftOut
    sequence: SequenceOut


class SequenceTranscriptWordOut(BaseModel):
    id: str
    idx: int
    segment_id: str
    asset_id: str
    source_start_frame: int
    source_end_frame: int
    seq_in_frame: int
    seq_out_frame_exclusive: int
    text: str
    confidence: float | None = None
    is_punctuation: bool = False


class SequenceTranscriptBlockOut(BaseModel):
    segment_id: str
    asset_id: str
    speaker_label: str | None = None
    source_start_frame: int
    source_end_frame: int
    seq_in_frame: int
    seq_out_frame_exclusive: int
    text: str
    alignment_status: str = "aligned"
    alignment_job_id: str | None = None
    alignment_language: str | None = None
    alignment_error: str | None = None
    alignment_updated_at: str | None = None
    words: list[SequenceTranscriptWordOut] = Field(default_factory=list)


class RenameRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class ImportStatusOut(BaseModel):
    phase: str  # queued | downloading | verifying | analyzing | ready | cancelled | error
    downloaded_bytes: int | None = None
    total_bytes: int | None = None
    speed_bps: float | None = None
    eta_seconds: float | None = None
    error: str | None = None


class HistoryStateOut(BaseModel):
    can_undo: bool
    can_redo: bool
    undo_label: str | None = None
    redo_label: str | None = None


# --- render-pipeline exports ------------------------------------------------
class SceneOut(BaseModel):
    id: str
    project_id: str
    source_timeline_id: str
    name: str
    order_index: int
    seq_in_frame: int
    seq_out_frame_exclusive: int
    scene_timeline_id: str | None = None
    music_asset_id: str | None = None
    music_gain_percent: int = 100
    # Enrichment populated only by list_project_scenes (the assemble bin): the source
    # video this scene came from and a representative source frame for a thumbnail.
    asset_id: str | None = None
    thumb_frame: int | None = None


class GenerateScenesRequest(BaseModel):
    asset_id: str
    gap_frames: int | None = Field(default=None, ge=0)


class SplitSceneRequest(BaseModel):
    at_seq_frame: int = Field(ge=0)


class CutAtFrameRequest(BaseModel):
    at_seq_frame: int = Field(ge=1)


class MergeScenesRequest(BaseModel):
    scene_id: str


class RenameSceneRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class SetSceneMusicRequest(BaseModel):
    asset_id: str
    gain_percent: int = Field(default=100, ge=0, le=400)


class RenderRequest(BaseModel):
    format: str = "mp4"
    # Optional quality gate: 422 when persisted quality is computed and below this threshold.
    # None means no gate (unverified stamp applied when quality is not computed).
    min_quality: float | None = Field(default=None, ge=0.0, le=1.0)
    # When True, burn the sequence transcript as SRT subtitles into the rendered MP4.
    # Only effective for format="mp4"; other formats ignore this flag.
    # Defaults to False so existing exports are byte-identical to before.
    burn_captions: bool = False
    # Styled karaoke captions (spec §5, narrated-reel). Independent of burn_captions
    # (the plain-SRT path); defaults to False so existing exports are unaffected.
    captions: bool = False
    # Word source for karaoke captions: "auto" prefers voiceover-authored words, falling
    # back to the source-video transcript when no voiceover words exist; "voiceover" and
    # "transcript" pin one source with no fallback.
    caption_source: Literal["auto", "voiceover", "transcript"] = "auto"
    caption_preset: Literal["reels", "tiktok", "shorts", "wide"] = "reels"


class ReelRenderRequest(BaseModel):
    hook_text: str | None = None
    disclosure_text: str = "KI · synthetisch"
    vertical: bool = True
    captions: bool = False
    caption_preset: str = "reels"
    caption_mode: str = "karaoke"
    caption_position: str = "bottom"
    caption_fontsize: int = Field(default=72, ge=24, le=160)
    caption_safe_margin: int = Field(default=250, ge=0, le=800)
    # Optional hard cap on the reel length (social platforms enforce max durations). When set,
    # the render keeps only the first N seconds of the cut (deterministic tail-trim). None = no cap.
    max_duration_seconds: int | None = Field(default=None, ge=1, le=86400)
    # Optional quality gate: 422 when persisted quality is computed and below this threshold.
    # None means no gate (unverified stamp applied when quality is not computed).
    min_quality: float | None = Field(default=None, ge=0.0, le=1.0)
    # KI-Hinweis nach Art des KI-Anteils (Video-Claw, 17.10.2026). "fest" = bisheriges
    # Verhalten (disclosure_text wird eingebrannt). "auto" = disclosure_text wird ignoriert;
    # der Render bestimmt den Text aus dem Inhalt ("Stimme: KI", "Gesicht: KI",
    # "KI · synthetisch") und blendet bei reinem Echtmaterial keinen ein.
    kennzeichnung: Literal["fest", "auto"] = Field(
        default="fest",
        description=(
            "'fest': disclosure_text wird eingebrannt (bisher). 'auto': Hinweis nach Inhalt - "
            "'Stimme: KI', 'Gesicht: KI' oder 'KI · synthetisch'; kein Hinweis, wenn nichts "
            "kuenstlich ist. Jede KI-Stimme zaehlt, auch untergemischt."
        ),
    )

    @field_validator("disclosure_text", mode="before")
    @classmethod
    def _disclosure_required(cls, v: str | None) -> str:
        """EU AI Act: disclosure presence is mandatory; blank → default label."""
        text = (v or "").strip()
        return text or "KI · synthetisch"


class RenderExportOut(BaseModel):
    id: str
    project_id: str
    timeline_id: str | None = None
    format: str
    status: str
    path: str | None = None
    size_bytes: int | None = None
    error: str | None = None
    created_at: str | None = None
    # Machine-readable quality stamp (P1-T2b). Populated from export options.
    quality_status: str | None = None
    quality_verified: bool | None = None
    # Wo eine Kopie des fertigen Videos ausserhalb dieser Maschine liegt
    # ("laura/exports/<id>.mp4"), oder None ohne Objektspeicher. `path` daneben bleibt
    # die lokale Datei und wird NIE ersetzt. Ohne dieses Feld blieb der Objektspeicher
    # fuer jeden Konsumenten unsichtbar: der Wert stand in der Datenbank und kam nie heraus.
    object_key: str | None = Field(
        default=None,
        description=(
            "Bucket-qualifizierter Schluessel der Kopie ausserhalb dieser Maschine, "
            "z.B. 'laura/exports/<id>.mp4'; None, wenn kein Objektspeicher "
            "konfiguriert ist oder der Upload fehlschlug. `path` daneben bleibt die "
            "lokale Datei und wird nie ersetzt. "
            "ACHTUNG BEIM POLLEN: status='ready' heisst NICHT, dass der Schluessel "
            "schon da ist. Der Render wird zuerst als fertig vermerkt, danach erst "
            "hochgeladen -- damit ein voller oder unerreichbarer Bucket keine Stunde "
            "Rendern kostet. Wer den Schluessel braucht, fragt nach 'ready' weiter, "
            "bis er auftaucht, und faellt sonst auf `path` zurueck."
        ),
    )


class DownloadUrlOut(BaseModel):
    """Ein Abruf-Link, der OHNE Zugangsdaten funktioniert.

    Gebaut am 22.09.2026, nachdem ein Agent die Objekt-Adresse gesehen, sie "kein
    direkt herunterladbarer Link" genannt und vorgeschlagen hatte, ein fertiges
    Video NEU ZU RENDERN. Der Schluessel allein ist eine Tuer ohne Klinke: wer ihn
    einloesen will, braeuchte den Service-Key -- und den darf ein Agent nie sehen.
    """

    url: str = Field(
        description=(
            "Vollstaendiger Abruf-Link. Traegt seine Berechtigung selbst, braucht "
            "also keinen Schluessel und keine Kopfzeile. Gilt nur, solange die "
            "Adresse des Objektspeichers erreichbar ist (heute im Tailnet)."
        )
    )
    expires_in_seconds: int = Field(
        description="Wie lange der Link gilt, ab dem Zeitpunkt dieser Antwort."
    )
    object_key: str = Field(description="Die Adresse, fuer die der Link gilt.")


# --- sequence (stage 5) ------------------------------------------------------
class SequenceItemOut(BaseModel):
    id: str
    scene_id: str
    scene_name: str
    order_index: int
    transition_after_kind: str = "hard"
    transition_after_frames: int = 0


class SequenceOut(BaseModel):
    timeline_id: str
    project_id: str
    items: list[SequenceItemOut] = Field(default_factory=list)


class SetSequenceScenesRequest(BaseModel):
    scene_ids: list[str]


class SequenceTransitionRequest(BaseModel):
    kind: str = "hard"
    duration_frames: int = Field(default=0, ge=0, le=240)


# --- transition smoothness review -------------------------------------------
class BoundaryIdentity(BaseModel):
    asset_a: str
    asset_b: str
    src_out_a: int
    src_in_b: int


class TransitionFixModel(BaseModel):
    kind: Literal["none", "resnap", "transition"]
    resnap_delta_frames: int = 0
    transition_style: Literal["crossfade", "fade"] = "crossfade"
    transition_frames: int = Field(default=0, ge=0, le=240)


class TransitionVerdictOut(BaseModel):
    boundary_seq_frame: int
    asset_a: str
    asset_b: str
    src_out_a: int
    src_in_b: int
    smoothness: float
    label: str
    reason: str
    suggested_fix: dict[str, Any]
    model_id: str
    created_at: str


class TransitionReviewOut(BaseModel):
    verdicts: list[TransitionVerdictOut]


class ApplyFixRequest(BaseModel):
    identity: BoundaryIdentity
    fix: TransitionFixModel


class ApplyFixOut(BaseModel):
    status: str
    applied: str | None = None
    reason: str | None = None
    delta: int | None = None
    style: str | None = None


# --- reenact / consent -------------------------------------------------------
class ConsentRequest(BaseModel):
    subject_label: str
    source_asset_id: str | None = None
    confirmed_by: str | None = None
    note: str | None = None


class ConsentOut(BaseModel):
    id: str
    project_id: str
    subject_label: str
    source_asset_id: str | None
    confirmed_by: str | None
    confirmed_at: str
    note: str | None
    revoked_at: str | None


class ReenactRequest(BaseModel):
    seq_in_frame: int
    seq_out_frame_exclusive: int
    portrait_asset_id: str
    consent_id: str  # MANDATORY — missing -> 422 automatically
    backend: str | None = None
    runtime_id: str | None = None


class LipsyncRequest(BaseModel):
    seq_in_frame: int = Field(ge=0)
    seq_out_frame_exclusive: int = Field(gt=0)
    audio_asset_id: str
    consent_id: str
    license_accepted: bool
    backend: str | None = None
    runtime_id: str | None = None
    quality_threshold: float = Field(default=0.6, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _valid_lipsync_request(self) -> LipsyncRequest:
        if self.seq_out_frame_exclusive <= self.seq_in_frame:
            raise ValueError("seq_out_frame_exclusive must be greater than seq_in_frame")
        if self.license_accepted is not True:
            raise ValueError("license_accepted must be true")
        return self


class LipsyncAccepted(BaseModel):
    job_id: str


# --- narrated reel (collage builder, spec §6) --------------------------------
class NarratedReelBeat(BaseModel):
    """One beat of a narrated-reel collage: a spoken line over a slice of source video.

    ``src_in_frame`` is the asset source frame the clip starts at; its length is derived
    from the synthesized voiceover's measured speech (spec §3 natural-fit), not requested
    here. ``pad_frames`` is trailing silence added after the measured speech before the
    beat's clip is clamped to the asset's end.
    """

    text: str = Field(min_length=1)
    asset_id: str
    src_in_frame: int = Field(ge=0)
    pad_frames: int = Field(default=12, ge=0, le=120)


class NarratedReelRequest(BaseModel):
    """Request body for POST /projects/{project_id}/narrated-reel (spec §6).

    Builds a fresh timeline from ``beats`` sequentially: each beat synthesizes its own
    natural-length voiceover, appends a video clip sized to the measured speech (clamped
    to the source asset's end), and places the voice track over it. ``crossfade_frames``
    is applied between every clip but the last; ``final_fade_frames`` fades the last clip
    out. ``render=True`` (default) chains an ``export.render`` job with captions burned
    from the voiceover words.
    """

    name: str | None = None
    beats: list[NarratedReelBeat] = Field(min_length=1, max_length=64)
    crossfade_frames: int = Field(default=8, ge=0, le=60)
    final_fade_frames: int = Field(default=12, ge=0, le=120)
    backend: str | None = None
    voice_id: str | None = None
    language: str | None = None
    runtime_id: str | None = None
    render: bool = True
    caption_preset: Literal["reels", "tiktok", "shorts", "wide"] = "wide"


class NarratedReelAccepted(BaseModel):
    timeline_id: str
    job_id: str


# --- ai runtimes / personas -------------------------------------------------
RuntimeKind = Literal["stub", "external_http", "container"]
RuntimeEffect = Literal["voice", "reenact", "lipsync", "faceswap", "restore"]
LicenseStatus = Literal["unknown", "accepted", "rejected", "not_required"]


class AiRuntimeCreate(BaseModel):
    kind: RuntimeKind
    effect: RuntimeEffect
    display_name: str = Field(min_length=1, max_length=200)
    base_url: str | None = None
    container_image: str | None = None
    container_name: str | None = None
    port: int | None = Field(default=None, ge=1, le=65535)
    workspace_mount: str | None = None
    model_mount: str | None = None
    container_env: dict[str, str] = Field(default_factory=dict)
    requires_gpu: bool = False
    enabled: bool = True
    license_status: LicenseStatus = "unknown"


class AiRuntimeOut(AiRuntimeCreate):
    id: str
    status: dict[str, Any] = Field(default_factory=dict)
    capabilities: dict[str, Any] = Field(default_factory=dict)
    last_health_at: str | None = None
    created_at: str
    updated_at: str


class AiRuntimeEventOut(BaseModel):
    id: str
    runtime_id: str
    event_type: str
    level: str
    message: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class AiPersonaCreate(BaseModel):
    project_id: str | None = None
    name: str = Field(min_length=1, max_length=200)
    consent_id: str
    face_reference_asset_id: str | None = None
    voice_reference_asset_id: str | None = None
    style: dict[str, Any] = Field(default_factory=dict)
    allowed_effects: list[RuntimeEffect] = Field(default_factory=list)
    preferred_runtimes: dict[str, str] = Field(default_factory=dict)


class AiPersonaOut(AiPersonaCreate):
    id: str
    created_at: str
    updated_at: str


# --- overlays (replacement-lane) -------------------------------------------
class OverlayRequest(BaseModel):
    asset_id: str
    seq_in_frame: int
    seq_out_frame_exclusive: int
    lane: int = 1
    src_in_frame: int = 0


class OverlayOut(BaseModel):
    id: str
    timeline_id: str
    asset_id: str
    lane: int
    role: str
    src_in_frame: int
    src_out_frame_exclusive: int
    seq_in_frame: int
    seq_out_frame_exclusive: int


# --- shorts / next-action read-model ----------------------------------------
class NextActionOut(BaseModel):
    """Pure read-model projection: the single next action to produce a finished reel.

    ``tool`` is None when nothing can be done right now (blocked or done).
    ``label_key`` is an i18n KEY string (not human prose) for the UI.
    ``blocked_by`` carries typed blocker tokens, e.g. ``["PROXY_PENDING"]``.
    """

    short_id: str
    tool: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    label_key: str
    reason: str
    blocked_by: list[str] = Field(default_factory=list)


# --- batch-plan (P7-T1) ------------------------------------------------------

class BatchPlanIn(BaseModel):
    """Request body for POST /shorts/batch-plan.

    ``short_ids`` must be non-empty (max 1000); empty list → 422.
    Note: recipe_id is deferred to v2 (no recipe store in v1).
    """

    short_ids: list[str] = Field(min_length=1, max_length=1000)


class BatchShortPlanOut(BaseModel):
    """Per-short entry in a batch plan response.

    ``found`` is False when the asset/short does not exist (action is None but
    the batch continues).  ``hash`` is the per-short sha256 leaf hash used to
    build the ``batch_hash``.
    """

    short_id: str
    found: bool
    action: NextActionOut | None = None
    hash: str


class BatchPlanOut(BaseModel):
    """Response for POST /shorts/batch-plan.

    ``plans`` are in the same order as the request's ``short_ids``.
    ``batch_hash`` is sha256(canonical_json([leaf_hash_0, ...])) — a flat
    deterministic root over the ordered per-short hashes (v1; a real binary
    merkle tree is not required and is deferred to v2).
    """

    plans: list[BatchShortPlanOut]
    batch_hash: str


# --- batch-status (P7-T2) ---------------------------------------------------


class BatchStageCounts(BaseModel):
    """Per-stage short counts for POST /shorts/batch-status.

    All 7 stage keys are always present (zero when no shorts are in that stage)
    so the shape is stable for consumers that iterate the keys without checking.

    Stage mapping (from resolve_next_action label_key):
      next_action.preparing  → preparing
      next_action.analyzing  → analyzing
      next_action.analyze    → analyze
      next_action.cut        → cut
      next_action.build_reel → build
      next_action.done       → done
      resolve returns None   → not_found
    """

    preparing: int = 0
    analyzing: int = 0
    analyze: int = 0
    cut: int = 0
    build: int = 0
    done: int = 0
    not_found: int = 0


class BatchStatusIn(BaseModel):
    """Request body for POST /shorts/batch-status.

    ``short_ids`` must be non-empty; empty list → 422.
    """

    short_ids: list[str] = Field(min_length=1, max_length=1000)


class BatchStatusOut(BaseModel):
    """Response for POST /shorts/batch-status — conveyor rollup over a manifest.

    ``total`` == len(short_ids).
    ``by_stage`` counts how many shorts are in each pipeline stage (all 7 keys
    always present, zero-filled).
    ``needs_human`` counts shorts where get_asset_policy returns a row with
    mode == "human".  Unverified count is deferred (needs short_runs).
    """

    total: int
    by_stage: BatchStageCounts
    needs_human: int


# --- recipe-from-trace (P7-T4) ----------------------------------------------


class RecipeFromTraceOut(BaseModel):
    """Response for GET /short-runs/{run_id}/recipe.

    Reconstructs a short_run's build recipe from the export it produced and
    verifies it against the stored ``recipe_hash``.  PURE READ — no writes.

    ``available``: False when the run has no trace (queued/failed with no export)
    or the linked export has since been deleted.
    ``verified``: True when ``compute_recipe_hash(recipe) == recipe_hash`` stored
    in the short_run row.  Always False when ``available`` is False.
    ``recipe``: the reconstructed render options (export options minus
    ``short_run_id``); None when not available.
    ``recipe_hash``: the hash stored in the short_run row (may differ from a
    freshly computed hash if the row was tampered or the record predates hashing).
    """

    short_id: str
    status: str
    recipe: dict[str, Any] | None
    recipe_hash: str | None
    verified: bool
    available: bool


# --- shorts candidates (auto-cutter S5b) ------------------------------------


class ExtractShortsRequest(BaseModel):
    """Optional overrides for the ``shorts.extract`` job. All omitted → module defaults."""

    min_duration_s: float | None = Field(default=None, gt=0)
    max_duration_s: float | None = Field(default=None, gt=0)
    max_candidates: int | None = Field(default=None, gt=0)


class ExtractShortsAccepted(BaseModel):
    """Returned when a ``shorts.extract`` job is enqueued for an asset."""

    job_id: str
    analysis_run_id: str


class RenderShortRequest(BaseModel):
    """Options for rendering one short candidate to a vertical 9:16 MP4.

    ``captions`` burns the candidate's karaoke transcript (default on);
    ``hook_text`` overlays an optional top-centre hook; ``loudnorm`` applies the
    EBU R128 social loudness target (default on).
    """

    captions: bool = True
    hook_text: str | None = None
    loudnorm: bool = True


class RenderShortAccepted(BaseModel):
    """Returned when a ``shorts.render`` job is enqueued for a candidate."""

    export_id: str
    job_id: str


class ShortsCandidateOut(BaseModel):
    """One persisted short candidate (a transcript-safe ``[start, end)`` window + scores)."""

    id: str
    asset_id: str
    source_timeline_id: str
    order_index: int
    start_frame: int
    end_frame_exclusive: int
    start_boundary: str
    end_boundary: str
    score: float
    rejected: bool
    reject_reason: str | None = None
    score_breakdown: dict[str, Any] | None = None
    qa_passed: bool
    qa_issues: list[str] = Field(default_factory=list)
    created_at: str
