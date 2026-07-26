"""Small data-access helpers. Plain SQL, returning dicts (JSON-serialisable)."""

from __future__ import annotations

import json
from typing import Any

from ..util import new_id, utcnow_iso
from .database import Database

# SQLite rejects OFFSET without LIMIT; bigint-max is a portable "no limit" sentinel
# (valid on SQLite and PostgreSQL) so callers can offset without a page size.
_NO_LIMIT = 2**63 - 1


def _paginate(
    sql: str, params: list[Any], limit: int | None, offset: int
) -> tuple[str, list[Any]]:
    """Append portable ``LIMIT/OFFSET`` when paginating; pass-through otherwise."""
    if limit is None and not offset:
        return sql, params
    effective = _NO_LIMIT if limit is None else limit
    return sql + " LIMIT ? OFFSET ?", [*params, effective, offset]


def create_project(
    db: Database,
    *,
    name: str,
    rate_num: int,
    rate_den: int,
    drop_frame: bool,
    workspace_root: str,
    project_id: str | None = None,
    org_id: str | None = None,
) -> dict[str, Any]:
    pid = project_id or new_id()
    now = utcnow_iso()
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO projects (id, name, sequence_rate_num, sequence_rate_den, "
            "drop_frame, workspace_root, created_at, org_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (pid, name, rate_num, rate_den, int(drop_frame), workspace_root, now, org_id),
        )
    project = get_project(db, pid)
    assert project is not None
    return project


def get_project(db: Database, project_id: str) -> dict[str, Any] | None:
    with db.connection() as conn:
        row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        return dict(row) if row is not None else None


def list_projects(
    db: Database, *, org_id: str | None = None,
    limit: int | None = None, offset: int = 0,
) -> list[dict[str, Any]]:
    sql = "SELECT * FROM projects"
    params: list[Any] = []
    if org_id is not None:
        sql += " WHERE org_id = ?"
        params.append(org_id)
    sql += " ORDER BY created_at DESC"
    sql, params = _paginate(sql, params, limit, offset)
    with db.connection() as conn:
        rows = (conn.execute(sql, tuple(params)) if params else conn.execute(sql)).fetchall()
        return [dict(r) for r in rows]


def count_projects(db: Database, *, org_id: str | None = None) -> int:
    with db.connection() as conn:
        if org_id is not None:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM projects WHERE org_id = ?", (org_id,)
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) AS n FROM projects").fetchone()
        return int(row["n"])


def rename_project(db: Database, project_id: str, name: str) -> bool:
    with db.transaction() as conn:
        cur = conn.execute("UPDATE projects SET name=? WHERE id=?", (name, project_id))
        return cur.rowcount > 0


def delete_project(db: Database, project_id: str) -> bool:
    with db.transaction() as conn:
        # Deleting the project cascades assets, timelines (and their clips / sequence_items / audio
        # clips / interchange exports) via ON DELETE CASCADE.
        cur = conn.execute("DELETE FROM projects WHERE id=?", (project_id,))
        # scenes / exports / consent_records carry a project_id column but have NO foreign key to
        # projects (historical schema), so the cascade never reaches them. Delete them explicitly
        # to avoid orphans. Run after the project delete: its cascade has already removed the
        # sequence_items that reference scenes, so there are no remaining referrers.
        conn.execute("DELETE FROM scenes WHERE project_id=?", (project_id,))
        conn.execute("DELETE FROM exports WHERE project_id=?", (project_id,))
        conn.execute("DELETE FROM consent_records WHERE project_id=?", (project_id,))
        return cur.rowcount > 0


def get_job(db: Database, job_id: str) -> dict[str, Any] | None:
    with db.connection() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return dict(row) if row is not None else None


def list_jobs(db: Database, *, limit: int = 50) -> list[dict[str, Any]]:
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs ORDER BY created_order DESC, created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]


def cancel_job(db: Database, job_id: str) -> bool:
    now = utcnow_iso()
    with db.transaction() as conn:
        row = conn.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            return False
        if row["status"] == "queued":
            conn.execute(
                "UPDATE jobs SET status='cancelled', cancel_requested=1, finished_at=?, "
                "updated_at=? WHERE id=?",
                (now, now, job_id),
            )
        else:
            conn.execute(
                "UPDATE jobs SET cancel_requested=1, updated_at=? WHERE id=?",
                (now, job_id),
            )
        return True


def get_fetch_job(db: Database, asset_id: str) -> dict[str, Any] | None:
    """The ingest.fetch job for an asset (keyed by its stable idempotency key)."""
    with db.connection() as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE idempotency_key = ?", (f"fetch:{asset_id}",)
        ).fetchone()
        return dict(row) if row is not None else None


def set_job_progress(db: Database, job_id: str, progress_json: str) -> None:
    """Store the latest progress sample for a job (throttled by the caller)."""
    with db.connection() as conn:
        conn.execute(
            "UPDATE jobs SET progress_json = ?, updated_at = ? WHERE id = ?",
            (progress_json, utcnow_iso(), job_id),
        )


def request_import_cancel(db: Database, asset_id: str) -> bool:
    """Flag the asset's ingest.fetch job for cooperative cancellation.

    The fetch handler polls :func:`is_import_cancelled` between progress ticks and
    aborts the in-flight download. Idempotent and a no-op when no fetch job exists
    (e.g. the import already finished). Returns True if a fetch job was flagged."""
    with db.transaction() as conn:
        cur = conn.execute(
            "UPDATE jobs SET cancel_requested = 1, updated_at = ? WHERE idempotency_key = ?",
            (utcnow_iso(), f"fetch:{asset_id}"),
        )
        return cur.rowcount > 0


def is_import_cancelled(db: Database, asset_id: str) -> bool:
    """True if the asset's ingest.fetch job has a pending cancel request."""
    with db.connection() as conn:
        row = conn.execute(
            "SELECT cancel_requested FROM jobs WHERE idempotency_key = ?",
            (f"fetch:{asset_id}",),
        ).fetchone()
        return bool(row["cancel_requested"]) if row is not None else False


_NON_TERMINAL = ("queued", "leased", "running")
_AI_KINDS = ("ai.voiceover", "ai.lipsync", "ai.reenact")


def is_job_cancel_requested(db: Database, job_id: str) -> bool:
    """True if the given job has a pending cancel request (cancel_requested=1)."""
    with db.connection() as conn:
        row = conn.execute(
            "SELECT cancel_requested FROM jobs WHERE id=?", (job_id,)
        ).fetchone()
    return bool(row["cancel_requested"]) if row is not None else False


def request_timeline_jobs_cancel(db: Database, timeline_id: str) -> list[str]:
    """Flag all non-terminal AI jobs for a timeline for cooperative cancellation.

    Returns the list of job IDs that were flagged. Calls :func:`cancel_job` on
    each matching job (queued → cancelled; running/leased → cancel_requested=1).
    """
    kinds_ph = ", ".join("?" for _ in _AI_KINDS)
    status_ph = ", ".join("?" for _ in _NON_TERMINAL)
    with db.connection() as conn:
        rows = conn.execute(
            f"SELECT id FROM jobs WHERE kind IN ({kinds_ph}) AND status IN ({status_ph}) "
            "AND json_extract(payload_json, '$.timeline_id') = ?",
            (*_AI_KINDS, *_NON_TERMINAL, timeline_id),
        ).fetchall()
    ids = [r["id"] for r in rows]
    for jid in ids:
        cancel_job(db, jid)
    return ids


# --- assets ---------------------------------------------------------------
def create_asset(
    db: Database,
    *,
    project_id: str,
    type: str,
    display_name: str,
    source_path: str,
    asset_id: str | None = None,
    online: bool = True,
    synthetic: bool = False,
    ai_effect: str | None = None,
) -> dict[str, Any]:
    aid = asset_id or new_id()
    now = utcnow_iso()
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO media_assets (id, project_id, type, display_name, source_path, "
            "online, synthetic, ai_effect, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (aid, project_id, type, display_name, source_path, int(online),
             int(synthetic), ai_effect, now),
        )
    asset = get_asset(db, aid)
    assert asset is not None
    return asset


def find_asset_by_source_path(
    db: Database, project_id: str, source_path: str
) -> dict[str, Any] | None:
    """First asset in a project with this source path (online preferred) — for relink."""
    with db.connection() as conn:
        row = conn.execute(
            "SELECT * FROM media_assets WHERE project_id=? AND source_path=? "
            "ORDER BY online DESC, created_at LIMIT 1",
            (project_id, source_path),
        ).fetchone()
        return dict(row) if row is not None else None


def get_asset(db: Database, asset_id: str) -> dict[str, Any] | None:
    with db.connection() as conn:
        row = conn.execute("SELECT * FROM media_assets WHERE id = ?", (asset_id,)).fetchone()
        return dict(row) if row is not None else None


def set_asset_synthetic(db: Database, asset_id: str, ai_effect: str | None) -> bool:
    """Mark an asset as AI-generated / synthetically modified.

    Sets ``synthetic=1`` and records the effect label (e.g. ``"reenact"``).
    Returns ``True`` when the row was found and updated.
    """
    with db.transaction() as conn:
        cur = conn.execute(
            "UPDATE media_assets SET synthetic=1, ai_effect=? WHERE id=?",
            (ai_effect, asset_id),
        )
        return cur.rowcount > 0


def list_assets(
    db: Database, project_id: str, *, limit: int | None = None, offset: int = 0
) -> list[dict[str, Any]]:
    sql = "SELECT * FROM media_assets WHERE project_id = ? ORDER BY created_at DESC"
    sql, params = _paginate(sql, [project_id], limit, offset)
    with db.connection() as conn:
        rows = conn.execute(sql, tuple(params)).fetchall()
        return [dict(r) for r in rows]


def count_assets(db: Database, project_id: str) -> int:
    with db.connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM media_assets WHERE project_id = ?", (project_id,)
        ).fetchone()
        return int(row["n"])


def update_asset_probe(
    db: Database,
    asset_id: str,
    *,
    type: str,
    duration_frames: int | None,
    rate_num: int | None,
    rate_den: int | None,
    audio_sample_rate: int | None,
    start_timecode: str | None,
    width: int | None,
    height: int | None,
    codec_video: str | None,
    codec_audio: str | None,
    is_vfr: bool,
    sha256: str | None,
) -> None:
    with db.transaction() as conn:
        conn.execute(
            "UPDATE media_assets SET type=?, duration_frames=?, rate_num=?, rate_den=?, "
            "audio_sample_rate=?, start_timecode=?, width=?, height=?, codec_video=?, "
            "codec_audio=?, is_vfr=?, sha256=? WHERE id=?",
            (type, duration_frames, rate_num, rate_den, audio_sample_rate, start_timecode,
             width, height, codec_video, codec_audio, int(is_vfr), sha256, asset_id),
        )


def set_asset_source(
    db: Database, asset_id: str, *, source_path: str, online: bool
) -> None:
    """Point an asset at its (now local) source file and flip its online flag.

    Used by the URL-ingest fetch stage once the download is verified complete.
    """
    with db.transaction() as conn:
        conn.execute(
            "UPDATE media_assets SET source_path=?, online=? WHERE id=?",
            (source_path, int(online), asset_id),
        )


def add_asset_file(
    db: Database,
    *,
    asset_id: str,
    kind: str,
    path: str,
    size_bytes: int | None = None,
    checksum: str | None = None,
    is_proxy: bool = False,
    is_waveform: bool = False,
    is_audio_extract: bool = False,
) -> dict[str, Any]:
    """Idempotent per (asset_id, kind): replaces any existing file of that kind."""
    fid = new_id()
    with db.transaction() as conn:
        conn.execute(
            "DELETE FROM asset_files WHERE asset_id=? AND kind=?", (asset_id, kind)
        )
        conn.execute(
            "INSERT INTO asset_files (id, asset_id, kind, path, size_bytes, is_proxy, "
            "is_waveform, is_audio_extract, checksum) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (fid, asset_id, kind, path, size_bytes, int(is_proxy), int(is_waveform),
             int(is_audio_extract), checksum),
        )
        row = conn.execute("SELECT * FROM asset_files WHERE id=?", (fid,)).fetchone()
    return dict(row)


def list_asset_files(db: Database, asset_id: str) -> list[dict[str, Any]]:
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT * FROM asset_files WHERE asset_id=? ORDER BY kind", (asset_id,)
        ).fetchall()
        return [dict(r) for r in rows]


# --- analysis -------------------------------------------------------------
def create_analysis_run(
    db: Database, *, asset_id: str, pipeline_version: str, config: dict[str, Any]
) -> dict[str, Any]:
    rid = new_id()
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO analysis_runs (id, asset_id, pipeline_version, status, config_json) "
            "VALUES (?, ?, ?, 'queued', ?)",
            (rid, asset_id, pipeline_version, json.dumps(config)),
        )
    run = get_analysis_run(db, rid)
    assert run is not None
    return run


def get_analysis_run(db: Database, run_id: str) -> dict[str, Any] | None:
    with db.connection() as conn:
        row = conn.execute("SELECT * FROM analysis_runs WHERE id=?", (run_id,)).fetchone()
        return dict(row) if row is not None else None


def get_latest_analysis_run(db: Database, asset_id: str) -> dict[str, Any] | None:
    with db.connection() as conn:
        row = conn.execute(
            "SELECT * FROM analysis_runs WHERE asset_id=? "
            "ORDER BY COALESCE(started_at, '') DESC, id DESC LIMIT 1",
            (asset_id,),
        ).fetchone()
        return dict(row) if row is not None else None


def start_analysis_run(db: Database, run_id: str) -> None:
    with db.transaction() as conn:
        conn.execute(
            "UPDATE analysis_runs SET status='running', started_at=? WHERE id=?",
            (utcnow_iso(), run_id),
        )


def finish_analysis_run(
    db: Database, run_id: str, *, status: str, diagnostics: dict[str, Any]
) -> None:
    with db.transaction() as conn:
        conn.execute(
            "UPDATE analysis_runs SET status=?, finished_at=?, diagnostics_json=? WHERE id=?",
            (status, utcnow_iso(), json.dumps(diagnostics), run_id),
        )


def clear_analysis_results(db: Database, *, asset_id: str, run_id: str) -> None:
    """Remove any prior shots/speakers/transcript for this run (idempotent re-run)."""
    with db.transaction() as conn:
        conn.execute("DELETE FROM shots WHERE analysis_run_id=?", (run_id,))
        conn.execute(
            "DELETE FROM transcript_words WHERE segment_id IN "
            "(SELECT id FROM transcript_segments WHERE analysis_run_id=?)",
            (run_id,),
        )
        conn.execute("DELETE FROM transcript_segments WHERE analysis_run_id=?", (run_id,))
        conn.execute("DELETE FROM speakers WHERE analysis_run_id=?", (run_id,))


def insert_shots(
    db: Database, *, asset_id: str, run_id: str, shots: list[dict[str, Any]]
) -> int:
    with db.transaction() as conn:
        for shot in shots:
            conn.execute(
                "INSERT INTO shots (id, asset_id, analysis_run_id, src_in_frame, "
                "src_out_frame_exclusive, confidence, method, thumbnail_path, "
                "black_ratio, static_score, phash, blur_score, keep, drop_reason) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (new_id(), asset_id, run_id, shot["src_in_frame"],
                 shot["src_out_frame_exclusive"], shot.get("confidence"),
                 shot.get("method"), shot.get("thumbnail_path"),
                 shot.get("black_ratio"), shot.get("static_score"), shot.get("phash"),
                 shot.get("blur_score"), 1 if shot.get("keep", True) else 0,
                 shot.get("drop_reason")),
            )
    return len(shots)


def list_shots(db: Database, asset_id: str, run_id: str) -> list[dict[str, Any]]:
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT * FROM shots WHERE asset_id=? AND analysis_run_id=? ORDER BY src_in_frame",
            (asset_id, run_id),
        ).fetchall()
        return [dict(r) for r in rows]


def insert_speaker(
    db: Database, *, asset_id: str, run_id: str, label: str
) -> str:
    sid = new_id()
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO speakers (id, asset_id, analysis_run_id, label) VALUES (?, ?, ?, ?)",
            (sid, asset_id, run_id, label),
        )
    return sid


def insert_segment_with_words(
    db: Database,
    *,
    asset_id: str,
    run_id: str,
    speaker_id: str | None,
    segment: dict[str, Any],
    words: list[dict[str, Any]],
) -> str:
    seg_id = new_id()
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO transcript_segments (id, asset_id, analysis_run_id, speaker_id, "
            "start_sample, end_sample, start_frame, end_frame, text, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                seg_id, asset_id, run_id, speaker_id,
                segment["start_sample"], segment["end_sample"],
                segment["start_frame"], segment["end_frame"],
                segment["text"], segment.get("confidence"),
            ),
        )
        for word in words:
            conn.execute(
                "INSERT INTO transcript_words (id, segment_id, idx, start_sample, end_sample, "
                "start_frame, end_frame, text, confidence, is_punctuation) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (new_id(), seg_id, word["idx"], word["start_sample"], word["end_sample"],
                 word["start_frame"], word["end_frame"], word["text"], word.get("confidence"),
                 int(word.get("is_punctuation", False))),
            )
    return seg_id


def get_transcript(db: Database, asset_id: str, run_id: str) -> list[dict[str, Any]]:
    with db.connection() as conn:
        segs = conn.execute(
            "SELECT s.*, sp.label AS speaker_label FROM transcript_segments s "
            "LEFT JOIN speakers sp ON sp.id = s.speaker_id "
            "WHERE s.asset_id=? AND s.analysis_run_id=? ORDER BY s.start_sample",
            (asset_id, run_id),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for seg in segs:
            words = conn.execute(
                "SELECT * FROM transcript_words WHERE segment_id=? ORDER BY idx", (seg["id"],)
            ).fetchall()
            row = dict(seg)
            row["words"] = [dict(w) for w in words]
            out.append(row)
        return out


def list_words_for_run(db: Database, asset_id: str, run_id: str) -> list[dict[str, Any]]:
    """All transcript words for an asset+run, ordered by source ``start_frame``.

    Flat list across segments (the segment grouping is irrelevant for editorial cut
    alignment, which only needs the word timings). Each row also carries the diarization
    ``speaker_label`` of its segment (via the segment -> speaker join; ``NULL`` when the asset was
    not diarized) so semantic placement can detect speaker turns without a per-word schema column.
    Empty when the run has no transcript.
    """
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT w.*, sp.label AS speaker_label FROM transcript_words w "
            "JOIN transcript_segments s ON s.id = w.segment_id "
            "LEFT JOIN speakers sp ON sp.id = s.speaker_id "
            "WHERE s.asset_id=? AND s.analysis_run_id=? "
            "ORDER BY w.start_frame, w.end_frame",
            (asset_id, run_id),
        ).fetchall()
        return [dict(r) for r in rows]


def get_speaker(db: Database, speaker_id: str) -> dict[str, Any] | None:
    with db.connection() as conn:
        row = conn.execute("SELECT * FROM speakers WHERE id=?", (speaker_id,)).fetchone()
        return dict(row) if row is not None else None


# --- timelines & exports --------------------------------------------------
def create_timeline(
    db: Database, *, project_id: str, name: str, kind: str, created_from: str | None = None
) -> dict[str, Any]:
    tid = new_id()
    now = utcnow_iso()
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO timelines (id, project_id, name, kind, otio_json, created_from, "
            "created_at) VALUES (?, ?, ?, ?, '{}', ?, ?)",
            (tid, project_id, name, kind, created_from, now),
        )
    timeline = get_timeline(db, tid)
    assert timeline is not None
    return timeline


def get_timeline(db: Database, timeline_id: str) -> dict[str, Any] | None:
    with db.connection() as conn:
        row = conn.execute("SELECT * FROM timelines WHERE id=?", (timeline_id,)).fetchone()
        return dict(row) if row is not None else None


def list_timelines(
    db: Database, project_id: str, *, limit: int | None = None, offset: int = 0
) -> list[dict[str, Any]]:
    sql = "SELECT * FROM timelines WHERE project_id=? ORDER BY created_at DESC"
    sql, params = _paginate(sql, [project_id], limit, offset)
    with db.connection() as conn:
        rows = conn.execute(sql, tuple(params)).fetchall()
        return [dict(r) for r in rows]


def count_timelines(db: Database, project_id: str) -> int:
    with db.connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM timelines WHERE project_id=?", (project_id,)
        ).fetchone()
        return int(row["n"])


def add_timeline_clip(
    db: Database,
    *,
    timeline_id: str,
    asset_id: str,
    src_in_frame: int,
    src_out_frame_exclusive: int,
    seq_in_frame: int,
    seq_out_frame_exclusive: int,
    lane: int = 0,
    speaker_id: str | None = None,
    origin_word_start_id: str | None = None,
    origin_word_end_id: str | None = None,
    speed_num: int = 1,
    speed_den: int = 1,
    audio_offset_samples: int = 0,
    role: str = "base",
) -> dict[str, Any]:
    cid = new_id()
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO timeline_clips (id, timeline_id, asset_id, src_in_frame, "
            "src_out_frame_exclusive, seq_in_frame, seq_out_frame_exclusive, lane, "
            "speaker_id, origin_word_start_id, origin_word_end_id, speed_num, speed_den, "
            "audio_offset_samples, role) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (cid, timeline_id, asset_id, src_in_frame, src_out_frame_exclusive,
             seq_in_frame, seq_out_frame_exclusive, lane, speaker_id,
             origin_word_start_id, origin_word_end_id, speed_num, speed_den,
             audio_offset_samples, role),
        )
        row = conn.execute("SELECT * FROM timeline_clips WHERE id=?", (cid,)).fetchone()
    return dict(row)


def update_timeline_clip_role(db: Database, clip_id: str, role: str) -> bool:
    with db.transaction() as conn:
        result = conn.execute(
            "UPDATE timeline_clips SET role=? WHERE id=?", (role, clip_id)
        )
        return result.rowcount > 0


def set_clip_transition(db: Database, *, clip_id: str, kind: str, frames: int) -> bool:
    """Set the transition that plays AFTER ``clip_id`` (mirrors set_sequence_item_transition).

    ``kind`` in {'hard','fade','crossfade'}; ``frames`` is the duration in TIMELINE frames.
    Returns ``True`` when a row was updated, ``False`` for an unknown clip.
    """
    with db.transaction() as conn:
        result = conn.execute(
            "UPDATE timeline_clips SET transition_after_kind=?, transition_after_frames=? "
            "WHERE id=?",
            (kind, int(frames), clip_id),
        )
        return result.rowcount > 0


def update_clip_frames(
    db: Database,
    clip_id: str,
    *,
    src_in_frame: int,
    src_out_frame_exclusive: int,
    seq_in_frame: int,
    seq_out_frame_exclusive: int,
) -> bool:
    """Update only a clip's four frame columns (used by a resnap roll), leaving everything else —
    crucially the transition_after_* fields — untouched. Returns True when a row changed."""
    with db.transaction() as conn:
        result = conn.execute(
            "UPDATE timeline_clips SET src_in_frame=?, src_out_frame_exclusive=?, "
            "seq_in_frame=?, seq_out_frame_exclusive=? WHERE id=?",
            (
                int(src_in_frame), int(src_out_frame_exclusive),
                int(seq_in_frame), int(seq_out_frame_exclusive), clip_id,
            ),
        )
        return result.rowcount > 0


# --- transition smoothness reviews (cached VLM verdicts) -------------------
def upsert_transition_review(
    db: Database,
    *,
    timeline_id: str,
    asset_a: str,
    asset_b: str,
    src_out_a: int,
    src_in_b: int,
    boundary_seq_frame: int,
    boundary_signature: str,
    smoothness: float,
    label: str,
    reason: str,
    suggested_fix_json: str,
    model_id: str,
    model_digest: str,
) -> None:
    """Insert or refresh a verdict, keyed by the SEMANTIC boundary identity + model digest.

    The unique key excludes ``boundary_seq_frame`` (it drifts on upstream edits), so re-reviewing
    the same source-frame pair after an unrelated edit updates the existing row instead of
    creating a duplicate (idempotency, invariant #7)."""
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO transition_reviews (id, timeline_id, asset_a, asset_b, src_out_a, "
            "src_in_b, boundary_seq_frame, boundary_signature, smoothness, label, reason, "
            "suggested_fix_json, model_id, model_digest, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(timeline_id, asset_a, asset_b, src_out_a, src_in_b, model_digest) "
            "DO UPDATE SET boundary_seq_frame=excluded.boundary_seq_frame, "
            "boundary_signature=excluded.boundary_signature, smoothness=excluded.smoothness, "
            "label=excluded.label, reason=excluded.reason, "
            "suggested_fix_json=excluded.suggested_fix_json, model_id=excluded.model_id, "
            "created_at=excluded.created_at",
            (
                new_id(), timeline_id, asset_a, asset_b, int(src_out_a), int(src_in_b),
                int(boundary_seq_frame), boundary_signature, float(smoothness), label, reason,
                suggested_fix_json, model_id, model_digest, utcnow_iso(),
            ),
        )


def list_transition_reviews(db: Database, timeline_id: str) -> list[dict[str, Any]]:
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT * FROM transition_reviews WHERE timeline_id=? ORDER BY boundary_seq_frame",
            (timeline_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_cached_review(
    db: Database,
    *,
    timeline_id: str,
    asset_a: str,
    asset_b: str,
    src_out_a: int,
    src_in_b: int,
    model_digest: str,
) -> dict[str, Any] | None:
    with db.connection() as conn:
        row = conn.execute(
            "SELECT * FROM transition_reviews WHERE timeline_id=? AND asset_a=? AND asset_b=? "
            "AND src_out_a=? AND src_in_b=? AND model_digest=?",
            (timeline_id, asset_a, asset_b, int(src_out_a), int(src_in_b), model_digest),
        ).fetchone()
        return dict(row) if row is not None else None


# --- timeline quality (persisted rough-cut quality verdict) ------------------
def set_timeline_quality(
    db: Database,
    timeline_id: str,
    *,
    status: str,
    overall: float | None = None,
    visual_exactness: float | None = None,
    editorial_cleanliness: float | None = None,
    n_cuts: int | None = None,
    n_split_cuts: int | None = None,
) -> None:
    """Upsert the quality verdict for a timeline (one row per timeline, replace on recompute)."""
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO timeline_quality "
            "(timeline_id, status, overall, visual_exactness, editorial_cleanliness, "
            "n_cuts, n_split_cuts, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(timeline_id) DO UPDATE SET "
            "status=excluded.status, overall=excluded.overall, "
            "visual_exactness=excluded.visual_exactness, "
            "editorial_cleanliness=excluded.editorial_cleanliness, "
            "n_cuts=excluded.n_cuts, n_split_cuts=excluded.n_split_cuts, "
            "created_at=excluded.created_at",
            (
                timeline_id, status,
                float(overall) if overall is not None else None,
                float(visual_exactness) if visual_exactness is not None else None,
                float(editorial_cleanliness) if editorial_cleanliness is not None else None,
                int(n_cuts) if n_cuts is not None else None,
                int(n_split_cuts) if n_split_cuts is not None else None,
                utcnow_iso(),
            ),
        )


def get_timeline_quality(db: Database, timeline_id: str) -> dict[str, Any] | None:
    """Return the persisted quality row for a timeline, or None if not yet computed."""
    with db.connection() as conn:
        row = conn.execute(
            "SELECT * FROM timeline_quality WHERE timeline_id=?", (timeline_id,)
        ).fetchone()
        return dict(row) if row is not None else None


def delete_timeline_clip(db: Database, clip_id: str) -> bool:
    with db.transaction() as conn:
        result = conn.execute(
            "DELETE FROM timeline_clips WHERE id=?", (clip_id,)
        )
        return result.rowcount > 0


def list_timeline_clips(db: Database, timeline_id: str) -> list[dict[str, Any]]:
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT * FROM timeline_clips WHERE timeline_id=? ORDER BY seq_in_frame, lane",
            (timeline_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def capture_timeline_snapshot(
    db: Database, timeline_id: str
) -> dict[str, list[dict[str, Any]]]:
    """Full editorial state of one rough-cut timeline (all columns, raw dict rows).

    Returns a dict with four keys: clips, scenes, audio_clips, transitions.
    Capturing all columns ensures that a future ADD COLUMN migration is
    not silently dropped from undo/redo snapshots.
    """
    return {
        "clips": list_timeline_clips(db, timeline_id),
        "scenes": list_scenes(db, timeline_id),
        "audio_clips": list_timeline_audio_clips(db, timeline_id),
        "transitions": list_transition_reviews(db, timeline_id),
    }


def _insert_rows(conn: Any, table: str, rows: list[dict[str, Any]]) -> None:
    # table + column names come from our OWN SELECT * snapshot — never user input.
    for r in rows:
        cols = list(r.keys())
        collist = ", ".join(cols)
        placeholders = ", ".join("?" for _ in cols)
        conn.execute(
            f"INSERT INTO {table} ({collist}) VALUES ({placeholders})",  # noqa: S608
            tuple(r[c] for c in cols),
        )


def _restore_into(
    conn: Any, timeline_id: str, snapshot: dict[str, list[dict[str, Any]]]
) -> None:
    conn.execute("DELETE FROM timeline_clips WHERE timeline_id=?", (timeline_id,))
    conn.execute("DELETE FROM timeline_audio_clips WHERE timeline_id=?", (timeline_id,))
    conn.execute("DELETE FROM transition_reviews WHERE timeline_id=?", (timeline_id,))
    conn.execute("DELETE FROM scenes WHERE source_timeline_id=?", (timeline_id,))
    _insert_rows(conn, "timeline_clips", snapshot["clips"])
    _insert_rows(conn, "timeline_audio_clips", snapshot["audio_clips"])
    _insert_rows(conn, "transition_reviews", snapshot["transitions"])
    _insert_rows(conn, "scenes", snapshot["scenes"])


def restore_timeline_snapshot(
    db: Database,
    timeline_id: str,
    snapshot: dict[str, list[dict[str, Any]]],
    *,
    conn: Any | None = None,
) -> None:
    """Atomically replace the timeline's four editorial row-groups from a snapshot.

    With ``conn`` given, runs in the caller's open transaction (used by perform_undo/redo, Task 8);
    otherwise opens its own immediate transaction. Restores EVERY column (dynamic INSERT from the
    snapshot row keys), so role/transition/linked/music columns survive — do NOT route through
    replace_timeline_clips, which drops them.
    """
    if conn is not None:
        _restore_into(conn, timeline_id, snapshot)
        return
    with db.transaction(immediate=True) as own:
        _restore_into(own, timeline_id, snapshot)


# --- sequence audio lane --------------------------------------------------
def add_timeline_audio_clip(
    db: Database,
    *,
    timeline_id: str,
    asset_id: str,
    seq_in_frame: int,
    seq_out_frame_exclusive: int,
    asset_in_frame: int = 0,
    gain_percent: int = 100,
    fade_in_frames: int = 0,
    fade_out_frames: int = 0,
    mix_mode: str = "mix",
    ducking_percent: int = 100,
    label: str | None = None,
) -> dict[str, Any]:
    cid = new_id()
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO timeline_audio_clips "
            "(id, timeline_id, asset_id, seq_in_frame, seq_out_frame_exclusive, "
            "asset_in_frame, gain_percent, fade_in_frames, fade_out_frames, mix_mode, "
            "ducking_percent, label, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                cid,
                timeline_id,
                asset_id,
                int(seq_in_frame),
                int(seq_out_frame_exclusive),
                int(asset_in_frame),
                int(gain_percent),
                int(fade_in_frames),
                int(fade_out_frames),
                mix_mode,
                int(ducking_percent),
                label,
                utcnow_iso(),
            ),
        )
        row = conn.execute("SELECT * FROM timeline_audio_clips WHERE id=?", (cid,)).fetchone()
    return dict(row)


def get_timeline_audio_clip(db: Database, clip_id: str) -> dict[str, Any] | None:
    with db.connection() as conn:
        row = conn.execute(
            "SELECT * FROM timeline_audio_clips WHERE id=?",
            (clip_id,),
        ).fetchone()
        return dict(row) if row is not None else None


def list_timeline_audio_clips(db: Database, timeline_id: str) -> list[dict[str, Any]]:
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT * FROM timeline_audio_clips WHERE timeline_id=? ORDER BY seq_in_frame, id",
            (timeline_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def update_timeline_audio_clip(
    db: Database,
    clip_id: str,
    **fields: object,
) -> dict[str, Any] | None:
    allowed = {
        "seq_in_frame",
        "seq_out_frame_exclusive",
        "asset_in_frame",
        "gain_percent",
        "fade_in_frames",
        "fade_out_frames",
        "mix_mode",
        "ducking_percent",
        "label",
    }
    updates = [(key, value) for key, value in fields.items() if key in allowed]
    if updates:
        assignments = ", ".join(f"{key}=?" for key, _ in updates)
        params = [value for _, value in updates]
        with db.transaction() as conn:
            conn.execute(
                f"UPDATE timeline_audio_clips SET {assignments} WHERE id=?",
                (*params, clip_id),
            )
    return get_timeline_audio_clip(db, clip_id)


def delete_timeline_audio_clip(db: Database, clip_id: str) -> bool:
    with db.transaction() as conn:
        result = conn.execute(
            "DELETE FROM timeline_audio_clips WHERE id=?",
            (clip_id,),
        )
        return result.rowcount > 0


def ripple_timeline_audio_clips(
    db: Database,
    timeline_id: str,
    del_seq_in: int,
    del_seq_out_excl: int,
) -> None:
    """Ripple timeline_audio_clips after a delete of [del_seq_in, del_seq_out_excl).

    Clips at/after del_seq_out_excl shift left by the deleted length.
    Clips fully inside the deleted span are dropped.
    Clips partially overlapping are clamped.
    Uses _shift_frame geometry from scenes.reconcile (same invariants).
    """
    from ..scenes.reconcile import _shift_frame

    length = del_seq_out_excl - del_seq_in
    if length <= 0:
        return
    clips = list_timeline_audio_clips(db, timeline_id)
    with db.transaction() as conn:
        for clip in clips:
            old_in = int(clip["seq_in_frame"])
            old_out = int(clip["seq_out_frame_exclusive"])
            new_in = _shift_frame(old_in, del_seq_in, del_seq_out_excl)
            new_out = _shift_frame(old_out, del_seq_in, del_seq_out_excl)
            if new_out <= new_in:
                # fully inside deleted span — drop
                conn.execute("DELETE FROM timeline_audio_clips WHERE id=?", (clip["id"],))
            elif new_in != old_in or new_out != old_out:
                conn.execute(
                    "UPDATE timeline_audio_clips "
                    "SET seq_in_frame=?, seq_out_frame_exclusive=? WHERE id=?",
                    (new_in, new_out, clip["id"]),
                )


def delete_timeline_audio_clips_overlapping(
    db: Database,
    timeline_id: str,
    seq_in: int,
    seq_out_excl: int,
    *,
    mix_mode: str | None = None,
) -> int:
    """Delete audio clips on timeline_id that overlap [seq_in, seq_out_excl).

    Overlap condition: clip.seq_in_frame < seq_out_excl AND clip.seq_out_frame_exclusive > seq_in
    Optionally filter by mix_mode. Returns count deleted.
    """
    params: list[object] = [timeline_id, seq_out_excl, seq_in]
    sql = (
        "DELETE FROM timeline_audio_clips "
        "WHERE timeline_id=? "
        "AND seq_in_frame < ? "
        "AND seq_out_frame_exclusive > ?"
    )
    if mix_mode is not None:
        sql += " AND mix_mode=?"
        params.append(mix_mode)
    with db.transaction() as conn:
        result = conn.execute(sql, params)
        return int(result.rowcount)


def create_interchange_export(
    db: Database,
    *,
    timeline_id: str,
    fmt: str,
    status: str,
    output_path: str | None,
    options: dict[str, Any],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    eid = new_id()
    now = utcnow_iso()
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO interchange_exports"
            " (id, timeline_id, format, status, output_path, options_json,"
            " diagnostics_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (eid, timeline_id, fmt, status, output_path, json.dumps(options),
             json.dumps(diagnostics), now),
        )
        row = conn.execute(
            "SELECT * FROM interchange_exports WHERE id=?", (eid,)
        ).fetchone()
    return dict(row)


def get_interchange_export(db: Database, export_id: str) -> dict[str, Any] | None:
    with db.connection() as conn:
        row = conn.execute(
            "SELECT * FROM interchange_exports WHERE id=?", (export_id,)
        ).fetchone()
        return dict(row) if row is not None else None


# --- render-pipeline exports -----------------------------------------------


def create_export(
    db: Database,
    *,
    project_id: str,
    timeline_id: str | None,
    format: str,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    eid = new_id()
    now = utcnow_iso()
    options_text = json.dumps(options) if options is not None else None
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO exports "
            "(id, project_id, timeline_id, format, status, options, created_at) "
            "VALUES (?, ?, ?, ?, 'rendering', ?, ?)",
            (eid, project_id, timeline_id, format, options_text, now),
        )
    row = get_export(db, eid)
    assert row is not None
    return row


def get_export(db: Database, export_id: str) -> dict[str, Any] | None:
    with db.connection() as conn:
        r = conn.execute("SELECT * FROM exports WHERE id=?", (export_id,)).fetchone()
        if r is None:
            return None
        row = dict(r)
        raw = row.get("options")
        row["options"] = json.loads(raw) if raw else {}
        return row


def list_exports(db: Database, project_id: str) -> list[dict[str, Any]]:
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT * FROM exports WHERE project_id=? ORDER BY created_at DESC",
            (project_id,),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            row = dict(r)
            raw = row.get("options")
            row["options"] = json.loads(raw) if raw else {}
            out.append(row)
        return out


def set_export_done(db: Database, export_id: str, *, path: str, size_bytes: int) -> None:
    with db.transaction() as conn:
        conn.execute("UPDATE exports SET status='ready', path=?, size_bytes=? WHERE id=?",
                     (path, size_bytes, export_id))


def set_export_error(db: Database, export_id: str, error: str) -> None:
    with db.transaction() as conn:
        conn.execute("UPDATE exports SET status='error', error=? WHERE id=?", (error, export_id))


def get_word(db: Database, word_id: str) -> dict[str, Any] | None:
    """A transcript word joined with its segment's asset id."""
    with db.connection() as conn:
        row = conn.execute(
            "SELECT w.*, s.asset_id AS asset_id FROM transcript_words w "
            "JOIN transcript_segments s ON s.id = w.segment_id WHERE w.id = ?",
            (word_id,),
        ).fetchone()
        return dict(row) if row is not None else None


def replace_timeline_clips(
    db: Database, timeline_id: str, rows: list[dict[str, Any]]
) -> None:
    """Atomically replace all clips of a timeline (materialised edit result).

    Every column including ``role`` is written from the row dict so a replace-overlay clip
    (role="replace") never silently reverts to the DB default ("base") after an op round-trip.
    """
    with db.transaction() as conn:
        conn.execute("DELETE FROM timeline_clips WHERE timeline_id=?", (timeline_id,))
        for r in rows:
            conn.execute(
                "INSERT INTO timeline_clips (id, timeline_id, asset_id, src_in_frame, "
                "src_out_frame_exclusive, seq_in_frame, seq_out_frame_exclusive, lane, "
                "speaker_id, origin_word_start_id, origin_word_end_id, speed_num, speed_den, "
                "audio_offset_samples, role) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (new_id(), timeline_id, r["asset_id"], r["src_in_frame"],
                 r["src_out_frame_exclusive"], r["seq_in_frame"], r["seq_out_frame_exclusive"],
                 r.get("lane", 0), r.get("speaker_id"), r.get("origin_word_start_id"),
                 r.get("origin_word_end_id"), r.get("speed_num", 1), r.get("speed_den", 1),
                 r.get("audio_offset_samples", 0), r.get("role", "base")),
            )


def set_timeline_clip_audio_offsets(
    db: Database, timeline_id: str, offsets_by_src_in: dict[int, int]
) -> None:
    """Persist accepted L/J split offsets onto the clips (2-lane editable state, samples).

    ``offsets_by_src_in`` maps a clip's ``src_in_frame`` (the inter-clip cut that begins it, i.e.
    the planner's ``seq_cut``) to that clip's signed ``audio_offset_samples`` head offset
    (invariant #3). The whole timeline is rewritten wholesale: every clip not present in the map is
    reset to ``0`` (a hard cut), so re-posting a reduced accepted set takes the omitted splits back
    to hard cuts. The first clip has no leading cut and so always resolves to ``0``.
    """
    with db.transaction() as conn:
        conn.execute(
            "UPDATE timeline_clips SET audio_offset_samples=0 WHERE timeline_id=?",
            (timeline_id,),
        )
        for src_in_frame, samples in offsets_by_src_in.items():
            conn.execute(
                "UPDATE timeline_clips SET audio_offset_samples=? "
                "WHERE timeline_id=? AND src_in_frame=?",
                (int(samples), timeline_id, int(src_in_frame)),
            )


def update_timeline_otio(db: Database, timeline_id: str, otio_json: str) -> None:
    with db.transaction() as conn:
        conn.execute(
            "UPDATE timelines SET otio_json=? WHERE id=?", (otio_json, timeline_id)
        )


# --- enterprise: orgs, users, memberships, api keys, audit -----------------
def create_org(db: Database, *, name: str) -> dict[str, Any]:
    oid = new_id()
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO organizations (id, name, created_at) VALUES (?, ?, ?)",
            (oid, name, utcnow_iso()),
        )
        row = conn.execute("SELECT * FROM organizations WHERE id=?", (oid,)).fetchone()
    return dict(row)


def get_org(db: Database, org_id: str) -> dict[str, Any] | None:
    with db.connection() as conn:
        row = conn.execute("SELECT * FROM organizations WHERE id=?", (org_id,)).fetchone()
        return dict(row) if row is not None else None


def create_user(db: Database, *, email: str, display_name: str | None = None) -> dict[str, Any]:
    uid = new_id()
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO users (id, email, display_name, created_at) VALUES (?, ?, ?, ?)",
            (uid, email, display_name, utcnow_iso()),
        )
        row = conn.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    return dict(row)


def add_membership(db: Database, *, org_id: str, user_id: str, role: str) -> dict[str, Any]:
    mid = new_id()
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO memberships (id, org_id, user_id, role, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (mid, org_id, user_id, role, utcnow_iso()),
        )
        row = conn.execute("SELECT * FROM memberships WHERE id=?", (mid,)).fetchone()
    return dict(row)


def create_api_key(
    db: Database, *, org_id: str, user_id: str | None, name: str | None, prefix: str,
    key_hash: str, role: str,
) -> dict[str, Any]:
    kid = new_id()
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO api_keys (id, org_id, user_id, name, prefix, key_hash, role, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (kid, org_id, user_id, name, prefix, key_hash, role, utcnow_iso()),
        )
        row = conn.execute("SELECT * FROM api_keys WHERE id=?", (kid,)).fetchone()
    return dict(row)


def get_api_key_by_hash(db: Database, key_hash: str) -> dict[str, Any] | None:
    with db.connection() as conn:
        row = conn.execute("SELECT * FROM api_keys WHERE key_hash=?", (key_hash,)).fetchone()
        return dict(row) if row is not None else None


def touch_api_key(db: Database, key_id: str) -> None:
    with db.transaction() as conn:
        conn.execute(
            "UPDATE api_keys SET last_used_at=? WHERE id=?", (utcnow_iso(), key_id)
        )


def revoke_api_key(db: Database, key_id: str) -> bool:
    with db.transaction() as conn:
        cur = conn.execute("UPDATE api_keys SET revoked=1 WHERE id=?", (key_id,))
        return cur.rowcount > 0


def insert_audit_event(
    db: Database, *, org_id: str | None, principal_kind: str, principal_id: str | None,
    action: str, entity_type: str | None, entity_id: str | None, payload: dict[str, Any],
) -> None:
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO audit_events (id, org_id, principal_kind, principal_id, action, "
            "entity_type, entity_id, payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (new_id(), org_id, principal_kind, principal_id, action, entity_type, entity_id,
             json.dumps(payload), utcnow_iso()),
        )


def list_audit_events(db: Database, *, limit: int = 100) -> list[dict[str, Any]]:
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT * FROM audit_events ORDER BY created_at DESC, id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


# --- deletes / renames / transcript edits / search ------------------------
def delete_asset(db: Database, asset_id: str) -> bool:
    with db.transaction() as conn:
        cur = conn.execute("DELETE FROM media_assets WHERE id=?", (asset_id,))
        return cur.rowcount > 0


def rename_timeline(db: Database, timeline_id: str, name: str) -> bool:
    with db.transaction() as conn:
        cur = conn.execute("UPDATE timelines SET name=? WHERE id=?", (name, timeline_id))
        return cur.rowcount > 0


def delete_timeline(db: Database, timeline_id: str) -> bool:
    with db.transaction() as conn:
        cur = conn.execute("DELETE FROM timelines WHERE id=?", (timeline_id,))
        return cur.rowcount > 0


def get_segment(db: Database, segment_id: str) -> dict[str, Any] | None:
    with db.connection() as conn:
        row = conn.execute(
            "SELECT s.*, sp.label AS speaker_label FROM transcript_segments s "
            "LEFT JOIN speakers sp ON sp.id = s.speaker_id WHERE s.id = ?",
            (segment_id,),
        ).fetchone()
        return dict(row) if row is not None else None


def update_segment(
    db: Database, segment_id: str, *, text: str | None = None, speaker_id: str | None = None
) -> bool:
    sets: list[str] = []
    params: list[Any] = []
    if text is not None:
        sets.append("text=?")
        params.append(text)
        sets.extend(
            [
                "alignment_status='stale'",
                "alignment_job_id=NULL",
                "alignment_error=NULL",
                "alignment_updated_at=?",
            ]
        )
        params.append(utcnow_iso())
    if speaker_id is not None:
        sets.append("speaker_id=?")
        params.append(speaker_id)
    if not sets:
        return False
    params.append(segment_id)
    with db.transaction() as conn:
        cur = conn.execute(f"UPDATE transcript_segments SET {', '.join(sets)} WHERE id=?", params)
        return cur.rowcount > 0


def mark_segments_alignment(
    db: Database,
    segment_ids: list[str],
    *,
    status: str,
    job_id: str | None = None,
    language: str | None = None,
    error: str | None = None,
) -> int:
    """Persist alignment lifecycle state for a set of transcript segments."""
    if not segment_ids:
        return 0
    placeholders = ",".join("?" for _ in segment_ids)
    with db.transaction() as conn:
        cur = conn.execute(
            "UPDATE transcript_segments SET alignment_status=?, alignment_job_id=?, "
            "alignment_language=?, alignment_error=?, alignment_updated_at=? "
            f"WHERE id IN ({placeholders})",
            [status, job_id, language, error, utcnow_iso(), *segment_ids],
        )
        return int(cur.rowcount)


def get_segment_words(db: Database, segment_id: str) -> list[dict[str, Any]]:
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT * FROM transcript_words WHERE segment_id=? ORDER BY idx", (segment_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def replace_segment_words(
    db: Database,
    segment_id: str,
    *,
    segment: dict[str, Any],
    words: list[dict[str, Any]],
) -> bool:
    """Replace one transcript segment's timing row and all child words atomically."""
    with db.transaction() as conn:
        cur = conn.execute(
            "UPDATE transcript_segments SET start_sample=?, end_sample=?, start_frame=?, "
            "end_frame=?, text=?, confidence=? WHERE id=?",
            (
                segment["start_sample"],
                segment["end_sample"],
                segment["start_frame"],
                segment["end_frame"],
                segment["text"],
                segment.get("confidence"),
                segment_id,
            ),
        )
        if cur.rowcount == 0:
            return False
        conn.execute("DELETE FROM transcript_words WHERE segment_id=?", (segment_id,))
        for word in words:
            conn.execute(
                "INSERT INTO transcript_words (id, segment_id, idx, start_sample, end_sample, "
                "start_frame, end_frame, text, confidence, is_punctuation) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    new_id(),
                    segment_id,
                    word["idx"],
                    word["start_sample"],
                    word["end_sample"],
                    word["start_frame"],
                    word["end_frame"],
                    word["text"],
                    word.get("confidence"),
                    int(word.get("is_punctuation", False)),
                ),
            )
    return True


def get_words_in_range(
    db: Database, start_word_id: str, end_word_id: str
) -> list[dict[str, Any]]:
    """Words from ``start_word_id`` to ``end_word_id`` inclusive, ordered by idx.

    Returns ``[]`` when either word is missing or they are in different segments
    (cross-segment selections are not captioned as a single cue)."""
    w0 = get_word(db, start_word_id)
    w1 = get_word(db, end_word_id)
    if w0 is None or w1 is None or w0["segment_id"] != w1["segment_id"]:
        return []
    lo, hi = sorted((int(w0["idx"]), int(w1["idx"])))
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT * FROM transcript_words WHERE segment_id=? AND idx BETWEEN ? AND ? "
            "ORDER BY idx",
            (w0["segment_id"], lo, hi),
        ).fetchall()
        return [dict(r) for r in rows]


def search_transcript(
    db: Database, *, project_id: str, query: str, limit: int = 50
) -> list[dict[str, Any]]:
    """Lexical, case-insensitive transcript search scoped to a project.

    Portable across SQLite/Postgres (LOWER + LIKE). FTS5/semantic search is a
    later optimisation (docs/15)."""
    pattern = f"%{query.lower()}%"
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT s.id AS segment_id, s.asset_id AS asset_id, a.display_name AS asset_name, "
            "s.start_frame AS start_frame, s.end_frame AS end_frame, s.text AS text, "
            "sp.label AS speaker_label "
            "FROM transcript_segments s "
            "JOIN media_assets a ON a.id = s.asset_id "
            "LEFT JOIN speakers sp ON sp.id = s.speaker_id "
            "WHERE a.project_id = ? AND LOWER(s.text) LIKE ? "
            "ORDER BY s.start_sample LIMIT ?",
            (project_id, pattern, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def get_shot(db: Database, shot_id: str) -> dict[str, Any] | None:
    with db.connection() as conn:
        row = conn.execute("SELECT * FROM shots WHERE id=?", (shot_id,)).fetchone()
        return dict(row) if row is not None else None


# --- scenes (rough-cut marker layer) ---------------------------------------

def get_scene(db: Database, scene_id: str) -> dict[str, Any] | None:
    with db.connection() as conn:
        row = conn.execute("SELECT * FROM scenes WHERE id=?", (scene_id,)).fetchone()
        return dict(row) if row is not None else None


def list_scenes(db: Database, source_timeline_id: str) -> list[dict[str, Any]]:
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT * FROM scenes WHERE source_timeline_id=? ORDER BY order_index",
            (source_timeline_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def replace_scenes(
    db: Database,
    project_id: str,
    source_timeline_id: str,
    ranges: list[tuple[int, int]],
) -> None:
    """Replace all scenes of a timeline with ``ranges`` (``(seq_in, seq_out_exclusive)``),
    ordered. Reassigns ids + ``order_index``; names are positional ("Szene N")."""
    now = utcnow_iso()
    with db.transaction() as conn:
        # Drop sequence references to the scenes we're about to delete. Regenerated scenes get
        # brand-new ids, so leaving the old sequence_items orphans them — they render as "?" and
        # 422 a later set_sequence_scenes. Cleaning here keeps cut data in sync at the source.
        conn.execute(
            "DELETE FROM sequence_items WHERE scene_id IN "
            "(SELECT id FROM scenes WHERE source_timeline_id=?)",
            (source_timeline_id,),
        )
        conn.execute("DELETE FROM scenes WHERE source_timeline_id=?", (source_timeline_id,))
        for i, (sin, sout) in enumerate(ranges):
            conn.execute(
                "INSERT INTO scenes (id, project_id, source_timeline_id, name, order_index, "
                "seq_in_frame, seq_out_frame_exclusive, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (new_id(), project_id, source_timeline_id, f"Szene {i + 1}", i, sin, sout, now),
            )


def update_scene_name(db: Database, scene_id: str, name: str) -> None:
    with db.transaction() as conn:
        conn.execute("UPDATE scenes SET name=? WHERE id=?", (name, scene_id))


def set_scene_timeline(db: Database, scene_id: str, timeline_id: str) -> None:
    with db.transaction() as conn:
        conn.execute("UPDATE scenes SET scene_timeline_id=? WHERE id=?", (timeline_id, scene_id))


def set_scene_music(db: Database, scene_id: str, asset_id: str, gain_percent: int) -> None:
    with db.transaction() as conn:
        conn.execute(
            "UPDATE scenes SET music_asset_id=?, music_gain_percent=? WHERE id=?",
            (asset_id, gain_percent, scene_id),
        )


def clear_scene_music(db: Database, scene_id: str) -> None:
    with db.transaction() as conn:
        conn.execute(
            "UPDATE scenes SET music_asset_id=NULL, music_gain_percent=100 WHERE id=?",
            (scene_id,),
        )


def get_scene_by_timeline(db: Database, scene_timeline_id: str) -> dict[str, Any] | None:
    with db.connection() as conn:
        row = conn.execute(
            "SELECT * FROM scenes WHERE scene_timeline_id=? LIMIT 1", (scene_timeline_id,)
        ).fetchone()
        return dict(row) if row is not None else None


# --- sequence items (stage 5) -----------------------------------------------

def get_or_create_project_sequence(db: Database, project_id: str) -> dict[str, Any]:
    with db.connection() as conn:
        row = conn.execute(
            "SELECT * FROM timelines WHERE project_id=? AND kind='sequence' "
            "ORDER BY created_at LIMIT 1",
            (project_id,),
        ).fetchone()
    if row is not None:
        return dict(row)
    return create_timeline(db, project_id=project_id, name="Sequenz", kind="sequence")


def list_sequence_items(db: Database, sequence_timeline_id: str) -> list[dict[str, Any]]:
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT * FROM sequence_items WHERE sequence_timeline_id=? ORDER BY order_index",
            (sequence_timeline_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def replace_sequence_items(
    db: Database, sequence_timeline_id: str, scene_ids: list[str]
) -> None:
    now = utcnow_iso()
    with db.transaction() as conn:
        conn.execute(
            "DELETE FROM sequence_items WHERE sequence_timeline_id=?", (sequence_timeline_id,)
        )
        for i, sid in enumerate(scene_ids):
            conn.execute(
                "INSERT INTO sequence_items (id, sequence_timeline_id, scene_id, order_index, "
                "transition_after_kind, transition_after_frames, created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (new_id(), sequence_timeline_id, sid, i, "hard", 0, now),
            )


def update_sequence_item_transition(
    db: Database,
    sequence_timeline_id: str,
    item_id: str,
    *,
    kind: str,
    duration_frames: int,
) -> bool:
    with db.transaction() as conn:
        row = conn.execute(
            "SELECT id FROM sequence_items WHERE id=? AND sequence_timeline_id=?",
            (item_id, sequence_timeline_id),
        ).fetchone()
        if row is None:
            return False
        cur = conn.execute(
            "UPDATE sequence_items SET transition_after_kind=?, transition_after_frames=? "
            "WHERE id=?",
            (kind, int(duration_frames), item_id),
        )
        return cur.rowcount > 0


# --- rough-cut per asset + project-wide scene list -------------------------

def get_asset_rough_cut(
    db: Database, project_id: str, asset_id: str
) -> dict[str, Any] | None:
    """The newest rough_cut timeline for this asset, or None — NEVER creates (the
    discovery ranking must not leave timelines behind; get_or_create_asset_rough_cut is
    the writing sibling)."""
    with db.connection() as conn:
        row = conn.execute(
            "SELECT * FROM timelines WHERE project_id=? AND kind='rough_cut' "
            "AND created_from=? ORDER BY created_at DESC, id DESC LIMIT 1",
            (project_id, asset_id),
        ).fetchone()
    return dict(row) if row is not None else None


def get_or_create_asset_rough_cut(
    db: Database, project_id: str, asset_id: str
) -> dict[str, Any]:
    """Return the newest rough_cut timeline for this asset, creating one if absent.

    Looks up ``timelines`` where ``project_id=?`` AND ``kind='rough_cut'`` AND
    ``created_from=asset_id``, ordered newest first.  If none exists, creates a
    fresh timeline via :func:`create_timeline` with ``created_from=asset_id``."""
    timeline = get_asset_rough_cut(db, project_id, asset_id)
    if timeline is not None:
        return timeline
    return create_timeline(
        db, project_id=project_id, name="Rough Cut", kind="rough_cut", created_from=asset_id
    )


def list_project_scenes(db: Database, project_id: str) -> list[dict[str, Any]]:
    """Scenes for the assemble bin — one set per source video, with thumbnail info.

    Returns scenes only from the *newest rough-cut timeline that still has scenes*
    for each source asset, so re-building a rough cut at a different bias no longer
    piles up stale duplicate scenes in the bin. Each scene is enriched with
    ``asset_id`` (the source video) and ``thumb_frame`` (a representative source
    frame) so the UI can render a thumbnail without extra round-trips.
    """
    with db.connection() as conn:
        timelines = conn.execute(
            "SELECT id, created_from FROM timelines "
            "WHERE project_id=? AND kind='rough_cut' ORDER BY created_at DESC, id DESC",
            (project_id,),
        ).fetchall()
        scene_rows = conn.execute(
            "SELECT * FROM scenes WHERE project_id=? ORDER BY source_timeline_id, order_index",
            (project_id,),
        ).fetchall()

    clip_cache: dict[str, list[dict[str, Any]]] = {}

    def clips_for(tid: str) -> list[dict[str, Any]]:
        if tid not in clip_cache:
            clip_cache[tid] = list_timeline_clips(db, tid)
        return clip_cache[tid]

    # Asset behind each rough-cut timeline: created_from, else its first clip's asset.
    timeline_asset: dict[str, str | None] = {}
    for t in timelines:
        aid = t["created_from"]
        if aid is None:
            cl = clips_for(t["id"])
            aid = cl[0]["asset_id"] if cl else None
        timeline_asset[t["id"]] = aid

    tids_with_scenes = {r["source_timeline_id"] for r in scene_rows}
    # Newest timeline (already ordered newest-first) per asset that actually has scenes.
    newest_for_asset: dict[str, str] = {}
    for t in timelines:
        aid = timeline_asset[t["id"]]
        if aid is not None and t["id"] in tids_with_scenes:
            newest_for_asset.setdefault(aid, t["id"])
    keep = set(newest_for_asset.values())
    # Never hide scenes whose source asset cannot be derived.
    keep |= {
        tid for tid in tids_with_scenes if timeline_asset.get(tid) is None
    }

    out: list[dict[str, Any]] = []
    for r in scene_rows:
        scene = dict(r)
        tid = scene["source_timeline_id"]
        if tid not in keep:
            continue
        clips = clips_for(tid)
        match = next(
            (
                c
                for c in clips
                if c["seq_in_frame"] <= scene["seq_in_frame"] < c["seq_out_frame_exclusive"]
            ),
            clips[0] if clips else None,
        )
        scene["asset_id"] = timeline_asset.get(tid) or (match["asset_id"] if match else None)
        scene["thumb_frame"] = int(match["src_in_frame"]) if match else 0
        out.append(scene)
    return out


# --- shorts candidates -------------------------------------------------------

def replace_shorts_candidates(
    db: Database,
    project_id: str,
    asset_id: str,
    source_timeline_id: str,
    candidates: list[Any],
) -> None:
    """Replace all shorts candidates for *asset_id* with *candidates*.

    Each element of *candidates* must expose (as attribute or dict key):
    ``start_frame``, ``end_frame_exclusive``, ``start_boundary``, ``end_boundary``,
    ``score``, ``rejected`` (bool), ``reject_reason`` (str | None),
    ``score_breakdown`` (dict → JSON), ``qa_passed`` (bool),
    ``qa_issues`` (list → JSON).

    Existing rows for the asset are deleted and replaced in one transaction;
    ``order_index`` is positional (0-based). New ``id`` and ``created_at`` are
    assigned per row. ``source_timeline_id`` is recorded as metadata but is NOT
    used as the delete key — deletion is keyed on ``asset_id`` so that re-extraction
    with a different timeline (e.g. first run used asset-id fallback, later a real
    rough_cut timeline exists) performs a true per-asset wholesale replace and never
    accumulates stale rows from a prior ``source_timeline_id``.
    """
    now = utcnow_iso()

    def _get(obj: Any, key: str) -> Any:
        return obj[key] if isinstance(obj, dict) else getattr(obj, key)

    with db.transaction() as conn:
        conn.execute(
            "DELETE FROM shorts_candidates WHERE asset_id=?",
            (asset_id,),
        )
        for i, c in enumerate(candidates):
            conn.execute(
                "INSERT INTO shorts_candidates ("
                "id, project_id, asset_id, source_timeline_id, order_index, "
                "start_frame, end_frame_exclusive, start_boundary, end_boundary, "
                "score, rejected, reject_reason, score_breakdown, "
                "qa_passed, qa_issues, created_at"
                ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    new_id(),
                    project_id,
                    asset_id,
                    source_timeline_id,
                    i,
                    int(_get(c, "start_frame")),
                    int(_get(c, "end_frame_exclusive")),
                    str(_get(c, "start_boundary")),
                    str(_get(c, "end_boundary")),
                    float(_get(c, "score")),
                    int(bool(_get(c, "rejected"))),
                    _get(c, "reject_reason"),
                    json.dumps(_get(c, "score_breakdown")),
                    int(bool(_get(c, "qa_passed"))),
                    json.dumps(_get(c, "qa_issues")),
                    now,
                ),
            )


def _decode_short_candidate(row: dict[str, Any]) -> dict[str, Any]:
    row["rejected"] = bool(row["rejected"])
    row["qa_passed"] = bool(row["qa_passed"])
    row["score_breakdown"] = json.loads(row["score_breakdown"] or "null")
    row["qa_issues"] = json.loads(row["qa_issues"] or "[]")
    return row


def list_shorts_candidates(
    db: Database, source_timeline_id: str
) -> list[dict[str, Any]]:
    """Return all shorts candidates for *source_timeline_id*, ordered by ``order_index``."""
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT * FROM shorts_candidates WHERE source_timeline_id=? ORDER BY order_index",
            (source_timeline_id,),
        ).fetchall()
    return [_decode_short_candidate(dict(r)) for r in rows]


def list_shorts_candidates_by_asset(
    db: Database, asset_id: str
) -> list[dict[str, Any]]:
    """Return all shorts candidates for *asset_id*, ordered by ``order_index``.

    Reads via the asset index (rather than the source-timeline key) so the API can list
    an asset's candidates without first resolving its binding timeline. Same JSON / bool
    deserialisation as :func:`list_shorts_candidates`.
    """
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT * FROM shorts_candidates WHERE asset_id=? ORDER BY order_index",
            (asset_id,),
        ).fetchall()
    return [_decode_short_candidate(dict(r)) for r in rows]


def get_short_candidate(db: Database, candidate_id: str) -> dict[str, Any] | None:
    """Return one candidate by primary key, or ``None`` if not found."""
    with db.connection() as conn:
        row = conn.execute(
            "SELECT * FROM shorts_candidates WHERE id=?", (candidate_id,)
        ).fetchone()
    return _decode_short_candidate(dict(row)) if row is not None else None


# --- ai runtimes / personas -------------------------------------------------

def _decode_ai_runtime(row: dict[str, Any]) -> dict[str, Any]:
    row["requires_gpu"] = bool(row.get("requires_gpu"))
    row["enabled"] = bool(row.get("enabled"))
    row["container_env"] = json.loads(row.pop("container_env_json") or "{}")
    row["status"] = json.loads(row.pop("status_cache_json") or "{}")
    row["capabilities"] = json.loads(row.pop("capabilities_json") or "{}")
    return row


def create_ai_runtime(
    db: Database,
    *,
    kind: str,
    effect: str,
    display_name: str,
    base_url: str | None = None,
    container_image: str | None = None,
    container_name: str | None = None,
    port: int | None = None,
    workspace_mount: str | None = None,
    model_mount: str | None = None,
    container_env: dict[str, str] | None = None,
    requires_gpu: bool = False,
    enabled: bool = True,
    license_status: str = "unknown",
) -> dict[str, Any]:
    runtime_id = new_id()
    now = utcnow_iso()
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO ai_runtimes "
            "(id, kind, effect, display_name, base_url, container_image, container_name, "
            "port, workspace_mount, model_mount, container_env_json, requires_gpu, enabled, "
            "license_status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                runtime_id,
                kind,
                effect,
                display_name,
                base_url,
                container_image,
                container_name,
                port,
                workspace_mount,
                model_mount,
                json.dumps(container_env or {}),
                int(requires_gpu),
                int(enabled),
                license_status,
                now,
                now,
            ),
        )
    runtime = get_ai_runtime(db, runtime_id)
    assert runtime is not None
    return runtime


def get_ai_runtime(db: Database, runtime_id: str) -> dict[str, Any] | None:
    with db.connection() as conn:
        row = conn.execute("SELECT * FROM ai_runtimes WHERE id=?", (runtime_id,)).fetchone()
    return _decode_ai_runtime(dict(row)) if row is not None else None


def list_ai_runtimes(db: Database, *, effect: str | None = None) -> list[dict[str, Any]]:
    sql = "SELECT * FROM ai_runtimes"
    params: list[Any] = []
    if effect is not None:
        sql += " WHERE effect=?"
        params.append(effect)
    sql += " ORDER BY effect, display_name"
    with db.connection() as conn:
        rows = conn.execute(sql, tuple(params)).fetchall()
    return [_decode_ai_runtime(dict(row)) for row in rows]


def update_ai_runtime_status(
    db: Database,
    runtime_id: str,
    *,
    status: dict[str, Any],
    capabilities: dict[str, Any] | None = None,
) -> bool:
    now = utcnow_iso()
    with db.transaction() as conn:
        if capabilities is None:
            cur = conn.execute(
                "UPDATE ai_runtimes SET status_cache_json=?, last_health_at=?, updated_at=? "
                "WHERE id=?",
                (json.dumps(status), now, now, runtime_id),
            )
        else:
            cur = conn.execute(
                "UPDATE ai_runtimes SET status_cache_json=?, capabilities_json=?, "
                "last_health_at=?, updated_at=? WHERE id=?",
                (json.dumps(status), json.dumps(capabilities), now, now, runtime_id),
            )
        return cur.rowcount > 0


def create_ai_runtime_event(
    db: Database,
    *,
    runtime_id: str,
    event_type: str,
    level: str,
    message: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    event_id = new_id()
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO ai_runtime_events "
            "(id, runtime_id, event_type, level, message, payload_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                event_id,
                runtime_id,
                event_type,
                level,
                message,
                json.dumps(payload or {}),
                utcnow_iso(),
            ),
        )
        row = conn.execute("SELECT * FROM ai_runtime_events WHERE id=?", (event_id,)).fetchone()
    event = dict(row)
    event["payload"] = json.loads(event.pop("payload_json") or "{}")
    return event


def list_ai_runtime_events(
    db: Database, runtime_id: str, *, limit: int = 100
) -> list[dict[str, Any]]:
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT * FROM ai_runtime_events WHERE runtime_id=? "
            "ORDER BY created_at DESC, id DESC LIMIT ?",
            (runtime_id, limit),
        ).fetchall()
    out = []
    for row in rows:
        event = dict(row)
        event["payload"] = json.loads(event.pop("payload_json") or "{}")
        out.append(event)
    return out


def _decode_ai_persona(row: dict[str, Any]) -> dict[str, Any]:
    row["style"] = json.loads(row.pop("style_json") or "{}")
    row["allowed_effects"] = json.loads(row.pop("allowed_effects_json") or "[]")
    row["preferred_runtimes"] = json.loads(row.pop("preferred_runtimes_json") or "{}")
    return row


def create_ai_persona(
    db: Database,
    *,
    name: str,
    consent_id: str,
    project_id: str | None = None,
    face_reference_asset_id: str | None = None,
    voice_reference_asset_id: str | None = None,
    style: dict[str, Any] | None = None,
    allowed_effects: list[str] | None = None,
    preferred_runtimes: dict[str, str] | None = None,
) -> dict[str, Any]:
    persona_id = new_id()
    now = utcnow_iso()
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO ai_personas "
            "(id, project_id, name, consent_id, face_reference_asset_id, "
            "voice_reference_asset_id, style_json, allowed_effects_json, "
            "preferred_runtimes_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                persona_id,
                project_id,
                name,
                consent_id,
                face_reference_asset_id,
                voice_reference_asset_id,
                json.dumps(style or {}),
                json.dumps(allowed_effects or []),
                json.dumps(preferred_runtimes or {}),
                now,
                now,
            ),
        )
    persona = get_ai_persona(db, persona_id)
    assert persona is not None
    return persona


def get_ai_persona(db: Database, persona_id: str) -> dict[str, Any] | None:
    with db.connection() as conn:
        row = conn.execute("SELECT * FROM ai_personas WHERE id=?", (persona_id,)).fetchone()
    return _decode_ai_persona(dict(row)) if row is not None else None


def list_ai_personas(db: Database, *, project_id: str | None = None) -> list[dict[str, Any]]:
    sql = "SELECT * FROM ai_personas"
    params: list[Any] = []
    if project_id is not None:
        sql += " WHERE project_id=? OR project_id IS NULL"
        params.append(project_id)
    sql += " ORDER BY created_at DESC"
    with db.connection() as conn:
        rows = conn.execute(sql, tuple(params)).fetchall()
    return [_decode_ai_persona(dict(row)) for row in rows]


# --- consent records -------------------------------------------------------

def create_consent_record(
    db: Database,
    *,
    project_id: str,
    subject_label: str,
    source_asset_id: str | None = None,
    confirmed_by: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    """Create a consent record for a named subject, confirmed right now."""
    cid = new_id()
    confirmed_at = utcnow_iso()
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO consent_records "
            "(id, project_id, subject_label, source_asset_id, confirmed_by, confirmed_at, note) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (cid, project_id, subject_label, source_asset_id, confirmed_by, confirmed_at, note),
        )
    record = get_consent_record(db, cid)
    assert record is not None
    return record


# --- demo drafts -----------------------------------------------------------

def create_demo_draft(
    db: Database,
    *,
    project_id: str,
    asset_id: str,
    status: str = "analyzing",
    items: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    draft_id = new_id()
    now = utcnow_iso()
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO demo_drafts "
            "(id, project_id, asset_id, status, items_json, result_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, '{}', ?, ?)",
            (draft_id, project_id, asset_id, status, json.dumps(items or []), now, now),
        )
    draft = get_demo_draft(db, draft_id)
    assert draft is not None
    return draft


def get_demo_draft(db: Database, draft_id: str) -> dict[str, Any] | None:
    with db.connection() as conn:
        row = conn.execute("SELECT * FROM demo_drafts WHERE id=?", (draft_id,)).fetchone()
    if row is None:
        return None
    return _decode_demo_draft(dict(row))


def list_demo_drafts_for_asset(db: Database, asset_id: str) -> list[dict[str, Any]]:
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT * FROM demo_drafts WHERE asset_id=? ORDER BY created_at DESC",
            (asset_id,),
        ).fetchall()
    return [_decode_demo_draft(dict(row)) for row in rows]


def update_demo_draft(
    db: Database,
    draft_id: str,
    *,
    status: str | None = None,
    items: list[dict[str, Any]] | None = None,
    result: dict[str, Any] | None = None,
    applied: bool = False,
) -> dict[str, Any] | None:
    current = get_demo_draft(db, draft_id)
    if current is None:
        return None
    next_status = status if status is not None else str(current["status"])
    next_items = items if items is not None else list(current["items"])
    next_result = result if result is not None else dict(current["result"])
    now = utcnow_iso()
    applied_at = now if applied else current.get("applied_at")
    with db.transaction() as conn:
        conn.execute(
            "UPDATE demo_drafts SET status=?, items_json=?, result_json=?, updated_at=?, "
            "applied_at=? WHERE id=?",
            (
                next_status,
                json.dumps(next_items),
                json.dumps(next_result),
                now,
                applied_at,
                draft_id,
            ),
        )
    return get_demo_draft(db, draft_id)


def _decode_demo_draft(row: dict[str, Any]) -> dict[str, Any]:
    row["items"] = json.loads(row.get("items_json") or "[]")
    row["result"] = json.loads(row.get("result_json") or "{}")
    return row


def get_consent_record(db: Database, consent_id: str) -> dict[str, Any] | None:
    with db.connection() as conn:
        row = conn.execute(
            "SELECT * FROM consent_records WHERE id=?", (consent_id,)
        ).fetchone()
        return dict(row) if row is not None else None


def list_consent_records(db: Database, project_id: str) -> list[dict[str, Any]]:
    """All consent records for a project, newest first."""
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT * FROM consent_records WHERE project_id=? ORDER BY confirmed_at DESC",
            (project_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_active_consent_id(db: Database, project_id: str) -> str | None:
    """Newest non-revoked consent id for a project, or None.

    Single-subject MVP for the auto-pipeline (spec §5 "consent once per subject"):
    the most recent confirmed-and-not-revoked record is reused by auto-lipsync jobs.
    """
    with db.connection() as conn:
        row = conn.execute(
            "SELECT id FROM consent_records WHERE project_id=? AND revoked_at IS NULL "
            "ORDER BY confirmed_at DESC LIMIT 1",
            (project_id,),
        ).fetchone()
        return str(row["id"]) if row is not None else None


_UNDO_DEPTH = 50


def push_row(conn: Any, timeline_id: str, stack: str, label: str, snapshot: dict[str, Any]) -> None:
    seq = conn.execute(
        "SELECT COALESCE(MAX(seq_no), 0) + 1 AS n FROM timeline_history WHERE timeline_id=?",
        (timeline_id,),
    ).fetchone()["n"]
    cols = (
        "id, timeline_id, seq_no, stack, label, payload_json, created_at"
    )
    conn.execute(
        f"INSERT INTO timeline_history ({cols}) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (new_id(), timeline_id, seq, stack, label, json.dumps(snapshot), utcnow_iso()),
    )


def pop_top(conn: Any, timeline_id: str, stack: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT id, seq_no, label, payload_json FROM timeline_history "
        "WHERE timeline_id=? AND stack=? ORDER BY seq_no DESC LIMIT 1",
        (timeline_id, stack),
    ).fetchone()
    if row is None:
        return None
    conn.execute("DELETE FROM timeline_history WHERE id=?", (row["id"],))
    return {
        "id": row["id"],
        "seq_no": row["seq_no"],
        "label": row["label"],
        "payload": json.loads(row["payload_json"]),
    }


def push_undo_checkpoint(db: Database, timeline_id: str, label: str) -> None:
    """Snapshot the current editorial state onto the undo stack; clear redo; cap depth."""
    snapshot = capture_timeline_snapshot(db, timeline_id)
    with db.transaction() as conn:
        push_row(conn, timeline_id, "undo", label, snapshot)
        conn.execute(
            "DELETE FROM timeline_history WHERE timeline_id=? AND stack='redo'",
            (timeline_id,),
        )
        cnt = conn.execute(
            "SELECT COUNT(*) AS c FROM timeline_history "
            "WHERE timeline_id=? AND stack='undo'",
            (timeline_id,),
        ).fetchone()["c"]
        if cnt > _UNDO_DEPTH:
            conn.execute(
                "DELETE FROM timeline_history WHERE id IN ("
                "SELECT id FROM timeline_history WHERE timeline_id=? AND stack='undo' "
                "ORDER BY seq_no ASC LIMIT ?)",
                (timeline_id, cnt - _UNDO_DEPTH),
            )


def get_history_state(db: Database, timeline_id: str) -> dict[str, Any]:
    with db.connection() as conn:
        u = conn.execute(
            "SELECT label FROM timeline_history WHERE timeline_id=? AND stack='undo' "
            "ORDER BY seq_no DESC LIMIT 1",
            (timeline_id,),
        ).fetchone()
        r = conn.execute(
            "SELECT label FROM timeline_history WHERE timeline_id=? AND stack='redo' "
            "ORDER BY seq_no DESC LIMIT 1",
            (timeline_id,),
        ).fetchone()
    return {
        "can_undo": u is not None,
        "can_redo": r is not None,
        "undo_label": u["label"] if u is not None else None,
        "redo_label": r["label"] if r is not None else None,
    }


def revoke_consent_record(db: Database, consent_id: str) -> bool:
    """Mark a consent record as revoked. The reenact gate then refuses it.

    Returns True when the record existed and was updated.
    """
    with db.transaction() as conn:
        cur = conn.execute(
            "UPDATE consent_records SET revoked_at=? WHERE id=?",
            (utcnow_iso(), consent_id),
        )
        return cur.rowcount > 0


# --- production sessions (agentic short-creator) ----------------------------


def create_production_session(
    db: Database, *, session_id: str, asset_id: str, created_utc: str
) -> None:
    """Create a production session record. session_id must be unique (PK)."""
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO production_sessions (session_id, asset_id, created_utc) "
            "VALUES (?, ?, ?)",
            (session_id, asset_id, created_utc),
        )


def set_production_session_job(db: Database, session_id: str, job_id: str) -> None:
    """Point a session at the job currently running it — the source of its liveness.

    Called on every run enqueued for the session (create and each follow-up), so the status
    endpoint always reports the LATEST run's state, not the first one's.
    """
    with db.transaction() as conn:
        conn.execute(
            "UPDATE production_sessions SET latest_job_id=? WHERE session_id=?",
            (job_id, session_id),
        )


def get_production_session(db: Database, session_id: str) -> dict[str, Any] | None:
    """Get a production session by session_id, or None if not found."""
    with db.connection() as conn:
        row = conn.execute(
            "SELECT * FROM production_sessions WHERE session_id=?", (session_id,)
        ).fetchone()
        return dict(row) if row is not None else None


def list_production_sessions(db: Database, asset_id: str) -> list[dict[str, Any]]:
    """List all production sessions for an asset, newest first."""
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT * FROM production_sessions WHERE asset_id=? ORDER BY created_utc DESC",
            (asset_id,),
        ).fetchall()
        return [dict(r) for r in rows]
