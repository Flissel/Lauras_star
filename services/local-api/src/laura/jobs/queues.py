"""Queue names and routing policy (docs/05-workers-queue.md).

The claim already orders by ``priority DESC, created_at ASC`` and filters by queue, so
priority and routing happen in the database. This module names the queues and groups
them into CPU vs GPU: CPU work runs anywhere, while the GPU queue (ASR / forced
alignment / diarization / embeddings) is consumed only by workers that have the heavy
ML extras and a GPU. The desktop runner consumes all queues; server-mode workers start
with a queue subset (``CPU_QUEUES`` or ``GPU_QUEUES``) so GPU stages land on GPU hosts.
"""

from __future__ import annotations

QUEUE_INGEST = "ingest.io"  # ffprobe / IO-bound import
QUEUE_PROXY = "proxy.cpu"  # proxy / audio / waveform (CPU, ffmpeg)
QUEUE_ANALYSIS_CPU = "analysis.scene"  # scene detection + orchestrator (CPU)
QUEUE_ANALYSIS_GPU = "analysis.gpu"  # ASR / align / diarize / embeddings (GPU)
QUEUE_EXPORT = "export"

CPU_QUEUES: tuple[str, ...] = (QUEUE_INGEST, QUEUE_PROXY, QUEUE_ANALYSIS_CPU, QUEUE_EXPORT)
GPU_QUEUES: tuple[str, ...] = (QUEUE_ANALYSIS_GPU,)
ALL_QUEUES: tuple[str, ...] = CPU_QUEUES + GPU_QUEUES

# job kind -> queue it should be enqueued on
_STAGE_QUEUE: dict[str, str] = {
    "ingest.fetch": QUEUE_INGEST,
    "ingest.probe": QUEUE_INGEST,
    "proxy.build": QUEUE_PROXY,
    "audio.extract": QUEUE_PROXY,
    "waveform.build": QUEUE_PROXY,
    "analysis.run": QUEUE_ANALYSIS_CPU,
    "analysis.align": QUEUE_ANALYSIS_GPU,
    "analysis.embed": QUEUE_ANALYSIS_GPU,
    "transcript.realign": QUEUE_ANALYSIS_GPU,
    "transition.review": QUEUE_ANALYSIS_GPU,
    "ai.voiceover": QUEUE_ANALYSIS_GPU,
    "ai.lipsync": QUEUE_ANALYSIS_GPU,
    "demo.analyze": QUEUE_ANALYSIS_CPU,
    "export.render": QUEUE_EXPORT,
    "shorts.render": QUEUE_EXPORT,
    "baustein.standbild": QUEUE_EXPORT,  # Reel-Bausteine (ffmpeg, CPU)
    "baustein.titelkarte": QUEUE_EXPORT,
    "baustein.standbild": QUEUE_EXPORT,  # Reel-Bausteine (ffmpeg, CPU)
    "baustein.titelkarte": QUEUE_EXPORT,
    "shorts.embed_frames": QUEUE_ANALYSIS_GPU,  # CLIP frame embeddings (visual search/hook)
    "short_creator.run": QUEUE_ANALYSIS_CPU,  # NL-agent orchestration (CPU; calls local Ollama)
    "production.run": QUEUE_ANALYSIS_CPU,  # v2 production-board orchestration (CPU; local Ollama)
}


def queue_for(kind: str, default: str = QUEUE_ANALYSIS_CPU) -> str:
    """Queue a job ``kind`` should run on (GPU stages route to the GPU queue)."""
    return _STAGE_QUEUE.get(kind, default)
