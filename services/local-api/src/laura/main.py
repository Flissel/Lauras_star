"""FastAPI application factory and entrypoint.

Binds to loopback only (local-first). Migrates the DB and starts the background
job runner on startup; stops it cleanly on shutdown.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from . import PIPELINE_VERSION, __version__
from .ai.handlers import register_ai_handlers
from .analysis.handlers import register_analysis_handlers
from .analysis.recovery import recover_stranded_analysis_runs
from .analysis.shorts_handlers import register_shorts_handlers
from .analysis.visual_embed import register_visual_handlers
from .api import (
    admin,
    ai_runtimes,
    analysis,
    assets,
    audio,
    batch,
    bausteine,
    chat,
    demo,
    generate,
    jobs,
    lipsync,
    narrated_reel,
    orchestration,
    overlays,
    projects,
    reels,
    reenact,
    scenes,
    search,
    sequences,
    short_creator,
    shorts,
    shorts_candidates,
    timelines,
    voiceover,
)
from .api.models import HealthOut
from .api.ratelimit import RateLimiter, make_rate_limit_middleware
from .config import Settings, ensure_workspace
from .db.database import create_database
from .demo.handlers import register_demo_handlers
from .bausteine.handlers import register_bausteine_handlers
from .generate.handlers import register_generate_handlers
from .ingest.handlers import register_ingest_handlers
from .jobs import JobRunner, default_registry
from .metrics import metrics_middleware, metrics_response
from .render.handlers import register_render_handlers
from .render.shorts_render import register_shorts_render_handler
from .short_creator.handlers import register_short_creator_handlers
from .telemetry import configure_tracing

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.load()
    ensure_workspace(settings)
    configure_tracing()  # no-op unless OTEL_EXPORTER_OTLP_ENDPOINT + otel extra are present
    db = create_database(settings)
    db.migrate()
    registry = default_registry()
    register_ingest_handlers(registry)
    register_analysis_handlers(registry)
    register_render_handlers(registry)
    register_ai_handlers(registry)
    register_demo_handlers(registry)
    register_shorts_handlers(registry)
    register_visual_handlers(registry)
    register_shorts_render_handler(registry)
    register_generate_handlers(registry)
    register_bausteine_handlers(registry)
    register_short_creator_handlers(registry)
    runner = JobRunner(
        db, registry,
        lease_seconds=settings.lease_seconds,
        concurrency=settings.worker_concurrency,
        max_runtime_seconds=settings.job_max_runtime_seconds,
        # The agent team runs far longer than any other job kind — give it its own cap
        # instead of loosening the one that catches hung quick jobs.
        runtime_overrides={"production.run": settings.production_run_max_runtime_seconds},
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # A worker that was SIGKILLed leaves its analysis run saying 'running' forever, and
        # the job reaper only ever sees leases that expire from now on. Heal what the last
        # process left behind before this one starts working. This is an opportunistic repair,
        # not load-bearing like the migration above: a locked DB (a second Laura process) or
        # any unexpected row must never take down the whole API on startup.
        try:
            recover_stranded_analysis_runs(db)
        except Exception:  # noqa: BLE001 - a repair pass must never block startup
            logger.exception("startup sweep for stranded analysis runs failed")
        if settings.start_runner:
            runner.start()
        try:
            yield
        finally:
            runner.stop()

    app = FastAPI(title="Laura Local API", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.db = db
    app.state.runner = runner
    # The desktop renderer (Electron) is a separate web origin from this loopback
    # service, and the X-Laura-Token header makes every call a CORS-preflighted request.
    # Allow the renderer origins so the browser doesn't block it: the Vite dev server
    # (localhost/127.0.0.1) and the packaged file:// renderer (Origin "null").
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["null"],
        allow_origin_regex=r"^http://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.middleware("http")(metrics_middleware)
    if settings.rate_limit_rpm > 0:
        capacity = settings.rate_limit_burst or settings.rate_limit_rpm
        limiter = RateLimiter(capacity=capacity, refill_per_sec=settings.rate_limit_rpm / 60.0)
        app.state.rate_limiter = limiter
        app.middleware("http")(make_rate_limit_middleware(limiter))

    @app.get("/healthz", response_model=HealthOut, tags=["health"])
    def healthz() -> HealthOut:
        return HealthOut(
            status="ok",
            version=__version__,
            pipeline_version=PIPELINE_VERSION,
            schema_version=db.schema_version(),
        )

    @app.get("/metrics", tags=["observability"])
    def metrics() -> Response:
        return metrics_response()

    app.include_router(projects.router)
    app.include_router(shorts.router)
    app.include_router(shorts_candidates.router)
    app.include_router(batch.router)
    app.include_router(assets.router)
    app.include_router(jobs.router)
    app.include_router(analysis.router)
    app.include_router(timelines.router)
    app.include_router(audio.router)
    app.include_router(reels.router)
    app.include_router(overlays.router)
    app.include_router(reenact.router)
    app.include_router(ai_runtimes.router)
    app.include_router(admin.router)
    app.include_router(search.router)
    app.include_router(scenes.router)
    app.include_router(sequences.router)
    app.include_router(voiceover.router)
    app.include_router(lipsync.router)
    app.include_router(narrated_reel.router)
    app.include_router(orchestration.router)
    app.include_router(demo.router)
    app.include_router(generate.router)
    app.include_router(bausteine.router)
    app.include_router(short_creator.router)
    app.include_router(chat.router)
    return app


def run() -> None:
    """Console entrypoint (`laura-api`)."""
    import uvicorn

    settings = Settings.load()
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)


if __name__ == "__main__":
    run()
