import { type ReactElement, useEffect, useRef, useState } from "react";

import type { AgentEvent, ChatMessage, LauraClient, ProductionStatus } from "../../api";
import { parseJobError, useJobStatus } from "../../hooks/useJobStatus";
import { log } from "../../shared/log";
import { EventLine } from "../ChatPanel";
import { CardErrorBoundary } from "./CardErrorBoundary";

/** Same cadence as `useProductionSession`'s board/job poll (see hooks/useProductionSession.ts) —
 * kept in step so a chat thread narrating a session feels like the rest of the app, not a
 * separate rhythm. */
const POLL_INTERVAL_MS = 2500;

/** How many of the accumulated events show before the „alle anzeigen" expander is needed. */
const EVENT_PREVIEW_COUNT = 5;

/** The load-bearing facts of an `action` message, narrowed defensively — `content` is typed
 * `Record<string, unknown>` (see api.ts's `ChatMessage`), and its real shape comes from the
 * backend's per-tool handlers (services/local-api/src/laura/chat/executor.py):
 * `{ tool, args, refs, outcome }`, where `refs` is tool-specific (session_id/job_id for
 * start_short/start_overview/follow_up, asset_ids/job_ids for import_urls). */
interface ActionContent {
  tool: string;
  refs: Record<string, unknown>;
  outcome: string;
}

function narrowActionContent(content: Record<string, unknown>): ActionContent {
  const tool = typeof content.tool === "string" ? content.tool : "";
  const refs =
    typeof content.refs === "object" && content.refs !== null
      ? (content.refs as Record<string, unknown>)
      : {};
  const outcome = typeof content.outcome === "string" ? content.outcome : "";
  return { tool, refs, outcome };
}

/** The single job this card tracks: `refs.job_id` (start_overview) or the first entry of
 * `refs.job_ids` (import_urls — a URL import can fan out into several jobs when the URL is a
 * playlist/channel; the card follows the first one, matching its single running/done/failed
 * line). */
function firstJobId(refs: Record<string, unknown>): string | null {
  if (typeof refs.job_id === "string") return refs.job_id;
  if (Array.isArray(refs.job_ids)) {
    const first = refs.job_ids.find((entry): entry is string => typeof entry === "string");
    if (first !== undefined) return first;
  }
  return null;
}

/** The load-bearing facts of a finished production run's board status, narrowed defensively
 * (`status` is null when the one-shot final fetch itself failed). `exportId` reads off the JOB
 * (`ProductionJobState.export_id`) — the artifacts map carries no export id of its own.
 * `qaVerdict`/`qaFailedChecks` read the qa_report entry's `checks_ok`/`failed_checks` — the only
 * QA signal `ProductionArtifactState` exposes today (QaReport carries no `checks` list yet, so
 * this stays unset on the current backend; ready for the day it does, per the brief's "QA
 * verdict when present"). */
interface ProductionResult {
  exportId: string | null;
  ratioPercent: number | null;
  qaVerdict: "ship" | "revise" | null;
  qaFailedChecks: string[];
}

function narrowProductionResult(status: ProductionStatus | null): ProductionResult {
  const exportId = status?.job?.export_id ?? null;
  let ratioPercent: number | null = null;
  let qaVerdict: "ship" | "revise" | null = null;
  let qaFailedChecks: string[] = [];
  if (status !== null && status.board_ready) {
    const render = status.artifacts.render_report;
    if (typeof render?.target_ratio === "number") {
      ratioPercent = Math.round(render.target_ratio * 100);
    }
    const qa = status.artifacts.qa_report;
    if (qa?.checks_ok === true) {
      qaVerdict = "ship";
    } else if (qa?.checks_ok === false) {
      qaVerdict = "revise";
      qaFailedChecks = qa.failed_checks ?? [];
    }
  }
  return { exportId, ratioPercent, qaVerdict, qaFailedChecks };
}

/** Statuses `useJobStatus` (and the backend jobs runner) consider terminal-non-success:
 * `failed` outright, or `cancelled` — a job someone/something killed never writes its own
 * "done" event line either, so it must finalize the card the same way a `failed` job does
 * (see the job-status backstop below). */
const JOB_TERMINAL_FAILURE = new Set(["failed", "cancelled"]);

/**
 * `start_short` / `follow_up`: narrates the live run via the production session's event log
 * (`GET /production/{sessionId}/events`, polled every {@link POLL_INTERVAL_MS}) — the last
 * {@link EVENT_PREVIEW_COUNT} lines via the re-exported `EventLine`, with an „alle anzeigen"
 * expander for the rest.
 *
 * The events reader always serves the NEWEST run log for the session, which two backstops
 * guard against here (via `jobId` — `refs.job_id`, written by the executor's
 * `_handle_start_short`/`_handle_start_overview`/`_handle_follow_up`):
 * - A follow-up's first poll can land on the PREVIOUS run's log, whose last line is already
 *   `{"type":"done"}` — the events effect defers finalizing on `done` to the job-status effect
 *   below instead of trusting it outright, so a stale done never terminally shows a stale result.
 * - A dead/killed job never writes `done` at all — the job-status effect finalizes as FAILED
 *   the moment the tracked job itself reports `failed`/`cancelled`, independent of whether the
 *   events log ever said anything, so the card stops spinning „⚙ läuft …" forever.
 * A `null` jobId (an old message from before this backstop existed, or a tool that never wrote
 * one) behaves exactly as before: `done` finalizes immediately, no job cross-check.
 */
function ProductionActionCard({
  client,
  sessionId,
  jobId,
  initialOutcome,
  onFocus,
}: {
  client: LauraClient;
  sessionId: string;
  jobId: string | null;
  initialOutcome: string;
  onFocus?: () => void;
}): ReactElement {
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [phase, setPhase] = useState<"running" | "done" | "failed">(
    initialOutcome === "running" ? "running" : "done",
  );
  const [showAll, setShowAll] = useState(false);
  const [status, setStatus] = useState<ProductionStatus | null>(null);
  // Set once the events reader itself has returned a `done` batch — see the module doc above.
  // Only meaningful when `jobId` is non-null; the null-jobId path finalizes straight off the
  // events poll and never reads this.
  const [eventsDone, setEventsDone] = useState(false);
  // The events cursor lives in a ref, not state: it must be read/written synchronously across
  // poll ticks without waiting on a re-render, the same reason useProductionSession keeps its
  // generation token in a ref rather than state.
  const cursorRef = useRef(0);
  const tickInFlightRef = useRef(false);
  // Guards the job-status effect's async success finalization against firing twice (e.g. a
  // second jobStatus update landing while the first getProductionStatus fetch is still in
  // flight) — the same overlap the events poll's tickInFlightRef guards against.
  const finalizingRef = useRef(false);

  // Independent job-status poll (same `useJobStatus` every other job-backed card in this file
  // uses): self-stops once the job reaches a terminal status, and keeps its last known value
  // around afterward (never re-nulled) so the failed-render below still has a reason to show.
  const { jobStatus } = useJobStatus(client, jobId);

  useEffect(() => {
    if (phase !== "running") return;
    let cancelled = false;

    const poll = async (): Promise<void> => {
      // Guards against an overlapping tick: a slow response still in flight when the next
      // interval fires must not race a second request against the same cursor.
      if (tickInFlightRef.current) return;
      tickInFlightRef.current = true;
      try {
        const batch = await client.getProductionEvents(sessionId, cursorRef.current);
        if (cancelled) return;
        cursorRef.current = batch.next;
        setEvents((prev) => [...prev, ...batch.events]);
        if (batch.done) {
          if (jobId === null) {
            window.clearInterval(intervalId);
            let finalStatus: ProductionStatus | null = null;
            try {
              finalStatus = await client.getProductionStatus(sessionId);
            } catch (e) {
              log.warn("ActionCard: final production status fetch failed", e);
            }
            if (!cancelled) {
              setStatus(finalStatus);
              setPhase("done");
            }
          } else {
            // Stop re-polling an events log that already said it's done — but let the
            // job-status effect below decide whether the card actually finalizes.
            window.clearInterval(intervalId);
            if (!cancelled) setEventsDone(true);
          }
        }
      } catch (e) {
        if (!cancelled) log.warn("ActionCard: production events poll failed, retrying", e);
      } finally {
        tickInFlightRef.current = false;
      }
    };

    const intervalId = window.setInterval(() => {
      void poll();
    }, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [client, sessionId, phase, jobId]);

  // The job-status backstop: a failure finalizes unconditionally (fixes the dead-job case); a
  // success only finalizes once the events log has ALSO said done (fixes the stale-log case) —
  // a job can flip to "succeeded" before its own run's events poll has caught up.
  useEffect(() => {
    if (phase !== "running" || jobId === null || jobStatus === null) return;
    if (JOB_TERMINAL_FAILURE.has(jobStatus.status)) {
      setPhase("failed");
      return;
    }
    if (jobStatus.status === "succeeded" && eventsDone) {
      if (finalizingRef.current) return;
      finalizingRef.current = true;
      void (async () => {
        let finalStatus: ProductionStatus | null = null;
        try {
          finalStatus = await client.getProductionStatus(sessionId);
        } catch (e) {
          log.warn("ActionCard: final production status fetch failed", e);
        }
        setStatus(finalStatus);
        setPhase("done");
      })();
    }
  }, [client, sessionId, jobId, jobStatus, phase, eventsDone]);

  const shown = showAll ? events : events.slice(-EVENT_PREVIEW_COUNT);
  const result = narrowProductionResult(status);
  const failReason = jobStatus !== null ? parseJobError(jobStatus) ?? "unbekannter Fehler" : "unbekannter Fehler";

  return (
    <div className="mb-1.5 rounded-md border border-bezel bg-surface-2 px-1.5 py-1 text-[11px]">
      {/* Each line individually guarded: the original white-screen came from exactly one
       * defective `done` event line (see CardErrorBoundary) — one bad event must not take the
       * card's status line (or the app) with it. Slim line fallback, not the card-shaped
       * default, since these render inside this card's own frame already. */}
      {shown.map((event, i) => (
        <CardErrorBoundary
          key={i}
          fallback={
            <div className="mb-1 text-content-faint">
              ⚠ Diese Zeile konnte nicht angezeigt werden.
            </div>
          }
        >
          <EventLine event={event} />
        </CardErrorBoundary>
      ))}
      {!showAll && events.length > EVENT_PREVIEW_COUNT && (
        <button
          type="button"
          onClick={() => setShowAll(true)}
          className="mb-1 block text-[10px] text-accent hover:underline"
        >
          alle anzeigen
        </button>
      )}
      {phase === "running" && (
        <div className="animate-pulse text-content-faint" role="status">
          ⚙ läuft …
        </div>
      )}
      {phase === "failed" && (
        <div className="mt-0.5 text-status-err" role="alert">
          ✗ fehlgeschlagen: {failReason}
        </div>
      )}
      {phase === "done" &&
        (result.exportId !== null ? (
          <div className="mt-0.5 rounded border border-bezel bg-surface-1 px-1.5 py-1">
            <div className="text-content-strong">
              Export: {result.exportId}
              {result.ratioPercent !== null && ` · ${result.ratioPercent}%`}
            </div>
            {result.qaVerdict !== null && (
              <div
                className={result.qaVerdict === "ship" ? "text-status-ok" : "text-status-warn"}
              >
                QA:{" "}
                {result.qaVerdict === "ship"
                  ? "✓ ship"
                  : `⚠ revise (${result.qaFailedChecks.join(", ")})`}
              </div>
            )}
            <button
              type="button"
              onClick={onFocus}
              className="mt-0.5 text-accent hover:underline"
            >
              ▶ ansehen
            </button>
          </div>
        ) : (
          <div className="mt-0.5 text-content-faint">Kein Export erzeugt.</div>
        ))}
    </div>
  );
}

/**
 * `start_overview` / `import_urls`: a single tracked job, polled via the existing
 * `useJobStatus` hook (same 1500 ms cadence every other job-backed panel in the app uses).
 */
function JobActionCard({ client, jobId }: { client: LauraClient; jobId: string }): ReactElement {
  const { jobStatus, error, isRunning } = useJobStatus(client, jobId);
  if (isRunning) {
    return (
      <div className="text-content-faint animate-pulse" role="status">
        ⚙ läuft
      </div>
    );
  }
  if (jobStatus?.status === "succeeded") {
    return <div className="text-status-ok">✓ fertig</div>;
  }
  return (
    <div className="text-status-err" role="alert">
      ✗ fehlgeschlagen: {error ?? "unbekannter Fehler"}
    </div>
  );
}

/** Fallback for a tool this card does not (yet) know how to narrate — never crashes on an
 * unrecognized shape, just shows the raw tool name (or a generic label when even that is
 * missing). */
function UnknownActionLine({ tool }: { tool: string }): ReactElement {
  return <div className="text-content-faint">{tool !== "" ? tool : "Aktion"}</div>;
}

export interface ActionCardProps {
  message: ChatMessage;
  client: LauraClient;
  /** Focus the just-created preview (Task 11's player) — wired once that lands. */
  onFocus?: () => void;
}

/**
 * One `action` message rendered as a thread card. Dispatches on `content.tool`: the production
 * tools (`start_short`/`follow_up`) narrate live via the session's event log
 * ({@link ProductionActionCard}); the one-shot job tools (`start_overview`/`import_urls`) show a
 * plain running/done/failed line ({@link JobActionCard}).
 */
export function ActionCard({ message, client, onFocus }: ActionCardProps): ReactElement {
  const { tool, refs, outcome } = narrowActionContent(message.content);

  if (tool === "start_short" || tool === "follow_up") {
    const sessionId = typeof refs.session_id === "string" ? refs.session_id : null;
    if (sessionId === null) return <UnknownActionLine tool={tool} />;
    const jobId = typeof refs.job_id === "string" ? refs.job_id : null;
    return (
      <ProductionActionCard
        client={client}
        sessionId={sessionId}
        jobId={jobId}
        initialOutcome={outcome}
        onFocus={onFocus}
      />
    );
  }

  if (tool === "start_overview" || tool === "import_urls") {
    const jobId = firstJobId(refs);
    if (jobId === null) return <UnknownActionLine tool={tool} />;
    return <JobActionCard client={client} jobId={jobId} />;
  }

  return <UnknownActionLine tool={tool} />;
}
