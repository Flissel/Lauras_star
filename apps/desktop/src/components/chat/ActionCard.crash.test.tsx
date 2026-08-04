import { act, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { AgentEvent, ChatMessage, JobStatus, LauraClient } from "../../api";
import { renderWithQuery } from "../../test-utils";
import { ActionCard } from "./ActionCard";

// Same rationale as ChatThread.crash.test.tsx, one level down: a single defective event LINE
// inside a running production card must not take the card (or the app) with it. `EventLine` is
// ActionCard's only import from ChatPanel, so the module mock stays this small.
vi.mock("../ChatPanel", () => ({
  EventLine: (): never => {
    throw new Error("kaputte Zeile");
  },
}));

function actionMessage(tool: string, refs: Record<string, unknown>): ChatMessage {
  return {
    id: "m1",
    conversation_id: "c1",
    seq: 3,
    role: "assistant",
    kind: "action",
    content: { tool, args: {}, refs, outcome: "running" },
    created_at: "2026-01-01T00:00:00Z",
  };
}

function job(overrides: Partial<JobStatus> = {}): JobStatus {
  return {
    id: "j1",
    queue: "default",
    kind: "ingest.fetch",
    status: "running",
    attempt: 1,
    max_attempts: 3,
    result_json: null,
    error_json: null,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    finished_at: null,
    ...overrides,
  };
}

beforeEach(() => {
  vi.useFakeTimers();
  vi.spyOn(console, "error").mockImplementation(() => {});
});

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe("ActionCard — per-event-line error boundary", () => {
  it("degrades a crashing event line without killing the running card", async () => {
    const events: AgentEvent[] = [{ type: "agent", agent: "scout", text: "sucht Momente" }];
    const c = {
      getProductionEvents: vi.fn().mockResolvedValue({ events, next: 1, done: false }),
      getProductionStatus: vi.fn(),
      getJob: vi.fn().mockResolvedValue(job()),
    } as unknown as LauraClient;
    renderWithQuery(
      <ActionCard
        message={actionMessage("start_short", { session_id: "s1", job_id: "j1" })}
        client={c}
      />,
    );

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2500);
    });

    expect(screen.getByText("⚠ Diese Zeile konnte nicht angezeigt werden.")).toBeTruthy();
    expect(screen.getByText("⚙ läuft …")).toBeTruthy();
  });
});
