import { screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { ChatMessage, LauraClient } from "../../api";
import { renderWithQuery } from "../../test-utils";
import { ChatThread } from "./ChatThread";

// The 2026-08-04 white-screen: ONE crashing card render (then: DoneCard, undefined.trim)
// unmounted the entire React tree. This file mocks a whole card component to throw — the
// stand-in for the next such defect — and asserts the thread degrades that one card instead
// of dying. Module mocks are file-wide, hence the separate spec next to ChatThread.test.tsx.
vi.mock("./ActionCard", () => ({
  ActionCard: (): never => {
    throw new Error("kaputte Karte");
  },
}));

function textMessage(role: "user" | "assistant", text: string, seq: number): ChatMessage {
  return {
    id: `m${seq}`,
    conversation_id: "c1",
    seq,
    role,
    kind: "text",
    content: { text },
    created_at: "2026-01-01T00:00:00Z",
  };
}

function actionMessage(seq: number): ChatMessage {
  return {
    id: `m${seq}`,
    conversation_id: "c1",
    seq,
    role: "assistant",
    kind: "action",
    content: { tool: "import_urls", args: {}, refs: { job_ids: ["j1"] }, outcome: "running" },
    created_at: "2026-01-01T00:00:00Z",
  };
}

beforeEach(() => {
  // jsdom does not implement scrollIntoView — stub it so the auto-scroll effect doesn't throw.
  Element.prototype.scrollIntoView = vi.fn();
  // React (and jsdom's virtual console) report even boundary-caught errors via console.error.
  vi.spyOn(console, "error").mockImplementation(() => {});
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("ChatThread — per-card error boundary", () => {
  it("degrades one crashing card to a fallback and keeps rendering the rest of the thread", () => {
    const messages = [
      textMessage("user", "davor", 1),
      actionMessage(2),
      textMessage("assistant", "danach", 3),
    ];
    renderWithQuery(
      <ChatThread
        messages={messages}
        client={{} as unknown as LauraClient}
        onDecide={vi.fn()}
      />,
    );
    expect(screen.getByText("davor")).toBeTruthy();
    expect(screen.getByText("danach")).toBeTruthy();
    expect(screen.getByText("⚠ Diese Karte konnte nicht angezeigt werden.")).toBeTruthy();
  });
});
