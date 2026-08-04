import { type ReactElement, useEffect, useRef } from "react";

import type { ChatMessage, LauraClient } from "../../api";
import { ActionCard } from "./ActionCard";
import { ApprovalCard } from "./ApprovalCard";
import { CardErrorBoundary } from "./CardErrorBoundary";

/** The load-bearing fact of a `text` message — `content` is typed `Record<string, unknown>`
 * (see api.ts's `ChatMessage`); the real shape comes from the backend's `{"text": text}` writes
 * for both user turns and `reply` assistant turns (services/local-api/src/laura/api/chat.py,
 * laura/chat/executor.py). */
function narrowTextContent(content: Record<string, unknown>): string {
  return typeof content.text === "string" ? content.text : "";
}

function TextBubble({ message }: { message: ChatMessage }): ReactElement {
  const text = narrowTextContent(message.content);
  const isUser = message.role === "user";
  return (
    <div className={`mb-1.5 flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[85%] whitespace-pre-wrap break-words rounded-md px-1.5 py-1 text-[11px] ${
          isUser
            ? "bg-accent/20 text-content-strong"
            : "border border-bezel bg-surface-2 text-content-muted"
        }`}
      >
        {text}
      </div>
    </div>
  );
}

export interface ChatThreadProps {
  messages: ChatMessage[];
  client: LauraClient;
  /** Per-message approval decision. The caller (ChatStage, Task 12) closes over the
   * conversation id: `onDecide={(messageId, decision) => client.decideApproval(conversationId,
   * messageId, decision)}`. `ChatThread` supplies the message id; each `ApprovalCard` only ever
   * sees its own `onDecide: (decision) => void`, so it stays ignorant of message ids entirely. */
  onDecide: (messageId: string, decision: "approve" | "reject") => void;
  /** Focus the artifact an `action` message produced (Task 11's preview). Omitted in tests/call
   * sites that don't need it — `ActionCard`'s own `onFocus` is optional for the same reason. */
  onFocusAction?: (messageId: string) => void;
}

/**
 * One conversation's message list: text bubbles for plain turns, `ApprovalCard` for
 * `approval_request`, `ActionCard` for `action` — dispatch is purely on `message.kind`, matching
 * the discriminated `ChatMessageKind` in api.ts. Auto-scrolls a sentinel at the tail into view
 * whenever `messages` changes, so a new turn (or a poll-driven card update) keeps the newest
 * content visible without the user scrolling manually.
 *
 * Every card renders inside a per-message `CardErrorBoundary` — the single wrap point in the
 * map below is deliberate, so a future card kind cannot ship unguarded (see CardErrorBoundary's
 * doc for the white-screen this prevents).
 */
export function ChatThread({
  messages,
  client,
  onDecide,
  onFocusAction,
}: ChatThreadProps): ReactElement {
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView?.({ block: "end" });
  }, [messages]);

  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-y-auto px-2 py-1.5">
      {messages.map((message) => (
        <CardErrorBoundary key={message.id}>
          {message.kind === "approval_request" ? (
            <ApprovalCard
              message={message}
              onDecide={(decision) => onDecide(message.id, decision)}
            />
          ) : message.kind === "action" ? (
            <ActionCard
              message={message}
              client={client}
              onFocus={onFocusAction ? () => onFocusAction(message.id) : undefined}
            />
          ) : (
            <TextBubble message={message} />
          )}
        </CardErrorBoundary>
      ))}
      <div ref={bottomRef} />
    </div>
  );
}
