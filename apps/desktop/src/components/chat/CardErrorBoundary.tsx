import { Component, type ErrorInfo, type ReactNode } from "react";

import { log } from "../../shared/log";

export interface CardErrorBoundaryProps {
  children: ReactNode;
  /** Replaces the default card-shaped fallback — for wrap sites where a full card frame would
   * look wrong (e.g. a single event line inside an already-framed ActionCard). */
  fallback?: ReactNode;
}

interface CardErrorBoundaryState {
  hasError: boolean;
}

/**
 * Per-card render guard for the chat thread. Without one, a single throwing card render
 * unmounts the ENTIRE React tree — the 2026-08-04 white-screen, where one DoneCard crashing on
 * a resume-path done event killed the whole app with no recovery. Wrapped per message (and per
 * event line), one defective card degrades to a small German fallback instead.
 *
 * A class component because React error boundaries cannot be written with hooks. No reset
 * logic: the boundary is keyed per message/line at its wrap sites, so a crashed card stays
 * degraded until its subtree remounts — re-rendering the same broken content would just throw
 * again.
 */
export class CardErrorBoundary extends Component<CardErrorBoundaryProps, CardErrorBoundaryState> {
  state: CardErrorBoundaryState = { hasError: false };

  static getDerivedStateFromError(): CardErrorBoundaryState {
    return { hasError: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    log.error("CardErrorBoundary: card render crashed", error, info.componentStack);
  }

  render(): ReactNode {
    if (this.state.hasError) {
      return (
        this.props.fallback ?? (
          <div className="mb-1.5 rounded-md border border-bezel bg-surface-2 px-1.5 py-1 text-[11px] text-content-faint">
            ⚠ Diese Karte konnte nicht angezeigt werden.
          </div>
        )
      );
    }
    return this.props.children;
  }
}
