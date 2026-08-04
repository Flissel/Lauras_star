import { render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { CardErrorBoundary } from "./CardErrorBoundary";

/** A child whose render always crashes — the stand-in for a future defective card. */
function Bomb(): never {
  throw new Error("kaboom");
}

beforeEach(() => {
  // React (and jsdom's virtual console) report even boundary-caught errors via console.error —
  // silence it so the crash tests keep the output pristine.
  vi.spyOn(console, "error").mockImplementation(() => {});
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("CardErrorBoundary", () => {
  it("renders its children when nothing throws", () => {
    render(
      <CardErrorBoundary>
        <div>gesund</div>
      </CardErrorBoundary>,
    );
    expect(screen.getByText("gesund")).toBeTruthy();
  });

  it("renders the German fallback card when a child render throws", () => {
    render(
      <CardErrorBoundary>
        <Bomb />
      </CardErrorBoundary>,
    );
    expect(screen.getByText("⚠ Diese Karte konnte nicht angezeigt werden.")).toBeTruthy();
  });

  it("renders a custom fallback when one is provided", () => {
    render(
      <CardErrorBoundary fallback={<div>eigener Fallback</div>}>
        <Bomb />
      </CardErrorBoundary>,
    );
    expect(screen.getByText("eigener Fallback")).toBeTruthy();
    expect(screen.queryByText("⚠ Diese Karte konnte nicht angezeigt werden.")).toBeNull();
  });

  it("only replaces its own subtree — a sibling boundary's healthy child survives", () => {
    render(
      <>
        <CardErrorBoundary>
          <Bomb />
        </CardErrorBoundary>
        <CardErrorBoundary>
          <div>Nachbar lebt</div>
        </CardErrorBoundary>
      </>,
    );
    expect(screen.getByText("⚠ Diese Karte konnte nicht angezeigt werden.")).toBeTruthy();
    expect(screen.getByText("Nachbar lebt")).toBeTruthy();
  });
});
