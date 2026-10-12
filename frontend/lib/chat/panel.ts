// State and dismissal policy of the sources side panel. The panel is non-modal: the chat
// behind it stays interactive and scrollable, so the reader can read the answer and the
// cited passage side by side. Plain TypeScript so it can be unit-tested.

import type { Source } from "./events.ts";

/** The source on show, and all sources of that answer (for the "other sources" list). */
export interface PanelState {
  source: Source;
  all: readonly Source[];
}

export type PanelAction =
  | { type: "open"; source: Source; all: readonly Source[] } // a chip was clicked
  | { type: "select"; source: Source } // a source was picked inside the panel
  | { type: "close" };

/** `null` is a closed panel. */
export function panelReducer(
  state: PanelState | null,
  action: PanelAction,
): PanelState | null {
  switch (action.type) {
    case "open":
      // Also when already open: a chip in the chat switches the panel to that source.
      return { source: action.source, all: action.all };
    case "select":
      if (state === null) return null;
      // Only a source of the same answer can be picked from the panel's own list.
      return state.all.some((s) => s.chunk_id === action.source.chunk_id)
        ? { ...state, source: action.source }
        : state;
    case "close":
      return null;
  }
}

export type DismissKind = "escape" | "close-button" | "outside";

/**
 * What closes the panel. Escape and the close button do. A click, scroll or focus change
 * outside it does not: the chat has to stay usable while the panel is open (and a click on
 * another chip should switch the panel, not close and reopen it).
 */
export function shouldDismiss(kind: DismissKind): boolean {
  return kind !== "outside";
}
