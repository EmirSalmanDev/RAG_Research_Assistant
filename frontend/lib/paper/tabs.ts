// The related-papers tabs: their order and the keyboard behaviour of the tab list
// (WAI-ARIA tabs pattern: arrow keys move between tabs, Home and End jump). Plain TypeScript.

export const RELATED_TABS = ["similar", "cites", "cited_by"] as const;
export type RelatedTab = (typeof RELATED_TABS)[number];

export const RELATED_TAB_LABELS: Record<RelatedTab, string> = {
  similar: "Similar",
  cites: "Cites",
  cited_by: "Cited by",
};

/** The tab a key press moves to, or null when the key is not a tab-list key. */
export function tabAfterKey(
  current: RelatedTab,
  key: string,
): RelatedTab | null {
  const i = RELATED_TABS.indexOf(current);
  const n = RELATED_TABS.length;
  switch (key) {
    case "ArrowRight":
      return RELATED_TABS[(i + 1) % n]!;
    case "ArrowLeft":
      return RELATED_TABS[(i - 1 + n) % n]!;
    case "Home":
      return RELATED_TABS[0];
    case "End":
      return RELATED_TABS[n - 1]!;
    default:
      return null;
  }
}
