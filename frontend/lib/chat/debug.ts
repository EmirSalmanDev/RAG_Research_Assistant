// Optional timing log for the chat stream. Off unless localStorage "chat-debug" is "1".
// It exists to answer one question with evidence: do answer tokens reach the browser
// one by one, or all at once (a proxy that buffers would turn the stream into one blob)?
// Each entry: which event, when it arrived (ms after the request was sent), how many
// characters it carried. Read it with `window.__chatTimeline` in the console.

export interface TimelineEntry {
  event: string;
  ms: number;
  chars: number;
}

declare global {
  interface Window {
    __chatTimeline?: TimelineEntry[];
  }
}

export function startTimeline(): void {
  if (typeof window === "undefined") return;
  try {
    if (window.localStorage.getItem("chat-debug") === "1")
      window.__chatTimeline = [];
    else window.__chatTimeline = undefined;
  } catch {
    // storage blocked: no log
  }
}

export function recordEvent(event: string, ms: number, chars: number): void {
  if (typeof window === "undefined" || !window.__chatTimeline) return;
  const entry = { event, ms: Math.round(ms), chars };
  window.__chatTimeline.push(entry);
  console.debug("[chat-sse]", entry);
}
