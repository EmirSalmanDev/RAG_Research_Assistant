"use client";

import { useEffect, useReducer, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { ChatProtocolError } from "@/lib/chat/events";
import { panelReducer } from "@/lib/chat/panel";
import { recordEvent, startTimeline } from "@/lib/chat/debug";
import { chatReducer, initialChatState, isBusy } from "@/lib/chat/reducer";
import { ChatHttpError, streamChat } from "@/lib/chat/stream";

import { MessageView } from "./message-view";
import { SourcePanel } from "./source-panel";

const MAX_CHARS = 2000; // the API refuses longer messages (ChatRequest.message)

// Starting points that fill the box; nothing is sent until the reader presses Ask.
const EXAMPLES = [
  "How does LoRA reduce the number of trained parameters?",
  "When does retrieval-augmented generation help, and when does it not?",
  "Which papers compare dense and sparse retrieval?",
];

function describeError(err: unknown): string {
  if (err instanceof ChatHttpError) return err.message;
  if (err instanceof ChatProtocolError) return err.message;
  if (err instanceof TypeError)
    return "Could not reach the API. Check that it is running.";
  return err instanceof Error ? err.message : "Something went wrong.";
}

export function ChatView() {
  const [state, dispatch] = useReducer(chatReducer, initialChatState);
  const [draft, setDraft] = useState("");
  const [panel, dispatchPanel] = useReducer(panelReducer, null);
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const stickRef = useRef(true);
  const busy = isBusy(state);

  // Follow the answer as it grows, unless the reader has scrolled up to read.
  useEffect(() => {
    const onScroll = () => {
      const gap =
        document.documentElement.scrollHeight -
        window.innerHeight -
        window.scrollY;
      stickRef.current = gap < 120;
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);
  useEffect(() => {
    if (stickRef.current) bottomRef.current?.scrollIntoView({ block: "end" });
  }, [state.messages]);

  // A page left open must not leave a stream running.
  useEffect(() => () => abortRef.current?.abort(), []);

  async function send(text: string) {
    const message = text.trim();
    if (!message || busy) return;
    setDraft("");
    stickRef.current = true;
    const controller = new AbortController();
    abortRef.current = controller;
    dispatch({
      type: "ask",
      text: message,
      userId: crypto.randomUUID(),
      assistantId: crypto.randomUUID(),
    });
    startTimeline();
    try {
      for await (const { event, receivedAt } of streamChat(
        { message, conversation_id: state.conversationId },
        { signal: controller.signal },
      )) {
        recordEvent(
          event.event,
          receivedAt,
          event.event === "token" ? event.data.text.length : 0,
        );
        dispatch({ type: "event", event });
      }
    } catch (err) {
      if (controller.signal.aborted) dispatch({ type: "stopped" });
      else dispatch({ type: "failed", message: describeError(err) });
    } finally {
      if (abortRef.current === controller) abortRef.current = null;
    }
  }

  function newChat() {
    abortRef.current?.abort();
    dispatchPanel({ type: "close" });
    dispatch({ type: "reset" });
  }

  const empty = state.messages.length === 0;

  return (
    <div
      className={cn(
        "flex w-full flex-1 flex-col transition-[padding] duration-200 motion-reduce:transition-none",
        // the panel is 32rem wide and does not dim or block the chat; on wide screens the
        // chat column moves over to sit beside it, on narrow ones the panel overlays it
        panel && "lg:pr-[32rem]",
      )}
    >
      <div className="mx-auto flex w-full max-w-2xl flex-1 flex-col px-4">
        <div className="flex-1 py-8">
          {empty ? (
            <div className="space-y-6">
              <div className="space-y-2">
                <h1 className="text-2xl font-semibold tracking-tight">
                  Ask the papers
                </h1>
                <p className="text-muted-foreground">
                  Answers cite the indexed papers. Open a citation to read the
                  passage it comes from.
                </p>
              </div>
              <ul className="space-y-2">
                {EXAMPLES.map((example) => (
                  <li key={example}>
                    <button
                      type="button"
                      onClick={() => setDraft(example)}
                      className="w-full rounded-md border px-4 py-3 text-left transition-colors hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                    >
                      {example}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          ) : (
            <div className="space-y-10">
              {state.messages.map((message, i) => (
                <div
                  key={message.id}
                  className={
                    message.role === "user" && i > 0 ? "border-t pt-10" : ""
                  }
                >
                  <MessageView
                    message={message}
                    activeChunkId={panel?.source.chunk_id}
                    onOpen={(source, all) =>
                      dispatchPanel({ type: "open", source, all })
                    }
                  />
                </div>
              ))}
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        <form
          className="sticky bottom-0 -mx-4 space-y-2 border-t bg-background/95 px-4 pb-4 pt-3 backdrop-blur"
          onSubmit={(e) => {
            e.preventDefault();
            void send(draft);
          }}
        >
          <label htmlFor="question" className="sr-only">
            Your question
          </label>
          <textarea
            id="question"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (
                e.key === "Enter" &&
                !e.shiftKey &&
                !e.nativeEvent.isComposing
              ) {
                e.preventDefault();
                void send(draft);
              }
            }}
            maxLength={MAX_CHARS}
            rows={2}
            placeholder="Ask about a method, a result, or a set of papers"
            className="block w-full resize-none rounded-md border border-input bg-card px-3 py-2 text-base shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          />
          <div className="flex items-center justify-between gap-2">
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={newChat}
              disabled={empty}
            >
              New chat
            </Button>
            <div className="flex items-center gap-3">
              {draft.length > MAX_CHARS - 200 && (
                <span className="text-xs text-muted-foreground">
                  {draft.length} / {MAX_CHARS}
                </span>
              )}
              {busy ? (
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => abortRef.current?.abort()}
                >
                  Stop
                </Button>
              ) : (
                <Button type="submit" disabled={draft.trim() === ""}>
                  Ask
                </Button>
              )}
            </div>
          </div>
        </form>

        <SourcePanel
          source={panel?.source ?? null}
          sources={panel?.all ?? []}
          onSelect={(source) => dispatchPanel({ type: "select", source })}
          onClose={() => dispatchPanel({ type: "close" })}
        />
      </div>
    </div>
  );
}
