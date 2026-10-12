// The chat page's state as a pure reducer: a conversation is a list of messages, and the
// stream's events are applied one by one to the assistant message being written.
// Keeping this free of React makes it testable on its own (node --test).

import type {
  ChatEvent,
  DoneEvent,
  MetaEvent,
  RefusalEvent,
  Source,
  Verification,
} from "./events.ts";

export interface UserMessage {
  role: "user";
  id: string;
  text: string;
}

export type AssistantStatus =
  "streaming" | "done" | "refused" | "error" | "stopped";

export interface AssistantMessage {
  role: "assistant";
  id: string;
  status: AssistantStatus;
  /** The answer so far (empty for a refusal; see `refusal`). */
  text: string;
  sources: Source[];
  meta: MetaEvent | null;
  verification: Verification | null;
  refusal: RefusalEvent | null;
  done: DoneEvent | null;
  error: string | null;
}

export type ChatMessage = UserMessage | AssistantMessage;

export interface ChatState {
  /** Sent with the next question so the backend continues the same conversation. */
  conversationId: string | null;
  messages: ChatMessage[];
}

export type ChatAction =
  | { type: "ask"; text: string; userId: string; assistantId: string }
  | { type: "event"; event: ChatEvent }
  | { type: "failed"; message: string } // no usable stream (HTTP error, network, bad data)
  | { type: "stopped" } // the reader pressed Stop
  | { type: "reset" };

export const initialChatState: ChatState = {
  conversationId: null,
  messages: [],
};

export function isBusy(state: ChatState): boolean {
  const last = state.messages[state.messages.length - 1];
  return last?.role === "assistant" && last.status === "streaming";
}

function newAssistant(id: string): AssistantMessage {
  return {
    role: "assistant",
    id,
    status: "streaming",
    text: "",
    sources: [],
    meta: null,
    verification: null,
    refusal: null,
    done: null,
    error: null,
  };
}

/** Apply `change` to the last message if it is a streaming assistant message. */
function updateStreaming(
  state: ChatState,
  change: (m: AssistantMessage) => AssistantMessage,
): ChatState {
  const last = state.messages[state.messages.length - 1];
  if (last?.role !== "assistant" || last.status !== "streaming") return state;
  return { ...state, messages: [...state.messages.slice(0, -1), change(last)] };
}

export function chatReducer(state: ChatState, action: ChatAction): ChatState {
  switch (action.type) {
    case "reset":
      return initialChatState;

    case "ask":
      if (isBusy(state)) return state; // one question at a time
      return {
        ...state,
        messages: [
          ...state.messages,
          { role: "user", id: action.userId, text: action.text },
          newAssistant(action.assistantId),
        ],
      };

    case "failed":
      return updateStreaming(state, (m) => ({
        ...m,
        status: "error",
        error: action.message,
      }));

    case "stopped":
      return updateStreaming(state, (m) => ({ ...m, status: "stopped" }));

    case "event": {
      const event = action.event;
      switch (event.event) {
        case "meta": {
          const next = updateStreaming(state, (m) => ({
            ...m,
            meta: event.data,
          }));
          return { ...next, conversationId: event.data.conversation_id };
        }
        case "sources":
          return updateStreaming(state, (m) => ({
            ...m,
            sources: event.data.sources,
          }));
        case "token":
          return updateStreaming(state, (m) => ({
            ...m,
            text: m.text + event.data.text,
          }));
        case "refusal":
          return updateStreaming(state, (m) => ({ ...m, refusal: event.data }));
        case "verification":
          return updateStreaming(state, (m) => ({
            ...m,
            verification: event.data,
          }));
        case "done": {
          const next = updateStreaming(state, (m) => ({
            ...m,
            done: event.data,
            status: event.data.refusal ? "refused" : "done",
          }));
          return { ...next, conversationId: event.data.conversation_id };
        }
        case "error":
          return updateStreaming(state, (m) => ({
            ...m,
            status: "error",
            error: event.data.message,
          }));
      }
    }
  }
}
