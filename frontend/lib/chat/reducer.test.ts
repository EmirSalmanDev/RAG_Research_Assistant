import { strict as assert } from "node:assert";
import { test } from "node:test";

import { parseChatEvent } from "./events.ts";
import type { ChatEvent } from "./events.ts";
import {
  ABSTRACT_SOURCE,
  DONE,
  META,
  SECTION_SOURCE,
  VERIFICATION,
} from "./fixtures.ts";
import { chatReducer, initialChatState, isBusy } from "./reducer.ts";
import type { AssistantMessage, ChatAction, ChatState } from "./reducer.ts";

function ev(name: string, data: unknown): ChatAction {
  const event = parseChatEvent(name, JSON.stringify(data)) as ChatEvent;
  return { type: "event", event };
}

function run(
  actions: ChatAction[],
  from: ChatState = initialChatState,
): ChatState {
  return actions.reduce(chatReducer, from);
}

const ask = (text: string, n = 1): ChatAction => ({
  type: "ask",
  text,
  userId: `u${n}`,
  assistantId: `a${n}`,
});

function lastAssistant(state: ChatState): AssistantMessage {
  const last = state.messages[state.messages.length - 1];
  assert.ok(last?.role === "assistant");
  return last;
}

test("asking adds the question and an empty streaming answer", () => {
  const state = run([ask("What are adapters?")]);
  assert.deepEqual(
    state.messages.map((m) => m.role),
    ["user", "assistant"],
  );
  assert.equal(lastAssistant(state).status, "streaming");
  assert.equal(isBusy(state), true);
});

test("a full turn: tokens are appended, sources and verification stored, done finishes it", () => {
  const state = run([
    ask("What are adapters?"),
    ev("meta", META),
    ev("sources", { sources: [ABSTRACT_SOURCE, SECTION_SOURCE] }),
    ev("token", { text: "Adapters train a " }),
    ev("token", { text: "small module [1]." }),
    ev("verification", VERIFICATION),
    ev("done", DONE),
  ]);
  const answer = lastAssistant(state);
  assert.equal(answer.text, "Adapters train a small module [1].");
  assert.equal(answer.sources.length, 2);
  assert.equal(answer.status, "done");
  assert.equal(answer.done?.abstract_only, true);
  assert.equal(answer.verification?.status, "checked");
  assert.equal(state.conversationId, META.conversation_id);
  assert.equal(isBusy(state), false);
});

test("a follow-up keeps the conversation id and adds a second pair of messages", () => {
  const first = run([ask("one"), ev("meta", META), ev("done", DONE)]);
  const second = run([ask("two", 2), ev("token", { text: "x" })], first);
  assert.equal(second.messages.length, 4);
  assert.equal(second.conversationId, META.conversation_id);
  assert.equal(lastAssistant(second).text, "x");
  assert.equal((second.messages[1] as AssistantMessage).text, ""); // the first answer is untouched
});

test("a refusal is shown from the refusal event, not from tokens", () => {
  const state = run([
    ask("q"),
    ev("meta", META),
    ev("refusal", { reason: "no_sources", text: "Nothing found." }),
    ev("done", {
      ...DONE,
      refusal: "no_sources",
      citations: [],
      abstract_only: false,
    }),
  ]);
  const answer = lastAssistant(state);
  assert.equal(answer.status, "refused");
  assert.equal(answer.refusal?.text, "Nothing found.");
  assert.equal(answer.text, "");
});

test("an error event ends the turn and keeps the partial answer", () => {
  const state = run([
    ask("q"),
    ev("token", { text: "Partial " }),
    ev("error", { stage: "generate", message: "generate failed: Timeout" }),
  ]);
  const answer = lastAssistant(state);
  assert.equal(answer.status, "error");
  assert.equal(answer.error, "generate failed: Timeout");
  assert.equal(answer.text, "Partial ");
  assert.equal(isBusy(state), false);
});

test("failed (no stream) and stopped (reader pressed Stop) end a streaming answer", () => {
  const failed = run([
    ask("q"),
    { type: "failed", message: "Could not reach the API." },
  ]);
  assert.equal(lastAssistant(failed).status, "error");
  const stopped = run([
    ask("q"),
    ev("token", { text: "So far" }),
    { type: "stopped" },
  ]);
  assert.equal(lastAssistant(stopped).status, "stopped");
  assert.equal(lastAssistant(stopped).text, "So far");
});

test("events after the answer has finished are ignored", () => {
  const state = run([
    ask("q"),
    ev("done", DONE),
    ev("token", { text: "late" }),
  ]);
  assert.equal(lastAssistant(state).text, "");
});

test("a second question while one is streaming is ignored; reset clears everything", () => {
  const busy = run([ask("one"), ask("two", 2)]);
  assert.equal(busy.messages.length, 2);
  assert.deepEqual(run([ask("one"), { type: "reset" }]), initialChatState);
});
