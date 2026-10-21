# 0006: Chat pipeline with citations

## Context
Retrieval (0005) finds the right chunks. Chat has to turn them into an answer in which
every claim can be checked against a paper, and stream it as it is written. Most papers
are indexed by their abstract only (full text exists for a few), so an answer must say
when it rests only on abstracts. A question the index cannot answer must get a refusal,
not a guess. Each stage needs its own switch so the evaluation can measure what it adds,
and every request's cost goes to `llm_usage`. Third-party text (paper chunks, earlier
answers built from them) is untrusted input.

## Decision

**Six stages**, each a typed function in `app/rag/` with a setting and unit tests:
- **condense** (`condense.py`, `condense_enabled`): one call rewrites a follow-up into a
  standalone question from the last `chat_history_messages` (6) messages. No call on a
  first message.
- **plan** (`plan.py`, `plan_enabled`): one Structured Outputs call returns the intent
  (`qa`, `explain`, `compare`, `find_papers`), two rephrasings and filters. The model's
  filters are checked in code, because a wrong filter hides every right answer. An arXiv
  id is kept only if it is written in the question; a category only if the index holds it;
  a date only if it parses and is not in the future. Removed values are recorded.
- **search** (`pipeline.py`): one `search()` (0005) for the question and one for each
  rephrasing (`plan_rephrasings_enabled`). The lists are merged with Reciprocal Rank
  Fusion, with the same k as inside Qdrant and positions from 0. The merged pool is then
  cut to `search_candidates` before rerank, so rephrasings never lengthen the rerank.
- **rerank**: as in 0005, off by default (`rerank_enabled`). Then at most 3 chunks per
  paper and the top `chat_sources` (8).
- **generate** (`generate.py`): the sources are numbered [1]..[8]. Each one carries its
  title, arXiv id, section and evidence level (`chunk_type`). The prompt asks for a
  citation after every factual sentence and is chosen by intent: `explain` has its own
  prompt and output limit; qa, compare and find_papers share `generate_v1`. The answer
  is streamed.
- **verify** (`verify.py`, `verify_enabled`):
  - Code checks always run: every `[n]` must exist, and uncited sentences are listed.
  - One Structured Outputs call then judges each cited sentence against its cited
    sources. Unsupported sentences are flagged, never removed.

**Prompts** are files in `app/rag/prompts/`, named `<name>_v<N>.txt`. A change means a
new file and a new version. The versions a turn used go into the trace and the message
meta.
- Each prompt is the system message and holds stable text only, so OpenAI's prompt
  caching can reuse it.
- Sources, the question and the conversation go in the user message, inside delimited
  blocks (`<source n="1">`, `<question>`, ...). Every prompt says to ignore instructions
  found inside blocks.
- Our tag names are neutralised inside third-party text (`<source` becomes `&lt;source`),
  so a chunk cannot close its block early.

**The abstract rule.** It has two parts:
- If every source given to generate is an `abstract` chunk, the user message carries a
  fixed note. The prompts (`generate_v1` and `generate_explain_v1`) say that an answer
  whose cited sources are all abstracts must say so and claim no detail beyond them.
- After generation, the code computes `abstract_only` from the CITED sources. It is
  stored in the meta and sent in the `done` event.

Abstract-only sources are not a reason to refuse.

**Refusal (the fallback for unanswerable questions).**
- No chunks after filters: a fixed refusal text is returned and generate is never called.
- Sources that do not answer: the prompt tells the model to reply exactly
  `INSUFFICIENT_SOURCES`. While streaming, the first characters are held back as long as
  they could still be that marker, so the marker never reaches the user. A full match
  becomes the same refusal.

A refusal is a finished turn and is saved, with `meta.refusal` set and no citations.

**Stage failures.**
- Condense, plan and verify fall back: to the raw message, to the default plan (qa, no
  filters, no rephrasings), and to "unavailable" (the code checks are kept). The error is
  recorded on the stage's span and in the meta.
- Search, rerank and generate cannot fall back: the stream ends with an `error` event.
  Rerank is included because a configured reranker that is missing is a setup problem,
  as in `GET /search`.

**Streaming.** `POST /chat` returns server-sent events: `meta`, `sources`, `token`...,
`refusal`, `verification`, then `done` or `error`. FastAPI's own `EventSourceResponse`
is used, so no dependency was added.
- Checks happen before the stream starts and return normal HTTP errors: 422 for a blank
  message, 404 for an unknown conversation, 503 when the OpenAI key is missing.
- The pipeline is synchronous and is iterated in a worker thread.
- On a disconnect the generator is closed. Its spans are ended, and the rest of the
  OpenAI stream is read so that its final usage chunk can still be logged.

**Retries.** Chat calls use our `call_with_retry` (429, 5xx, network errors). A stream is
retried only while it is being opened, because a retry after tokens were shown would
repeat the answer.

**Saving.** A turn is saved in one transaction at its end: the conversation if it is new,
the user message and the assistant message. There is never half a turn: a failed turn
saves nothing, and a retry does not leave a dangling question. Spend is separate:
- each request writes its own committed `llm_usage` row (`chat-condense`, `chat-plan`,
  `chat-generate`, `chat-verify`, `embed-query`), so a failed turn's spend is kept;
- the assistant message gets the turn's totals: tokens of the chat calls, and the cost of
  the chat calls plus the question embeddings;
- a stream that breaks before its final usage chunk gets an ESTIMATED row instead, under
  its own feature name (`chat-generate-estimated`). The count uses the local `cl100k_base`
  tokenizer, which is only an approximation for gpt-4o-mini (that model uses
  `o200k_base`, which is not downloaded for this). It counts the prompt plus a few tokens
  per message, and only the text received, so the output count is a lower bound.

**`messages.meta` (migration 0003).** One nullable jsonb column on `messages`. It holds
the intent, the checked plan, the verification, the skipped stages, the errors fallen
back from, the prompt versions, the standalone question and `abstract_only`.
- Why one jsonb column and not typed columns: this shape will change while the chat
  features are built (verification output, filter chips from the plan, comparisons), and
  a jsonb column avoids a migration for every change.
- The price is that type checking moves from the database to the application. A pydantic
  model (`app/rag/meta.py`) checks every write and refuses unknown fields. It carries a
  `version`, so rows written under an older shape stay readable.
- A field that becomes stable and is queried often can move to its own column later.

**Tracing.** The pipeline uses a small `Tracer` interface (`app/observability/tracing.py`)
rather than a tracing SDK.
- Every turn has one trace with exactly six spans. A stage that did not run still has its
  span, ended as skipped with the reason (for example rerank: disabled), so every trace
  has the same shape.
- Spans are started and ended explicitly, because the streamed answer pauses between
  pieces.
- Langfuse (`app/observability/langfuse_tracer.py`, SDK 4.15.6, OpenTelemetry-based) is
  used only when both Langfuse keys are set; otherwise, in tests, and when the SDK cannot
  start, the tracer is a no-op. Tests use a recording fake, a fake SDK client, and the
  real SDK with an in-memory exporter.
- A turn sends exactly the six stage spans, in one trace and with no root span. Each span
  is started in a trace id made by the adapter, which the SDK marks as a top-level
  observation. The trace name ("chat-turn") and the session (the conversation id) are
  attached to each span. The turn's summary (outcome, prompt versions, skipped stages,
  errors) goes on the last span, `verify`. One flush per turn sends them to
  `{LANGFUSE_HOST}/api/public/otel/v1/traces`.
- The SDK gets a private OpenTelemetry tracer provider (it would otherwise install itself
  as the process-wide one and could receive other libraries' spans), and it exports only
  its own spans. Keys, host, timeout, debug and sample rate are passed as parameters, so
  environment variables such as `LANGFUSE_BASE_URL` cannot redirect the export.
- Span data is sanitised first. Key-like strings and bearer tokens are redacted, and
  base64 `data:` URIs are dropped, because the SDK would upload those as media. Long
  strings are cut. The same redaction is applied to error texts stored in
  `messages.meta`. The SDK itself attaches the PUBLIC key to its instrumentation scope;
  that is the same key it already sends to the same host in its auth header. The secret
  key never leaves the auth header.
- A failure inside the SDK is logged and dropped: tracing never fails a chat turn.

## Departures from the original design
1. **No author filter.** The planner returns no author filter, because chunk payloads have
   no author field. This replaces 0005's note that author filters would come with the
   planner. Author names stay in the rephrasings, where BM25 can still match them.
2. **No running summary.** Condense reads only the last N messages (`chat_history_messages`);
   the original design also kept a running summary for long chats.
3. **compare and find_papers use the qa prompt.** Their own outputs (a comparison table,
   ranked paper cards) come later. The intent is already stored and traced.
4. **One `messages.meta` jsonb column** (see above), not part of the original data model.
5. **Six top-level spans per trace, no root span.** The pipeline stages are the trace's
   observations; there is no extra "chat-turn" span wrapping them.
6. **Reranking is re-measured for speed and memory only** (fewer candidates, an ONNX
   thread limit, possibly a smaller model). Rerank quality is compared at the full
   evaluation milestone, on non-golden questions. Rerank stays off by default, and the
   golden set is never tuned against.

## Consequences
- A turn costs up to four chat requests (condense only with history, verify only when an
  answer cites something) and one question embedding per search query (up to three in
  hybrid mode). Each is visible in `llm_usage`.
- Verification roughly doubles the input tokens of a turn. It can be switched off.
- Holding back the start of the answer delays the first visible token until the text
  can no longer be the marker (at most its 20 characters).
- The simple sentence splitter can split at abbreviations such as "e.g.". Verify then
  judges two halves, which can flag more but never hides a claim.
- Ties are resolved inside Qdrant, in no guaranteed order. Two chunks with the same
  fused RRF score (for example dense #1 and BM25 #2 against dense #2 and BM25 #1) can
  swap between identical runs, and a tie at a list's cutoff decides which chunk makes the
  limit. The chat's own merge of the question and rephrasing lists breaks its ties
  deterministically (first appearance), but it inherits the order Qdrant returned. The
  retrieval evaluation is therefore not exactly repeatable for questions with tied top
  chunks. A deterministic tie-break in the search stage is planned as a separate change,
  with the rule chosen in advance and a before/after retrieval evaluation. (Done: equal
  scores are now ordered by chunk id, also at the cut; docs/adr/0012.)
- `/chat` needs migration 0003 applied (`make migrate`).
- Changing a prompt means a new file and version. Comparisons across versions should use
  the version recorded on each message.
