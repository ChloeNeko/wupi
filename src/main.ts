import "./style.css";
import { invoke, Channel } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

// The events the backend actually emits today (chat_send in lib.rs):
//   chunk  — streamed reply text, token-by-token
//   done   — generation complete; carries final_text + the parsed thought
//            channel (`reasoning`), which the UI renders as a collapsible
//            "thinking" panel above the reply.
//   error  — generation failed; the orphaned user message is rolled back.
//
// The agent-loop events (`tool`, `tool_result`, `round_done`) are intentionally
// ABSENT here: agent.rs is a stub (bails with "not wired"), so the backend
// never emits them. Re-add these union members when the agent loop lands —
// same principle as Bug #13 (don't ship UI for features that don't exist).
type StreamEvent =
  | { type: "chunk"; text: string }
  | { type: "done"; final_text: string; reasoning?: string }
  | { type: "error"; message: string };

const chatEl = document.getElementById("chat") as HTMLDivElement;
const inputEl = document.getElementById("input") as HTMLTextAreaElement;
const sendBtn = document.getElementById("send") as HTMLButtonElement;
const stopBtn = document.getElementById("stop") as HTMLButtonElement;
const statusEl = document.getElementById("status") as HTMLSpanElement;

// Memory debug panel (pillar 4). Cached once like the chat refs above.
const memoryToggle = document.getElementById("memory-toggle") as HTMLButtonElement;
const memoryPanel = document.getElementById("memory-debug") as HTMLDivElement;
const memoryCloseBtn = document.getElementById("memory-debug-close") as HTMLButtonElement;
const memoryInput = document.getElementById("memory-debug-input") as HTMLInputElement;
const memoryRunBtn = document.getElementById("memory-debug-run") as HTMLButtonElement;
const memoryResults = document.getElementById("memory-debug-results") as HTMLDivElement;
// Optional dense cosine floor override (AGENTS.md §2M Checkpoint D). Empty
// string → null → the backend uses its compiled DENSE_COSINE_FLOOR const.
const memoryFloorInput = document.getElementById("memory-debug-floor") as HTMLInputElement;

// Schema delta debug panel (B/C runtime test). Mirrors the memory panel but
// takes a synthetic user+assistant exchange pair (a delta is computed against
// a turn pair, not a single query). The "apply" checkbox toggles dry-run vs.
// live schema mutation so the panel can both probe the model AND demonstrate
// schema evolution across chained calls.
const schemaToggle = document.getElementById("schema-toggle") as HTMLButtonElement;
const schemaPanel = document.getElementById("schema-debug") as HTMLDivElement;
const schemaCloseBtn = document.getElementById("schema-debug-close") as HTMLButtonElement;
const schemaUserInput = document.getElementById("schema-debug-user") as HTMLTextAreaElement;
const schemaAsstInput = document.getElementById("schema-debug-asst") as HTMLTextAreaElement;
const schemaApplyCheckbox = document.getElementById("schema-debug-apply") as HTMLInputElement;
const schemaRunBtn = document.getElementById("schema-debug-run") as HTMLButtonElement;
const schemaResults = document.getElementById("schema-debug-results") as HTMLDivElement;

// Mirrors the Rust `RankedMemory` (memory.rs) + nested `MemoryEntry` +
// `DebugScores`. The backend serializes via serde, so field names are
// snake_case and the role comes through as the serde-lowercased enum string
// ("user"/"assistant"/...). The `debug` block carries the raw dense cosine
// (the calibration readout for DENSE_COSINE_FLOOR) and per-list ranks.
interface MemoryEntry {
  id: number;
  text_content: string;
  timestamp: number;
  role: string;
  chunk_index: number;
  salience: number;
  metadata_json: string | null;
  card_id: string;
  session_id: string | null;
}
interface DebugScores {
  dense_cosine?: number | null;
  dense_rank?: number | null;
  sparse_rank?: number | null;
}
interface RankedMemory {
  entry: MemoryEntry;
  score: number;
  debug?: DebugScores;
}

let sending = false;

function addMessage(role: "user" | "wupi" | "system", text: string): HTMLDivElement {
  const el = document.createElement("div");
  el.className = `msg ${role}`;

  if (role !== "system") {
    const roleLabel = document.createElement("div");
    roleLabel.className = "role";
    roleLabel.textContent = role === "user" ? "You" : "Wupi";
    el.appendChild(roleLabel);
  }

  const body = document.createElement("div");
  body.className = "body";
  body.textContent = text;
  el.appendChild(body);

  chatEl.appendChild(el);
  scrollToBottom();
  return el;
}

/**
 * Render the model's parsed thought channel (`reasoning`) as a collapsible
 * panel ABOVE the reply body inside a Wupi bubble. Uses native <details> so
 * the toggle is free (no JS, keyboard-accessible, respects prefers-reduced-motion).
 *
 * The backend holds thought content until the `done` event (ThoughtGate
 * buffers it during generation — see chat_format.rs), so this is called once
 * per completed turn, not streamed. If `reasoning` is empty (direct reply,
 * no thought channel) we render nothing — no empty clutter.
 */
function attachReasoning(bubble: HTMLDivElement, reasoning: string): void {
  if (!reasoning.trim()) return;

  const body = bubble.querySelector(".body") as HTMLDivElement | null;
  if (!body) return;

  const details = document.createElement("details");
  details.className = "reasoning";

  const summary = document.createElement("summary");
  summary.textContent = "thinking";
  details.appendChild(summary);

  const thought = document.createElement("div");
  thought.className = "reasoning-body";
  thought.textContent = reasoning;
  details.appendChild(thought);

  // Insert ABOVE the reply body (thought channel precedes reply in the
  // Gemma 4 protocol, and reading-order should match).
  bubble.insertBefore(details, body);
}

let streamingBubble: HTMLDivElement | null = null;
let streamingBody: HTMLDivElement | null = null;

function beginStreaming(): void {
  streamingBubble = addMessage("wupi", "");
  streamingBody = streamingBubble.querySelector(".body") as HTMLDivElement;
  streamingBubble.classList.add("streaming");
}

function appendStreamText(text: string): void {
  if (!streamingBody) return;
  streamingBody.textContent += text;
  scrollToBottom();
}

function finalizeStream(text: string): void {
  if (streamingBubble) streamingBubble.classList.remove("streaming");
  if (streamingBody) streamingBody.textContent = text;
  streamingBubble = null;
  streamingBody = null;
  scrollToBottom();
}

function scrollToBottom(): void {
  chatEl.scrollTop = chatEl.scrollHeight;
}

function setStatus(state: "ready" | "busy" | "error" | string, text: string): void {
  statusEl.textContent = text;
  statusEl.className = `status ${state}`;
}

async function send(): Promise<void> {
  const text = inputEl.value.trim();
  if (!text || sending) return;

  sending = true;
  setSending(true);
  addMessage("user", text);
  inputEl.value = "";
  autoGrow();

  beginStreaming();
  setStatus("busy", "Wupi is thinking…");

  const channel = new Channel<StreamEvent>();
  channel.onmessage = (event) => handleStreamEvent(event);

  try {
    await invoke("chat_send", { text, onEvent: channel });
  } catch (err) {
    handleStreamEvent({
      type: "error",
      message: String(err?.message ?? err),
    });
  }
}

function handleStreamEvent(event: StreamEvent): void {
  switch (event.type) {
    case "chunk":
      appendStreamText(event.text);
      break;
    case "done":
      // Attach the parsed thought channel (if any) BEFORE finalizing, so the
      // panel is in place when the streaming class is removed.
      if (event.reasoning && streamingBubble) {
        attachReasoning(streamingBubble, event.reasoning);
      }
      finalizeStream(event.final_text);
      setStatus("ready", "ready");
      setSending(false);
      sending = false;
      break;
    case "error":
      finalizeStream(`⚠️ ${event.message}`);
      setStatus("error", "error");
      setSending(false);
      sending = false;
      break;
  }
}

async function stop(): Promise<void> {
  try {
    await invoke("chat_stop");
  } catch {
  }
}

function setSending(inFlight: boolean): void {
  sendBtn.classList.toggle("hidden", inFlight);
  stopBtn.classList.toggle("hidden", !inFlight);
  inputEl.disabled = inFlight;
  inputEl.setAttribute("aria-busy", String(inFlight));
}

function autoGrow(): void {
  inputEl.style.height = "auto";
  inputEl.style.height = `${Math.min(inputEl.scrollHeight, 120)}px`;
}

sendBtn.addEventListener("click", send);
stopBtn.addEventListener("click", stop);
inputEl.addEventListener("input", autoGrow);
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});

// ── Schema delta debug panel (B/C runtime test) ─────────────
// Toggle via 🌍, supply a synthetic user+assistant exchange, run the delta
// pass. Shows the raw model output + parsed delta + resulting schema. With
// "apply delta" checked, the delta is merged into AppState.schema so chained
// calls demonstrate schema evolution across turns.
schemaToggle.addEventListener("click", () => {
  const willOpen = schemaPanel.classList.contains("hidden");
  schemaPanel.classList.toggle("hidden", !willOpen);
  if (willOpen) schemaUserInput.focus();
});

schemaCloseBtn.addEventListener("click", () => {
  schemaPanel.classList.add("hidden");
});

schemaRunBtn.addEventListener("click", runSchemaDelta);

async function runSchemaDelta(): Promise<void> {
  const userExchange = schemaUserInput.value.trim();
  const assistantExchange = schemaAsstInput.value.trim();
  if (!userExchange || !assistantExchange) return;

  schemaResults.innerHTML = "";
  const placeholder = document.createElement("div");
  placeholder.className = "memory-result-empty";
  placeholder.textContent = "generating delta…";
  schemaResults.appendChild(placeholder);

  try {
    const result = await invoke<SchemaDeltaResult>("debug_schema_delta", {
      userExchange,
      assistantExchange,
      apply: schemaApplyCheckbox.checked,
    });
    renderSchemaResult(result);
  } catch (err) {
    schemaResults.innerHTML = "";
    const errEl = document.createElement("div");
    errEl.className = "memory-result-error";
    errEl.textContent = `⚠️ ${String(err?.message ?? err)}`;
    schemaResults.appendChild(errEl);
  }
}

// Mirrors the JSON returned by debug_schema_delta (lib.rs).
interface SchemaDeltaResult {
  raw_output: string;
  delta: {
    summary?: string | null;
    recent_events?: string[] | null;
    entities?: Record<string, string | null> | null;
  } | null;
  error: string;
  schema_after: string;
}

function renderSchemaResult(result: SchemaDeltaResult): void {
  schemaResults.innerHTML = "";

  // Error banner (parse failure / generation error). Even on error we show
  // the raw output below so the malformed emission is visible.
  if (result.error) {
    const errEl = document.createElement("div");
    errEl.className = "memory-result-error";
    errEl.textContent = `⚠️ ${result.error}`;
    schemaResults.appendChild(errEl);
  }

  // Raw model output — preformatted so the exact bytes are visible. This is
  // the diagnostic heart of the panel: if the model wraps JSON in fences,
  // rambles, or emits garbage, it shows here.
  if (result.raw_output) {
    const rawBlock = document.createElement("div");
    rawBlock.className = "schema-delta-block";
    const rawLabel = document.createElement("div");
    rawLabel.className = "schema-delta-label";
    rawLabel.textContent = "raw model output";
    rawBlock.appendChild(rawLabel);
    const raw = document.createElement("pre");
    raw.className = "schema-raw";
    raw.textContent = result.raw_output;
    rawBlock.appendChild(raw);
    schemaResults.appendChild(rawBlock);
  }

  // Parsed delta (as JSON). null when parsing failed.
  const deltaBlock = document.createElement("div");
  deltaBlock.className = "schema-delta-block";
  const deltaLabel = document.createElement("div");
  deltaLabel.className = "schema-delta-label";
  deltaLabel.textContent = result.delta ? "parsed delta" : "parsed delta (none)";
  deltaBlock.appendChild(deltaLabel);
  const delta = document.createElement("pre");
  delta.className = "schema-raw";
  delta.textContent = result.delta ? JSON.stringify(result.delta, null, 2) : "(parse failed)";
  deltaBlock.appendChild(delta);
  schemaResults.appendChild(deltaBlock);

  // Resulting schema state (the full WorldSchema after optional apply).
  const schemaBlock = document.createElement("div");
  schemaBlock.className = "schema-delta-block";
  const schemaLabel = document.createElement("div");
  schemaLabel.className = "schema-delta-label";
  schemaLabel.textContent = "schema after";
  schemaBlock.appendChild(schemaLabel);
  const schema = document.createElement("pre");
  schema.className = "schema-raw";
  schema.textContent = result.schema_after;
  schemaBlock.appendChild(schema);
  schemaResults.appendChild(schemaBlock);
}

// ── Memory debug panel (pillar 4) ───────────────────────────
// Toggle via 🧠, query via the input + search button. Results render as
// ranked rows showing the fused RRF score + role + text. The panel is the
// tuning surface for the hybrid engine — fire queries independently of
// generation to see what retrieval actually returns.
memoryToggle.addEventListener("click", () => {
  const willOpen = memoryPanel.classList.contains("hidden");
  memoryPanel.classList.toggle("hidden", !willOpen);
  if (willOpen) memoryInput.focus();
});

memoryCloseBtn.addEventListener("click", () => {
  memoryPanel.classList.add("hidden");
});

memoryRunBtn.addEventListener("click", runMemoryQuery);
memoryInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    runMemoryQuery();
  }
});
memoryFloorInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    runMemoryQuery();
  }
});

async function runMemoryQuery(): Promise<void> {
  const query = memoryInput.value.trim();
  if (!query) return;

  // Parse the optional floor override. Empty/non-numeric → null → backend
  // uses its compiled default. Tauri's serde bridge maps JS null → Rust None.
  const floorRaw = memoryFloorInput.value.trim();
  const floorParsed = floorRaw === "" ? null : Number.parseFloat(floorRaw);
  const denseFloor = floorParsed !== null && Number.isFinite(floorParsed) ? floorParsed : null;

  memoryResults.innerHTML = "";
  const placeholder = document.createElement("div");
  placeholder.className = "memory-result-empty";
  placeholder.textContent = "searching…";
  memoryResults.appendChild(placeholder);

  try {
    // topK + denseFloor map to the Rust `top_k: Option<usize>` and
    // `dense_floor: Option<f32>` params via Tauri's camelCase→snake_case
    // serde convention.
    const hits = await invoke<RankedMemory[]>("debug_memory_query", {
      query,
      topK: 10,
      denseFloor,
    });
    renderMemoryResults(hits);
  } catch (err) {
    memoryResults.innerHTML = "";
    const errEl = document.createElement("div");
    errEl.className = "memory-result-error";
    errEl.textContent = `⚠️ ${String(err?.message ?? err)}`;
    memoryResults.appendChild(errEl);
  }
}

function renderMemoryResults(hits: RankedMemory[]): void {
  memoryResults.innerHTML = "";

  if (hits.length === 0) {
    const empty = document.createElement("div");
    empty.className = "memory-result-empty";
    empty.textContent = "no memories matched.";
    memoryResults.appendChild(empty);
    return;
  }

  for (const hit of hits) {
    const row = document.createElement("div");
    row.className = "memory-result";

    const head = document.createElement("div");
    head.className = "memory-result-head";

    const role = document.createElement("span");
    role.className = "memory-result-role";
    role.textContent = hit.entry.role;

    // Score row: fused RRF + raw dense cosine + per-list ranks. The cosine
    // is the calibration readout — watch it on borderline hits to decide
    // whether DENSE_COSINE_FLOOR should move.
    const score = document.createElement("span");
    score.className = "memory-result-score";
    const parts: string[] = [`rrf ${hit.score.toFixed(4)}`];
    const dbg = hit.debug ?? {};
    if (dbg.dense_cosine != null) {
      parts.push(`cos ${dbg.dense_cosine.toFixed(3)}`);
    }
    if (dbg.dense_rank != null || dbg.sparse_rank != null) {
      const ranks: string[] = [];
      if (dbg.sparse_rank != null) ranks.push(`s${dbg.sparse_rank}`);
      if (dbg.dense_rank != null) ranks.push(`d${dbg.dense_rank}`);
      parts.push(ranks.join(" "));
    }
    score.textContent = parts.join(" · ");

    head.appendChild(role);
    head.appendChild(score);
    row.appendChild(head);

    const text = document.createElement("div");
    text.className = "memory-result-text";
    text.textContent = hit.entry.text_content;
    row.appendChild(text);

    memoryResults.appendChild(row);
  }
}

async function boot(): Promise<void> {
  try {
    const ready: string = await invoke("app_ready");
    setStatus("ready", ready);
    inputEl.focus();
  } catch (err) {
    setStatus("error", `init failed: ${String(err)}`);
  }

  await listen<{ status: string; model?: string; message?: string }>(
    "model-status",
    (event) => {
      const p = event.payload;
      if (p.status === "ready") {
        setStatus("ready", `ready · ${p.model ?? "model"} loaded`);
      } else if (p.status === "error") {
        setStatus("error", `model error: ${p.message ?? "unknown"}`);
      } else if (p.status === "no_model") {
        setStatus("ready", "ready · no model (echo mode)");
      }
    },
  );
}

boot();
