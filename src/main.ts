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

// Mirrors the Rust `RankedMemory` (memory.rs) + nested `MemoryEntry`. The
// backend serializes via serde, so field names are snake_case and the role
// comes through as the serde-lowercased enum string ("user"/"assistant"/...).
interface MemoryEntry {
  id: number;
  text_content: string;
  timestamp: number;
  role: string;
  chunk_index: number;
  salience: number;
  metadata_json: string | null;
}
interface RankedMemory {
  entry: MemoryEntry;
  score: number;
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

async function runMemoryQuery(): Promise<void> {
  const query = memoryInput.value.trim();
  if (!query) return;

  memoryResults.innerHTML = "";
  const placeholder = document.createElement("div");
  placeholder.className = "memory-result-empty";
  placeholder.textContent = "searching…";
  memoryResults.appendChild(placeholder);

  try {
    // topK maps to the Rust `top_k: Option<usize>` param via Tauri's
    // camelCase→snake_case serde convention.
    const hits = await invoke<RankedMemory[]>("debug_memory_query", {
      query,
      topK: 10,
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

    const score = document.createElement("span");
    score.className = "memory-result-score";
    // Score scale is ~1/61..2/61; show raw to 4dp for diagnostic precision.
    score.textContent = `rrf ${hit.score.toFixed(4)}`;

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
