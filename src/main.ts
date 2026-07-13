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
