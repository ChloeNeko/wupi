import "./style.css";
import { invoke, Channel } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

type StreamEvent =
  | { type: "chunk"; text: string }
  | { type: "round_done"; text: string; had_tool_calls: boolean }
  | { type: "tool"; name: string; input_summary: string }
  | { type: "tool_result"; name: string; ok: boolean; summary: string }
  | { type: "done"; final_text: string }
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

function flashSystem(text: string): void {
  const el = addMessage("system", text);
  setTimeout(() => {
    el.style.transition = "opacity 0.6s";
    el.style.opacity = "0";
    setTimeout(() => el.remove(), 700);
  }, 3500);
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
    case "tool":
      flashSystem(`🔧 ${event.name}(${event.input_summary})`);
      setStatus("busy", `calling ${event.name}…`);
      break;
    case "tool_result":
      flashSystem(
        `${event.ok ? "✓" : "✗"} ${event.name}: ${event.summary}`,
      );
      setStatus("busy", "Wupi is thinking…");
      break;
    case "round_done":
      if (!event.had_tool_calls) {
        finalizeStream(event.text);
      }
      break;
    case "done":
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
