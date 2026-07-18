//! LLM backend façade + the process-wide llama.cpp backend singleton.
//!
//! The heavy generation logic now lives in [`crate::engine`] — a dedicated
//! thread owning a persistent `LlamaContext` with Q8_0 KV cache and a
//! [`KvBuffer`] that tracks the token IDs resident in the cache so each turn
//! only prefills the **delta** since the last turn.
//!
//! This module is a thin façade: it loads the model off-thread, leaks it
//! to `&'static` (so the engine can hold a `LlamaContext<'static>`), spawns
//! the engine, and exposes a [`GenerationClient`] impl that posts requests to
//! the engine thread.
//!
//! # Why `Box::leak`
//!
//! `LlamaContext<'a>` borrows `&'a LlamaModel`. Storing model + context together
//! is self-referential and rejected by the borrow checker. Leaking the model to
//! `&'static LlamaModel` dissolves the borrow — `new_context(&'static self)`
//! yields `LlamaContext<'static>`, which the engine thread can own freely.
//!
//! This is the idiomatic choice for a **process-lifetime singleton**: the
//! model is loaded once and lives until the OS exits. The memory is never
//! reclaimed, which is exactly what we want (we don't want to unload the
//! model mid-session). If hot-swap lands later (a P-phase settings feature),
//! reclaim via `Box::from_raw(ptr)` + `drop` before loading the replacement.
//!
//! # The shared backend
//!
//! [`shared_backend`] is the single chokepoint for `LlamaBackend::init()`,
//! which the crate documents as panic-on-double-init. Both the chat loader
//! (here) and the Memory embedder ([`crate::memory_embedder_llama`]) call it
//! — the `OnceLock` makes the race safe even if both load concurrently.

use crate::chat_format::ParsedOutput;
use crate::chat_format::ModelFamily;
use crate::engine::{ChatEngine, EngineReply, EngineRequest};
use crate::session::ApiMessage;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::LlamaModel;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::sync::OnceLock;

pub type StreamFuture = Pin<Box<dyn Future<Output = anyhow::Result<ParsedOutput>> + Send>>;
pub type ChunkFn = Arc<dyn Fn(&str) + Send + Sync>;
/// Cancellation flag shared between `chat_send` and `chat_stop`. The engine's
/// decode loop checks this between tokens.
pub type CancelToken = Arc<std::sync::atomic::AtomicBool>;

pub trait GenerationClient: Send + Sync {
    fn stream(
        &self,
        messages: Vec<ApiMessage>,
        memory_block: Option<String>,
        world_state: Option<String>,
        context_size: u32,
        on_chunk: ChunkFn,
        cancel: CancelToken,
    ) -> StreamFuture;

    fn is_ready(&self) -> bool {
        false
    }
}

/// The process-wide shared llama.cpp backend, leaked to `&'static`.
///
/// `LlamaBackend::init()` must be called exactly once per process — a second
/// call returns `Err(BackendAlreadyInitialized)` (the crate guards itself with
/// an internal `AtomicBool`). The chat engine (`load_blocking`) and the Memory
/// embedder both need the backend, so this function is the single chokepoint
/// that serializes them: the first caller runs `init()` and leaks the backend,
/// every later caller reuses the same `&'static` ref. The `OnceLock` makes the
/// race safe even if both threads load concurrently at startup.
///
/// The backend is a ZST (`pub struct LlamaBackend {}`) with no raw fields, so
/// `&'static LlamaBackend` is `Send + Sync` and safe to share across the
/// `wupi-engine` and `wupi-embedder` threads. Leaking is correct for a
/// process-lifetime singleton — it lives until the OS exits, matching the
/// model leak in `LlamaModelHandle::into_static`.
static SHARED_BACKEND: OnceLock<&'static LlamaBackend> = OnceLock::new();

pub fn shared_backend() -> &'static LlamaBackend {
    SHARED_BACKEND.get_or_init(|| {
        let backend = LlamaBackend::init()
            .expect("LlamaBackend::init failed — cannot start llama.cpp");
        Box::leak(Box::new(backend))
    })
}

/// Echo fallback used when no model file is found. Unchanged from Layer 1.
pub struct EchoBackend;

impl GenerationClient for EchoBackend {
    fn stream(
        &self,
        _messages: Vec<ApiMessage>,
        _memory_block: Option<String>,
        _world_state: Option<String>,
        _context_size: u32,
        on_chunk: ChunkFn,
        _cancel: CancelToken,
    ) -> StreamFuture {
        Box::pin(async move {
            let reply = "(echo backend) Wupi's model isn't loaded yet.";
            on_chunk(reply);
            Ok(ParsedOutput {
                content: reply.to_string(),
                reasoning: String::new(),
                raw: String::new(),
            })
        })
    }
}

/// HTTP API backend — talks to an OpenAI-compatible chat completions endpoint
/// (Z.AI, NanoGPT, OpenRouter, OpenAI itself, llama.cpp/vLLM/Ollama servers).
/// Implements the same [`GenerationClient`] trait as [`LlamaCppBackend`] so
/// `chat_send` can dispatch on `ModelSource` without caring which backend is
/// active.
///
/// **Streaming:** POST `{endpoint}/chat/completions` with
/// `{model, messages, stream:true, temperature?}`, read the SSE response
/// incrementally. Each `data: {...}` line carries a `choices[0].delta.content`
/// token; forward each to `on_chunk` for live UI rendering. Honors `cancel`
/// by aborting mid-stream (the equivalent of the local engine's between-token
/// cancel check).
///
/// **Memory + world_state injection:** the local backend splices these into
/// the inter-turn region via `render_prompt`. An API only takes a flat
/// `messages` list, so we fold them into the system message (they're already
/// XML-tagged blocks — `<retrieved_memory>`, `<world_state>` — and read fine
/// as additional system context). This preserves the retrieval + schema
/// injection that makes Wupi's memory work, just routed through the system
/// role instead of a protocol splice.
///
/// **No reasoning/raw:** the OpenAI streaming format has no equivalent of the
/// Gemma4 thought channel. `ParsedOutput.reasoning` + `.raw` are left empty
/// (the post-generation archiving + schema-delta pipeline keys off `.content`
/// only, so this is safe).
pub struct HttpBackend {
    profile: crate::api::ApiProfile,
    client: reqwest::Client,
}

impl HttpBackend {
    /// Construct from a saved profile. Builds a reqwest client with a generous
    /// timeout (generation can take minutes for long replies) + the bearer
    /// token pre-attached so every request on this client is authenticated.
    pub fn new(profile: crate::api::ApiProfile) -> Self {
        let mut headers = reqwest::header::HeaderMap::new();
        if !profile.api_key.is_empty() {
            if let Ok(v) = reqwest::header::HeaderValue::from_str(&format!(
                "Bearer {}",
                profile.api_key
            )) {
                headers.insert(reqwest::header::AUTHORIZATION, v);
            }
        }
        let client = reqwest::Client::builder()
            .default_headers(headers)
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self { profile, client }
    }

    /// Resolve the full chat-completions URL from the profile's base endpoint.
    /// Accepts either a bare base (`https://nano-gpt.com/api/v1`) or one that
    /// already includes the path (`https://x/api/v1/chat/completions`). If the
    /// endpoint ends with `/`, it's trimmed first.
    fn completions_url(&self) -> String {
        let base = self.profile.endpoint.trim_end_matches('/');
        if base.ends_with("/chat/completions") {
            base.to_string()
        } else {
            format!("{base}/chat/completions")
        }
    }
}

/// A single message in the OpenAI chat request body. The local `ApiMessage`
/// has a `raw_output` field the API doesn't want — this is the slim wire view.
/// (Could `#[serde(skip)]` raw_output on ApiMessage instead, but that would
/// couple the session type to the API wire format; a local view is cleaner.)
#[derive(serde::Serialize)]
struct ChatRequestMessage {
    role: String,
    content: String,
}

/// The streaming chunk envelope: `{ choices: [ { delta: { content: "..." } } ] }.
/// `content` is `Option` because the first chunk typically carries only `role`,
/// and the final chunk carries `finish_reason` instead. Everything else is
/// ignored — we only want the delta text.
#[derive(serde::Deserialize)]
struct ChatStreamChunk {
    choices: Vec<ChatStreamChoice>,
}
#[derive(serde::Deserialize)]
struct ChatStreamChoice {
    delta: ChatStreamDelta,
}
#[derive(serde::Deserialize)]
struct ChatStreamDelta {
    #[serde(default)]
    content: Option<String>,
}

impl GenerationClient for HttpBackend {
    fn stream(
        &self,
        messages: Vec<ApiMessage>,
        memory_block: Option<String>,
        world_state: Option<String>,
        _context_size: u32,
        on_chunk: ChunkFn,
        cancel: CancelToken,
    ) -> StreamFuture {
        let url = self.completions_url();
        let model = self.profile.model.clone();
        let temperature = self.profile.temperature;
        let client = self.client.clone();
        Box::pin(async move {
            // Fold memory_block + world_state into the system message. They're
            // already XML-tagged blocks; appending them to the existing system
            // content keeps the retrieval/schema context that Wupi depends on.
            let mut wire_messages: Vec<ChatRequestMessage> = Vec::with_capacity(messages.len());
            let mut extra_ctx = String::new();
            if let Some(mb) = memory_block.as_ref() {
                if !mb.trim().is_empty() {
                    extra_ctx.push_str("\n\n");
                    extra_ctx.push_str(mb);
                }
            }
            if let Some(ws) = world_state.as_ref() {
                if !ws.trim().is_empty() {
                    extra_ctx.push_str("\n\n");
                    extra_ctx.push_str(ws);
                }
            }
            for (i, m) in messages.into_iter().enumerate() {
                let content = if i == 0 && m.role == "system" && !extra_ctx.is_empty() {
                    format!("{}{extra_ctx}", m.content)
                } else {
                    m.content
                };
                wire_messages.push(ChatRequestMessage {
                    role: m.role,
                    content,
                });
            }

            // Build the request body. `stream: true` requests SSE.
            // Sampler params mirror the locked local-engine config (AGENTS.md
            // §0 Sampler config): temp 1.0, top_p 0.95, min_p 0.1, top_k 0.
            // min_p + top_k are llama.cpp-native and non-standard for the
            // OpenAI /chat/completions contract — providers that don't
            // recognize them should ignore them, and the few that reject
            // unknown fields will surface a 400 (acceptable per the explicit
            // "full mirror" decision; aligns the API path with local).
            let body = serde_json::json!({
                "model": model,
                "messages": wire_messages,
                "stream": true,
                "temperature": temperature.unwrap_or(1.0),
                "top_p": 0.95,
                "min_p": 0.1,
                "top_k": 0,
            });

            let response = client
                .post(&url)
                .json(&body)
                .send()
                .await
                .map_err(|e| anyhow::anyhow!("API request to {url} failed: {e}"))?;

            if !response.status().is_success() {
                let status = response.status();
                let text = response.text().await.unwrap_or_default();
                return Err(anyhow::anyhow!(
                    "API returned {status}: {}",
                    text.chars().take(500).collect::<String>()
                ));
            }

            // Stream the SSE body. Each `data: {json}` line is one token chunk.
            // `data: [DONE]` terminates the stream. Lines not starting with
            // `data:` (comments, event headers) are ignored.
            use futures_util::StreamExt;
            let mut stream = response.bytes_stream();
            let mut buffer = String::new();
            let mut full_content = String::new();

            while let Some(chunk_res) = stream.next().await {
                // Honor cancel: stop reading + return what we have so far.
                if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                    break;
                }
                let bytes = chunk_res.map_err(|e| anyhow::anyhow!("SSE read error: {e}"))?;
                // The chunk may not be UTF8-aligned at boundaries; lossy-convert
                // since SSE is ASCII-framed and the JSON payloads are UTF8.
                buffer.push_str(&String::from_utf8_lossy(&bytes));

                // Process complete lines. Keep any trailing partial line in buffer.
                while let Some(newline_pos) = buffer.find('\n') {
                    let line = buffer[..newline_pos].trim().to_string();
                    buffer = buffer[newline_pos + 1..].to_string();
                    if line.is_empty() || !line.starts_with("data:") {
                        continue;
                    }
                    let data = line["data:".len()..].trim();
                    if data == "[DONE]" {
                        buffer.clear();
                        break;
                    }
                    // Parse the JSON chunk; on failure skip (some providers
                    // send keep-alive comments or partial events we don't care
                    // about). A malformed chunk must never kill the stream.
                    if let Ok(parsed) = serde_json::from_str::<ChatStreamChunk>(data) {
                        if let Some(choice) = parsed.choices.into_iter().next() {
                            if let Some(piece) = choice.delta.content {
                                if !piece.is_empty() {
                                    on_chunk(&piece);
                                    full_content.push_str(&piece);
                                }
                            }
                        }
                    }
                }
            }

            Ok(ParsedOutput {
                content: full_content,
                reasoning: String::new(),
                raw: String::new(),
            })
        })
    }

    fn is_ready(&self) -> bool {
        // An HttpBackend exists only when a profile is connected, so it's
        // always ready to stream (the network call itself will surface any
        // connectivity error at stream time).
        true
    }
}

/// The loaded model, as loaded from disk. This is an intermediate value — it
/// exists only between `load_blocking` and `into_static`, after which the model
/// is leaked to `&'static` and handed to the engine. The backend is NOT owned
/// here; it's the process-wide singleton from `shared_backend`.
pub struct LlamaModelHandle {
    model: LlamaModel,
    family: ModelFamily,
}

unsafe impl Send for LlamaModelHandle {}
unsafe impl Sync for LlamaModelHandle {}

impl LlamaModelHandle {
    /// Leak the model to a `&'static` reference so the engine can own a
    /// `LlamaContext<'static>`. See the module docs for the rationale.
    /// The shared backend ref is returned alongside it (already `&'static`).
    ///
    /// The family is returned by value (it's `Copy`).
    #[must_use]
    fn into_static(self) -> (&'static LlamaBackend, &'static LlamaModel, ModelFamily) {
        let backend_ref: &'static LlamaBackend = shared_backend();
        let model_ref: &'static LlamaModel = Box::leak(Box::new(self.model));
        (backend_ref, model_ref, self.family)
    }
}

/// The backend façade. Holds a handle to the engine thread (or `None` while
/// loading). Fully `Send`/`Sync` — no `LlamaContext` or `!Send` type crosses
/// out of the engine thread.
pub struct LlamaCppBackend {
    engine: Arc<std::sync::Mutex<Option<ChatEngine>>>,
}

/// Process-level slot for the leaked `&'static LlamaModel`. Filled once when
/// the chat backend loads (so the leaked model survives the loader thread
/// exiting). The schema delta engine reads this to create its OWN isolated
/// `LlamaContext` on the same model — true context isolation, the same
/// pattern as the embedder (§3B). `LlamaModel` is `Sync`, so a `&'static` ref
/// is safely shareable across the chat, embedder, and schema threads.
static SHARED_MODEL: std::sync::OnceLock<&'static LlamaModel> = std::sync::OnceLock::new();

impl LlamaCppBackend {
    /// Load the model off-thread, then spawn the persistent engine. Returns
    /// immediately with a backend handle; `on_result` fires when loading +
    /// engine init completes (success or failure).
    ///
    /// `context_size` fixes the `n_ctx` of the persistent context. It cannot
    /// change without re-spawning the engine — that's a future P concern
    /// (settings hot-reload).
    pub fn spawn_load(
        path: PathBuf,
        n_gpu_layers: u32,
        context_size: u32,
        on_result: Box<dyn FnOnce(Result<String, String>) + Send>,
    ) -> Arc<Self> {
        let engine_slot: Arc<std::sync::Mutex<Option<ChatEngine>>> =
            Arc::new(std::sync::Mutex::new(None));
        let slot_clone = Arc::clone(&engine_slot);

        std::thread::spawn(move || match Self::load_blocking(&path, n_gpu_layers) {
            Ok(handle) => {
                tracing::info!("model loaded from {}", path.display());
                let name = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("model")
                    .to_string();

                // Leak the model to &'static (the backend is already the
                // process-wide &'static from shared_backend). The engine thread
                // owns both for the process lifetime.
                let (backend_ref, model_ref, family) = handle.into_static();

                // Stash the model ref in the process-level slot so the schema
                // delta engine can create an isolated context on the same model.
                // set() is a no-op if already set (it won't be — first load).
                let _ = SHARED_MODEL.set(model_ref);

                // Spawn the persistent engine with Q8_0 KV cache + delta prefill.
                let (engine, init_rx) =
                    ChatEngine::spawn(backend_ref, model_ref, family, context_size);

                // Bug #6: await engine init confirmation BEFORE signaling
                // readiness. We're already on a background thread, so
                // blocking here doesn't stall the UI. If init_runtime failed
                // (CUDA context alloc error, etc.), report the error instead
                // of falsely claiming "ready".
                match init_rx.recv() {
                    Ok(Ok(())) => {
                        {
                            let mut g = slot_clone.lock().expect("engine mutex");
                            *g = Some(engine);
                        }
                        on_result(Ok(name));
                    }
                    Ok(Err(e)) => {
                        tracing::error!(error = %e, "engine init failed");
                        on_result(Err(e));
                    }
                    Err(_) => {
                        let msg = "engine init channel closed unexpectedly".to_string();
                        tracing::error!(error = %msg);
                        on_result(Err(msg));
                    }
                }
            }
            Err(e) => {
                tracing::error!(error = %e, "model load failed");
                on_result(Err(format!("{e}")));
            }
        });

        Arc::new(LlamaCppBackend {
            engine: engine_slot,
        })
    }

    fn load_blocking(path: &Path, n_gpu_layers: u32) -> anyhow::Result<LlamaModelHandle> {
        use llama_cpp_2::model::params::LlamaModelParams;
        let backend: &'static LlamaBackend = shared_backend();

        let params = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers);
        let model = LlamaModel::load_from_file(backend, path, &params)
            .map_err(|e| anyhow::anyhow!("model load: {e:?}"))?;

        let filename = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        let family = ModelFamily::from_model_name(filename);
        tracing::info!(family = ?family, filename, "detected model family");

        Ok(LlamaModelHandle {
            model,
            family,
        })
    }
}

impl GenerationClient for LlamaCppBackend {
    fn stream(
        &self,
        messages: Vec<ApiMessage>,
        memory_block: Option<String>,
        world_state: Option<String>,
        _context_size: u32,
        on_chunk: ChunkFn,
        cancel: CancelToken,
    ) -> StreamFuture {
        let engine = Arc::clone(&self.engine);
        Box::pin(async move {
            let (reply_tx, reply_rx) = std::sync::mpsc::channel::<EngineReply>();
            {
                let guard = engine.lock().expect("engine mutex");
                let eng = guard
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("model not loaded yet"))?;
                eng.request(EngineRequest {
                    messages,
                    on_chunk,
                    cancel,
                    memory_block,
                    world_state,
                    reply: reply_tx,
                })
                .map_err(|e| anyhow::anyhow!(e))?;
            }

            // Await the reply off the async runtime — generation takes seconds
            // and we must not block a tokio worker. The engine streams chunks
            // directly to `on_chunk` (the Tauri Channel) while we wait.
            let reply = tokio::task::spawn_blocking(move || reply_rx.recv())
                .await
                .map_err(|e| anyhow::anyhow!("join: {e}"))?
                .map_err(|_| anyhow::anyhow!("engine reply channel closed"))?;

            match reply {
                EngineReply::Ok(parsed) => Ok(parsed),
                EngineReply::Err(msg) => Err(anyhow::anyhow!(msg)),
            }
        })
    }

    fn is_ready(&self) -> bool {
        self.engine
            .lock()
            .map(|g| g.is_some())
            .unwrap_or(false)
    }
}

impl LlamaCppBackend {
    /// Shut down the engine thread + clear the slot. Posts `EngineMsg::Shutdown`
    /// AND blocks on the JoinHandle until the thread has fully exited + dropped
    /// its `EngineRuntime` (LlamaContext + the borrowed `&'static LlamaModel`
    /// → VRAM actually freed), then sets the inner slot to `None` so further
    /// `stream()` calls return the "not ready" error instead of posting to a
    /// dead thread. The synchronous join is load-bearing during model swaps —
    /// the old fire-and-forget version raced VRAM teardown and OOM'd the next
    /// `load_from_file` (Chloe's 2026-07-18 VRAM-overlap diagnosis). Callers
    /// using this from an async context should wrap it in `spawn_blocking`.
    pub fn shutdown(&self) {
        if let Some(engine) = self.engine.lock().map(|mut g| g.take()).unwrap_or(None) {
            engine.shutdown();
            tracing::info!("chat engine shutdown complete (thread joined + context dropped)");
        }
    }
}

/// Free function: the leaked `&'static LlamaModel`, available after the chat
/// backend finishes loading. Used by the schema delta engine to create an
/// isolated `LlamaContext` on the same model. Returns `None` if the model
/// hasn't loaded yet (callers should gate on backend readiness first).
pub fn shared_model() -> Option<&'static LlamaModel> {
    SHARED_MODEL.get().copied()
}
