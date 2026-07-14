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

/// Free function: the leaked `&'static LlamaModel`, available after the chat
/// backend finishes loading. Used by the schema delta engine to create an
/// isolated `LlamaContext` on the same model. Returns `None` if the model
/// hasn't loaded yet (callers should gate on backend readiness first).
pub fn shared_model() -> Option<&'static LlamaModel> {
    SHARED_MODEL.get().copied()
}
