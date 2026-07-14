pub mod chat_format;
pub mod engine;
pub mod kv_buffer;
pub mod llm;
pub mod memory;
pub mod memory_embedder;
pub mod memory_embedder_llama;
pub mod memory_rrf;
pub mod prompts;
pub mod schema;
pub mod schema_engine;
pub mod session;
pub mod stream_filter;

use std::sync::Arc;
use tauri::{Emitter, Manager};
use llm::GenerationClient;

/// The Memory engine's concrete embedder type, decided ONCE at startup. Using
/// `Box<dyn Embedder + Send + Sync>` lets `AppState` hold one concrete
/// `MemoryEngine` regardless of whether `Embed.gguf` was found — `LlamaCppEmbedder`
/// (real BERT backend) or `StubEmbedder` (byte-histogram fallback) both box into
/// this slot. One virtual call per `embed`, negligible next to multi-ms GPU work.
/// The `Embedder` trait is verified dyn-compatible (no `Self`, no generic
/// methods, manually-desugared `EmbedFuture` instead of `async fn`).
pub type DynEmbedder = Box<dyn memory_embedder::Embedder + Send + Sync>;

#[derive(Clone)]
pub struct AppState {
    pub session: Arc<tokio::sync::Mutex<session::Conversation>>,
    pub backend: Arc<std::sync::Mutex<Option<Arc<llm::LlamaCppBackend>>>>,
    pub settings: Arc<std::sync::Mutex<prompts::WupiSettings>>,
    /// The cancel token for the CURRENTLY active generation (if any). Each
    /// `chat_send` creates a fresh `CancelToken` and stores it here; `chat_stop`
    /// signals whatever is in this slot. This prevents overlapping sends from
    /// cross-wiring each other's cancellation (Bug #7).
    pub active_cancel: Arc<std::sync::Mutex<Option<llm::CancelToken>>>,
    /// The Memory engine. Wrapped in `OnceLock` because the embedder needs a
    /// model path resolved from the Tauri `app` handle, which isn't available
    /// when `AppState::new()` runs (before `setup()`). `setup()` fills it once;
    /// reads after init are lock-free. Always `Some` after `setup()` completes.
    pub memory: Arc<std::sync::OnceLock<Arc<memory::MemoryEngine<DynEmbedder>>>>,
    /// The world-state schema — "the schema IS the summarizer." A persistent,
    /// semi-structured record of the simulated world's state, updated after
    /// every chat turn by the background state-delta pass (schema_engine.rs).
    /// Held under tokio::sync::Mutex because it's read by chat_send (to inject
    /// into the prompt) and written by the delta-completion path.
    pub schema: Arc<tokio::sync::Mutex<schema::WorldSchema>>,
    /// Handle to the in-flight schema delta pass (if any). chat_send checks
    /// this to implement the invisible queue: if a pass is running when the
    /// user sends, the message waits for it to finish before the next
    /// generation starts. None = no pass running, proceed immediately.
    /// Always `Some(JoinHandle)` between turn-finalize and the next chat_send.
    pub pending_delta: Arc<tokio::sync::Mutex<Option<tokio::task::JoinHandle<()>>>>,
    /// The schema delta engine. Wrapped in `OnceLock` because spawning it
    /// requires the chat model to have loaded first (`shared_model()` is the
    /// leaked `&'static LlamaModel` the schema context is created from). For
    /// the B/C runtime test the engine is spawned LAZILY on first
    /// `debug_schema_delta` call; Component E will move this to an eager spawn
    /// at model-ready.
    pub schema_engine: Arc<std::sync::OnceLock<Arc<schema_engine::SchemaEngine>>>,
}

impl AppState {
    fn new() -> Self {
        Self {
            session: Arc::new(tokio::sync::Mutex::new(session::Conversation::new())),
            backend: Arc::new(std::sync::Mutex::new(None)),
            settings: Arc::new(std::sync::Mutex::new(prompts::WupiSettings::default())),
            active_cancel: Arc::new(std::sync::Mutex::new(None)),
            memory: Arc::new(std::sync::OnceLock::new()),
            schema: Arc::new(tokio::sync::Mutex::new(schema::WorldSchema::default())),
            pending_delta: Arc::new(tokio::sync::Mutex::new(None)),
            schema_engine: Arc::new(std::sync::OnceLock::new()),
        }
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    std::panic::set_hook(Box::new(|info| {
        let msg = format!("{info}\nbacktrace: {}", std::backtrace::Backtrace::force_capture());
        let _ = std::fs::write(std::env::temp_dir().join("wupi_panic.txt"), &msg);
    }));

    let log_dir = std::env::temp_dir();
    let file_appender = tracing_appender::rolling::never(&log_dir, "wupi_os.log");
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);
    std::mem::forget(_guard);
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("debug")),
        )
        .with_target(false)
        .with_writer(non_blocking)
        .init();

    tracing::info!("=== WUPI OS starting ===");
    tauri::Builder::default()
        .manage(AppState::new())
        .setup(|app| {
            tracing::info!("setup hook entered");
            let data_dir = app
                .path()
                .app_data_dir()
                .expect("app data dir is available");
            std::fs::create_dir_all(&data_dir).ok();
            tracing::info!("app data dir: {}", data_dir.display());

            let state: tauri::State<AppState> = app.state();
            let session_path = data_dir.join("session.json");
            if let Ok(loaded) = session::Conversation::load(&session_path) {
                // Bug #10: hold the guard once instead of locking twice
                // back-to-back (once to assign, once to read .len()).
                let count = {
                    let mut s = state.session.blocking_lock();
                    *s = loaded;
                    s.messages.len()
                };
                tracing::info!("loaded persisted session ({} messages)", count);
            }

            let model_path = resolve_model_path(app.handle());
            if let Some(path) = model_path {
                tracing::info!("spawning model load: {}", path.display());
                let app_handle = app.handle().clone();
                // context_size fixes the persistent context's n_ctx for the
                // session. Changing it requires re-spawning the engine (a
                // future P concern). Default comes from WupiSettings.
                let context_size = {
                    let s = state.settings.lock().expect("settings mutex");
                    s.context_size
                };
                let backend = llm::LlamaCppBackend::spawn_load(path, 99, context_size, Box::new(move |result| {
                    let payload = match &result {
                        Ok(name) => serde_json::json!({ "status": "ready", "model": name }),
                        Err(msg) => serde_json::json!({ "status": "error", "message": msg }),
                    };
                    let _ = app_handle.emit("model-status", payload);
                }));
                *state.backend.lock().expect("backend mutex") = Some(backend);
            } else {
                tracing::warn!("no model file found; running without LLM (echo mode)");
                let app_handle = app.handle().clone();
                let _ = app_handle.emit(
                    "model-status",
                    serde_json::json!({ "status": "no_model" }),
                );
            }

            // ── Memory engine (pillar 1) ────────────────────────────────
            // Build the MemoryEngine with the real BERT embedder if
            // `Embed.gguf` is on disk; fall back to StubEmbedder otherwise
            // (graceful degradation — documented contract in
            // memory_embedder_llama.rs::resolve_embed_model). The embedder is
            // boxed into `Box<dyn Embedder + Send + Sync>` so AppState holds
            // one concrete type regardless of which backend was chosen.
            //
            // `shared_backend()` (§2H) is the single `LlamaBackend::init()`
            // chokepoint: both the chat loader (above) and the embedder route
            // through it. The embedder thread does NOT block on chat-model
            // loading — `shared_backend` is a `OnceLock` that resolves on first
            // call; whichever loader hits it first inits, the other reuses.
            let embedder: DynEmbedder = match resolve_embed_model_dirs(app.handle()) {
                Some(path) => {
                    tracing::info!("spawning embed model load: {}", path.display());
                    let (embedder, init_rx) =
                        memory_embedder_llama::LlamaCppEmbedder::spawn_load(path, 99);
                    // Block on the readiness channel — same contract as the
                    // chat engine's Bug #6 fix. If init failed, fall back to
                    // the stub so the app still runs (memory just won't be
                    // semantic). This recv runs on the setup thread, which is
                    // fine — setup is allowed to block.
                    match init_rx.recv() {
                        Ok(Ok(())) => {
                            tracing::info!("memory engine: LlamaCppEmbedder ready");
                            Box::new(embedder)
                        }
                        Ok(Err(msg)) => {
                            tracing::warn!(
                                error = %msg,
                                "embedder init failed; falling back to StubEmbedder"
                            );
                            Box::new(memory_embedder::StubEmbedder {
                                dim: memory_embedder::EMBED_DIM,
                            })
                        }
                        Err(_) => {
                            tracing::warn!(
                                "embedder init channel closed; falling back to StubEmbedder"
                            );
                            Box::new(memory_embedder::StubEmbedder {
                                dim: memory_embedder::EMBED_DIM,
                            })
                        }
                    }
                }
                None => {
                    tracing::warn!(
                        "no Embed.gguf found; memory engine using StubEmbedder (no semantic search)"
                    );
                    Box::new(memory_embedder::StubEmbedder {
                        dim: memory_embedder::EMBED_DIM,
                    })
                }
            };

            let memory_db_path = data_dir.join("memory.sqlite");
            match memory::MemoryEngine::open(&memory_db_path, embedder) {
                Ok(engine) => {
                    let _ = state.memory.set(Arc::new(engine));
                    tracing::info!(db = %memory_db_path.display(), "memory engine initialized");
                }
                Err(e) => {
                    // DB open failure is fatal for memory but must not kill
                    // the app. Leave the OnceLock empty; callers check `get`.
                    tracing::error!(error = %format!("{e:#}"), "memory engine init failed");
                }
            }

            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { .. } = event {
                if window.app_handle().webview_windows().len() <= 1 {
                    window.app_handle().exit(0);
                }
            }
        })
        .invoke_handler(tauri::generate_handler![
            app_ready,
            chat_send,
            chat_stop,
            get_settings,
            debug_memory_query,
            debug_schema_delta,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
    tracing::info!("=== WUPI OS event loop exited ===");
}

#[tauri::command]
fn app_ready(state: tauri::State<'_, AppState>) -> String {
    let backend = state.backend.lock().expect("backend mutex");
    if let Some(b) = backend.as_ref() {
        if b.is_ready() {
            return "ready · model loaded".to_string();
        }
        return "loading model…".to_string();
    }
    "ready · no model (echo mode)".to_string()
}

fn resolve_model_path(app: &tauri::AppHandle) -> Option<std::path::PathBuf> {
    use tauri::Manager;

    let candidates: Vec<std::path::PathBuf> = {
        let mut v = Vec::new();
        if let Some(d) = app.path().resource_dir().ok() {
            v.push(d.join("models"));
        }
        if let Some(exe) = std::env::current_exe().ok() {
            if let Some(parent) = exe.parent() {
                v.push(parent.join("models"));
                if let Some(grand) = parent.parent().and_then(|g| g.parent()) {
                    v.push(grand.join("src-tauri").join("models"));
                }
                if let Some(gg) = parent.parent().and_then(|g| g.parent()).and_then(|g| g.parent()) {
                    v.push(gg.join("src-tauri").join("models"));
                }
            }
        }
        if let Some(data) = app.path().app_data_dir().ok() {
            v.push(data.join("models"));
        }
        v
    };

    for dir in &candidates {
        if dir.exists() {
            if let Some(picked) = pick_main_model(dir) {
                tracing::info!("resolved model: {} (from {})", picked.display(), dir.display());
                return Some(picked);
            }
        }
    }
    None
}

fn pick_main_model(dir: &std::path::Path) -> Option<std::path::PathBuf> {
    let ggufs: Vec<_> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("gguf"))
        .collect();
    // Locked naming convention (2026-07-12): the chat model is always
    // `WUPI.gguf`. Match the canonical name first (case-insensitive) so
    // resolution never depends on the size fallback. Embed.gguf is excluded
    // implicitly — it isn't named WUPI and is far smaller than WUPI.gguf, so
    // the size fallback would skip it anyway.
    if let Some(m) = ggufs.iter().find(|e| {
        e.file_name().to_string_lossy().to_lowercase() == "wupi.gguf"
    }) {
        return Some(m.path());
    }
    ggufs
        .into_iter()
        .max_by_key(|e| e.metadata().ok().map(|m| m.len()).unwrap_or(0))
        .map(|e| e.path())
}

/// Walk the same candidate dirs as `resolve_model_path`, but for the embeddings
/// model (`Embed.gguf`). Sibling to the chat model's discovery so the embedder
/// loader is self-contained at the wiring seam. Returns `None` when no embed
/// model is present — the caller falls back to `StubEmbedder` (graceful, not a
/// crash). Exact-name match only; no size fallback (only one file will ever be
/// named `Embed.gguf`).
fn resolve_embed_model_dirs(app: &tauri::AppHandle) -> Option<std::path::PathBuf> {
    use tauri::Manager;
    let mut dirs: Vec<std::path::PathBuf> = Vec::new();
    if let Some(d) = app.path().resource_dir().ok() {
        dirs.push(d.join("models"));
    }
    if let Some(exe) = std::env::current_exe().ok() {
        if let Some(parent) = exe.parent() {
            dirs.push(parent.join("models"));
            if let Some(grand) = parent.parent().and_then(|g| g.parent()) {
                dirs.push(grand.join("src-tauri").join("models"));
            }
            if let Some(gg) = parent.parent().and_then(|g| g.parent()).and_then(|g| g.parent()) {
                dirs.push(gg.join("src-tauri").join("models"));
            }
        }
    }
    if let Some(data) = app.path().app_data_dir().ok() {
        dirs.push(data.join("models"));
    }
    memory_embedder_llama::resolve_embed_model(&dirs)
}

#[tauri::command]
async fn chat_send(
    text: String,
    on_event: tauri::ipc::Channel<serde_json::Value>,
    state: tauri::State<'_, AppState>,
    app: tauri::AppHandle,
) -> Result<(), String> {
    tracing::info!(?text, "chat_send");

    // Bug #7: create a FRESH cancel token for this request only. Each
    // chat_send gets its own token stored in active_cancel; chat_stop signals
    // whatever is there. This prevents overlapping sends from un-canceling
    // each other.
    let cancel: llm::CancelToken =
        Arc::new(std::sync::atomic::AtomicBool::new(false));
    {
        let mut slot = state.active_cancel.lock().expect("active_cancel mutex");
        *slot = Some(Arc::clone(&cancel));
    }

    let settings = state.settings.lock().expect("settings mutex").clone();

    // ── Memory retrieval (pillar 3, §2F Option 3) ───────────────────────
    // Embed the user's just-typed text and pull top hits BEFORE the session
    // lock. This is ON the chat path by design (§3A) — embedding takes ms on
    // GPU, the SQLite work is spawn_blocking-internal. The just-typed message
    // isn't archived yet (pillar 2 archives after generation), so we never
    // retrieve the thing we're about to send.
    //
    // §2F cost: the retrieved block differs per query → the prompt structure
    // changes every turn → the structural-divergence guard (engine.rs) cold-
    // resets the KV cache. Delta-prefill is dead on Memory-enabled turns. This
    // is the accepted v1 cost; the cache-layout optimization is a later pass.
    // ── Memory retrieval (§2F eager-prefill layout, 2026-07-13) ────────
    // The retrieved block is NO LONGER baked into the system prompt. It's
    // threaded separately as `memory_block` and injected into the inter-turn
    // region by `render_prompt`. This keeps the system+turns prefix
    // byte-identical across turns (the precondition for eager prefill).
    let memory_block = match state.memory.get() {
        Some(engine) => match engine.search(&text, 5).await {
            Ok(hits) if !hits.is_empty() => Some(memory::render_memory_block(&hits)),
            Ok(_) => None,
            Err(e) => {
                tracing::warn!(error = %format!("{e:#}"), "memory search failed; injecting nothing");
                None
            }
        },
        // OnceLock empty = memory engine failed to init at startup. Memory is
        // best-effort; chat proceeds with no retrieved context.
        None => {
            tracing::trace!("memory engine not initialized; skipping retrieval");
            None
        }
    };
    let system_prompt = prompts::build_system_content(&settings);

    // §2F eager-prefill sliding window (2026-07-13): cap visible history to
    // the last VISIBLE_WINDOW messages regardless of token budget. Memory (M)
    // backfills evicted turns via retrieval. Truncation in the engine becomes
    // a safety net that effectively never fires (4 short turns ≪ ~3000 budget).
    // 6 messages = 3 full user↔assistant turns. The sweet spot: enough
    // recency that the model has natural conversational continuity, small
    // enough that truncation never fires and the prompt stays cheap. Gemma
    // 12B handles this with zero performance hit. Tunable.
    const VISIBLE_WINDOW: usize = 6;

    let messages = {
        let mut s = state.session.lock().await;
        s.add_message(session::Role::User, text.clone());
        save_session(&app, &s).await;
        s.assemble_api_messages_windowed(&system_prompt, VISIBLE_WINDOW)
    };

    let on_chunk: llm::ChunkFn = Arc::new({
        let on_event = on_event.clone();
        move |piece: &str| {
            let _ = on_event.send(serde_json::json!({ "type": "chunk", "text": piece }));
        }
    });

    let backend_opt = state.backend.lock().expect("backend mutex").clone();
    let result = if let Some(backend) = backend_opt {
        match backend
            .stream(messages, memory_block, settings.context_size, on_chunk, cancel.clone())
            .await
        {
            Ok(text) => text,
            Err(e) => {
                clear_active_cancel(&state);
                rollback_last_user_message(&state, &app).await;
                on_event
                    .send(serde_json::json!({ "type": "error", "message": format!("{e}") }))
                    .map_err(|e| e.to_string())?;
                return Ok(());
            }
        }
    } else {
        let echo = llm::EchoBackend;
        match echo.stream(messages, None, settings.context_size, on_chunk, cancel.clone()).await {
            Ok(t) => t,
            Err(e) => {
                clear_active_cancel(&state);
                rollback_last_user_message(&state, &app).await;
                on_event
                    .send(serde_json::json!({ "type": "error", "message": format!("{e}") }))
                    .map_err(|e| e.to_string())?;
                return Ok(());
            }
        }
    };

    // Bug #3 Step 4: persist the raw model output alongside the cleaned
    // content + reasoning so the formatter can re-render cache-coherently
    // next turn (no full re-prefill of the previous reply).
    {
        let mut s = state.session.lock().await;
        s.add_assistant_turn(
            result.content.clone(),
            result.reasoning.clone(),
            result.raw.clone(),
        );
        save_session(&app, &s).await;

        // ── Memory archiving (pillar 2) ───────────────────────────────
        // Trigger is turn-COMPLETION, not truncation. We read from the
        // Conversation (clean strings), sidestepping the engine.rs:480
        // token-boundary-drift landmine entirely — truncate_to_fit operates on
        // LlamaToken slices with no safe mapping back to Message text.
        //
        // Both turns archived (user + assistant) so search can match either.
        // spawn detaches → add_memory's internal spawn_blocking runs the SQLite
        // insert off the hot path. The chat loop never awaits it. Errors are
        // logged-and-dropped inside the task: memory is best-effort, a failed
        // archive must not break chat.
        //
        // Salience flat 1.0 for v1 (the field is stored but unused by
        // retrieval today; a heuristic is a later concern). chunk_index stays
        // 0 (whole-message; no chunking yet).
        if let Some(engine) = state.memory.get() {
            // The user message is the second-to-last (last is the assistant
            // turn we just appended). checked_sub(2) guards the cold-start
            // edge where messages is unexpectedly short.
            let user_text = s.messages.len().checked_sub(2).and_then(|i| s.messages.get(i)).map(|m| m.content.clone());
            let asst_text = result.content.clone();
            let engine = Arc::clone(engine);
            tokio::spawn(async move {
                if let Some(text) = user_text {
                    if let Err(e) = engine.add_memory(text, memory::Role::User, 1.0).await {
                        tracing::warn!(error = %format!("{e:#}"), "archive user turn failed");
                    }
                }
                if let Err(e) = engine.add_memory(asst_text, memory::Role::Assistant, 1.0).await {
                    tracing::warn!(error = %format!("{e:#}"), "archive assistant turn failed");
                }
            });
        }
    }

    clear_active_cancel(&state);

    on_event
        .send(serde_json::json!({
            "type": "done",
            "final_text": result.content,
            "reasoning": result.reasoning,
        }))
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// Clear the active cancel slot. Called at every exit path of `chat_send`
/// (success + both error branches) so a stale token is never left behind.
fn clear_active_cancel(state: &tauri::State<'_, AppState>) {
    let mut slot = state.active_cancel.lock().expect("active_cancel mutex");
    *slot = None;
}

async fn rollback_last_user_message(state: &tauri::State<'_, AppState>, app: &tauri::AppHandle) {
    let mut s = state.session.lock().await;
    if s.last_message_is_user() {
        s.pop_last_message();
        save_session(app, &s).await;
    }
}

#[tauri::command]
async fn chat_stop(state: tauri::State<'_, AppState>) -> Result<(), String> {
    tracing::info!("chat_stop requested");
    let slot = state.active_cancel.lock().expect("active_cancel mutex");
    if let Some(cancel) = slot.as_ref() {
        cancel.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    Ok(())
}

/// Debug probe into the Memory engine (pillar 4). Embeds the query, runs the
/// hybrid FTS5 + vec0 search, and returns the RRF-fused ranked results with
/// scores. Off the chat path entirely — this is the observability surface for
/// tuning retrieval independently of generation.
///
/// `top_k` defaults to 10 when `None`. Returns an error string (not a panic)
/// if the memory engine isn't initialized or the query fails — the panel
/// renders it as a red message.
#[tauri::command]
async fn debug_memory_query(
    query: String,
    top_k: Option<usize>,
    state: tauri::State<'_, AppState>,
) -> Result<Vec<memory::RankedMemory>, String> {
    let engine = state
        .memory
        .get()
        .ok_or_else(|| "memory engine not initialized".to_string())?;
    engine
        .search(&query, top_k.unwrap_or(10))
        .await
        .map_err(|e| format!("{e:#}"))
}

/// Debug probe into the schema delta engine (B/C runtime test). Posts a
/// SYNTHETIC exchange (the caller supplies both sides) + the current schema,
/// waits for the delta pass to complete, and returns:
///   - the raw model output (what the schema model actually emitted)
///   - the parsed delta (if JSON was valid; else null)
///   - any error string
///   - the resulting schema JSON (after optionally applying the delta)
///
/// `apply: true` merges the delta into AppState.schema so the caller can
/// chain multiple calls and watch the schema evolve. `apply: false` is a dry
/// run — the schema is untouched, useful for prompt-tuning without side effects.
///
/// The schema engine is spawned LAZILY on first call (gated on the chat model
/// being loaded — `shared_model()` must be `Some`). Mirrors the Memory engine's
/// OnceLock-once pattern; Component E will move this to an eager spawn at
/// model-ready. Returns an error string if the chat model isn't loaded yet.
#[tauri::command]
async fn debug_schema_delta(
    user_exchange: String,
    assistant_exchange: String,
    apply: Option<bool>,
    state: tauri::State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    // Lazy-spawn the schema engine on first call. `get_or_init` is race-safe
    // if two debug calls land concurrently.
    let engine = if let Some(e) = state.schema_engine.get() {
        Arc::clone(e)
    } else {
        // The chat model must be loaded first — shared_model() is None until
        // the loader thread finishes + hands back the &'static LlamaModel.
        if llm::shared_model().is_none() {
            return Err("chat model not loaded yet — schema engine cannot start".into());
        }
        let (engine, init_rx) = schema_engine::SchemaEngine::spawn();
        // Block on the readiness channel (Bug #6 contract). This recv runs on
        // the tokio worker — fine, setup-style blocking, not a hot path.
        let ready = tokio::task::spawn_blocking(move || init_rx.recv())
            .await
            .map_err(|e| format!("init join: {e}"))?
            .map_err(|e| format!("init channel: {e}"))?;
        match ready {
            Ok(()) => {
                tracing::info!("schema engine ready (lazy spawn via debug IPC)");
            }
            Err(msg) => {
                return Err(format!("schema engine init failed: {msg}"));
            }
        }
        let engine = Arc::new(engine);
        let _ = state.schema_engine.set(Arc::clone(&engine));
        engine
    };

    // Snapshot the current schema (the delta pass diffs against this).
    let current = state.schema.lock().await.clone();

    // Post the delta request + await the reply off the tokio worker (the
    // schema thread is a bare std::thread; its mpsc::Receiver is blocking).
    let reply_rx = engine
        .request_delta((user_exchange, assistant_exchange), &current)
        .map_err(|e| format!("{e:#}"))?;
    let reply = tokio::task::spawn_blocking(move || reply_rx.recv())
        .await
        .map_err(|e| format!("reply join: {e}"))?
        .map_err(|e| format!("reply channel: {e}"))?;

    // Optionally apply the delta so the caller can chain calls and watch the
    // schema evolve across a multi-turn scenario.
    let schema_after = if apply.unwrap_or(false) {
        if let Some(ref delta) = reply.delta {
            let mut s = state.schema.lock().await;
            s.apply_delta(delta.clone());
            s.to_json_pretty()
        } else {
            // Parse failed — return the unchanged schema.
            state.schema.lock().await.to_json_pretty()
        }
    } else {
        current.to_json_pretty()
    };

    Ok(serde_json::json!({
        "raw_output": reply.raw_output,
        "delta": reply.delta,
        "error": reply.error,
        "schema_after": schema_after,
    }))
}

#[tauri::command]
async fn get_settings(state: tauri::State<'_, AppState>) -> Result<serde_json::Value, String> {
    // Bug #9: read the actual settings instead of hardcoded values.
    // Clone the values out of the guard before awaiting the session lock so
    // the non-Send MutexGuard isn't held across an await point.
    let (context_size, conversation_budget) = {
        let s = state.settings.lock().expect("settings mutex");
        (s.context_size, s.conversation_budget)
    };
    Ok(serde_json::json!({
        "contextSize": context_size,
        "conversationBudget": conversation_budget,
        "messageCount": state.session.lock().await.messages.len(),
    }))
}

/// Persist the session off the Tokio worker pool.
///
/// `Conversation::save` is atomic (temp + fsync + rename, see §2E) but
/// synchronous — `File::create` / `write_all` / `sync_all` / `rename` all
/// block the calling thread on the disk. Running that on a Tokio worker
/// (which is what the old sync `save_session` did) stalls the async runtime
/// for the duration of the write + fsync. Harmless today (one user, one
/// chat, save is ~ms on SSD), but the moment the Memory engine adds
/// concurrent async work racing the save, a blocked worker becomes a real
/// stall. `spawn_blocking` moves the I/O onto the dedicated blocking thread
/// pool (default 512 threads) so workers stay free to poll futures.
///
/// The session mutex guard is still held across the `.await` by the caller
/// — that's correct for a `tokio::sync::Mutex` (its guard is await-safe) and
/// serializes concurrent saves, which we want anyway.
async fn save_session(app: &tauri::AppHandle, conv: &session::Conversation) {
    use tauri::Manager;
    let Some(data_dir) = app.path().app_data_dir().ok() else {
        return;
    };
    let path = data_dir.join("session.json");
    // Clone so the closure owns its data (spawn_blocking needs 'static). The
    // Conversation is a Vec of small messages — cheap to clone relative to a
    // disk fsync.
    let conv = conv.clone();
    let _ = tokio::task::spawn_blocking(move || {
        if let Err(e) = conv.save(&path) {
            tracing::warn!(?e, "failed to persist session");
        }
    })
    .await;
}
