pub mod chat_format;
pub mod codex;
pub mod engine;
#[cfg(windows)]
pub mod hardware;
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
pub mod sim_card;
pub mod stream_filter;
pub mod system_menu;
pub mod terminal;
pub mod theme;
pub mod user_profile;

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
    /// The active simulation card's id — the partition key for Memory
    /// retrieval and archiving (AGENTS.md §2M). Defaults to
    /// [`memory::WUPI_OS_CARD_ID`] (the Wupi-as-assistant namespace) until
    /// the character/simulation card system exists; when a card loads, its
    /// loader sets this. Read on every chat turn (search + 2× archive).
    pub active_card_id: Arc<std::sync::Mutex<String>>,
    /// The active Simulation Card (the parsed persona artifact). Filled once
    /// in `setup()` from `cards/Wupi.sim`; reads after init are lock-free.
    /// `chat_send` renders it into the system-prompt persona section;
    /// `get_intro` reads its randomized introduction list. Always `Some`
    /// after `setup()` (the loader falls back to a stub, never `None`).
    pub active_card: Arc<std::sync::OnceLock<sim_card::SimCard>>,
    /// The resolved path to the operator's profile (`cards/Operator.xml`),
    /// filled once in `setup()`. `None` when no profile resolved (the common
    /// case until the operator authors one). The PATH is stable; the CONTENT
    /// is re-read fresh each `chat_send` (hot-reload — see `user_profile`).
    /// Lock-free reads after `setup`. Held as `Option<PathBuf>` so a missing
    /// profile is `None`, distinct from "not yet resolved."
    pub operator_path: Arc<std::sync::OnceLock<Option<std::path::PathBuf>>>,
    /// The active theme + color code (defaults Aurora / Vibrant). Read by the
    /// frontend to paint the cascade panels; written by `theme_set`. Held
    /// under a std Mutex — never awaited across.
    pub theme: Arc<std::sync::Mutex<theme::ThemeSettings>>,
    /// The resolved path to `theme.json` in app data. Filled once in setup;
    /// `theme_set` saves to it. OnceLock because it needs the Tauri app handle
    /// to resolve app_data_dir (not available in AppState::new()).
    pub theme_path: Arc<std::sync::OnceLock<std::path::PathBuf>>,
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
            active_card_id: Arc::new(std::sync::Mutex::new(
                memory::WUPI_OS_CARD_ID.to_owned(),
            )),
            active_card: Arc::new(std::sync::OnceLock::new()),
            operator_path: Arc::new(std::sync::OnceLock::new()),
            theme: Arc::new(std::sync::Mutex::new(theme::ThemeSettings::default())),
            theme_path: Arc::new(std::sync::OnceLock::new()),
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
        .manage(terminal::new_registry())
        .manage(hardware::AudioRegistry)
        .setup(|app| {
            tracing::info!("setup hook entered");
            let data_dir = app
                .path()
                .app_data_dir()
                .expect("app data dir is available");
            std::fs::create_dir_all(&data_dir).ok();
            tracing::info!("app data dir: {}", data_dir.display());

            let state: tauri::State<AppState> = app.state();

            // ── Theme (persisted; defaults Aurora / Vibrant) ───────────────
            // Resolved path cached on AppState so theme_get/theme_set don't
            // need the app handle; load now so the frontend can read the
            // persisted choice on boot.
            {
                let theme_path = theme::ThemeSettings::resolve_path(&data_dir);
                let loaded = theme::ThemeSettings::load(&theme_path);
                tracing::info!(
                    theme = %loaded.theme,
                    color_code = %loaded.color_code,
                    "theme loaded"
                );
                *state.theme.lock().expect("theme mutex") = loaded;
                let _ = state.theme_path.set(theme_path);
            }

            // ── Session + schema are EPHEMERAL (2026-07-14) ───────────────
            // WUPI OS launches into a FRESH session every time — no
            // session.json or world_schema.json load. Memory (memory.sqlite)
            // is the ONLY persistent state; it survives across launches and
            // is how Wupi "remembers" you. The session + schema live only in
            // memory for the current launch.
            //
            // Why: Wupi is a meta-assistant / Copilot (§1), not a roleplay
            // chat app. You don't resume your last Windows session every
            // reboot. Persisting the session caused cross-topic contamination
            // (a cyberpunk story's messages bled into a fresh dungeon run).
            //
            // The character/simulation card system (future, unbuilt) will
            // re-introduce SCOPED persistence: a card carries its own session
            // + its own schema, resumable on demand. That's an opt-in layer
            // on top of the ephemeral default, NOT a replacement for it. The
            // atomic save/load methods in session.rs + schema.rs are retained
            // for that future use (marked #[allow(dead_code)] until then).
            tracing::info!("fresh session + empty schema (ephemeral mode)");

            // ── Simulation Card (Wupi's persona) ─────────────────────────
            // Load the default card (`cards/Wupi.sim`) before anything else —
            // it's a single cheap file read + parse, independent of model
            // loading, and `get_intro` (called from the frontend's boot) may
            // race the model load. `load_or_fallback` degrades gracefully to
            // a stub persona on any error (missing file, bad XML), so the OS
            // always boots. The card's `id` becomes the active card partition
            // key for Memory once cards own their partition; today Memory
            // stays on the Wupi sentinel namespace.
            let card = match resolve_card_path(app.handle()) {
                Some(path) => sim_card::load_or_fallback(&path),
                None => {
                    tracing::warn!(
                        "no cards/Wupi.sim found; using minimal fallback persona \
                         (persona section suppressed in the prompt)"
                    );
                    sim_card::fallback()
                }
            };
            let _ = state.active_card.set(card);

            // ── Operator profile (User Profile system) ───────────────────────
            // Resolve the operator's profile path (`cards/Operator.xml`) once
            // and cache it. The CONTENT is re-read fresh each chat_send
            // (hot-reload: a live edit takes effect on the very next message,
            // no reboot); only the PATH is stable. `None` when no profile
            // exists — the common case until the operator authors one. Wupi
            // then runs without a <user_profile> section (graceful: she just
            // doesn't know who she's talking to until the file exists).
            let operator = resolve_operator_path(app.handle());
            if let Some(p) = &operator {
                tracing::info!("resolved operator profile: {}", p.display());
            } else {
                tracing::info!("no Operator.xml found; running without a user profile");
            }
            let _ = state.operator_path.set(operator);

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
                    match &result {
                        Ok(name) => {
                            let _ = app_handle.emit(
                                "model-status",
                                serde_json::json!({ "status": "ready", "model": name }),
                            );

                            // ── Eager schema-engine spawn (Component E) ────
                            // The chat model is loaded → shared_model() is now
                            // Some. Spawn the schema delta engine so it's ready
                            // before the first chat turn (Component D's queue
                            // assumes it exists). Mirrors the embedder's block-
                            // on-readiness pattern. Runs on the loader thread
                            // (disposable background thread) — blocking recv is
                            // fine here. The schema context alloc is just KV
                            // allocation (ms, reuses the leaked model), so the
                            // delay before "ready" is negligible.
                            let app_state = app_handle.state::<AppState>();
                            if app_state.schema_engine.get().is_none() {
                                let (engine, init_rx) = schema_engine::SchemaEngine::spawn();
                                match init_rx.recv() {
                                    Ok(Ok(())) => {
                                        tracing::info!(
                                            "schema engine ready (eager spawn at model-ready)"
                                        );
                                        let _ = app_state.schema_engine.set(Arc::new(engine));
                                    }
                                    Ok(Err(msg)) => {
                                        tracing::warn!(
                                            error = %msg,
                                            "schema engine init failed; schema updates disabled"
                                        );
                                    }
                                    Err(_) => {
                                        tracing::warn!(
                                            "schema engine init channel closed; \
                                             schema updates disabled"
                                        );
                                    }
                                }
                            }
                        }
                        Err(msg) => {
                            let _ = app_handle.emit(
                                "model-status",
                                serde_json::json!({ "status": "error", "message": msg }),
                            );
                        }
                    }
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

                    // ── Codex seed (Codex v1, 2026-07-14) ──────────────────────
                    // Reconcile authored `.md` files in `codex/` against the
                    // Codex-tagged entries already stored in memory.sqlite.
                    // Idempotent (hash-based): re-runs against an unchanged
                    // source set do zero writes. Best-effort — a failed seed
                    // is logged-and-dropped, never fatal (same contract as the
                    // embedder fallback). Runs synchronously here (setup is
                    // allowed to block — it already blocks on the embedder
                    // readiness channel above).
                    if let Some(codex_dir) = resolve_codex_dir(app.handle()) {
                        if let Some(engine) = state.memory.get() {
                            match tauri::async_runtime::block_on(
                                codex::seed_codex(engine, &codex_dir, memory::WUPI_OS_CARD_ID),
                            ) {
                                Ok(report) => tracing::info!(
                                    seeded = report.seeded,
                                    updated = report.updated,
                                    purged = report.purged,
                                    unchanged = report.unchanged,
                                    "codex seeded"
                                ),
                                Err(e) => tracing::warn!(
                                    error = %format!("{e:#}"),
                                    "codex seed failed; continuing without authored lore"
                                ),
                            }
                        }
                    } else {
                        tracing::info!("no codex/ dir found; skipping codex seed");
                    }
                }
                Err(e) => {
                    // DB open failure is fatal for memory but must not kill
                    // the app. Leave the OnceLock empty; callers check `get`.
                    tracing::error!(error = %format!("{e:#}"), "memory engine init failed");
                }
            }

            // ── System tray (paw icon) — installed once the app handle exists.
            // Built last so an icon-build failure can't strand the earlier
            // engine init. A failure here is non-fatal: log and continue; the
            // app still runs, just without a tray (Sleep would then hide the
            // window with no way back except Restart/relaunch).
            if let Err(e) = system_menu::build_tray(&app.handle()) {
                tracing::error!(error = %format!("{e:#}"), "tray icon build failed");
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
        // Tray-menu item dispatch: "Wake" restores the window, "Quit" is a
        // full shutdown. Routed through the same power actions the paw
        // dropdown uses.
        .on_menu_event(|app, event| {
            match event.id().as_ref() {
                system_menu::TRAY_WAKE => system_menu::power_wake(&app),
                system_menu::TRAY_QUIT => system_menu::power_shutdown(&app),
                _ => {}
            }
        })
        .invoke_handler(tauri::generate_handler![
            app_ready,
            chat_send,
            chat_stop,
            get_settings,
            get_intro,
            debug_memory_query,
            debug_schema_delta,
            system_menu::power_shutdown_cmd,
            system_menu::power_restart_cmd,
            system_menu::power_sleep_cmd,
            theme_get,
            theme_set,
            terminal::terminal_create_window,
            terminal::terminal_init,
            terminal::terminal_input,
            terminal::terminal_resize,
            terminal::terminal_close,
            hardware::audio::audio_get_state,
            hardware::audio::audio_set_volume,
            hardware::audio::audio_list_outputs,
            hardware::audio::audio_set_default_output,
            hardware::wifi::wifi_get_current,
            hardware::wifi::wifi_scan,
            hardware::wifi::wifi_connect,
            hardware::wifi::wifi_toggle_radio,
            hardware::bluetooth::bluetooth_get_state,
            hardware::bluetooth::bluetooth_toggle_radio,
            hardware::bluetooth::bluetooth_list_devices,
            hardware::bluetooth::bluetooth_discover,
            hardware::bluetooth::bluetooth_pair,
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

/// Randomized boot greeting — picks one line from the active card's
/// `<introductions>` list. The result is a UI-only flourish: the frontend
/// renders it as a Wupi bubble but it is NEVER added to the conversation,
/// sent to the model, or archived to memory (an assistant turn with no
/// preceding user turn would be a malformed structure + memory noise). Returns
/// `null` when the card has no introductions (e.g. the fallback stub) → the
/// frontend shows no boot bubble.
#[tauri::command]
fn get_intro(state: tauri::State<'_, AppState>) -> Option<String> {
    state
        .active_card
        .get()
        .and_then(|c| c.random_intro().map(|s| s.to_owned()))
}

/// Read the active theme + color code. The frontend paints the cascade
/// panels from this and applies the palette to the aurora canvas.
#[tauri::command]
fn theme_get(state: tauri::State<'_, AppState>) -> serde_json::Value {
    let t = state.theme.lock().expect("theme mutex");
    serde_json::json!({ "theme": t.theme, "colorCode": t.color_code })
}

/// Persist a new theme + color code and return the updated value. The
/// frontend re-paints the canvas on the next frame after the round-trip.
#[tauri::command]
fn theme_set(
    theme_name: String,
    color_code: String,
    state: tauri::State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let path = state
        .theme_path
        .get()
        .ok_or_else(|| "theme path not initialized".to_string())?
        .clone();
    let new_settings = theme::ThemeSettings {
        theme: theme_name,
        color_code,
    };
    new_settings.save(&path);
    *state.theme.lock().expect("theme mutex") = new_settings.clone();
    tracing::info!(
        theme = %new_settings.theme,
        color_code = %new_settings.color_code,
        "theme updated"
    );
    Ok(serde_json::json!({
        "theme": new_settings.theme,
        "colorCode": new_settings.color_code,
    }))
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

/// Resolve the default Simulation Card (`cards/Wupi.sim`) by walking the same
/// candidate-dir list as [`resolve_model_path`], but joining `"cards"` instead
/// of `"models"` and exact-matching `Wupi.sim` (case-insensitive). Locked-name
/// single file — no size fallback (only one file will ever be named
/// `Wupi.sim`). Returns `None` when no card is found; the caller falls back to
/// a minimal stub persona so the app still boots.
fn resolve_card_path(app: &tauri::AppHandle) -> Option<std::path::PathBuf> {
    use tauri::Manager;

    let mut candidates: Vec<std::path::PathBuf> = Vec::new();
    if let Some(d) = app.path().resource_dir().ok() {
        candidates.push(d.join("cards"));
    }
    if let Some(exe) = std::env::current_exe().ok() {
        if let Some(parent) = exe.parent() {
            candidates.push(parent.join("cards"));
            if let Some(grand) = parent.parent().and_then(|g| g.parent()) {
                candidates.push(grand.join("cards"));
            }
            if let Some(gg) = parent.parent().and_then(|g| g.parent()).and_then(|g| g.parent()) {
                candidates.push(gg.join("cards"));
            }
        }
    }
    if let Some(data) = app.path().app_data_dir().ok() {
        candidates.push(data.join("cards"));
    }

    for dir in &candidates {
        if !dir.exists() {
            continue;
        }
        // Exact name match (case-insensitive). The dev path resolves to the
        // repo's `cards/` dir; the exe-sibling + resource paths resolve to
        // wherever cards ship alongside a packaged build.
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                if entry.file_name().to_string_lossy().to_lowercase() == "wupi.sim" {
                    let path = entry.path();
                    tracing::info!("resolved card: {} (from {})", path.display(), dir.display());
                    return Some(path);
                }
            }
        }
    }
    None
}

/// Resolve the operator's profile (`cards/Operator.xml`) by walking the same
/// candidate-dir list as [`resolve_card_path`], joining `"cards"`, and exact-
/// matching `Operator.xml` (case-insensitive). Sibling to Wupi.sim in the same
/// dir. Returns `None` when no profile is found — the common case until the
/// operator authors one; the caller runs without a `<user_profile>` section
/// (graceful, not a crash).
///
/// Only the PATH is resolved here (once, in setup). The CONTENT is re-read
/// fresh each `chat_send` via `user_profile::load` — that's the hot-reload
/// mechanism (live edits take effect on the next message, no reboot, no
/// watcher thread).
fn resolve_operator_path(app: &tauri::AppHandle) -> Option<std::path::PathBuf> {
    use tauri::Manager;

    let mut candidates: Vec<std::path::PathBuf> = Vec::new();
    if let Some(d) = app.path().resource_dir().ok() {
        candidates.push(d.join("cards"));
    }
    if let Some(exe) = std::env::current_exe().ok() {
        if let Some(parent) = exe.parent() {
            candidates.push(parent.join("cards"));
            if let Some(grand) = parent.parent().and_then(|g| g.parent()) {
                candidates.push(grand.join("cards"));
            }
            if let Some(gg) = parent.parent().and_then(|g| g.parent()).and_then(|g| g.parent()) {
                candidates.push(gg.join("cards"));
            }
        }
    }
    if let Some(data) = app.path().app_data_dir().ok() {
        candidates.push(data.join("cards"));
    }

    for dir in &candidates {
        if !dir.exists() {
            continue;
        }
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                if entry.file_name().to_string_lossy().to_lowercase() == "operator.xml" {
                    return Some(entry.path());
                }
            }
        }
    }
    None
}

/// Resolve the `codex/` directory (Codex v1, 2026-07-14). Mirrors
/// [`resolve_card_path`] — same 5-candidate walk — but joins `"codex"` and
/// returns the *directory* (not a single file), since the codex dir holds a
/// set of `*.md` files. Returns `None` if no `codex/` dir exists in any
/// candidate location (graceful — the Codex is optional; the seed loader
/// treats a missing dir as "nothing to seed").
fn resolve_codex_dir(app: &tauri::AppHandle) -> Option<std::path::PathBuf> {
    let mut candidates: Vec<std::path::PathBuf> = Vec::new();
    if let Some(d) = app.path().resource_dir().ok() {
        candidates.push(d.join("codex"));
    }
    if let Some(exe) = std::env::current_exe().ok() {
        if let Some(parent) = exe.parent() {
            candidates.push(parent.join("codex"));
            if let Some(grand) = parent.parent().and_then(|g| g.parent()) {
                candidates.push(grand.join("codex"));
            }
            if let Some(gg) = parent.parent().and_then(|g| g.parent()).and_then(|g| g.parent()) {
                candidates.push(gg.join("codex"));
            }
        }
    }
    if let Some(data) = app.path().app_data_dir().ok() {
        candidates.push(data.join("codex"));
    }

    for dir in &candidates {
        if dir.is_dir() {
            tracing::info!("resolved codex dir: {}", dir.display());
            return Some(dir.clone());
        }
    }
    None
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

    // ── State-Delta queue (invisible, Component D) ─────────────────────
    // If a background schema delta pass is still in flight from the PREVIOUS
    // turn, await it before doing anything else. To the user this looks like
    // normal thinking time — the frontend gets no signal until the first chunk
    // arrives, so a pre-stream delay is indistinguishable from model latency.
    // The await resolves when the delta task completes (success or failure);
    // the schema is already updated in AppState by the task before it exits.
    // Errors are ignored — schema is best-effort, a failed delta must not
    // block chat (the schema stays at its last-good state).
    if let Some(handle) = state.pending_delta.lock().await.take() {
        let _ = handle.await;
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
    let memory_block = {
        // Read the active card id once (cheap clone of a short string) so the
        // search and both archive calls below use the same scope within a turn.
        let card_id = state
            .active_card_id
            .lock()
            .expect("active_card_id mutex")
            .clone();
        match state.memory.get() {
            Some(engine) => match engine.search(&text, &card_id, 5, None).await {
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
        }
    };

    // ── Echo-skip signal (Codex v1, 2026-07-14) ──────────────────────────
    // Capture BEFORE `memory_block` is moved into `.stream()` below. If the
    // block contained a Codex reference, the post-turn archiver skips saving
    // the assistant's reply (which would otherwise echo authored lore back
    // into retrieval — the self-contamination loop, §2N landmine #5). The
    // marker is shared with `render_memory_block` via `CODEX_FRAME_MARKER`.
    let codex_was_injected = memory_block
        .as_deref()
        .map(|b| b.contains(memory::CODEX_FRAME_MARKER))
        .unwrap_or(false);

    // ── World-state schema injection (Component D) ──────────────────────
    // Render the current schema into the inter-turn region as a sibling
    // annotation to memory_block. `render_for_prompt()` returns "" for an
    // empty schema → we pass None → no <world_state> block on the first turn
    // (before any deltas have landed). Same empty-skip pattern as memory.
    // The schema is read here (before the session lock) so the chat engine
    // sees the state as of turn-start; any delta fired by the PREVIOUS turn
    // has already landed via the pending_delta await above.
    let world_state = {
        let s = state.schema.lock().await;
        let rendered = s.render_for_prompt();
        if rendered.is_empty() { None } else { Some(rendered) }
    };
    // Persona: rendered once per turn from the active Simulation Card. The
    // card is immutable after setup, so the rendered string is byte-identical
    // across turns → the persona block in the system prompt is stable and does
    // NOT trigger the §2F cold-reset guard (only the inter-turn memory block
    // does, by design). The fallback card renders to "" → section suppressed.
    let persona = state
        .active_card
        .get()
        .map(|c| c.render_for_prompt());
    // Operator profile: re-read FRESH from disk each turn (hot-reload). The
    // path is cached (stable); only the content refreshes — so a live edit to
    // Operator.xml takes effect on the very next message. `load` returns None
    // on missing/malformed → section silently suppressed (graceful). Like the
    // persona, the rendered text is byte-identical across turns until the file
    // is edited → no cold-reset (cache-friendly, Prime Directive).
    let user_profile = user_profile::load(
        state.operator_path.get().and_then(std::option::Option::as_deref),
    )
    .map(|p| p.render_for_prompt());
    let system_prompt =
        prompts::build_system_content(&settings, persona.as_deref(), user_profile.as_deref());

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
            .stream(messages, memory_block, world_state, settings.context_size, on_chunk, cancel.clone())
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
        match echo.stream(messages, None, None, settings.context_size, on_chunk, cancel.clone()).await {
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

    // Bug #3 Step 4: hold the raw model output alongside the cleaned content +
    // reasoning so the formatter can re-render cache-coherently next turn (no
    // full re-prefill of the previous reply). Session is ephemeral now
    // (2026-07-14) — no save; the turn lives only in memory for this launch.
    {
        let mut s = state.session.lock().await;
        s.add_assistant_turn(
            result.content.clone(),
            result.reasoning.clone(),
            result.raw.clone(),
        );

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
        //
        // ── Echo-skip (Codex v1, 2026-07-14) ──────────────────────────────
        // `codex_was_injected` was captured before `memory_block` was moved
        // into `.stream()`. If true, skip archiving the assistant's reply —
        // it's a paraphrase of authored Codex lore, and saving it would
        // pollute retrieval with echoes of the Codex itself (the self-
        // contamination loop, §2N landmine #5).
        if codex_was_injected {
            tracing::debug!("codex echo-skip: archiving suppressed (codex reference was injected this turn)");
        }
        if !codex_was_injected {
        if let Some(engine) = state.memory.get() {
            // The user message is the second-to-last (last is the assistant
            // turn we just appended). checked_sub(2) guards the cold-start
            // edge where messages is unexpectedly short.
            let user_text = s.messages.len().checked_sub(2).and_then(|i| s.messages.get(i)).map(|m| m.content.clone());
            let asst_text = result.content.clone();
            let card_id = state
                .active_card_id
                .lock()
                .expect("active_card_id mutex")
                .clone();
            let engine = Arc::clone(engine);
            tokio::spawn(async move {
                if let Some(text) = user_text {
                    if let Err(e) = engine.add_memory(text, &card_id, memory::Role::User, 1.0).await {
                        tracing::warn!(error = %format!("{e:#}"), "archive user turn failed");
                    }
                }
                if let Err(e) = engine.add_memory(asst_text, &card_id, memory::Role::Assistant, 1.0).await {
                    tracing::warn!(error = %format!("{e:#}"), "archive assistant turn failed");
                }
            });
        }
        } // end echo-skip gate (if !codex_was_injected)
    }

    // ── State-Delta fire (Component D) ──────────────────────────────────
    // Fire the background schema delta pass for the turn that just completed.
    // Mirrors the memory archive spawn above: detached, best-effort, errors
    // logged-and-dropped. The handle is stored in pending_delta so the NEXT
    // chat_send awaits it (the invisible queue) before reading the schema —
    // guaranteeing the next turn sees this turn's schema update.
    //
    // The delta pass runs on the dedicated wupi-schema thread (isolated
    // context, never touches the chat KV cache). The JoinHandle wraps the
    // post-generation work: post the request, await the reply via
    // spawn_blocking, apply the delta, persist. If the schema engine isn't
    // available (init failed, or chat proceeded in echo mode), skip silently.
    if let Some(schema_engine) = state.schema_engine.get() {
        // Capture the exchange from the session (clean strings, same source
        // as the memory archive — sidesteps the token-boundary-drift landmine
        // the same way). Read inside a brief lock, clone out, then drop the
        // guard before spawning so the task doesn't pin the session mutex.
        let (user_text, asst_text) = {
            let s = state.session.lock().await;
            let user = s.messages.len().checked_sub(2).and_then(|i| s.messages.get(i)).map(|m| m.content.clone());
            (user, result.content.clone())
        };
        // ── Content gate (M2, 2026-07-14) ────────────────────────────────
        // The delta pass is a full 12B forward pass. Skip it for clearly non-
        // substantive turns (short filler like "ok"/"thanks", or empty replies)
        // — see `should_fire_delta` for the conservative heuristic. 99% of real
        // turns still fire; the user's typing time masks the generation cost.
        // A skipped turn leaves pending_delta empty, so the next chat_send
        // doesn't wait — zero latency hit for filler turns.
        let user_text_for_gate = user_text.as_deref().unwrap_or("");
        if !schema_engine::should_fire_delta(user_text_for_gate, &asst_text) {
            tracing::debug!(
                user_words = user_text_for_gate.split_whitespace().count(),
                "schema delta skipped by content gate (non-substantive turn)"
            );
        } else {
            let current_schema = state.schema.lock().await.clone();
            let schema_engine = Arc::clone(schema_engine);
            let schema_slot = state.schema.clone();
            let handle = tokio::spawn(async move {
                // Post the delta request. The reply comes back on a std::mpsc
                // channel (the schema thread is a bare std::thread), so we await
                // it via spawn_blocking — same pattern as the chat engine reply.
                let reply_rx = match schema_engine
                    .request_delta((user_text.unwrap_or_default(), asst_text), &current_schema)
                {
                    Ok(rx) => rx,
                    Err(e) => {
                        tracing::warn!(error = %format!("{e:#}"), "schema delta request failed; schema unchanged");
                        return;
                    }
                };
                let reply = match tokio::task::spawn_blocking(move || reply_rx.recv()).await {
                    Ok(Ok(r)) => r,
                    Ok(Err(e)) => {
                        tracing::warn!(error = %format!("{e}"), "schema delta reply channel closed");
                        return;
                    }
                    Err(e) => {
                        tracing::warn!(error = %format!("{e}"), "schema delta reply join failed");
                        return;
                    }
                };
                if let Some(delta) = reply.delta {
                    let mut s = schema_slot.lock().await;
                    s.apply_delta(delta);
                    tracing::debug!("schema delta applied (in-memory; ephemeral)");
                } else if !reply.error.is_empty() {
                    tracing::warn!(error = %reply.error, "schema delta produced no delta (parse/generation failure); schema unchanged");
                }
            });
            *state.pending_delta.lock().await = Some(handle);
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

async fn rollback_last_user_message(state: &tauri::State<'_, AppState>, _app: &tauri::AppHandle) {
    // Pop the orphaned user message on generation failure so the next send
    // doesn't see two consecutive user turns (Bug C, §2D). Session is
    // ephemeral now (2026-07-14) — no disk save, just in-memory correction.
    // The `_app` param is retained for signature stability (callers pass it).
    let mut s = state.session.lock().await;
    if s.last_message_is_user() {
        s.pop_last_message();
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
/// hybrid FTS5 + vec0 search, and returns the score-aware-RRF-fused ranked
/// results with raw dense cosine + per-list ranks per hit. Off the chat path
/// entirely — this is the observability surface for tuning retrieval
/// independently of generation, AND the calibration surface for
/// [`memory_rrf::DENSE_COSINE_FLOOR`] (AGENTS.md §2M Checkpoint E).
///
/// `top_k` defaults to 10 when `None`. `dense_floor` overrides the const for
/// live calibration — pass a value to see how the result set changes at that
/// threshold without a rebuild; leave `None` to use the compiled default.
/// Returns an error string (not a panic) if the memory engine isn't
/// initialized or the query fails — the panel renders it as a red message.
///
/// Retrieval is scoped to the active card id (AGENTS.md §2M) — cards never
/// see each other's memory.
#[tauri::command]
async fn debug_memory_query(
    query: String,
    top_k: Option<usize>,
    dense_floor: Option<f32>,
    state: tauri::State<'_, AppState>,
) -> Result<Vec<memory::RankedMemory>, String> {
    let engine = state
        .memory
        .get()
        .ok_or_else(|| "memory engine not initialized".to_string())?;
    let card_id = state
        .active_card_id
        .lock()
        .expect("active_card_id mutex")
        .clone();
    engine
        .search(&query, &card_id, top_k.unwrap_or(10), dense_floor)
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
///
/// UNUSED since 2026-07-14 (ephemeral sessions). Retained for the future
/// character/simulation card system, which will re-introduce SCOPED
/// persistence (a card carries its own resumable session). The atomic-save
/// machinery is tested infrastructure — don't rebuild it when cards land.
#[allow(dead_code)]
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

/// Persist the world-state schema off the Tokio worker pool. Mirrors
/// `save_session`: `WorldSchema::save` is atomic (temp + fsync + rename) but
/// synchronous, so `spawn_blocking` keeps the async runtime free.
///
/// UNUSED since 2026-07-14 (ephemeral schema). Retained for the future
/// character/simulation card system alongside `save_session`.
#[allow(dead_code)]
async fn save_schema(app: &tauri::AppHandle, schema: &schema::WorldSchema) {
    use tauri::Manager;
    let Some(data_dir) = app.path().app_data_dir().ok() else {
        return;
    };
    let path = data_dir.join("world_schema.json");
    let schema = schema.clone();
    let _ = tokio::task::spawn_blocking(move || {
        if let Err(e) = schema.save(&path) {
            tracing::warn!(?e, "failed to persist world schema");
        }
    })
    .await;
}
