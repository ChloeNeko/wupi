pub mod api;
pub mod bracket_parser;
pub mod chat_format;
pub mod codex;
pub mod engine;
pub mod game_command;
pub mod game_engine;
#[cfg(windows)]
pub mod hardware;
pub mod kv_buffer;
pub mod llm;
pub mod memory;
pub mod memory_embedder;
pub mod memory_embedder_llama;
pub mod memory_rrf;
pub mod model_downloader;
pub mod narrator_prompt;
pub mod prompts;
pub mod schema;
pub mod schema_engine;
pub mod session;
pub mod sim_card;
pub mod stream_filter;
pub mod system_menu;
pub mod theme;
pub mod user_profile;

use std::sync::Arc;
use tauri::{Emitter, Manager};
use llm::GenerationClient;

/// The Memory engine's concrete embedder type, decided ONCE at startup. Using
/// `Box<dyn Embedder + Send + Sync>` lets `AppState` hold one concrete
/// `MemoryEngine` regardless of whether `Embed.gguf` was found: `LlamaCppEmbedder`
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
    /// The world-state schema: "the schema IS the summarizer." A persistent,
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
    /// The schema delta engine. Held under a resettable Mutex<Option<...>> so
    /// the model-swap code (api_connect/api_disconnect, chunk 4b) can tear it
    /// down + respawn it on a different model (WUPI.gguf ↔ Agent.gguf). Was
    /// OnceLock before the API feature; OnceLock can't be reset, which blocked
    /// the swap. None = not running (chat proceeds without schema deltas).
    pub schema_engine: Arc<std::sync::Mutex<Option<Arc<schema_engine::SchemaEngine>>>>,
    /// The active simulation card's id: the partition key for Memory
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
    /// is re-read fresh each `chat_send` (hot-reload: see `user_profile`).
    /// Lock-free reads after `setup`. Held as `Option<PathBuf>` so a missing
    /// profile is `None`, distinct from "not yet resolved."
    pub operator_path: Arc<std::sync::OnceLock<Option<std::path::PathBuf>>>,
    /// The resolved `docs/` directory (Codex lore library; renamed from
    /// `codex/` 2026-07-17). Filled once in setup; the codex_* IPC commands
    /// read/write `.md` files here. `None` when no docs/ dir resolved: the
    /// Codex UI shows empty.
    pub codex_dir: Arc<std::sync::OnceLock<Option<std::path::PathBuf>>>,
    /// The active theme + color code (defaults Aurora / Vibrant). Read by the
    /// frontend to paint the cascade panels; written by `theme_set`. Held
    /// under a std Mutex: never awaited across.
    pub theme: Arc<std::sync::Mutex<theme::ThemeSettings>>,
    /// The resolved path to `theme.json` in app data. Filled once in setup;
    /// `theme_set` saves to it. OnceLock because it needs the Tauri app handle
    /// to resolve app_data_dir (not available in AppState::new()).
    pub theme_path: Arc<std::sync::OnceLock<std::path::PathBuf>>,
    /// The API connection config (saved profiles + active source). Read by
    /// the `api_*` IPC commands; written by `api_profile_save`/`api_connect`/
    /// `api_disconnect`. Held under a std Mutex: short critical sections.
    pub api_config: Arc<std::sync::Mutex<api::ApiConfig>>,
    /// The resolved path to `api_config.json` in app data. Filled once in
    /// setup; the `api_*` IPC saves to it. OnceLock for the same reason as
    /// `theme_path`: needs the Tauri app handle to resolve app_data_dir.
    pub api_config_path: Arc<std::sync::OnceLock<std::path::PathBuf>>,
    /// The active chat source (`Local` = WUPI.gguf 12B, `Api` = HTTP endpoint).
    /// Mirrors `api_config.model_source` but held separately so `chat_send`
    /// reads it without locking the whole config (and so the swap logic can
    /// flip it atomically with the model teardown). Defaults to Local.
    pub model_source: Arc<std::sync::Mutex<api::ModelSource>>,

    // The GameEngine (narrator) lives here, NOT eagerly spawned at boot: it
    // spawns on `game_start` and shuts down on `game_end`. Costs VRAM only
    // while a game is actually running. Same shape as `schema_engine` (Mutex
    // of Option of Arc). None = no game running.
    pub game_engine: Arc<std::sync::Mutex<Option<Arc<game_engine::GameEngine>>>>,
    /// Per-game cancel token, parallel to `active_cancel`. Distinct slot so
    /// chat-stop and game-stop never cross-wire (Bug #7 pattern, §2C).
    pub active_game_cancel: Arc<std::sync::Mutex<Option<llm::CancelToken>>>,
    /// The game's scoped world-state schema (sibling to `schema`, which is
    /// Wupi-assistant's). Per-card: wiped/reloaded on card switch. Held
    /// under tokio Mutex because `game_send` reads it + Wupi's game-manager
    /// path writes it (via `game_command` deltas).
    pub game_schema: Arc<tokio::sync::Mutex<schema::WorldSchema>>,
    /// The game's scoped conversation (sibling to `session`, which is
    /// Wupi-assistant's). Per-card: loaded on `game_start` from
    /// `sessions/<card_id>.json`, saved on `game_end`. Held under tokio Mutex
    /// because `game_send` reads + writes it (windowing the narrator prompt +
    /// appending each turn). Phase 3 per-card persistence (AGENTS.md §2AA).
    pub game_session: Arc<tokio::sync::Mutex<session::Conversation>>,
    /// The active roleplay card. `None` when no game is running. Set on
    /// `game_start`, cleared on `game_end`. The narrator prompt builder
    /// reads this each `game_send` turn.
    pub active_game_card: Arc<std::sync::Mutex<Option<sim_card::SimCard>>>,
    /// The card id BEFORE a game started, so `game_end` can restore it. The
    /// system card (`__wupi_os__`) is the default; games swap to the
    /// roleplay card's id and restore on exit.
    pub pre_game_card_id: Arc<std::sync::Mutex<String>>,
    /// First-run GGUF download progress (see `model_downloader.rs`). Polled
    /// by `get_download_progress` and emitted as the `download-progress`
    /// event. Held under a std Mutex: short critical sections only, never
    /// awaited across (the download task itself runs on a tokio task and
    /// briefly locks to update fields between awaits).
    pub download_progress: Arc<std::sync::Mutex<model_downloader::DownloadProgress>>,
    /// Cancel token for an in-flight first-run download. Signaled by
    /// `cancel_download`; read at the top of each chunk in the download loop
    /// (same `Ordering::Relaxed` invariant as the engine decode loop, §3).
    pub download_cancel: Arc<std::sync::Mutex<Option<model_downloader::CancelToken>>>,
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
            schema_engine: Arc::new(std::sync::Mutex::new(None)),
            active_card_id: Arc::new(std::sync::Mutex::new(
                memory::WUPI_OS_CARD_ID.to_owned(),
            )),
            active_card: Arc::new(std::sync::OnceLock::new()),
            operator_path: Arc::new(std::sync::OnceLock::new()),
            codex_dir: Arc::new(std::sync::OnceLock::new()),
            theme: Arc::new(std::sync::Mutex::new(theme::ThemeSettings::default())),
            theme_path: Arc::new(std::sync::OnceLock::new()),
            api_config: Arc::new(std::sync::Mutex::new(api::ApiConfig::default())),
            api_config_path: Arc::new(std::sync::OnceLock::new()),
            model_source: Arc::new(std::sync::Mutex::new(api::ModelSource::default())),
            game_engine: Arc::new(std::sync::Mutex::new(None)),
            active_game_cancel: Arc::new(std::sync::Mutex::new(None)),
            game_schema: Arc::new(tokio::sync::Mutex::new(schema::WorldSchema::default())),
            game_session: Arc::new(tokio::sync::Mutex::new(session::Conversation::new())),
            active_game_card: Arc::new(std::sync::Mutex::new(None)),
            pre_game_card_id: Arc::new(std::sync::Mutex::new(memory::WUPI_OS_CARD_ID.to_owned())),
            download_progress: Arc::new(std::sync::Mutex::new(
                model_downloader::DownloadProgress::default(),
            )),
            download_cancel: Arc::new(std::sync::Mutex::new(None)),
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
        // Updater: signed-update distribution. The JS side drives check() +
        // downloadAndInstall(); process plugin provides relaunch() after
        // install. Both plugins MUST be registered before .setup() so the
        // IPC surface is ready when the frontend loads.
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_process::init())
        .manage(AppState::new())
        .manage(hardware::AudioRegistry)
        .setup(|app| {
            tracing::info!("setup hook entered");
            let data_dir = app
                .path()
                .app_data_dir()
                .expect("app data dir is available");
            std::fs::create_dir_all(&data_dir).ok();
            // Phase 3 per-card persistence: the sessions/ and schemas/ subdirs
            // hold `<card_id>.json` files for each roleplay card that's been
            // played. Created once at boot; cheap no-op if they already exist.
            std::fs::create_dir_all(data_dir.join("sessions")).ok();
            std::fs::create_dir_all(data_dir.join("schemas")).ok();
            tracing::info!("app data dir: {}", data_dir.display());

            let state: tauri::State<AppState> = app.state();

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

            // Same pattern as theme: resolve path → load → cache path on
            // AppState so the api_* IPC commands don't need the app handle.
            // model_source is restored here; the actual model swap (if it was
            // Api at last shutdown) is re-performed later in setup once the
            // local model has finished loading, NOT here: we can't swap
            // models before the local model has loaded.
            {
                let api_path = api::ApiConfig::resolve_path(&data_dir);
                let loaded = api::ApiConfig::load(&api_path);
                tracing::info!(
                    profiles = loaded.profiles.len(),
                    source = ?loaded.model_source,
                    active = ?loaded.active_profile_id,
                    "api config loaded"
                );
                // Sync model_source to the loaded value (both track the same
                // thing; model_source is the fast-read copy for chat_send).
                *state.model_source.lock().expect("model_source mutex") = loaded.model_source;
                *state.api_config.lock().expect("api_config mutex") = loaded;
                let _ = state.api_config_path.set(api_path);
            }

            // WUPI OS launches into a FRESH session every time: no
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

            // Load the default card (`cards/Wupi.sim`) before anything else -
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

            // Resolve the operator's profile path (`cards/Operator.xml`) once
            // and cache it. The CONTENT is re-read fresh each chat_send
            // (hot-reload: a live edit takes effect on the very next message,
            // no reboot); only the PATH is stable. `None` when no profile
            // exists: the common case until the operator authors one. Wupi
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
                let backend = llm::LlamaCppBackend::spawn_load(path.clone(), 99, context_size, Box::new(move |result| {
                    match &result {
                        Ok(name) => {
                            let _ = app_handle.emit(
                                "model-status",
                                serde_json::json!({ "status": "ready", "model": name }),
                            );

                            // The schema engine loads its OWN model now (no
                            // shared_model() coupling). In Local mode it reuses
                            // the same WUPI.gguf path the chat engine just
                            // loaded. Spawned here (after chat model ready) so
                            // the two loads don't compete for VRAM during boot;
                            // runs on the loader thread, blocking recv is fine.
                            let app_state = app_handle.state::<AppState>();
                            let already = app_state
                                .schema_engine
                                .lock()
                                .map(|g| g.is_some())
                                .unwrap_or(false);
                            if !already {
                                let (engine, init_rx) = schema_engine::SchemaEngine::spawn_load(
                                    path.clone(),
                                    99,
                                );
                                match init_rx.recv() {
                                    Ok(Ok(())) => {
                                        tracing::info!(
                                            "schema engine ready (eager spawn at model-ready)"
                                        );
                                        if let Ok(mut slot) = app_state.schema_engine.lock() {
                                            *slot = Some(Arc::new(engine));
                                        }

                                        // If the user was on an API profile at last
                                        // shutdown, boot brought the 12B up as a safe
                                        // default. Now that both engines are ready,
                                        // re-perform the API swap so Wupi comes back
                                        // up on the same connection the user last had.
                                        // The schema engine stays on WUPI.gguf either
                                        // way (no Agent.gguf dependency). On any error
                                        // we stay on local 12B: boot must never fail.
                                        let restore = {
                                            let cfg = app_state
                                                .api_config
                                                .lock()
                                                .expect("api_config mutex");
                                            (
                                                cfg.model_source,
                                                cfg.active_profile_id.clone(),
                                            )
                                        };
                                        if matches!(restore.0, api::ModelSource::Api) {
                                            if let Some(profile_id) = restore.1 {
                                                tracing::info!(
                                                    profile_id = %profile_id,
                                                    "boot: restoring last-used API connection"
                                                );
                                                // Tear down the freshly-loaded 12B
                                                // chat backend (schema stays put).
                                                let taken = app_state
                                                    .backend
                                                    .lock()
                                                    .expect("backend mutex")
                                                    .take();
                                                if let Some(b) = taken {
                                                    b.shutdown();
                                                }
                                                *app_state
                                                    .model_source
                                                    .lock()
                                                    .expect("model_source mutex") =
                                                    api::ModelSource::Api;
                                                let _ = app_handle.emit(
                                                    "model-status",
                                                    serde_json::json!({
                                                        "status": "ready",
                                                        "model": "api (restored)",
                                                    }),
                                                );
                                                tracing::info!(
                                                    "boot: API connection restored"
                                                );
                                            } else {
                                                // model_source was Api but no active
                                                // profile: downgrade to Local so
                                                // chat_send doesn't route to a
                                                // non-existent API path.
                                                tracing::warn!(
                                                    "boot: model_source was Api but no \
                                                     active profile; downgrading to Local"
                                                );
                                                *app_state
                                                    .model_source
                                                    .lock()
                                                    .expect("model_source mutex") =
                                                    api::ModelSource::Local;
                                                let mut cfg = app_state
                                                    .api_config
                                                    .lock()
                                                    .expect("api_config mutex");
                                                cfg.model_source =
                                                    api::ModelSource::Local;
                                            }
                                        }
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
                // No GGUF found. On a fresh install (the beta-tester path),
                // this means the first-run downloader hasn't run yet — emit
                // "missing" so the frontend shows the download overlay instead
                // of silently booting into echo mode. The frontend's overlay
                // then drives `download_models`; on completion it triggers a
                // reload that re-enters setup with the now-present WUPI.gguf.
                // If the user explicitly dismissed the download (or it failed
                // unrecoverably), `download_models` leaves the state as-is
                // and the title indicator falls back to "offline".
                tracing::info!("no model file found; emitting 'missing' for first-run downloader");
                let app_handle = app.handle().clone();
                let _ = app_handle.emit(
                    "model-status",
                    serde_json::json!({ "status": "missing" }),
                );
            }

            // Build the MemoryEngine with the real BERT embedder if
            // `Embed.gguf` is on disk; fall back to StubEmbedder otherwise
            // (graceful degradation: documented contract in
            // memory_embedder_llama.rs::resolve_embed_model). The embedder is
            // boxed into `Box<dyn Embedder + Send + Sync>` so AppState holds
            // one concrete type regardless of which backend was chosen.
            //
            // `shared_backend()` (§2H) is the single `LlamaBackend::init()`
            // chokepoint: both the chat loader (above) and the embedder route
            // through it. The embedder thread does NOT block on chat-model
            // loading: `shared_backend` is a `OnceLock` that resolves on first
            // call; whichever loader hits it first inits, the other reuses.
            let embedder: DynEmbedder = match resolve_embed_model_dirs(app.handle()) {
                Some(path) => {
                    tracing::info!("spawning embed model load: {}", path.display());
                    let (embedder, init_rx) =
                        memory_embedder_llama::LlamaCppEmbedder::spawn_load(path, 99);
                    // Block on the readiness channel: same contract as the
                    // chat engine's Bug #6 fix. If init failed, fall back to
                    // the stub so the app still runs (memory just won't be
                    // semantic). This recv runs on the setup thread, which is
                    // fine: setup is allowed to block.
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

                    // Reconcile authored `.md` files in `docs/` against the
                    // Codex-tagged entries already stored in memory.sqlite.
                    // Idempotent (hash-based): re-runs against an unchanged
                    // source set do zero writes. Best-effort: a failed seed
                    // is logged-and-dropped, never fatal (same contract as the
                    // embedder fallback). Runs synchronously here (setup is
                    // allowed to block: it already blocks on the embedder
                    // readiness channel above).
                    // (1) User-authored codex from `docs/` → CODEX_CARD_ID. The user's
                    //     blank slate: empty by default, populated only via the
                    //     codex_* IPC. Pinned to CODEX_CARD_ID (not active_card_id)
                    //     so editing lore during a game lands in the user's namespace,
                    //     NOT the active roleplay card (the pre-Phase-2 bug).
                    // (2) Wupi's non-editable system knowledge from
                    //     `cards/wupi_knowledge/` → WUPI_SYSTEM_CARD_ID. The firewall:
                    //     no user IPC writes here; only this boot seed does. Wupi
                    //     reads it cross-card via search_wupi_visible regardless of
                    //     which roleplay card is active.
                    if let Some(codex_dir) = resolve_codex_dir(app.handle()) {
                        // Cache the resolved path for the codex_* IPC (file CRUD).
                        let _ = state.codex_dir.set(Some(codex_dir.clone()));
                        if let Some(engine) = state.memory.get() {
                            match tauri::async_runtime::block_on(
                                codex::seed_codex(engine, &codex_dir, memory::CODEX_CARD_ID, "codex"),
                            ) {
                                Ok(report) => tracing::info!(
                                    seeded = report.seeded,
                                    updated = report.updated,
                                    purged = report.purged,
                                    unchanged = report.unchanged,
                                    "user codex seeded"
                                ),
                                Err(e) => tracing::warn!(
                                    error = %format!("{e:#}"),
                                    "user codex seed failed; continuing without authored lore"
                                ),
                            }
                        }
                    } else {
                        tracing::info!("no docs/ dir found; skipping user codex seed");
                    }

                    // Wupi-system seed: her own OS docs (the firewall's read-only side).
                    if let Some(wupi_knowledge_dir) = resolve_wupi_knowledge_dir(app.handle()) {
                        if let Some(engine) = state.memory.get() {
                            match tauri::async_runtime::block_on(
                                codex::seed_codex(engine, &wupi_knowledge_dir, memory::WUPI_SYSTEM_CARD_ID, "wupi_system"),
                            ) {
                                Ok(report) => tracing::info!(
                                    seeded = report.seeded,
                                    updated = report.updated,
                                    purged = report.purged,
                                    unchanged = report.unchanged,
                                    "wupi system knowledge seeded"
                                ),
                                Err(e) => tracing::warn!(
                                    error = %format!("{e:#}"),
                                    "wupi system knowledge seed failed; continuing"
                                ),
                            }
                        }
                    } else {
                        tracing::info!("no cards/wupi_knowledge/ dir found; skipping system knowledge seed");
                    }
                }
                Err(e) => {
                    // DB open failure is fatal for memory but must not kill
                    // the app. Leave the OnceLock empty; callers check `get`.
                    tracing::error!(error = %format!("{e:#}"), "memory engine init failed");
                }
            }

            // ── System tray (paw icon): installed once the app handle exists.
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
                    // Destroy the tray BEFORE exit so Windows receives NIM_DELETE
                    // while we're still alive (prevents ghost-icon caching).
                    system_menu::destroy_tray(&window.app_handle());
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
            memory_list,
            memory_update,
            memory_delete,
            memory_wipe_card,
            codex_list,
            codex_save,
            codex_delete,
            operator_profile_get,
            operator_profile_set,
            api_profiles_list,
            api_profile_save,
            api_profile_delete,
            api_profile_test,
            api_connect,
            api_disconnect,
            model_source_get,
            check_models,
            download_models,
            get_download_progress,
            cancel_download,
            game_cards_list,
            game_start,
            game_send,
            game_stop,
            game_end,
            system_menu::power_shutdown_cmd,
            system_menu::power_restart_cmd,
            system_menu::power_sleep_cmd,
            theme_get,
            theme_set,
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
        // Build the app, then run the event loop with a callback. Splitting
        // .build() + App::run(callback) — instead of Builder::run(context)
        // which takes no callback — gives us the RunEvent hook for
        // belt-and-suspenders tray cleanup. Builder::run exists too but
        // internally calls App::run with `|_, _| {}` (see tauri app.rs:2449).
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app_handle, event| {
            // Belt-and-suspenders tray cleanup. The dominant exit paths
            // (power_shutdown → std::process::exit, on_window_event close)
            // already destroy the tray explicitly. This catches any other
            // RunEvent::ExitRequested path (e.g. programmatic app.exit from
            // a future code path) so a ghost icon can never accumulate from
            // a graceful-exit route we didn't anticipate.
            if let tauri::RunEvent::ExitRequested { .. } = event {
                system_menu::destroy_tray(&app_handle);
            }
        });
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

/// Randomized boot greeting: picks one line from the active card's
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
    let candidates = model_search_dirs(app);
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

/// The candidate `models/` directories, in search order. Shared by the chat
/// model resolver, the embedder resolver, and the schema-engine model
/// resolver (so they all agree on where `.gguf` files live). Extracted from
/// `resolve_model_path` so the swap logic can resolve Agent.gguf / WUPI.gguf
/// by name against the same dirs.
fn model_search_dirs(app: &tauri::AppHandle) -> Vec<std::path::PathBuf> {
    use tauri::Manager;
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
    // implicitly: it isn't named WUPI and is far smaller than WUPI.gguf, so
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
/// model is present: the caller falls back to `StubEmbedder` (graceful, not a
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

// ── First-run GGUF downloader IPC ──────────────────────────────────────────
//
// Four commands power the boot download overlay:
//   - check_models            : are WUPI.gguf / Embed.gguf present?
//   - download_models         : stream both from HF into app_data_dir/models
//   - get_download_progress   : polled snapshot of the in-flight download
//   - cancel_download         : signal an in-flight download to stop
//
// The flow (driven from script.js's setupBootSplash gate):
//   1. setup emits `model-status: missing` when no gguf is found (lib.rs
//      setup() above).
//   2. script.js calls check_models to confirm, shows #download-overlay.
//   3. User clicks "Download" → download_models fires; the overlay subscri
//      cribes to `download-progress` events + polls get_download_progress.
//   4. On Done, script.js calls app_ready (which re-resolves the model and
//      triggers a reload via the existing model-status path).

/// Check whether the chat model + embedder model are present in any candidate
/// `models/` dir. Returns a JSON object the frontend uses to decide whether
/// to show the download overlay. Both-present ⇒ boot normally; either
/// missing ⇒ show the overlay (the downloader fetches BOTH regardless, so
/// the simple "either missing" gate is correct).
#[tauri::command]
fn check_models(app: tauri::AppHandle) -> serde_json::Value {
    let wupi = resolve_model_path(&app).is_some();
    let embed = resolve_embed_model_dirs(&app).is_some();
    serde_json::json!({
        "wupi": if wupi { "present" } else { "missing" },
        "embed": if embed { "present" } else { "missing" },
    })
}

/// The target `models/` dir for downloads. Always `app_data_dir/models` —
/// the 5th candidate in `model_search_dirs`, the only one that's reliably
/// user-writable on a fresh install (resource_dir is read-only after
/// install, exe-parent may be Program Files). Resolving it here keeps the
/// downloader module free of Tauri path APIs (testable in isolation).
fn download_target_dir(app: &tauri::AppHandle) -> Option<std::path::PathBuf> {
    use tauri::Manager;
    app.path().app_data_dir().ok().map(|d| d.join("models"))
}

/// Stream both GGUFs from HF into `app_data_dir/models`. Long-running; the
/// frontend subscribes to `download-progress` events (throttled to 2/sec)
/// and polls `get_download_progress` for the authoritative snapshot between
/// emits. Returns `Ok(())` when both files land; `Err(msg)` on any failure
/// (network, HTTP status, cancel). On cancel, the `.part` files are left in
/// place for the next attempt to resume from.
#[tauri::command]
async fn download_models(
    app: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    // Resolve target dir + mint a fresh cancel token. The cancel slot is
    // scoped into its own block so the MutexGuard drops before we await
    // (the `!Send` guard invariant, same as api_connect at lib.rs:2092).
    let dest_dir = download_target_dir(&app)
        .ok_or_else(|| "could not resolve app_data_dir/models".to_owned())?;
    let cancel = {
        let fresh = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut slot = state
            .download_cancel
            .lock()
            .expect("download_cancel mutex");
        *slot = Some(std::sync::Arc::clone(&fresh));
        fresh
    };

    // Reset progress to a clean Idle so a re-run after a prior failure
    // doesn't show stale totals.
    {
        let mut p = state.download_progress.lock().expect("progress mutex");
        *p = model_downloader::DownloadProgress::default();
    }

    let result = model_downloader::download_all(
        dest_dir,
        Arc::clone(&state.download_progress),
        cancel,
        app.clone(),
    )
    .await;

    // Clear the cancel slot regardless of outcome (no in-flight download
    // to cancel anymore).
    {
        let mut slot = state
            .download_cancel
            .lock()
            .expect("download_cancel mutex");
        *slot = None;
    }

    match result {
        Ok(()) => Ok(()),
        Err(e) => {
            // Mark progress Failed so the overlay can show the error. The
            // `.part` files are retained by the downloader for resume.
            {
                let mut p = state.download_progress.lock().expect("progress mutex");
                if p.phase != model_downloader::DownloadPhase::Done {
                    p.phase = model_downloader::DownloadPhase::Failed;
                    p.error = e.clone();
                }
            }
            let _ = app.emit(
                "download-progress",
                state.download_progress.lock().expect("progress mutex").clone(),
            );
            Err(e)
        }
    }
}

/// Polled snapshot of download progress. The frontend calls this on a timer
/// (e.g. every 250ms) as the authoritative source between throttled
/// `download-progress` events. Cheaper and more reliable than relying on
/// catching every emitted event (events can coalesce or drop under load).
#[tauri::command]
fn get_download_progress(state: tauri::State<'_, AppState>) -> model_downloader::DownloadProgress {
    state
        .download_progress
        .lock()
        .expect("progress mutex")
        .clone()
}

/// Cancel an in-flight download. The download loop checks the token at the
/// top of each chunk and exits with `Err("cancelled")`; the `.part` files
/// stay on disk for the next attempt to resume from. No-op if no download
/// is running.
#[tauri::command]
fn cancel_download(state: tauri::State<'_, AppState>) {
    if let Some(token) = state
        .download_cancel
        .lock()
        .expect("download_cancel mutex")
        .as_ref()
    {
        token.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Resolve the default Simulation Card (`cards/Wupi.sim`) by walking the same
/// candidate-dir list as [`resolve_model_path`], but joining `"cards"` instead
/// of `"models"` and exact-matching `Wupi.sim` (case-insensitive). Locked-name
/// single file: no size fallback (only one file will ever be named
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
/// dir. Returns `None` when no profile is found: the common case until the
/// operator authors one; the caller runs without a `<user_profile>` section
/// (graceful, not a crash).
///
/// Only the PATH is resolved here (once, in setup). The CONTENT is re-read
/// fresh each `chat_send` via `user_profile::load`: that's the hot-reload
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

/// Resolve the `docs/` directory: the Codex lore source (renamed from
/// `codex/` 2026-07-17, after the `.md` files moved there). Mirrors
/// [`resolve_card_path`]: same 5-candidate walk - but joins `"docs"` and
/// returns the *directory* (not a single file), since it holds a set of
/// `*.md` files. Returns `None` if no `docs/` dir exists in any candidate
/// location (graceful: the Codex is optional; the seed loader treats a
/// missing dir as "nothing to seed").
fn resolve_codex_dir(app: &tauri::AppHandle) -> Option<std::path::PathBuf> {
    let mut candidates: Vec<std::path::PathBuf> = Vec::new();
    if let Some(d) = app.path().resource_dir().ok() {
        candidates.push(d.join("docs"));
    }
    if let Some(exe) = std::env::current_exe().ok() {
        if let Some(parent) = exe.parent() {
            candidates.push(parent.join("docs"));
            if let Some(grand) = parent.parent().and_then(|g| g.parent()) {
                candidates.push(grand.join("docs"));
            }
            if let Some(gg) = parent.parent().and_then(|g| g.parent()).and_then(|g| g.parent()) {
                candidates.push(gg.join("docs"));
            }
        }
    }
    if let Some(data) = app.path().app_data_dir().ok() {
        candidates.push(data.join("docs"));
    }

    for dir in &candidates {
        if dir.is_dir() {
            tracing::info!("resolved codex (docs/) dir: {}", dir.display());
            return Some(dir.clone());
        }
    }
    None
}

/// Resolve the `cards/wupi_knowledge/` directory: the home of Wupi's non-
/// editable system knowledge (the Phase 2 firewall's read-only seed source).
///
/// Mirrors [`resolve_codex_dir`]: same 5-candidate walk (resource_dir, exe
/// parent/grandparent/great-grandparent, app_data_dir), but joins
/// `cards/wupi_knowledge`. Returns `None` if no such dir exists (graceful -
/// the system knowledge is optional; the seed loader treats a missing dir as
/// "nothing to seed"). The path sits alongside `cards/Wupi.sim` and
/// `cards/game_cards/`: all three are bundled tracked text assets, not
/// user-writable state.
fn resolve_wupi_knowledge_dir(app: &tauri::AppHandle) -> Option<std::path::PathBuf> {
    let mut candidates: Vec<std::path::PathBuf> = Vec::new();
    if let Some(d) = app.path().resource_dir().ok() {
        candidates.push(d.join("cards").join("wupi_knowledge"));
    }
    if let Some(exe) = std::env::current_exe().ok() {
        if let Some(parent) = exe.parent() {
            candidates.push(parent.join("cards").join("wupi_knowledge"));
            if let Some(grand) = parent.parent().and_then(|g| g.parent()) {
                candidates.push(grand.join("cards").join("wupi_knowledge"));
            }
            if let Some(gg) = parent.parent().and_then(|g| g.parent()).and_then(|g| g.parent()) {
                candidates.push(gg.join("cards").join("wupi_knowledge"));
            }
        }
    }
    if let Some(data) = app.path().app_data_dir().ok() {
        candidates.push(data.join("cards").join("wupi_knowledge"));
    }

    for dir in &candidates {
        if dir.is_dir() {
            tracing::info!("resolved wupi_knowledge dir: {}", dir.display());
            return Some(dir.clone());
        }
    }
    None
}

// Three small functions: game_is_active checks whether a GameEngine is
// running; route_to_game_manager handles the MutateWorldState intent
// (translates the player's request to a SchemaDelta via the schema engine's
// isolated context, applies it, streams a confirmation); route_to_game_query
// handles QueryWorldState (returns a slice of the game's world-state schema).
//
// Both route helpers are invoked from the top of chat_send via an early
// return, so the existing Wupi-assistant chat body is never entered when a
// management intent is detected.

/// True when a GameEngine is currently running (a game is active). Cheap:
/// locks the Mutex briefly, checks for Some, drops the guard. Used at the top
/// of `chat_send` to decide whether to run the management-intent classifier.
fn game_is_active(state: &tauri::State<'_, AppState>) -> bool {
    state
        .game_engine
        .lock()
        .map(|g| g.is_some())
        .unwrap_or(false)
}

/// Handle a `MutateWorldState` intent: translate the player's natural-language
/// request into a `SchemaDelta` via the schema engine's isolated context,
/// apply it to the active game's scoped `game_schema`, and stream a
/// confirmation back through the same `on_event` Channel Wupi's chat uses.
///
/// The translation reuses `SchemaEngine::request_translation` (Phase E,
/// 2026-07-18): the same isolated context the auto-summarizer runs on, no
/// KV pollution to chat or narrator. The confirmation text is a template
/// filled from the actual delta (no LLM needed for the confirmation itself -
/// keeps the management path cheap).
///
/// Errors surface as a single `error` Channel event + an `Err` return, so the
/// UI can render them like any chat error. The active game's schema is left
/// unchanged on any failure path.
async fn route_to_game_manager(
    text: String,
    on_event: tauri::ipc::Channel<serde_json::Value>,
    state: &tauri::State<'_, AppState>,
) -> Result<(), String> {
    // 1. Pull the schema engine out under a brief lock. If it's not running
    //    (rare: eager-spawned at boot), surface an error rather than crash.
    let schema_engine = state
        .schema_engine
        .lock()
        .map_err(|e| format!("schema_engine mutex: {e}"))?
        .clone()
        .ok_or_else(|| "schema engine not running: cannot translate request".to_string())?;

    // 2. Snapshot the current game schema (the delta diffs against this).
    //    Clone out + drop the guard before the awaited translation call.
    let current_schema = state.game_schema.lock().await.clone();

    // 3. Post the translation request + await the reply off the tokio worker
    //    (the schema thread is a bare std::thread; its mpsc::Receiver blocks).
    let reply_rx = schema_engine
        .request_translation(text.clone(), &current_schema)
        .map_err(|e| format!("{e:#}"))?;
    let reply = tokio::task::spawn_blocking(move || reply_rx.recv())
        .await
        .map_err(|e| format!("translation reply join: {e}"))?
        .map_err(|e| format!("translation reply channel: {e}"))?;

    if !reply.error.is_empty() {
        on_event
            .send(serde_json::json!({
                "type": "error",
                "message": format!("couldn't translate that: {}", reply.error),
            }))
            .map_err(|e| e.to_string())?;
        return Ok(());
    }

    // 4. Apply the delta to the game schema. If the model emitted `{}` (no
    //    changes), the delta is empty: treat as "didn't understand, nothing
    //    changed" and confirm that specifically.
    let Some(delta) = reply.delta else {
        on_event
            .send(serde_json::json!({
                "type": "error",
                "message": "couldn't translate that into a state change".to_string(),
            }))
            .map_err(|e| e.to_string())?;
        return Ok(());
    };

    let delta_applied = delta.has_changes();
    if delta_applied {
        let mut s = state.game_schema.lock().await;
        s.apply_delta(delta.clone());
    }

    // 5. Build + stream the confirmation. The text is template-filled from
    //    the delta (no LLM call): keeps the management path cheap. The
    //    frontend renders it as a normal Wupi bubble via the same `chunk` +
    //    `done` event shape chat uses.
    let confirmation = if delta_applied {
        format_confirmation(&delta, &text)
    } else {
        "I couldn't turn that into a state change: try rephrasing? \
         For example: \"make it stormy\", \"set the weather to clear\", \
         \"give Alex a torch\".".to_string()
    };
    on_event
        .send(serde_json::json!({ "type": "chunk", "text": &confirmation }))
        .map_err(|e| e.to_string())?;
    on_event
        .send(serde_json::json!({
            "type": "done",
            "final_text": confirmation,
            "reasoning": "",
            "game_manager": true,
        }))
        .map_err(|e| e.to_string())?;
    tracing::info!(
        request = %text.chars().take(80).collect::<String>(),
        applied = delta_applied,
        "game-manager: mutation request handled"
    );
    Ok(())
}

/// Handle a `QueryWorldState` intent: return a slice of the active game's
/// world-state schema so Wupi can narrate it. The `focus` (e.g. "weather",
/// "inventory") is matched against the schema's entity keys; if nothing
/// matches, the whole schema is returned (so Wupi can still describe the
/// state of the world generally).
async fn route_to_game_query(
    focus: String,
    on_event: tauri::ipc::Channel<serde_json::Value>,
    state: &tauri::State<'_, AppState>,
) -> Result<(), String> {
    let snapshot = state.game_schema.lock().await.clone();
    let state_json = snapshot.to_json_pretty();

    // Best-effort focus match: look for entity keys containing the focus
    // substring. If none match, send the full schema.
    let focused = if focus.is_empty() {
        state_json.clone()
    } else {
        let lower = focus.to_lowercase();
        snapshot
            .entities
            .iter()
            .filter(|(k, _)| k.to_lowercase().contains(&lower))
            .map(|(k, v)| format!("  {k}: {v}"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let body = if focused.is_empty() {
        format!("Here's what I know about the world right now:\n{state_json}")
    } else {
        format!("Here's what I know about {focus}:\n{focused}")
    };

    // Emit two messages: the structured `game_state_query` (machine-readable,
    // for any future UI that wants to render state differently) + the
    // chunk/done pair Wupi's chat UI renders as a normal bubble.
    on_event
        .send(serde_json::json!({
            "type": "game_state_query",
            "focus": focus,
            "state": state_json,
        }))
        .map_err(|e| e.to_string())?;
    on_event
        .send(serde_json::json!({ "type": "chunk", "text": &body }))
        .map_err(|e| e.to_string())?;
    on_event
        .send(serde_json::json!({
            "type": "done",
            "final_text": body,
            "reasoning": "",
            "game_manager": true,
        }))
        .map_err(|e| e.to_string())?;
    tracing::info!(
        focus = %focus,
        "game-manager: query handled"
    );
    Ok(())
}

/// Build a short natural-language confirmation of a `SchemaDelta`. Avoids an
/// extra LLM call: the mutation translation already did the work; this just
/// narrates the result. Falls back to a generic "Done." if the delta has no
/// recognizable changes (the `has_changes` gate upstream should prevent that).
fn format_confirmation(delta: &schema::SchemaDelta, original_request: &str) -> String {
    let mut bits = Vec::new();
    if let Some(summary) = delta.summary.as_deref() {
        bits.push(format!("Summary updated: \"{summary}\""));
    }
    if let Some(events) = delta.recent_events.as_ref() {
        if !events.is_empty() {
            let preview = events.last().map(|s| s.as_str()).unwrap_or("");
            let preview: String = preview.chars().take(80).collect();
            bits.push(format!("Logged event: {preview}"));
        }
    }
    if let Some(ents) = delta.entities.as_ref() {
        for (k, v) in ents.iter() {
            match v {
                Some(val) => bits.push(format!("{k} → {val}")),
                None => bits.push(format!("{k} removed")),
            }
        }
    }
    if bits.is_empty() {
        format!("Done: \"{}\" applied.", original_request)
    } else {
        format!("Done! {}", bits.join("; "))
    }
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

    // When a game is active, check whether the player's message to Wupi is a
    // game-management intent (mutate world state / query world state). If so,
    // route to the dedicated handlers and RETURN EARLY: the existing chat
    // body is never entered, so Wupi-assistant chat behavior is unchanged
    // when no game is active OR when the message isn't management-shaped.
    // See docs/games-app-design.md §1.4 + game_command.rs for the heuristic.
    if game_is_active(&state) {
        match game_command::classify(&text) {
            game_command::GameCommand::MutateWorldState(_) => {
                clear_active_cancel(&state);
                return route_to_game_manager(text, on_event, &state).await;
            }
            game_command::GameCommand::QueryWorldState(focus) => {
                clear_active_cancel(&state);
                return route_to_game_query(focus, on_event, &state).await;
            }
            game_command::GameCommand::NotACommand => {
                // Fall through to normal Wupi-assistant chat.
            }
        }
    }

    // If a background schema delta pass is still in flight from the PREVIOUS
    // turn, await it before doing anything else. To the user this looks like
    // normal thinking time: the frontend gets no signal until the first chunk
    // arrives, so a pre-stream delay is indistinguishable from model latency.
    // The await resolves when the delta task completes (success or failure);
    // the schema is already updated in AppState by the task before it exits.
    // Errors are ignored: schema is best-effort, a failed delta must not
    // block chat (the schema stays at its last-good state).
    if let Some(handle) = state.pending_delta.lock().await.take() {
        let _ = handle.await;
    }

    let settings = state.settings.lock().expect("settings mutex").clone();

    // Embed the user's just-typed text and pull top hits BEFORE the session
    // lock. This is ON the chat path by design (§3A): embedding takes ms on
    // GPU, the SQLite work is spawn_blocking-internal. The just-typed message
    // isn't archived yet (pillar 2 archives after generation), so we never
    // retrieve the thing we're about to send.
    //
    // §2F cost: the retrieved block differs per query → the prompt structure
    // changes every turn → the structural-divergence guard (engine.rs) cold-
    // resets the KV cache. Delta-prefill is dead on Memory-enabled turns. This
    // is the accepted v1 cost; the cache-layout optimization is a later pass.
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
            // Phase 2 firewall: Wupi-as-assistant retrieves from BOTH the
            // active card AND her reserved system-knowledge partition
            // (WUPI_SYSTEM_CARD_ID) via search_wupi_visible. She always knows
            // her own OS docs regardless of which card is active. Roleplay
            // cards never see each other: only system knowledge leaks through.
            Some(engine) => match engine.search_wupi_visible(&text, &card_id, 5, None).await {
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

    // Capture BEFORE `memory_block` is moved into `.stream()` below. If the
    // block contained a Codex reference, the post-turn archiver skips saving
    // the assistant's reply (which would otherwise echo authored lore back
    // into retrieval: the self-contamination loop, §2N landmine #5). The
    // marker is shared with `render_memory_block` via `CODEX_FRAME_MARKER`.
    let codex_was_injected = memory_block
        .as_deref()
        .map(|b| b.contains(memory::CODEX_FRAME_MARKER))
        .unwrap_or(false);

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
    // path is cached (stable); only the content refreshes: so a live edit to
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
    // Model-source dispatch: API path constructs a fresh HttpBackend from the
    // active profile; Local path uses the persistent LlamaCppBackend (or
    // EchoBackend if no local model loaded). The memory_block + world_state
    // are folded into the system message by the HttpBackend (APIs take a flat
    // messages list, not the inter-turn splice the local backend uses).
    let source = *state.model_source.lock().expect("model_source mutex");
    let result = if source == api::ModelSource::Api {
        // Active profile must exist (api_connect validates + sets it). If it's
        // somehow missing (e.g. config edited mid-session), fall through to
        // the local path rather than crashing.
        let profile_opt = {
            let cfg = state.api_config.lock().expect("api_config mutex");
            cfg.active_profile().cloned()
        };
        match profile_opt {
            Some(profile) => {
                let http = llm::HttpBackend::new(profile);
                match http
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
            }
            None => {
                // No active profile but source=Api: corrupted state. Surface
                // it so the user knows to reconnect, then bail.
                clear_active_cancel(&state);
                rollback_last_user_message(&state, &app).await;
                on_event
                    .send(serde_json::json!({
                        "type": "error",
                        "message": "API source selected but no profile connected. Reconnect in the API panel."
                    }))
                    .map_err(|e| e.to_string())?;
                return Ok(());
            }
        }
    } else if let Some(backend) = backend_opt {
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
    // (2026-07-14): no save; the turn lives only in memory for this launch.
    {
        let mut s = state.session.lock().await;
        s.add_assistant_turn(
            result.content.clone(),
            result.reasoning.clone(),
            result.raw.clone(),
        );

        // Trigger is turn-COMPLETION, not truncation. We read from the
        // Conversation (clean strings), sidestepping the engine.rs:480
        // token-boundary-drift landmine entirely: truncate_to_fit operates on
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
        // `codex_was_injected` was captured before `memory_block` was moved
        // into `.stream()`. If true, skip archiving the assistant's reply -
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

    // Fire the background schema delta pass for the turn that just completed.
    // Mirrors the memory archive spawn above: detached, best-effort, errors
    // logged-and-dropped. The handle is stored in pending_delta so the NEXT
    // chat_send awaits it (the invisible queue) before reading the schema -
    // guaranteeing the next turn sees this turn's schema update.
    //
    // The delta pass runs on the dedicated wupi-schema thread (isolated
    // context, never touches the chat KV cache). The JoinHandle wraps the
    // post-generation work: post the request, await the reply via
    // spawn_blocking, apply the delta, persist. If the schema engine isn't
    // available (init failed, or chat proceeded in echo mode, or mid-swap),
    // skip silently. Clone the Arc out of the Mutex and drop the guard
    // before the spawned task (the task holds the clone across awaits).
    let schema_engine_opt = state
        .schema_engine
        .lock()
        .map(|g| g.clone())
        .unwrap_or(None);
    if let Some(schema_engine) = schema_engine_opt {
        // Capture the exchange from the session (clean strings, same source
        // as the memory archive: sidesteps the token-boundary-drift landmine
        // the same way). Read inside a brief lock, clone out, then drop the
        // guard before spawning so the task doesn't pin the session mutex.
        let (user_text, asst_text) = {
            let s = state.session.lock().await;
            let user = s.messages.len().checked_sub(2).and_then(|i| s.messages.get(i)).map(|m| m.content.clone());
            (user, result.content.clone())
        };
        // The delta pass is a full 12B forward pass. Skip it for clearly non-
        // substantive turns (short filler like "ok"/"thanks", or empty replies)
        //: see `should_fire_delta` for the conservative heuristic. 99% of real
        // turns still fire; the user's typing time masks the generation cost.
        // A skipped turn leaves pending_delta empty, so the next chat_send
        // doesn't wait: zero latency hit for filler turns.
        let user_text_for_gate = user_text.as_deref().unwrap_or("");
        if !schema_engine::should_fire_delta(user_text_for_gate, &asst_text) {
            tracing::debug!(
                user_words = user_text_for_gate.split_whitespace().count(),
                "schema delta skipped by content gate (non-substantive turn)"
            );
        } else {
            let current_schema = state.schema.lock().await.clone();
            let schema_engine = Arc::clone(&schema_engine);
            let schema_slot = state.schema.clone();
            let handle = tokio::spawn(async move {
                // Post the delta request. The reply comes back on a std::mpsc
                // channel (the schema thread is a bare std::thread), so we await
                // it via spawn_blocking: same pattern as the chat engine reply.
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
    // ephemeral now (2026-07-14): no disk save, just in-memory correction.
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
/// entirely: this is the observability surface for tuning retrieval
/// independently of generation, AND the calibration surface for
/// [`memory_rrf::DENSE_COSINE_FLOOR`] (AGENTS.md §2M Checkpoint E).
///
/// `top_k` defaults to 10 when `None`. `dense_floor` overrides the const for
/// live calibration: pass a value to see how the result set changes at that
/// threshold without a rebuild; leave `None` to use the compiled default.
/// Returns an error string (not a panic) if the memory engine isn't
/// initialized or the query fails: the panel renders it as a red message.
///
/// Retrieval is scoped to the active card id (AGENTS.md §2M): cards never
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

// The Codex UI lists / searches / edits / removes memories and can hard-reset
// the active card's episodic store. `debug_memory_query` above is the search
// path (reused by the Codex search box); these four commands cover enumerate /
// mutate / wipe. All scope to the active card id exactly as the search does.

/// Enumerate memories in the active card, newest first. The Codex browser's
/// default view. `limit` defaults to 200 (the per-card corpus is small); an
/// explicit `0` is clamped to 1 so the UI always gets at least the head row.
#[tauri::command]
async fn memory_list(
    limit: Option<usize>,
    offset: Option<usize>,
    state: tauri::State<'_, AppState>,
) -> Result<Vec<memory::MemoryEntry>, String> {
    let engine = state
        .memory
        .get()
        .ok_or_else(|| "memory engine not initialized".to_string())?;
    let card_id = state
        .active_card_id
        .lock()
        .expect("active_card_id mutex")
        .clone();
    let limit = limit.unwrap_or(200).max(1);
    let offset = offset.unwrap_or(0);
    engine
        .list_memories(&card_id, limit, offset)
        .await
        .map_err(|e| format!("{e:#}"))
}

/// Edit one memory's text in place (re-embeds + rewrites all three tables).
/// Silent no-op if `id` doesn't exist. Used by the Codex browser's inline
/// editor.
#[tauri::command]
async fn memory_update(
    id: i64,
    text: String,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    let engine = state
        .memory
        .get()
        .ok_or_else(|| "memory engine not initialized".to_string())?;
    engine
        .update_memory(id, text)
        .await
        .map_err(|e| format!("{e:#}"))
}

/// Delete one memory by id (all three tables). Used by the Codex browser's
/// per-row Remove button. Wraps the existing engine method; lifted to IPC so
/// the frontend doesn't need a separate delete surface.
#[tauri::command]
async fn memory_delete(id: i64, state: tauri::State<'_, AppState>) -> Result<(), String> {
    let engine = state
        .memory
        .get()
        .ok_or_else(|| "memory engine not initialized".to_string())?;
    engine
        .delete_memory(id)
        .await
        .map_err(|e| format!("{e:#}"))
}

/// Hard reset: wipe every EPISODIC memory in the active card, preserving
/// authored Codex lore. Returns the deleted count so the UI can confirm.
/// The Codex browser's "Hard Reset" button (confirm-gated on the frontend).
#[tauri::command]
async fn memory_wipe_card(state: tauri::State<'_, AppState>) -> Result<usize, String> {
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
        .wipe_episodic_card(&card_id)
        .await
        .map_err(|e| format!("{e:#}"))
}

// The Codex is a library of authored reference "books" (world lore, TV/wiki
// facts, worldbuilding). Source of truth = `.md` files in the resolved
// `docs/` dir; the DB is a derived retrieval index re-seeded at boot. These
// three commands operate on the FILES directly, then re-seed so retrieval
// stays in sync within the running session. Nothing here touches episodic
// chat memory: the Codex is a separate, authored-only surface.

/// List every Codex entry (filename, title, tags, body). The Codex UI's
/// library view. Returns an empty Vec when no docs/ dir resolved.
#[tauri::command]
fn codex_list(state: tauri::State<'_, AppState>) -> Result<Vec<codex::CodexFile>, String> {
    let dir = state.codex_dir.get().and_then(|o| o.as_ref());
    let Some(dir) = dir else { return Ok(Vec::new()); };
    codex::list_files(dir).map_err(|e| format!("{e:#}"))
}

/// Create or overwrite a Codex `.md` file, then re-seed so retrieval sees the
/// change this session. `filename` is the stem (sanitized on disk). Returns
/// the (possibly-sanitized) filename so the UI can track the real key.
#[tauri::command]
async fn codex_save(
    filename: String,
    title: String,
    tags: Vec<String>,
    body: String,
    state: tauri::State<'_, AppState>,
) -> Result<String, String> {
    let dir = state
        .codex_dir
        .get()
        .and_then(|o| o.as_ref().cloned())
        .ok_or_else(|| "no codex dir resolved".to_string())?;
    // Write the file off the tokio worker (synchronous FS I/O). `save_file`
    // returns the sanitized stem it actually wrote: echo it back so the UI
    // tracks the entry by its real on-disk key.
    let saved_name = tokio::task::spawn_blocking(move || codex::save_file(&dir, &filename, &title, &tags, &body))
        .await
        .map_err(|e| format!("codex save join: {e}"))?
        .map_err(|e| format!("{e:#}"))?;
    // Re-seed so the retrieval index reflects the edit without a reboot.
    // Pinned to CODEX_CARD_ID (NOT active_card_id): Phase 2 firewall fix:
    // pre-fix this read active_card_id, so editing lore DURING a game wrote
    // it into the active roleplay card's partition. User lore always lands in
    // the user's namespace regardless of what game is running.
    if let (Some(engine), Some(dir)) = (state.memory.get(), state.codex_dir.get().and_then(|o| o.as_ref())) {
        let _ = codex::seed_codex(engine, dir, memory::CODEX_CARD_ID, "codex").await;
    }
    Ok(saved_name)
}

/// Delete a Codex `.md` file by stem, then re-seed. Silent no-op if missing.
#[tauri::command]
async fn codex_delete(
    filename: String,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    let dir = state
        .codex_dir
        .get()
        .and_then(|o| o.as_ref().cloned())
        .ok_or_else(|| "no codex dir resolved".to_string())?;
    tokio::task::spawn_blocking(move || codex::delete_file(&dir, &filename))
        .await
        .map_err(|e| format!("codex delete join: {e}"))?
        .map_err(|e| format!("{e:#}"))?;
    // Same CODEX_CARD_ID pin as codex_save (Phase 2 firewall).
    if let (Some(engine), Some(dir)) = (state.memory.get(), state.codex_dir.get().and_then(|o| o.as_ref())) {
        let _ = codex::seed_codex(engine, dir, memory::CODEX_CARD_ID, "codex").await;
    }
    Ok(())
}

// Two commands mirror the theme get/set pattern: read fresh from the cached
// path, write atomically back. Hot-reload is automatic (chat_send re-reads
// every turn), so a saved profile applies on the next chat turn with no extra
// wiring. `UserProfile` is Serialize/Deserialize so it crosses IPC directly.

/// Read the operator profile fresh from disk. Returns `None` when no
/// Operator.xml resolved at startup (the Profile Editor renders empty fields
/// and a Create prompt in that case).
#[tauri::command]
async fn operator_profile_get(
    state: tauri::State<'_, AppState>,
) -> Result<Option<user_profile::UserProfile>, String> {
    // `operator_path` is `Arc<OnceLock<Option<PathBuf>>>`. `.get()` yields
    // `Option<&Option<PathBuf>>`; flatten + clone the inner PathBuf to an
    // OWNED Option<PathBuf> so it can move into the 'static spawn_blocking
    // closure (a borrow of `state` can't cross that boundary). `load` takes
    // Option<&Path>; `.as_deref()` on the owned Option<PathBuf> at the call
    // site yields exactly that.
    let path = state
        .operator_path
        .get()
        .and_then(|o| o.clone());
    // spawn_blocking: load does synchronous file I/O. Cheap, but keep it off
    // the tokio worker for consistency with the rest of the profile/memory IPC.
    tokio::task::spawn_blocking(move || user_profile::load(path.as_deref()))
        .await
        .map_err(|e| format!("profile get join: {e}"))
}

/// Write the operator profile atomically to the resolved `Operator.xml` path.
/// Creates the file (and its parent dir) if missing. Returns an error string
/// if no path resolved at startup (shouldn't happen: `setup` always resolves
/// the candidates; `None` means none existed, in which case we can't write).
#[tauri::command]
async fn operator_profile_set(
    name: String,
    description: String,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    // `operator_path` is `Arc<OnceLock<Option<PathBuf>>>`. `.get()` yields
    // `Option<&Option<PathBuf>>`; flatten + clone the inner PathBuf so we own
    // it and can move it into the spawn_blocking closure.
    let path = state
        .operator_path
        .get()
        .and_then(|o| o.clone())
        .ok_or_else(|| "no operator profile path resolved".to_string())?;
    let profile = user_profile::UserProfile { name, description };
    tokio::task::spawn_blocking(move || user_profile::save(&path, &profile))
        .await
        .map_err(|e| format!("profile set join: {e}"))?
        .map_err(|e| format!("{e:#}"))
}

// Source of truth = `api_config.json` in the app data dir; the in-memory
// `AppState.api_config` is the fast read copy. These commands cover enumerate
// / mutate profiles + read/switch the active model source. `api_connect` and
// `api_disconnect` perform the actual model swap (chunk 4); in chunk 2 they
// just validate + set state so the IPC surface is testable end-to-end before
// the risky teardown code lands.

/// Read the full API config (all profiles + active source). The API panel's
/// default view. Returns the `ApiConfig` as-is: the frontend renders the
/// profile list + the model-source radio from it.
#[tauri::command]
fn api_profiles_list(state: tauri::State<'_, AppState>) -> api::ApiConfig {
    state.api_config.lock().expect("api_config mutex").clone()
}

/// Upsert a profile by id (replace if same id, append otherwise), persist,
/// return the saved profile. The id is sanitized from the name if the caller
/// passes an empty one: the UI tracks entries by the returned id.
#[tauri::command]
async fn api_profile_save(
    mut profile: api::ApiProfile,
    state: tauri::State<'_, AppState>,
) -> Result<api::ApiProfile, String> {
    if profile.id.trim().is_empty() {
        profile.id = api::sanitize_profile_id(&profile.name);
    } else {
        profile.id = api::sanitize_profile_id(&profile.id);
    }
    let path = state
        .api_config_path
        .get()
        .cloned()
        .ok_or_else(|| "api_config path not initialized".to_string())?;
    let saved = profile.clone();
    // Mutate under the lock, snapshot, then DROP the guard before awaiting
    // (std::sync::MutexGuard is !Send; can't hold across spawn_blocking.await).
    let cfg_snapshot = {
        let mut cfg = state.api_config.lock().expect("api_config mutex");
        cfg.upsert(profile);
        cfg.clone()
    };
    tokio::task::spawn_blocking(move || cfg_snapshot.save(&path))
        .await
        .map_err(|e| format!("api_config save join: {e}"))?;
    Ok(saved)
}

/// Delete a profile by id. If it was the active profile, clears active +
/// downgrades model_source to Local (can't stay on API with no profile).
/// Returns true if a profile was removed.
#[tauri::command]
async fn api_profile_delete(
    profile_id: String,
    state: tauri::State<'_, AppState>,
) -> Result<bool, String> {
    let path = state
        .api_config_path
        .get()
        .cloned()
        .ok_or_else(|| "api_config path not initialized".to_string())?;
    // Mutate under the lock, snapshot, drop guard before awaiting.
    let (removed, downgrade_source, cfg_snapshot) = {
        let mut cfg = state.api_config.lock().expect("api_config mutex");
        let was_active = cfg.active_profile_id.as_deref() == Some(profile_id.as_str());
        let removed = cfg.remove(&profile_id);
        // If we just deleted the active profile, we can't stay on API. This
        // flips the in-memory state; the actual model swap (reload local)
        // happens in chunk 4's full disconnect path. For chunk 2 it's just
        // bookkeeping: the frontend reads model_source_get to reflect it.
        let downgrade_source = removed && was_active;
        if downgrade_source {
            cfg.model_source = api::ModelSource::Local;
        }
        (removed, downgrade_source, cfg.clone())
    };
    tokio::task::spawn_blocking(move || cfg_snapshot.save(&path))
        .await
        .map_err(|e| format!("api_config save join: {e}"))?;
    if downgrade_source {
        *state.model_source.lock().expect("model_source mutex") = api::ModelSource::Local;
    }
    Ok(removed)
}

/// Read the current model source + readiness flags. The frontend's source
/// selector reads this. `api_ready` = an active profile exists (so the API
/// radio is enabled); `local_ready` = the local backend is loaded.
#[tauri::command]
fn model_source_get(state: tauri::State<'_, AppState>) -> serde_json::Value {
    let source = *state.model_source.lock().expect("model_source mutex");
    let cfg = state.api_config.lock().expect("api_config mutex");
    let api_ready = cfg.active_profile().is_some();
    let local_ready = state
        .backend
        .lock()
        .expect("backend mutex")
        .as_ref()
        .map(|b| b.is_ready())
        .unwrap_or(false);
    serde_json::json!({
        "source": source,
        "apiReady": api_ready,
        "localReady": local_ready,
    })
}

/// Connect an API profile: set it active, perform the model swap (Local→API),
/// flip model_source to Api. In chunk 2 the swap is a stub: it just validates
/// the profile exists + sets state. The real teardown lands in chunk 4.
#[tauri::command]
async fn api_connect(
    profile_id: String,
    app: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    match api_connect_inner(profile_id, app.clone(), &state).await {
        Ok(()) => Ok(()),
        // Safety net for the title indicator: if connect failed, the 12B is
        // still loaded (validation runs before any teardown; if teardown ran
        // and then something downstream failed, the runtime is genuinely
        // offline). Emit model-status so the frontend's title self-corrects
        //: never gets stuck in the red "swapping" state. JS handles this
        // too, but the backend is the authority.
        Err(e) => {
            let backend_loaded = state
                .backend
                .lock()
                .expect("backend mutex")
                .as_ref()
                .map(|b| b.is_ready())
                .unwrap_or(false);
            let status = if backend_loaded {
                serde_json::json!({ "status": "ready", "model": "WUPI.gguf" })
            } else {
                serde_json::json!({ "status": "error", "message": &e })
            };
            let _ = app.emit("model-status", status);
            Err(e)
        }
    }
}

async fn api_connect_inner(
    profile_id: String,
    app: tauri::AppHandle,
    state: &tauri::State<'_, AppState>,
) -> Result<(), String> {
    let path = state
        .api_config_path
        .get()
        .cloned()
        .ok_or_else(|| "api_config path not initialized".to_string())?;
    // Validate the profile exists + has the required fields (under lock, then
    // drop the guard before any await).
    {
        let cfg = state.api_config.lock().expect("api_config mutex");
        let profile = cfg
            .profiles
            .iter()
            .find(|p| p.id == profile_id)
            .ok_or_else(|| format!("no API profile with id {profile_id}"))?;
        if profile.endpoint.trim().is_empty() {
            return Err("profile endpoint is empty".into());
        }
        if profile.model.trim().is_empty() {
            return Err("profile model is empty".into());
        }
        if profile.api_key.trim().is_empty() {
            return Err("profile api_key is empty".into());
        }
    }

    tracing::info!("api_connect: beginning model swap (Local → API)");
    // The original design swapped the schema/memory engine to Agent.gguf on
    // API connect (to free VRAM for the API chat path). That required
    // Agent.gguf to actually load: and the file we had (Gemma 4 E4B)
    // returns NullResult in llama-cpp-2 0.1.151. Rather than block API mode
    // on a separate sidekick model, the schema engine now stays on WUPI.gguf
    // in both modes. Cost: ~1-2GB extra VRAM for the schema's isolated
    // context on the 12B when in API mode. Benefit: API mode works without
    // any external model dependency.
    tracing::info!("api_connect: keeping schema engine on WUPI.gguf (no Agent.gguf swap)");

    // Tear down the 12B CHAT engine (the schema engine stays put). Posts
    // EngineMsg::Shutdown; the thread exits + drops its LlamaContext (freeing
    // VRAM). shutdown() blocks on the JoinHandle so VRAM is actually released
    // (load-bearing for subsequent loads: see the 2026-07-18 VRAM-overlap
    // fix). Wrapped in spawn_blocking because the join is a synchronous
    // block we don't want on a Tokio worker.
    //
    // Scope the mutex guard in its own block so it drops BEFORE the .await -
    // holding a std::sync::MutexGuard across an await makes the future !Send
    // and Tauri commands require Send futures.
    let backend_opt = {
        let mut guard = state.backend.lock().expect("backend mutex");
        guard.take()
    };
    if let Some(backend) = backend_opt {
        tokio::task::spawn_blocking(move || backend.shutdown())
            .await
            .map_err(|e| format!("chat backend shutdown join: {e}"))?;
    }
    // 3. Flip model_source FIRST (before persisting) so chat_send routes to
    //    the API path on the very next message. Then persist the config.
    *state.model_source.lock().expect("model_source mutex") = api::ModelSource::Api;
    let cfg_snapshot = {
        let mut cfg = state.api_config.lock().expect("api_config mutex");
        cfg.active_profile_id = Some(profile_id.clone());
        cfg.model_source = api::ModelSource::Api;
        cfg.clone()
    };
    tokio::task::spawn_blocking(move || cfg_snapshot.save(&path))
        .await
        .map_err(|e| format!("api_config save join: {e}"))?;
    tracing::info!(profile_id = %profile_id, "api connected: chat via API, schema stays on WUPI.gguf");
    // Emit model-status so the title indicator flips from "swapping" (red)
    // to "idle" (steady white). api_disconnect's reload path already emits
    // via the spawn_load callback; api_connect had no emit, so the title
    // stayed red indefinitely after an ONLINE connect. The model name is
    // the profile's model string (what the API will actually serve).
    let model_name = {
        let cfg = state.api_config.lock().expect("api_config mutex");
        cfg.active_profile().map(|p| p.model.clone()).unwrap_or_default()
    };
    let _ = app.emit(
        "model-status",
        serde_json::json!({ "status": "ready", "model": model_name }),
    );
    Ok(())
}

/// Disconnect the API: flip back to Local, perform the reverse model swap
/// (API→Local). Reloads WUPI.gguf as the chat engine + schema engine.
#[tauri::command]
async fn api_disconnect(app: tauri::AppHandle, state: tauri::State<'_, AppState>) -> Result<(), String> {
    let path = state
        .api_config_path
        .get()
        .cloned()
        .ok_or_else(|| "api_config path not initialized".to_string())?;

    tracing::info!("api_disconnect: beginning chat-engine reload (API to Local)");
    // The schema engine stayed on WUPI.gguf during API mode (no swap on
    // connect, see api_connect), so there's no schema swap to reverse here.
    // Just reload the 12B chat engine.
    let model_path = resolve_model_path(&app)
        .ok_or_else(|| "no WUPI.gguf found for reconnect".to_string())?;
    let context_size = state.settings.lock().expect("settings mutex").context_size;
    let app_handle = app.clone();
    let backend = llm::LlamaCppBackend::spawn_load(
        model_path,
        99,
        context_size,
        Box::new(move |result| match &result {
            Ok(name) => {
                let _ = app_handle.emit(
                    "model-status",
                    serde_json::json!({ "status": "ready", "model": name }),
                );
            }
            Err(msg) => {
                let _ = app_handle.emit(
                    "model-status",
                    serde_json::json!({ "status": "error", "message": msg }),
                );
            }
        }),
    );
    *state.backend.lock().expect("backend mutex") = Some(backend);
    // 3. Flip model_source to Local + persist.
    *state.model_source.lock().expect("model_source mutex") = api::ModelSource::Local;
    let cfg_snapshot = {
        let mut cfg = state.api_config.lock().expect("api_config mutex");
        cfg.model_source = api::ModelSource::Local;
        cfg.clone()
    };
    tokio::task::spawn_blocking(move || cfg_snapshot.save(&path))
        .await
        .map_err(|e| format!("api_config save join: {e}"))?;
    tracing::info!("api disconnected: chat + schema back on WUPI.gguf (local)");
    Ok(())
}

/// Test whether an API profile is reachable. Issues a lightweight GET to the
/// endpoint's `/models` path (the OpenAI-standard list endpoint). Returns Ok
/// with the model list if reachable, Err with a diagnostic if not. Used by
/// the API panel's "Test connection" button before the user commits.
#[tauri::command]
async fn api_profile_test(
    profile: api::ApiProfile,
) -> Result<serde_json::Value, String> {
    let base = profile.endpoint.trim_end_matches('/').to_string();
    // Hit /models if the endpoint is a bare base; if it already points at
    // /chat/completions, strip back to the base and try /models from there.
    let base = base.trim_end_matches("/chat/completions").to_string();
    let url = format!("{base}/models");
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .map_err(|e| format!("build client: {e}"))?;
    let resp = client
        .get(&url)
        .bearer_auth(&profile.api_key)
        .send()
        .await
        .map_err(|e| format!("request to {url} failed: {e}"))?;
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        return Err(format!(
            "API returned {status}: {}",
            body.chars().take(300).collect::<String>()
        ));
    }
    // Return the raw JSON (shape varies by provider; the frontend just shows
    // "connected" + optionally the model count). Best-effort parse: if it's
    // not JSON, return a success marker with the text body.
    match resp.json::<serde_json::Value>().await {
        Ok(v) => Ok(v),
        Err(_) => Ok(serde_json::json!({ "connected": true })),
    }
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
/// run: the schema is untouched, useful for prompt-tuning without side effects.
///
/// The schema engine is spawned LAZILY on first call (gated on the chat model
/// being loaded: `shared_model()` must be `Some`). Mirrors the Memory engine's
/// OnceLock-once pattern; Component E will move this to an eager spawn at
/// model-ready. Returns an error string if the chat model isn't loaded yet.
#[tauri::command]
async fn debug_schema_delta(
    user_exchange: String,
    assistant_exchange: String,
    apply: Option<bool>,
    state: tauri::State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    // The schema engine is eager-spawned at chat-model-ready in setup(). This
    // debug command requires it to already be running: the lazy-spawn fallback
    // was removed when the engine stopped using shared_model() (it now loads
    // its own model by path, which needs the app handle for resolution: not
    // worth threading through a debug-only path). If the eager spawn failed at
    // boot, surface that here.
    let engine = state
        .schema_engine
        .lock()
        .map_err(|e| format!("schema_engine mutex: {e}"))?
        .clone()
        .ok_or_else(|| "schema engine not running (eager spawn failed at boot, or no model found)".to_string())?;

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
            // Parse failed: return the unchanged schema.
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

// ===========================================================================
// Games app IPC (Seam 1 + Seam 2, 2026-07-18): see docs/games-app-design.md
// ===========================================================================
// Five commands: enumerate roleplay cards, start a game (spawn GameEngine +
// swap active_card_id), send a narrator turn (streaming), stop a turn, end
// the game (shutdown engine + restore card id). The narrator system prompt
// is built per-turn from the active roleplay card + the card's scoped
// schema. Bracket commands are parsed from the final raw output + emitted
// as structured scene_event Channel messages so the (deferred) UI can route
// them. Memory archiving + schema delta reuse the existing paths: both
// scope to the active card_id automatically.

/// Lightweight metadata for one roleplay card, returned by `game_cards_list`.
/// Carries enough for a card-picker UI (name, id, short description) without
/// loading the full persona body.
#[derive(Debug, Clone, serde::Serialize)]
struct GameCardMeta {
    id: String,
    name: String,
    card_type: String,
    setting_preview: String,
    tone: Option<String>,
}

/// Enumerate every `.sim` file in `cards/game_cards/` and return parsed
/// metadata. The card-picker UI's data source. Returns an empty Vec when no
/// game_cards/ dir exists (the common case until cards are authored or
/// imported): graceful, not an error.
#[tauri::command]
fn game_cards_list(app: tauri::AppHandle) -> Result<Vec<GameCardMeta>, String> {
    let dir = resolve_game_cards_dir(&app);
    let Some(dir) = dir else {
        return Ok(Vec::new());
    };
    let mut out = Vec::new();
    let entries = std::fs::read_dir(&dir).map_err(|e| format!("read game_cards/: {e}"))?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|x| x.to_str()) != Some("sim") {
            continue;
        }
        let card = sim_card::load_or_fallback(&path);
        // Skip fallback stubs (a malformed file produced the fallback). The
        // id sentinel is the signal: see sim_card::FALLBACK_ID.
        if card.id == "__wupi_fallback__" {
            tracing::warn!(path = %path.display(), "skipping malformed game card");
            continue;
        }
        // Only list roleplay cards in this registry: the system card
        // (Wupi.sim) lives in `cards/`, not `cards/game_cards/`, so this is
        // belt-and-suspenders against a misplaced file.
        if card.card_type != "roleplay" {
            continue;
        }
        let setting_preview = card
            .setting
            .as_deref()
            .map(|s| s.chars().take(160).collect::<String>())
            .unwrap_or_default();
        out.push(GameCardMeta {
            id: card.id.clone(),
            name: card.name.clone(),
            card_type: card.card_type.clone(),
            setting_preview,
            tone: card.tone.clone(),
        });
    }
    // Stable order: alphabetical by name so the picker doesn't jitter.
    out.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    Ok(out)
}

/// Start a game: load the roleplay card, spawn the GameEngine (loads
/// WUPI.gguf as its own isolated context), swap `active_card_id` to the
/// card's id, and reset the game's scoped schema to empty (fresh scene).
/// The `pre_game_card_id` is saved so `game_end` can restore it.
#[tauri::command]
async fn game_start(
    card_id: String,
    app: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    tracing::info!(card_id = %card_id, "game_start: spawning GameEngine");

    // 1. Refuse if a game is already running (the UI shouldn't allow this,
    //    but defense-in-depth).
    {
        let existing = state.game_engine.lock().expect("game_engine mutex");
        if existing.is_some() {
            return Err("a game is already running: call game_end first".into());
        }
    }

    // 2. Resolve + load the roleplay card by id. The id comes from
    //    `game_cards_list`, so it must exist in the registry.
    let card = {
        let dir = resolve_game_cards_dir(&app)
            .ok_or_else(|| "no cards/game_cards/ dir resolved".to_string())?;
        find_card_by_id(&dir, &card_id)?
    };

    // 3. Resolve the model path (WUPI.gguf: same file the chat engine uses,
    //    freshly leaked as the GameEngine's own &'static ref).
    let model_path = resolve_model_path(&app)
        .ok_or_else(|| "no WUPI.gguf found: cannot start game".to_string())?;

    // 4. Spawn the GameEngine + block on readiness (the engine loads its
    //    own model on a dedicated std::thread; recv() runs on the tokio
    //    worker via spawn_blocking: same pattern as the schema engine).
    let (engine, init_rx) = game_engine::GameEngine::spawn_load(model_path, 99);
    let ready = tokio::task::spawn_blocking(move || init_rx.recv())
        .await
        .map_err(|e| format!("game engine init join: {e}"))?
        .map_err(|e| format!("game engine init channel: {e}"))?;
    match ready {
        Ok(()) => {
            if let Ok(mut slot) = state.game_engine.lock() {
                *slot = Some(Arc::new(engine));
            }
        }
        Err(msg) => {
            return Err(format!("game engine init failed: {msg}"));
        }
    }

    // 5. Swap active_card_id + save the pre-game value for restoration.
    //    This scopes all memory retrieval + archiving to the roleplay card
    //    automatically (§2M: already wired through chat_send + the debug
    //    panel). Phase 3: instead of zeroing the schema/session, LOAD any
    //    prior per-card state from disk so a game resumes where it left off.
    //    First launch of a card → NotFound → default/empty (the loaders handle
    //    this gracefully).
    {
        let mut pre = state.pre_game_card_id.lock().expect("pre_game_card_id mutex");
        let mut active = state.active_card_id.lock().expect("active_card_id mutex");
        *pre = active.clone();
        *active = card.id.clone();
    }
    // Load prior per-card state. Both fall back to default/empty when no save
    // exists (the loaders' NotFound path returns Default::default()). This is
    // what makes games resumable across reboots.
    let prior_schema = load_schema(&app, &card.id).await
        .unwrap_or_else(schema::WorldSchema::default);
    let prior_session = load_session(&app, &card.id).await
        .unwrap_or_else(session::Conversation::new);
    *state.game_schema.lock().await = prior_schema;
    *state.game_session.lock().await = prior_session;
    *state.active_game_card.lock().expect("active_game_card mutex") = Some(card);

    tracing::info!("game started: narrator engine live, memory scoped to card, per-card state loaded");
    Ok(())
}

/// Send a narrator turn: render the narrator prompt from the active card +
/// current game schema, post the request to the GameEngine, stream chunks
/// to the Channel, parse bracket commands from the final raw output, and
/// emit them as scene_event messages. After the turn, archive to memory
/// (card-scoped) and fire the schema delta (card-scoped). Mirrors `chat_send`
/// shape but routes to the GameEngine + uses the narrator system prompt.
#[tauri::command]
async fn game_send(
    text: String,
    on_event: tauri::ipc::Channel<serde_json::Value>,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    tracing::info!(?text, "game_send");

    // Fresh cancel token for this turn (Bug #7 pattern, scoped to game).
    let cancel: llm::CancelToken =
        Arc::new(std::sync::atomic::AtomicBool::new(false));
    {
        let mut slot = state.active_game_cancel.lock().expect("active_game_cancel mutex");
        *slot = Some(Arc::clone(&cancel));
    }

    // Pull the engine + card out under brief locks, then drop the guards
    // before the .await (the locks are std::sync::Mutex: guards are !Send).
    let engine = {
        let guard = state.game_engine.lock().expect("game_engine mutex");
        guard.clone().ok_or_else(|| "no game running: call game_start first".to_string())?
    };
    let card = {
        let guard = state.active_game_card.lock().expect("active_game_card mutex");
        guard.clone().ok_or_else(|| "no active game card".to_string())?
    };

    // Build the narrator system prompt from the card + current game schema.
    let world_state = {
        let s = state.game_schema.lock().await;
        let rendered = s.render_for_prompt();
        if rendered.is_empty() { None } else { Some(rendered) }
    };
    let system_prompt = narrator_prompt::build_narrator_system_prompt(&card, world_state.as_deref());

    // Append the user turn to the per-card game conversation, then window the
    // visible history. Same sliding-window strategy as chat_send's VISIBLE_WINDOW
    // (§2I M2): old turns drop from the prompt (memory backfills via retrieval)
    // so the prompt stays small (~5KB not ~80KB). The full conversation is
    // persisted on game_end so games resume across reboots.
    const GAME_VISIBLE_WINDOW: usize = 8; // narrator turns are shorter, so allow more than chat's 6
    {
        let mut gs = state.game_session.lock().await;
        gs.add_message(session::Role::User, text.clone());
    }

    // Build a windowed prompt: system + last GAME_VISIBLE_WINDOW messages +
    // generation cue. Same Gemma4 `<|turn>` protocol the chat path uses
    // (assistant → "model"). We render inline (no ChatFormat trait dependency)
    // because the narrator prompt is a single-shot prefill into the GameEngine
    // (no KV-cache reuse across turns: the GameEngine clears KV every turn,
    // see game_engine.rs:375). So cache-coherent re-render from raw_output
    // isn't required here; cleaned content is fine.
    let window: Vec<session::Message> = {
        let gs = state.game_session.lock().await;
        let msgs = &gs.messages;
        let start = msgs.len().saturating_sub(GAME_VISIBLE_WINDOW);
        msgs[start..].to_vec()
    };
    let mut prompt = String::with_capacity(4096);
    prompt.push_str("<|turn>system\n");
    prompt.push_str(system_prompt.trim());
    prompt.push_str("<turn|>\n");
    for m in &window {
        let role = match m.role {
            session::Role::Assistant => "model",
            session::Role::User => "user",
            session::Role::System => "system",
        };
        prompt.push_str("<|turn>");
        prompt.push_str(role);
        prompt.push('\n');
        prompt.push_str(&m.content);
        prompt.push_str("<turn|>\n");
    }
    prompt.push_str("<|turn>model\n");

    // Streaming callback wraps the Channel send.
    let on_chunk: llm::ChunkFn = Arc::new({
        let on_event = on_event.clone();
        move |piece: &str| {
            let _ = on_event.send(serde_json::json!({ "type": "chunk", "text": piece }));
        }
    });

    // Post the turn + await the reply off the tokio worker (the game thread
    // is a bare std::thread; its mpsc::Receiver is blocking).
    let reply_rx = engine
        .request_turn(prompt, on_chunk, cancel.clone())
        .map_err(|e| format!("{e:#}"))?;
    let reply = tokio::task::spawn_blocking(move || reply_rx.recv())
        .await
        .map_err(|e| format!("game reply join: {e}"))?
        .map_err(|e| format!("game reply channel: {e}"))?;

    // Clear the cancel slot now that the turn is done.
    {
        let mut slot = state.active_game_cancel.lock().expect("active_game_cancel mutex");
        *slot = None;
    }

    if !reply.error.is_empty() {
        on_event
            .send(serde_json::json!({ "type": "error", "message": reply.error }))
            .map_err(|e| e.to_string())?;
        return Ok(());
    }

    // Parse bracket commands from the final raw output + emit them as
    // scene_event messages (one per command). The cleaned prose is emitted
    // as a final "narration" message so the UI renders it as the dialogue.
    //
    // First strip the Gemma4 channel-protocol wrapping (the model emits
    // `<|channel>thought\n...<channel|>reply`: we only want the reply).
    // Reuses `schema::extract_reply_channel` (the same rsplit_once helper
    // the schema engine + chat_format use). Without this the protocol
    // markers leak into the narrator prose: runtime-discovered during the
    // 2026-07-18 MVP test. Also strips `<audio|>` (the Gemma4 audio-channel
    // closer the model emits mid-prose; without this it leaks as literal
    // text like "Mira's voice is a<audio|> whisper").
    let cleaned_raw = schema::extract_reply_channel(&reply.raw_output);
    let parsed = bracket_parser::parse(&cleaned_raw);
    for cmd in &parsed.commands {
        on_event
            .send(serde_json::json!({ "type": "scene_event", "command": cmd }))
            .map_err(|e| e.to_string())?;
    }

    // Archive both turns to the card-scoped memory. Best-effort, detached,
    // same pattern as chat_send's pillar-2 archive.
    let card_id = state.active_card_id.lock().expect("active_card_id mutex").clone();
    if let Some(memory_engine) = state.memory.get() {
        let memory_engine = Arc::clone(memory_engine);
        let user_text = text.clone();
        let asst_text = parsed.prose.clone();
        tokio::spawn(async move {
            if let Err(e) = memory_engine
                .add_memory(user_text, &card_id, memory::Role::User, 1.0)
                .await
            {
                tracing::warn!(error = %format!("{e:#}"), "archive game user turn failed");
            }
            if let Err(e) = memory_engine
                .add_memory(asst_text, &card_id, memory::Role::Assistant, 1.0)
                .await
            {
                tracing::warn!(error = %format!("{e:#}"), "archive game assistant turn failed");
            }
        });
    }

    // Phase 3: append the assistant turn to the per-card game conversation so
    // the next turn's windowed prompt includes it. We store the CLEANED prose
    // (parsed.prose): the GameEngine clears its KV cache every turn (no delta-
    // prefill), so cache-coherent raw_output re-render isn't required here
    // (unlike the chat path's Bug #3 fix). The reasoning channel is empty for
    // narrator turns (the bracket parser doesn't extract a thought channel).
    {
        let mut gs = state.game_session.lock().await;
        gs.add_assistant_turn(parsed.prose.clone(), String::new(), reply.raw_output.clone());
    }

    on_event
        .send(serde_json::json!({
            "type": "done",
            "final_text": parsed.prose,
            "cancelled": reply.cancelled,
        }))
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// Cancel the in-flight narrator turn (parallel to `chat_stop`). Signals
/// the per-request token; the engine's decode loop checks it between tokens
/// and breaks cleanly (§2C KV-consistency contract).
#[tauri::command]
async fn game_stop(state: tauri::State<'_, AppState>) -> Result<(), String> {
    tracing::info!("game_stop requested");
    let slot = state.active_game_cancel.lock().expect("active_game_cancel mutex");
    if let Some(cancel) = slot.as_ref() {
        cancel.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    Ok(())
}

/// End the game: shut down the GameEngine (frees VRAM), persist the per-card
/// session + schema (Phase 3: resumable across reboots), restore the
/// pre-game `active_card_id`, clear the game state. After this, Wupi-assistant
/// chat works exactly as before the game (memory retrieval + schema delta
/// scope back to the system card).
#[tauri::command]
async fn game_end(
    app: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    tracing::info!("game_end: shutting down GameEngine");

    // 1. Take the engine out of AppState (so concurrent game_send sees None
    //    and bails), then shut it down. shutdown() blocks on the JoinHandle
    //    until VRAM is freed (load-bearing: see GameEngine::shutdown doc).
    let engine_opt = {
        let mut guard = state.game_engine.lock().expect("game_engine mutex");
        guard.take()
    };
    if let Some(engine) = engine_opt {
        tokio::task::spawn_blocking(move || engine.shutdown())
            .await
            .map_err(|e| format!("game engine shutdown join: {e}"))?;
    }

    // 2. Phase 3 per-card persistence: capture the roleplay card id BEFORE
    //    the restore (step 3 swaps active_card_id back to the system value),
    //    then save the session + schema under the roleplay id. Both saves are
    //    best-effort: a failure logs a warning but doesn't block game_end
    //    (the in-memory state is cleared regardless; the user just loses the
    //    resume point on a disk error, not the running game).
    let roleplay_card_id = state.active_card_id.lock().expect("active_card_id mutex").clone();
    if roleplay_card_id != memory::WUPI_OS_CARD_ID {
        let schema_snapshot = state.game_schema.lock().await.clone();
        let session_snapshot = state.game_session.lock().await.clone();
        save_schema(&app, &roleplay_card_id, &schema_snapshot).await;
        save_session(&app, &roleplay_card_id, &session_snapshot).await;
        tracing::info!(card_id = %roleplay_card_id, "per-card state saved");
    }

    // 3. Restore the pre-game card id + clear the game-scoped state.
    {
        let pre = state.pre_game_card_id.lock().expect("pre_game_card_id mutex").clone();
        *state.active_card_id.lock().expect("active_card_id mutex") = pre;
    }
    *state.game_schema.lock().await = schema::WorldSchema::default();
    *state.game_session.lock().await = session::Conversation::new();
    *state.active_game_card.lock().expect("active_game_card mutex") = None;

    // 4. Clear any leftover game cancel token.
    *state.active_game_cancel.lock().expect("active_game_cancel mutex") = None;

    tracing::info!("game ended: narrator engine down, per-card state persisted, memory scope restored");
    Ok(())
}

/// Resolve the `cards/game_cards/` directory by walking the same candidate
/// dirs as `resolve_card_path`, joining `"cards/game_cards"`. Returns `None`
/// if no such dir exists in any candidate location (graceful: the picker
/// shows empty).
fn resolve_game_cards_dir(app: &tauri::AppHandle) -> Option<std::path::PathBuf> {
    use tauri::Manager;
    let mut candidates: Vec<std::path::PathBuf> = Vec::new();
    if let Some(d) = app.path().resource_dir().ok() {
        candidates.push(d.join("cards").join("game_cards"));
    }
    if let Some(exe) = std::env::current_exe().ok() {
        if let Some(parent) = exe.parent() {
            candidates.push(parent.join("cards").join("game_cards"));
            if let Some(grand) = parent.parent().and_then(|g| g.parent()) {
                candidates.push(grand.join("cards").join("game_cards"));
            }
            if let Some(gg) = parent.parent().and_then(|g| g.parent()).and_then(|g| g.parent()) {
                candidates.push(gg.join("cards").join("game_cards"));
            }
        }
    }
    if let Some(data) = app.path().app_data_dir().ok() {
        candidates.push(data.join("cards").join("game_cards"));
    }

    for dir in &candidates {
        if dir.is_dir() {
            tracing::info!("resolved game_cards dir: {}", dir.display());
            return Some(dir.clone());
        }
    }
    None
}

/// Find a roleplay card by id within `cards/game_cards/`. Returns an error
/// string (not a panic) if no card with that id exists.
fn find_card_by_id(dir: &std::path::Path, target_id: &str) -> Result<sim_card::SimCard, String> {
    let entries = std::fs::read_dir(dir).map_err(|e| format!("read game_cards/: {e}"))?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|x| x.to_str()) != Some("sim") {
            continue;
        }
        let card = sim_card::load_or_fallback(&path);
        if card.id == target_id && card.card_type == "roleplay" {
            return Ok(card);
        }
    }
    Err(format!("no roleplay card with id '{target_id}' in {}", dir.display()))
}

/// Persist the session off the Tokio worker pool.
///
/// `Conversation::save` is atomic (temp + fsync + rename, see §2E) but
/// synchronous: `File::create` / `write_all` / `sync_all` / `rename` all
/// block the calling thread on the disk. Running that on a Tokio worker
/// (which is what the old sync `save_session` did) stalls the async runtime
/// for the duration of the write + fsync. Harmless today (one user, one
/// chat, save is ~ms on SSD), but the moment the Memory engine adds
/// concurrent async work racing the save, a blocked worker becomes a real
/// stall. `spawn_blocking` moves the I/O onto the dedicated blocking thread
/// pool (default 512 threads) so workers stay free to poll futures.
///
/// The session mutex guard is still held across the `.await` by the caller
///: that's correct for a `tokio::sync::Mutex` (its guard is await-safe) and
/// serializes concurrent saves, which we want anyway.
///
/// **Phase 3 per-card persistence (AGENTS.md §2AA):** now scoped by `card_id`
/// → `sessions/<card_id>.json`. The Wupi-assistant session stays ephemeral
/// (§2K); only roleplay game sessions persist (a card carries its own
/// resumable session). The atomic-save machinery is reused as-is.
async fn save_session(
    app: &tauri::AppHandle,
    card_id: &str,
    conv: &session::Conversation,
) {
    use tauri::Manager;
    let Some(data_dir) = app.path().app_data_dir().ok() else {
        return;
    };
    let path = resolve_session_path(&data_dir, card_id);
    // Clone so the closure owns its data (spawn_blocking needs 'static). The
    // Conversation is a Vec of small messages: cheap to clone relative to a
    // disk fsync.
    let conv = conv.clone();
    let _ = tokio::task::spawn_blocking(move || {
        if let Err(e) = conv.save(&path) {
            tracing::warn!(?e, "failed to persist session");
        }
    })
    .await;
}

/// Load a card-scoped session. Returns a fresh empty `Conversation` when no
/// saved file exists (the `Conversation::load` NotFound path already does
/// this: we just route through it). Symmetric to `save_session`.
async fn load_session(
    app: &tauri::AppHandle,
    card_id: &str,
) -> Option<session::Conversation> {
    use tauri::Manager;
    let data_dir = app.path().app_data_dir().ok()?;
    let path = resolve_session_path(&data_dir, card_id);
    let path_cloned = path.clone();
    tokio::task::spawn_blocking(move || session::Conversation::load(&path_cloned))
        .await
        .ok()?
        .ok()
}

/// Persist the world-state schema off the Tokio worker pool. Mirrors
/// `save_session`: `WorldSchema::save` is atomic (temp + fsync + rename) but
/// synchronous, so `spawn_blocking` keeps the async runtime free.
///
/// **Phase 3:** now scoped by `card_id` → `schemas/<card_id>.json`. Only the
/// active game's schema persists (Wupi-assistant's schema stays ephemeral).
async fn save_schema(
    app: &tauri::AppHandle,
    card_id: &str,
    schema: &schema::WorldSchema,
) {
    use tauri::Manager;
    let Some(data_dir) = app.path().app_data_dir().ok() else {
        return;
    };
    let path = resolve_schema_path(&data_dir, card_id);
    let schema = schema.clone();
    let _ = tokio::task::spawn_blocking(move || {
        if let Err(e) = schema.save(&path) {
            tracing::warn!(?e, "failed to persist world schema");
        }
    })
    .await;
}

/// Load a card-scoped world schema. Returns a fresh default `WorldSchema`
/// when no saved file exists (the `WorldSchema::load` NotFound path already
/// does this). Symmetric to `save_schema`.
async fn load_schema(
    app: &tauri::AppHandle,
    card_id: &str,
) -> Option<schema::WorldSchema> {
    use tauri::Manager;
    let data_dir = app.path().app_data_dir().ok()?;
    let path = resolve_schema_path(&data_dir, card_id);
    let path_cloned = path.clone();
    tokio::task::spawn_blocking(move || schema::WorldSchema::load(&path_cloned))
        .await
        .ok()?
        .ok()
}

/// `<data_dir>/sessions/<card_id>.json`. The `sessions/` subdir is created
/// once in `setup()` (extends the existing `create_dir_all(&data_dir)`). The
/// card_id is the filename stem: roleplay card ids are filesystem-safe
/// (lowercased, derived from `<metadata><id>` in `sim_card.rs`).
fn resolve_session_path(data_dir: &std::path::Path, card_id: &str) -> std::path::PathBuf {
    data_dir.join("sessions").join(format!("{card_id}.json"))
}

/// `<data_dir>/schemas/<card_id>.json`. Sibling to `resolve_session_path`;
/// same subdir convention + filesystem-safety assumption.
fn resolve_schema_path(data_dir: &std::path::Path, card_id: &str) -> std::path::PathBuf {
    data_dir.join("schemas").join(format!("{card_id}.json"))
}
