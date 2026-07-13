pub mod agent;
pub mod chat_format;
pub mod engine;
pub mod kv_buffer;
pub mod llm;
pub mod prompts;
pub mod session;
pub mod stream_filter;
pub mod tools;

use std::sync::Arc;
use tauri::{Emitter, Manager};
use llm::GenerationClient;

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
}

impl AppState {
    fn new() -> Self {
        Self {
            session: Arc::new(tokio::sync::Mutex::new(session::Conversation::new())),
            backend: Arc::new(std::sync::Mutex::new(None)),
            settings: Arc::new(std::sync::Mutex::new(prompts::WupiSettings::default())),
            active_cancel: Arc::new(std::sync::Mutex::new(None)),
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
    let system_prompt = prompts::build_system_content(&settings);

    let messages = {
        let mut s = state.session.lock().await;
        s.add_message(session::Role::User, text.clone());
        save_session(&app, &s).await;
        s.assemble_api_messages(&system_prompt)
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
            .stream(messages, settings.context_size, on_chunk, cancel.clone())
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
        match echo.stream(messages, settings.context_size, on_chunk, cancel.clone()).await {
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
