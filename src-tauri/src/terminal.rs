//! Terminal windows: a real PTY (portable-pty, ConPTY on Windows 10+) wired
//! to an xterm.js frontend in a borderless, draggable, violet-glassmorphism
//! `WebviewWindow`.
//!
//! Lifecycle:
//! - `terminal_open` spawns the PTY + child shell, builds the window, and
//!   starts a reader thread that pumps PTY stdout → the frontend Channel.
//! - `terminal_input` writes keystrokes from xterm.js → the PTY stdin.
//! - `terminal_close` HIDES the window without killing the PTY, so reopening
//!   revives the same shell session. The window's custom magenta X calls this.
//! - `terminal_resize` forwards xterm.js size changes to the PTY.
//!
//! Each terminal is held in a process-wide registry so multiple can coexist
//! and so `terminal_close`/`terminal_input` can address one by label.

use std::collections::HashMap;
use std::io::Read;
use std::sync::{Arc, Mutex};
use tauri::{Manager, Runtime, WebviewUrl, WebviewWindowBuilder};

/// Single canonical label for the terminal window. One terminal at a time;
/// reopening focuses/hides the same window + PTY rather than spawning more.
const TERMINAL_WINDOW_LABEL: &str = "wupi-terminal";

/// One live terminal: the PTY writer (for stdin), the master (for resize),
/// the child handle, and the window label. The reader is consumed by a
/// dedicated thread at spawn time.
///
/// All trait objects are explicitly `+ Send` so the registry (an
/// `Arc<Mutex<HashMap<…>>>`) is `Send + Sync`, which Tauri's `State` requires.
/// `Sync` comes from the outer `Mutex`: only one command touches a handle at a
/// time. (`MasterPty` is declared `: Send` in portable-pty, but the bare trait
/// object `dyn MasterPty` doesn't pick up `Send` unless you write it.)
pub struct TerminalHandle {
    /// Writer into the PTY master → becomes the shell's stdin. Plain std::io::Write.
    writer: Box<dyn std::io::Write + Send>,
    /// The PTY master, kept for `terminal_resize` (resize lives on MasterPty,
    /// not on the writer).
    master: Box<dyn portable_pty::MasterPty + Send>,
    /// The spawned shell child. Held so we could kill it on true-exit later;
    /// today `terminal_close` only hides the window.
    #[allow(dead_code)]
    child: Box<dyn portable_pty::Child + Send>,
}

/// Process-wide registry of open terminals, keyed by window label.
pub type TerminalRegistry = Arc<Mutex<HashMap<String, TerminalHandle>>>;

pub fn new_registry() -> TerminalRegistry {
    Arc::new(Mutex::new(HashMap::new()))
}

/// Default shell. PowerShell if present (nicer on modern Windows), else the
/// COMSPEC (cmd.exe). Falls back to cmd.exe if nothing is set.
fn default_shell() -> (String, Vec<String>) {
    if let Ok(ps) = std::env::var("PSExePath") {
        // Honored only if explicitly set by the user; otherwise prefer the
        // system PowerShell below.
        return (ps, vec![]);
    }
    // pwsh.exe (PowerShell 7+) then powershell.exe (Windows PowerShell).
    for candidate in ["C:\\Program Files\\PowerShell\\7\\pwsh.exe", "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe"] {
        if std::path::Path::new(candidate).exists() {
            return (candidate.to_string(), vec![]);
        }
    }
    std::env::var("COMSPEC")
        .map(|c| (c, vec![]))
        .unwrap_or_else(|_| ("cmd.exe".to_string(), vec![]))
}

/// Create the terminal window, or focus it if one already exists. Called by
/// the paw-menu Terminal button. Does NOT spawn the PTY — the window's own
/// `terminal.js` owns the PTY lifecycle (so the PTY lives and dies with its
/// window). Returns the window label.
#[tauri::command]
pub fn terminal_create_window<R: Runtime>(
    app: tauri::AppHandle<R>,
) -> Result<String, String> {
    let label = TERMINAL_WINDOW_LABEL;
    // Reuse a single terminal window: if it exists (visible or hidden), show
    // + focus it instead of opening a second.
    if let Some(win) = app.get_webview_window(label) {
        let _ = win.set_always_on_top(true);
        let _ = win.show();
        let _ = win.set_focus();
        return Ok(label.to_string());
    }
    let url = WebviewUrl::App("terminal.html".into());
    WebviewWindowBuilder::new(&app, label, url)
        .title("WUPI Terminal")
        .inner_size(900.0, 560.0)
        .min_inner_size(420.0, 280.0)
        .decorations(false)
        .transparent(true)
        .shadow(false)
        .resizable(true)
        .always_on_top(true)
        .center()
        .build()
        .map_err(|e| format!("build terminal window: {e}"))?;
    Ok(label.to_string())
}

/// Spawn the PTY + shell for the calling window and begin streaming stdout.
/// Called by `terminal.js` once the window has mounted xterm. The window's
/// own label (passed from the frontend) is the registry key. If a PTY is
/// already registered for this label (window was hidden, now revived), we
/// just return — the existing reader thread is still pumping.
#[tauri::command]
pub fn terminal_init<R: Runtime>(
    on_event: tauri::ipc::Channel<serde_json::Value>,
    app: tauri::AppHandle<R>,
    registry: tauri::State<'_, TerminalRegistry>,
) -> Result<String, String> {
    let label = app
        .get_webview_window(TERMINAL_WINDOW_LABEL)
        .map(|_| TERMINAL_WINDOW_LABEL.to_string())
        .unwrap_or_else(|| format!("terminal-{}", std::process::id()));

    // Already initialized (window was hidden and reopened) — nothing to do.
    {
        let reg = registry.lock().map_err(|e| e.to_string())?;
        if reg.contains_key(&label) {
            return Ok(label);
        }
    }

    // ── Spawn the PTY + shell ──────────────────────────────────────────────
    let pty_system = portable_pty::native_pty_system();
    let pair = pty_system
        .openpty(portable_pty::PtySize {
            rows: 28,
            cols: 92,
            pixel_width: 0,
            pixel_height: 0,
        })
        .map_err(|e| format!("openpty: {e}"))?;

    let (shell, args) = default_shell();
    let mut cmd = portable_pty::CommandBuilder::new(&shell);
    for a in args {
        cmd.arg(a);
    }

    let child = pair
        .slave
        .spawn_command(cmd)
        .map_err(|e| format!("spawn shell: {e}"))?;

    let mut reader = pair
        .master
        .try_clone_reader()
        .map_err(|e| format!("clone reader: {e}"))?;
    let writer = pair
        .master
        .take_writer()
        .map_err(|e| format!("take writer: {e}"))?;

    // Register the terminal. `pair.master` is moved into the handle for resize.
    {
        let mut reg = registry.lock().map_err(|e| e.to_string())?;
        reg.insert(
            label.clone(),
            TerminalHandle {
                writer,
                master: pair.master,
                child,
            },
        );
    }

    // ── Reader thread: PTY stdout → frontend Channel ───────────────────────
    // Reads in a loop and sends each chunk as { kind: "data", data: <bytes> }.
    // On EOF or read error the loop exits and a { kind: "exit" } is sent; the
    // frontend can then stop. The window stays open (shell exited) so the user
    // sees the final output.
    let app_for_thread = app.clone();
    let label_for_thread = label.clone();
    let chan = on_event;
    std::thread::spawn(move || {
        let mut buf = [0u8; 8192];
        loop {
            match reader.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    let payload = serde_json::json!({
                        "kind": "data",
                        "data": base64_encode(&buf[..n]),
                        "label": label_for_thread,
                    });
                    let _ = chan.send(payload);
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => continue,
                Err(_) => break,
            }
        }
        let _ = chan.send(serde_json::json!({
            "kind": "exit",
            "label": label_for_thread,
        }));
        // Shell exited: clean the registry entry. Window close is left to the
        // user (the X), but the PTY is already gone.
        if let Some(reg) = app_for_thread.try_state::<TerminalRegistry>() {
            let mut reg = reg.lock().unwrap();
            reg.remove(&label_for_thread);
        }
    });

    Ok(label)
}

/// Forward keystrokes from xterm.js to the PTY stdin.
#[tauri::command]
pub fn terminal_input(
    label: String,
    data: String,
    registry: tauri::State<'_, TerminalRegistry>,
) -> Result<(), String> {
    let mut reg = registry.lock().map_err(|e| e.to_string())?;
    let term = reg
        .get_mut(&label)
        .ok_or_else(|| format!("no terminal '{label}'"))?;
    let bytes = base64_decode(&data);
    term.writer.write_all(&bytes).map_err(|e| e.to_string())?;
    term.writer.flush().map_err(|e| e.to_string())?;
    Ok(())
}

/// Forward an xterm.js resize to the PTY (cols/rows). Resize lives on the
/// master, not the writer.
#[tauri::command]
pub fn terminal_resize(
    label: String,
    cols: u16,
    rows: u16,
    registry: tauri::State<'_, TerminalRegistry>,
) -> Result<(), String> {
    let reg = registry.lock().map_err(|e| e.to_string())?;
    let term = reg
        .get(&label)
        .ok_or_else(|| format!("no terminal '{label}'"))?;
    term.master
        .resize(portable_pty::PtySize {
            rows,
            cols,
            pixel_width: 0,
            pixel_height: 0,
        })
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// Hide the window (do NOT kill the PTY). The custom magenta X calls this so
/// "pressing X won't literally exit the actual terminal."
#[tauri::command]
pub fn terminal_close<R: Runtime>(
    label: String,
    app: tauri::AppHandle<R>,
) -> Result<(), String> {
    if let Some(win) = app.get_webview_window(&label) {
        let _ = win.hide();
    }
    Ok(())
}

// ── Helpers ─────────────────────────────────────────────────────────────────
// The PTY writer is `Box<dyn std::io::Write + Send>` directly — portable-pty's
// `take_writer()` already returns a `Send` writer, so no wrapper is needed.
// (Earlier iterations wrapped a custom trait; the real API is std::io::Write.)

// Minimal base64 (URL-unsafe, standard alphabet) — avoids pulling a base64 dep.
fn base64_encode(input: &[u8]) -> String {
    const ALPHA: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((input.len() + 2) / 3 * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        out.push(ALPHA[((triple >> 18) & 0x3F) as usize] as char);
        out.push(ALPHA[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            out.push(ALPHA[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(ALPHA[(triple & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

fn base64_decode(input: &str) -> Vec<u8> {
    fn val(c: u8) -> u8 {
        match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => 0,
        }
    }
    let bytes: Vec<u8> = input.bytes().filter(|&b| b != b'=' && !b.is_ascii_whitespace()).collect();
    let mut out = Vec::with_capacity(bytes.len() * 3 / 4);
    for chunk in bytes.chunks(4) {
        let v0 = val(chunk[0]) as u32;
        let v1 = val(chunk[1]) as u32;
        let v2 = if chunk.len() > 2 { val(chunk[2]) as u32 } else { 0 };
        let v3 = if chunk.len() > 3 { val(chunk[3]) as u32 } else { 0 };
        let triple = (v0 << 18) | (v1 << 12) | (v2 << 6) | v3;
        out.push((triple >> 16) as u8);
        if chunk.len() > 2 { out.push((triple >> 8) as u8); }
        if chunk.len() > 3 { out.push(triple as u8); }
    }
    out
}
