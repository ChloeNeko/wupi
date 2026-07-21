//! In-app terminal: a real cmd.exe shell running headless in a ConPTY on the
//! Rust side, with xterm.js as the visible surface in the webview.
//!
//! ## The "middle man" architecture
//!
//! xterm.js is a terminal *emulator*, not a shell. It renders ANSI output and
//! emits keystrokes; the real shell (`cmd.exe`) runs in a pseudoconsole
//! (ConPTY on Win10+, the only path on Windows since Win7's `cmd.exe` console
//! host doesn't expose a portable TTY). They talk over a byte pipe:
//!
//! ```text
//! [xterm.js in #terminal]  ←TerminalData channel→  [Tauri IPC]  ←bytes→  [ConPTY cmd.exe]
//!    (renders ANSI)            (typed, backpressure)                (the real shell)
//!         ↓ keystrokes from the message-box input only
//!    [invoke('terminal_input', { text })]
//! ```
//!
//! The shell has no idea a window exists — this is the same pattern VS Code,
//! Hyper, and Tabby use, and it's what fixes the focus-trap / zombie-process
//! disaster of the old second-window terminal design (which spawned a second
//! Tauri window inside a fullscreen + alwaysOnTop app, then lost control of
//! the child process when the window glitched).
//!
//! ## Lifecycle (kill-on-close — the load-bearing guardrail)
//!
//! One session at a time, held in `AppState::terminal` as
//! `Arc<Mutex<Option<TerminalSession>>>`. The shell dies when the drawer
//! closes — no background session, no zombie risk. Kill paths, all idempotent:
//!
//! 1. The `terminal_close` IPC (drawer ✕, Esc, paw-menu re-click, dock toggle).
//! 2. `windowCloseHooks.set('terminal', ...)` in script.js → `terminal_close`.
//! 3. `RunEvent::ExitRequested` in lib.rs calls `kill_if_any()` — guarantees
//!    no `cmd.exe` survives WUPI shutdown even on alt-F4 / Task Manager kill.
//!    This is the path the old version was missing.
//! 4. `ChildKiller::kill()` is also called on `Drop` of the session as
//!    belt-and-suspenders.
//!
//! ## Thread split: ChildKiller vs Child
//!
//! portable-pty's `Child` trait composes `ChildKiller` (for `kill` + the
//! `clone_killer` factory) and adds `wait` + `process_id`. `wait` blocks until
//! the child exits and returns the exit code — so it must run on the reader
//! thread (which exits its loop on EOF = child died). The kill path needs to
//! run from ANY thread (the IPC handler, the Drop impl, the ExitRequested
//! hook) — that's what `clone_killer` is for. So:
//!
//! - The reader thread owns the `Box<dyn Child + Send>` and calls `wait` once
//!   after its read loop ends → emits the exit code.
//! - The session stored in AppState owns a `Box<dyn ChildKiller + Send + Clone>`
//!   for cross-thread kill. Dropping the session calls `kill` on it.
//!
//! This is the portable-pty-canonical pattern (see the ChildKiller trait docs).
//!
//! ## Input model
//!
//! The frontend feeds lines via `terminal_input(text)`; this module writes
//! `text + "\r\n"` (CRLF, what ConPTY expects for Enter). The Stop button
//! calls `terminal_stop`, which writes the Windows break sequence (the raw
//! Ctrl-C byte `0x03`) to the writer — that's what interrupts a running
//! command. (Browser-style keys in the frontend: Ctrl+C copies selection,
//! it does NOT send break — the Stop button is the break path.)
//!
//! ## Why raw bytes, not String
//!
//! Terminal output is arbitrary encoding: UTF-8, CP437 (the Windows console
//! default), ANSI escape sequences. Sending as a `String` would mangle
//! multibyte sequences across chunk boundaries (a 3-byte UTF-8 char split
//! between two reads would decode to U+FFFD twice). Keeping bytes raw and
//! letting xterm.js decode via its own `WriteBuffer` is correct.
//!
//! ## Reference
//!
//! - portable-pty API: <https://docs.rs/portable-pty>
//! - ChildKiller trait: <https://docs.rs/portable-pty/latest/portable_pty/trait.ChildKiller.html>

use std::io::{Read, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use portable_pty::{native_pty_system, ChildKiller, CommandBuilder, PtySize};
use serde::Serialize;
use tauri::ipc::Channel;

/// On Windows, ensure the process has the right window-station + desktop
/// binding BEFORE spawning a ConPTY child.
///
/// This is the fix for the **`cmd.exe - Application Error (0xc0000142)` /
/// `STATUS_DLL_INIT_FAILED`** dialog that appears when a GUI-subsystem app
/// (`windows_subsystem = "windows"`) spawns a console process via ConPTY.
/// Internally ConPTY calls `CreatePseudoConsole`, which spawns `conhost.exe`
/// — itself a GUI-subsystem binary. conhost's `user32.dll` `DllMain` needs
/// access to the interactive window station + desktop; if the spawning
/// thread isn't bound to `winsta0\default`, `user32` init fails with
/// `STATUS_DLL_INIT_FAILED` and the whole spawn chain surfaces 0xc0000142.
/// (Per Microsoft's WSL issue #5448: "0xc0000142 is the error code you get
/// when conhost fails to launch before the process finishes launching.")
///
/// explorer.exe-spawned processes (the Win+R → cmd path) inherit the right
/// station binding automatically; GUI apps calling CreatePseudoConsole don't.
/// This explicit SetThreadDesktop on the spawning thread is the canonical
/// fix (see comp.os.ms-windows.programmer.win32 on CreateProcessW 0xc0000142
/// from a non-interactive context).
///
/// We do NOT call SetProcessWindowStation (that would be process-wide and
/// could disrupt Tauri's webview). SetThreadDesktop is per-thread + safe.
///
/// No-op (returns Ok(())) on non-Windows so the same source compiles.
#[cfg(windows)]
fn bind_to_interactive_desktop() -> Result<(), String> {
    use windows::core::PCWSTR;
    use windows::Win32::System::StationsAndDesktops::{
        OpenDesktopW, SetThreadDesktop, DESKTOP_CONTROL_FLAGS,
    };

    // "default" as a UTF-16 (wide) string with NUL terminator — PCWSTR needs
    // *const u16, not *const u8. The encode_wide + chain(0) is the canonical
    // pattern for building a wide literal at runtime.
    let wide: Vec<u16> = "default".encode_utf16().chain(std::iter::once(0)).collect();

    unsafe {
        // Open the default desktop on the process's current window station.
        // We don't change the window station itself (SetProcessWindowStation
        // is process-wide + could disrupt Tauri's webview); we only ensure
        // the spawning thread is bound to the default desktop so conhost
        // inherits the binding. If the process is already on WinSta0 (the
        // normal case for a user-launched GUI app), this just re-asserts
        // what's already true and is a no-op.
        //
        // OpenDesktopW signature: (lpszdesktop, dwflags, finherit, dwdesiredaccess).
        // dwflags=0 (no DF_ALLOWOTHERACCOUNT), finherit=false, access mask =
        // GENERIC_READ | GENERIC_EXECUTE = 0x80000000 | 0x20000000 = 0xA0000000.
        let desktop = OpenDesktopW(
            PCWSTR(wide.as_ptr()),
            DESKTOP_CONTROL_FLAGS::default(),
            false,
            0xA0000000u32,
        )
        .map_err(|e| format!("OpenDesktopW(default): {e}"))?;
        SetThreadDesktop(desktop)
            .map_err(|e| format!("SetThreadDesktop: {e}"))?;
        // Intentionally leak the desktop handle — it must remain valid for
        // the thread's lifetime so the binding sticks. Closing it would
        // undo the assignment. ~12 bytes, negligible.
    }
    Ok(())
}

#[cfg(not(windows))]
fn bind_to_interactive_desktop() -> Result<(), String> { Ok(()) }

/// The payload streamed to the frontend over the typed `Channel`. Tagged enum
/// so xterm.js can switch on `kind`: `'output'` writes raw bytes to the
/// terminal, `'exited'` notifies the UI the shell died (so it can close the
/// drawer + show a status line).
///
/// `bytes` is a `Vec<u8>` (not String) — terminal output is arbitrary
/// encoding (UTF-8 / CP437 / ANSI escapes); sending as a string would mangle
/// multibyte sequences across chunk boundaries.
#[derive(Clone, Serialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum TerminalData {
    Output { bytes: Vec<u8> },
    Exited { code: Option<u32> },
}

/// A live terminal session: the writer pipe + a child-killer handle + cancel
/// flag. Held in `AppState::terminal` under a `Mutex<Option<_>>` (single
/// slot: one session at a time, matching the "one terminal drawer" UI).
///
/// The reader thread owns the original `Box<dyn Child + Send>` (it needs
/// `wait()` for the exit code); this session owns the killer clone for the
/// cross-thread kill path. On `Drop`, the killer is invoked — the
/// load-bearing zombie guardrail. Even if a code path forgets to call
/// `terminal_close`, the session's destructor ensures the child doesn't
/// outlive WUPI.
pub struct TerminalSession {
    /// stdin pipe to the pty child. `Box<dyn Write + Send>` is the
    /// portable-pty writer type erased for storage.
    writer: Box<dyn Write + Send>,
    /// Cloned killer — `ChildKiller::clone_killer` returns a separate handle
    /// (`Box<dyn ChildKiller + Send + Sync>`) so we can kill from this thread
    /// while the reader thread blocks in `wait`. Invoked on `Drop` as the
    /// zombie guardrail.
    killer: Box<dyn ChildKiller + Send + Sync>,
    /// Signals the reader thread to stop after the next `read()` returns.
    /// The reader can't be cancelled mid-`read()` (portable-pty's reader is
    /// a blocking std::Read), so this is checked between reads; a kill on
    /// the child causes the read to EOF naturally.
    cancel: Arc<AtomicBool>,
}

impl Drop for TerminalSession {
    fn drop(&mut self) {
        // Signal the reader to stop (it'll also EOF when the child dies).
        self.cancel.store(true, Ordering::Relaxed);
        // Kill the child. The reader thread's `wait` then returns + the
        // thread emits Exited + exits. Best-effort: if the child is already
        // dead (e.g. the user typed `exit`), this is a no-op.
        let _ = self.killer.kill();
    }
}

/// The default shell command. On Windows: `cmd.exe` (the universal default;
/// PowerShell is opt-in for users who want it, but cmd is guaranteed present
/// on every install). On other platforms: the user's default shell via
/// `new_default_prog` (kept for non-Windows test builds).
///
/// ## Load-bearing detail: environment inheritance
///
/// `CommandBuilder::new("cmd.exe")` starts with an EMPTY environment (the
/// portable-pty docs are explicit: "By default, the environment is NOT
/// inherited from the parent process"). On Windows this breaks `cmd.exe`
/// immediately: without `SystemRoot`, `PATH`, `COMSPEC`, `TEMP`, etc. the
/// loader can't resolve the system DLLs `cmd.exe` needs → `cmd.exe` pops the
/// "Application was unable to start correctly (0xc0000142)" /
/// `STATUS_DLL_INIT_FAILED` error dialog and never produces a prompt.
///
/// Fix: explicitly iterate `std::env::vars()` and re-add each one. This
/// mirrors what `new_default_prog()` does internally, but lets us pin the
/// shell to `cmd.exe` (the default shell might be PowerShell or something
/// exotic on a customized system; cmd is the safe universal choice).
fn default_shell() -> CommandBuilder {
    let mut cmd = if cfg!(windows) {
        CommandBuilder::new("cmd.exe")
    } else {
        // Non-Windows (test builds, future cross-platform): defer to the
        // system's default shell via $SHELL. new_default_prog inherits env.
        return CommandBuilder::new_default_prog();
    };
    // Windows path: inherit the parent's env verbatim. Without this cmd.exe
    // 0xc0000142s on launch (see the doc comment above).
    for (k, v) in std::env::vars() {
        cmd.env(k, v);
    }
    cmd
}

/// Spawn a new shell in a ConPTY, store the session, and start streaming
/// output to `on_data` until EOF or cancel. Idempotent: if a session already
/// exists, returns Ok without re-spawning (the frontend can call this on
/// every drawer-open without leaking shells).
///
/// Errors are surfaced to the frontend as an IPC `Err(String)`; the drawer
/// then writes a red `[failed to start shell: …]` line to xterm and closes.
pub fn open_session(
    state: &Arc<Mutex<Option<TerminalSession>>>,
    on_data: Channel<TerminalData>,
) -> Result<(), String> {
    let mut slot = state.lock().expect("terminal mutex");
    if slot.is_some() {
        // Already live — no-op. The frontend's Channel listener is still
        // attached from the first open; calling open again would create a
        // second Channel the backend never emits to. Idempotent = correct.
        return Ok(());
    }

    // CRITICAL: bind the spawning thread to winsta0\default BEFORE
    // openpty/spawn. Without this, ConPTY's internal conhost spawn fails
    // user32.dll init → 0xc0000142 STATUS_DLL_INIT_FAILED (see the long doc
    // comment on bind_to_interactive_desktop). Best-effort: if it fails we
    // log + try the spawn anyway (the binding might already be correct and
    // the spawn could still succeed; the error path will surface if not).
    if let Err(e) = bind_to_interactive_desktop() {
        tracing::warn!(?e, "terminal: could not bind to interactive desktop (may still work)");
    }

    // 80×24 is the canonical default; ConPTY + cmd.exe render the standard
    // Windows console prompt at this size. xterm.js will fit to its container
    // on the frontend; this size only governs the initial pty dimensions.
    let pair = native_pty_system()
        .openpty(PtySize {
            rows: 24,
            cols: 80,
            pixel_width: 0,
            pixel_height: 0,
        })
        .map_err(|e| format!("openpty: {e}"))?;

    let mut cmd = default_shell();
    // A clean cwd avoids inheriting whatever dir wupi.exe was launched from
    // (often Program Files or the user's home — neither is a useful shell
    // starting point). Use the user profile dir on Windows, $HOME elsewhere.
    if let Some(home) = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")) {
        cmd.cwd(home);
    }
    tracing::info!(
        cwd = ?std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")),
        env_count = std::env::vars().count(),
        "terminal: spawning shell",
    );

    // Spawn the child attached to the pty slave.
    let mut child: Box<dyn portable_pty::Child + Send> = pair
        .slave
        .spawn_command(cmd)
        .map_err(|e| format!("spawn: {e}"))?;

    // Take the reader + writer from the master BEFORE dropping the slave.
    // The order matters on Windows ConPTY: dropping the slave too early can
    // tear down the pseudoconsole before the child has a chance to write its
    // banner, leaving the drawer empty. The wezterm portable-pty examples
    // take reader+writer first, then drop slave, then spawn the reader
    // thread. Match that order exactly.
    let mut reader = pair
        .master
        .try_clone_reader()
        .map_err(|e| format!("try_clone_reader: {e}"))?;
    let writer = pair
        .master
        .take_writer()
        .map_err(|e| format!("take_writer: {e}"))?;

    // Clone a killer handle BEFORE handing the child off to the reader
    // thread. The reader thread will own the `Child` (for `wait`); this
    // killer clone is what the session uses to kill from any thread.
    // clone_killer returns `Box<dyn ChildKiller + Send + Sync>` directly
    // (no Result) per the portable-pty trait signature — the platform
    // implementation is always able to produce a second handle.
    let killer = child.clone_killer();

    // Now safe to drop our slave reference. The child holds its own internal
    // reference, so this doesn't kill it.
    drop(pair.slave);

    tracing::info!("terminal: shell spawned, reader thread starting");

    let cancel = Arc::new(AtomicBool::new(false));
    let cancel_for_thread = Arc::clone(&cancel);

    // Reader thread: loops read() → emit Output, until EOF (child exited) or
    // cancel. On EOF, calls `wait()` on the child to get the exit status +
    // emits Exited, then the frontend closes the drawer. The thread owns the
    // `child` (for `wait`); the session owns the `killer` clone (for kill).
    let on_data_for_thread = on_data.clone();
    std::thread::spawn(move || {
        let mut buf = [0u8; 8192];
        loop {
            if cancel_for_thread.load(Ordering::Relaxed) {
                break;
            }
            match reader.read(&mut buf) {
                Ok(0) => break, // EOF: child closed its stdout (usually exited)
                Ok(n) => {
                    let bytes = buf[..n].to_vec();
                    let _ = on_data_for_thread.send(TerminalData::Output { bytes });
                }
                Err(e) => {
                    // A read error mid-stream is unrecoverable; treat as exit.
                    tracing::warn!(?e, "terminal reader error");
                    break;
                }
            }
        }
        // Stream is over — wait on the child for the exit code + emit Exited.
        // `wait` blocks until the child has fully terminated, so by the time
        // this returns the OS has reaped the process (no zombie). Best-effort:
        // if `wait` itself fails (rare), emit None so the UI still closes.
        let code = child.wait().ok().map(|s| s.exit_code());
        let _ = on_data_for_thread.send(TerminalData::Exited { code });
    });

    // Store the session (writer + killer clone + cancel). The reader thread
    // owns the original `child` (moved into the closure above).
    *slot = Some(TerminalSession {
        writer,
        killer,
        cancel,
    });
    Ok(())
}

/// Write `text + "\r\n"` to the shell's stdin. CRLF is what ConPTY expects
/// for the Enter key; raw LF would be treated as a soft line break, not a
/// command submission, in cmd.exe.
pub fn write_input(state: &Arc<Mutex<Option<TerminalSession>>>, text: &str) -> Result<(), String> {
    let mut slot = state.lock().expect("terminal mutex");
    let Some(session) = slot.as_mut() else {
        // No live session — silently no-op. The frontend's drawer is closed
        // in this case (the close hook killed the session), so this is only
        // reachable if the user somehow submits input after a close race.
        return Ok(());
    };
    // CRLF terminator. cmd.exe treats \r\n as Enter; \n alone is mid-line.
    session
        .writer
        .write_all(format!("{text}\r\n").as_bytes())
        .map_err(|e| format!("write stdin: {e}"))?;
    session.writer.flush().ok();
    Ok(())
}

/// Write the Windows break sequence (the raw Ctrl-C byte, 0x03) to the
/// shell. This is what the Stop button does. cmd.exe + ConPTY interpret
/// `0x03` as the interrupt signal — same as if the user pressed Ctrl-C in a
/// real console. Idempotent: no-op if no session is live.
pub fn send_break(state: &Arc<Mutex<Option<TerminalSession>>>) {
    let mut slot = state.lock().expect("terminal mutex");
    let Some(session) = slot.as_mut() else {
        return;
    };
    let _ = session.writer.write_all(&[0x03u8]);
    let _ = session.writer.flush();
}

/// Kill the child + drop the session. Called on drawer close, Esc, paw-menu
/// re-click, and app shutdown (via `kill_if_any`). Idempotent: no-op if no
/// session is live. The session's `Drop` impl does the actual kill (via the
/// ChildKiller clone) — we just take it out of the slot.
pub fn close_session(state: &Arc<Mutex<Option<TerminalSession>>>) {
    let mut slot = state.lock().expect("terminal mutex");
    // Drop the session — its Drop impl signals cancel + kills via the killer.
    // No explicit action needed beyond taking it out of the slot.
    *slot = None;
}

/// Forcibly kill any live session. Called from `RunEvent::ExitRequested` in
/// lib.rs as the load-bearing zombie guardrail: guarantees no `cmd.exe`
/// survives WUPI shutdown even if the user alt-F4s, pulls Task Manager, or
/// a code path forgets to call `terminal_close`. Just delegates to
/// `close_session` (which drops + kills via the session's Drop impl).
pub fn kill_if_any(state: &tauri::State<'_, crate::AppState>) {
    close_session(&state.terminal);
}
