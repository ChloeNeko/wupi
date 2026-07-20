//! System / power menu logic: tray icon + Shutdown / Restart / Sleep / Wake.
//!
//! "Sleep" hides the main window to the tray and pauses the aurora canvas
//! (the app's dominant idle CPU/GPU cost) while keeping the model, memory, and
//! schema engines warm in RAM/VRAM: the "barely noticeable" state. The render
//! loop is what makes sleep cheap, so we emit `canvas-pause` / `canvas-resume`
//! to the frontend which gates `requestAnimationFrame`.
//!
//! All four actions are exposed as `#[tauri::command]`s invoked from the paw
//! dropdown in `index.html`. The tray icon's double-click and its menu "Wake"
//! / "Quit" items route through the same power_wake / power_shutdown paths.

use tauri::{
    menu::{Menu, MenuItem},
    tray::{MouseButton, TrayIconBuilder, TrayIconEvent},
    AppHandle, Emitter, Manager, Runtime,
};

/// Menu item IDs (compared as `&str` against `event.id().as_ref()`).
pub const TRAY_WAKE: &str = "wupi_wake";
pub const TRAY_QUIT: &str = "wupi_quit";

const MAIN_WINDOW: &str = "main";
const EVT_CANVAS_PAUSE: &str = "canvas-pause";
const EVT_CANVAS_RESUME: &str = "canvas-resume";

/// Build + install the system-tray icon. Called once from `setup()`.
///
/// The icon reuses the paw asset bundled into the binary via
/// `tauri::generate_context!`. The menu offers "Wake" (restore the window)
/// and "Quit" (full shutdown); a double-click on the icon itself also wakes.
pub fn build_tray<R: Runtime>(app: &AppHandle<R>) -> tauri::Result<()> {
    let wake = MenuItem::with_id(app, TRAY_WAKE, "Wake", true, None::<&str>)?;
    let quit = MenuItem::with_id(app, TRAY_QUIT, "Quit", true, None::<&str>)?;
    let menu = Menu::with_items(app, &[&wake, &quit])?;

    // The icon: prefer the bundled paw PNG (32x32) shipped as a Tauri icon.
    // Fall back to no explicit icon if it can't be resolved: the tray still
    // works, just with the platform default.
    let icon = app
        .default_window_icon()
        .cloned();

    let mut builder = TrayIconBuilder::with_id("wupi-tray")
        .tooltip("WUPI OS")
        .menu(&menu)
        .on_tray_icon_event(move |tray, event| {
            // Double-click (left button) wakes the app from sleep / brings it
            // forward. Single-click is left to the OS default (show menu).
            if let TrayIconEvent::DoubleClick {
                button: MouseButton::Left,
                ..
            } = event
            {
                power_wake(tray.app_handle());
            }
        });
    if let Some(ic) = icon {
        builder = builder.icon(ic);
    }
    builder.build(app)?;
    Ok(())
}


/// Best-effort teardown of the system-tray icon. MUST be called before any
/// `std::process::exit` path so the Windows shell receives `NIM_DELETE` while
/// the process is still alive to service it. Without this, `std::process::exit`
/// skips Tauri's `Drop` for the tray, Windows is never notified, and Explorer
/// leaves a "ghost" icon cached in the hidden-icons popover until the user
/// hovers over it (the well-known Windows shell caching quirk).
///
/// `remove_tray_by_id` is the correct Tauri 2 API: it takes the icon out of
/// Tauri's internal state AND calls `icon.close()` (the platform-level
/// teardown — `Shell_NotifyIcon(NIM_DELETE)` on Windows). The returned icon
/// drops at the end of this fn, finalizing the cleanup. Idempotent + non-fatal:
/// a missing icon (None return) is the normal case on a second call, just
/// swallowed silently.
pub fn destroy_tray<R: Runtime>(app: &AppHandle<R>) {
    // Returns Option<TrayIcon>; dropping it finalizes the platform cleanup.
    // No error path — `remove_tray_by_id` returns None if the icon doesn't
    // exist (already destroyed, never built), which is fine for cleanup.
    drop(app.remove_tray_by_id("wupi-tray"));
}

/// Full shutdown: terminate the process unconditionally. We use
/// `std::process::exit(0)`: an immediate OS-level process kill that bypasses
/// Tauri's exit flow entirely. `app.exit(0)` runs the graceful window/webview
/// teardown, which can STALL when a secondary window is open or wedged, forcing
/// the user to Task Manager. `std::process::exit` kills every window + webview
/// affiliated with the process in one shot, no waiting. (The terminal window
/// that originally surfaced this has been removed, but the hard-kill remains
/// the right call for a power-off action.)
///
/// BEFORE the hard exit we explicitly `destroy_tray`: `std::process::exit`
/// skips Rust destructors, so Tauri's tray `Drop` would never run and Windows
/// would leave a ghost icon cached. Destroying first sends `NIM_DELETE` while
/// we're still alive to service it.
pub fn power_shutdown<R: Runtime>(app: &AppHandle<R>) {
    let _ = app.emit(EVT_CANVAS_PAUSE, ());
    destroy_tray(app);
    // Flush the emit + destroy above before the hard kill so the frontend gets
    // the pause event and the shell gets NIM_DELETE. (Both best-effort; if
    // they don't land, the kill still happens.)
    std::process::exit(0);
}

/// Restart: spawn a fresh copy of this executable, then shut down.
///
/// `current_exe()` is the canonical way to re-launch; detached spawn so the
/// new process survives this one's exit.
pub fn power_restart<R: Runtime>(app: &AppHandle<R>) {
    let exe = match std::env::current_exe() {
        Ok(p) => p,
        Err(e) => {
            tracing::error!(error = %e, "restart: could not resolve current_exe");
            return;
        }
    };
    match std::process::Command::new(&exe).spawn() {
        Ok(_) => tracing::info!("restart: spawned new instance, shutting down"),
        Err(e) => {
            tracing::error!(error = %e, exe = %exe.display(), "restart: spawn failed");
            // If we can't relaunch, do NOT shut down: that would leave the
            // user with nothing. Surface the failure and stay alive.
            return;
        }
    }
    power_shutdown(app);
}

/// Sleep: hide the main window to the tray and pause the canvas. Engines
/// stay warm. The window leaves the taskbar entirely (hidden, not minimized).
pub fn power_sleep<R: Runtime>(app: &AppHandle<R>) {
    let _ = app.emit(EVT_CANVAS_PAUSE, ());
    if let Some(win) = app.get_webview_window(MAIN_WINDOW) {
        let _ = win.hide();
    }
}

/// Wake: restore + focus the main window and resume the canvas.
pub fn power_wake<R: Runtime>(app: &AppHandle<R>) {
    if let Some(win) = app.get_webview_window(MAIN_WINDOW) {
        let _ = win.show();
        let _ = win.set_focus();
    }
    let _ = app.emit(EVT_CANVAS_RESUME, ());
}


#[tauri::command]
pub fn power_shutdown_cmd<R: Runtime>(app: tauri::AppHandle<R>) -> Result<(), String> {
    // Defer the actual shutdown so the IPC call returns to the frontend first;
    // otherwise the window can tear down before the reply is ack'd, which on
    // some WebView2 builds logs a harmless-but-ugly disconnect warning.
    let app2 = app.clone();
    tauri::async_runtime::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        power_shutdown(&app2);
    });
    Ok(())
}

#[tauri::command]
pub fn power_restart_cmd<R: Runtime>(app: tauri::AppHandle<R>) -> Result<(), String> {
    power_restart(&app);
    Ok(())
}

#[tauri::command]
pub fn power_sleep_cmd<R: Runtime>(app: tauri::AppHandle<R>) -> Result<(), String> {
    power_sleep(&app);
    Ok(())
}
