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


/// Full shutdown: terminate the process unconditionally. We use
/// `std::process::exit(0)`: an immediate OS-level process kill that bypasses
/// Tauri's exit flow entirely. `app.exit(0)` runs the graceful window/webview
/// teardown, which can STALL when a secondary window is open or wedged, forcing
/// the user to Task Manager. `std::process::exit` kills every window + webview
/// affiliated with the process in one shot, no waiting. (The terminal window
/// that originally surfaced this has been removed, but the hard-kill remains
/// the right call for a power-off action.)
pub fn power_shutdown<R: Runtime>(app: &AppHandle<R>) {
    let _ = app.emit(EVT_CANVAS_PAUSE, ());
    // Flush the emit above before the hard kill so the frontend gets it.
    // (The emit is best-effort; if it doesn't land, the kill still happens.)
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
