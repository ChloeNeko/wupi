//! Hardware integrations — Windows API bridges for the status-bar modules.
//!
//! Each submodule wraps a Windows subsystem (Core Audio, WLAN, Bluetooth) and
//! exposes plain `#[tauri::command]`s returning JSON values to the frontend.
//! All COM work happens on a short-lived thread-per-call so the Tauri async
//! runtime never blocks on COM apartment concerns.

#[cfg(windows)]
pub mod audio;

#[cfg(windows)]
pub use audio::{
    audio_get_state, audio_list_outputs, audio_set_default_output, audio_set_volume, AudioRegistry,
    new_registry as new_audio_registry,
};
