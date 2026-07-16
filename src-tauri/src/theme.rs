//! Theme settings: the active theme + color code, persisted to `theme.json`
//! in the app data dir. Kept separate from `WupiSettings` (which is LLM-only)
//! so UI/visual config doesn't bleed into the prompt-building struct.
//!
//! The current defaults are `theme = "Aurora"` and `color_code = "Vibrant"`.
//! These are the ONLY names recognized today; the frontend owns the list of
//! options in the cascade panels. New themes/color codes are additive — add
//! an entry to the frontend `COLOR_CODES` map and they light up live.

use std::path::PathBuf;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ThemeSettings {
    pub theme: String,
    pub color_code: String,
}

impl Default for ThemeSettings {
    fn default() -> Self {
        Self {
            theme: "Aurora".to_owned(),
            color_code: "Vibrant".to_owned(),
        }
    }
}

impl ThemeSettings {
    /// Path to `theme.json` inside the app data dir. Computed once in setup
    /// and cached on AppState (the `theme_path` OnceLock) so the load/save
    /// helpers below stay `&Path`-based and need no `AppHandle`.
    pub fn resolve_path(app_data_dir: &std::path::Path) -> PathBuf {
        app_data_dir.join("theme.json")
    }

    /// Load from disk, falling back to defaults on any error (missing file,
    /// malformed JSON, IO). Persistence is best-effort: a corrupt theme.json
    /// should never block app launch.
    pub fn load(path: &std::path::Path) -> Self {
        match std::fs::read_to_string(path) {
            Ok(s) => serde_json::from_str::<ThemeSettings>(&s).unwrap_or_default(),
            Err(_) => ThemeSettings::default(),
        }
    }

    /// Persist atomically: temp file + rename (same pattern as session.rs::save).
    /// On failure we log and continue — theme state still lives in memory and
    /// can be re-saved on the next change.
    pub fn save(&self, path: &std::path::Path) {
        let tmp = path.with_extension("json.tmp");
        let json = match serde_json::to_string_pretty(self) {
            Ok(j) => j,
            Err(e) => {
                tracing::error!(error = %e, "theme: serialize failed");
                return;
            }
        };
        if let Err(e) = std::fs::write(&tmp, json) {
            tracing::error!(error = %e, "theme: write tmp failed");
            return;
        }
        // Windows rename over existing uses MOVEFILE_REPLACE_EXISTING — atomic.
        if let Err(e) = std::fs::rename(&tmp, path) {
            tracing::error!(error = %e, "theme: rename failed");
            let _ = std::fs::remove_file(&tmp);
        }
    }
}
