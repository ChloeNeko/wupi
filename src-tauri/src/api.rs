//! API connection config: saved endpoint profiles + the active model-source
//! selector, persisted to `api_config.json` in the app data dir.
//!
//! This is the config half of the API feature. The HTTP backend impl
//! (`HttpBackend: GenerationClient`) lives in `llm.rs`; this module owns only
//! the persisted state — the list of saved API profiles (endpoint URL + model
//! name + API key) and which model source (local WUPI.gguf vs API) is active.
//!
//! **Storage contract** (mirrors `theme.rs` exactly): `api_config.json` in the
//! app data dir, atomic save (temp + rename), graceful default on any error.
//! The file holds real API keys in plaintext — acceptable per the design call
//! that WUPI OS runs on private offline personal computers with no network
//! exposure of the file. NOT cryptographically protected; if WUPI ever gains
//! network-facing features, move keys to Windows DPAPI first.
//!
//! **Profile model:** a flat `Vec<ApiProfile>` + an `active_profile_id`. The
//! UI upserts by `id` (a generated slug from the name); delete is by `id`. The
//! `model_source` field persists across reboots so the user's last selection
//! (local vs API) is restored at boot — but the actual model swap is
//! re-performed in `setup()` based on this value, not assumed.

use std::path::PathBuf;

/// One saved API endpoint configuration. The on-disk identity is `id` (a
/// stable slug derived from `name`); the UI tracks profiles by this key. A
/// rename = upsert with a new id (the old one is deleted by the caller).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ApiProfile {
    /// Stable slug (lowercase name, non-alphanumerics → `-`). The on-disk key.
    pub id: String,
    /// Human-readable label shown in the UI ("Z.AI personal", "NanoGPT", ...).
    pub name: String,
    /// The base endpoint URL, OpenAI-compatible. e.g.
    /// `https://api.z.ai/api/coding/paas/v4` or `https://nano-gpt.com/api/v1`.
    /// `HttpBackend` joins `/chat/completions` for the actual request path.
    pub endpoint: String,
    /// The model string the API expects ("gpt-4o-mini", "zai-glm-4.6", etc.).
    /// Provider-specific; the user copies it from the provider's model list.
    pub model: String,
    /// The API key (secret). Plaintext on disk per the design contract above.
    /// Serialized as normal JSON — `serde_json` doesn't special-case it.
    #[serde(default)]
    pub api_key: String,
    /// Optional per-profile temperature. `None` lets the provider pick its
    /// default (usually 1.0). `WupiSettings` today carries no sampling params,
    /// so the profile is the natural home for API-only sampling overrides.
    #[serde(default)]
    pub temperature: Option<f32>,
}

/// The active chat source. Persisted in `api_config.json` so the user's last
/// choice is restored at boot; the model swap is re-performed in `setup()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelSource {
    /// Local `WUPI.gguf` (12B) — the default. Chat + schema both run locally.
    Local,
    /// An API endpoint is the chat source. The schema/memory engine moves to
    /// `Agent.gguf` (4B) so the API can focus on roleplay; never two local
    /// chat models loaded simultaneously.
    Api,
}

impl Default for ModelSource {
    fn default() -> Self {
        Self::Local
    }
}

/// The full persisted API config: every saved profile + the active source.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ApiConfig {
    /// All saved profiles. Upsert by `id`; the UI lists these alphabetically.
    #[serde(default)]
    pub profiles: Vec<ApiProfile>,
    /// Which profile is currently connected (last successful `api_connect`).
    /// `None` when no profile is active — `ModelSource::Api` requires this to
    /// be `Some`; the UI gates the API radio on a connected profile existing.
    #[serde(default)]
    pub active_profile_id: Option<String>,
    /// The selected chat source. Restored at boot; the swap re-runs in setup.
    #[serde(default)]
    pub model_source: ModelSource,
}

impl ApiConfig {
    /// Path to `api_config.json` inside the app data dir. Computed once in
    /// setup and cached on AppState (the `api_config_path` OnceLock) so the
    /// load/save helpers below stay `&Path`-based and need no `AppHandle`.
    /// Mirrors `ThemeSettings::resolve_path` exactly.
    pub fn resolve_path(app_data_dir: &std::path::Path) -> PathBuf {
        app_data_dir.join("api_config.json")
    }

    /// Load from disk, falling back to default (empty config, Local source)
    /// on any error (missing file, malformed JSON, IO). Persistence is
    /// best-effort: a corrupt `api_config.json` must never block app launch —
    /// the user just sees an empty profile list and re-enters their configs.
    /// Mirrors `ThemeSettings::load`.
    pub fn load(path: &std::path::Path) -> Self {
        match std::fs::read_to_string(path) {
            Ok(s) => serde_json::from_str::<ApiConfig>(&s).unwrap_or_default(),
            Err(_) => ApiConfig::default(),
        }
    }

    /// Persist atomically: temp file + rename (same pattern as `theme.rs` +
    /// `session.rs::save`). On failure we log and continue — the in-memory
    /// config still lives in AppState and can be re-saved on the next change.
    pub fn save(&self, path: &std::path::Path) {
        let tmp = path.with_extension("json.tmp");
        let json = match serde_json::to_string_pretty(self) {
            Ok(j) => j,
            Err(e) => {
                tracing::error!(error = %e, "api_config: serialize failed");
                return;
            }
        };
        if let Err(e) = std::fs::write(&tmp, json) {
            tracing::error!(error = %e, "api_config: write tmp failed");
            return;
        }
        // Windows rename over existing uses MOVEFILE_REPLACE_EXISTING — atomic.
        if let Err(e) = std::fs::rename(&tmp, path) {
            tracing::error!(error = %e, "api_config: rename failed");
            let _ = std::fs::remove_file(&tmp);
        }
    }

    /// Upsert a profile by `id`: replace if an existing profile has the same
    /// id, append otherwise. Returns the index the profile ended up at. The
    /// caller is responsible for saving + any model-swap side effects.
    pub fn upsert(&mut self, profile: ApiProfile) -> usize {
        if let Some(pos) = self.profiles.iter().position(|p| p.id == profile.id) {
            self.profiles[pos] = profile;
            pos
        } else {
            self.profiles.push(profile);
            self.profiles.len() - 1
        }
    }

    /// Remove a profile by id. Returns true if a profile was removed. If the
    /// removed profile was the active one, `active_profile_id` is cleared —
    /// the caller must handle the `ModelSource` downgrade (can't be Api with
    /// no active profile).
    pub fn remove(&mut self, id: &str) -> bool {
        let before = self.profiles.len();
        self.profiles.retain(|p| p.id != id);
        let removed = self.profiles.len() < before;
        if removed && self.active_profile_id.as_deref() == Some(id) {
            self.active_profile_id = None;
        }
        removed
    }

    /// Borrow the active profile, if any. `None` when no profile is active
    /// OR the active id points at a profile that no longer exists (orphaned
    /// id after a delete that didn't clear it — defensive).
    pub fn active_profile(&self) -> Option<&ApiProfile> {
        let id = self.active_profile_id.as_ref()?;
        self.profiles.iter().find(|p| &p.id == id)
    }
}

/// Sanitize a profile name into a stable id slug: lowercase, replace any
/// non-alphanumeric/`-`/`_` char with `-`, trim leading/trailing `-`. Returns
/// a fallback (`"profile"`) if the result is empty. Public so the IPC layer
/// can echo back the exact id `upsert` will use (the UI tracks entries by id).
/// Mirrors `codex::sanitize_stem` — same contract, different default.
pub fn sanitize_profile_id(name: &str) -> String {
    let slug: String = name
        .trim()
        .to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '-' })
        .collect();
    let slug = slug.trim_matches('-').to_owned();
    if slug.is_empty() {
        "profile".to_owned()
    } else {
        slug
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_local_with_no_profiles() {
        let c = ApiConfig::default();
        assert!(c.profiles.is_empty());
        assert_eq!(c.active_profile_id, None);
        assert_eq!(c.model_source, ModelSource::Local);
    }

    #[test]
    fn upsert_replaces_same_id_appends_new() {
        let mut c = ApiConfig::default();
        let p1 = ApiProfile {
            id: "zai".into(),
            name: "Z.AI".into(),
            endpoint: "https://api.z.ai/api/coding/paas/v4".into(),
            model: "glm-4.6".into(),
            api_key: "k1".into(),
            temperature: None,
        };
        assert_eq!(c.upsert(p1.clone()), 0);
        assert_eq!(c.profiles.len(), 1);

        // Same id → replace, not append.
        let mut p1b = p1.clone();
        p1b.api_key = "k2".into();
        assert_eq!(c.upsert(p1b), 0);
        assert_eq!(c.profiles.len(), 1);
        assert_eq!(c.profiles[0].api_key, "k2");

        // Different id → append.
        let p2 = ApiProfile {
            id: "nanogpt".into(),
            name: "NanoGPT".into(),
            endpoint: "https://nano-gpt.com/api/v1".into(),
            model: "gpt-4o-mini".into(),
            api_key: "k3".into(),
            temperature: Some(0.7),
        };
        assert_eq!(c.upsert(p2), 1);
        assert_eq!(c.profiles.len(), 2);
    }

    #[test]
    fn remove_clears_active_when_it_was_the_target() {
        let mut c = ApiConfig::default();
        c.upsert(ApiProfile {
            id: "zai".into(),
            name: "Z.AI".into(),
            endpoint: "x".into(),
            model: "m".into(),
            api_key: "k".into(),
            temperature: None,
        });
        c.active_profile_id = Some("zai".into());
        assert!(c.remove("zai"));
        assert!(c.profiles.is_empty());
        assert_eq!(c.active_profile_id, None, "active must clear on delete");
    }

    #[test]
    fn remove_keeps_active_when_deleting_a_different_profile() {
        let mut c = ApiConfig::default();
        c.upsert(ApiProfile {
            id: "a".into(),
            name: "A".into(),
            endpoint: "x".into(),
            model: "m".into(),
            api_key: "k".into(),
            temperature: None,
        });
        c.upsert(ApiProfile {
            id: "b".into(),
            name: "B".into(),
            endpoint: "x".into(),
            model: "m".into(),
            api_key: "k".into(),
            temperature: None,
        });
        c.active_profile_id = Some("a".into());
        assert!(c.remove("b"));
        assert_eq!(c.active_profile_id.as_deref(), Some("a"));
    }

    #[test]
    fn active_profile_finds_by_id() {
        let mut c = ApiConfig::default();
        c.upsert(ApiProfile {
            id: "zai".into(),
            name: "Z.AI".into(),
            endpoint: "x".into(),
            model: "m".into(),
            api_key: "k".into(),
            temperature: None,
        });
        c.active_profile_id = Some("zai".into());
        assert_eq!(c.active_profile().map(|p| p.name.as_str()), Some("Z.AI"));
    }

    #[test]
    fn active_profile_none_for_orphaned_id() {
        let mut c = ApiConfig::default();
        c.active_profile_id = Some("does-not-exist".into());
        assert!(c.active_profile().is_none());
    }

    #[test]
    fn sanitize_replaces_specials_with_dash() {
        assert_eq!(sanitize_profile_id("Z.AI personal"), "z-ai-personal");
        assert_eq!(sanitize_profile_id("NanoGPT!"), "nanogpt");
        assert_eq!(sanitize_profile_id("---"), "profile");
        assert_eq!(sanitize_profile_id(""), "profile");
        assert_eq!(sanitize_profile_id("  Work OpenRouter  "), "work-openrouter");
    }

    #[test]
    fn model_source_serializes_lowercase() {
        // serde rename_all = "lowercase" → matches the frontend's string check.
        let s = serde_json::to_string(&ModelSource::Api).unwrap();
        assert_eq!(s, "\"api\"");
        let s = serde_json::to_string(&ModelSource::Local).unwrap();
        assert_eq!(s, "\"local\"");
        // Round-trips through deserialization.
        let l: ModelSource = serde_json::from_str("\"local\"").unwrap();
        assert_eq!(l, ModelSource::Local);
    }

    #[test]
    fn config_roundtrips_through_json() {
        let mut c = ApiConfig::default();
        c.upsert(ApiProfile {
            id: "zai".into(),
            name: "Z.AI".into(),
            endpoint: "https://api.z.ai/api/coding/paas/v4".into(),
            model: "glm-4.6".into(),
            api_key: "secret".into(),
            temperature: Some(0.8),
        });
        c.active_profile_id = Some("zai".into());
        c.model_source = ModelSource::Api;

        let json = serde_json::to_string(&c).unwrap();
        let back: ApiConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.profiles.len(), 1);
        assert_eq!(back.profiles[0].api_key, "secret");
        assert_eq!(back.profiles[0].temperature, Some(0.8));
        assert_eq!(back.active_profile_id.as_deref(), Some("zai"));
        assert_eq!(back.model_source, ModelSource::Api);
    }

    #[test]
    fn legacy_config_without_new_fields_loads_with_defaults() {
        // A config missing `active_profile_id` or `model_source` (older shape)
        // must deserialize via #[serde(default)] rather than fail.
        let json = r#"{"profiles":[{"id":"x","name":"X","endpoint":"e","model":"m","api_key":"k"}]}"#;
        let c: ApiConfig = serde_json::from_str(json).expect("serde defaults fill missing fields");
        assert_eq!(c.profiles.len(), 1);
        assert_eq!(c.active_profile_id, None);
        assert_eq!(c.model_source, ModelSource::Local);
    }

    #[test]
    fn load_missing_file_returns_default() {
        // Graceful degradation: a missing api_config.json at boot → empty
        // default config, no panic. This is the common case until the user
        // authors their first profile.
        let bogus = std::path::Path::new("/this/does/not/exist/api_config.json");
        let c = ApiConfig::load(bogus);
        assert!(c.profiles.is_empty());
        assert_eq!(c.model_source, ModelSource::Local);
    }
}
