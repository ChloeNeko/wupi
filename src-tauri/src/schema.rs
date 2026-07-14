//! The world-state schema — "the schema IS the summarizer."
//!
//! A persistent, semi-structured record of the simulated world's current
//! state: a running narrative summary, recent salient events, and a flexible
//! key→value entity map (characters, inventory, locations, stats, quest
//! flags). Updated after every chat turn by the background state-delta pass
//! (see `schema_engine.rs`), which emits ONLY the changed keys as a
//! [`SchemaDelta`]; this module's [`WorldSchema::apply_delta`] merges that
//! delta into the global state.
//!
//! # The micro-delta contract
//!
//! The delta pass NEVER rewrites the whole schema. It emits a small JSON
//! object containing only the keys that changed this turn. This keeps the
//! delta pass fast (token-bound autoregression: a 20-token delta takes
//! ~0.6s vs ~60s for a full regen) and lets the model focus on what
//! actually moved rather than re-describing the whole world each turn.
//!
//! # Key removal
//!
//! `null` in a delta's `entities` map means "delete this key." A non-null
//! value means "set/overwrite." This is unambiguous for a string-valued
//! schema and JSON-native.
//!
//! # Persistence
//!
//! `world_schema.json` in the app data dir (sibling to `session.json`).
//! Atomic save (temp + fsync + rename), same pattern as
//! [`crate::session::Conversation::save`]. Loaded at startup into `AppState`.

use std::collections::HashMap;
use std::path::Path;

/// The persistent world-state schema. The single source of truth for the
/// simulated world's current state, maintained by the background delta pass.
///
/// Semi-structured by design: a fixed envelope (`summary`, `recent_events`)
/// gives the model a stable narrative anchor, while the flexible `entities`
/// map adapts to any scenario (fantasy inventory, sci-fi ship status, modern
/// day relationship tracker) without code changes. Keys are model-defined.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
pub struct WorldSchema {
    /// One-paragraph running narrative summary. The model rewrites this when
    /// the narrative arc shifts — NOT every turn. Carries the "where are we
    /// in the story" thread that the 6-message window can't hold alone.
    #[serde(default)]
    pub summary: String,

    /// Recent salient events, newest appended at the end. The model appends
    /// new events and may trim old ones in a delta. Bounded growth is the
    /// model's responsibility (the prompt instructs it to keep this list
    /// short); a Rust-side cap could be added later if it drifts.
    #[serde(default)]
    pub recent_events: Vec<String>,

    /// Flexible key→value store for hard data. Keys are model-defined and
    /// namespaced by convention (e.g. `"item.iron_sword"`,
    /// `"char.mira.trust"`, `"loc.current"`). Values are strings — structured
    /// enough to read programmatically, loose enough to hold anything.
    ///
    /// In a delta, a `None` value (JSON `null`) means "delete this key";
    /// `Some(v)` means "set/overwrite."
    #[serde(default)]
    pub entities: HashMap<String, String>,
}

impl WorldSchema {
    /// Deep-merge a micro-delta into self. The "native Rust merging"
    /// requirement — the model emits only changed keys, Rust applies them.
    ///
    /// Semantics:
    /// - `summary`: overwrite if the delta carries one (the model only emits
    ///   this when the narrative arc actually shifted).
    /// - `recent_events`: append the delta's events at the end (newest last).
    ///   No dedupe — the model is responsible for not re-emitting existing
    ///   events. Trimming old events is the model's job too (it sees the full
    ///   current list in the delta prompt and can drop stale ones by emitting
    ///   a replacement... actually no — append-only is the v1 contract; the
    ///   model can rewrite `summary` to fold old events in, and a future
    ///   "replace recent_events" signal could land if trimming becomes needed).
    /// - `entities`: for each (key, value) in the delta: `Some(v)` → upsert,
    ///   `None` → remove the key (no-op if it didn't exist).
    pub fn apply_delta(&mut self, delta: SchemaDelta) {
        if let Some(summary) = delta.summary {
            self.summary = summary;
        }
        if let Some(mut events) = delta.recent_events {
            self.recent_events.append(&mut events);
        }
        if let Some(ents) = delta.entities {
            for (key, value) in ents {
                match value {
                    Some(v) => {
                        self.entities.insert(key, v);
                    }
                    None => {
                        self.entities.remove(&key);
                    }
                }
            }
        }
    }

    /// Render the schema into a compact, prompt-friendly string for injection
    /// into the chat turn's `<world_state>` block. Compactness matters: this
    /// goes into the inter-turn region alongside the memory block, and every
    /// token is prefill cost. We emit the summary, the last few recent events
    /// (not all — the model doesn't need the deep history list in chat, that's
    /// what the delta pass sees in full), and the entities as `key: value`
    /// lines.
    ///
    /// Returns an empty string for an empty schema so the caller can skip
    /// emitting the `<world_state>` block entirely (matches the memory block's
    /// empty-skip behavior in `chat_format.rs`).
    pub fn render_for_prompt(&self) -> String {
        let empty = self.summary.trim().is_empty()
            && self.recent_events.is_empty()
            && self.entities.is_empty();
        if empty {
            return String::new();
        }

        let mut out = String::with_capacity(512);
        if !self.summary.trim().is_empty() {
            out.push_str("summary: ");
            out.push_str(self.summary.trim());
            out.push('\n');
        }
        // Cap recent events shown in chat at the last 5 — older events live
        // in the persisted schema + memory retrieval, not the chat prompt.
        let show_events = self.recent_events.len().saturating_sub(5);
        if !self.recent_events[show_events..].is_empty() {
            out.push_str("recent_events:\n");
            for ev in &self.recent_events[show_events..] {
                out.push_str("  - ");
                out.push_str(ev);
                out.push('\n');
            }
        }
        if !self.entities.is_empty() {
            out.push_str("entities:\n");
            // Sort keys for deterministic output (stable prompt = stable tokens).
            let mut keys: Vec<&String> = self.entities.keys().collect();
            keys.sort();
            for key in keys {
                out.push_str("  ");
                out.push_str(key);
                out.push_str(": ");
                out.push_str(&self.entities[key]);
                out.push('\n');
            }
        }
        // Trim the trailing newline — the caller wraps this in a tag block.
        out.trim_end().to_string()
    }

    /// Serialize for the delta pass's "current schema" prompt input. Pretty-
    /// printed JSON so the model can read it clearly; the schema is small.
    pub fn to_json_pretty(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Atomic save to `world_schema.json` (temp + fsync + rename, same pattern
    /// as `session::Conversation::save`). A crash mid-write can never truncate
    /// the existing file.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        let tmp_path = temp_path_for(path);
        let _ = std::fs::remove_file(&tmp_path); // clear stale temp

        {
            let mut file = std::fs::File::create(&tmp_path)?;
            std::io::Write::write_all(&mut file, json.as_bytes())?;
            std::io::Write::flush(&mut file)?;
            let _ = file.sync_all();
        }
        std::fs::rename(&tmp_path, path)
    }

    /// Load from `world_schema.json`. Returns an empty schema if the file
    /// doesn't exist yet (first run) — never an error for the NotFound case.
    pub fn load(path: &Path) -> std::io::Result<Self> {
        match std::fs::read_to_string(path) {
            Ok(text) => serde_json::from_str(&text)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(e),
        }
    }
}

/// A micro-delta against [`WorldSchema`]. All fields optional — the model
/// emits ONLY the keys that changed this turn. Omitted fields = unchanged.
///
/// Deserialized from the JSON object the schema-delta model pass emits. The
/// `entities` field's inner `Option<String>` is load-bearing: outer `Option`
/// = "did any entity change?", inner `Option` = "is this a delete (`null`)
/// or a set (`Some`)?". `serde` deserializes JSON `null` to `None` and a
/// string to `Some(string)`, giving us the unambiguous delete-vs-set signal
/// for free.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
pub struct SchemaDelta {
    #[serde(default)]
    pub summary: Option<String>,
    #[serde(default)]
    pub recent_events: Option<Vec<String>>,
    #[serde(default)]
    pub entities: Option<HashMap<String, Option<String>>>,
}

impl SchemaDelta {
    /// Parse a model-emitted string into a delta. Tolerant of three layers of
    /// wrapping the model may apply:
    /// 1. The Gemma4 channel protocol (`<|channel>thought\n...<channel|>reply`).
    ///    The model emits this protocol for ALL output (including the schema
    ///    delta pass, which is instructed to emit raw JSON). The JSON lives in
    ///    the REPLY channel — the text after the last `<channel|>` marker.
    /// 2. Markdown fences (```` ```json ... ``` ````) — stripped if present.
    /// 3. Surrounding whitespace.
    ///
    /// Runtime-discovered 2026-07-13: the delta pass emitted
    /// `<|channel>thought\n<channel|>{}` — a valid empty delta `{}` wrapped in
    /// the channel protocol. Without extracting the reply channel, serde saw
    /// `<|channel>...` and bailed at column 1.
    pub fn from_model_output(raw: &str) -> Result<Self, serde_json::Error> {
        let reply = extract_reply_channel(raw);
        let cleaned = strip_markdown_fences(reply).trim();
        serde_json::from_str(cleaned)
    }
}

/// Extract the reply channel from Gemma4 protocol output. The model emits
/// `<|channel>thought\n...<channel|>reply` — the thought channel (internal
/// reasoning) comes first, closed by `<channel|>`, then the reply text
/// follows. The JSON delta is in the reply.
///
/// Uses `rsplit_once("<channel|>")` to take everything after the LAST closing
/// marker. This correctly handles:
/// - Protocol-wrapped output (the common case): the thought block is
///   discarded, the reply JSON is kept.
/// - Thought-only output (no reply): returns empty string → parse fails
///   gracefully (the repair prompt or error path takes over).
/// - Raw JSON with no protocol wrapping: returns the whole string unchanged
///   (the rare case where the model emits JSON directly).
///
/// This mirrors `chat_format.rs::Gemma4Format::parse_output`'s split-on-
/// `<channel|>` logic, specialized to the schema's "I only want the reply"
/// need (parse_output splits into both channels; we discard thought entirely).
fn extract_reply_channel(raw: &str) -> &str {
    match raw.rsplit_once("<channel|>") {
        Some((_, reply)) => reply,
        None => raw,
    }
}

/// Strip ```json ... ``` markdown fences if present. The model is told to
/// emit raw JSON but may wrap it anyway; this is cheaper than fighting the
/// model and more robust than erroring.
fn strip_markdown_fences(s: &str) -> &str {
    let trimmed = s.trim();
    if let Some(rest) = trimmed.strip_prefix("```json") {
        if let Some(body) = rest.strip_suffix("```") {
            return body;
        }
    }
    if let Some(rest) = trimmed.strip_prefix("```") {
        if let Some(body) = rest.strip_suffix("```") {
            return body;
        }
    }
    trimmed
}

/// Build a sibling temp-file path for an atomic save. Mirrors
/// `session::temp_path_for` — same directory/volume so `rename` is atomic.
fn temp_path_for(path: &Path) -> std::path::PathBuf {
    let mut name = path
        .file_name()
        .map(std::ffi::OsString::from)
        .unwrap_or_else(|| std::ffi::OsString::from("wupi.tmp"));
    name.push(".tmp");
    path.with_file_name(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_delta_upserts_entities() {
        let mut schema = WorldSchema::default();
        let delta = SchemaDelta {
            summary: None,
            recent_events: None,
            entities: Some(HashMap::from([
                ("iron_sword".to_string(), Some("acquired".to_string())),
                ("loc.current".to_string(), Some("tavern".to_string())),
            ])),
        };
        schema.apply_delta(delta);
        assert_eq!(schema.entities.get("iron_sword"), Some(&"acquired".to_string()));
        assert_eq!(schema.entities.get("loc.current"), Some(&"tavern".to_string()));
    }

    #[test]
    fn apply_delta_null_deletes_key() {
        let mut schema = WorldSchema {
            summary: String::new(),
            recent_events: vec![],
            entities: HashMap::from([
                ("iron_sword".to_string(), "acquired".to_string()),
                ("loc.current".to_string(), "tavern".to_string()),
            ]),
        };
        // Drop the sword, move locations.
        let delta = SchemaDelta {
            summary: None,
            recent_events: None,
            entities: Some(HashMap::from([
                ("iron_sword".to_string(), None), // delete
                ("loc.current".to_string(), Some("forest".to_string())),
            ])),
        };
        schema.apply_delta(delta);
        assert!(!schema.entities.contains_key("iron_sword"), "null should delete");
        assert_eq!(schema.entities.get("loc.current"), Some(&"forest".to_string()));
    }

    #[test]
    fn apply_delta_null_on_missing_key_is_noop() {
        let mut schema = WorldSchema::default();
        let delta = SchemaDelta {
            summary: None,
            recent_events: None,
            entities: Some(HashMap::from([("ghost".to_string(), None)])),
        };
        schema.apply_delta(delta);
        assert!(schema.entities.is_empty());
    }

    #[test]
    fn apply_delta_appends_recent_events() {
        let mut schema = WorldSchema {
            summary: String::new(),
            recent_events: vec!["entered tavern".to_string()],
            entities: HashMap::new(),
        };
        let delta = SchemaDelta {
            summary: None,
            recent_events: Some(vec!["ordered ale".to_string(), "heard rumor".to_string()]),
            entities: None,
        };
        schema.apply_delta(delta);
        assert_eq!(
            schema.recent_events,
            vec!["entered tavern", "ordered ale", "heard rumor"]
        );
    }

    #[test]
    fn apply_delta_overwrites_summary() {
        let mut schema = WorldSchema {
            summary: "old summary".to_string(),
            recent_events: vec![],
            entities: HashMap::new(),
        };
        let delta = SchemaDelta {
            summary: Some("new summary".to_string()),
            recent_events: None,
            entities: None,
        };
        schema.apply_delta(delta);
        assert_eq!(schema.summary, "new summary");
    }

    #[test]
    fn apply_delta_empty_is_noop() {
        let mut schema = WorldSchema {
            summary: "kept".to_string(),
            recent_events: vec!["kept".to_string()],
            entities: HashMap::from([("k".to_string(), "v".to_string())]),
        };
        schema.apply_delta(SchemaDelta::default());
        assert_eq!(schema.summary, "kept");
        assert_eq!(schema.recent_events, vec!["kept"]);
        assert_eq!(schema.entities.get("k"), Some(&"v".to_string()));
    }

    #[test]
    fn from_model_output_parses_clean_json() {
        let raw = r#"{"summary":"new","entities":{"x":"1"}}"#;
        let delta = SchemaDelta::from_model_output(raw).unwrap();
        assert_eq!(delta.summary.as_deref(), Some("new"));
        assert_eq!(
            delta.entities.unwrap().get("x"),
            Some(&Some("1".to_string()))
        );
    }

    #[test]
    fn from_model_output_strips_markdown_fences() {
        let raw = "```json\n{\"summary\":\"x\"}\n```";
        let delta = SchemaDelta::from_model_output(raw).unwrap();
        assert_eq!(delta.summary.as_deref(), Some("x"));
    }

    #[test]
    fn from_model_output_null_entity_value_is_delete_signal() {
        // JSON null deserializes to Option::None — the delete signal.
        let raw = r#"{"entities":{"drop_me":null}}"#;
        let delta = SchemaDelta::from_model_output(raw).unwrap();
        assert_eq!(delta.entities.unwrap().get("drop_me"), Some(&None));
    }

    #[test]
    fn from_model_output_strips_gemma4_channel_protocol() {
        // Regression for the 2026-07-13 runtime failure: the delta pass
        // emitted `<|channel>thought\n<channel|>{}` — a valid empty delta
        // wrapped in the Gemma4 channel protocol. Without extracting the
        // reply channel serde saw `<|channel>...` and bailed at column 1.
        let raw = "<|channel>thought\n<channel|>{}";
        let delta = SchemaDelta::from_model_output(raw).unwrap();
        assert!(delta.summary.is_none());
        assert!(delta.recent_events.is_none());
        assert!(delta.entities.is_none());
    }

    #[test]
    fn from_model_output_extracts_json_after_thought_channel() {
        // The realistic case: model thinks briefly, then emits the JSON delta
        // in the reply channel.
        let raw = "<|channel>thought\nI should record the sword pickup.\n<channel|>{\"entities\":{\"item.iron_sword\":\"acquired\"}}";
        let delta = SchemaDelta::from_model_output(raw).unwrap();
        assert_eq!(
            delta.entities.unwrap().get("item.iron_sword"),
            Some(&Some("acquired".to_string()))
        );
    }

    #[test]
    fn from_model_output_channel_protocol_with_markdown_fence() {
        // Double wrapping: channel protocol + markdown fence. The reply
        // channel is extracted first, then the fence is stripped.
        let raw = "<|channel>thought\n<channel|>```json\n{\"summary\":\"updated\"}\n```";
        let delta = SchemaDelta::from_model_output(raw).unwrap();
        assert_eq!(delta.summary.as_deref(), Some("updated"));
    }

    #[test]
    fn from_model_output_thought_only_no_reply_is_error() {
        // The model emitted only a thought channel (no reply). Extraction
        // returns empty → parse fails gracefully. The repair prompt or error
        // path takes over; the schema is left unchanged for that turn.
        let raw = "<|channel>thought\nthinking...\n<channel|>";
        assert!(SchemaDelta::from_model_output(raw).is_err());
    }

    #[test]
    fn from_model_output_raw_json_without_protocol_passes_through() {
        // No channel markers at all — the model emitted JSON directly (rare
        // but possible). rsplit_once finds no `<channel|>` and returns the
        // whole string unchanged.
        let raw = r#"{"recent_events":["saw a fox"]}"#;
        let delta = SchemaDelta::from_model_output(raw).unwrap();
        assert_eq!(delta.recent_events.unwrap(), vec!["saw a fox".to_string()]);
    }

    #[test]
    fn render_for_prompt_empty_schema_is_empty_string() {
        assert_eq!(WorldSchema::default().render_for_prompt(), "");
    }

    #[test]
    fn render_for_prompt_caps_recent_events_at_five() {
        let schema = WorldSchema {
            summary: String::new(),
            recent_events: (0..10).map(|i| format!("event{i}")).collect(),
            entities: HashMap::new(),
        };
        let rendered = schema.render_for_prompt();
        // Only the last 5 events should appear.
        assert!(rendered.contains("event5"));
        assert!(rendered.contains("event9"));
        assert!(!rendered.contains("event4"));
    }

    #[test]
    fn save_load_roundtrip() {
        let dir = std::env::temp_dir();
        let path = dir.join("wupi_schema_test.json");
        let _ = std::fs::remove_file(&path);
        let schema = WorldSchema {
            summary: "test summary".to_string(),
            recent_events: vec!["e1".to_string()],
            entities: HashMap::from([("k".to_string(), "v".to_string())]),
        };
        schema.save(&path).unwrap();
        let loaded = WorldSchema::load(&path).unwrap();
        assert_eq!(loaded.summary, "test summary");
        assert_eq!(loaded.recent_events, vec!["e1"]);
        assert_eq!(loaded.entities.get("k"), Some(&"v".to_string()));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_missing_file_returns_default() {
        let path = std::env::temp_dir().join("wupi_schema_does_not_exist_xyz.json");
        let _ = std::fs::remove_file(&path);
        let loaded = WorldSchema::load(&path).unwrap();
        assert!(loaded.summary.is_empty());
        assert!(loaded.recent_events.is_empty());
        assert!(loaded.entities.is_empty());
    }
}
