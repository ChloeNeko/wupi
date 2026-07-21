//! Pure-Rust shape validator for `SchemaDelta` — the §1B defense layer.
//!
//! Sits between `SchemaDelta::from_model_output` (parses JSON) and
//! `WorldSchema::apply_delta` (commits to state). Catches structurally
//! malformed deltas BEFORE they touch world state, at zero LLM cost.
//!
//! ## Why this exists (the §1B argument)
//!
//! Retries (the 3-pass repair loop in `schema_engine.rs`) defeat JSON
//! *malformedness* — they get the model to emit something that parses. But
//! retries do NOT defeat *hallucination*: a model that confidently emits
//! `{"entities":{"char.mira.trust":"banana"}}` will, on repair, emit something
//! equally wrong but parseable. Validation by *shape* is the only structural
//! defense, and it's a microsecond Rust check, not a 5–8s LLM pass.
//!
//! ## Contract (intentionally narrow at v1)
//!
//! The validator is **deliberately permissive at v1**. The entity map is
//! free-form by design (`schema.rs` line 56: "Keys are model-defined and
//! namespaced by convention"). Tight type-shape enforcement (`char.*.trust`
//! must be f64 in [0,1]) would over-constrain scenarios the engine hasn't
//! seen yet. What v1 enforces:
//!
//! 1. **Structural integrity.** Keys are non-empty, bounded length, no
//!    control chars. Values bounded length. (Defense against garbage deltas
//!    the model sometimes emits under prompt corruption.)
//! 2. **Recent-events integrity.** Each event is non-empty and bounded length.
//! 3. **Summary integrity.** Bounded length (no model rambling that bloats
//!    every subsequent turn's prompt).
//!
//! What v1 deliberately does NOT enforce (future phases):
//! - Numeric range on `char.*.trust`-style keys (deferred until Phase 2/3
//!   adds a typed entity schema).
//! - `loc.*` references known spatial nodes (deferred until Phase 2 spatial
//!   engine exists; the validator takes an optional `known_nodes` set that's
//!   `None` today).
//! - `secret.*` exposure-status vocabulary (Phase 3 secrets).
//!
//! The `ValidationContext` struct is the extension point: future phases add
//! fields here (typed entity specs, spatial graph, etc.) without touching the
//! validator's public API.
//!
//! ## Failure mode
//!
//! `validate` returns `Result<(), ValidationFailure>`. The 3-pass loop in
//! `schema_engine.rs` treats a validation failure exactly like a parse
//! failure: the delta is re-attempted with a repair prompt that INCLUDES the
//! validation error message (so the model can correct it). If all 3 passes
//! fail validation, the delta enters the failure queue (`lib.rs`'s
//! `failed_delta_queue`) and folds into the next turn's prompt — never
//! silently dropped.

use crate::schema::SchemaDelta;

/// Maximum lengths. Generous: the goal is to catch runaway output, not to
/// constrain legitimate play. Tuned from the existing schema's observed
/// ranges (summary ~200-500 chars, events ~40-200 chars, entity values
/// ~5-100 chars).
const MAX_SUMMARY_LEN: usize = 4_000;
const MAX_EVENTS_PER_DELTA: usize = 20;
const MAX_EVENT_LEN: usize = 1_000;
const MAX_ENTITY_KEYS_PER_DELTA: usize = 50;
const MAX_KEY_LEN: usize = 200;
const MAX_VALUE_LEN: usize = 4_000;

/// Future-phase extension point. Today the validator runs with
/// `ValidationContext::default()` (no spatial graph, no typed entity specs).
/// Phase 2 adds `known_nodes: Option<&HashSet<String>>` so `loc.*` keys can
/// be checked against the spatial graph. Phase 3 adds typed entity specs so
/// `char.*.trust` can be range-checked. Each addition is a new field here,
/// not an API break.
#[derive(Debug, Clone, Default)]
pub struct ValidationContext<'a> {
    /// Known spatial node ids. When `Some`, `loc.*` values (and `loc.*`-shaped
    /// keys) must reference a node in this set. `None` today (Phase 2 fills it).
    #[allow(dead_code)]
    pub known_nodes: Option<&'a std::collections::HashSet<String>>,
}

/// Why a delta was rejected. Carries enough detail that the schema engine can
/// surface it to the model in the next repair pass (so the model can correct
/// the specific issue, not just "try again"). `Display` impl produces a
/// model-facing one-liner.
///
/// `reason` is owned `String` (not `&'static str`) because some variants
/// interpolate length values. The validator runs once per turn at most, so
/// the allocation cost is irrelevant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationFailure {
    /// A key was empty, too long, or contained control characters.
    InvalidKey { key: String, reason: String },
    /// A value was too long or contained control characters (excluding
    /// newlines, which are legitimately part of multi-line prose values).
    InvalidValue { key: String, reason: String },
    /// The delta carried more events than the per-delta cap. Prevents the
    /// model from dumping a whole scene as "events" and bloating every
    /// future turn's prompt.
    TooManyEvents { count: usize },
    /// An individual event was empty or too long.
    InvalidEvent { index: usize, reason: String },
    /// The summary exceeded the length cap.
    SummaryTooLong { len: usize },
    /// The delta carried more entity keys than the per-delta cap.
    TooManyEntityKeys { count: usize },
}

impl std::fmt::Display for ValidationFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidKey { key, reason } => write!(
                f,
                "invalid entity key {key:?}: {reason}"
            ),
            Self::InvalidValue { key, reason } => write!(
                f,
                "invalid value for entity key {key:?}: {reason}"
            ),
            Self::TooManyEvents { count } => write!(
                f,
                "delta emitted {count} recent_events; max {MAX_EVENTS_PER_DELTA} (keep only genuinely new salient events)"
            ),
            Self::InvalidEvent { index, reason } => write!(
                f,
                "recent_events[{index}] {reason}"
            ),
            Self::SummaryTooLong { len } => write!(
                f,
                "summary is {len} chars; max {MAX_SUMMARY_LEN} (rewrite concisely)"
            ),
            Self::TooManyEntityKeys { count } => write!(
                f,
                "delta emitted {count} entity keys; max {MAX_ENTITY_KEYS_PER_DELTA} (emit only changed keys)"
            ),
        }
    }
}

/// Validate a parsed delta against the structural rules. Pure, allocation-
/// light (one `.len()` per field, no cloning of values). Returns the first
/// failure encountered; the 3-pass loop in `schema_engine.rs` surfaces it to
/// the model on the next repair pass.
///
/// `ctx` is the extension point for future-phase typed validation (spatial
/// nodes, entity spec ranges). Pass `ValidationContext::default()` today.
pub fn validate(delta: &SchemaDelta, _ctx: &ValidationContext<'_>) -> Result<(), ValidationFailure> {
    // Summary: cap runaway length.
    if let Some(summary) = &delta.summary {
        let len = summary.chars().count();
        if len > MAX_SUMMARY_LEN {
            return Err(ValidationFailure::SummaryTooLong { len });
        }
    }

    // Recent events: per-delta count cap + per-event length + non-empty.
    if let Some(events) = &delta.recent_events {
        if events.len() > MAX_EVENTS_PER_DELTA {
            return Err(ValidationFailure::TooManyEvents { count: events.len() });
        }
        for (i, ev) in events.iter().enumerate() {
            if ev.trim().is_empty() {
                return Err(ValidationFailure::InvalidEvent {
                    index: i,
                    reason: "is empty (emit only genuine salient events)".to_string(),
                });
            }
            let len = ev.chars().count();
            if len > MAX_EVENT_LEN {
                return Err(ValidationFailure::InvalidEvent {
                    index: i,
                    reason: format!("is {len} chars; max {MAX_EVENT_LEN}"),
                });
            }
        }
    }

    // Entities: per-delta count cap + per-key/value shape.
    if let Some(ents) = &delta.entities {
        if ents.len() > MAX_ENTITY_KEYS_PER_DELTA {
            return Err(ValidationFailure::TooManyEntityKeys { count: ents.len() });
        }
        for (key, value_opt) in ents {
            // Key shape: non-empty, bounded length, no control chars.
            let key_len = key.chars().count();
            if key.trim().is_empty() {
                return Err(ValidationFailure::InvalidKey {
                    key: key.clone(),
                    reason: "is empty".to_string(),
                });
            }
            if key_len > MAX_KEY_LEN {
                return Err(ValidationFailure::InvalidKey {
                    key: key.clone(),
                    reason: format!("is {key_len} chars; max {MAX_KEY_LEN}"),
                });
            }
            if has_control_chars(key) {
                return Err(ValidationFailure::InvalidKey {
                    key: key.clone(),
                    reason: "contains control characters".to_string(),
                });
            }

            // Value shape (only when Some — None is the delete signal, always
            // valid). Allow newlines in values (multi-line prose is fine);
            // reject other control chars + cap length.
            if let Some(value) = value_opt {
                let val_len = value.chars().count();
                if val_len > MAX_VALUE_LEN {
                    return Err(ValidationFailure::InvalidValue {
                        key: key.clone(),
                        reason: format!("is {val_len} chars; max {MAX_VALUE_LEN}"),
                    });
                }
                if has_disallowed_control_chars(value) {
                    return Err(ValidationFailure::InvalidValue {
                        key: key.clone(),
                        reason: "contains control characters (newlines are allowed; strip other control chars)".to_string(),
                    });
                }
            }

            // Phase 2 will add: if ctx.known_nodes is Some and key starts
            // with "loc.", the value must be in known_nodes. Skipped today
            // (the spatial graph doesn't exist yet — would over-constrain).
        }
    }

    Ok(())
}

/// True if the string contains any ASCII control char (excluding newline
/// `\n` and tab `\t`, which are legitimate in prose values).
fn has_disallowed_control_chars(s: &str) -> bool {
    s.chars().any(|c| {
        let code = c as u32;
        // Allow tab (0x09) and newline (0x0A). Carriage return (0x0D) is
        // tolerated (Windows line endings). Reject other C0 control chars
        // and DEL (0x7F). Unicode control chars beyond ASCII are allowed
        // (they're rare in prose and over-restricting them breaks non-Latin
        // scripts).
        (code < 0x20 && code != 0x09 && code != 0x0A && code != 0x0D) || code == 0x7F
    })
}

/// Stricter variant for KEYS: keys should never contain control chars at all
/// (including newlines — a key with a newline is always model noise).
fn has_control_chars(s: &str) -> bool {
    s.chars().any(|c| {
        let code = c as u32;
        code < 0x20 || code == 0x7F
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // The default context (no spatial graph, no typed specs) — what every
    // caller passes today. Tests that don't specifically exercise future-
    // phase features use this.
    fn ctx() -> ValidationContext<'static> {
        ValidationContext::default()
    }

    // ---------- ACCEPT cases (the validator is deliberately permissive) ----------

    #[test]
    fn accepts_empty_delta() {
        let delta = SchemaDelta::default();
        assert!(validate(&delta, &ctx()).is_ok());
    }

    #[test]
    fn accepts_normal_entities() {
        let mut ents = HashMap::new();
        ents.insert("item.iron_sword".to_string(), Some("acquired".to_string()));
        ents.insert("char.mira.trust".to_string(), Some("0.85".to_string()));
        ents.insert("loc.current".to_string(), Some("tavern".to_string()));
        let delta = SchemaDelta {
            summary: None,
            recent_events: None,
            entities: Some(ents),
        };
        assert!(validate(&delta, &ctx()).is_ok());
    }

    #[test]
    fn accepts_delete_signal() {
        // None = delete key. Always valid regardless of key shape (it's
        // removing, not setting).
        let mut ents = HashMap::new();
        ents.insert("item.iron_sword".to_string(), None);
        let delta = SchemaDelta {
            summary: None,
            recent_events: None,
            entities: Some(ents),
        };
        assert!(validate(&delta, &ctx()).is_ok());
    }

    #[test]
    fn accepts_multi_line_value() {
        // Newlines in values are fine (multi-line prose, e.g. a long
        // character note). The pre-existing convention in schema.rs has
        // no rule against it.
        let mut ents = HashMap::new();
        ents.insert(
            "char.mira.notes".to_string(),
            Some("Line one.\nLine two.\nLine three.".to_string()),
        );
        let delta = SchemaDelta {
            summary: None,
            recent_events: None,
            entities: Some(ents),
        };
        assert!(validate(&delta, &ctx()).is_ok());
    }

    #[test]
    fn accepts_unicode_in_value() {
        // Non-Latin scripts must pass — over-restricting Unicode control
        // chars would break play in non-English scenarios.
        let mut ents = HashMap::new();
        ents.insert(
            "item.katana".to_string(),
            Some("刀".to_string()),
        );
        let delta = SchemaDelta {
            summary: None,
            recent_events: None,
            entities: Some(ents),
        };
        assert!(validate(&delta, &ctx()).is_ok());
    }

    #[test]
    fn accepts_reasonable_summary_and_events() {
        let delta = SchemaDelta {
            summary: Some("The party reached the dungeon entrance.".to_string()),
            recent_events: Some(vec![
                "Mira joined the party.".to_string(),
                "Found a glowing key.".to_string(),
            ]),
            entities: None,
        };
        assert!(validate(&delta, &ctx()).is_ok());
    }

    // ---------- REJECT cases (structural corruption) ----------

    #[test]
    fn rejects_empty_key() {
        let mut ents = HashMap::new();
        ents.insert("   ".to_string(), Some("v".to_string()));
        let delta = SchemaDelta {
            summary: None,
            recent_events: None,
            entities: Some(ents),
        };
        let err = validate(&delta, &ctx()).unwrap_err();
        assert_eq!(
            err,
            ValidationFailure::InvalidKey {
                key: "   ".to_string(),
                reason: "is empty".to_string()
            }
        );
    }

    #[test]
    fn rejects_key_with_control_chars() {
        // A newline in a key is always model noise.
        let mut ents = HashMap::new();
        ents.insert("bad\nkey".to_string(), Some("v".to_string()));
        let delta = SchemaDelta {
            summary: None,
            recent_events: None,
            entities: Some(ents),
        };
        assert!(matches!(
            validate(&delta, &ctx()),
            Err(ValidationFailure::InvalidKey { .. })
        ));
    }

    #[test]
    fn rejects_value_with_disallowed_control_chars() {
        // Null byte in value = corruption. Newlines would be allowed; null
        // is not.
        let mut ents = HashMap::new();
        ents.insert(
            "char.mira.notes".to_string(),
            Some("corrupt\x00value".to_string()),
        );
        let delta = SchemaDelta {
            summary: None,
            recent_events: None,
            entities: Some(ents),
        };
        assert!(matches!(
            validate(&delta, &ctx()),
            Err(ValidationFailure::InvalidValue { .. })
        ));
    }

    #[test]
    fn rejects_too_many_events() {
        let events: Vec<String> = (0..(MAX_EVENTS_PER_DELTA + 1))
            .map(|i| format!("event {i}"))
            .collect();
        let delta = SchemaDelta {
            summary: None,
            recent_events: Some(events),
            entities: None,
        };
        let err = validate(&delta, &ctx()).unwrap_err();
        assert!(matches!(
            err,
            ValidationFailure::TooManyEvents { count } if count == MAX_EVENTS_PER_DELTA + 1
        ));
    }

    #[test]
    fn rejects_empty_event() {
        let delta = SchemaDelta {
            summary: None,
            recent_events: Some(vec!["   ".to_string()]),
            entities: None,
        };
        let err = validate(&delta, &ctx()).unwrap_err();
        assert!(matches!(
            err,
            ValidationFailure::InvalidEvent { index: 0, .. }
        ));
    }

    #[test]
    fn rejects_too_many_entity_keys() {
        let mut ents = HashMap::new();
        for i in 0..(MAX_ENTITY_KEYS_PER_DELTA + 1) {
            ents.insert(format!("k{i}"), Some("v".to_string()));
        }
        let delta = SchemaDelta {
            summary: None,
            recent_events: None,
            entities: Some(ents),
        };
        let err = validate(&delta, &ctx()).unwrap_err();
        assert!(matches!(
            err,
            ValidationFailure::TooManyEntityKeys { count } if count == MAX_ENTITY_KEYS_PER_DELTA + 1
        ));
    }

    #[test]
    fn rejects_runaway_summary() {
        let delta = SchemaDelta {
            summary: Some("x".repeat(MAX_SUMMARY_LEN + 1)),
            recent_events: None,
            entities: None,
        };
        let err = validate(&delta, &ctx()).unwrap_err();
        assert!(matches!(err, ValidationFailure::SummaryTooLong { .. }));
    }

    // ---------- Display impl (used in repair prompts) ----------

    #[test]
    fn display_produces_model_facing_message() {
        let fail = ValidationFailure::InvalidKey {
            key: "bad key".to_string(),
            reason: "contains control characters".to_string(),
        };
        let msg = format!("{fail}");
        assert!(msg.contains("\"bad key\""));
        assert!(msg.contains("control characters"));
    }

    #[test]
    fn display_length_failure_shows_numbers() {
        let fail = ValidationFailure::SummaryTooLong { len: 4321 };
        let msg = format!("{fail}");
        assert!(msg.contains("4321"));
        assert!(msg.contains(&MAX_SUMMARY_LEN.to_string()));
    }
}
