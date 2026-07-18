//! NPC runtime data model — Phase 2 stub (Games app Seam 5).
//!
//! These types model the deterministic NPC behavior sidecar that Phase 2
//! will implement (the UIE `app/` Character Runtime Engine philosophy — see
//! docs/games-app-design.md §4 reference table). The MVP ships the type
//! definitions only, so the Phase 2 engines have a place to land without
//! painting us into a corner.
//!
//! **Nothing in this module is wired into the game loop yet.** All fields
//! are `pub` so Phase 2 can construct + mutate freely; nothing here is
//! `#[allow(dead_code)]`-suppressed because the types are intentionally
//! forward-compat scaffolding (Rust only warns on unused items at the
//! module level if they're truly never referenced — the type defs here
//! will be referenced the moment Phase 2 lands).

use std::collections::HashMap;

use crate::schema::SchemaDelta;

/// One NPC's runtime state — the per-turn input to the (Phase 2) behavior
/// scorer. Mirrors UIE's `app/models/character.py:16` + `runtime.py`.
#[derive(Debug, Clone, Default)]
pub struct NpcState {
    /// Stable id matching `SimCard.start_npc_ids` + the `[CHARACTER_TURN:id]`
    /// handoff tag.
    pub id: String,
    /// Display name.
    pub name: String,
    /// Archetype bundle id (UIE state pack: "bartender", "guard", "vendor",
    /// "rival", "romantic_slow_burn"...). Phase 2 loads the matching
    /// `StatePack` from a registry.
    pub archetype: String,
    /// Emotional state map (dimensional: anger, affection, fear, joy, etc.).
    /// Each value is a 0.0-1.0 scalar with decay toward baseline (Phase 2).
    pub emotions: HashMap<String, f32>,
    /// The NPC's relationship to the player (and, Phase 3, to other NPCs).
    pub relationship: Relationship,
    /// Memory distortion profile — makes the NPC an unreliable narrator.
    /// UIE pattern (`uie_backend.py:550`): reliability, distortion_chance,
    /// forgets_names. Phase 2 uses this when the NPC recalls past events.
    pub memory_profile: MemoryProfile,
    /// The NPC's last computed behavior + render guidance (Phase 2 output).
    /// `None` until the first behavior-scoring pass runs.
    pub current_behavior: Option<BehaviorOutput>,
}

/// The 9-scalar relationship model (UIE `relationship.py:6`). Each is -1.0
/// to 1.0 (or 0.0 to 1.0 for ones that don't have a negative valence).
#[derive(Debug, Clone, Default)]
pub struct Relationship {
    pub trust: f32,
    pub respect: f32,
    pub affection: f32,
    pub attraction: f32,
    pub attachment: f32,
    pub resentment: f32,
    pub suspicion: f32,
    pub fear: f32,
    pub comfort: f32,
    /// Outstanding obligation (UIE `debt`). Positive = NPC feels the player
    /// owes them; negative = NPC owes the player.
    pub debt: f32,
}

/// Per-NPC memory distortion — the unreliable-narrator mechanic. Phase 2's
/// recall path multiplies recall fidelity by these.
#[derive(Debug, Clone)]
pub struct MemoryProfile {
    /// 0.0-1.0. 1.0 = perfect recall, 0.0 = confabulates freely.
    pub reliability: f32,
    /// 0.0-1.0. Probability any given recalled detail is distorted.
    pub distortion_chance: f32,
    /// If true, the NPC forgets proper names (uses descriptions instead).
    pub forgets_names: bool,
}

impl Default for MemoryProfile {
    fn default() -> Self {
        // Sane defaults: most NPCs remember things accurately.
        Self {
            reliability: 0.9,
            distortion_chance: 0.05,
            forgets_names: false,
        }
    }
}

/// The (Phase 2) behavior scorer's output for one NPC on one turn. Mirrors
/// UIE's `compiler_service.build_renderer_payload` — the chosen behavior,
/// observable cues the LLM is told to show, and what it's told to AVOID
/// (never narrate the hidden math).
#[derive(Debug, Clone)]
pub struct BehaviorOutput {
    /// The chosen behavior id (e.g. "comfort_player", "block_exit").
    pub behavior: String,
    /// Context-specific variant (e.g. "block_exit" → "physically_step_into_path").
    pub variant: String,
    /// Top-N dominant emotional states, sorted by intensity (for narration).
    pub dominant_states: Vec<String>,
    /// Observable physical cues the LLM should portray (UIE `SHOW_THROUGH_MAP`).
    /// E.g. ["body blocking", "stepping into path", "firm voice"].
    pub show_through: Vec<String>,
    /// What the LLM should AVOID portraying (UIE `AVOID_ALWAYS`):
    /// narrating internal state values, explaining hidden math, etc.
    pub avoid: Vec<String>,
    /// Detected internal conflict (top-2 behaviors within 15 points of each
    /// other). Dramatically improves NPC realism when present (UIE pattern).
    pub internal_conflict: Option<String>,
}

/// The schema-delta shape that NPC behavior consequences emit — applied to
/// the card's scoped game schema. Same type the schema engine + Wupi's
/// game-manager path produce (`SchemaDelta`). Re-exported here so Phase 2's
/// behavior scorer can return one type across all callers.
pub type BehaviorDelta = SchemaDelta;
