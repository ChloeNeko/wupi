//! Narrator system-prompt builder (Games app Seam 3).
//!
//! The narrator is the AI's role in a roleplay game: it portrays the world,
//! the environment, the sensory detail, and (for MVP) the named NPCs'
//! observable behavior: but it is forbidden from speaking for the player
//! or generating the player's actions. The contract is borrowed from UIE's
//! `omniscientEngine.js:8-28` (the "narrator agency split"), adapted to
//! WUPI's strict-XML-prompt aesthetic (Prime Directive §1B.3).
//!
//! # Bracket-command protocol
//!
//! The narrator emits bracket commands alongside its prose so the engine
//! can route structured events to the UI deterministically:
//!
//! - `[CHARACTER_TURN:npc_id]`: signals an NPC is about to speak; the
//!   Phase 2 NPC sidecar will produce the line. For MVP the narrator is
//!   told to skip this and instead narrate the NPC's reaction in prose
//!   (documented MVP compromise: true handoffs need the Phase 2 engine).
//! - `[OBJECT id=iron_chest state=open]`: an object's state changed.
//! - `[FX rain]`, `[FX letterbox]`, `[FX shake-heavy]`: a scene-FX class
//!   should activate. The names match UIE's `sceneEffects.js` vocabulary
//!   so the eventual UI port is direct.
//!
//! The parser lives in `stream_filter.rs::BracketCommand`.

use crate::sim_card::SimCard;

/// Build the narrator system prompt for a roleplay card. The prompt is
/// injected as the `<|turn>system` block of the Gemma4 chat format. It tells
/// the model:
///   1. It is the Narrator, not any character.
///   2. It must never speak for the player (Alex): third-person references
///      only, never decide what Alex does/says/feels.
///   3. The setting + tone (from the card's `<scenario>` block).
///   4. The bracket-command protocol (see module doc).
///   5. The current `<world_state>` (passed in by the caller, scoped to the
///      card's schema).
pub fn build_narrator_system_prompt(
    card: &SimCard,
    world_state: Option<&str>,
) -> String {
    let mut out = String::with_capacity(2048);

    out.push_str("<narrator_role>\n");
    out.push_str(NARRATOR_CORE);
    out.push_str("\n</narrator_role>\n\n");

    // Scenario context (setting + tone). Always present on a roleplay card;
    // guards against a malformed card missing `<scenario>`.
    out.push_str("<scenario>\n");
    if let Some(setting) = card.setting.as_deref() {
        out.push_str("setting: ");
        out.push_str(setting.trim());
        out.push_str("\n\n");
    }
    if let Some(tone) = card.tone.as_deref() {
        out.push_str("tone: ");
        out.push_str(tone.trim());
        out.push_str("\n\n");
    }
    if !card.start_npc_ids.is_empty() {
        out.push_str("present_npcs: ");
        out.push_str(&card.start_npc_ids.join(", "));
        out.push_str("\n");
    }
    out.push_str("</scenario>\n\n");

    // Bracket-command protocol.
    out.push_str("<bracket_commands>\n");
    out.push_str(BRACKET_PROTOCOL);
    out.push_str("\n</bracket_commands>\n\n");

    // World state (card-scoped schema snapshot: what's true right now).
    if let Some(state) = world_state {
        if !state.trim().is_empty() {
            out.push_str("<world_state>\n");
            out.push_str(state.trim());
            out.push_str("\n</world_state>\n\n");
        }
    }

    // DELIBERATELY last: closest to the user input - so it's the loudest
    // signal the model sees when generating. The GameEngine's KV cache may
    // hold residual state from a PRIOR card (the 2026-07-18 "Alex
    // hallucination": the cyberpunk narrator used the dungeon protagonist's
    // name). Gemma 4, like all transformer LLMs, weights recent tokens
    // heavily; putting the explicit card-identity reinforcement at the tail
    // overrides those lingering vibes. Same principle that made §2O's persona
    // injection work: explicit, structured, recently-positioned context wins
    // over implicit residual state.
    out.push_str("<active_reality>\n");
    out.push_str(&format!(
        "You are narrating {}, NOT any other scenario. ",
        card.name.trim(),
    ));
    if let Some(name) = card.protagonist_name.as_deref() {
        out.push_str(&format!(
            "The protagonist is {name}: use this name exclusively; never use a different protagonist's name. "
        ));
    } else {
        out.push_str("Refer to the protagonist generically (e.g. \"the traveler\"); never invent or import a protagonist name. ");
    }
    if let Some(setting) = card.setting.as_deref() {
        // Brief recap (full setting already lives in <scenario> above). The
        // recap here is the recency-reinforcement, not the source of truth.
        let brief: String = setting.trim().chars().take(160).collect();
        out.push_str(&format!("Setting recap: {brief}… "));
    }
    out.push_str("Do NOT reference characters, locations, items, or elements from any other scenario: only what belongs to this one.\n");
    out.push_str("</active_reality>\n\n");

    out
}

/// The narrator's ground-truth identity + the player contract. Kept as a
/// const so it's byte-identical across cards (the card-specific bits are the
/// scenario + world_state sections appended after).
const NARRATOR_CORE: &str = "\
You are the NARRATOR of this scenario: not a character, not the player.

Your job:
- Portray the WORLD: the environment, the weather, the sounds, the smells, the small details that make it feel lived-in.
- Portray NPCs: their observable behavior, reactions, and (if they speak) their dialogue. When an NPC speaks, use the [CHARACTER_TURN:npc_id] tag at the start of their line, then close with [CHARACTER_TURN:end].
- Drive the scene with tension and momentum, but END your turn the moment the player needs to act.

THE PLAYER:
- The player's name is Alex. Refer to Alex in third person, by name.
- NEVER speak for Alex. NEVER decide what Alex does, says, thinks, or feels.
- NEVER write Alex's dialogue, choices, or reactions. Wait for the player's input.

NARRATIVE DISCIPLINE:
- Keep prose tight: 2-4 sentences per beat unless the scene demands more.
- Lean on sensory detail over spectacle.
- Show, don't summarize. A scene beat should leave the next move to the player.";

/// The bracket-command vocabulary the narrator emits alongside prose.
/// Mirrors UIE's scene-effect names so the eventual UI port is direct.
const BRACKET_PROTOCOL: &str = "\
Emit bracket commands alongside your prose to drive the UI deterministically:

- [CHARACTER_TURN:npc_id] ... [CHARACTER_TURN:end]
    Wrap an NPC's spoken line. Use the npc_id from <scenario>present_npcs.

- [OBJECT id=object_id state=new_state]
    Announce an object's state changed. Use stable snake_case ids.

- [FX effect_name]
    Trigger a scene effect. Valid names: rain, snow, fog, letterbox, flash,
    vignette, shake-light, shake-heavy, spotlight, thunder, glitch, blackout,
    whiteout. Use sparingly: only when the ambiance meaningfully shifts.

Bracket commands are machine-read; keep their syntax exact (square brackets,
colon for character turns, equals sign for object state). Put them on their
own line, separate from prose.";

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn dungeon_card() -> SimCard {
        SimCard {
            id: "dungeon_tavern".into(),
            name: "The Rusty Tankard".into(),
            card_type: "roleplay".into(),
            core_persona: "A dungeon scenario.".into(),
            traits: String::new(),
            appearance: String::new(),
            role_instruction: String::new(),
            responsibilities: String::new(),
            conversational_rules: String::new(),
            technical_rules: String::new(),
            introductions: Vec::new(),
            setting: Some("A frontier tavern at the edge of the Goblinwood.".into()),
            tone: Some("grim, atmospheric".into()),
            opening_scene: Some("Rain lashes the shutters.".into()),
            start_npc_ids: vec!["gorm".into(), "goblin".into()],
            declared_activities: vec!["combat".into()],
            protagonist_name: Some("Alex".into()),
        }
    }

    #[test]
    fn narrator_prompt_contains_core_sections() {
        let card = dungeon_card();
        let prompt = build_narrator_system_prompt(&card, None);
        assert!(prompt.contains("<narrator_role>"));
        assert!(prompt.contains("<scenario>"));
        assert!(prompt.contains("frontier tavern"));
        assert!(prompt.contains("grim, atmospheric"));
        assert!(prompt.contains("gorm, goblin"));
        assert!(prompt.contains("<bracket_commands>"));
    }

    #[test]
    fn narrator_prompt_forbids_speaking_for_player() {
        let card = dungeon_card();
        let prompt = build_narrator_system_prompt(&card, None);
        assert!(prompt.contains("NEVER speak for Alex"));
        assert!(prompt.contains("third person"));
    }

    #[test]
    fn narrator_prompt_includes_world_state_when_provided() {
        let card = dungeon_card();
        let state = "weather: stormy\nchest_state: locked";
        let prompt = build_narrator_system_prompt(&card, Some(state));
        assert!(prompt.contains("<world_state>"));
        assert!(prompt.contains("stormy"));
    }

    #[test]
    fn narrator_prompt_omits_world_state_when_empty() {
        let card = dungeon_card();
        let prompt = build_narrator_system_prompt(&card, Some("   "));
        assert!(!prompt.contains("<world_state>"));
    }

    #[test]
    fn narrator_prompt_handles_minimal_card() {
        // A card missing scenario fields degrades gracefully: empty
        // <scenario> block, no panic.
        let card = SimCard {
            id: "minimal".into(),
            name: "Minimal".into(),
            card_type: "roleplay".into(),
            core_persona: String::new(),
            traits: String::new(),
            appearance: String::new(),
            role_instruction: String::new(),
            responsibilities: String::new(),
            conversational_rules: String::new(),
            technical_rules: String::new(),
            introductions: Vec::new(),
            setting: None,
            tone: None,
            opening_scene: None,
            start_npc_ids: Vec::new(),
            declared_activities: Vec::new(),
            protagonist_name: None,
        };
        let prompt = build_narrator_system_prompt(&card, None);
        assert!(prompt.contains("<narrator_role>"));
        assert!(prompt.contains("<scenario>"));
    }

    /// The `<active_reality>` anchor (Phase E, 2026-07-18): reinforces the
    /// active card's identity at the prompt tail to override cross-card KV
    /// contamination. Must include the card name + protagonist name when
    /// declared.
    #[test]
    fn narrator_prompt_has_active_reality_with_protagonist() {
        let card = dungeon_card();
        let prompt = build_narrator_system_prompt(&card, None);
        assert!(prompt.contains("<active_reality>"));
        assert!(prompt.contains("The Rusty Tankard"));
        assert!(prompt.contains("The protagonist is Alex"));
        assert!(prompt.contains("NOT any other scenario"));
    }

    /// When `protagonist_name` is None (a card that doesn't declare one),
    /// the anchor falls back to generic phrasing and explicitly forbids
    /// inventing a name: which is the actual defense against the Alex
    /// hallucination when no protagonist is named.
    #[test]
    fn narrator_prompt_active_reality_falls_back_when_no_protagonist() {
        let card = SimCard {
            id: "minimal".into(),
            name: "Some Scene".into(),
            card_type: "roleplay".into(),
            core_persona: String::new(),
            traits: String::new(),
            appearance: String::new(),
            role_instruction: String::new(),
            responsibilities: String::new(),
            conversational_rules: String::new(),
            technical_rules: String::new(),
            introductions: Vec::new(),
            setting: Some("A place.".into()),
            tone: None,
            opening_scene: None,
            start_npc_ids: Vec::new(),
            declared_activities: Vec::new(),
            protagonist_name: None,
        };
        let prompt = build_narrator_system_prompt(&card, None);
        assert!(prompt.contains("<active_reality>"));
        // Generic fallback: must NOT contain a hardcoded name.
        assert!(!prompt.contains("The protagonist is Alex"));
        assert!(prompt.contains("never invent or import a protagonist name"));
        assert!(prompt.contains("Some Scene"));
    }

    /// The `<active_reality>` block is the LAST section in the prompt
    /// (closest to the user input). Verify ordering: <narrator_role> first,
    /// <active_reality> last.
    #[test]
    fn active_reality_is_last_section() {
        let card = dungeon_card();
        let prompt = build_narrator_system_prompt(&card, Some("weather: stormy"));
        let narrator_idx = prompt.find("<narrator_role>").expect("narrator_role present");
        let scenario_idx = prompt.find("<scenario>").expect("scenario present");
        let bracket_idx = prompt.find("<bracket_commands>").expect("bracket_commands present");
        let world_idx = prompt.find("<world_state>").expect("world_state present");
        let reality_idx = prompt.find("<active_reality>").expect("active_reality present");
        assert!(narrator_idx < scenario_idx);
        assert!(scenario_idx < bracket_idx);
        assert!(bracket_idx < world_idx);
        assert!(world_idx < reality_idx);
    }
}
