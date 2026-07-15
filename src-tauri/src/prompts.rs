//! System-prompt construction.
//!
//! The system prompt is now THREE cleanly separated layers (SIM card system
//! 2026-07-14, User Profile system 2026-07-14):
//!
//! 1. **`OS_DIRECTIVES`** — the universal OS scaffold. Engine-level rules true
//!    for EVERY card (the simulation framing + the semantics of the
//!    `<retrieved_memory>` / `<world_state>` / `<user_profile>` tags). Lives
//!    here in Rust because it's an engineering concern tied to the
//!    architecture, NOT a content artifact. Every future card shares it; write
//!    it once.
//! 2. **The persona** — rendered from the active Simulation Card
//!    (`sim_card.rs`) and passed in as `Option<&str>`. Wupi gets NOTHING from
//!    this file — her entire identity comes from `cards/Wupi.sim`. A dungeon
//!    card would supply its own persona; the directives above are unchanged.
//! 3. **The user profile** — rendered from the operator's profile
//!    (`user_profile.rs`, `cards/Operator.xml`) and passed in as `Option<&str>`.
//!    The static "who am I talking to" counterpart to the persona. Re-read
//!    fresh each turn (hot-reload) so live edits take effect immediately.
//!
//! Ordering in the assembled prompt: `<os_directives>` → `<persona>` →
//! `<user_profile>` → `<current_context>`. The operator's identity comes AFTER
//! Wupi's so the model grounds itself in its own personality first, then learns
//! who it's addressing — the same ordering the context stack uses.
//!
//! The old `DEFAULT_SYSTEM_PROMPT` (the "out-of-character assistant, not part
//! of any roleplay" text) was placeholder scaffolding from the barebones-P
//! phase and is DELETED. The catgirl card was always the intended destination.

/// The universal OS-level directives — card-agnostic scaffolding shared by
/// every Simulation Card. Persona content comes from the card, NOT from here.
pub const OS_DIRECTIVES: &str = "\
You are operating as a process within WUPI OS — a local, AI-native simulation \
runtime. You are the active Simulation Card: a simulation interface reasoning \
through a structured environment, not a generic chatbot.

Structural discipline: respect the tags and channels provided. Context marked \
<retrieved_memory> holds PAST records, not the current scene — they are \
reference material only, never continuity to adopt. Context marked \
<world_state> is persistent ground truth about the simulated world. Context \
marked <user_profile> describes the operator you are speaking with — treat it \
as authoritative identity, not a suggestion. When memory and the live \
conversation disagree, the live conversation always wins.";

/// Assemble the system-prompt content from its layers.
///
/// - `<os_directives>` — always present (universal scaffolding).
/// - `<persona>` — present only when a real card loaded; `None` or empty
///   suppresses the section (e.g. the fallback stub persona).
/// - `<user_profile>` — present only when an operator profile loaded; `None`
///   or empty suppresses the section (e.g. no `Operator.xml` resolved). Re-read
///   fresh each turn by the caller so live edits take effect immediately
///   (hot-reload). Byte-identical across turns until edited → does NOT trigger
///   the §2F cold-reset guard (same cache-friendliness as the persona).
/// - `<current_context>` — the live `WupiSettings` readout.
pub fn build_system_content(
    settings: &WupiSettings,
    persona: Option<&str>,
    user_profile: Option<&str>,
) -> String {
    let mut sections = Vec::new();

    sections.push(format!(
        "<os_directives>\n{}\n</os_directives>",
        OS_DIRECTIVES
    ));

    if let Some(p) = persona.filter(|s| !s.trim().is_empty()) {
        sections.push(p.to_owned());
    }

    if let Some(p) = user_profile.filter(|s| !s.trim().is_empty()) {
        sections.push(p.to_owned());
    }

    sections.push(format!(
        "<current_context>\ncontext_size: {}\nconversation_budget: {}\n</current_context>",
        settings.context_size, settings.conversation_budget
    ));

    // Note (2026-07-13, §2F eager-prefill design): the retrieved-memory block
    // NO LONGER lives in the system prompt. It moved to the inter-turn region
    // (chat_format.rs::render_prompt injects it after all turns, before the
    // generation prompt). Keeping it out of the system prompt is what makes
    // the system+turns prefix stable across turns, which lets the eager
    // prefill establish a cache the next turn can delta-prefill against.

    sections.join("\n\n")
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WupiSettings {
    pub context_size: u32,
    pub conversation_budget: u32,
}

impl Default for WupiSettings {
    fn default() -> Self {
        Self {
            context_size: 4000,
            conversation_budget: 16000,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_system_content_includes_live_settings() {
        let settings = WupiSettings {
            context_size: 2048,
            conversation_budget: 8192,
        };

        let content = build_system_content(&settings, None, None);
        assert!(content.contains("<os_directives>"));
        assert!(content.contains("context_size: 2048"));
        assert!(content.contains("conversation_budget: 8192"));
    }

    #[test]
    fn persona_section_is_optional() {
        let settings = WupiSettings::default();

        // No persona → no <persona> section.
        let without = build_system_content(&settings, None, None);
        assert!(!without.contains("<persona>"));

        // With persona → section present.
        let with = build_system_content(&settings, Some("<persona>\nWupi\n</persona>"), None);
        assert!(with.contains("<persona>"));
    }

    #[test]
    fn empty_persona_is_suppressed() {
        let settings = WupiSettings::default();
        let content = build_system_content(&settings, Some("   "), None);
        assert!(!content.contains("<persona>"));
    }

    #[test]
    fn user_profile_section_is_optional_and_ordered_after_persona() {
        // The profile is a sibling to the persona. None → suppressed. When
        // present it lands AFTER <persona> and BEFORE <current_context>.
        // Discriminate on the CLOSING tag `</user_profile>`: the opening tag
        // name also appears in OS_DIRECTIVES (the tag-semantics sentence), but
        // the closing tag only ever exists in a rendered section.
        let settings = WupiSettings::default();

        let without = build_system_content(&settings, None, None);
        assert!(!without.contains("</user_profile>"));

        let profile = "<user_profile>\nname: Chloe\n</user_profile>";
        let with = build_system_content(&settings, None, Some(profile));
        assert!(with.contains("</user_profile>"));
        assert!(with.contains("name: Chloe"));

        // Ordering: when both persona + profile are present, persona comes first.
        let persona = "<persona>\nname: Wupi\n</persona>";
        let both = build_system_content(&settings, Some(persona), Some(profile));
        let persona_pos = both.find("</persona>").unwrap();
        let profile_pos = both.find("</user_profile>").unwrap();
        let ctx_pos = both.find("<current_context>").unwrap();
        assert!(persona_pos < profile_pos, "persona before user_profile");
        assert!(profile_pos < ctx_pos, "user_profile before current_context");
    }

    #[test]
    fn empty_user_profile_is_suppressed() {
        let settings = WupiSettings::default();
        let content = build_system_content(&settings, None, Some("   "));
        // Closing tag only appears in a rendered section (see note above).
        assert!(!content.contains("</user_profile>"));
    }
}
