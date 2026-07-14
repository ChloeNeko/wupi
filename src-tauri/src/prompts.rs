//! System-prompt construction.
//!
//! The system prompt is now TWO cleanly separated layers (2026-07-14, SIM card
//! system):
//!
//! 1. **`OS_DIRECTIVES`** — the universal OS scaffold. Engine-level rules true
//!    for EVERY card (the simulation framing + the semantics of the
//!    `<retrieved_memory>` / `<world_state>` tags). Lives here in Rust because
//!    it's an engineering concern tied to the architecture, NOT a content
//!    artifact. Every future card shares it; write it once.
//! 2. **The persona** — rendered from the active Simulation Card
//!    (`sim_card.rs`) and passed in as `Option<&str>`. Wupi gets NOTHING from
//!    this file — her entire identity comes from `cards/Wupi.sim`. A dungeon
//!    card would supply its own persona; the directives above are unchanged.
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
<world_state> is persistent ground truth about the simulated world. When \
memory and the live conversation disagree, the live conversation always wins.";

/// Assemble the system-prompt content from its three layers.
///
/// - `<os_directives>` — always present (universal scaffolding).
/// - `<persona>` — present only when a real card loaded; `None` or empty
///   suppresses the section (e.g. the fallback stub persona).
/// - `<current_context>` — the live `WupiSettings` readout.
pub fn build_system_content(settings: &WupiSettings, persona: Option<&str>) -> String {
    let mut sections = Vec::new();

    sections.push(format!(
        "<os_directives>\n{}\n</os_directives>",
        OS_DIRECTIVES
    ));

    if let Some(p) = persona.filter(|s| !s.trim().is_empty()) {
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

        let content = build_system_content(&settings, None);
        assert!(content.contains("<os_directives>"));
        assert!(content.contains("context_size: 2048"));
        assert!(content.contains("conversation_budget: 8192"));
    }

    #[test]
    fn persona_section_is_optional() {
        let settings = WupiSettings::default();

        // No persona → no <persona> section.
        let without = build_system_content(&settings, None);
        assert!(!without.contains("<persona>"));

        // With persona → section present.
        let with = build_system_content(&settings, Some("<persona>\nWupi\n</persona>"));
        assert!(with.contains("<persona>"));
    }

    #[test]
    fn empty_persona_is_suppressed() {
        let settings = WupiSettings::default();
        let content = build_system_content(&settings, Some("   "));
        assert!(!content.contains("<persona>"));
    }
}
