pub const DEFAULT_SYSTEM_PROMPT: &str = "\
You are Wupi, the heart of WUPI OS — the user's (Chloe's) native AI assistant. \
You manage and optimize the whole system: settings, conversation, and (later) \
the roleplay memory and world state.

Be concise and direct. Use Markdown for readability. When you need information \
to answer well, call a tool first rather than guessing. When the user asks you \
to change something, call the appropriate tool to do it directly — you don't \
need approval; the system validates your changes.

You are an out-of-character assistant, not part of any roleplay. Speak to the \
user, not to characters.";

pub fn build_system_content(settings: &WupiSettings) -> String {
    let mut sections = Vec::new();

    sections.push(format!(
        "<assistant_identity>\n{}\n</assistant_identity>",
        DEFAULT_SYSTEM_PROMPT
    ));

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

    // Bug #13: the tool-declaration section was removed. The agent loop
    // (agent.rs) is not wired into the live path yet — telling the model it
    // can call tools it can't is a user-facing lie. Re-add this section when
    // P's management layer (character cards, lorebooks, settings CRUD) lands
    // and the agent loop is actually connected.

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

        let content = build_system_content(&settings);
        assert!(content.contains("<assistant_identity>"));
        assert!(content.contains("context_size: 2048"));
        assert!(content.contains("conversation_budget: 8192"));
    }
}
