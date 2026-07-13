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

pub fn build_system_content(settings: &WupiSettings, memory_block: Option<&str>) -> String {
    let mut sections = Vec::new();

    sections.push(format!(
        "<assistant_identity>\n{}\n</assistant_identity>",
        DEFAULT_SYSTEM_PROMPT
    ));

    // Retrieved memory block (pillar 3, §2F Option 3). Sits between identity
    // and current_context so it's close to the user's attention without
    // displacing Wupi's core identity. Empty/whitespace blocks are skipped
    // entirely — no empty tag pollution. When this block's content changes
    // turn-to-turn (which it will, since each query retrieves a different
    // set), the §2F structural-divergence guard cold-resets the KV cache.
    // That is the accepted v1 cost; the cache-layout optimization is a
    // dedicated later pass.
    if let Some(block) = memory_block {
        let trimmed = block.trim();
        if !trimmed.is_empty() {
            sections.push(format!("<retrieved_memory>\n{trimmed}\n</retrieved_memory>"));
        }
    }

    sections.push(format!(
        "<current_context>\ncontext_size: {}\nconversation_budget: {}\n</current_context>",
        settings.context_size, settings.conversation_budget
    ));

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

        let content = build_system_content(&settings, None);
        assert!(content.contains("<assistant_identity>"));
        assert!(content.contains("context_size: 2048"));
        assert!(content.contains("conversation_budget: 8192"));
        // No memory block supplied → no tag leaks.
        assert!(!content.contains("<retrieved_memory>"));
    }

    #[test]
    fn build_system_content_includes_memory_block_when_supplied() {
        let settings = WupiSettings::default();
        let block = "[user] earlier I mentioned the project plan";

        let content = build_system_content(&settings, Some(block));
        assert!(content.contains("<retrieved_memory>"));
        assert!(content.contains(block));
        // Block sits AFTER identity, BEFORE current_context.
        let id_pos = content.find("<assistant_identity>").unwrap();
        let mem_pos = content.find("<retrieved_memory>").unwrap();
        let ctx_pos = content.find("<current_context>").unwrap();
        assert!(id_pos < mem_pos && mem_pos < ctx_pos);
    }

    #[test]
    fn build_system_content_skips_empty_memory_block() {
        let settings = WupiSettings::default();

        let content = build_system_content(&settings, Some("   \n  "));
        assert!(!content.contains("<retrieved_memory>"));
    }
}
