use serde::{Deserialize, Serialize};

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

#[allow(dead_code)]
pub const TOOL_CALL_FORMAT: &str = "\
```tool_call
{\"name\": \"tool_name\", \"input\": { ...parameters... }}
```";

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDef {
    pub name: &'static str,
    pub description: &'static str,
    pub schema: serde_json::Value,
}

#[allow(dead_code)]
pub fn tool_definitions() -> Vec<ToolDef> {
    vec![
        ToolDef {
            name: "get_settings",
            description: "Read WUPI OS's current settings (context size, conversation budget, \
                          message count, etc.). Call this before changing anything so you work \
                          from the real current values.",
            schema: serde_json::json!({ "type": "object", "properties": {}, "required": [] }),
        },
        ToolDef {
            name: "ask_user",
            description: "Ask the user a clarifying question and wait for their typed answer. \
                          Use this when you lack information needed to proceed.",
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "question": { "type": "string", "description": "The question to show the user" },
                    "context": { "type": "string", "description": "Optional extra context shown under the question" }
                },
                "required": ["question"],
            }),
        },
    ]
}

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
