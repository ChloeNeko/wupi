use crate::llm::GenerationClient;
use crate::prompts;
use crate::session::ApiMessage;
use crate::tools;

pub const MAX_TOOL_ROUNDS: usize = 5;

#[allow(dead_code)]
pub async fn run<C: GenerationClient>(
    _client: &C,
    _settings: &prompts::WupiSettings,
    _persisted: &[crate::session::Message],
) -> anyhow::Result<String> {
    anyhow::bail!("agent loop not wired in Layer 1")
}

#[allow(dead_code)]
fn assemble_round_messages(
    system_prompt: &str,
    persisted: &[crate::session::Message],
    transient: &[ApiMessage],
) -> Vec<ApiMessage> {
    let mut out = crate::session::Conversation {
        messages: persisted.to_vec(),
    }
    .assemble_api_messages(system_prompt);
    out.extend_from_slice(transient);
    out
}

#[allow(dead_code)]
fn format_tool_result(name: &str, result: &tools::ToolResult) -> ApiMessage {
    let content = match result {
        tools::ToolResult::Value(v) => {
            format!("[Tool \"{name}\" result]\n{}", serde_json::to_string_pretty(v).unwrap_or_default())
        }
        tools::ToolResult::AskUser { question, context } => {
            format!("[Tool \"{name}\" → ask_user]\n{question}\n{context}")
        }
    };
    ApiMessage {
        role: "user".into(),
        content,
        raw_output: String::new(),
    }
}
