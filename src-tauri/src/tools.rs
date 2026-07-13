use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    #[serde(default)]
    pub input: serde_json::Value,
}

#[derive(Debug, Clone)]
pub enum ToolResult {
    Value(serde_json::Value),
    AskUser { question: String, context: String },
}

pub fn parse_tool_calls_from_text(text: &str) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    let re = tool_call_re();
    for caps in re.captures_iter(text) {
        let body = caps.name("body").map(|m| m.as_str()).unwrap_or("").trim();
        if body.is_empty() {
            continue;
        }
        if let Some(parsed) = parse_tool_call_body(body) {
            match parsed {
                ParsedBody::Single(tc) => calls.push(tc),
                ParsedBody::Many(list) => calls.extend(list.into_iter().filter(|t| !t.name.is_empty())),
            }
        }
    }
    calls
}

enum ParsedBody {
    Single(ToolCall),
    Many(Vec<ToolCall>),
}

fn parse_tool_call_body(body: &str) -> Option<ParsedBody> {
    let cleaned = repair_json(body);
    if let Ok(v) = serde_json::from_str::<ToolCall>(&cleaned) {
        return Some(ParsedBody::Single(v));
    }
    if let Ok(list) = serde_json::from_str::<Vec<ToolCall>>(&cleaned) {
        return Some(ParsedBody::Many(list));
    }
    let joined = cleaned.replace("}{", "},{");
    if let Ok(list) = serde_json::from_str::<Vec<ToolCall>>(&joined) {
        return Some(ParsedBody::Many(list));
    }
    if let Ok(list) = serde_json::from_str::<Vec<ToolCall>>(&format!("[{cleaned}]")) {
        return Some(ParsedBody::Many(list));
    }
    None
}

fn repair_json(raw: &str) -> String {
    let s = raw.trim();
    s.replace(",}", "}").replace(",]", "]")
}

pub fn strip_tool_calls_from_text(text: &str) -> String {
    tool_call_re().replace_all(text, "").trim().to_string()
}

fn tool_call_re() -> &'static regex::Regex {
    use std::sync::OnceLock;
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    RE.get_or_init(|| {
        regex::Regex::new(r"(?is)```tool_call\s*(?P<body>[\s\S]*?)```").unwrap()
    })
}

pub async fn execute_tool(
    name: &str,
    input: &serde_json::Value,
    settings: &crate::prompts::WupiSettings,
    message_count: usize,
) -> ToolResult {
    match name {
        "get_settings" => ToolResult::Value(serde_json::json!({
            "contextSize": settings.context_size,
            "conversationBudget": settings.conversation_budget,
            "messageCount": message_count,
        })),
        "ask_user" => {
            let question = input
                .get("question")
                .and_then(|v| v.as_str())
                .unwrap_or("Please answer:")
                .to_string();
            let context = input
                .get("context")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            ToolResult::AskUser { question, context }
        }
        _ => ToolResult::Value(serde_json::json!({
            "error": format!("Unknown tool: {name}"),
        })),
    }
}
