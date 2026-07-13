use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
    System,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: String,
    pub role: Role,
    pub content: String,
    #[serde(default)]
    pub reasoning: String,
    /// The raw model output (pre-parse) for assistant turns. Used by the
    /// chat formatter to re-render the turn so the rendered token sequence
    /// matches the KV cache exactly — preserving delta-prefill across turns
    /// (Bug #3: Prefix Cache Extinction). Empty for user/system messages and
    /// legacy sessions; `render_prompt` falls back to `strip_thinking` then.
    #[serde(default)]
    pub raw_output: String,
    pub timestamp: i64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Conversation {
    pub messages: Vec<Message>,
}

impl Conversation {
    pub fn new() -> Self {
        Self { messages: Vec::new() }
    }

    pub fn add_message(&mut self, role: Role, content: String) -> &Message {
        self.add_message_with_reasoning(role, content, String::new())
    }

    /// Add a message with an explicit reasoning (thought-channel) payload.
    /// Used for assistant turns where the model emitted a `<|channel>thought`
    /// block that we parsed out of the raw output.
    pub fn add_message_with_reasoning(
        &mut self,
        role: Role,
        content: String,
        reasoning: String,
    ) -> &Message {
        let msg = Message {
            id: gen_id(),
            role,
            content,
            reasoning,
            raw_output: String::new(),
            timestamp: chrono_now_millis(),
        };
        self.messages.push(msg);
        self.messages.last().expect("just pushed")
    }

    /// Add an assistant turn with the raw model output alongside the cleaned
    /// content + reasoning. The raw output is what the chat formatter
    /// re-renders from so the token sequence matches the KV cache (Bug #3).
    pub fn add_assistant_turn(
        &mut self,
        content: String,
        reasoning: String,
        raw_output: String,
    ) -> &Message {
        let msg = Message {
            id: gen_id(),
            role: Role::Assistant,
            content,
            reasoning,
            raw_output,
            timestamp: chrono_now_millis(),
        };
        self.messages.push(msg);
        self.messages.last().expect("just pushed")
    }

    /// True iff the most recent message is a user turn. Used by the error
    /// rollback in `chat_send` (Bug C fix, 2026-07-12) to decide whether the
    /// just-added user message should be popped when the backend errored
    /// before producing any assistant reply.
    pub fn last_message_is_user(&self) -> bool {
        self.messages.last().map(|m| m.role == Role::User).unwrap_or(false)
    }

    /// Remove the most recent message, if any. Used to roll back an orphaned
    /// user message when a generation fails (Bug C fix, 2026-07-12) so that
    /// session.json matches the actual visible conversation and the next
    /// send doesn't render two consecutive user turns.
    pub fn pop_last_message(&mut self) {
        self.messages.pop();
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }

    pub fn load(path: &Path) -> std::io::Result<Self> {
        match std::fs::read_to_string(path) {
            Ok(text) => serde_json::from_str(&text)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::new()),
            Err(e) => Err(e),
        }
    }

    pub fn assemble_api_messages(&self, system_prompt: &str) -> Vec<ApiMessage> {
        let mut out = Vec::with_capacity(self.messages.len() + 1);
        if !system_prompt.is_empty() {
            out.push(ApiMessage {
                role: "system".into(),
                content: system_prompt.into(),
                raw_output: String::new(),
            });
        }
        for m in &self.messages {
            out.push(ApiMessage {
                role: match m.role {
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    Role::System => "system",
                }
                .into(),
                content: m.content.clone(),
                raw_output: m.raw_output.clone(),
            });
        }
        out
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiMessage {
    pub role: String,
    pub content: String,
    /// Raw model output for assistant turns. The formatter renders model
    /// turns from this (when present) so the rendered tokens match the KV
    /// cache exactly. Empty for non-assistant turns. See Bug #3.
    #[serde(default)]
    pub raw_output: String,
}

fn gen_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let mut bytes = [0u8; 6];
    getrandom_fill(&mut bytes);
    let rand: String = bytes.iter().map(|b| format!("{:02x}", b)).collect();
    format!("m_{:x}_{}", ts, &rand[..6])
}

fn getrandom_fill(buf: &mut [u8]) {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let mut x = nanos as u64;
    for b in buf.iter_mut() {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *b = (x & 0xff) as u8;
    }
}

fn chrono_now_millis() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}
