//! Chat-completion format presets.
//!
//! Each model family has its own turn protocol — the special tokens that
//! delimit system/user/model turns, thinking channels, and tool calls.
//! Rather than depend on llama.cpp's heuristic template matcher (which only
//! recognizes ~50 hardcoded formats and returns `-1` / FfiError on anything
//! modern like Gemma 4's `<|turn>` protocol), we hand-write the formatter
//! against each family's *documented* protocol.
//!
//! This is deterministic, dependency-free, and avoids shipping a Jinja engine.
//! Adding a new model family means writing one more `ChatFormat` impl — no
//! per-model completion logic.

use crate::session::ApiMessage;

/// What a turn looks like for a given model family.
pub trait ChatFormat: Send + Sync {
    /// Render a full conversation + system prompt into the model's native
    /// token protocol. The returned string is passed to `str_to_token`.
    ///
    /// `add_generation_prompt = true` should append the opening of a model
    /// turn (no closing) so the model continues from there.
    ///
    /// `memory_block` — an optional retrieved-memory annotation injected into
    /// the inter-turn region (between the last conversation turn and the
    /// generation prompt). `None`/empty renders nothing. This position is
    /// deliberate (2026-07-13, §2F eager-prefill design): keeping the memory
    /// block OUT of the system prompt means the stable prefix (system +
    /// turns, rendered with `memory_block=None`) is a true byte-prefix of the
    /// full prompt, which is what lets the eager prefill establish a cache
    /// the next turn can delta-prefill against. The block is a non-turn
    /// annotation (no turn markers around it) so it reads as context, not a
    /// conversational turn.
    fn render_prompt(
        &self,
        system: &str,
        messages: &[ApiMessage],
        tools: &[ToolSpec],
        memory_block: Option<&str>,
        add_generation_prompt: bool,
    ) -> String;

    /// Parse raw model output into (reply, thought) channels.
    /// `reply` is the user-visible text; `thought` is the model's internal
    /// reasoning (may be empty if the model didn't think).
    fn parse_output(&self, raw: &str) -> ParsedOutput;

    /// Human-readable name for logging.
    fn name(&self) -> &'static str;
}

/// A tool declaration rendered into the prompt's system turn.
#[derive(Debug, Clone)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
}

/// Result of splitting model output into its channels.
#[derive(Debug, Clone, Default)]
pub struct ParsedOutput {
    /// The reply channel — what the user sees.
    pub content: String,
    /// The thought channel — the model's reasoning, if any.
    pub reasoning: String,
    /// The complete raw model output (pre-parse). Set by the engine's decode
    /// loop, NOT by `parse_output` itself. Persisted onto assistant `Message`s
    /// so `render_prompt` can re-render the turn cache-coherently (Bug #3).
    pub raw: String,
}

/// The set of model families we know how to format for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    /// Gemma 4 E4B / Gemma 4 variants. Uses `<|turn>` / `<turn|>` turn
    /// delimiters, `<|channel>thought` / `<channel|>` thinking channels,
    /// `<|tool_call>` / `<tool_call|>` for tool invocation.
    Gemma4,
    /// Fallback: plain text turns, no special protocol. Used when the loaded
    /// model isn't recognized — generation will work but without thinking or
    /// tool channels.
    Plain,
}

impl ModelFamily {
    /// Pick a family from the model's filename. Case-insensitive substring
    /// match. Extend as new models are added.
    pub fn from_model_name(filename: &str) -> Self {
        let lower = filename.to_lowercase();
        // The chat model is always shipped as `WUPI.gguf` (locked naming
        // convention 2026-07-12: any future chat model reuses this name).
        // Today's WUPI.gguf is a Gemma 4 12B quant, so it resolves to Gemma4.
        // If you ever ship a NON-Gemma chat model under `WUPI.gguf`, add a new
        // variant + ChatFormat impl and route on the model's GGUF metadata
        // (`general.architecture`) instead of the filename.
        if lower.contains("gemma") || lower.contains("wupi") {
            // Gemma 2/3 use <start_of_turn>; Gemma 4 uses <|turn>. The 4B/E4B
            // quants in this project are Gemma 4. If you load a Gemma 2/3 model,
            // add a separate variant and matcher.
            ModelFamily::Gemma4
        } else {
            ModelFamily::Plain
        }
    }

    /// Return the formatter for this family.
    pub fn formatter(&self) -> Box<dyn ChatFormat> {
        match self {
            ModelFamily::Gemma4 => Box::new(Gemma4Format),
            ModelFamily::Plain => Box::new(PlainFormat),
        }
    }

    /// The literal string that opens a new turn in this family's protocol, if
    /// it has one. Used by the engine to find turn boundaries for safe cache
    /// eviction. Returns `None` for families with no turn delimiter (Plain).
    pub fn turn_marker_literal(&self) -> Option<&'static str> {
        match self {
            ModelFamily::Gemma4 => Some("<|turn>"),
            ModelFamily::Plain => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Gemma 4 protocol
// ---------------------------------------------------------------------------

/// Renders the Gemma 4 E4B dialogue protocol.
///
/// Reference: https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4
///
/// Token summary:
///   `<|turn>{role}\n` ... `<turn|>\n`   — dialogue turn delimiters
///   `<|think|>`                         — activates thinking (system turn)
///   `<|channel>thought\n` ... `<channel|>` — internal reasoning channel
///   `<|tool>declaration:...{...}<tool|>` — tool definition
///   `<|tool_call>call:name{args}<tool_call|>` — model requests a tool
///   `<|tool_response>response:name{val}<tool_response|>` — tool result back
///
/// Roles: `system`, `user`, `model` (note: assistant → model).
pub struct Gemma4Format;

impl ChatFormat for Gemma4Format {
    fn name(&self) -> &'static str {
        "gemma4"
    }

    fn render_prompt(
        &self,
        system: &str,
        messages: &[ApiMessage],
        tools: &[ToolSpec],
        memory_block: Option<&str>,
        add_generation_prompt: bool,
    ) -> String {
        let mut out = String::with_capacity(2048);

        // --- System turn (with optional tools + thinking activation) ---
        let has_system = !system.trim().is_empty();
        let has_tools = !tools.is_empty();
        if has_system || has_tools {
            out.push_str("<|turn>system\n");
            if has_system {
                out.push_str(system.trim());
            }
            // Tool declarations live inside the system turn, each wrapped in
            // <|tool> ... <tool|>.
            for t in tools {
                out.push_str("<|tool>declaration:");
                out.push_str(&t.name);
                out.push_str("{description:\"");
                push_escaped(&mut out, &t.description);
                out.push_str("\"}<tool|>");
            }
            out.push_str("<turn|>\n");
        }

        // --- Conversation turns ---
        for m in messages {
            // Gemma calls the assistant "model".
            let role = match m.role.as_str() {
                "assistant" => "model",
                other => other,
            };
            out.push_str("<|turn>");
            out.push_str(role);
            out.push('\n');

            if role == "model" {
                // Cache-coherent re-render (Bug #3): when raw_output is present,
                // render it verbatim so the token sequence matches what's
                // resident in the KV cache. Without this, the cleaned content
                // diverges from the cache and forces a full re-prefill each turn.
                // Legacy turns (no raw_output) fall back to strip_thinking.
                if !m.raw_output.is_empty() {
                    out.push_str(&m.raw_output);
                } else {
                    out.push_str(&strip_thinking(&m.content));
                }
            } else {
                out.push_str(m.content.trim());
            }
            out.push_str("<turn|>\n");
        }

        // --- Retrieved-memory block (inter-turn injection, §2F eager-prefill) ---
        // Sits AFTER all conversation turns, BEFORE the generation prompt. This
        // is the load-bearing position: it's a non-turn annotation (no `<|turn>`
        // markers around it) injected into the inter-turn region, so it reads as
        // context for the upcoming model turn rather than a turn itself. The
        // model hasn't seen this exact structure in training — flagged as a
        // runtime-tested risk in the plan. Only emitted when there's content
        // AND a generation prompt follows (no point annotating a render that
        // isn't going to generate).
        if add_generation_prompt {
            if let Some(block) = memory_block {
                let trimmed = block.trim();
                if !trimmed.is_empty() {
                    out.push_str("<retrieved_memory>\n");
                    out.push_str(trimmed);
                    out.push_str("\n</retrieved_memory>\n");
                }
            }
        }

        // --- Generation prompt ---
        if add_generation_prompt {
            out.push_str("<|turn>model\n");
        }

        out
    }

    fn parse_output(&self, raw: &str) -> ParsedOutput {
        // The model emits zero or more `<|channel>thought\n ... <channel|>`
        // blocks, optionally followed by a reply channel. We split on the
        // closing marker `<channel|>`: anything before it that contains the
        // opening `<|channel>` (possibly with `thought`) is reasoning; the
        // remainder is the reply.
        //
        // The template's own strip_thinking macro uses this same split logic.
        let mut content = String::new();
        let mut reasoning = String::new();

        for part in raw.split("<channel|>") {
            if let Some(before) = part.split("<|channel>").next() {
                // `before` is everything prior to the opening `<|channel>` on
                // this segment. If the segment started with `<|channel>thought`,
                // `before` is "" and what follows (in `part` after the marker)
                // is the thinking text — but we already consumed it via split.
                // The text after `<channel|>` (next iteration) is the reply.
                if part.contains("<|channel>") {
                    // This was a thought block — capture as reasoning.
                    let thought = part
                        .split("<|channel>")
                        .last()
                        .unwrap_or("")
                        .trim_start_matches("thought")
                        .trim();
                    if !thought.is_empty() {
                        if !reasoning.is_empty() {
                            reasoning.push('\n');
                        }
                        reasoning.push_str(thought);
                    }
                    // Also preserve any text that came before the `<|channel>`
                    // in this segment (rare; usually empty).
                    if !before.trim().is_empty() {
                        content.push_str(before.trim());
                        content.push('\n');
                    }
                } else {
                    // No opening marker — this is reply text (or trailing junk).
                    content.push_str(part);
                }
            }
        }

        ParsedOutput {
            content: content.trim().to_string(),
            reasoning: reasoning.trim().to_string(),
            raw: String::new(),
        }
    }
}

/// The Gemma 4 template's strip_thinking logic, in Rust. Removes
/// `<|channel>thought\n...<channel|>` blocks entirely and keeps the rest.
/// Used when re-rendering prior assistant turns so we don't re-feed the
/// raw thinking markers back to the model as literal text.
fn strip_thinking(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for part in text.split("<channel|>") {
        if part.contains("<|channel>") {
            // Keep only what came before the opening marker (usually nothing).
            out.push_str(part.split("<|channel>").next().unwrap_or(""));
        } else {
            out.push_str(part);
        }
    }
    out.trim().to_string()
}

/// Minimal JSON-string escaping for embedding values into the Gemma 4 tool
/// declaration / argument syntax. Escapes the characters that would break
/// the `{key:"value"}` rendering.
fn push_escaped(out: &mut String, s: &str) {
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(ch),
        }
    }
}

// ---------------------------------------------------------------------------
// ThoughtGate — stateful streaming filter for the variable-length thought block
// ---------------------------------------------------------------------------

/// The opening marker for Gemma 4's thinking channel.
const THOUGHT_OPEN: &str = "<|channel>thought";
/// The closing marker for any Gemma 4 channel (thought or reply).
const CHANNEL_CLOSE: &str = "<channel|>";

/// A stateful streaming filter that handles Gemma 4's variable-length thought
/// block (`<|channel>thought\n...<channel|>`).
///
/// Unlike `StreamFilter` (which handles bounded markers via regex), the thought
/// block has no known length — we can't predict when `<channel|>` will arrive.
/// The gate tracks three states:
///
/// - `Detecting`: we haven't seen enough to know if this is a thought turn or
///   a direct reply. A tiny buffer (len of the opening marker) is held until
///   we can tell. This is the first-token mode detection.
/// - `InThought`: we're inside the thought block. Everything is held back;
///   the UI should show a "thinking" indicator instead.
/// - `Reply`: the thought block closed (or there never was one). All text
///   passes through immediately with zero buffering.
///
/// The gate outputs clean reply text. The thought *content* is not emitted —
/// it's captured separately by `parse_output` at end of generation.
pub struct ThoughtGate {
    state: GateState,
    /// Small buffer used only in Detecting mode to peek at the first tokens.
    detect_buf: String,
    /// Accumulated text in InThought mode, scanned for the closer.
    thought_buf: String,
}

#[derive(Debug, PartialEq, Eq)]
enum GateState {
    /// First tokens — deciding if this turn uses the thought channel.
    Detecting,
    /// Inside the thought block, holding everything until `<channel|>`.
    InThought,
    /// Past the thought block (or never had one). Stream freely.
    Reply,
}

impl ThoughtGate {
    pub fn new() -> Self {
        ThoughtGate {
            state: GateState::Detecting,
            detect_buf: String::with_capacity(THOUGHT_OPEN.len()),
            thought_buf: String::new(),
        }
    }

    /// Feed a token piece. Returns `(reply_text, is_thinking)`.
    /// `reply_text` is text safe to show the user. `is_thinking` tells the
    /// UI to display a thinking indicator when true.
    pub fn feed(&mut self, piece: &str) -> (String, bool) {
        match self.state {
            GateState::Detecting => self.feed_detecting(piece),
            GateState::InThought => self.feed_in_thought(piece),
            GateState::Reply => (piece.to_string(), false),
        }
    }

    /// Emit any remaining held text at end of generation.
    pub fn flush(&mut self) -> String {
        match self.state {
            GateState::Detecting => {
                // Never saw enough to enter thought mode — emit the buffer.
                let out = std::mem::take(&mut self.detect_buf);
                out
            }
            GateState::InThought => {
                // Generation ended mid-thought (truncated). Discard the
                // incomplete thought — it's not useful reply text.
                self.thought_buf.clear();
                String::new()
            }
            GateState::Reply => String::new(),
        }
    }

    fn feed_detecting(&mut self, piece: &str) -> (String, bool) {
        self.detect_buf.push_str(piece);

        // Have we seen enough to decide?
        if self.detect_buf.len() >= THOUGHT_OPEN.len() {
            if self.detect_buf.starts_with(THOUGHT_OPEN) {
                // Thought turn. Strip the opening marker, capture the rest.
                let after_marker = &self.detect_buf[THOUGHT_OPEN.len()..];
                self.thought_buf.push_str(after_marker);
                self.detect_buf.clear();
                self.state = GateState::InThought;
                return (String::new(), true);
            } else {
                // Not a thought turn — emit the whole buffer as reply.
                let out = std::mem::take(&mut self.detect_buf);
                self.state = GateState::Reply;
                return (out, false);
            }
        }

        // Not enough yet to tell. Check if it COULD still become the marker.
        if THOUGHT_OPEN.starts_with(self.detect_buf.as_str()) {
            // Still a valid prefix — hold it.
            (String::new(), false)
        } else {
            // Can't possibly become the marker — emit and switch to Reply.
            let out = std::mem::take(&mut self.detect_buf);
            self.state = GateState::Reply;
            (out, false)
        }
    }

    fn feed_in_thought(&mut self, piece: &str) -> (String, bool) {
        self.thought_buf.push_str(piece);

        // Look for the channel closer. Everything after it is reply text.
        if let Some(idx) = self.thought_buf.find(CHANNEL_CLOSE) {
            let reply_start = idx + CHANNEL_CLOSE.len();
            let reply = self.thought_buf[reply_start..].to_string();
            self.thought_buf.clear();
            self.state = GateState::Reply;
            // Strip any leading whitespace/newline the model puts after the closer.
            (reply.trim_start().to_string(), false)
        } else {
            // Still thinking — hold everything.
            (String::new(), true)
        }
    }
}

impl Default for ThoughtGate {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod thought_gate_tests {
    use super::*;

    #[test]
    fn direct_reply_streams_immediately() {
        let mut g = ThoughtGate::new();
        let (out, thinking) = g.feed("Hello!");
        assert!(!thinking);
        assert!(out.contains("Hello!"));
    }

    #[test]
    fn thought_then_reply() {
        let mut g = ThoughtGate::new();
        // Feed the opening marker across chunks.
        let (out1, thinking1) = g.feed("<|channel>thought\n");
        assert_eq!(out1, "");
        assert!(thinking1, "should be thinking after open marker");

        let (out2, thinking2) = g.feed("reasoning here");
        assert_eq!(out2, "");
        assert!(thinking2);

        let (out3, thinking3) = g.feed("<channel|>visible reply");
        assert!(!thinking3, "should exit thinking after closer");
        assert!(out3.contains("visible reply"), "got: {:?}", out3);
    }

    #[test]
    fn detecting_prefix_not_marker_emits_immediately() {
        // Text that starts with < but isn't the thought marker.
        let mut g = ThoughtGate::new();
        let (out, thinking) = g.feed("Just a reply");
        assert!(!thinking);
        assert_eq!(out, "Just a reply");
    }

    #[test]
    fn partial_marker_prefix_held_then_released() {
        let mut g = ThoughtGate::new();
        // Feed a prefix of the marker that's ambiguous.
        let (out1, _) = g.feed("<|chan");
        assert_eq!(out1, "", "ambiguous prefix should be held");
        // Next piece makes it clearly not the marker.
        let (out2, thinking) = g.feed("not a marker");
        assert!(!thinking);
        assert!(out2.contains("not a marker"));
    }

    #[test]
    fn flush_in_detecting_emits_buffer() {
        // "partial" starts with 'p', not '<', so it can't be a prefix of the
        // thought marker. feed() emits it immediately and switches to Reply.
        // flush() then returns nothing (the buffer was already drained).
        let mut g = ThoughtGate::new();
        let (out, thinking) = g.feed("partial");
        assert!(!thinking);
        assert_eq!(out, "partial");
        let flushed = g.flush();
        assert_eq!(flushed, "");
    }

    #[test]
    fn flush_in_thought_discards() {
        let mut g = ThoughtGate::new();
        g.feed("<|channel>thought\nincomplete");
        let flushed = g.flush();
        assert_eq!(flushed, "", "incomplete thought should be discarded");
    }
}

// ---------------------------------------------------------------------------
// Plain fallback
// ---------------------------------------------------------------------------

/// No special protocol. Renders turns as `Role: content\n` so at least
/// generation works on an unrecognized model. No thinking/tools support.
pub struct PlainFormat;

impl ChatFormat for PlainFormat {
    fn name(&self) -> &'static str {
        "plain"
    }

    fn render_prompt(
        &self,
        system: &str,
        messages: &[ApiMessage],
        _tools: &[ToolSpec],
        memory_block: Option<&str>,
        add_generation_prompt: bool,
    ) -> String {
        let mut out = String::new();
        if !system.trim().is_empty() {
            out.push_str("System: ");
            out.push_str(system.trim());
            out.push_str("\n\n");
        }
        for m in messages {
            let role = match m.role.as_str() {
                "assistant" => "Assistant".to_string(),
                other => {
                    let mut c = other.chars();
                    match c.next() {
                        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                        None => String::new(),
                    }
                }
            };
            out.push_str(&role);
            out.push_str(": ");
            out.push_str(&m.content);
            out.push('\n');
        }
        // Best-effort memory annotation for the fallback family. Plain has no
        // turn protocol to respect, so a plain [memory] line before the
        // generation prompt is the natural shape.
        if add_generation_prompt {
            if let Some(block) = memory_block {
                let trimmed = block.trim();
                if !trimmed.is_empty() {
                    out.push_str("[memory] ");
                    out.push_str(trimmed);
                    out.push('\n');
                }
            }
        }
        if add_generation_prompt {
            out.push_str("Assistant: ");
        }
        out
    }

    fn parse_output(&self, raw: &str) -> ParsedOutput {
        ParsedOutput {
            content: raw.trim().to_string(),
            reasoning: String::new(),
            raw: String::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str) -> ApiMessage {
        ApiMessage {
            role: role.into(),
            content: content.into(),
            raw_output: String::new(),
        }
    }

    #[test]
    fn gemma4_renders_basic_chat() {
        let f = Gemma4Format;
        let out = f.render_prompt(
            "You are Wupi.",
            &[msg("user", "Hello"), msg("model", "Hi there")],
            &[],
            None,
            true,
        );
        assert!(out.contains("<|turn>system\nYou are Wupi.<turn|>"));
        assert!(out.contains("<|turn>user\nHello<turn|>"));
        assert!(out.contains("<|turn>model\nHi there<turn|>"));
        assert!(out.ends_with("<|turn>model\n"));
    }

    #[test]
    fn gemma4_injects_memory_block_in_inter_turn_region() {
        // §2F eager-prefill design (2026-07-13): the retrieved-memory block sits
        // AFTER all conversation turns, BEFORE the generation prompt — and
        // crucially NOT inside the system prompt. This is what makes the stable
        // prefix (rendered with memory_block=None) a true byte-prefix of the
        // full prompt, enabling eager prefill. Verifies both position AND the
        // no-turn-marker annotation shape.
        let f = Gemma4Format;
        let block = "[user] earlier I mentioned the project plan";
        let out = f.render_prompt(
            "You are Wupi.",
            &[msg("user", "Hello"), msg("model", "Hi there")],
            &[],
            Some(block),
            true,
        );
        // Block appears AFTER the last turn close, BEFORE the generation prompt.
        let last_turn_end = out.rfind("<turn|>\n").unwrap();
        let mem_pos = out.find("<retrieved_memory>").unwrap();
        let gen_pos = out.rfind("<|turn>model\n").unwrap();
        assert!(last_turn_end < mem_pos, "memory block must come after all turns");
        assert!(mem_pos < gen_pos, "memory block must come before generation prompt");
        assert!(out.contains(block));
        assert!(out.ends_with("<|turn>model\n"), "still ends with the generation prompt");
        // No turn markers wrap the memory block — it's an annotation.
        assert!(!out.contains("<|turn>retrieved_memory"));
    }

    #[test]
    fn gemma4_memory_block_omitted_when_none() {
        // The stable-prefix render path passes None — no annotation leaks.
        let f = Gemma4Format;
        let out = f.render_prompt(
            "You are Wupi.",
            &[msg("user", "Hello")],
            &[],
            None,
            true,
        );
        assert!(!out.contains("<retrieved_memory>"));
        assert!(out.ends_with("<|turn>model\n"));
    }

    #[test]
    fn gemma4_memory_block_omitted_when_empty_or_whitespace() {
        let f = Gemma4Format;
        for empty in &["", "   ", "\n\n"] {
            let out = f.render_prompt(
                "You are Wupi.",
                &[msg("user", "Hello")],
                &[],
                Some(empty),
                true,
            );
            assert!(!out.contains("<retrieved_memory>"), "empty block should not render: {out}");
        }
    }

    #[test]
    fn gemma4_assistant_role_becomes_model() {
        let f = Gemma4Format;
        let out = f.render_prompt("", &[msg("assistant", "hi")], &[], None, false);
        assert!(out.contains("<|turn>model\nhi<turn|>"));
        assert!(!out.contains("<|turn>assistant"));
    }

    #[test]
    fn gemma4_parses_thought_then_reply() {
        let f = Gemma4Format;
        let raw = "<|channel>thought\nI should greet them.\n<channel|>Hello there!";
        let parsed = f.parse_output(raw);
        assert_eq!(parsed.reasoning, "I should greet them.");
        assert_eq!(parsed.content, "Hello there!");
    }

    #[test]
    fn gemma4_parses_reply_only() {
        let f = Gemma4Format;
        let parsed = f.parse_output("Just a plain reply.");
        assert_eq!(parsed.content, "Just a plain reply.");
        assert_eq!(parsed.reasoning, "");
    }

    #[test]
    fn gemma4_strip_thinking_removes_thought_blocks() {
        let cleaned = strip_thinking("<|channel>thought\nsecret\n<channel|>visible");
        assert_eq!(cleaned, "visible");
    }

    #[test]
    fn gemma4_renders_model_turn_from_raw_output_when_present() {
        // Bug #3: when raw_output is present, the formatter renders it verbatim
        // (cache-coherent) instead of stripping thinking from cleaned content.
        let f = Gemma4Format;
        let mut m = msg("assistant", "visible reply");
        m.raw_output = "<|channel>thought\nsecret\n<channel|>visible reply".into();
        let out = f.render_prompt("", &[m], &[], None, false);
        assert!(
            out.contains("<|channel>thought\nsecret\n<channel|>visible reply"),
            "raw_output should be rendered verbatim for cache coherence, got: {out}"
        );
    }

    #[test]
    fn gemma4_falls_back_to_strip_thinking_without_raw_output() {
        // Legacy turns (no raw_output) still get the strip_thinking path.
        let f = Gemma4Format;
        let m = msg("assistant", "<|channel>thought\nsecret\n<channel|>visible");
        let out = f.render_prompt("", &[m], &[], None, false);
        assert!(
            !out.contains("<|channel>"),
            "legacy turn should strip thinking, got: {out}"
        );
        assert!(out.contains("visible"));
    }

    #[test]
    fn detect_gemma4_from_name() {
        // Locked naming convention (2026-07-12): chat model is always
        // `WUPI.gguf`, embeddings model is always `Embed.gguf`.
        assert_eq!(ModelFamily::from_model_name("WUPI.gguf"), ModelFamily::Gemma4);
        assert_eq!(ModelFamily::from_model_name("wupi.gguf"), ModelFamily::Gemma4);
        // Legacy/foreign Gemma filenames still detect.
        assert_eq!(ModelFamily::from_model_name("Gemma 12B.gguf"), ModelFamily::Gemma4);
        assert_eq!(ModelFamily::from_model_name("gemma-4-E4B.gguf"), ModelFamily::Gemma4);
        // Non-Gemma foreign files fall through to Plain.
        assert_eq!(ModelFamily::from_model_name("llama.gguf"), ModelFamily::Plain);
        assert_eq!(ModelFamily::from_model_name("Embed.gguf"), ModelFamily::Plain);
    }
}
