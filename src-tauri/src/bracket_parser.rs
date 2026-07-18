//! Bracket-command extractor for narrator output (Games app Seam 3).
//!
//! The narrator emits bracket commands alongside its prose to drive the UI
//! deterministically. This module parses those out of the *final raw output*
//! (post-generation), NOT from the token stream — brackets are scene-level
//! events, not token-level concerns, so they're best extracted once from the
//! complete text rather than incrementally during streaming.
//!
//! # Supported commands (mirror `narrator_prompt::BRACKET_PROTOCOL`)
//!
//! - `[CHARACTER_TURN:npc_id]` ... `[CHARACTER_TURN:end]` — an NPC spoke.
//! - `[OBJECT id=iron_chest state=open]` — an object's state changed.
//! - `[FX rain]` — a scene effect should activate.
//!
//! # Design
//!
//! Pure string parsing — no regex backtracking, no re-tokenizing (Prime
//! Directive §1B.2). One linear scan over the text, extracting bracketed
//! regions. The prose left over after extraction is the cleaned narrator
//! output the UI renders.
//!
//! Robustness: malformed brackets (`[OBJECT id=x]` missing `state=`,
//! `[CHARACTER_TURN:` unterminated) are silently dropped, not fatal. The
//! narrator is a 12B model; we tolerate noisy output.

use serde::Serialize;

/// One bracket command extracted from narrator output.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BracketCommand {
    /// An NPC spoke. `npc_id` matches a card's `start_npc_ids`. `line` is
    /// the prose between the open and close tags.
    CharacterTurn { npc_id: String, line: String },
    /// An object's state changed.
    Object { id: String, state: String },
    /// A scene effect should activate.
    Fx { effect: String },
}

/// The result of parsing narrator output: the bracket commands found + the
/// prose with brackets removed (for UI rendering).
#[derive(Debug, Clone, Default, Serialize)]
pub struct ParsedNarration {
    /// Bracket commands in the order they appeared.
    pub commands: Vec<BracketCommand>,
    /// The narrator prose with all bracket regions stripped out. What the
    /// UI renders as the dialogue box.
    pub prose: String,
}

/// Parse a narrator's complete raw output into commands + cleaned prose.
///
/// The output is the verbatim text the model emitted (Gemma4 channel
/// protocol is stripped upstream by `chat_format::extract_reply_channel` or
/// equivalent; this function sees pure narrator text).
///
/// Strategy: walk the text, when we see `[`, attempt to match a known
/// command pattern. On match, push a command + skip past the bracket. On
/// no match, copy the `[` into prose and continue (graceful — better to
/// leak a literal bracket than misparse).
///
/// `CHARACTER_TURN` is the only multi-region command (open + body + close).
/// `OBJECT` and `FX` are single-region. This keeps the parser linear and
/// the brackets-plus-prose invariant simple.
pub fn parse(raw: &str) -> ParsedNarration {
    let bytes = raw.as_bytes();
    let mut commands = Vec::new();
    let mut prose = String::with_capacity(raw.len());
    let mut i = 0;

    while i < bytes.len() {
        // Find the next `[` from the current position.
        let Some(rel) = bytes[i..].iter().position(|&b| b == b'[') else {
            prose.push_str(&raw[i..]);
            break;
        };
        let start = i + rel;

        // Emit any prose before the bracket.
        prose.push_str(&raw[i..start]);

        // Find the closing `]`.
        let Some(end_rel) = bytes[start..].iter().position(|&b| b == b']') else {
            // Unterminated bracket — emit the `[` literally and advance one
            // byte (so we don't loop forever on a stray `[`).
            prose.push('[');
            i = start + 1;
            continue;
        };
        let end = start + end_rel;
        let bracket = &raw[start + 1..end]; // contents between [ and ]

        // Try to match a command. On match, push it; on miss, the bracket
        // content is emitted as literal prose (preserves original text).
        // `text_after` is the raw text starting just past the closing `]` —
        // used by CHARACTER_TURN to find its `[CHARACTER_TURN:end]` body
        // terminator. Indices returned by `parse_one` are relative to this
        // slice (not the full `raw`), so the caller adds `end + 1`.
        let text_after = &raw[end + 1..];
        match parse_one(bracket, text_after) {
            Some((cmd, consumed_after_bracket)) => {
                commands.push(cmd);
                // For CHARACTER_TURN we also consumed the body + close tag;
                // advance past them.
                i = end + 1 + consumed_after_bracket;
            }
            None => {
                // Not a recognized command. Emit the bracket verbatim.
                prose.push('[');
                prose.push_str(bracket);
                prose.push(']');
                i = end + 1;
            }
        }
    }

    ParsedNarration { commands, prose }
}

/// Attempt to parse one bracket's contents into a `BracketCommand`.
/// Returns `(command, bytes_consumed_after_the_closing_bracket)` — the
/// after-bracket consumption is nonzero only for `CHARACTER_TURN`, which
/// swallows its body + close tag.
///
/// `text_after` is the raw text starting just after the closing `]` (used
/// to find the `CHARACTER_TURN:end` terminator). Indices returned are
/// relative to this slice.
fn parse_one(bracket: &str, text_after: &str) -> Option<(BracketCommand, usize)> {
    let bracket = bracket.trim();

    if let Some(rest) = bracket.strip_prefix("CHARACTER_TURN:") {
        let npc_id = rest.trim().to_string();
        if npc_id == "end" || npc_id.is_empty() {
            // A stray close tag or empty open tag — drop it.
            return Some((BracketCommand::CharacterTurn {
                npc_id: String::new(),
                line: String::new(),
            }, 0));
        }
        // Find the matching [CHARACTER_TURN:end] in `text_after`. The body
        // between is the NPC's spoken line.
        let close = "[CHARACTER_TURN:end]";
        if let Some(end_idx) = text_after.find(close) {
            let line = text_after[..end_idx].trim().to_string();
            return Some((
                BracketCommand::CharacterTurn { npc_id, line },
                end_idx + close.len(),
            ));
        }
        // No close tag — treat the rest of the output as the line (graceful).
        let line = text_after.trim().to_string();
        return Some((
            BracketCommand::CharacterTurn { npc_id, line },
            text_after.len(),
        ));
    }

    if let Some(rest) = bracket.strip_prefix("OBJECT") {
        // Parse `id=x state=y` (whitespace-tolerant).
        let mut id = None;
        let mut state = None;
        for tok in rest.split_whitespace() {
            if let Some(v) = tok.strip_prefix("id=") {
                id = Some(v.to_string());
            } else if let Some(v) = tok.strip_prefix("state=") {
                state = Some(v.to_string());
            }
        }
        if let (Some(id), Some(state)) = (id, state) {
            return Some((BracketCommand::Object { id, state }, 0));
        }
        return None;
    }

    if let Some(rest) = bracket.strip_prefix("FX") {
        let effect = rest.trim().to_string();
        if !effect.is_empty() {
            return Some((BracketCommand::Fx { effect }, 0));
        }
        return None;
    }

    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_object_command() {
        let raw = "Alex approaches the hearth. [OBJECT id=iron_chest state=open] The lock gives way.";
        let parsed = parse(raw);
        assert_eq!(parsed.commands.len(), 1);
        assert_eq!(
            parsed.commands[0],
            BracketCommand::Object {
                id: "iron_chest".into(),
                state: "open".into(),
            }
        );
        assert!(parsed.prose.contains("Alex approaches the hearth."));
        assert!(parsed.prose.contains("The lock gives way."));
        assert!(!parsed.prose.contains("[OBJECT"));
    }

    #[test]
    fn extracts_fx_command() {
        let raw = "The storm breaks. [FX rain] Water drums on the shutters.";
        let parsed = parse(raw);
        assert_eq!(parsed.commands.len(), 1);
        assert_eq!(
            parsed.commands[0],
            BracketCommand::Fx { effect: "rain".into() }
        );
        assert!(parsed.prose.contains("The storm breaks."));
        assert!(!parsed.prose.contains("[FX"));
    }

    #[test]
    fn extracts_character_turn_with_body() {
        let raw = "[CHARACTER_TURN:gorm] Rain's bad tonight. [CHARACTER_TURN:end] Gorm dries a mug.";
        let parsed = parse(raw);
        assert_eq!(parsed.commands.len(), 1);
        match &parsed.commands[0] {
            BracketCommand::CharacterTurn { npc_id, line } => {
                assert_eq!(npc_id, "gorm");
                assert_eq!(line, "Rain's bad tonight.");
            }
            _ => panic!("expected CharacterTurn"),
        }
        // The body was consumed into the command; prose has only the trailing bit.
        assert!(parsed.prose.contains("Gorm dries a mug."));
        assert!(!parsed.prose.contains("Rain's bad tonight."));
    }

    #[test]
    fn extracts_multiple_commands_in_order() {
        let raw = "[FX thunder] [OBJECT id=door state=closed] A shape moves outside.";
        let parsed = parse(raw);
        assert_eq!(parsed.commands.len(), 2);
        assert!(matches!(parsed.commands[0], BracketCommand::Fx { .. }));
        assert!(matches!(parsed.commands[1], BracketCommand::Object { .. }));
    }

    #[test]
    fn no_brackets_passes_through_unchanged() {
        let raw = "The fire crackles. Rain falls steadily.";
        let parsed = parse(raw);
        assert!(parsed.commands.is_empty());
        assert_eq!(parsed.prose, raw);
    }

    #[test]
    fn unknown_bracket_emitted_as_literal() {
        // `[NOTE:foo]` isn't a recognized command — preserve it in prose.
        let raw = "Strange [NOTE:foo] marker.";
        let parsed = parse(raw);
        assert!(parsed.commands.is_empty());
        assert_eq!(parsed.prose, raw);
    }

    #[test]
    fn unterminated_bracket_emits_literal() {
        let raw = "Trailing [unterminated";
        let parsed = parse(raw);
        assert!(parsed.commands.is_empty());
        assert!(parsed.prose.contains("[unterminated"));
    }

    #[test]
    fn malformed_object_dropped() {
        // Missing state= → not a valid command → bracket emitted verbatim.
        let raw = "Alex looks. [OBJECT id=chest] Nothing happens.";
        let parsed = parse(raw);
        assert!(parsed.commands.is_empty());
        assert!(parsed.prose.contains("[OBJECT id=chest]"));
    }

    #[test]
    fn character_turn_without_close_consumes_rest() {
        // Graceful: no end tag → treat rest of output as the line.
        let raw = "Alex nods. [CHARACTER_TURN:gorm] Welcome, traveller.";
        let parsed = parse(raw);
        assert_eq!(parsed.commands.len(), 1);
        if let BracketCommand::CharacterTurn { npc_id, line } = &parsed.commands[0] {
            assert_eq!(npc_id, "gorm");
            assert_eq!(line, "Welcome, traveller.");
        } else {
            panic!("expected CharacterTurn");
        }
    }
}
