//! Wupi-as-game-manager intent router (Games app Seam E: the pivot's
//! headline feature).
//!
//! When a game is active (`GameEngine.is_some()`), Wupi's chat context
//! gains a second capability: she can read and mutate the active game's
//! scoped `<world_state>` via natural language. This module classifies the
//! player's message to Wupi and decides whether it's:
//!
//! - `MutateWorldState(delta)`: apply a state mutation ("make it stormy"
//!   → `{entities: {weather: "stormy"}}`).
//! - `QueryWorldState(what)`: return part of the state for Wupi to narrate
//!   ("what's the weather?").
//! - `NotACommand`: fall through to normal Wupi-assistant chat.
//!
//! # MVP compromise
//!
//! Intent detection is **heuristic** (keyword matching) for the MVP. This
//! WILL misroute edge cases. Phase 2 replaces it with an LLM-judge pre-pass
//! or a small classifier. Documented as a known limitation, not a hidden
//! bug: see the inline comments for what each branch covers and what it
//! misses.

use crate::schema::SchemaDelta;

/// Wupi's classification of a player message directed at her (not the
/// narrator) while a game is running.
#[derive(Debug, Clone)]
pub enum GameCommand {
    /// The player wants to change the game world. `delta` is the
    /// `SchemaDelta` to apply to the card's scoped schema.
    MutateWorldState(SchemaDelta),
    /// The player is asking about the game state. `what` is the focus
    /// (e.g. "weather", "inventory", "npcs"). Wupi will narrate the answer
    /// in her own voice: no mutation.
    QueryWorldState(String),
    /// Not a game-management request. Fall through to normal Wupi chat.
    NotACommand,
}

/// Classify a message to Wupi (while a game is active) into a GameCommand.
///
/// Returns `NotACommand` quickly for clearly non-management messages so
/// the chat path doesn't pay the heuristic cost in the common case where
/// the player is just chatting with Wupi.
///
/// The heuristic is **conservative toward `NotACommand`**: false-positives
/// (treating normal chat as a command) are worse than false-negatives
/// (missing a command: the player can rephrase). The bar to route to a
/// command is HIGH.
pub fn classify(text: &str) -> GameCommand {
    let lower = text.to_lowercase();
    let trimmed = lower.trim();

    if trimmed.is_empty() {
        return GameCommand::NotACommand;
    }

    // "what's the X", "show me X", "how is X", "status of X".
    let query_starters = [
        "what's ", "what is ", "whats ", "show me ", "show ",
        "how is ", "how's ", "status of ", "tell me about ",
        "list my ", "what do i have", "what am i carrying",
        "where am i", "who is here", "who's here",
    ];
    if query_starters.iter().any(|s| trimmed.starts_with(s)) {
        return GameCommand::QueryWorldState(extract_focus(trimmed));
    }

    // "make it X", "set X to Y", "change X to Y", "give me X", "remove X",
    // "teleport/travel to X", "fast-travel to X".
    let mutation_starters = [
        "make it ", "make the ", "make ",
        "set ", "change ", "turn ", "switch ",
        "give me ", "give alex ", "add ",
        "remove ", "delete ", "drop ",
        "teleport ", "travel to ", "fast-travel to ", "fast travel to ",
        "spawn ",
    ];
    if mutation_starters.iter().any(|s| trimmed.starts_with(s)) {
        // For the MVP we return a PLACEHOLDER delta: the actual LLM
        // translation ("make it stormy" → {weather: stormy}) happens in
        // `game_command::translate_to_delta`, called from `chat_send` after
        // classification. Returning an empty delta here keeps the type
        // signature honest; the caller will populate it.
        return GameCommand::MutateWorldState(SchemaDelta::default());
    }

    // Some management intents don't start with a clear verb but contain
    // strong domain keywords. Match a few high-value ones.
    let keyword_signals = ["inventory", "weather", "time of day", "fast travel"];
    if keyword_signals.iter().any(|kw| trimmed.contains(kw)) {
        // Distinguish query vs mutation by verb presence.
        if contains_mutation_verb(trimmed) {
            return GameCommand::MutateWorldState(SchemaDelta::default());
        }
        return GameCommand::QueryWorldState(extract_focus(trimmed));
    }

    GameCommand::NotACommand
}

/// Extract the focus noun from a query ("what's the weather" → "weather").
/// For MVP this is a simple last-word grab; Phase 2 will use the LLM.
fn extract_focus(text: &str) -> String {
    // Take the last whitespace-delimited token, stripped of punctuation.
    text.split_whitespace()
        .last()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .filter(|w| !w.is_empty())
        .unwrap_or_else(|| "state".to_string())
}

/// Mutation verbs that flip a keyword match from query to mutation.
fn contains_mutation_verb(text: &str) -> bool {
    let verbs = ["make", "set", "change", "turn", "give", "add", "remove", "spawn"];
    text.split_whitespace()
        .any(|w| verbs.contains(&w))
}

/// Translate a player's natural-language mutation request into a
/// `SchemaDelta` by asking the LLM (Wupi's chat context, briefly). This is
/// called from `chat_send` AFTER `classify` returns `MutateWorldState`.
///
/// The LLM is given the request + current world_state JSON and asked to
/// emit ONLY the changed keys as a delta. Same prompt structure as the
/// schema engine's delta pass, but driven by an explicit player command
/// rather than an automatic per-turn summarization.
///
/// **MVP note:** this function returns the prompt text; the actual LLM call
/// + parse happens in the caller (which has access to the GameEngine/
/// SchemaEngine). Keeping the prompt-construction pure lets us unit-test it
/// without a model.
pub fn render_translation_prompt(
    player_request: &str,
    current_state_json: &str,
    deferred_attempts: &[crate::schema_engine::FailedAttempt],
) -> String {
    let mut out = String::with_capacity(1024);
    out.push_str("<|turn>system\n");
    out.push_str(TRANSLATION_INSTRUCTION);
    out.push_str("<turn|>\n");
    out.push_str("<|turn>user\n");
    out.push_str("Current world state:\n");
    out.push_str(current_state_json);
    out.push_str("\n\nPlayer's request to Wupi:\n");
    out.push_str(player_request);
    // Deferred re-attempt context (fail-proof contract §5 layer 3). When the
    // player's previous request failed all 3 passes, fold its trigger + errors
    // in here so the model has a fresh shot with the new request as anchor.
    // The carrier carries the *trigger*, not the broken raw output: the new
    // request + the prior errors are the useful signal.
    if !deferred_attempts.is_empty() {
        out.push_str("\n\n[Previously deferred state changes — re-attempt with the above request as the primary context:]\n");
        for (i, attempt) in deferred_attempts.iter().enumerate() {
            let trigger = attempt
                .trigger
                .as_deref()
                .or_else(|| attempt.exchange.as_ref().map(|(u, _)| u.as_str()))
                .unwrap_or("(no trigger recorded)");
            out.push_str(&format!(
                "  {}. prior request: {:?}\n     prior errors: {}\n",
                i + 1,
                trigger.chars().take(200).collect::<String>(),
                attempt.errors
            ));
        }
    }
    out.push_str("\n\nEmit ONLY the JSON delta object (changed keys only). If the request is not a state mutation, emit {}.\n");
    out.push_str("<turn|>\n");
    out.push_str("<|turn>model\n");
    out
}

const TRANSLATION_INSTRUCTION: &str = "\
You are translating a player's natural-language request into a state delta
for the roleplay game's world_state. The world_state is a JSON object with
three top-level keys: summary (string), recent_events (array of strings),
entities (object of key → string).

Emit ONLY the changed keys as a JSON delta with this shape:
{
  \"summary\": \"...\" (optional, only if the arc shifted),
  \"recent_events\": [\"...\"] (optional, append-only),
  \"entities\": {\"key\": \"value\" | null} (optional; null deletes a key)
}

Do NOT re-emit unchanged keys. Do NOT wrap the JSON in markdown fences.
If the request cannot be expressed as a state change (e.g. it's a question
or a normal chat message), emit an empty object: {}";

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_message_is_not_a_command() {
        assert!(matches!(classify(""), GameCommand::NotACommand));
        assert!(matches!(classify("   "), GameCommand::NotACommand));
    }

    #[test]
    fn normal_chat_is_not_a_command() {
        assert!(matches!(classify("hey wupi how are you"), GameCommand::NotACommand));
        assert!(matches!(classify("tell me a joke"), GameCommand::NotACommand));
        assert!(matches!(classify("nya~"), GameCommand::NotACommand));
    }

    #[test]
    fn query_starters_route_to_query() {
        match classify("what's the weather") {
            GameCommand::QueryWorldState(focus) => assert_eq!(focus, "weather"),
            _ => panic!("expected QueryWorldState"),
        }
        match classify("show me my inventory") {
            GameCommand::QueryWorldState(_) => {}
            _ => panic!("expected QueryWorldState"),
        }
        match classify("where am i") {
            GameCommand::QueryWorldState(_) => {}
            _ => panic!("expected QueryWorldState"),
        }
    }

    #[test]
    fn mutation_starters_route_to_mutate() {
        assert!(matches!(
            classify("make it stormy"),
            GameCommand::MutateWorldState(_)
        ));
        assert!(matches!(
            classify("give me a sword"),
            GameCommand::MutateWorldState(_)
        ));
        assert!(matches!(
            classify("travel to the dungeon"),
            GameCommand::MutateWorldState(_)
        ));
        assert!(matches!(
            classify("set weather to rain"),
            GameCommand::MutateWorldState(_)
        ));
    }

    #[test]
    fn keyword_without_verb_routes_to_query() {
        // "the weather is nice": mentions weather but no mutation verb.
        match classify("the weather is nice today") {
            GameCommand::QueryWorldState(_) => {}
            other => panic!("expected QueryWorldState, got {other:?}"),
        }
    }

    #[test]
    fn keyword_with_verb_routes_to_mutate() {
        // "change the weather": keyword + mutation verb.
        assert!(matches!(
            classify("change the weather"),
            GameCommand::MutateWorldState(_)
        ));
    }

    #[test]
    fn render_translation_prompt_contains_request_and_state() {
        let prompt = render_translation_prompt(
            "make it stormy",
            "{\"entities\":{\"weather\":\"clear\"}}",
            &[], // no deferred attempts in the common case
        );
        assert!(prompt.contains("make it stormy"));
        assert!(prompt.contains("\"weather\":\"clear\""));
        assert!(prompt.contains("<|turn>system"));
        assert!(prompt.contains("<|turn>model"));
    }

    #[test]
    fn render_translation_prompt_folds_deferred_attempts() {
        // Fail-proof contract layer 3: prior translation failures must
        // surface in the next request's prompt.
        let deferred = vec![crate::schema_engine::FailedAttempt {
            exchange: None,
            trigger: Some("prior failed request".to_string()),
            errors: "pass 1 parse: ... | pass 2 validation: ...".to_string(),
            passes_used: 3,
        }];
        let prompt = render_translation_prompt(
            "new request",
            "{}",
            &deferred,
        );
        assert!(prompt.contains("Previously deferred"));
        assert!(prompt.contains("prior failed request"));
        assert!(prompt.contains("pass 1 parse"));
    }

    #[test]
    fn extract_focus_strips_punctuation() {
        assert_eq!(extract_focus("what's the weather?"), "weather");
        assert_eq!(extract_focus("show me my inventory."), "inventory");
    }
}

// Top-level `Display` impl (Phase E cleanup, 2026-07-18). Was previously
// inside `#[cfg(test)]` to silence unused-Debug-format warnings on `_` match
// arms in tests. Promoted to the module level: it's useful for log lines in
// the route helpers too (`tracing::info!(?cmd, ...)` falls back to Display
// when Debug isn't used). No behavior change.
impl std::fmt::Display for GameCommand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GameCommand::MutateWorldState(_) => write!(f, "MutateWorldState"),
            GameCommand::QueryWorldState(focus) => write!(f, "QueryWorldState({focus})"),
            GameCommand::NotACommand => write!(f, "NotACommand"),
        }
    }
}
