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

    /// Persist the conversation **atomically**: serialize, write to a sibling
    /// temp file, then `rename` it over the destination.
    ///
    /// Atomicity matters because `save()` runs on every message: a plain
    /// `fs::write(path, ...)` truncates-then-writes, so a crash / power loss /
    /// disk-full mid-write leaves `session.json` truncated and the ENTIRE
    /// conversation unrecoverable. The temp+rename pattern guarantees the
    /// destination is either the previous complete file or the new complete
    /// file — never a half-written middle state.
    ///
    /// - The temp file is in the same directory as `path` (same volume →
    ///   `rename` is atomic; a cross-device rename would degrade to copy+delete
    ///   and lose the atomicity guarantee).
    /// - On Windows, `std::fs::rename` over an existing file uses
    ///   `MOVEFILE_REPLACE_EXISTING`, so the overwrite is atomic there too.
    /// - A stale `.tmp` from a prior crashed save is removed first so we never
    ///   accidentally rename a leftover corrupt temp over the good file.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        // Temp file: sibling of the destination, same directory/volume.
        let tmp_path = temp_path_for(path);

        // Clear any stale temp from a crashed prior save (ignore NotFound).
        let _ = std::fs::remove_file(&tmp_path);

        // Write + flush the temp so the bytes are durable before the rename.
        // Without fsync, a crash after rename could expose an empty/journaled
        // file once the OS writeback catches up.
        {
            let mut file = std::fs::File::create(&tmp_path)?;
            std::io::Write::write_all(&mut file, json.as_bytes())?;
            std::io::Write::flush(&mut file)?;
            // Sync the file's data to disk. AllDataSync because metadata
            // (size) for an existing file is cheap; for the rename we only
            // truly need the data, but AllDataSync is the safer choice and
            // the perf cost is one extra syscall on a tiny JSON file.
            let _ = file.sync_all();
        }

        // Atomic replace. On Windows this uses MOVEFILE_REPLACE_EXISTING.
        std::fs::rename(&tmp_path, path)
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

    /// Windowed variant of [`assemble_api_messages`]: prepends the system
    /// message in full, then takes only the LAST `window` stored messages.
    ///
    /// This is the §2F eager-prefill sliding window (2026-07-13). Capping
    /// visible history to a fixed message count (regardless of token budget)
    /// does two things:
    /// 1. Makes `truncate_to_fit` effectively never fire (4 short turns +
    ///    system ≪ the ~3000-token budget), eliminating truncation-driven
    ///    cold-resets.
    /// 2. Keeps the stable prefix short and predictable so eager prefill is
    ///    cheap and the delta (memory block + new user) stays small.
    ///
    /// Memory (M) is supposed to backfill the evicted older turns via
    /// retrieval — that's the whole point of the offload. If retrieval misses,
    /// the model genuinely sees less recency than before; the cap is in a
    /// `const` at the call site, trivially tunable.
    ///
    /// **Alternating roll-up (2026-07-17):** before returning, consecutive
    /// same-role messages are merged into one block (content joined with
    /// `\n\n`, `raw_output` joined with `\n` for assistant turns). This
    /// guarantees a clean user↔assistant alternation for the downstream
    /// backend — GLM gets the strictly-alternating payload its chat template
    /// expects, and local Gemma 4 never emits adjacent `<|turn>user` or
    /// `<|turn>model` blocks that would confuse the chat-template tracking.
    /// The roll-up is a pure normalization on the assembled slice; stored
    /// session state is untouched. See `normalize_alternating`.
    pub fn assemble_api_messages_windowed(
        &self,
        system_prompt: &str,
        window: usize,
    ) -> Vec<ApiMessage> {
        let start = self.messages.len().saturating_sub(window);
        let visible = &self.messages[start..];

        let mut out = Vec::with_capacity(visible.len() + 1);
        if !system_prompt.is_empty() {
            out.push(ApiMessage {
                role: "system".into(),
                content: system_prompt.into(),
                raw_output: String::new(),
            });
        }
        for m in visible {
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
        normalize_alternating(out)
    }
}

/// Merge consecutive same-role messages into single blocks. Applied to the
/// assembled `Vec<ApiMessage>` before it leaves `assemble_api_messages_windowed`,
/// so BOTH backends (local Gemma 4 via `Gemma4Format::render_prompt`, online
/// GLM via `HttpBackend::stream`) receive the same strictly-alternating
/// payload. Session storage is left untouched — this is a presentation-layer
/// transform only.
///
/// Rules:
/// - Walk the slice; if message `i` and `i+1` share a role, their `content`
///   strings are joined with `\n\n` and their `raw_output` strings with `\n`.
///   The pair collapses into one entry; the walk continues from the merged
///   entry so runs of 3+ same-role messages fold fully.
/// - The system block (index 0) participates: it never has a same-role
///   neighbor in practice (the conversation starts with user), but if a
///   legacy/hand-edited session had a leading system+system pair it would
///   merge cleanly rather than emit two system turns.
/// - `raw_output` is merged too because the local formatter renders assistant
///   turns from `raw_output` when present (Bug #3, §2C) — joining only
///   `content` would desync the rendered tokens from the KV cache. Assistant
///   `raw_output` blocks are the Gemma4 channel protocol; joining with `\n`
///   (not `\n\n`) keeps the turn boundary inside the merged block legible.
///
/// Empty messages are NOT dropped — an empty user turn is still a turn (the
/// backend's alternation contract doesn't care about content length).
pub fn normalize_alternating(messages: Vec<ApiMessage>) -> Vec<ApiMessage> {
    if messages.len() < 2 {
        return messages;
    }
    let mut out: Vec<ApiMessage> = Vec::with_capacity(messages.len());
    for m in messages {
        if let Some(last) = out.last_mut() {
            if last.role == m.role {
                // Same-role neighbor: roll up into the last block.
                if !last.content.is_empty() && !m.content.is_empty() {
                    last.content.push_str("\n\n");
                }
                last.content.push_str(&m.content);
                if !last.raw_output.is_empty() && !m.raw_output.is_empty() {
                    last.raw_output.push('\n');
                }
                last.raw_output.push_str(&m.raw_output);
                continue;
            }
        }
        out.push(m);
    }
    out
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

/// Build a sibling temp-file path for an atomic save: same directory + volume
/// as `path` (so `rename` is atomic), with a `.tmp` suffix on the file stem.
///
/// `session.json` → `session.json.tmp`. We keep the full original name as a
/// prefix so the temp is visually obvious in the dir and `save`'s stale-temp
/// cleanup (`remove_file`) targets exactly this one file — not a random one.
fn temp_path_for(path: &Path) -> std::path::PathBuf {
    let mut name = path
        .file_name()
        .map(std::ffi::OsString::from)
        .unwrap_or_else(|| std::ffi::OsString::from("wupi.tmp"));
    name.push(".tmp");
    path.with_file_name(name)
}

fn gen_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let mut bytes = [0u8; 6];
    prng_fill(&mut bytes);
    let rand: String = bytes.iter().map(|b| format!("{:02x}", b)).collect();
    format!("m_{:x}_{}", ts, &rand[..6])
}

/// Fill `buf` with pseudo-random bytes via a xorshift64 seeded from wall-clock
/// nanos. NOT cryptographic — used only for message-ID uniqueness in local
/// chat (see `gen_id`). The name reflects what it is: a PRNG, not the OS
/// CSPRNG (the old name `getrandom_fill` falsely implied the `getrandom`
/// syscall / crate). Renamed 2026-07-13 (Gemini review).
fn prng_fill(buf: &mut [u8]) {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Unique path in the temp dir, namespaced by pid + a counter so parallel
    /// test runs and repeated invocations don't collide. Avoids pulling in the
    /// `tempfile` crate for what is a one-line unique-name need.
    fn unique_test_path(name: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!(
            "wupi_test_{}{}_{}_{}.json",
            std::process::id(),
            ts,
            n,
            name
        ))
    }

    /// Build a throwaway conversation with one message for round-trip tests.
    fn sample_conv() -> Conversation {
        let mut c = Conversation::new();
        c.add_message(Role::User, "hello".into());
        c
    }

    /// Clean up both the main file and its sibling temp so the temp dir
    /// doesn't accumulate test artifacts. NotFound is fine.
    fn cleanup(path: &Path) {
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(temp_path_for(path));
    }

    #[test]
    fn temp_path_is_sibling_with_tmp_suffix() {
        let p = std::path::PathBuf::from("dir/session.json");
        assert_eq!(temp_path_for(&p), std::path::PathBuf::from("dir/session.json.tmp"));
        // No-extension path still gets .tmp.
        let p2 = std::path::PathBuf::from("dir/session");
        assert_eq!(temp_path_for(&p2), std::path::PathBuf::from("dir/session.tmp"));
    }

    #[test]
    fn save_then_load_roundtrips() {
        let path = unique_test_path("roundtrip");
        cleanup(&path);
        let conv = sample_conv();

        conv.save(&path).expect("save should succeed");

        // Temp must be gone after a successful save (renamed away).
        assert!(!temp_path_for(&path).exists(), "temp file left behind");
        // Main file exists and round-trips.
        let loaded = Conversation::load(&path).expect("load should succeed");
        assert_eq!(loaded.messages.len(), 1);
        assert_eq!(loaded.messages[0].content, "hello");

        cleanup(&path);
    }

    #[test]
    fn save_does_not_leave_temp_file_on_success() {
        // Regression guard: if a future refactor drops the rename or makes it
        // non-atomic, this test fails because the .tmp would remain.
        let path = unique_test_path("no_temp_leftover");
        cleanup(&path);
        sample_conv().save(&path).expect("save");
        assert!(path.exists(), "destination must exist");
        assert!(
            !temp_path_for(&path).exists(),
            "temp file must be renamed away, not left behind"
        );
        cleanup(&path);
    }

    #[test]
    fn save_overwrites_existing_file_in_place() {
        // Save once, then save a second time with different content. The
        // destination must reflect the second save and the temp must be gone.
        let path = unique_test_path("overwrite");
        cleanup(&path);

        let mut first = Conversation::new();
        first.add_message(Role::User, "first".into());
        first.save(&path).expect("first save");

        let mut second = Conversation::new();
        second.add_message(Role::User, "second".into());
        second.add_message(Role::User, "third".into());
        second.save(&path).expect("second save");

        let loaded = Conversation::load(&path).expect("load");
        assert_eq!(loaded.messages.len(), 2, "second save should win");
        assert_eq!(loaded.messages[0].content, "second");
        assert!(!temp_path_for(&path).exists(), "temp leaked after overwrite");

        cleanup(&path);
    }

    #[test]
    fn stale_temp_from_prior_crash_is_cleared_before_rename() {
        // Simulate a prior crashed save: drop a stale temp file in place, then
        // save. The stale temp must not survive (it gets removed + replaced by
        // the new one, which is then renamed away).
        let path = unique_test_path("stale_temp");
        cleanup(&path);

        let tmp = temp_path_for(&path);
        std::fs::write(&tmp, b"stale garbage from a crashed save").expect("seed stale temp");
        assert!(tmp.exists(), "precondition: stale temp exists");

        sample_conv().save(&path).expect("save should clear stale temp");
        assert!(path.exists(), "destination written");
        assert!(
            !tmp.exists(),
            "stale temp must be cleared (removed then renamed away), not left as garbage"
        );

        cleanup(&path);
    }

    #[test]
    fn destination_survives_when_only_temp_would_be_corrupt() {
        // The atomicity guarantee: if a write were to fail partway, the
        // DESTINATION must remain the previously-saved good file. We can't
        // easily simulate a mid-write crash, but we CAN prove the invariant
        // indirectly: save a known-good file, then confirm a second save that
        // completes leaves a valid file. The point of the temp+rename design
        // is that the destination is never opened for write directly.
        let path = unique_test_path("atomicity");
        cleanup(&path);

        let mut good = Conversation::new();
        good.add_message(Role::User, "known-good state".into());
        good.save(&path).expect("first save");
        let before = std::fs::read(&path).expect("read good file");

        // Second save with new content.
        sample_conv().save(&path).expect("second save");
        let after = std::fs::read(&path).expect("read new file");

        assert_ne!(before, after, "second save must actually replace content");
        // Both reads must be valid JSON (no partial-write corruption).
        assert!(
            serde_json::from_slice::<Conversation>(&before).is_ok(),
            "pre-overwrite file must be valid"
        );
        assert!(
            serde_json::from_slice::<Conversation>(&after).is_ok(),
            "post-overwrite file must be valid"
        );

        cleanup(&path);
    }

    // ── normalize_alternating (the Alternating Roll-Up) ─────────────────
    fn api(role: &str, content: &str) -> ApiMessage {
        ApiMessage {
            role: role.into(),
            content: content.into(),
            raw_output: String::new(),
        }
    }
    fn api_raw(role: &str, content: &str, raw: &str) -> ApiMessage {
        ApiMessage {
            role: role.into(),
            content: content.into(),
            raw_output: raw.into(),
        }
    }

    #[test]
    fn normalize_keeps_already_alternating_unchanged() {
        let msgs = vec![
            api("system", "sys"),
            api("user", "hi"),
            api("assistant", "hello"),
            api("user", "how are you?"),
            api("assistant", "fine"),
        ];
        let out = normalize_alternating(msgs);
        assert_eq!(out.len(), 5);
        assert_eq!(out.iter().map(|m| m.role.as_str()).collect::<Vec<_>>(),
                   vec!["system", "user", "assistant", "user", "assistant"]);
    }

    #[test]
    fn normalize_merges_consecutive_user_messages() {
        // Simulates the "user clicks Save / fires multiple commands" case:
        // two user turns in a row should collapse into one.
        let msgs = vec![
            api("system", "sys"),
            api("user", "save this"),
            api("user", "and also that"),
            api("assistant", "done"),
        ];
        let out = normalize_alternating(msgs);
        assert_eq!(out.len(), 3, "two user msgs merge into one");
        assert_eq!(out[1].role, "user");
        assert_eq!(out[1].content, "save this\n\nand also that");
        assert_eq!(out[2].role, "assistant");
    }

    #[test]
    fn normalize_merges_consecutive_assistant_messages_with_raw_output() {
        // Cache-coherence (Bug #3): raw_output must be merged alongside
        // content, otherwise the local formatter renders the merged turn
        // from a stale raw_output and desyncs the KV cache.
        let msgs = vec![
            api("user", "q"),
            api_raw("assistant", "part 1", "<raw1>"),
            api_raw("assistant", "part 2", "<raw2>"),
            api("user", "next"),
        ];
        let out = normalize_alternating(msgs);
        assert_eq!(out.len(), 3, "two assistant msgs merge into one");
        assert_eq!(out[1].content, "part 1\n\npart 2");
        assert_eq!(out[1].raw_output, "<raw1>\n<raw2>",
                   "raw_output joined with single \\n (not \\n\\n)");
    }

    #[test]
    fn normalize_folds_runs_of_three_or_more() {
        let msgs = vec![
            api("system", "sys"),
            api("user", "a"),
            api("user", "b"),
            api("user", "c"),
            api("assistant", "reply"),
        ];
        let out = normalize_alternating(msgs);
        assert_eq!(out.len(), 3);
        assert_eq!(out[1].content, "a\n\nb\n\nc");
    }

    #[test]
    fn normalize_handles_empty_messages_without_dropping() {
        // An empty user turn is still a turn — don't drop it. When the first
        // of the merged pair is empty, no leading separator is emitted (the
        // \n\n guard fires only when BOTH sides have content).
        let msgs = vec![
            api("user", ""),
            api("user", "real message"),
            api("assistant", "reply"),
        ];
        let out = normalize_alternating(msgs);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].content, "real message",
                   "empty-leading case yields just the non-empty content");
        assert_eq!(out[1].content, "reply");
    }

    #[test]
    fn normalize_preserves_system_at_index_zero() {
        // A legacy session with two leading system messages should merge them
        // into the index-0 system block, not emit two system turns.
        let msgs = vec![
            api("system", "directive A"),
            api("system", "directive B"),
            api("user", "hi"),
        ];
        let out = normalize_alternating(msgs);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].role, "system");
        assert_eq!(out[0].content, "directive A\n\ndirective B");
        assert_eq!(out[1].role, "user");
    }

    #[test]
    fn normalize_empty_and_single_pass_through() {
        assert_eq!(normalize_alternating(vec![]).len(), 0);
        let one = vec![api("user", "solo")];
        assert_eq!(normalize_alternating(one).len(), 1);
    }
}
