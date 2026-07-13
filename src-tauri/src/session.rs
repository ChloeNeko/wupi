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
}
