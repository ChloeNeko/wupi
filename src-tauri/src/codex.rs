//! Codex: authored reference lore, seeded from disk at startup.
//!
//! A Codex entry is reference knowledge (system documentation, world
//! background) that lives in the SAME `memories` table as episodic turns,
//! distinguished by a `metadata_json` tag: `{"kind":"codex","title":...,
//! "hash":...}`. It is retrieved by the existing Memory v2 pipeline (same
//! embedder, same vec0 index, same RRF fusion) and rendered by
//! `memory::render_memory_block` under a distinct "reference knowledge"
//! epistemic frame (factual background to internalize, NOT archival records to
//! distrust). See AGENTS.md §2P.
//!
//! Source format: plain `.md` files in a `docs/` directory (renamed from
//! `codex/` on 2026-07-17: `resolve_codex_dir` in `lib.rs` walks for `docs`),
//! each with an optional YAML-ish front-matter block (`---\ntitle: X\ntags:
//! a, b\n---`) + a prose body. The seed loader parses each file, computes a
//! content hash, and reconciles the source set against what's already stored -
//! inserting new entries, updating changed ones (delete + re-insert), and
//! purging orphans (source file deleted). This is idempotent: re-running
//! against an unchanged source set produces no writes.
//!
//! Design contract (mirrors `sim_card.rs` + the embedder's graceful-
//! degradation pattern): a missing/empty `docs/` dir or a malformed file is
//! logged-and-skipped, never fatal. The Codex is best-effort; a bad source
//! file must never kill the OS boot.
//!
//! Per-file length budget: each `.md` body must stay under ~350 tokens (~1400
//! chars). `Embed.gguf` (bge-small) truncates silently at 512 tokens, so a
//! long reference doc gets a garbage embedding and scores near the floor even
//! on a perfect match. Split long docs into multiple small files rather than
//! building a chunking engine (Codex v1 deliberately defers chunking: see
//! §2N landmine #6). The loader warns (does not reject) when a body exceeds
//! the heuristic budget.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::Path;

use crate::memory::{MemoryEngine, MemoryId};
use crate::memory_embedder::Embedder;

/// The result of a seed run: logged at startup so the operator can see at a
/// glance whether the Codex synced cleanly. All four counts are mutually
/// exclusive (each source file resolves to exactly one outcome).
#[derive(Debug, Clone, Default)]
pub struct ReconcileReport {
    /// New source files inserted into the store.
    pub seeded: usize,
    /// Source files whose content hash changed since last seed (delete + re-insert).
    pub updated: usize,
    /// Stored entries whose source file no longer exists (purged).
    pub purged: usize,
    /// Source files whose hash matches the stored entry (no write needed).
    pub unchanged: usize,
}

/// One parsed Codex source file: title + tags from front-matter, body is the
/// prose, hash is over the raw file bytes. Ephemeral; lives only for the
/// reconcile pass.
struct ParsedEntry {
    title: String,
    tags: Vec<String>,
    body: String,
    hash: u64,
}

/// Seed the Codex: parse every `.md` in `codex_dir`, reconcile against the
/// Codex entries already stored in the active card partition, and apply the
/// minimal set of inserts/updates/deletes.
///
/// The reconcile matches on `title` (the stable key: a renamed file is a
/// delete + insert, by design) and detects changes via `hash` (over raw file
/// bytes). All DB ops go through the existing `MemoryEngine` async methods;
/// this fn is async and awaits them in sequence (N is small, ~5-10 files).
///
/// `codex_dir` missing or empty → returns an empty report (graceful, not an
/// error). A parse failure on one file → logs a warning and skips that file;
/// the rest still seed. Only a systemic failure (e.g. the DB list call dies)
/// returns `Err`.
///
/// **Phase 2 firewall:** `namespace` tags the seeded entries so callers can
/// distinguish user-authored codex (`"codex"`) from Wupi's non-editable system
/// knowledge (`"wupi_system"`). Both reuse the `kind=codex` discriminator
/// downstream (so the per-class floor + render frame apply automatically); the
/// `namespace` field is for future filtering and the audit log. The two seed
/// paths (user codex from `docs/`, Wupi-system from `cards/wupi_knowledge/`)
/// write to disjoint `card_id` partitions: see `CODEX_CARD_ID` and
/// `WUPI_SYSTEM_CARD_ID` in `memory.rs`.
pub async fn seed_codex(
    engine: &MemoryEngine<impl Embedder>,
    codex_dir: &Path,
    card_id: &str,
    namespace: &str,
) -> anyhow::Result<ReconcileReport> {
    let mut report = ReconcileReport::default();

    // Parse all source files first. A missing dir is not an error: the
    // Codex is optional.
    let sources = match parse_dir(codex_dir) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(
                dir = %codex_dir.display(),
                error = %format!("{e}"),
                namespace,
                "codex dir unreadable or missing; skipping seed"
            );
            return Ok(report);
        }
    };

    if sources.is_empty() {
        tracing::info!(dir = %codex_dir.display(), namespace, "codex dir empty; nothing to seed");
        return Ok(report);
    }

    // Load the existing Codex entries for this card. Keyed by title for the
    // reconcile diff. Each value carries (id, hash): id for delete, hash for
    // change detection.
    let existing = engine.list_codex_entries(card_id).await?;
    let mut existing_by_title: HashMap<String, (MemoryId, Option<String>)> = HashMap::new();
    for (id, metadata_json) in existing {
        let title = extract_metadata_field(metadata_json.as_deref(), "title")
            .unwrap_or_default();
        existing_by_title.insert(title, (id, extract_metadata_field(metadata_json.as_deref(), "hash")));
    }

    // Track which existing titles we consumed, so leftovers = orphans to purge.
    let mut consumed: std::collections::HashSet<&str> = std::collections::HashSet::new();

    for src in &sources {
        let stored_hash = existing_by_title
            .get(&src.title)
            .and_then(|(_, h)| h.clone());
        let stored_hash_u64 = stored_hash.as_deref().and_then(|s| s.parse::<u64>().ok());

        match stored_hash_u64 {
            Some(h) if h == src.hash => {
                // Unchanged: no write.
                report.unchanged += 1;
                consumed.insert(&src.title);
            }
            Some(_) => {
                // Changed: delete old, insert new (re-embed with new text).
                if let Some(&(old_id, _)) = existing_by_title.get(&src.title) {
                    if let Err(e) = engine.delete_memory(old_id).await {
                        tracing::warn!(
                            title = %src.title,
                            error = %format!("{e}"),
                            "codex update: failed to delete old entry; skipping"
                        );
                        continue;
                    }
                }
                match insert_entry(engine, src, card_id, namespace).await {
                    Ok(()) => {
                        report.updated += 1;
                        consumed.insert(&src.title);
                    }
                    Err(e) => {
                        tracing::warn!(title = %src.title, error = %format!("{e}"), "codex update insert failed");
                    }
                }
            }
            None => {
                // New: insert.
                match insert_entry(engine, src, card_id, namespace).await {
                    Ok(()) => {
                        report.seeded += 1;
                        consumed.insert(&src.title);
                    }
                    Err(e) => {
                        tracing::warn!(title = %src.title, error = %format!("{e}"), "codex seed insert failed");
                    }
                }
            }
        }
    }

    // Purge orphans: stored Codex entries whose title wasn't consumed above
    // (their source file is gone).
    for (title, (id, _)) in &existing_by_title {
        if !consumed.contains(title.as_str()) {
            match engine.delete_memory(*id).await {
                Ok(()) => report.purged += 1,
                Err(e) => tracing::warn!(title = %title, error = %format!("{e}"), "codex orphan purge failed"),
            }
        }
    }

    Ok(report)
}

/// Insert one parsed entry via `add_codex_entry`, building its `metadata_json`.
/// Salience is flat 1.0 (matches episodic; salience weighting is deferred per
/// §2N landmine #4). `namespace` flows into the metadata so the entry's origin
/// (user codex vs Wupi-system) is queryable for future filtering.
async fn insert_entry(
    engine: &MemoryEngine<impl Embedder>,
    src: &ParsedEntry,
    card_id: &str,
    namespace: &str,
) -> anyhow::Result<()> {
    // Body-length guard: warn (don't reject) when the body exceeds the
    // ~350-token heuristic budget. The entry still seeds: the operator sees
    // the warning and can split the file.
    const BUDGET_CHARS: usize = 1400;
    if src.body.len() > BUDGET_CHARS {
        tracing::warn!(
            title = %src.title,
            body_chars = src.body.len(),
            budget = BUDGET_CHARS,
            "codex entry exceeds the ~350-token budget; bge-small may truncate the embedding. Split into smaller files."
        );
    }

    let metadata = build_metadata_json(&src.title, &src.tags, src.hash, namespace);
    engine
        .add_codex_entry(src.body.clone(), card_id, 1.0, metadata)
        .await
        .map(|_| ())
}

// The Codex UI treats the `.md` files in docs/ as the source of truth: the
// DB is a derived retrieval index, re-seeded at boot. These functions read and
// write the FILES directly, so edits persist across reboots and stay
// git-trackable. After any mutation the caller re-seeds so retrieval stays in
// sync within the running session.

/// One Codex file as the UI sees it. `filename` is the stem (no `.md`, no
/// path): it's the stable identity of the entry across edits. A rename =
/// delete-old + save-new (the caller's job).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CodexFile {
    /// Stem of the `.md` file (e.g. `neo-kyoto`). The on-disk key.
    pub filename: String,
    pub title: String,
    pub tags: Vec<String>,
    /// The prose body (everything after the front-matter).
    pub body: String,
}

/// List every Codex `.md` file in `dir`, parsed into `CodexFile` rows. Sorted
/// by title for a stable library view. Empty Vec for a missing/empty dir.
pub fn list_files(dir: &Path) -> anyhow::Result<Vec<CodexFile>> {
    let read = match std::fs::read_dir(dir) {
        Ok(r) => r,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(anyhow::anyhow!("read codex dir {}: {e}", dir.display())),
    };
    let mut paths: Vec<std::path::PathBuf> = read
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()).map_or(false, |s| s.eq_ignore_ascii_case("md")))
        .collect();
    paths.sort();

    let mut out = Vec::new();
    for path in paths {
        match parse_file(&path) {
            Ok(entry) => {
                let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("untitled").to_owned();
                out.push(CodexFile {
                    filename: stem,
                    title: entry.title,
                    tags: entry.tags,
                    body: entry.body,
                });
            }
            Err(e) => tracing::warn!(file = %path.display(), error = %format!("{e}"), "codex file parse failed; skipping in list"),
        }
    }
    // Sort by title (case-insensitive) for a clean library order.
    out.sort_by(|a, b| a.title.to_lowercase().cmp(&b.title.to_lowercase()));
    Ok(out)
}

/// Sanitize a filename into a file-system-safe stem: lowercase, replace any
/// non-alphanumeric/`-`/`_` char with `-`, trim leading/trailing `-`. Returns
/// `None` if the result is empty. Public so the IPC layer can echo back the
/// exact stem `save_file` will use (the UI tracks entries by this key).
pub fn sanitize_stem(filename: &str) -> Option<String> {
    let stem: String = filename
        .trim()
        .to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '-' })
        .collect();
    let stem = stem.trim_matches('-').to_owned();
    if stem.is_empty() { None } else { Some(stem) }
}

/// Serialize a Codex entry back to its `.md` form and write it atomically.
/// `filename` is the stem; `.md` is appended. The front-matter is regenerated
/// from title + tags; the body is written verbatim below it. Atomic write
/// (temp + rename) mirrors the operator-profile save pattern. Returns the
/// sanitized stem actually written (for the UI to track).
pub fn save_file(dir: &Path, filename: &str, title: &str, tags: &[String], body: &str) -> anyhow::Result<String> {
    std::fs::create_dir_all(dir).map_err(|e| anyhow::anyhow!("create codex dir: {e:?}"))?;

    let safe_stem = sanitize_stem(filename)
        .ok_or_else(|| anyhow::anyhow!("codex filename empty after sanitization"))?;

    let md = render_md(title, tags, body);
    let target = dir.join(format!("{safe_stem}.md"));
    let tmp = dir.join(format!(".{safe_stem}.md.tmp"));

    {
        use std::io::Write;
        let mut f = std::fs::File::create(&tmp).map_err(|e| anyhow::anyhow!("create codex temp: {e:?}"))?;
        f.write_all(md.as_bytes()).map_err(|e| anyhow::anyhow!("write codex temp: {e:?}"))?;
        f.sync_all().map_err(|e| anyhow::anyhow!("fsync codex temp: {e:?}"))?;
    }
    std::fs::rename(&tmp, &target).map_err(|e| anyhow::anyhow!("rename codex temp → target: {e:?}"))?;
    Ok(safe_stem)
}

/// Delete a Codex `.md` file by stem. Silent no-op if it doesn't exist.
pub fn delete_file(dir: &Path, filename: &str) -> anyhow::Result<()> {
    let path = dir.join(format!("{filename}.md"));
    match std::fs::remove_file(&path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(anyhow::anyhow!("delete codex file {}: {e:?}", path.display())),
    }
}

/// Render a Codex entry to its canonical `.md` form: YAML-ish front-matter
/// (title + tags) then a blank line then the body. Separated from `save_file`
/// so a round-trip test can exercise it without touching disk.
fn render_md(title: &str, tags: &[String], body: &str) -> String {
    let tags_line = tags.join(", ");
    format!(
        "---\ntitle: {title}\ntags: {tags_line}\n---\n\n{body}\n",
        body = body.trim_end(),
    )
}


/// Parse every `.md` file in `dir` (non-recursive). Returns an empty Vec for
/// an empty/missing dir (caller treats as "nothing to seed"). Files are sorted
/// by filename for deterministic seed order.
fn parse_dir(dir: &Path) -> anyhow::Result<Vec<ParsedEntry>> {
    let mut entries = Vec::new();
    let read = std::fs::read_dir(dir).map_err(|e| anyhow::anyhow!("read codex dir {}: {e}", dir.display()))?;

    let mut paths: Vec<std::path::PathBuf> = read
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()).map_or(false, |s| s.eq_ignore_ascii_case("md")))
        .collect();
    paths.sort();

    for path in paths {
        match parse_file(&path) {
            Ok(entry) => entries.push(entry),
            Err(e) => tracing::warn!(file = %path.display(), error = %format!("{e}"), "codex file parse failed; skipping"),
        }
    }
    Ok(entries)
}

/// Parse one `.md` file into a `ParsedEntry`. Reads bytes, computes the hash
/// over the raw bytes (not the parsed fields: so whitespace-only edits to
/// front-matter still register as a change), then splits front-matter from body.
fn parse_file(path: &Path) -> anyhow::Result<ParsedEntry> {
    let bytes = std::fs::read(path).map_err(|e| anyhow::anyhow!("read {}: {e}", path.display()))?;
    let text = String::from_utf8_lossy(&bytes).into_owned();

    let mut hasher = std::hash::DefaultHasher::new();
    bytes.hash(&mut hasher);
    let hash = hasher.finish();

    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("untitled")
        .to_owned();

    let (front, body) = split_front_matter(&text);
    let (title, tags) = parse_front_matter(front, &stem);

    Ok(ParsedEntry {
        title,
        tags,
        body: body.to_owned(),
        hash,
    })
}

/// Split a markdown file into `(front_matter, body)`. Front-matter is the
/// text between leading `---\n` and the next `\n---\n` (or end). If the file
/// doesn't start with `---`, there's no front-matter: the whole thing is body.
fn split_front_matter(text: &str) -> (Option<&str>, &str) {
    let after_opener = text.strip_prefix("---\n").or_else(|| text.strip_prefix("---\r\n"));
    let Some(rest) = after_opener else {
        return (None, text);
    };
    // Find the closing `---` on its own line.
    if let Some(end) = rest.find("\n---\n") {
        let front = &rest[..end];
        let body = &rest[end + "\n---\n".len()..];
        (Some(front), body)
    } else if let Some(end) = rest.find("\n---\r\n") {
        let front = &rest[..end];
        let body = &rest[end + "\n---\r\n".len()..];
        (Some(front), body)
    } else {
        // Opening fence but no closer: treat the whole thing as body (no
        // front-matter). Malformed, but don't lose the content.
        (None, text)
    }
}

/// Parse front-matter text into `(title, tags)`. Hand-rolled: recognizes
/// `title: X` and `tags: a, b, c` lines via `split_once(':')`. Unknown keys
/// are ignored. No YAML engine (Prime Directive §1B.4: compose, don't nest).
fn parse_front_matter(front: Option<&str>, fallback_stem: &str) -> (String, Vec<String>) {
    let front = match front {
        Some(f) => f,
        None => return (fallback_stem.to_owned(), Vec::new()),
    };

    let mut title = None;
    let mut tags = Vec::new();

    for line in front.lines() {
        let line = line.trim();
        if let Some((key, val)) = line.split_once(':') {
            let key = key.trim();
            let val = val.trim();
            match key {
                "title" => {
                    if !val.is_empty() {
                        title = Some(val.to_owned());
                    }
                }
                "tags" => {
                    tags = val
                        .split(',')
                        .map(|t| t.trim().to_owned())
                        .filter(|t| !t.is_empty())
                        .collect();
                }
                _ => {}
            }
        }
    }

    (title.unwrap_or_else(|| fallback_stem.to_owned()), tags)
}

/// Build the `metadata_json` string for a Codex entry. Hand-rolled JSON
/// construction (the structure is fixed and small; a serde round-trip would be
/// overkill). All values are JSON-escaped via `escape_json_string`.
fn build_metadata_json(title: &str, tags: &[String], hash: u64, namespace: &str) -> String {
    let title_escaped = escape_json_string(title);
    let tags_array = tags
        .iter()
        .map(|t| format!("\"{}\"", escape_json_string(t)))
        .collect::<Vec<_>>()
        .join(",");
    // `kind=codex` is the downstream discriminator (is_codex / codex floor /
    // render frame). `namespace` is the origin tag: "codex" for user-authored
    // lore, "wupi_system" for Wupi's non-editable system docs. Both reuse the
    // same retrieval/render pipeline; namespace is for future filtering + audit.
    format!(
        "{{\"kind\":\"codex\",\"namespace\":\"{}\",\"title\":\"{}\",\"tags\":[{}],\"hash\":\"{}\"}}",
        escape_json_string(namespace),
        title_escaped,
        tags_array,
        hash
    )
}

/// Escape a string for safe inclusion inside a JSON string value. Handles the
/// six mandatory JSON escapes. The title/tags are author-controlled and may
/// contain quotes or backslashes.
fn escape_json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Extract a string field's value from a `metadata_json` blob. Shared with
/// `memory::codex_title` in spirit but lives here too (the seed pass needs
/// `title` AND `hash`). Substring probe: finds `"key":"..."` and returns the
/// unescaped value. Returns `None` if the key is absent.
fn extract_metadata_field(metadata_json: Option<&str>, key: &str) -> Option<String> {
    let s = metadata_json?;
    let needle = format!("\"{key}\"");
    let idx = s.find(&needle)?;
    let after_key = &s[idx + needle.len()..];
    let after_colon = after_key.trim_start();
    let after_colon = after_colon.strip_prefix(':')?;
    let after_colon = after_colon.trim_start();
    let value = after_colon.strip_prefix('"')?;
    let mut end = None;
    let mut chars = value.char_indices();
    while let Some((i, c)) = chars.next() {
        if c == '\\' {
            chars.next();
            continue;
        }
        if c == '"' {
            end = Some(i);
            break;
        }
    }
    let raw = &value[..end?];
    Some(raw.replace("\\\"", "\"").replace("\\\\", "\\"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn front_matter_parses_title_and_tags() {
        let md = "---\ntitle: Card Format\ntags: cards, xml, format\n---\nThe body text.";
        let (front, body) = split_front_matter(md);
        let (title, tags) = parse_front_matter(front, "fallback");
        assert_eq!(title, "Card Format");
        assert_eq!(tags, vec!["cards", "xml", "format"]);
        assert_eq!(body, "The body text.");
    }

    #[test]
    fn front_matter_missing_falls_back_to_stem() {
        let md = "Just body, no front-matter.";
        let (front, body) = split_front_matter(md);
        assert!(front.is_none());
        assert_eq!(body, "Just body, no front-matter.");
        let (title, tags) = parse_front_matter(front, "my-file");
        assert_eq!(title, "my-file");
        assert!(tags.is_empty());
    }

    #[test]
    fn front_matter_with_only_title() {
        let md = "---\ntitle: Solo Title\n---\nBody.";
        let (front, body) = split_front_matter(md);
        let (title, tags) = parse_front_matter(front, "x");
        assert_eq!(title, "Solo Title");
        assert!(tags.is_empty());
        assert_eq!(body, "Body.");
    }

    #[test]
    fn front_matter_unclosed_fence_treats_all_as_body() {
        // Opening `---` but no closing fence: don't lose the content.
        let md = "---\ntitle: Broken\nNo closing fence.";
        let (front, body) = split_front_matter(md);
        assert!(front.is_none());
        assert!(body.contains("No closing fence."));
    }

    #[test]
    fn build_metadata_json_round_trips_through_extract() {
        let tags = vec!["a".to_owned(), "b".to_owned()];
        let json = build_metadata_json("My Title", &tags, 12345, "codex");
        assert_eq!(extract_metadata_field(Some(&json), "title"), Some("My Title".to_owned()));
        assert_eq!(extract_metadata_field(Some(&json), "hash"), Some("12345".to_owned()));
        assert!(json.contains("\"kind\":\"codex\""));
        assert!(json.contains("\"namespace\":\"codex\""));
        assert!(json.contains("\"tags\":[\"a\",\"b\"]"));
    }

    #[test]
    fn build_metadata_json_escapes_quotes_in_title() {
        let json = build_metadata_json("He said \"hi\"", &[], 1, "codex");
        assert!(json.contains("\"title\":\"He said \\\"hi\\\"\""));
        assert_eq!(extract_metadata_field(Some(&json), "title"), Some("He said \"hi\"".to_owned()));
    }

    #[test]
    fn build_metadata_json_tags_wupi_system_namespace() {
        // The firewall's distinguishing field: Wupi-system docs carry the same
        // kind=codex (so the floor + render frame apply) but a different
        // namespace (for future filtering / audit).
        let json = build_metadata_json("Critical Wall", &[], 42, "wupi_system");
        assert!(json.contains("\"kind\":\"codex\""));
        assert!(json.contains("\"namespace\":\"wupi_system\""));
        assert_eq!(extract_metadata_field(Some(&json), "namespace"), Some("wupi_system".to_owned()));
    }

    #[test]
    fn hash_is_deterministic_for_identical_bytes() {
        // parse_file hashes raw bytes, so identical files → identical hash.
        // Verified by hashing the same content twice via the hasher directly.
        let bytes = b"hello codex";
        let h1 = {
            let mut h = std::hash::DefaultHasher::new();
            bytes.hash(&mut h);
            h.finish()
        };
        let h2 = {
            let mut h = std::hash::DefaultHasher::new();
            bytes.hash(&mut h);
            h.finish()
        };
        assert_eq!(h1, h2);
    }

    #[test]
    fn hash_differs_when_content_changes() {
        let h1 = {
            let mut h = std::hash::DefaultHasher::new();
            b"version one".hash(&mut h);
            h.finish()
        };
        let h2 = {
            let mut h = std::hash::DefaultHasher::new();
            b"version two".hash(&mut h);
            h.finish()
        };
        assert_ne!(h1, h2);
    }

    #[test]
    fn extract_field_handles_missing_key() {
        let json = r#"{"kind":"codex","title":"x"}"#;
        assert_eq!(extract_metadata_field(Some(json), "hash"), None);
        assert_eq!(extract_metadata_field(None, "title"), None);
    }
}
