//! User Profile (`Operator.xml`) loader, parser, and renderer.
//!
//! The User Profile is the operator's static identity artifact: the "who am I
//! talking to" counterpart to the Simulation Card's "who am I." It lives at
//! `cards/Operator.xml` next to `Wupi.sim`, uses the same strict-XML +
//! CDATA-wrapped prose format, and parses with the same `roxmltree` DOM parser
//! (CDATA auto-merged into text nodes: zero escape handling).
//!
//! Unlike the Simulation Card, the profile is **re-read fresh on every chat
//! turn** rather than cached. This is the hot-reload mechanism: because the
//! profile is a ~1KB file consumed only at the single moment a prompt is
//! assembled (the top of `chat_send`), reading it synchronously each turn is
//! cheaper than a file-watcher thread, zero-staleness, no dependency, and no
//! partial-write races (Prime Directive §1B: the cheapest path that preserves
//! token integrity). The resolved *path* is cached (stable; resolved once in
//! `setup`); only the *content* refreshes.
//!
//! The profile bypasses the Memory engine entirely. It is identity, not
//! episodic recall: it belongs in the stable system-prompt prefix (sibling to
//! `<persona>`), NOT in the inter-turn `<retrieved_memory>` block. Because the
//! rendered text is byte-identical across turns (until the file is edited), it
//! does NOT trigger the §2F cold-reset guard: it's as cache-friendly as the
//! persona.
//!
//! Design contract (mirrors the SIM card's graceful-degradation pattern in
//! §2O): if the file is missing or malformed, `load` returns `None` and the
//! `<user_profile>` section is simply suppressed. A bad or absent profile must
//! never kill the OS: it just means Wupi doesn't know who she's talking to.

use std::path::Path;

/// The parsed operator profile. Both fields are optional in the XML: a
/// field that's absent or empty renders as nothing, and a profile with both
/// blank renders to an empty string (suppressed downstream by the
/// `Option<&str>` gate, same empty-skip as the SIM card fallback).
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct UserProfile {
    /// How Wupi should address the operator (e.g. "Operator", "Master", "Creator").
    pub name: String,
    /// A freeform character description for Wupi to refer to the operator as -
    /// who they are, how she should treat them, their relationship/tone, etc.
    /// Replaces the old role/background/dynamics split (simplified 2026-07-17).
    pub description: String,
}

impl UserProfile {
    /// Render the profile into a compact `<user_profile>` block for the system
    /// prompt. Only non-blank fields are emitted; the block is skipped
    /// entirely (empty return) when both fields are blank, so the caller's
    /// `Option<&str>` gate suppresses the section cleanly.
    ///
    /// XML-tagged fields match the prompt's existing aesthetic (Prime
    /// Directive §1B.3: rigid structure exploits instruction-tuned attention).
    /// Ordering is name → description: identity first, then the character
    /// framing last so it lands closest to the conversation.
    pub fn render_for_prompt(&self) -> String {
        let mut sections = Vec::new();

        if !self.name.trim().is_empty() {
            sections.push(format!("name: {}", self.name.trim()));
        }
        if !self.description.trim().is_empty() {
            sections.push(format!(
                "description:\n{}\n",
                indent(self.description.trim())
            ));
        }

        if sections.is_empty() {
            return String::new();
        }
        format!("<user_profile>\n{}\n</user_profile>", sections.join("\n"))
    }
}

/// Indent every non-empty line of a block by two spaces, mirroring the SIM
/// card's `indent` helper so multi-line prose (description) nests cleanly
/// inside its parent field.
fn indent(block: &str) -> String {
    block
        .lines()
        .map(|line| {
            if line.trim().is_empty() {
                String::new()
            } else {
                format!("  {}", line)
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The hot-reload entry point. Re-reads + re-parses the file on every call so
/// live edits take effect on the very next `chat_send` (no watcher, no cache -
/// see the module docs for why per-turn re-read is the right call).
///
/// - `None` path → `None` (no Operator.xml resolved at startup; Wupi runs
///   without a profile: the common case until the operator authors one).
/// - `Some(path)` missing or malformed → `None`, debug-logged. Debug (not
///   warn) because this fires every turn when the file is absent, and a warn
///   per turn would spam the log. The startup resolution already logged the
///   one-time "no Operator.xml" notice.
/// - `Some(path)` valid → `Some(UserProfile)`.
///
/// This single function implements hot-reload + graceful degradation in one
/// mechanism: read → parse → `None` on any failure → section suppressed.
pub fn load(path: Option<&Path>) -> Option<UserProfile> {
    let path = path?;
    match std::fs::read_to_string(path).map_err(anyhow::Error::from).and_then(|text| parse(&text)) {
        Ok(profile) => Some(profile),
        Err(e) => {
            tracing::debug!(
                path = %path.display(),
                error = %format!("{e}"),
                "operator profile unreadable; <user_profile> section suppressed this turn"
            );
            None
        }
    }
}

/// Serialize a profile back to `Operator.xml` and write it atomically.
///
/// The inverse of [`load`]: renders both fields as `<user_profile>` XML
/// with every field CDATA-wrapped (so prose containing quotes, angle brackets,
/// or smart quotes round-trips with zero escape handling: same contract the
/// parser already assumes). The on-disk format is byte-stable: an unchanged
/// profile re-saves to byte-identical text, so it stays cache-friendly for the
/// §2F guard.
///
/// **Atomic write** (temp file → fsync → rename over the target) so a crash or
/// power loss mid-write can never truncate `Operator.xml`: mirrors the atomic
/// pattern in `session.rs` (AGENTS.md §2E). The temp lives next to the target
/// (same volume → `rename` is atomic; on Windows it uses
/// `MOVEFILE_REPLACE_EXISTING`).
///
/// Hot-reload is automatic: `chat_send` re-reads the file every turn, so the
/// saved values take effect on the next message with zero extra wiring.
pub fn save(path: &Path, profile: &UserProfile) -> anyhow::Result<()> {
    let xml = render_xml(profile);
    let parent = path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("operator profile path has no parent: {}", path.display()))?;
    std::fs::create_dir_all(parent)
        .map_err(|e| anyhow::anyhow!("create operator profile dir: {e:?}"))?;

    // Write to a sibling temp, fsync, then atomic-rename over the target.
    let mut tmp = std::env::temp_dir();
    tmp.push(format!(
        ".wupi-operator-{}.tmp",
        path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("xml")
    ));
    // Place the temp NEXT TO the target (same volume) so rename is atomic.
    let tmp = parent.join(
        tmp.file_name()
            .ok_or_else(|| anyhow::anyhow!("bad temp file name"))?,
    );

    {
        use std::io::Write;
        let mut f = std::fs::File::create(&tmp)
            .map_err(|e| anyhow::anyhow!("create operator temp: {e:?}"))?;
        f.write_all(xml.as_bytes())
            .map_err(|e| anyhow::anyhow!("write operator temp: {e:?}"))?;
        f.sync_all()
            .map_err(|e| anyhow::anyhow!("fsync operator temp: {e:?}"))?;
    }

    std::fs::rename(&tmp, path)
        .map_err(|e| anyhow::anyhow!("rename operator temp → target: {e:?}"))?;
    Ok(())
}

/// Render the profile to its canonical on-disk XML form. Separated from
/// [`save`] so a round-trip unit test can exercise it without touching disk.
/// Both fields are emitted (even when blank) so the file shape is stable;
/// blank fields render as an empty CDATA section and parse back to an empty
/// string.
fn render_xml(profile: &UserProfile) -> String {
    fn field(tag: &str, value: &str) -> String {
        format!("  <{tag}><![CDATA[{}]]></{tag}>", value)
    }
    format!(
        "<user_profile>\n{}\n{}\n</user_profile>\n",
        field("name", &profile.name),
        field("description", &profile.description),
    )
}

/// Parse a `Operator.xml` profile from its XML text. Separated from `load` so
/// the unit tests exercise the parser without touching the filesystem. Root
/// must be `<user_profile>`; the two child tags are both optional and default
/// to empty strings. CDATA is already merged into `.text()` by roxmltree, so
/// prose with smart quotes or literal angle brackets parses with zero escape
/// handling (same contract as the SIM card).
///
/// **Backward compat (2026-07-17 simplification):** the old 4-field format
/// (name/role/background/dynamics) is silently tolerated: `role` is dropped,
/// and `background` + `dynamics` (if present) are concatenated into
/// `description` so an old Operator.xml isn't lost on first load. The next
/// save rewrites it in the new 2-field shape.
fn parse(xml: &str) -> anyhow::Result<UserProfile> {
    let doc = roxmltree::Document::parse(xml)
        .map_err(|e| anyhow::anyhow!("parsing operator profile XML: {e}"))?;
    let root = doc
        .root_element()
        .has_tag_name("user_profile")
        .then_some(doc.root_element())
        .ok_or_else(|| anyhow::anyhow!("root element must be <user_profile>"))?;

    // New format: name + description. Description is the sole prose field.
    let name = child_text(root, "name").unwrap_or_default();
    let mut description = child_text(root, "description").unwrap_or_default();

    // Backward compat: if the old 4-field tags are present (and the new
    // description is absent), fold background + dynamics into description so
    // the old prose isn't lost. `role` is intentionally dropped: it was the
    // least useful field and the simplification explicitly removes it.
    if description.trim().is_empty() {
        let background = child_text(root, "background").unwrap_or_default();
        let dynamics = child_text(root, "dynamics").unwrap_or_default();
        let mut folded = String::new();
        if !background.trim().is_empty() {
            folded.push_str(background.trim());
        }
        if !dynamics.trim().is_empty() {
            if !folded.is_empty() {
                folded.push_str("\n\n");
            }
            folded.push_str(dynamics.trim());
        }
        if !folded.is_empty() {
            description = folded;
        }
    }

    Ok(UserProfile { name, description })
}

// Tiny roxmltree wrappers, local to this module. They duplicate the SIM card's
// helpers: intentional: the project style is self-contained modules, and a
// shared `xml_util` module would couple two unrelated loaders for ~10 lines of
// savings. CDATA is already merged into `.text()` by roxmltree.

/// Find the first direct child element with the given tag name.
fn first_child<'a, 'input>(
    node: roxmltree::Node<'a, 'input>,
    tag: &str,
) -> Option<roxmltree::Node<'a, 'input>> {
    node.children().find(|c| c.is_element() && c.has_tag_name(tag))
}

/// Text of a direct child element, trimmed. `None` if the child is absent.
fn child_text(node: roxmltree::Node, tag: &str) -> Option<String> {
    first_child(node, tag)
        .map(|n| n.text().unwrap_or("").trim().to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"<?xml version="1.0"?>
<user_profile>
  <name>Operator</name>
  <description><![CDATA[
The operator. Wupi's Master: direct, ambitious, wants an honest engineering
partner, not a yes-man. Build WUPI OS alongside her.
  ]]></description>
</user_profile>"#;

    /// The pre-2026-07-17 4-field shape (name/role/background/dynamics). The
    /// parser must tolerate it: background + dynamics fold into `description`,
    /// `role` is dropped. An old Operator.xml isn't lost on first load.
    const LEGACY_4FIELD: &str = r#"<?xml version="1.0"?>
<user_profile>
  <name>Operator</name>
  <role>Lead Developer</role>
  <background><![CDATA[
- Builds WUPI OS from the ground up.
  ]]></background>
  <dynamics><![CDATA[
- Wupi is intensely devoted to her Master.
  ]]></dynamics>
</user_profile>"#;

    #[test]
    fn parse_extracts_all_fields() {
        let p = parse(SAMPLE).expect("sample parses");
        assert_eq!(p.name, "Operator");
        assert!(p.description.contains("honest engineering"));
    }

    #[test]
    fn parse_legacy_4field_folds_into_description() {
        // The old format still loads: background + dynamics → description.
        let p = parse(LEGACY_4FIELD).expect("legacy parses");
        assert_eq!(p.name, "Operator");
        assert!(p.description.contains("Builds WUPI OS"));
        assert!(p.description.contains("devoted to her Master"));
    }

    #[test]
    fn parse_missing_fields_default_empty() {
        // A minimal profile with only a name still parses; absent tags → "".
        let minimal = r#"<user_profile>
  <name>Guest</name>
</user_profile>"#;
        let p = parse(minimal).expect("parses");
        assert_eq!(p.name, "Guest");
        assert_eq!(p.description, "");
    }

    #[test]
    fn parse_rejects_wrong_root() {
        let bad = "<not_a_profile><name>x</name></not_a_profile>";
        assert!(parse(bad).is_err());
    }

    #[test]
    fn parse_rejects_malformed_xml() {
        // Graceful degradation on malformed XML (the hot-reload "saved mid-edit"
        // case). parse() errors → load() returns None → section suppressed.
        let broken = "<user_profile><name>Operator</name"; // truncated mid-tag
        assert!(parse(broken).is_err());
    }

    #[test]
    fn render_emits_tagged_section_with_present_fields() {
        let p = parse(SAMPLE).expect("parses");
        let rendered = p.render_for_prompt();
        assert!(rendered.starts_with("<user_profile>"));
        assert!(rendered.contains("name: Operator"));
        assert!(rendered.contains("description:"));
        assert!(rendered.contains("honest engineering"));
    }

    #[test]
    fn render_empty_when_all_fields_blank() {
        // The all-blank profile renders to "" → the caller's Option gate
        // suppresses the section entirely (same empty-skip as the SIM fallback).
        let p = UserProfile::default();
        assert_eq!(p.render_for_prompt(), "");
    }

    #[test]
    fn load_returns_none_for_none_path() {
        // No path resolved at startup → None, no panic, no IO.
        assert!(load(None).is_none());
    }

    #[test]
    fn load_returns_none_for_missing_file() {
        // A path that doesn't exist → None (graceful, not an error). This is
        // the hot-reload deletion case: deleting Operator.xml mid-chat silently
        // suppresses the section on the next turn.
        let bogus = Path::new("/this/does/not/exist/Operator.xml");
        assert!(load(Some(bogus)).is_none());
    }

    #[test]
    fn roundtrip_preserves_both_fields() {
        // save → load round-trip via render_xml → parse. Guards the byte-
        // stable contract (an unchanged profile re-saves identically).
        let original = UserProfile {
            name: "Operator".to_owned(),
            description: "Master of Wupi.\nDirect and ambitious.".to_owned(),
        };
        let xml = render_xml(&original);
        let reloaded = parse(&xml).expect("round-trip parses");
        assert_eq!(reloaded.name, original.name);
        assert_eq!(reloaded.description, original.description);
    }

    #[test]
    fn shipped_operator_xml_parses_and_renders() {
        // Integration check against the REAL seed profile shipped in the repo.
        // Guards against hand-edits to cards/Operator.xml breaking the parse
        // (a malformed profile silently suppresses the section at runtime -
        // this test makes a regression visible in CI instead). Locates the file
        // relative to CARGO_MANIFEST_DIR (src-tauri/).
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("cards")
            .join("Operator.xml");
        let xml = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(_) => {
                // File is optional in the format contract; skip rather than
                // fail if the tree doesn't ship it (e.g. a fresh checkout
                // before authoring). The parse logic is covered by other tests.
                eprintln!("Operator.xml not found at {}; skipping", path.display());
                return;
            }
        };
        let p = parse(&xml).expect("shipped Operator.xml must parse cleanly");
        let rendered = p.render_for_prompt();
        assert!(rendered.starts_with("<user_profile>"), "rendered: {rendered}");
        // The shipped seed ships both fields. CDATA content (description) must
        // survive: if it were escaped instead of CDATA-wrapped, the prose
        // wouldn't parse as text.
        assert!(!p.name.trim().is_empty());
        assert!(!p.description.trim().is_empty());
        assert!(rendered.contains("description:"));
    }
}
