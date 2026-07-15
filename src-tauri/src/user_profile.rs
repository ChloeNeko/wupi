//! User Profile (`Operator.xml`) loader, parser, and renderer.
//!
//! The User Profile is the operator's static identity artifact — the "who am I
//! talking to" counterpart to the Simulation Card's "who am I." It lives at
//! `cards/Operator.xml` next to `Wupi.sim`, uses the same strict-XML +
//! CDATA-wrapped prose format, and parses with the same `roxmltree` DOM parser
//! (CDATA auto-merged into text nodes — zero escape handling).
//!
//! Unlike the Simulation Card, the profile is **re-read fresh on every chat
//! turn** rather than cached. This is the hot-reload mechanism: because the
//! profile is a ~1KB file consumed only at the single moment a prompt is
//! assembled (the top of `chat_send`), reading it synchronously each turn is
//! cheaper than a file-watcher thread, zero-staleness, no dependency, and no
//! partial-write races (Prime Directive §1B — the cheapest path that preserves
//! token integrity). The resolved *path* is cached (stable; resolved once in
//! `setup`); only the *content* refreshes.
//!
//! The profile bypasses the Memory engine entirely. It is identity, not
//! episodic recall — it belongs in the stable system-prompt prefix (sibling to
//! `<persona>`), NOT in the inter-turn `<retrieved_memory>` block. Because the
//! rendered text is byte-identical across turns (until the file is edited), it
//! does NOT trigger the §2F cold-reset guard — it's as cache-friendly as the
//! persona.
//!
//! Design contract (mirrors the SIM card's graceful-degradation pattern in
//! §2O): if the file is missing or malformed, `load` returns `None` and the
//! `<user_profile>` section is simply suppressed. A bad or absent profile must
//! never kill the OS — it just means Wupi doesn't know who she's talking to.

use std::path::Path;

/// The parsed operator profile. All four fields are optional in the XML — a
/// field that's absent or empty renders as nothing, and a profile with all
/// four blank renders to an empty string (suppressed downstream by the
/// `Option<&str>` gate, same empty-skip as the SIM card fallback).
#[derive(Debug, Clone, Default)]
pub struct UserProfile {
    /// How Wupi should address the operator (e.g. "Chloe", "Master", "Creator").
    pub name: String,
    /// The operator's function (e.g. "Lead Developer", "Operator").
    pub role: String,
    /// Who the operator is in the context of this world — freeform prose.
    pub background: String,
    /// How Wupi should treat the operator (relationship, tone, dynamics).
    pub dynamics: String,
}

impl UserProfile {
    /// Render the profile into a compact `<user_profile>` block for the system
    /// prompt. Only non-blank fields are emitted; the block is skipped
    /// entirely (empty return) when every field is blank, so the caller's
    /// `Option<&str>` gate suppresses the section cleanly.
    ///
    /// XML-tagged fields match the prompt's existing aesthetic (Prime
    /// Directive §1B.3 — rigid structure exploits instruction-tuned attention).
    /// Ordering is name → role → background → dynamics: identity first, then
    /// the relational framing last so it lands closest to the conversation.
    pub fn render_for_prompt(&self) -> String {
        let mut sections = Vec::new();

        if !self.name.trim().is_empty() {
            sections.push(format!("name: {}", self.name.trim()));
        }
        if !self.role.trim().is_empty() {
            sections.push(format!("role: {}", self.role.trim()));
        }
        if !self.background.trim().is_empty() {
            sections.push(format!(
                "background:\n{}\n",
                indent(self.background.trim())
            ));
        }
        if !self.dynamics.trim().is_empty() {
            sections.push(format!(
                "dynamics:\n{}\n",
                indent(self.dynamics.trim())
            ));
        }

        if sections.is_empty() {
            return String::new();
        }
        format!("<user_profile>\n{}\n</user_profile>", sections.join("\n"))
    }
}

/// Indent every non-empty line of a block by two spaces, mirroring the SIM
/// card's `indent` helper so multi-line prose (background, dynamics) nests
/// cleanly inside its parent field.
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
/// live edits take effect on the very next `chat_send` (no watcher, no cache —
/// see the module docs for why per-turn re-read is the right call).
///
/// - `None` path → `None` (no Operator.xml resolved at startup; Wupi runs
///   without a profile — the common case until the operator authors one).
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

/// Parse a `Operator.xml` profile from its XML text. Separated from `load` so
/// the unit tests exercise the parser without touching the filesystem. Root
/// must be `<user_profile>`; the four child tags are all optional and default
/// to empty strings. CDATA is already merged into `.text()` by roxmltree, so
/// prose with smart quotes or literal angle brackets parses with zero escape
/// handling (same contract as the SIM card).
fn parse(xml: &str) -> anyhow::Result<UserProfile> {
    let doc = roxmltree::Document::parse(xml)
        .map_err(|e| anyhow::anyhow!("parsing operator profile XML: {e}"))?;
    let root = doc
        .root_element()
        .has_tag_name("user_profile")
        .then_some(doc.root_element())
        .ok_or_else(|| anyhow::anyhow!("root element must be <user_profile>"))?;

    Ok(UserProfile {
        name: child_text(root, "name").unwrap_or_default(),
        role: child_text(root, "role").unwrap_or_default(),
        background: child_text(root, "background").unwrap_or_default(),
        dynamics: child_text(root, "dynamics").unwrap_or_default(),
    })
}

// ── XML traversal helpers ──────────────────────────────────────────────────
// Tiny roxmltree wrappers, local to this module. They duplicate the SIM card's
// helpers — intentional: the project style is self-contained modules, and a
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
  <name>Chloe</name>
  <role>Lead Developer</role>
  <background><![CDATA[
- Builds WUPI OS from the ground up.
- Wants an honest engineering partner, not a yes-man.
  ]]></background>
  <dynamics><![CDATA[
- Wupi is intensely devoted to her Master.
- Direct communication is prized over flattery.
  ]]></dynamics>
</user_profile>"#;

    #[test]
    fn parse_extracts_all_fields() {
        let p = parse(SAMPLE).expect("sample parses");
        assert_eq!(p.name, "Chloe");
        assert_eq!(p.role, "Lead Developer");
        assert!(p.background.contains("Builds WUPI OS"));
        assert!(p.dynamics.contains("devoted to her Master"));
    }

    #[test]
    fn parse_missing_fields_default_empty() {
        // A minimal profile with only a name still parses; absent tags → "".
        let minimal = r#"<user_profile>
  <name>Guest</name>
</user_profile>"#;
        let p = parse(minimal).expect("parses");
        assert_eq!(p.name, "Guest");
        assert_eq!(p.role, "");
        assert_eq!(p.background, "");
        assert_eq!(p.dynamics, "");
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
        let broken = "<user_profile><name>Chloe</name"; // truncated mid-tag
        assert!(parse(broken).is_err());
    }

    #[test]
    fn render_emits_tagged_section_with_present_fields() {
        let p = parse(SAMPLE).expect("parses");
        let rendered = p.render_for_prompt();
        assert!(rendered.starts_with("<user_profile>"));
        assert!(rendered.contains("name: Chloe"));
        assert!(rendered.contains("role: Lead Developer"));
        assert!(rendered.contains("background:"));
        assert!(rendered.contains("Builds WUPI OS"));
        assert!(rendered.contains("dynamics:"));
        assert!(rendered.contains("devoted to her Master"));
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
    fn shipped_operator_xml_parses_and_renders() {
        // Integration check against the REAL seed profile shipped in the repo.
        // Guards against hand-edits to cards/Operator.xml breaking the parse
        // (a malformed profile silently suppresses the section at runtime —
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
        // The shipped seed has all four fields populated.
        assert!(p.name.contains("Chloe"));
        assert!(!p.role.is_empty());
        assert!(!p.background.is_empty());
        assert!(!p.dynamics.is_empty());
        // CDATA content (background, dynamics) must survive — if it were
        // escaped instead of CDATA-wrapped, the prose wouldn't parse as text.
        assert!(rendered.contains("background:"));
        assert!(rendered.contains("dynamics:"));
    }
}
