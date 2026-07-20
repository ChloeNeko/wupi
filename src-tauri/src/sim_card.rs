//! Simulation Card (`.sim`) loader, parser, and renderer.
//!
//! A Simulation Card is the persona artifact for a WUPI entity: Wupi's
//! own card (the interface persona) or, later, a roleplay scenario card.
//! Each card carries its own identity, appearance, role, conversational style,
//! and an introduction list used for the randomized boot greeting.
//!
//! The card is strict XML with CDATA-wrapped prose blocks (so emoticons,
//! quotes, and any literal `<>` in the persona text parse cleanly). We parse
//! it once at startup with `roxmltree` (a tiny DOM parser that auto-merges
//! CDATA into text nodes: zero special handling), render the persona into a
//! `<persona>` block for the system prompt, and expose a randomized intro for
//! the boot UI flourish.
//!
//! Design contract (mirrors the embedder's graceful-degradation pattern in
//! §2M): if the card file is missing or malformed, `load_or_fallback` returns
//! a minimal stub persona so the app still boots. The persona is best-effort;
//! a bad card must never kill the OS.

use std::path::Path;

use rand::seq::IndexedRandom;

/// One Simulation Card, parsed from a `.sim` file. Owned and immutable for the
/// process lifetime after `setup()` loads it.
#[derive(Debug, Clone)]
pub struct SimCard {
    pub id: String,
    pub name: String,
    /// `"system"` for the OS interface persona (Wupi), `"roleplay"` for
    /// future scenario cards. Drives behavior upstream (e.g. whether the card
    /// owns a resumable session + schema).
    pub card_type: String,
    pub core_persona: String,
    pub traits: String,
    pub appearance: String,
    pub role_instruction: String,
    pub responsibilities: String,
    pub conversational_rules: String,
    pub technical_rules: String,
    /// One greeting string per line in `<introductions>`. Empty if the card
    /// omits the block. Used by [`random_intro`] for the boot flourish.
    pub introductions: Vec<String>,
    // All `None` / empty for the system card (Wupi). A roleplay scenario card
    // carries a `<scenario>` block that populates these. The parser already
    // handles optional elements via `nested_text` returning `None` for absent
    // parents, so adding fields here is non-breaking: `Wupi.sim` parses as
    // before with every field below at its default.
    /// The world/setting premise. Injected into the narrator's system prompt
    /// as the ground-truth scenario context. `None` for system cards.
    pub setting: Option<String>,
    /// Narrative tone directive ("grim, atmospheric, slow-burn"). Guides the
    /// narrator's voice. `None` for system cards.
    pub tone: Option<String>,
    /// Seed text for the first narrator turn (the opening scene). The
    /// GameEngine uses this to prime the first generation if the conversation
    /// is empty. `None` for system cards.
    pub opening_scene: Option<String>,
    /// Stable NPC ids present at scene start. Used by the Phase 2 NPC runtime
    /// to spawn the initial cast. Empty for system cards.
    pub start_npc_ids: Vec<String>,
    /// Activities this card activates (e.g. `["combat","crafting"]`). Phase
    /// 2+ hint: the engine registry will match these against available
    /// activity modules. Empty for system cards.
    pub declared_activities: Vec<String>,
    /// The protagonist's name for roleplay cards (e.g. "Alex", "Kaelen").
    /// Used by the narrator prompt's `<active_reality>` tail block (Phase E,
    /// 2026-07-18) to anchor the model in the current card's identity and
    /// prevent cross-card KV-cache contamination (the "Alex hallucination"
    /// where the cyberpunk narrator used the dungeon protagonist's name).
    /// `None` for system cards; narrator hardening falls back to generic
    /// "the protagonist" phrasing.
    pub protagonist_name: Option<String>,
}

impl SimCard {
    /// Render the persona into a compact `<persona>` block for the system
    /// prompt. Only the identity-shaping fields are rendered: `introductions`
    /// are a UI concern, not model context. Returns an empty `String` for the
    /// minimal fallback (so the caller's `Option<&str>` gate suppresses the
    /// section cleanly when there's no real persona).
    ///
    /// XML-tagged sections match the prompt's existing aesthetic (Prime
    /// Directive §1B.3: rigid structure exploits instruction-tuned attention).
    pub fn render_for_prompt(&self) -> String {
        if self.is_fallback() {
            return String::new();
        }
        let mut sections = Vec::new();

        sections.push(format!(
            "<identity>\nname: {}\ncore_persona: {}\ntraits:\n{}\n</identity>",
            self.name.trim(),
            self.core_persona.trim(),
            indent(self.traits.trim()),
        ));

        if !self.appearance.trim().is_empty() {
            sections.push(format!(
                "<appearance>\n{}\n</appearance>",
                self.appearance.trim()
            ));
        }

        if !self.role_instruction.trim().is_empty() {
            let mut block = format!("<role>\ninstruction: {}\n", self.role_instruction.trim());
            if !self.responsibilities.trim().is_empty() {
                block.push_str(&format!("responsibilities:\n{}\n", indent(self.responsibilities.trim())));
            }
            block.push_str("</role>");
            sections.push(block);
        }

        if !self.conversational_rules.trim().is_empty() {
            sections.push(format!(
                "<conversational_style>\nrules:\n{}\n</conversational_style>",
                indent(self.conversational_rules.trim())
            ));
        }

        if !self.technical_rules.trim().is_empty() {
            sections.push(format!(
                "<technical_protocols>\nrules:\n{}\n</technical_protocols>",
                indent(self.technical_rules.trim())
            ));
        }

        format!("<persona>\n{}\n</persona>", sections.join("\n\n"))
    }

    /// Pick one introduction line at random. Returns `None` if the card has no
    /// introductions (the caller then shows no boot bubble). Called once per
    /// boot via the `get_intro` IPC command.
    pub fn random_intro(&self) -> Option<&str> {
        if self.introductions.is_empty() {
            return None;
        }
        let mut rng = rand::rng();
        self.introductions.choose(&mut rng).map(String::as_str)
    }

    /// The fallback stub has this sentinel id so `render_for_prompt` can detect
    /// it and emit nothing (suppressing the `<persona>` section entirely).
    fn is_fallback(&self) -> bool {
        self.id == FALLBACK_ID
    }
}

/// Indent every non-empty line of a block by two spaces, so list items nest
/// cleanly inside their parent XML section.
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

const FALLBACK_ID: &str = "__wupi_fallback__";

/// Build the minimal fallback card used when the real card file is missing or
/// unparseable. The app still boots; the persona section is simply suppressed
/// (`render_for_prompt` returns empty for the fallback). Loud warning is the
/// caller's job: this fn is silent. Public so `setup()` can reach it directly
/// when no card path resolved at all.
pub fn fallback() -> SimCard {
    SimCard {
        id: FALLBACK_ID.to_owned(),
        name: "Wupi".to_owned(),
        card_type: "system".to_owned(),
        core_persona: String::new(),
        traits: String::new(),
        appearance: String::new(),
        role_instruction: String::new(),
        responsibilities: String::new(),
        conversational_rules: String::new(),
        technical_rules: String::new(),
        introductions: Vec::new(),
        // Roleplay-only fields: all empty for the system-card fallback.
        setting: None,
        tone: None,
        opening_scene: None,
        start_npc_ids: Vec::new(),
        declared_activities: Vec::new(),
        protagonist_name: None,
    }
}

/// Load a `.sim` card from disk, falling back to a minimal stub on any error
/// (missing file, IO error, malformed XML, missing required fields). The
/// persona is best-effort: a bad card must never kill the OS boot. Mirrors
/// the embedder's graceful-degradation contract (§2M).
pub fn load_or_fallback(path: &Path) -> SimCard {
    match try_load(path) {
        Ok(card) => {
            tracing::info!(
                card_path = %path.display(),
                card_id = %card.id,
                card_name = %card.name,
                intros = card.introductions.len(),
                "simulation card loaded"
            );
            card
        }
        Err(e) => {
            tracing::warn!(
                card_path = %path.display(),
                error = %format!("{e}"),
                "simulation card load failed; using minimal fallback (persona section suppressed)"
            );
            fallback()
        }
    }
}

fn try_load(path: &Path) -> anyhow::Result<SimCard> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("reading card file: {e}"))?;
    parse(&text)
}

/// Parse a `.sim` card from its XML text. Separated from `try_load` so the
/// unit tests can exercise the parser without touching the filesystem.
fn parse(xml: &str) -> anyhow::Result<SimCard> {
    let doc = roxmltree::Document::parse(xml)
        .map_err(|e| anyhow::anyhow!("parsing card XML: {e}"))?;
    let root = doc
        .root_element()
        .has_tag_name("sim_card")
        .then_some(doc.root_element())
        .ok_or_else(|| anyhow::anyhow!("root element must be <sim_card>"))?;

    // `id` is OPTIONAL and derived from <identity><name> (lowercased) when
    // <metadata> is absent. The metadata block is NOT part of the card format
    // by design: cards stay clean and persona-only. The id is vestigial today
    // anyway: memory partitioning uses the WUPI_CARD_ID sentinel, not the
    // card's id. Keeping a derived id preserves the field for a future
    // roleplay-card partition path without forcing metadata onto every card.
    let name = first_child(root, "identity")
        .and_then(|n| child_text(n, "name"))
        .unwrap_or_else(|| "unknown".to_owned());
    let id = nested_text(root, "metadata", "id")
        .unwrap_or_else(|| name.to_lowercase());
    let card_type = nested_text(root, "metadata", "type").unwrap_or_else(|| "system".to_owned());

    let identity = first_child(root, "identity");
    let core_persona = identity
        .and_then(|n| child_text(n, "core_persona"))
        .unwrap_or_default();
    let traits = identity
        .and_then(|n| child_text(n, "traits"))
        .unwrap_or_default();

    let appearance = first_child(root, "appearance")
        .map(|n| {
            // Render the whole appearance block as-is: each child element on
            // its own line as `tag: text`, preserving the list-style children
            // (hair, clothing) verbatim.
            let mut lines = Vec::new();
            for child in n.children().filter(|c| c.is_element()) {
                let tag = child.tag_name().name();
                let val = text_content(child);
                if val.trim().is_empty() {
                    lines.push(tag.to_owned());
                } else {
                    lines.push(format!("{tag}: {}", val.trim()));
                }
            }
            lines.join("\n")
        })
        .unwrap_or_default();

    let role = first_child(root, "role");
    let role_instruction = role
        .and_then(|n| child_text(n, "instruction"))
        .unwrap_or_default();
    let responsibilities = role
        .and_then(|n| child_text(n, "responsibilities"))
        .unwrap_or_default();

    let conversational_rules = first_child(root, "conversational_style")
        .and_then(|n| child_text(n, "rules"))
        .unwrap_or_default();

    let technical_rules = first_child(root, "technical_protocols")
        .and_then(|n| child_text(n, "rules"))
        .unwrap_or_default();

    let introductions = first_child(root, "introductions")
        .map(|n| {
            // The block is a CDATA bullet list: one intro per `- ` line.
            // Strip the leading `- ` and trim each. Empty lines drop.
            text_content(n)
                .lines()
                .map(|l| l.trim())
                .filter(|l| !l.is_empty())
                .map(|l| l.strip_prefix("- ").unwrap_or(l).trim().to_owned())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    // All fields optional; absent on system cards (Wupi). `setting`/`tone`/
    // `opening_scene` are nested text children; `start_npcs`/`activities` are
    // CDATA bullet lists parsed the same way as `introductions`. A missing
    // `<scenario>` block leaves every field at its default (None / empty) -
    // `Wupi.sim` parses unchanged.
    let scenario = first_child(root, "scenario");
    let setting = scenario
        .and_then(|n| child_text(n, "setting"))
        .filter(|s| !s.is_empty());
    let tone = scenario
        .and_then(|n| child_text(n, "tone"))
        .filter(|s| !s.is_empty());
    let opening_scene = scenario
        .and_then(|n| child_text(n, "opening_scene"))
        .filter(|s| !s.is_empty());
    let start_npc_ids = scenario
        .and_then(|n| first_child(n, "start_npcs"))
        .map(|n| parse_bullet_list(&text_content(n)))
        .unwrap_or_default();
    let declared_activities = scenario
        .and_then(|n| first_child(n, "activities"))
        .map(|n| parse_bullet_list(&text_content(n)))
        .unwrap_or_default();
    // Protagonist name (Phase E narrator hardening, 2026-07-18). Optional;
    // absent on system cards and on roleplay cards that don't declare one.
    let protagonist_name = scenario
        .and_then(|n| child_text(n, "protagonist"))
        .filter(|s| !s.is_empty());

    Ok(SimCard {
        id,
        name,
        card_type,
        core_persona,
        traits,
        appearance,
        role_instruction,
        responsibilities,
        conversational_rules,
        technical_rules,
        introductions,
        setting,
        tone,
        opening_scene,
        start_npc_ids,
        declared_activities,
        protagonist_name,
    })
}

/// Parse a CDATA bullet list (`- item one\n- item two`) into owned Strings.
/// Shared by `<introductions>`, `<scenario><start_npcs>`, and
/// `<scenario><activities>`. Strips the leading `- ` and trims each line;
/// empty lines drop. Factored out so the three callers don't duplicate the
/// same line-walk.
fn parse_bullet_list(text: &str) -> Vec<String> {
    text.lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .map(|l| l.strip_prefix("- ").unwrap_or(l).trim().to_owned())
        .collect::<Vec<_>>()
}

// roxmltree's API is verbose; these thin wrappers keep the parser readable.
// CDATA is already merged into `.text()` by roxmltree, so `text_content`
// returns the full text of a node regardless of how it was wrapped.

/// The concatenated text of a node (CDATA + plain text children merged).
fn text_content(node: roxmltree::Node) -> String {
    node.text().unwrap_or("").to_owned()
}

/// Find the first direct child element with the given tag name.
fn first_child<'a, 'input>(
    node: roxmltree::Node<'a, 'input>,
    tag: &str,
) -> Option<roxmltree::Node<'a, 'input>> {
    node.children().find(|c| c.is_element() && c.has_tag_name(tag))
}

/// Text of a direct child element, trimmed.
fn child_text(node: roxmltree::Node, tag: &str) -> Option<String> {
    first_child(node, tag).map(text_content).map(|s| s.trim().to_owned())
}

/// Text of `root → <parent> → <child>`. Returns `None` if either step is
/// absent (so optional fields like `metadata/type` degrade cleanly).
fn nested_text(root: roxmltree::Node, parent: &str, child: &str) -> Option<String> {
    first_child(root, parent).and_then(|n| child_text(n, child))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"<?xml version="1.0"?>
<sim_card>
  <metadata>
    <id>wupi</id>
    <name>Wupi</name>
    <type>system</type>
  </metadata>
  <identity>
    <name>Wupi</name>
    <core_persona>A cheerful catgirl.</core_persona>
    <traits><![CDATA[
- Devoted to Master.
- Clumsy but eager.
    ]]></traits>
  </identity>
  <appearance>
    <race>Catgirl</race>
    <cat_ears>Perky and expressive.</cat_ears>
  </appearance>
  <role>
    <instruction>Help Master manage the system.</instruction>
    <responsibilities><![CDATA[
- Chat naturally.
- Manage settings.
    ]]></responsibilities>
  </role>
  <conversational_style>
    <rules><![CDATA[
- Use "nya~".
    ]]></rules>
  </conversational_style>
  <technical_protocols>
    <rules><![CDATA[
- Code must be sterile.
    ]]></rules>
  </technical_protocols>
  <introductions><![CDATA[
- "Hello Master~" (=^-ω-^=)
- "Booted up, nya~" ฅ^>⩊<^ฅ
  ]]></introductions>
</sim_card>"#;

    #[test]
    fn parse_extracts_all_fields() {
        let card = parse(SAMPLE).expect("sample parses");
        assert_eq!(card.id, "wupi");
        assert_eq!(card.name, "Wupi");
        assert_eq!(card.card_type, "system");
        assert_eq!(card.core_persona, "A cheerful catgirl.");
        assert!(card.traits.contains("Devoted to Master."));
        assert!(card.appearance.contains("race: Catgirl"));
        assert!(card.appearance.contains("cat_ears: Perky and expressive."));
        assert_eq!(card.role_instruction, "Help Master manage the system.");
        assert!(card.responsibilities.contains("Manage settings."));
        assert!(card.conversational_rules.contains("nya~"));
        assert!(card.technical_rules.contains("sterile"));
        assert_eq!(card.introductions.len(), 2);
        assert!(card.introductions[0].contains("Hello Master"));
        // The literal `>` in the emoticon survives (the XML/CDATA contract).
        assert!(card.introductions[1].contains("ฅ^>⩊<^ฅ"));
    }

    #[test]
    fn parse_strips_intro_bullet_prefix() {
        let card = parse(SAMPLE).expect("parses");
        // Intros should not carry the leading `- ` marker into the UI text.
        for intro in &card.introductions {
            assert!(!intro.starts_with("- "), "intro kept its bullet: {intro}");
        }
    }

    #[test]
    fn render_for_prompt_emits_tagged_sections() {
        let card = parse(SAMPLE).expect("parses");
        let rendered = card.render_for_prompt();
        assert!(rendered.starts_with("<persona>"));
        assert!(rendered.contains("<identity>"));
        assert!(rendered.contains("name: Wupi"));
        assert!(rendered.contains("<appearance>"));
        assert!(rendered.contains("<role>"));
        assert!(rendered.contains("<conversational_style>"));
        assert!(rendered.contains("<technical_protocols>"));
        // Introductions must NOT leak into the model persona block.
        assert!(!rendered.contains("Hello Master"));
    }

    #[test]
    fn random_intro_returns_none_when_empty() {
        let card = SimCard {
            id: "x".into(),
            name: "x".into(),
            card_type: "system".into(),
            core_persona: String::new(),
            traits: String::new(),
            appearance: String::new(),
            role_instruction: String::new(),
            responsibilities: String::new(),
            conversational_rules: String::new(),
            technical_rules: String::new(),
            introductions: Vec::new(),
            setting: None,
            tone: None,
            opening_scene: None,
            start_npc_ids: Vec::new(),
            declared_activities: Vec::new(),
            protagonist_name: None,
        };
        assert!(card.random_intro().is_none());
    }

    #[test]
    fn random_intro_picks_from_list() {
        let card = parse(SAMPLE).expect("parses");
        let pick = card.random_intro().expect("non-empty list yields a pick");
        assert!(card.introductions.iter().any(|i| i == pick));
    }

    #[test]
    fn fallback_card_renders_empty() {
        // The fallback suppresses the persona section entirely: empty render.
        let card = fallback();
        assert_eq!(card.render_for_prompt(), "");
        assert!(card.random_intro().is_none());
    }

    #[test]
    fn parse_rejects_wrong_root() {
        let bad = "<not_a_sim_card><id>x</id></not_a_sim_card>";
        assert!(parse(bad).is_err());
    }

    #[test]
    fn parse_derives_id_from_name_when_no_metadata() {
        // Metadata is OPTIONAL: a clean, persona-only card (no <metadata>
        // block) must still parse. The id derives from <identity><name>,
        // lowercased. This is the card format going forward.
        let no_meta = r#"<sim_card>
  <identity>
    <name>Wupi</name>
    <core_persona>A catgirl.</core_persona>
  </identity>
</sim_card>"#;
        let card = parse(no_meta).expect("metadata-free card parses");
        assert_eq!(card.name, "Wupi");
        assert_eq!(card.id, "wupi");
        assert_eq!(card.card_type, "system");
    }

    /// A roleplay scenario card (Games app Seam 1). Same strict-XML + CDATA
    /// format as `Wupi.sim`, but with a `<scenario>` block holding setting,
    /// tone, opening_scene, start_npcs, and activities. The system card
    /// (Wupi) omits this block entirely: those fields stay at their default
    /// (None / empty). The dungeon card below is also the §2L-test seed
    /// (the dungeon half of the cross-topic memory rejection test).
    #[test]
    fn parse_roleplay_scenario_block() {
        let roleplay = r#"<?xml version="1.0"?>
<sim_card>
  <metadata>
    <id>dungeon_tavern</id>
    <name>The Rusty Tankard</name>
    <type>roleplay</type>
  </metadata>
  <identity>
    <name>The Rusty Tankard</name>
    <core_persona>A one-shot dungeon scenario.</core_persona>
  </identity>
  <scenario>
    <setting><![CDATA[
A remote frontier tavern at the edge of the Goblinwood. Travellers
shelter here before braving the ruined keep to the north.
    ]]></setting>
    <tone>grim, atmospheric, slow-burn</tone>
    <opening_scene><![CDATA[
Rain lashes the shutters of the Rusty Tankard. The barkeeper polishes
a mug and watches the door. A goblin was seen in the back room an hour
ago. A locked iron chest sits under a table by the hearth.
    ]]></opening_scene>
    <start_npcs><![CDATA[
- barkeeper
- goblin
    ]]></start_npcs>
    <activities><![CDATA[
- combat
    ]]></activities>
    <protagonist>Alex</protagonist>
  </scenario>
</sim_card>"#;
        let card = parse(roleplay).expect("roleplay card parses");
        assert_eq!(card.id, "dungeon_tavern");
        assert_eq!(card.card_type, "roleplay");
        assert!(card.setting.as_deref().unwrap().contains("frontier tavern"));
        assert_eq!(card.tone.as_deref(), Some("grim, atmospheric, slow-burn"));
        assert!(card.opening_scene.as_deref().unwrap().contains("Rain lashes"));
        assert_eq!(card.start_npc_ids, vec!["barkeeper".to_string(), "goblin".to_string()]);
        assert_eq!(card.declared_activities, vec!["combat".to_string()]);
        assert_eq!(card.protagonist_name.as_deref(), Some("Alex"));
    }

    /// The system card (Wupi.sim) has NO `<scenario>` block. Every roleplay
    /// field stays at its default. This guards against the additive fields
    /// accidentally picking up stray values from a system card.
    #[test]
    fn system_card_has_no_scenario_fields() {
        let card = parse(SAMPLE).expect("system card parses");
        assert_eq!(card.card_type, "system");
        assert!(card.setting.is_none());
        assert!(card.tone.is_none());
        assert!(card.opening_scene.is_none());
        assert!(card.start_npc_ids.is_empty());
        assert!(card.declared_activities.is_empty());
        assert!(card.protagonist_name.is_none());
    }
}
