---
title: os-directives-vs-persona
tags: prompts, architecture, system-prompt
---

The system prompt has two cleanly separated layers. OS-level protocol lives in Rust; persona lives in the card. Never mix them.

OS_DIRECTIVES (a Rust const in prompts.rs) is the universal scaffold shared by every Simulation Card. It carries engine-level rules true for ALL cards: the simulation framing, and the semantics of the `<retrieved_memory>` and `<world_state>` tags. These are engineering concerns tied to the Rust architecture — the channel protocol, tag semantics, the rule that memory is past-records-not-authoritative.

The persona is rendered from the active card and injected as `<persona>`. A dungeon card supplies its own persona; the directives above are unchanged.

Why the split: coupling OS-level protocol to a content artifact (.sim file) means every future card author must re-encode engine internals into prose. Persona-specific operational flavor (like Wupi's CRITICAL WALL) stays in the card; a dungeon card may omit it entirely. OS = Rust, persona = card.
