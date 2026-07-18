---
title: sim-card-format
tags: cards, format, sim, xml
---

The Simulation Card (.sim) is the persona artifact for a WUPI OS entity. It is strict XML with a root element `<sim_card>` and CDATA-wrapped prose blocks.

Required structure: an `<identity>` block containing `<name>` and `<core_persona>`. Optional blocks: `<appearance>`, `<role>` (with `<instruction>` + `<responsibilities>`), `<conversational_style>` (with `<rules>`), `<technical_protocols>` (with `<rules>`), and `<introductions>` (a CDATA bullet list of boot greetings).

CDATA is load-bearing: prose carries smart quotes and any literal angle brackets, and CDATA lets the XML parser auto-merge all of it into text nodes with zero escape handling. Never escape characters inside CDATA.

The card id is derived from `<identity><name>` lowercased (e.g. "Wupi" -> "wupi"). There is no `<metadata>` block — cards are persona-only and clean. This convention is locked for all future cards.

The loader degrades gracefully: a missing or malformed card never kills the OS boot; a minimal stub persona is used instead. Never author a card that the parser must fail loudly on — always validate locally before shipping.
