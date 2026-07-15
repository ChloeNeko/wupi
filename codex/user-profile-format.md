---
title: user-profile-format
tags: cards, format, operator, xml, profile
---

The User Profile (`cards/Operator.xml`) is the operator's static identity artifact — the "who am I talking to" counterpart to the Simulation Card. Same strict-XML + CDATA format as `.sim`, same `roxmltree` parser.

Structure: root `<user_profile>`. Four optional snake_case child tags: `<name>` (how Wupi addresses the operator), `<role>` (function), `<background>` (freeform prose — who they are), `<dynamics>` (how Wupi should treat them). CDATA wraps prose so smart quotes and angle brackets parse with zero escape handling.

Hot-reload: the path is resolved once at startup (cached), but the content is re-read fresh on every chat turn. Edit the file mid-conversation and the change lands on the next message — no reboot, no watcher thread. Synchronous re-read of a ~1KB file is cheaper and simpler than a watcher (Prime Directive).

The profile bypasses Memory entirely. It is identity, not recall — injected into the stable system-prompt prefix as `<user_profile>` (sibling to `<persona>`), never into the inter-turn `<retrieved_memory>` block. Byte-identical across turns until edited, so it never triggers the cache cold-reset.

Graceful degradation: missing, absent, or malformed → `None` → section silently suppressed. Wupi runs without knowing who she's talking to, never a crash.
