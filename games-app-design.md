# WUPI OS — Games App Design Doc

**Status:** LOCKED (UI shape) — 2026-07-18, **REVISED same day** (Direction 3 pivot)
**Owner:** Chloe (UI/UX direction), implementation per the MVP build plan
**Scope:** The full-screen "Games" application launched from the WUPI OS home grid.

> Companion to AGENTS.md §0 (the "BLOCKED ON UI/UX DESIGN" note). This doc
> satisfies the §0 requirement: *"Until a direction exists, roleplay cards,
> the §2L test, scoped persistence, and floor recalibration are all gated."*
> A direction now exists. The gate is cleared for the MVP build plan.

---

## 1. The Decision (revised 2026-07-18 — Direction 3 pivot)

> **PIVOT FROM THE EARLIER DRAFT:** the Recursive Game-OS direction
> (Direction 2) with data-heavy dashboards is **scrapped.** In its place:
> **pure full-screen immersion** (Direction 3) with **WUPI herself** as the
> management interface via a slide-out drawer overlay. This is a stronger,
> more original thesis — see §1.2 for why.

The Games app is **pure full-screen immersion**: the player never sees a
settings menu, inventory screen, or panel of any kind. The entire screen is
the simulation stage (background + sprites + dialogue + scene FX). **All
management happens through WUPI**, who lives in a sleek slide-out drawer
overlay.

- **Pure immersion stage.** Full-screen background, character sprites, a
  dialogue box, scene FX. No HUD chrome by default. The LLM's bracket
  commands (`[CHARACTER_TURN:name]`, `[OBJECT id= state=]`, `[FX ...]`) drive
  what's visible. Cutscenes are the natural state, not a special mode.
- **WUPI as game manager (the side drawer).** The player slides Wupi out
  (hotkey, edge-swipe, or a hot-spot in the scene) and asks in natural
  language: *"show me my inventory,"* *"make it rain,"* *"make the barkeeper
  suspicious of me,"* *"fast-travel to the dungeon."* Wupi parses the intent,
  executes it as a `SchemaDelta` mutation on the game's scoped world-state in
  the background, confirms in natural language, and slides away. **She is the
  OS operator applied to the game** — the same role she plays for WUPI OS
  itself.

### 1.1 The dual-context requirement (load-bearing)

Because Wupi must be available *while the game is running* — without
interrupting the narrator — the **chat context (Wupi-assistant) and the
game context (Narrator) MUST coexist** as two live `LlamaContext`s on the
same leaked `&'static LlamaModel`. The `schema_engine.rs` isolated-2nd-
context pattern (§2J) is the reference architecture. On Chloe's 12GB GPU:

```
WUPI.gguf weights (shared, leaked &'static)   ~9.8 GB
Chat context    (Q8_0 KV, n_ctx=4000)          ~75 MB   (Wupi-assistant — existing)
Schema context  (Q8_0 KV, n_ctx=2048)          ~75 MB   (state-delta summarizer — existing)
Game context    (Q8_0 KV, n_ctx=4000)          ~75 MB   (Narrator — NEW)
Embedder context (n_ctx=512)                   ~10 MB   (BERT — existing)
                                                ─────────
                                                ~10.0 GB → ~2 GB headroom on 12GB ✓
```

Comfortable. **The dual-context setup is the architectural foundation of
this design** — it's what makes "Wupi manages the game without leaving it"
possible. The GameEngine loads its OWN model path (mirroring
`schema_engine.rs::spawn_load`, which loads its own path independent of
`shared_model()`); the four contexts share weights + `shared_backend()`
(§2H) but own their KV state independently.

### 1.2 Why this honors the WUPI OS thesis (better than Direction 2 did)

AGENTS.md §1: *"'P' is what everything else relies on. Without it the other
engines wouldn't work at all — she is the control plane / management layer,
not a peer module."*

Direction 2 made the game a *peer* OS alongside WUPI OS — duplicating Wupi's
management role inside the game shell. Direction 3 makes Wupi the *control
plane* of the game itself — **consistent with her role in the rest of the
OS.** The game doesn't have its own panels because **Wupi IS the panel
layer**, the same way she is for the OS. There is one catgirl, one
management interface, one natural-language command channel.

### 1.3 Rejected alternatives (for the record)

- **Direction 1 (UIE-orthodox "Stage + Command Drawer"):** feels like a web
  app ported into a window, not a native OS. Rejected.
- **Direction 2 (Recursive Game-OS with dashboards):** scrapped in the
  pivot. Data-heavy dashboards conflict with the immersion thesis and
  duplicate Wupi's role as management layer. The full-screen + Wupi-drawer
  model is strictly stronger.
- **Direction 4 (Contextual Rail):** safe but uncommitted. The pure-
  immersion + Wupi model is more original.

### 1.4 Wupi as game manager — the command path

When a game is active (`GameEngine.is_some()`), Wupi's chat context gains a
**second capability**: natural-language game-state mutation. The intent
detector in `chat_send` routes management requests to a
`game_command::interpret(text, &game_schema) -> GameCommand` path that
emits the same `SchemaDelta` type the schema engine produces. Three MVP
variants: `MutateWorldState(delta)`, `QueryWorldState(what)`,
`NotACommand` (falls through to normal Wupi chat).

Wupi's system prompt gains a "game manager" addendum when a game is active:
she can read and mutate the active game's `<world_state>`. She stays the OS
catgirl; she just additionally has authority over the game world. See the
MVP build plan Phase E for the contract.

---

## 2. The Engine Seams

Each seam is a Rust module + (where needed) UI surface + IPC contract.
**Under Direction 3, most seams have NO dedicated UI** — they're surfaced
to the player through Wupi's natural-language command path instead of
panels. **Cards declare which seams they activate** — a high-fantasy card
enables combat+crafting; a modern life-sim enables economy+phone; a
mystery card enables neither.

| # | Seam | MVP? | Rust module (planned) | Surfaced via |
|---|------|:----:|----------------------|---------------|
| 1 | **Scenario Card Lifecycle** | 🔴 YES | extend `sim_card.rs` + new registry | Card picker UI (only panel that survives Direction 3) |
| 2 | **Game Turn Loop** (`game_send`) | 🔴 YES | new `game_engine.rs` | The stage itself — streaming dialogue render |
| 3 | **Narrator-Agency Protocol** | 🔴 YES | new `narrator_prompt.rs`, extend `stream_filter.rs` | Bracket commands → scene_event Channel messages |
| + | **Wupi game-manager path** | 🔴 YES (pivot adds it) | new `game_command.rs` | Wupi's drawer — natural language |
| 4 | **Tick System + Consequence Queue** | 🟢 P2 | new `game_clock.rs`, `consequence.rs` | Wupi narrates outcomes; no clock HUD |
| 5 | **NPC Runtime (behavior sidecar)** | 🟢 P2 | new `npc_runtime.rs`, `npc_secret.rs` | Sprites + expressions + Wupi's "relationship" queries |
| 6 | **World State (map/spatial/travel)** | 🟢 P2 | new `world_state.rs` | Wupi's "travel to..." commands; no map panel |
| 7 | **Player State (you/inventory/party)** | 🟢 P2 | new `player_state.rs` | Wupi's "show my inventory" queries; no inventory panel |
| 8 | **Activity Engines (combat/econ/craft/minigames)** | 🟢 P2+ | per-activity | Stage-driven + Wupi commands; no shop/trade panels |
| 9 | **Presentation Layer (sprites/FX)** | 🟢 DEFERRED-UI | pure TS/CSS | The stage (when UI design lands) |
| 10 | **Asset Pipeline (image gen / TTS)** | 🟢 DEFERRED | TBD | Portrait slots, bg loader, voice preview |

### MVP scope (first playable)

Seams **1 + 2 + 3 + the Wupi command path (+)**. Backend-complete for the
first playable: load a dungeon-fantasy card → start a game → take a turn →
the Narrator responds with prose + bracket commands → the game-state
schema delta fires after the turn → meanwhile the player can ask Wupi (via
her drawer) to mutate game state ("make it stormy") and she executes it as
a `SchemaDelta`. **No sprites, no FX rendering, no panels** — those land
when the UI design does.

### What the MVP is NOT

- **No UI work** (the Games home tile stays inert; full-screen UI is the
  separate post-design implementation pass).
- **No NPCs that "feel alive" yet** (Seam 5 is Phase 2).
- **No map, travel, or world simulation** (Seam 6 is Phase 2).
- **No inventory, stats, or party** (Seam 7 is Phase 2).
- **No combat, economy, or crafting** (Seam 8 is Phase 2+, card-declared).
- **No image generation or TTS** (Seam 10 is deferred).

**The MVP still delivers the §0 unblocks:** roleplay cards live (Seam 1) →
§2L cross-topic test runnable, scoped persistence shippable, floor
recalibration on real roleplay data possible.

---

## 3. Architectural Constraints (what the code forces)

These are the load-bearing decisions the MVP plan must respect. Each is a
constraint from existing WUPI architecture, not a choice.

### 3.1 A 4th `LlamaContext` — DECIDED: dedicated, shared weights (Direction 3)

The Games app gets its OWN `LlamaContext<'static>` for roleplay turns —
sibling to chat (4000) + embedder (512) + schema (2048). The decision is
**dedicated game context with shared weights**, mirroring the schema
engine's isolated-2nd-context pattern (§2J). This is **load-bearing for
Direction 3**: Wupi-as-game-manager requires the chat context AND the game
context alive simultaneously (see §1.1). A shared-context swap would
require exiting the game to talk to Wupi — defeating the thesis.

All four contexts share one `LlamaBackend` via `shared_backend()` (§2H) and
one leaked `&'static LlamaModel` (one ~9.8 GB VRAM copy). The GameEngine
loads its own model PATH (mirroring `schema_engine.rs::spawn_load`, which
loads its own path independent of `shared_model()`) — same file, freshly
leaked ref, independent KV state. The GameEngine is **NOT eager-spawned at
boot** (unlike the schema engine) — it spawns on `game_start` and shuts
down on `game_end`, costing VRAM only while a game is actually running.

VRAM verification: ~10.0 GB total → ~2 GB headroom on 12GB. If live
measurement shows it's tight, the fallback is reducing the game's `n_ctx`
to 3072 (saves ~20MB KV).

### 3.2 Card-switch invalidates the KV cache

Switching roleplay cards changes the system prompt entirely (different
persona, different world). The structural-divergence guard (§2F) will
cold-reset on every card switch by definition. **This is correct and
expected** — design for it. Card switches are infrequent; the cold-reset
cost (~1s) is acceptable.

### 3.3 Per-card persistence (re-introduces §2K scoped saves)

§2K made sessions + schema ephemeral by default. Roleplay cards re-introduce
**scoped persistence**: each card owns its own session + schema, resumable
across launches. The atomic save/load machinery in `session.rs` + `schema.rs`
is retained `#[allow(dead_code)]` for exactly this — don't rebuild it.
Per-card files: `sessions/<card_id>.json` + `schemas/<card_id>.json` in app
data dir. Wupi's own chat stays ephemeral (the §2K thesis holds for the
system card).

### 3.4 Memory partition is already wired

`active_card_id` is already threaded through `chat_send` +
`debug_memory_query` (§2M). Retrieval scopes to the active card via
`AND rowid IN (SELECT id FROM memories WHERE card_id = ?)`. The Games app
sets `active_card_id` to the loaded roleplay card's id; memory partition is
free. **But see §2N landmine #1:** the vec0 `rowid IN (...)` multi-card
recall trap is still latent and only exercisable once a second card exists.
The §2L test will surface it.

### 3.5 The `.sim` format is reusable with no schema change

`SimCard.card_type: String` already has the doc: *"system for the OS
interface persona (Wupi), roleplay for future scenario cards. Drives
behavior upstream (e.g. whether the card owns a resumable session +
schema)"* (`sim_card.rs:30-33`). A roleplay card is the same strict-XML +
CDATA format as `cards/Wupi.sim`, just with `card_type="roleplay"` and
additional scenario metadata (setting, tone, starting scene, declared
activities). The loader/renderer infrastructure from §2O is reusable as-is.

### 3.6 The narrator protocol must enforce `{{forbidden_character_voices}}`

UIE's `omniscientEngine.js:8-28` is the reference. The narrator system prompt
forbids speaking named NPCs' dialogue; instead the LLM emits
`[CHARACTER_TURN:name]` handoffs and `[OBJECT id= state=]` tags. The
StreamFilter (`stream_filter.rs`) extends to parse these into UI events.
**This is the cheap LLM-discipline fix that prevents voice-stealing** and
should bake into the MVP system prompt from day one.

---

## 4. Reference: UIE Source Material

For implementation reference (NOT for direct porting — see "What we already
do better" in the deep-dive notes):

| Concept | UIE reference file |
|---|---|
| Deterministic NPC behavior sidecar | `app/services/character_runtime_service.py:47` |
| State packs (NPC archetypes) | `app/services/state_pack_service.py:46` |
| Behavior scoring + inertia + contradictions | `app/services/behavior_service.py:30-203` |
| Renderer payload (`SHOW_THROUGH_MAP`/`AVOID_ALWAYS`) | `app/services/compiler_service.py:9,51,61` |
| Secrets engine four-channel context | `src/modules/secretsEngine.js:378-466` |
| Narrator agency protocol | `src/modules/omniscientEngine.js:8-28` |
| Delayed/probabilistic consequences | `src/modules/consequenceEngine.js:34,225` |
| Cross-domain ripple rules | `src/modules/causalityEngine.js:93` |
| 3-pillar hybrid combat | `src/modules/combatEngine.js:55-92,313-332` |
| Scaled design-canvas panels | `src/modules/modalViewport.js:1-36` |
| Layered character compositing | `src/css/vnCharacterEngine.css:46-66`, `src/modules/vnCharacterEngine.js:66-104` |
| Declarative scene-FX layer | `src/modules/sceneEffects.js` |
| Parametric CSS-var animation families | `src/styles/layeredAnimations.css` |
| Spatial hierarchy + forbidden-regex | `src/modules/mapTemplate.js:55-112,118-130` |
| Travel-time BFS | `src/modules/timeProgress.js:94-130` |
| Schedule deviation from wants/needs | `src/modules/schedules.js:111-125` |
| Gossip distortion propagation | `src/modules/gossipNetwork.js:86-162` |

**Honest port-worth:** the behavior sidecar philosophy (Seam 5), consequence
queue (Seam 4), secrets four-channel (Seam 5), narrator protocol (Seam 3),
and bracket-command protocol (Seam 3) are the genuinely worth-stealing
ideas. WUPI's retrieval, typed state-deltas, and local LLM are strictly
ahead of UIE's equivalents — do not port those.

---

## 5. Cross-References to AGENTS.md

- §0 — the BLOCKED note this doc clears
- §1 — the WUPI OS thesis this design honors
- §2F — the cold-reset tax (accepted for card switches)
- §2J — the schema engine (reused per-card)
- §2K — ephemeral sessions (re-introduced scoped for roleplay cards)
- §2L — the memory retrieval test this MVP unblocks
- §2M — per-card memory partition (already wired)
- §2N — landmines for cards (especially #1, the vec0 recall trap)
- §2O — the SIM card system (reusable for roleplay cards)
- §2X — the AI panel redesign (separate concern; Games gets its own engine)
