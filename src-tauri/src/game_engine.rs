//! The GameEngine — the Narrator's dedicated generation thread (Games app Seam 2).
//!
//! A dedicated `std::thread` ("wupi-game") owning an ISOLATED
//! `LlamaContext<'static>` on the same `WUPI.gguf` model the chat engine
//! uses. The narrator's roleplay turns run here, fully isolated from the
//! Wupi-assistant chat context (and from the schema/embedder contexts).
//!
//! # Why a fourth context (the load-bearing isolation)
//!
//! The Games app design (docs/games-app-design.md §1.1) is built on
//! DUAL-CONTEXT: Wupi-as-game-manager (her chat context) must be available
//! *while* the Narrator is mid-scene. The two cannot share a context — they
//! are different personas with different system prompts and different KV
//! state. A fourth isolated `LlamaContext` on the same leaked model
//! accomplishes this. Mirrors the schema engine pattern (§2J) and the
//! embedder pattern (§3B). Four contexts now coexist:
//! **chat (4000) + embedder (512) + schema (2048) + game (4000)** — all
//! sharing one leaked `&'static LlamaModel` + one `shared_backend()`.
//! Total VRAM ~10GB → ~2GB headroom on 12GB (verified by design doc §3.1).
//!
//! # Streaming, not one-shot
//!
//! Unlike the schema engine (which returns a single JSON blob), the game
//! engine streams tokens to a Tauri Channel via the same `ChunkFn` callback
//! type the chat engine uses (`llm::ChunkFn`). The caller (`game_send` IPC)
//! wraps the Channel's `send` into the chunk callback, the same way
//! `chat_send` does. Bracket commands (`[CHARACTER_TURN:...]`, `[OBJECT ...]`,
//! `[FX ...]`) ride alongside prose as `type: "scene_event"` Channel messages
//! (parsed by the `BracketCommand` extractor in `stream_filter.rs`).
//!
//! # Lifecycle
//!
//! NOT eager-spawned at boot. Spawns on `game_start` (when the user picks a
//! roleplay card), shuts down on `game_end`. Costs VRAM only while a game is
//! actually running. Mirrors `SchemaEngine`'s handle shape (`mpsc::Sender` +
//! `Mutex<Option<JoinHandle>>` + `unsafe impl Send+Sync`).

use std::path::{Path, PathBuf};
use std::sync::mpsc;

use llama_cpp_2::context::params::{KvCacheType, LlamaContextParams};
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;

use crate::llm::{shared_backend, shared_model, CancelToken, ChunkFn};

/// The game context's token budget. Matches the chat context (4000) — the
/// narrator's turns are the same shape as chat turns (system + history + new
/// turn) and need the same headroom for long roleplay exchanges.
const GAME_CTX: u32 = 4000;
const GAME_BATCH: u32 = 512;
/// Cap on generated tokens for a single narrator turn. 1024 is generous for
/// a narrative beat (2-4 paragraphs); the narrator system prompt tells the
/// model to keep prose tight. The clamp (engine.rs pattern) further bounds
/// this by `n_ctx - n_cur` at decode time.
const GAME_MAX_TOKENS: i32 = 1024;

// ---------------------------------------------------------------------------
// Control plane — channel types
// ---------------------------------------------------------------------------

/// A request to the game thread: stream a narrator turn for `prompt`.
struct GameRequest {
    /// Fully-rendered prompt (system + visible history + new user turn +
    /// generation prompt). The engine tokenizes + prefills + decodes it.
    prompt: String,
    /// Streaming callback — invoked once per decoded token piece. Wraps the
    /// Tauri Channel's `send` (mirrors `chat_send`'s `on_chunk`).
    on_chunk: ChunkFn,
    /// Per-request cancellation token. The decode loop checks
    /// `cancel.load(Relaxed)` between tokens (same pattern as the chat
    /// engine, §2C). Distinct slot from `active_cancel` so game/chat cancels
    /// never cross-wire.
    cancel: CancelToken,
    /// One-shot reply channel. Sent exactly once when the turn completes
    /// (success, cancel, or error).
    reply: mpsc::Sender<GameReply>,
}

/// What the game thread sends back when a narrator turn completes. Carries
/// the full cleaned text + raw model output + any bracket commands the
/// parser extracted. On error, `error` is populated and the others are empty.
#[derive(Debug, Clone, serde::Serialize)]
pub struct GameReply {
    /// The verbatim model output (post generation, pre-cleanup). Empty on
    /// generation failure.
    pub raw_output: String,
    /// Human-readable error if the turn failed. Empty on success.
    pub error: String,
    /// True if the turn was cancelled mid-generation (`game_stop`). The
    /// caller decides whether to persist a partial reply.
    pub cancelled: bool,
}

enum GameMsg {
    Request(Box<GameRequest>),
    Shutdown,
}

// ---------------------------------------------------------------------------
// Handle (held by callers; fully Send + Sync)
// ---------------------------------------------------------------------------

/// The handle callers hold. Fully `Send + Sync` — a channel sender + the
/// thread's JoinHandle so `shutdown()` can block until VRAM is actually freed
/// (same load-bearing concern as `SchemaEngine`: the next `game_start` must
/// not race the previous `game_end`'s VRAM teardown). Mirrors `SchemaEngine`
/// and `LlamaCppEmbedder`.
pub struct GameEngine {
    tx: mpsc::Sender<GameMsg>,
    join: std::sync::Mutex<Option<std::thread::JoinHandle<()>>>,
}

// SAFETY: mpsc::Sender<GameMsg> is Send (GameMsg owns only Send data).
// Mutex<Option<JoinHandle<()>>> is Send+Sync. No `LlamaContext` crosses out.
unsafe impl Send for GameEngine {}
unsafe impl Sync for GameEngine {}

impl GameEngine {
    /// Spawn the game thread. Loads `WUPI.gguf` (or whatever path resolves)
    /// as this engine's OWN model — freshly leaked `&'static`, independent
    /// KV state. The readiness receiver yields `Ok(())` once the context is
    /// live (or `Err` if init failed — the caller should treat the engine as
    /// unavailable, same contract as `SchemaEngine::spawn_load`).
    pub fn spawn_load(
        path: PathBuf,
        n_gpu_layers: u32,
    ) -> (Self, mpsc::Receiver<Result<(), String>>) {
        let (tx, rx) = mpsc::channel::<GameMsg>();
        let (init_tx, init_rx) = mpsc::channel::<Result<(), String>>();

        let builder = std::thread::Builder::new().name("wupi-game".into());
        let join = builder
            .spawn(move || {
                let mut runtime = match Self::init_runtime(&path, n_gpu_layers) {
                    Ok(rt) => {
                        let _ = init_tx.send(Ok(()));
                        rt
                    }
                    Err(e) => {
                        let msg = format!("game engine init failed: {e}");
                        tracing::error!(error = %msg, "game engine init failed; thread exiting");
                        let _ = init_tx.send(Err(msg.clone()));
                        Self::drain_failed(&rx, msg);
                        return;
                    }
                };
                tracing::info!("wupi-game thread ready");

                loop {
                    match rx.recv() {
                        Ok(GameMsg::Request(req)) => {
                            // Self-healing: isolate each turn so one panic
                            // doesn't kill the thread.
                            let outcome = std::panic::catch_unwind(
                                std::panic::AssertUnwindSafe(|| {
                                    runtime.generate_turn(&req)
                                }),
                            );
                            let reply_msg = match outcome {
                                Ok(Ok(raw)) => GameReply {
                                    raw_output: raw,
                                    error: String::new(),
                                    cancelled: false,
                                },
                                Ok(Err(GenerationOutcome::Cancelled(raw))) => GameReply {
                                    raw_output: raw,
                                    error: String::new(),
                                    cancelled: true,
                                },
                                Ok(Err(GenerationOutcome::GenerationErr(e))) => {
                                    tracing::warn!(error = %format!("{e:#}"), "game turn failed");
                                    runtime.ctx.clear_kv_cache();
                                    GameReply {
                                        raw_output: String::new(),
                                        error: format!("{e:#}"),
                                        cancelled: false,
                                    }
                                }
                                Err(payload) => {
                                    let msg = payload
                                        .downcast_ref::<String>()
                                        .map(|s| s.clone())
                                        .or_else(|| {
                                            payload.downcast_ref::<&str>().map(|s| s.to_string())
                                        })
                                        .unwrap_or_else(|| {
                                            "game turn panic (unknown cause)".to_string()
                                        });
                                    tracing::error!(panic = %msg, "game turn panicked");
                                    runtime.ctx.clear_kv_cache();
                                    GameReply {
                                        raw_output: String::new(),
                                        error: format!("game panic: {msg}"),
                                        cancelled: false,
                                    }
                                }
                            };
                            let _ = req.reply.send(reply_msg);
                        }
                        Ok(GameMsg::Shutdown) => {
                            tracing::info!("wupi-game shutting down");
                            break;
                        }
                        Err(mpsc::RecvError) => {
                            tracing::info!("wupi-game: all senders dropped, exiting");
                            break;
                        }
                    }
                }
            })
            .expect("failed to spawn wupi-game thread");

        (
            GameEngine {
                tx,
                join: std::sync::Mutex::new(Some(join)),
            },
            init_rx,
        )
    }

    /// Shut down the game thread and block until VRAM is freed. Same
    /// load-bearing concern as `SchemaEngine::shutdown` — required so the
    /// next `game_start` doesn't race the teardown.
    pub fn shutdown(&self) {
        let _ = self.tx.send(GameMsg::Shutdown);
        if let Ok(mut guard) = self.join.lock() {
            if let Some(handle) = guard.take() {
                if let Err(e) = handle.join() {
                    tracing::warn!(error = ?e, "wupi-game thread join failed during shutdown");
                }
            }
        }
    }

    /// Post a narrator turn request. The caller awaits the reply via the
    /// receiver it created. The streaming chunks arrive via `on_chunk` *as
    /// they decode* — the reply comes once when generation completes.
    pub fn request_turn(
        &self,
        prompt: String,
        on_chunk: ChunkFn,
        cancel: CancelToken,
    ) -> anyhow::Result<mpsc::Receiver<GameReply>> {
        let (reply_tx, reply_rx) = mpsc::channel::<GameReply>();
        let req = GameRequest {
            prompt,
            on_chunk,
            cancel,
            reply: reply_tx,
        };
        self.tx
            .send(GameMsg::Request(Box::new(req)))
            .map_err(|_| anyhow::anyhow!("game engine thread closed"))?;
        Ok(reply_rx)
    }

    /// Drain any queued requests after a failed init so callers don't block
    /// forever waiting on a reply from a dead thread. Mirrors
    /// `SchemaEngine::drain_failed`.
    fn drain_failed(rx: &mpsc::Receiver<GameMsg>, why: String) {
        while let Ok(msg) = rx.recv_timeout(std::time::Duration::from_millis(50)) {
            if let GameMsg::Request(req) = msg {
                let _ = req.reply.send(GameReply {
                    raw_output: String::new(),
                    error: why.clone(),
                    cancelled: false,
                });
            }
        }
    }

    /// Initialize the game runtime. Prefers the chat engine's already-loaded
    /// `&'static LlamaModel` via `shared_model()` — sharing weights is the
    /// ONLY way four contexts (chat 4000 + embedder 512 + schema 2048 +
    /// game 4000) fit on a 12GB GPU. Loading a second 12B copy would OOM
    /// (the 2026-07-18 `NullResult` lesson). The `path` arg is kept for
    /// forward-compat (a future dedicated narrator model); it's only used
    /// if `shared_model()` returns `None`.
    fn init_runtime(path: &Path, n_gpu_layers: u32) -> anyhow::Result<GameRuntime> {
        let backend = shared_backend();

        // Prefer the shared model (the load-bearing path — avoids VRAM OOM).
        // Only load a separate copy if there's no shared model to reuse
        // (e.g. API mode where the chat engine's local model is torn down).
        let model_ref: &'static LlamaModel = match shared_model() {
            Some(m) => {
                tracing::info!("game engine reusing shared chat model (VRAM-efficient)");
                m
            }
            None => {
                tracing::warn!(
                    "no shared model available; game engine loading its own copy (may OOM)"
                );
                let params = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers);
                let model = LlamaModel::load_from_file(backend, path, &params)
                    .map_err(|e| anyhow::anyhow!("game model load {}: {e:?}", path.display()))?;
                tracing::info!(path = %path.display(), "game model loaded (own copy)");
                Box::leak(Box::new(model))
            }
        };

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(GAME_CTX))
            .with_n_batch(GAME_BATCH)
            .with_embeddings(false)
            // Match the chat engine's KV quantization exactly — the narrator
            // context is the same shape as a chat context.
            .with_type_k(KvCacheType::Q8_0)
            .with_type_v(KvCacheType::Q8_0);
        let ctx = model_ref
            .new_context(backend, ctx_params)
            .map_err(|e| anyhow::anyhow!("game context init: {e:?}"))?;
        tracing::info!(n_ctx = GAME_CTX, "game context created (isolated)");

        Ok(GameRuntime { ctx, model: model_ref })
    }
}

/// Distinguishes a mid-generation cancel from a real error so the reply can
/// set `cancelled: true` appropriately.
enum GenerationOutcome {
    Cancelled(String),
    GenerationErr(anyhow::Error),
}

// ---------------------------------------------------------------------------
// Runtime (owned by the game thread; never crosses thread boundaries)
// ---------------------------------------------------------------------------

struct GameRuntime {
    ctx: llama_cpp_2::context::LlamaContext<'static>,
    model: &'static LlamaModel,
}

impl GameRuntime {
    /// Generate one narrator turn: tokenize → prefill → sample-and-decode,
    /// streaming chunks via `req.on_chunk`. Checks `req.cancel` between
    /// tokens (Relaxed ordering, same correctness argument as the chat
    /// engine, §2B). Returns the full raw model output (Gemma4 channel
    /// protocol included — the caller parses/extracts).
    ///
    /// Uses the locked sampler config (temp 1.0 + top_p 0.95 + min_p 0.1 +
    /// greedy argmax) — same as the chat engine (AGENTS.md "Sampler config
    /// LOCKED"). Creative but not unhinged.
    ///
    /// No delta-prefill optimization for v1 — each turn does a full
    /// prefill. The accepted §2F cold-reset tax on memory-injected turns
    /// applies here too. Optimize later if TTFT becomes a constraint.
    fn generate_turn(&mut self, req: &GameRequest) -> Result<String, GenerationOutcome> {
        let mut tokens = self
            .model
            .str_to_token(&req.prompt, AddBos::Always)
            .map_err(|e| GenerationOutcome::GenerationErr(anyhow::anyhow!("game tokenize: {e:?}")))?;
        if tokens.is_empty() {
            return Err(GenerationOutcome::GenerationErr(anyhow::anyhow!(
                "game tokenized prompt is empty"
            )));
        }
        // Truncate from the front if the prompt alone exceeds context (keep
        // the system prompt's tail + recent turns + generation cue). Mirror
        // of the schema engine's guard.
        let max_prompt = (GAME_CTX as usize).saturating_sub(GAME_MAX_TOKENS as usize);
        if tokens.len() > max_prompt {
            let drop = tokens.len() - max_prompt;
            tokens.drain(0..drop);
            tracing::warn!(dropped = drop, "game prompt exceeded context; truncated from front");
        }

        // One-shot full prefill each turn (no KV reuse for v1).
        self.ctx.clear_kv_cache();

        let n_prompt = tokens.len() as i32;
        let mut batch = LlamaBatch::new(GAME_BATCH as usize, 1);
        let mut consumed = 0usize;
        while consumed < tokens.len() {
            let take = std::cmp::min(GAME_BATCH as usize, tokens.len() - consumed);
            let is_last_chunk = consumed + take == tokens.len();
            batch.clear();
            for (i, tok) in tokens[consumed..consumed + take].iter().enumerate() {
                let is_final = is_last_chunk && i == take - 1;
                batch
                    .add(*tok, (consumed + i) as i32, &[0], is_final)
                    .map_err(|e| {
                        GenerationOutcome::GenerationErr(anyhow::anyhow!("game batch add: {e:?}"))
                    })?;
            }
            self.ctx
                .decode(&mut batch)
                .map_err(|e| {
                    GenerationOutcome::GenerationErr(anyhow::anyhow!("game prefill decode: {e:?}"))
                })?;
            consumed += take;
        }

        // Locked sampler config (see module doc + AGENTS.md).
        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::temp(1.0),
            LlamaSampler::top_p(0.95, 1),
            LlamaSampler::min_p(0.1, 1),
            LlamaSampler::greedy(),
        ]);
        let eos = self.model.token_eos();
        let mut n_cur = n_prompt;
        let mut step_batch = LlamaBatch::new(1, 1);
        let mut out = String::new();
        let max_tokens = GAME_MAX_TOKENS
            .min((GAME_CTX as i32 - n_prompt).max(64));

        for _ in 0..max_tokens {
            // Cancellation check at the TOP of the loop (between tokens,
            // never mid-decode — same KV-consistency contract as the chat
            // engine, §2C).
            if req.cancel.load(std::sync::atomic::Ordering::Relaxed) {
                tracing::debug!("game turn cancelled by request");
                return Err(GenerationOutcome::Cancelled(out));
            }

            // sample(&ctx, -1) reads logits from the last decoded position.
            // Same direct API the chat engine uses (engine.rs:773).
            let new_token: LlamaToken = sampler.sample(&self.ctx, -1);
            sampler.accept(new_token);

            if self.model.is_eog_token(new_token) || new_token == eos {
                break;
            }

            // Detokenize + stream the piece (encoding_rs decoder for
            // multibyte safety, mirrors engine.rs:750-754).
            let mut decoder = encoding_rs::UTF_8.new_decoder();
            let piece = self
                .model
                .token_to_piece(new_token, &mut decoder, true, None)
                .map_err(|e| {
                    GenerationOutcome::GenerationErr(anyhow::anyhow!("game token to piece: {e:?}"))
                })?;
            if !piece.is_empty() {
                out.push_str(&piece);
                // Stream to the caller via the chunk callback.
                (req.on_chunk)(&piece);
            }

            // Feed the token back at position n_cur.
            step_batch.clear();
            step_batch
                .add(new_token, n_cur, &[0], true)
                .map_err(|e| {
                    GenerationOutcome::GenerationErr(anyhow::anyhow!("game decode batch: {e:?}"))
                })?;
            self.ctx
                .decode(&mut step_batch)
                .map_err(|e| {
                    GenerationOutcome::GenerationErr(anyhow::anyhow!("game decode: {e:?}"))
                })?;
            n_cur += 1;
        }

        // Sampler drops implicitly on scope exit — no explicit free() needed.
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// The handle is Send+Sync (manually asserted via unsafe impl). This test
    /// just confirms the type compiles with the right trait bounds — it
    /// doesn't construct one (that requires a real model load).
    #[test]
    fn game_engine_traits_compile() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<GameEngine>();
    }

    /// The reply struct serializes (it crosses the IPC boundary as JSON).
    #[test]
    fn game_reply_serializes() {
        let reply = GameReply {
            raw_output: "scene text".into(),
            error: String::new(),
            cancelled: false,
        };
        let json = serde_json::to_string(&reply).expect("serializes");
        assert!(json.contains("scene text"));
        assert!(json.contains("\"cancelled\":false"));
    }

    /// Constants are sane (compile-time sanity check).
    #[test]
    fn constants_are_sane() {
        assert!(GAME_CTX >= 2048, "game context must be generous");
        assert!(GAME_BATCH >= 256, "batch must fit a chunk");
        assert!(GAME_MAX_TOKENS >= 256, "max tokens must allow a meaty beat");
    }
}
