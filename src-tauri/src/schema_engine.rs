//! The background state-delta schema engine.
//!
//! A dedicated `std::thread` ("wupi-schema") owning an ISOLATED
//! `LlamaContext<'static>` on a model it loads ITSELF (not the chat model).
//! After each chat turn, `chat_send` posts a [`SchemaRequest`] here; the
//! thread generates a micro-delta JSON (only the changed keys), parses it,
//! and replies. The chat KV cache is never touched — true context isolation.
//!
//! # Independent model loading (2026-07-17, API feature)
//!
//! Previously this engine shared the chat model via `shared_model()`. That
//! coupling broke the moment chat could move to an HTTP API (no local model
//! to share). The engine now loads its OWN model by path — mirroring the
//! embedder (`memory_embedder_llama.rs`). In Local mode it loads `WUPI.gguf`;
//! in API mode it loads `Agent.gguf` (4B Gemma4, the dedicated agent). The
//! Gemma4 turn markers in the delta prompt work for both — Agent.gguf is
//! Gemma4 family.
//!
//! # Why a separate context (the load-bearing isolation requirement)
//!
//! The schema pass MUST NOT pollute the chat engine's rolling KV cache. A
//! second `LlamaContext` on the schema's own `&'static LlamaModel` achieves
//! this: independent KV state, no cross-contamination. This is the same
//! pattern as the embedder (§3B) — proven architecture.
//!
//! # The micro-delta contract
//!
//! The pass emits ONLY changed keys, not a full schema rewrite. A typical
//! delta is 20-100 tokens → sub-second generation. See `schema.rs` for the
//! merge semantics (`null` = delete key).
//!
//! # JSON robustness
//!
//! `SchemaDelta::from_model_output` strips markdown fences. If parsing still
//! fails, the thread retries once with a repair prompt; on second failure it
//! replies `Err` and the schema is left unchanged for that turn (graceful —
//! chat proceeds with stale schema).

use std::path::{Path, PathBuf};
use std::sync::mpsc;

use llama_cpp_2::context::params::{KvCacheType, LlamaContextParams};
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;

use crate::llm::shared_backend;
use crate::schema::{SchemaDelta, WorldSchema};

/// The schema context's token budget. Smaller than chat's 4000 — the delta
/// pass only needs: system instruction (~150 tokens) + current schema JSON
/// (~200-800) + last exchange (~100-400) + generation room. 2048 is generous
/// headroom; the KV cost at Q8_0 is ~75MB.
const SCHEMA_CTX: u32 = 2048;
const SCHEMA_BATCH: u32 = 512;
/// Cap on generated tokens for a delta pass. A compliant micro-delta is
/// 20-100 tokens; 256 is hard headroom before truncation forces the model to
/// stop rambling. If it hits this cap the output likely isn't valid JSON
/// anyway and the repair path or error path handles it.
const SCHEMA_MAX_TOKENS: i32 = 256;

// ---------------------------------------------------------------------------
// Control plane — channel types
// ---------------------------------------------------------------------------

/// A request to the schema thread: diff `last_exchange` against
/// `current_schema` and emit the changed keys.
struct SchemaRequest {
    /// (user_message, assistant_message) from the turn that just completed.
    last_exchange: (String, String),
    /// The current schema serialized as pretty JSON, so the model knows what
    /// to diff against.
    current_schema_json: String,
    /// One-shot reply channel.
    reply: mpsc::Sender<SchemaReply>,
}

/// What the schema thread sends back when a delta pass completes. Carries the
/// RAW model output alongside the parsed delta so callers (the debug IPC, and
/// Component D's queue) can see exactly what the model emitted — essential for
/// diagnosing JSON malformedness. On parse failure, `delta` is `None` and
/// `error` explains why. `raw_output` is always populated on a completed pass.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SchemaReply {
    /// The verbatim model output (post generation). Empty only if generation
    /// itself failed before producing tokens.
    pub raw_output: String,
    /// The parsed delta, if JSON was valid. `None` on parse failure or
    /// generation error.
    pub delta: Option<SchemaDelta>,
    /// Human-readable error if the pass failed (tokenize/prefill/decode, or
    /// JSON parse failure after both passes). Empty string on success.
    pub error: String,
}

enum SchemaMsg {
    Request(Box<SchemaRequest>),
    /// Translate a player's natural-language game-management request into a
    /// `SchemaDelta` (Phase E, 2026-07-18). Distinct from `Request` (the auto-
    /// summarizer's per-turn delta): the translation takes an explicit player
    /// command, not a just-finished chat exchange. Reuses the same JSON-delta
    /// parser + the schema engine's isolated context — no new infrastructure.
    RequestTranslation(Box<TranslationRequest>),
    /// Kept for future hot-swap / clean shutdown, mirroring `ChatEngine`.
    #[allow(dead_code)]
    Shutdown,
}

/// A request to translate a player's natural-language request ("make it
/// stormy") into a `SchemaDelta` against the current game-world schema.
/// Carries the raw player text + the current schema JSON. The handler uses
/// `game_command::render_translation_prompt` to build the LLM prompt, then
/// parses the reply via the same `SchemaDelta::from_model_output` the
/// auto-summarizer uses.
struct TranslationRequest {
    /// The player's verbatim request to Wupi (e.g. "make it stormy").
    player_request: String,
    /// The current game-world schema as pretty JSON (what to diff against).
    current_schema_json: String,
    /// One-shot reply channel.
    reply: mpsc::Sender<SchemaReply>,
}

// ---------------------------------------------------------------------------
// Handle (held by callers; fully Send + Sync)
// ---------------------------------------------------------------------------

/// The handle callers hold. Fully `Send + Sync` — a channel sender + the
/// thread's JoinHandle so `shutdown()` can block until VRAM is actually freed
/// (the fire-and-forget pattern was causing VRAM-overlap OOM during model
/// swaps — see the 2026-07-18 fix in `swap_schema_engine`). Mirrors
/// `ChatEngine` and `LlamaCppEmbedder`.
pub struct SchemaEngine {
    tx: mpsc::Sender<SchemaMsg>,
    join: std::sync::Mutex<Option<std::thread::JoinHandle<()>>>,
}

// SAFETY: mpsc::Sender<SchemaMsg> is Send (SchemaMsg owns only Send data).
// Mutex<Option<JoinHandle<()>>> is Send+Sync. No `LlamaContext` crosses out.
unsafe impl Send for SchemaEngine {}
unsafe impl Sync for SchemaEngine {}

impl SchemaEngine {
    /// Spawn the schema thread. The chat backend MUST be loaded first (we read
    /// `shared_model()` to get the leaked `&'static LlamaModel`). Returns
    /// `None` if no model is available — callers should treat the schema
    /// engine as optional (chat proceeds without schema updates).
    ///
    /// The readiness receiver yields `Ok(())` once the schema context is live
    /// (or `Err` if init failed). The caller SHOULD `recv()` before treating
    /// the engine as ready, same contract as `ChatEngine::spawn` (Bug #6).
    ///
    /// `path` is the model file this engine loads as ITS OWN model — no longer
    /// `shared_model()`. In Local mode pass WUPI.gguf; in API mode pass
    /// Agent.gguf. Mirrors `LlamaCppEmbedder::spawn_load`.
    pub fn spawn_load(
        path: PathBuf,
        n_gpu_layers: u32,
    ) -> (Self, mpsc::Receiver<Result<(), String>>) {
        let (tx, rx) = mpsc::channel::<SchemaMsg>();
        let (init_tx, init_rx) = mpsc::channel::<Result<(), String>>();

        let builder = std::thread::Builder::new().name("wupi-schema".into());
        let join = builder
            .spawn(move || {
                let mut runtime = match Self::init_runtime(&path, n_gpu_layers) {
                    Ok(rt) => {
                        let _ = init_tx.send(Ok(()));
                        rt
                    }
                    Err(e) => {
                        let msg = format!("schema engine init failed: {e}");
                        tracing::error!(error = %msg, "schema engine init failed; thread exiting");
                        let _ = init_tx.send(Err(msg.clone()));
                        Self::drain_failed(&rx, msg);
                        return;
                    }
                };
                tracing::info!("wupi-schema thread ready");

                loop {
                    // Both Request and RequestTranslation produce the same
                    // `(raw, Result<delta, err>)` outcome shape; we share the
                    // outcome → SchemaReply mapping (Phase E, 2026-07-18).
                    let parsed_msg = match rx.recv() {
                        Ok(SchemaMsg::Request(req)) => {
                            // Self-healing: isolate each delta pass so one
                            // panic doesn't kill the thread.
                            let outcome = std::panic::catch_unwind(
                                std::panic::AssertUnwindSafe(|| {
                                    runtime.generate_delta(&req)
                                }),
                            );
                            Some((outcome, req.reply))
                        }
                        Ok(SchemaMsg::RequestTranslation(req)) => {
                            // Phase E: same self-healing wrap, different runtime
                            // call. The translation prompt is built by
                            // `game_command::render_translation_prompt`; the
                            // parser is the same `SchemaDelta::from_model_output`.
                            let outcome = std::panic::catch_unwind(
                                std::panic::AssertUnwindSafe(|| {
                                    runtime.generate_translation(&req)
                                }),
                            );
                            Some((outcome, req.reply))
                        }
                        Ok(SchemaMsg::Shutdown) => {
                            tracing::info!("wupi-schema shutting down");
                            break;
                        }
                        Err(mpsc::RecvError) => {
                            tracing::info!("wupi-schema: all senders dropped, exiting");
                            break;
                        }
                    };
                    let Some((outcome, reply_tx)) = parsed_msg else { continue };
                    let reply_msg = match outcome {
                        Ok(Ok((raw, Ok(delta)))) => SchemaReply {
                            raw_output: raw,
                            delta: Some(delta),
                            error: String::new(),
                        },
                        Ok(Ok((raw, Err(e)))) => {
                            // Generation succeeded but JSON parse failed
                            // twice. Surface the raw output so the debug
                            // panel can show what the model emitted.
                            tracing::warn!(error = %e, "schema delta parse failed");
                            runtime.ctx.clear_kv_cache();
                            SchemaReply {
                                raw_output: raw,
                                delta: None,
                                error: e,
                            }
                        }
                        Ok(Err(e)) => {
                            // Generation itself failed (tokenize/prefill/
                            // decode). No raw output to report.
                            tracing::warn!(error = %format!("{e:#}"), "schema delta failed");
                            runtime.ctx.clear_kv_cache();
                            SchemaReply {
                                raw_output: String::new(),
                                delta: None,
                                error: format!("{e:#}"),
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
                                    "schema delta panic (unknown cause)".to_string()
                                });
                            tracing::error!(panic = %msg, "schema delta panicked");
                            runtime.ctx.clear_kv_cache();
                            SchemaReply {
                                raw_output: String::new(),
                                delta: None,
                                error: format!("schema panic: {msg}"),
                            }
                        }
                    };
                    let _ = reply_tx.send(reply_msg);
                }
            })
            .expect("failed to spawn wupi-schema thread");

        (
            SchemaEngine {
                tx,
                join: std::sync::Mutex::new(Some(join)),
            },
            init_rx,
        )
    }

    /// Shut down the schema thread AND block until it has fully exited +
    /// dropped its `SchemaRuntime` (LlamaContext + leaked model ref → VRAM
    /// freed). This synchronous wait is load-bearing during model swaps: the
    /// old fire-and-forget pattern posted Shutdown and returned immediately,
    /// so the next `spawn_load` raced the thread's VRAM teardown and OOM'd
    /// `load_from_file` → `NullResult` (Chloe's 2026-07-18 diagnosis of the
    /// E4B/Agent.gguf load failures). By blocking on the JoinHandle we
    /// guarantee VRAM is actually released before any new model allocates.
    pub fn shutdown(&self) {
        let _ = self.tx.send(SchemaMsg::Shutdown);
        // Take the JoinHandle (so repeated shutdown() calls are safe) and
        // block on it. A panic in the thread surfaces as Err; log + ignore
        // since shutdown is best-effort anyway.
        if let Ok(mut guard) = self.join.lock() {
            if let Some(handle) = guard.take() {
                if let Err(e) = handle.join() {
                    tracing::warn!(error = ?e, "wupi-schema thread join failed during shutdown");
                }
            }
        }
    }

    /// Post a delta request. The caller awaits the reply via the receiver
    /// it created. Fire-and-forget is NOT the contract here — the caller
    /// (chat_send's queue) needs the result before proceeding.
    pub fn request_delta(
        &self,
        last_exchange: (String, String),
        current_schema: &WorldSchema,
    ) -> anyhow::Result<mpsc::Receiver<SchemaReply>> {
        let (reply_tx, reply_rx) = mpsc::channel::<SchemaReply>();
        let req = SchemaRequest {
            last_exchange,
            current_schema_json: current_schema.to_json_pretty(),
            reply: reply_tx,
        };
        self.tx
            .send(SchemaMsg::Request(Box::new(req)))
            .map_err(|_| anyhow::anyhow!("schema engine thread closed"))?;
        Ok(reply_rx)
    }

    /// Post a TRANSLATION request (Phase E, 2026-07-18): translate a player's
    /// natural-language game-management request into a `SchemaDelta`. Used by
    /// `route_to_game_manager` when Wupi intercepts a "make it stormy" /
    /// "give me a sword" / "travel to the dungeon" command. Same reply
    /// contract as `request_delta` — caller awaits via the returned receiver.
    pub fn request_translation(
        &self,
        player_request: String,
        current_schema: &WorldSchema,
    ) -> anyhow::Result<mpsc::Receiver<SchemaReply>> {
        let (reply_tx, reply_rx) = mpsc::channel::<SchemaReply>();
        let req = TranslationRequest {
            player_request,
            current_schema_json: current_schema.to_json_pretty(),
            reply: reply_tx,
        };
        self.tx
            .send(SchemaMsg::RequestTranslation(Box::new(req)))
            .map_err(|_| anyhow::anyhow!("schema engine thread closed"))?;
        Ok(reply_rx)
    }

    fn drain_failed(rx: &mpsc::Receiver<SchemaMsg>, why: String) {
        while let Ok(msg) = rx.recv_timeout(std::time::Duration::from_millis(50)) {
            if let SchemaMsg::Request(req) = msg {
                let _ = req.reply.send(SchemaReply {
                    raw_output: String::new(),
                    delta: None,
                    error: why.clone(),
                });
            }
        }
    }

    /// Initialize the schema runtime: load the model by path (this engine's
    /// OWN model — no `shared_model()`), leak it to `&'static`, create an
    /// isolated context. Mirrors `memory_embedder_llama.rs::init_runtime`.
    /// Runs on the schema thread.
    fn init_runtime(path: &Path, n_gpu_layers: u32) -> anyhow::Result<SchemaRuntime> {
        let backend = shared_backend();

        let params = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers);
        let model = LlamaModel::load_from_file(backend, path, &params)
            .map_err(|e| anyhow::anyhow!("schema model load {}: {e:?}", path.display()))?;
        tracing::info!(path = %path.display(), "schema model loaded");

        // Leak the model to &'static so the context can borrow it for its
        // whole life. Same rationale as llm.rs::into_static + the embedder.
        let model_ref: &'static LlamaModel = Box::leak(Box::new(model));

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(SCHEMA_CTX))
            .with_n_batch(SCHEMA_BATCH)
            .with_embeddings(false)
            // Match the chat engine's KV quantization for consistency.
            .with_type_k(KvCacheType::Q8_0)
            .with_type_v(KvCacheType::Q8_0);
        let ctx = model_ref
            .new_context(backend, ctx_params)
            .map_err(|e| anyhow::anyhow!("schema context init: {e:?}"))?;
        tracing::info!(n_ctx = SCHEMA_CTX, "schema context created (isolated)");

        Ok(SchemaRuntime { ctx, model: model_ref })
    }
}

// ---------------------------------------------------------------------------
// Runtime (owned by the schema thread; never crosses thread boundaries)
// ---------------------------------------------------------------------------

struct SchemaRuntime {
    ctx: llama_cpp_2::context::LlamaContext<'static>,
    model: &'static LlamaModel,
}

impl SchemaRuntime {
    /// Generate a micro-delta for the given exchange + current schema.
    ///
    /// Two-pass: render the delta prompt, generate, parse JSON. If parsing
    /// fails, retry once with a repair prompt. On second failure, return Err
    /// (the schema is left unchanged for this turn — graceful degradation).
    ///
    /// Returns `(raw_output, Result<delta, error>)` so the caller can always
    /// see what the model emitted, even when parsing failed — essential for the
    /// debug panel's JSON-malformedness diagnosis.
    fn generate_delta(
        &mut self,
        req: &SchemaRequest,
    ) -> Result<(String, Result<SchemaDelta, String>), anyhow::Error> {
        let prompt = render_delta_prompt(&req.current_schema_json, &req.last_exchange);
        let raw = self.generate_text(&prompt)?;
        match SchemaDelta::from_model_output(&raw) {
            Ok(delta) => {
                tracing::debug!(tokens = raw.len(), "schema delta parsed on first pass");
                Ok((raw, Ok(delta)))
            }
            Err(first_err) => {
                tracing::warn!(
                    error = %first_err,
                    raw_preview = %raw.chars().take(200).collect::<String>(),
                    "schema delta JSON parse failed; retrying with repair prompt"
                );
                let repair = render_repair_prompt(&raw);
                let raw2 = self.generate_text(&repair)?;
                // The returned raw_output is the RETRY's output (the most
                // recent model emission) — that's what's diagnostically
                // useful when the caller wants to see why parsing failed.
                match SchemaDelta::from_model_output(&raw2) {
                    Ok(delta) => Ok((raw2, Ok(delta))),
                    Err(e) => Ok((
                        raw2,
                        Err(format!(
                            "schema delta parse failed twice: first={first_err}, retry={e}"
                        )),
                    )),
                }
            }
        }
    }

    /// Translate a player's natural-language game-management request into a
    /// `SchemaDelta` (Phase E, 2026-07-18). Same shape as `generate_delta` —
    /// two-pass with repair, same JSON parser — but the prompt is built by
    /// `game_command::render_translation_prompt` from the player's verbatim
    /// text + the current game-world schema. Used by Wupi-as-game-manager
    /// when she intercepts "make it stormy" / "give me a sword" via chat_send.
    fn generate_translation(
        &mut self,
        req: &TranslationRequest,
    ) -> Result<(String, Result<SchemaDelta, String>), anyhow::Error> {
        let prompt = crate::game_command::render_translation_prompt(
            &req.player_request,
            &req.current_schema_json,
        );
        let raw = self.generate_text(&prompt)?;
        match SchemaDelta::from_model_output(&raw) {
            Ok(delta) => {
                tracing::debug!(
                    tokens = raw.len(),
                    request = %req.player_request.chars().take(80).collect::<String>(),
                    "schema translation parsed on first pass"
                );
                Ok((raw, Ok(delta)))
            }
            Err(first_err) => {
                tracing::warn!(
                    error = %first_err,
                    request = %req.player_request.chars().take(80).collect::<String>(),
                    "schema translation parse failed; retrying with repair prompt"
                );
                let repair = render_repair_prompt(&raw);
                let raw2 = self.generate_text(&repair)?;
                match SchemaDelta::from_model_output(&raw2) {
                    Ok(delta) => Ok((raw2, Ok(delta))),
                    Err(e) => Ok((
                        raw2,
                        Err(format!(
                            "translation parse failed twice: first={first_err}, retry={e}"
                        )),
                    )),
                }
            }
        }
    }

    /// Tokenize → prefill → sample-and-decode a single response. One-shot
    /// generation with a max-tokens cap and greedy sampling (the delta is
    /// deterministic JSON; no creativity needed). Returns the decoded text.
    ///
    /// The context is fully reset each call (clear_kv_cache + re-prefill from
    /// zero). Unlike the chat engine, there's no delta-prefill optimization
    /// here — each prompt is a different schema + exchange, and the prompt is
    /// small (~1-2KB), so a full prefill each call is cheap and correct.
    ///
    /// The sample/detokenize/batch pattern mirrors `engine.rs::decode_loop`
    /// exactly: `sample(&ctx, -1)` reads from the last logits position,
    /// `accept` advances sampler state, `token_to_piece` with an encoding_rs
    /// decoder handles multibyte boundaries, and the sampled token is fed
    /// back at position `n_cur - 1`.
    fn generate_text(&mut self, prompt: &str) -> anyhow::Result<String> {
        let mut tokens = self
            .model
            .str_to_token(prompt, AddBos::Always)
            .map_err(|e| anyhow::anyhow!("schema tokenize: {e:?}"))?;
        if tokens.is_empty() {
            anyhow::bail!("schema tokenized prompt is empty");
        }
        // Guard: if the prompt alone exceeds the context, truncate from the
        // FRONT (keep the generation prompt + last exchange). Losing the
        // oldest schema detail beats failing entirely.
        let max_prompt = (SCHEMA_CTX as usize).saturating_sub(SCHEMA_MAX_TOKENS as usize);
        if tokens.len() > max_prompt {
            let drop = tokens.len() - max_prompt;
            tokens.drain(0..drop);
            tracing::warn!(dropped = drop, "schema prompt exceeded context; truncated from front");
        }

        // Fresh cache each call — the schema context is one-shot, no reuse.
        self.ctx.clear_kv_cache();

        // Prefill in batches (mirrors engine.rs::prefill).
        let n_prompt = tokens.len() as i32;
        let mut batch = LlamaBatch::new(SCHEMA_BATCH as usize, 1);
        let mut consumed = 0usize;
        while consumed < tokens.len() {
            let take = std::cmp::min(SCHEMA_BATCH as usize, tokens.len() - consumed);
            let is_last_chunk = consumed + take == tokens.len();
            batch.clear();
            for (i, tok) in tokens[consumed..consumed + take].iter().enumerate() {
                let is_final = is_last_chunk && i == take - 1;
                batch
                    .add(*tok, (consumed + i) as i32, &[0], is_final)
                    .map_err(|e| anyhow::anyhow!("schema batch add: {e:?}"))?;
            }
            self.ctx
                .decode(&mut batch)
                .map_err(|e| anyhow::anyhow!("schema prefill decode: {e:?}"))?;
            consumed += take;
        }

        // Sample-and-decode loop. Greedy (argmax) — JSON wants determinism,
        // and there's no ThoughtGate/StreamFilter here (the output is JSON,
        // not the Gemma4 channel protocol). n_cur = next position to decode.
        let mut sampler = LlamaSampler::greedy();
        let eos = self.model.token_eos();
        let mut n_cur = n_prompt;
        let mut step_batch = LlamaBatch::new(1, 1);
        let mut out = String::new();

        for _ in 0..SCHEMA_MAX_TOKENS {
            // sample(&ctx, -1) reads logits from the last decoded position.
            let new_token: LlamaToken = sampler.sample(&self.ctx, -1);
            sampler.accept(new_token);

            if self.model.is_eog_token(new_token) || new_token == eos {
                break;
            }

            // Detokenize with an encoding_rs decoder for multibyte safety
            // (mirrors engine.rs:750-754).
            let mut decoder = encoding_rs::UTF_8.new_decoder();
            let piece = self
                .model
                .token_to_piece(new_token, &mut decoder, true, None)
                .map_err(|e| anyhow::anyhow!("schema token to piece: {e:?}"))?;
            if !piece.is_empty() {
                out.push_str(&piece);
            }

            // Feed the sampled token back at position n_cur (one past the
            // last prefilled/decoded), then decode to produce the next
            // position's logits. Mirrors engine.rs:770-776.
            step_batch.clear();
            step_batch
                .add(new_token, n_cur, &[0], true)
                .map_err(|e| anyhow::anyhow!("schema decode batch: {e:?}"))?;
            self.ctx
                .decode(&mut step_batch)
                .map_err(|e| anyhow::anyhow!("schema decode: {e:?}"))?;
            n_cur += 1;
        }

        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Prompt rendering (Component C)
// ---------------------------------------------------------------------------

/// Render the schema-delta generation prompt. Uses the Gemma4 turn markers so
/// the model sees familiar structure, but the content is schema-specific.
/// NOT routed through `ChatFormat::render_prompt` — this is a dedicated
/// renderer (the schema pass isn't a chat turn).
fn render_delta_prompt(current_schema_json: &str, last_exchange: &(String, String)) -> String {
    let mut out = String::with_capacity(2048);
    out.push_str("<|turn>system\n");
    out.push_str(DELTA_SYSTEM_INSTRUCTION);
    out.push_str("<turn|>\n");
    out.push_str("<|turn>user\n");
    out.push_str("Current schema:\n");
    out.push_str(current_schema_json);
    out.push_str("\n\nLast exchange:\n[user]: ");
    out.push_str(&last_exchange.0);
    out.push_str("\n[model]: ");
    out.push_str(&last_exchange.1);
    out.push_str("\n<turn|>\n");
    out.push_str("<|turn>model\n");
    out
}

/// Repair prompt: when the first output wasn't valid JSON, ask again tightly.
fn render_repair_prompt(bad_output: &str) -> String {
    let mut out = String::with_capacity(1024);
    out.push_str("<|turn>system\n");
    out.push_str("Your previous output was not valid JSON. Emit ONLY the JSON delta object — no prose, no markdown fences, no commentary. If nothing changed, emit {}.");
    out.push_str("<turn|>\n");
    out.push_str("<|turn>user\n");
    out.push_str("Previous (invalid) output:\n");
    out.push_str(bad_output);
    out.push_str("\n<turn|>\n");
    out.push_str("<|turn>model\n");
    out
}

/// Cheap content gate for whether the schema delta pass should fire this turn.
///
/// The delta pass is a FULL 12B forward pass (tokenize + prefill + greedy
/// decode up to 256 tokens). Firing it unconditionally on every turn —
/// including "ok", "thanks", "lol", "yes" — burns ~1-4s of dedicated GPU time
/// for a turn that changed nothing in the world. This gate skips those.
///
/// **Conservative by design.** The cost of a false skip (missing a real world-
/// state change) is far higher than the cost of a false fire (one wasted pass),
/// so the bar to skip is HIGH: only short, clearly-non-substantive user turns
/// with a short assistant reply. Anything ambiguous fires the pass.
///
/// # What skips
///
/// - User message ≤ 4 words AND ≤ 32 chars (covers "ok", "thanks", "lol",
///   "yes", "no", "sure", "k", "yep", "continue", "ok cool", etc.).
/// - No assistant content (empty/error reply — nothing to record).
///
/// # What does NOT skip (deliberately)
///
/// - Short roleplay actions ("I nod", "I draw" — 2 words but world-moving).
///   These are 2 words but contain a verb in first person, so the word-count
///   gate alone is wrong for them. We can't distinguish "I nod" from "ok"
///   cheaply without a model call, so we FIRE on anything that looks like it
///   could be an action. The signal we use: presence of a pronoun ("i", "you",
///   "he", "she", "they", "we") or a verb-shape. Cheaper and safer to just
///   fire on anything that isn't obviously filler.
/// - Long assistant replies (a meaty reply likely reflects a meaty exchange).
///
/// Pure + allocation-light so it's testable in isolation (Prime Directive §3A:
/// retrieval/control logic stays decoupled from the model backend).
pub fn should_fire_delta(user_text: &str, assistant_text: &str) -> bool {
    // No assistant content → nothing to record (error turn, empty reply).
    if assistant_text.trim().is_empty() {
        return false;
    }
    let user = user_text.trim();
    // Word count via split_whitespace (handles runs of spaces/tabs/newlines).
    let word_count = user.split_whitespace().count();
    // Short AND compact → almost certainly filler. The char ceiling catches
    // 4 "words" that are actually one long token blob; the word ceiling catches
    // long rambling filler. Both must hold to skip.
    if word_count <= 4 && user.len() <= 32 {
        // Final guard: if the short message contains a first/second-person
        // pronoun, it might be a roleplay action ("I nod", "you see"). Fire
        // rather than risk skipping world state. Pronoun check is case-
        // insensitive on a small set; cheaper than a verb lookup.
        let lower = user.to_lowercase();
        const PRONOUNS: &[&str] = &[
            "i ", "i'", "i’m", "i'm", "i’ll", "i'll", "i’ve", "i've",
            "you ", "you'", "u ", "he ", "she ", "they ", "we ",
        ];
        let looks_like_action = PRONOUNS.iter().any(|p| lower.starts_with(p));
        if looks_like_action {
            return true; // ambiguous — fire to be safe
        }
        return false; // short, compact, no pronoun → filler, skip
    }
    // Everything else: fire. Long or substantive exchanges always get a pass.
    true
}

const DELTA_SYSTEM_INSTRUCTION: &str = "\
You are a world-state tracker. Given the current schema and the last exchange, emit ONLY the keys that changed as a JSON delta. Do NOT rewrite unchanged keys.

Output format (raw JSON only — no markdown fences, no prose):
{
  \"summary\": \"<updated summary string, or omit if unchanged>\",
  \"recent_events\": [\"<new event>\", ...],
  \"entities\": {\"<key>\": \"<new value>\", \"<key_to_delete>\": null}
}

Rules:
- Emit ONLY changed keys. Omit unchanged sections entirely. If nothing tracked changed this turn, emit {}.
- entities: a null value means DELETE the key. A non-null string means SET/overwrite.
- Keep the delta minimal — a few keys at most per turn.
- summary: only emit when the narrative arc meaningfully shifts, not every turn.
- recent_events: append only genuinely new salient events.\n";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delta_prompt_contains_system_instruction_and_exchange() {
        let prompt = render_delta_prompt(
            "{\"summary\":\"\"}",
            &("I pick up the sword".to_string(), "You grab it.".to_string()),
        );
        assert!(prompt.contains("world-state tracker"));
        assert!(prompt.contains("I pick up the sword"));
        assert!(prompt.contains("You grab it."));
        assert!(prompt.starts_with("<|turn>system\n"));
        assert!(prompt.ends_with("<|turn>model\n"));
    }

    #[test]
    fn repair_prompt_references_invalid_output() {
        let prompt = render_repair_prompt("not json at all");
        assert!(prompt.contains("not valid JSON"));
        assert!(prompt.contains("not json at all"));
    }

    // ── should_fire_delta gate tests ────────────────────────────────────
    // The gate is the M2 overhead fix: skip the full 12B forward pass on
    // clearly non-substantive turns. The contract is conservative — when in
    // doubt, fire (the cost of a missed world-state change > one wasted pass).

    #[test]
    fn gate_skips_short_filler_user_messages() {
        // The canonical skip cases: 1-4 word filler with a real assistant reply.
        let reply = "Sure thing, here's the info you asked for.";
        for filler in &[
            "ok", "thanks", "lol", "yes", "no", "sure", "k", "yep",
            "ok cool", "got it", "sounds good", "will do",
        ] {
            assert!(
                !should_fire_delta(filler, reply),
                "filler {filler:?} should skip the delta pass"
            );
        }
    }

    #[test]
    fn gate_skips_when_assistant_reply_is_empty() {
        // Empty/error reply → nothing to record, regardless of user message.
        assert!(!should_fire_delta("Tell me about the dungeon", ""));
        assert!(!should_fire_delta("Tell me about the dungeon", "   "));
    }

    #[test]
    fn gate_fires_on_normal_substantive_exchange() {
        // A real question + real reply → always fire.
        assert!(should_fire_delta(
            "What's in the iron chest?",
            "You open it and find a glowing amulet inside."
        ));
    }

    #[test]
    fn gate_fires_on_long_user_message_even_if_filler_sounding() {
        // 5+ words clears the word ceiling regardless of content — fires.
        assert!(should_fire_delta(
            "ok so anyway let me think about that for a second",
            "Take your time."
        ));
    }

    #[test]
    fn gate_fires_on_short_roleplay_action_with_pronoun() {
        // The critical false-negative guard: "I nod" is 2 words (would skip by
        // count alone) but it's a world-moving roleplay action. The pronoun
        // check catches it and fires.
        assert!(should_fire_delta("I nod", "She acknowledges you."));
        assert!(should_fire_delta("I draw my sword", "Roll for initiative."));
        assert!(should_fire_delta("you see a goblin", "It snarls."));
    }

    #[test]
    fn gate_fires_on_first_person_contraction() {
        // "I'm" / "I'll" — pronoun check covers contractions too.
        assert!(should_fire_delta("I'm going north", "The path narrows."));
        assert!(should_fire_delta("I'll attack", "You strike."));
    }

    #[test]
    fn gate_skips_short_message_without_pronoun_or_verb_shape() {
        // 3 words, no pronoun, not action-shaped → filler, skip.
        assert!(!should_fire_delta("lol that's funny", "Glad you enjoyed it."));
    }
}

// ---------------------------------------------------------------------------
// Model discovery (mirrors memory_embedder_llama.rs)
// ---------------------------------------------------------------------------

/// Find a named `.gguf` model in `dir` (case-insensitive exact-name match).
/// Used by [`resolve_schema_model`] to locate either `WUPI.gguf` (Local mode)
/// or `Agent.gguf` (API mode) for the schema engine's own model load.
fn pick_named_model(dir: &Path, name: &str) -> Option<PathBuf> {
    let target = name.to_lowercase();
    std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .find(|e| {
            e.path().extension().and_then(|x| x.to_str()).map_or(false, |x| x.eq_ignore_ascii_case("gguf"))
                && e.file_name().to_string_lossy().to_lowercase() == target
        })
        .map(|e| e.path())
}

/// Resolve the schema engine's model across the standard candidate dirs.
/// `name` is `"WUPI.gguf"` in Local mode or `"Agent.gguf"` in API mode.
/// Mirrors `memory_embedder_llama::resolve_embed_model` + the candidate walk
/// in `lib.rs::resolve_model_path`. Returns `None` if no matching file exists
/// — the caller falls back to running without schema deltas (graceful).
pub fn resolve_schema_model(dirs: &[PathBuf], name: &str) -> Option<PathBuf> {
    for dir in dirs {
        if dir.exists() {
            if let Some(p) = pick_named_model(dir, name) {
                tracing::info!("resolved schema model ({name}): {}", p.display());
                return Some(p);
            }
        }
    }
    tracing::warn!("no {name} found — schema engine will not start (no world-state deltas)");
    None
}
