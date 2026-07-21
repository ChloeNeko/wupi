//! The background state-delta schema engine.
//!
//! A dedicated `std::thread` ("wupi-schema") owning an ISOLATED
//! `LlamaContext<'static>` on `WUPI.gguf`. After each chat turn, `chat_send`
//! posts a [`SchemaRequest`] here; the thread generates a micro-delta JSON
//! (only the changed keys), parses it, and replies. The chat KV cache is
//! never touched: true context isolation.
//!
//! # Why a separate context (the load-bearing isolation requirement)
//!
//! The schema pass MUST NOT pollute the chat engine's rolling KV cache. A
//! second `LlamaContext` on the schema's own `&'static LlamaModel` achieves
//! this: independent KV state, no cross-contamination. Same pattern as the
//! embedder (§3B): proven architecture.
//!
//! # The micro-delta contract
//!
//! Emits ONLY changed keys, not a full schema rewrite. A typical delta is
//! 20-100 tokens for sub-second generation. See `schema.rs` for the merge
//! semantics (`null` = delete key).
//!
//! # The fail-proof contract (3-pass + Rust validator + failure queue)
//!
//! Replaces the earlier "two-pass, drop on second fail" behavior. The new
//! invariant (locked 2026-07-20, §5): **no world-state evolution is ever
//! silently dropped.** Three layers, cheapest-first:
//!
//! 1. **Pure-Rust shape validator** (`schema_validator::validate`, ~0 cost).
//!    Enforces structural integrity (key/value length, no control chars,
//!    per-delta count caps). Defense by *structure*: a delta that fails
//!    validation gets fed its specific error back via the repair prompt so
//!    the model can correct the *issue*, not just regenerate blindly.
//! 2. **3-pass repair loop with accumulating context.** Initial generation →
//!    if parse OR validation fails, repair pass 1 (shows pass 1's raw output
//!    + the specific error) → repair pass 2 (shows BOTH prior errors + both
//!    raw outputs). Cap is 3 (empirically the LLM-JSON-repair cliff; passes
//!    4+ mostly produce different versions of the same failure). Worst case
//!    ~15-24s vs 35-56s for the rejected 7-pass proposal.
//! 3. **Failure queue (`failed_delta_queue` on AppState).** A delta that
//!    still fails all 3 passes is NOT dropped: it's queued. The next turn's
//!    delta prompt folds in the failed attempt as "previously deferred state
//!    change — re-attempt with new context." The new conversational context
//!    is a strictly better retry signal than re-running the same failed
//!    prompt. `SchemaReply::failed_attempt` carries the data the caller
//!    needs to enqueue.

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
use crate::schema_validator;

/// The schema context's token budget. Smaller than chat's 4000: the delta
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

/// Maximum number of generation passes per delta attempt (initial + 2
/// repairs = 3 total). The 4th-and-beyond cliff is empirically steep for
/// LLM-JSON-repair; the failure queue (fold-into-next-turn) is the strictly
/// better retry strategy past this cap. See module doc "The fail-proof
/// contract" layer 2.
const MAX_DELTA_PASSES: u8 = 3;

// ---------------------------------------------------------------------------
// Control plane: channel types
// ---------------------------------------------------------------------------

/// A request to the schema thread: diff `last_exchange` against
/// `current_schema` and emit the changed keys.
struct SchemaRequest {
    /// (user_message, assistant_message) from the turn that just completed.
    last_exchange: (String, String),
    /// The current schema serialized as pretty JSON, so the model knows what
    /// to diff against.
    current_schema_json: String,
    /// Deferred attempts from prior turns that the engine couldn't commit
    /// (all 3 passes failed). Folded into this turn's prompt as
    /// "previously deferred state changes — re-attempt with the new exchange
    /// as context." Empty in the common case (no prior failures).
    deferred_attempts: Vec<FailedAttempt>,
    /// One-shot reply channel.
    reply: mpsc::Sender<SchemaReply>,
}

/// What the schema thread sends back when a delta pass completes. Carries the
/// RAW model output alongside the parsed delta so callers (the debug IPC, and
/// Component D's queue) can see exactly what the model emitted: essential for
/// diagnosing JSON malformedness. On parse failure, `delta` is `None` and
/// `error` explains why. `raw_output` is always populated on a completed pass.
///
/// `failed_attempt` is `Some` ONLY when all 3 passes failed AND the failure
/// looks retryable (parse failures, validation failures). Generation errors
/// (tokenize/prefill/decode infrastructure failures) leave it `None` — those
/// aren't going to fix themselves on the next turn. The caller (lib.rs's
/// delta-fire spawn) pushes the `FailedAttempt` onto the failure queue; the
/// next turn's delta prompt folds it in. See module doc layer 3.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SchemaReply {
    /// The verbatim model output (post generation). Empty only if generation
    /// itself failed before producing tokens.
    pub raw_output: String,
    /// The parsed delta, if JSON was valid AND passed validation. `None` on
    /// parse failure, validation failure, or generation error.
    pub delta: Option<SchemaDelta>,
    /// Human-readable error if the pass failed (tokenize/prefill/decode, or
    /// JSON parse failure after all passes, or validation failure after all
    /// passes). Empty string on success.
    pub error: String,
    /// Populated when all 3 passes failed AND the failure is retryable
    /// (parse/validation errors). The caller enqueues this; the next turn
    /// re-attempts with fresh conversational context. `None` on success, on
    /// infrastructure errors, or on generation panics (those don't benefit
    /// from a retry).
    #[serde(default)]
    pub failed_attempt: Option<FailedAttempt>,
}

/// A deferred delta: the schema engine's claim check for a turn's
/// world-state evolution that it couldn't commit. The caller (lib.rs) holds
/// these in `failed_delta_queue`; the next turn's `request_delta` /
/// `request_translation` call passes them in via `deferred_attempts` so the
/// prompt can fold them in ("previously deferred state change — re-attempt
/// with new context").
///
/// Carries the *triggering context*, not the failed model output: re-running
/// the same broken output through the model rarely helps. What helps is
/// giving the model a fresh generation pass with the *exchange* that
/// produced the broken delta, alongside the new turn's exchange. The model
/// gets two shots worth of conversational signal.
#[derive(Debug, Clone, serde::Serialize)]
pub struct FailedAttempt {
    /// The (user, assistant) exchange that produced the failed delta. Empty
    /// for translation attempts (which carry `trigger` instead).
    pub exchange: Option<(String, String)>,
    /// The player request that produced the failed translation. `None` for
    /// auto-summarizer attempts (which carry `exchange` instead).
    pub trigger: Option<String>,
    /// The accumulated errors from all 3 passes, joined. The next attempt's
    /// prompt can include this so the model knows what went wrong last time.
    pub errors: String,
    /// How many times this attempt has been retried (always 3 on first
    /// enqueue; the caller bumps it if a deferred re-attempt ALSO fails and
    /// re-enqueues). The queue caps total retries to avoid pathological
    /// loops — see lib.rs's `failed_delta_queue` cap.
    pub passes_used: u8,
}

/// Type alias distinguishing the two kinds of triggering context an attempt
/// carries. Internal to the engine; `FailedAttempt` exposes them as
/// Option<(exchange)> / Option<request> for the IPC boundary.
#[derive(Clone)]
enum AttemptSource {
    /// Auto-summarizer: triggered by a chat exchange.
    Auto { exchange: (String, String) },
    /// Game-manager translation: triggered by an explicit player request.
    Translation { request: String },
}

/// The outcome of a delta-or-translation attempt. The engine's internal
/// return type; the message handler maps this to `SchemaReply` for the IPC
/// boundary. Distinguishes "committed cleanly" from "retryable failure" (the
/// carrier lets the caller enqueue for next turn).
enum AttemptOutcome {
    /// The delta parsed + validated cleanly. Ready to apply.
    Committed { raw_output: String, delta: SchemaDelta },
    /// All passes failed. `last_raw_output` is for the debug panel; `errors`
    /// is the joined accumulated diagnostics; `carrier` is what the caller
    /// pushes onto `failed_delta_queue` so the next turn re-attempts.
    Failed {
        last_raw_output: String,
        errors: String,
        carrier: FailedAttempt,
    },
}

enum SchemaMsg {
    Request(Box<SchemaRequest>),
    /// Translate a player's natural-language game-management request into a
    /// `SchemaDelta` (Phase E, 2026-07-18). Distinct from `Request` (the auto-
    /// summarizer's per-turn delta): the translation takes an explicit player
    /// command, not a just-finished chat exchange. Reuses the same JSON-delta
    /// parser + the schema engine's isolated context, no new infrastructure.
    RequestTranslation(Box<TranslationRequest>),
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
    /// Deferred translation attempts from prior player requests that the
    /// engine couldn't commit. Folded into this request's prompt so the
    /// model gets another shot with new context. Empty in the common case.
    deferred_attempts: Vec<FailedAttempt>,
    /// One-shot reply channel.
    reply: mpsc::Sender<SchemaReply>,
}

// ---------------------------------------------------------------------------
// Handle (held by callers; fully Send + Sync)
// ---------------------------------------------------------------------------

/// The handle callers hold. Fully `Send + Sync`: a channel sender to the
/// dedicated schema thread. No `LlamaContext` crosses out. The thread lives
/// for process lifetime (no hot-swap path now that the schema engine stays
/// on WUPI.gguf in both Local and API modes, §2X).
pub struct SchemaEngine {
    tx: mpsc::Sender<SchemaMsg>,
}

// SAFETY: mpsc::Sender<SchemaMsg> is Send (SchemaMsg owns only Send data).
// Mutex<Option<JoinHandle<()>>> is Send+Sync. No `LlamaContext` crosses out.
unsafe impl Send for SchemaEngine {}
unsafe impl Sync for SchemaEngine {}

impl SchemaEngine {
    /// Spawn the schema thread. The chat backend MUST be loaded first (we read
    /// `shared_model()` to get the leaked `&'static LlamaModel`). Returns
    /// `None` if no model is available: callers should treat the schema
    /// engine as optional (chat proceeds without schema updates).
    ///
    /// The readiness receiver yields `Ok(())` once the schema context is live
    /// (or `Err` if init failed). The caller SHOULD `recv()` before treating
    /// the engine as ready, same contract as `ChatEngine::spawn` (Bug #6).
    ///
    /// `path` is the model file this engine loads as ITS OWN model: no longer
    /// `shared_model()`. In Local mode pass WUPI.gguf; in API mode pass
    /// Agent.gguf. Mirrors `LlamaCppEmbedder::spawn_load`.
    pub fn spawn_load(
        path: PathBuf,
        n_gpu_layers: u32,
    ) -> (Self, mpsc::Receiver<Result<(), String>>) {
        let (tx, rx) = mpsc::channel::<SchemaMsg>();
        let (init_tx, init_rx) = mpsc::channel::<Result<(), String>>();

        let builder = std::thread::Builder::new().name("wupi-schema".into());
        let _join = builder
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
                        Err(mpsc::RecvError) => {
                            tracing::info!("wupi-schema: all senders dropped, exiting");
                            break;
                        }
                    };
                    let Some((outcome, reply_tx)) = parsed_msg else { continue };
                    let reply_msg = match outcome {
                        // Generation succeeded; delta parsed + validated.
                        Ok(Ok(AttemptOutcome::Committed { raw_output, delta })) => SchemaReply {
                            raw_output,
                            delta: Some(delta),
                            error: String::new(),
                            failed_attempt: None,
                        },
                        // Generation succeeded but all 3 passes failed
                        // (parse/validation). Retryable: surface the carrier
                        // so the caller enqueues for next-turn re-attempt.
                        // The schema is unchanged for THIS turn.
                        Ok(Ok(AttemptOutcome::Failed { last_raw_output, errors, carrier })) => {
                            tracing::warn!(
                                error = %errors,
                                passes = carrier.passes_used,
                                "schema attempt failed all passes; queuing for re-attempt"
                            );
                            runtime.ctx.clear_kv_cache();
                            SchemaReply {
                                raw_output: last_raw_output,
                                delta: None,
                                error: errors,
                                failed_attempt: Some(carrier),
                            }
                        }
                        // Generation itself failed (tokenize/prefill/decode).
                        // Infrastructure failure: not retryable, no carrier.
                        Ok(Err(e)) => {
                            tracing::warn!(error = %format!("{e:#}"), "schema generation failed (infrastructure)");
                            runtime.ctx.clear_kv_cache();
                            SchemaReply {
                                raw_output: String::new(),
                                delta: None,
                                error: format!("{e:#}"),
                                failed_attempt: None,
                            }
                        }
                        // The catch_unwind caught a panic. Thread survives
                        // (KV cleared below); not retryable, no carrier.
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
                                failed_attempt: None,
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
            },
            init_rx,
        )
    }

    /// Post a delta request. The caller awaits the reply via the receiver
    /// it created. Fire-and-forget is NOT the contract here: the caller
    /// (chat_send's queue) needs the result before proceeding.
    ///
    /// `deferred_attempts` carries failures from prior turns (folded into
    /// the prompt so the model gets another shot with fresh context). Pass
    /// an empty vec in the common case; the caller is responsible for
    /// draining the failure queue.
    pub fn request_delta(
        &self,
        last_exchange: (String, String),
        current_schema: &WorldSchema,
        deferred_attempts: Vec<FailedAttempt>,
    ) -> anyhow::Result<mpsc::Receiver<SchemaReply>> {
        let (reply_tx, reply_rx) = mpsc::channel::<SchemaReply>();
        let req = SchemaRequest {
            last_exchange,
            current_schema_json: current_schema.to_json_pretty(),
            deferred_attempts,
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
    /// contract as `request_delta`: caller awaits via the returned receiver.
    ///
    /// `deferred_attempts` carries translation failures from prior player
    /// requests. Pass an empty vec in the common case.
    pub fn request_translation(
        &self,
        player_request: String,
        current_schema: &WorldSchema,
        deferred_attempts: Vec<FailedAttempt>,
    ) -> anyhow::Result<mpsc::Receiver<SchemaReply>> {
        let (reply_tx, reply_rx) = mpsc::channel::<SchemaReply>();
        let req = TranslationRequest {
            player_request,
            current_schema_json: current_schema.to_json_pretty(),
            deferred_attempts,
            reply: reply_tx,
        };
        self.tx
            .send(SchemaMsg::RequestTranslation(Box::new(req)))
            .map_err(|_| anyhow::anyhow!("schema engine thread closed"))?;
        Ok(reply_rx)
    }

    fn drain_failed(rx: &mpsc::Receiver<SchemaMsg>, why: String) {
        while let Ok(msg) = rx.recv_timeout(std::time::Duration::from_millis(50)) {
            // Both Request and RequestTranslation carry a reply sender that
            // needs an error response on init failure. The deferred_attempts
            // are dropped (the caller will re-enqueue them next turn from
            // its own failure queue — the schema thread's queue is separate
            // from the caller's AppState queue and is always empty between
            // turns).
            let reply_tx = match msg {
                SchemaMsg::Request(r) => r.reply,
                SchemaMsg::RequestTranslation(r) => r.reply,
            };
            let _ = reply_tx.send(SchemaReply {
                raw_output: String::new(),
                delta: None,
                error: why.clone(),
                failed_attempt: None, // infrastructure failure, not retryable
            });
        }
    }

    /// Initialize the schema runtime: load the model by path (this engine's
    /// OWN model: no `shared_model()`), leak it to `&'static`, create an
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
    /// Fail-proof contract (see module doc): 3-pass max with accumulating
    /// repair context, validator between parse and success, failure queue
    /// carrier on the returned `AttemptOutcome::Failed` variant.
    fn generate_delta(
        &mut self,
        req: &SchemaRequest,
    ) -> Result<AttemptOutcome, anyhow::Error> {
        let initial_prompt = render_delta_prompt(
            &req.current_schema_json,
            &req.last_exchange,
            &req.deferred_attempts,
        );
        self.generate_with_repair(
            &initial_prompt,
            AttemptSource::Auto {
                exchange: req.last_exchange.clone(),
            },
            &req.deferred_attempts,
            "schema delta",
        )
    }

    /// Translate a player's natural-language game-management request into a
    /// `SchemaDelta` (Phase E, 2026-07-18). Same fail-proof contract as
    /// `generate_delta`: 3-pass + validator + failure queue. The initial
    /// prompt is built by `game_command::render_translation_prompt` from the
    /// player's verbatim text + the current game-world schema. Used by
    /// Wupi-as-game-manager when she intercepts "make it stormy" / "give me
    /// a sword" via chat_send.
    fn generate_translation(
        &mut self,
        req: &TranslationRequest,
    ) -> Result<AttemptOutcome, anyhow::Error> {
        let initial_prompt = crate::game_command::render_translation_prompt(
            &req.player_request,
            &req.current_schema_json,
            &req.deferred_attempts,
        );
        self.generate_with_repair(
            &initial_prompt,
            AttemptSource::Translation {
                request: req.player_request.clone(),
            },
            &req.deferred_attempts,
            "schema translation",
        )
    }

    /// The shared 3-pass repair loop. Runs the model up to `MAX_DELTA_PASSES`
    /// times. Each pass parses the output via `SchemaDelta::from_model_output`
    /// AND validates it via `schema_validator::validate`. A pass succeeds only
    /// if both parse and validation succeed. Repair prompts accumulate prior
    /// errors + prior raw outputs so the model sees what it got wrong, not
    /// just a generic "try again."
    ///
    /// Returns `AttemptOutcome` (success / parse-or-validation failure /
    /// retryable-failure-with-carrier) so the message handler can build the
    /// right `SchemaReply` including the failure-queue carrier.
    ///
    /// `label` is a short diagnostic ("schema delta" / "schema translation")
    /// used in tracing. `source` carries the trigger context (exchange or
    /// player request) so a failed attempt can be re-attempted on the next
    /// turn. `prior_deferred` is the failures folded in from previous turns;
    /// it does NOT count toward this attempt's pass budget.
    fn generate_with_repair(
        &mut self,
        initial_prompt: &str,
        source: AttemptSource,
        prior_deferred: &[FailedAttempt],
        label: &'static str,
    ) -> Result<AttemptOutcome, anyhow::Error> {
        let validation_ctx = schema_validator::ValidationContext::default();

        // Track every failure across all passes so we can (a) accumulate them
        // into the repair prompt and (b) carry them on the FailedAttempt if
        // all passes fail.
        let mut errors: Vec<String> = Vec::with_capacity(MAX_DELTA_PASSES as usize);
        let mut raw_outputs: Vec<String> = Vec::with_capacity(MAX_DELTA_PASSES as usize);
        let mut last_raw = String::new();

        for pass in 1..=MAX_DELTA_PASSES {
            let prompt: String = if pass == 1 {
                // First pass: the caller-built initial prompt (delta or
                // translation), already includes deferred-attempts context
                // if any.
                initial_prompt.to_string()
            } else {
                // Repair pass: shows the accumulated raw outputs + errors
                // from every prior pass. The model sees what it got wrong
                // and why, so it can correct the specific issue.
                render_accumulated_repair_prompt(&raw_outputs, &errors)
            };
            let raw = self.generate_text(&prompt)?;
            last_raw = raw.clone();
            raw_outputs.push(raw.clone());

            // Parse the JSON (channel-protocol + fence strip happens inside
            // from_model_output).
            let parsed = SchemaDelta::from_model_output(&raw);
            let delta = match parsed {
                Ok(d) => d,
                Err(e) => {
                    let msg = format!("pass {pass} JSON parse: {e}");
                    tracing::warn!(
                        label,
                        pass,
                        error = %e,
                        raw_preview = %raw.chars().take(200).collect::<String>(),
                        "{label} parse failed"
                    );
                    errors.push(msg);
                    continue; // next pass
                }
            };

            // Validate structure. This is the §1B defense layer: catches
            // parseable-but-corrupt deltas (control chars, runaway length,
            // count-cap violations) at zero LLM cost.
            if let Err(vfail) = schema_validator::validate(&delta, &validation_ctx) {
                let msg = format!("pass {pass} validation: {vfail}");
                tracing::warn!(label, pass, failure = %vfail, "{label} validation failed");
                errors.push(msg);
                continue; // next pass — repair prompt will show the failure
            }

            // Success: parse OK + validation OK. Trace + return.
            tracing::debug!(
                label,
                pass,
                tokens = raw.len(),
                deferred = prior_deferred.len(),
                "{label} committed on pass {pass}"
            );
            return Ok(AttemptOutcome::Committed { raw_output: raw, delta });
        }

        // All passes exhausted. Build the failure-queue carrier so the caller
        // can enqueue this for re-attempt on the next turn. The carrier
        // carries the SOURCE (exchange or request) + the accumulated errors;
        // it does NOT carry the broken raw outputs (re-running those through
        // the model rarely helps; fresh context does).
        let (exchange_opt, trigger_opt) = match &source {
            AttemptSource::Auto { exchange } => (Some(exchange.clone()), None),
            AttemptSource::Translation { request } => (None, Some(request.clone())),
        };
        tracing::warn!(
            label,
            passes = MAX_DELTA_PASSES,
            errors = errors.join(" | "),
            "{label} failed all {MAX_DELTA_PASSES} passes; carrying for re-attempt"
        );
        Ok(AttemptOutcome::Failed {
            last_raw_output: last_raw,
            errors: errors.join(" | "),
            carrier: FailedAttempt {
                exchange: exchange_opt,
                trigger: trigger_opt,
                errors: errors.join(" | "),
                passes_used: MAX_DELTA_PASSES,
            },
        })
    }

    /// Tokenize → prefill → sample-and-decode a single response. One-shot
    /// generation with a max-tokens cap and greedy sampling (the delta is
    /// deterministic JSON; no creativity needed). Returns the decoded text.
    ///
    /// The context is fully reset each call (clear_kv_cache + re-prefill from
    /// zero). Unlike the chat engine, there's no delta-prefill optimization
    /// here: each prompt is a different schema + exchange, and the prompt is
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

        // Fresh cache each call: the schema context is one-shot, no reuse.
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

        // Sample-and-decode loop. Greedy (argmax): JSON wants determinism,
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
/// NOT routed through `ChatFormat::render_prompt`: this is a dedicated
/// renderer (the schema pass isn't a chat turn).
///
/// `deferred_attempts` carries failures from prior turns (fail-proof contract
/// §5 layer 3). Folded in as "previously deferred state changes — re-attempt
/// with this turn's exchange as anchor." Empty in the common case.
fn render_delta_prompt(
    current_schema_json: &str,
    last_exchange: &(String, String),
    deferred_attempts: &[FailedAttempt],
) -> String {
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
    // Deferred re-attempt context. When the previous turn's delta failed all
    // 3 passes, fold its triggering exchange + accumulated errors in here so
    // the model gets another shot with the new exchange as anchor.
    if !deferred_attempts.is_empty() {
        out.push_str(
            "\n\n[Previously deferred state changes — re-attempt with the above exchange as the primary context:]\n",
        );
        for (i, attempt) in deferred_attempts.iter().enumerate() {
            let (u, a) = attempt
                .exchange
                .clone()
                .unwrap_or(("".to_string(), "".to_string()));
            out.push_str(&format!(
                "  {}. prior [user]: {:?}\n      prior [model]: {:?}\n      prior errors: {}\n",
                i + 1,
                u.chars().take(200).collect::<String>(),
                a.chars().take(200).collect::<String>(),
                attempt.errors
            ));
        }
    }
    out.push_str("\n<turn|>\n");
    out.push_str("<|turn>model\n");
    out
}

/// Accumulating repair prompt. Shows the model EVERY prior pass's raw output
/// + every prior error, so it can correct the *specific* issue rather than
/// regenerate blindly. This is the §1B-aligned repair: structured feedback
/// at the cost of one LLM pass, not 7 blind retries.
///
/// The accumulated-context shape is load-bearing: pass 3 sees both pass 1
/// and pass 2's outputs + errors, giving the model maximum signal on its
/// final attempt before the failure queue takes over.
fn render_accumulated_repair_prompt(prior_raw: &[String], prior_errors: &[String]) -> String {
    let mut out = String::with_capacity(1024 + prior_raw.len() * 256);
    out.push_str("<|turn>system\n");
    out.push_str(
        "Your previous output(s) were invalid. Emit ONLY the JSON delta object: no prose, no markdown fences, no commentary. Address EACH error below. If nothing actually changed, emit {}.",
    );
    out.push_str("<turn|>\n");
    out.push_str("<|turn>user\n");
    out.push_str(&format!("{} prior attempt(s) failed:\n", prior_raw.len()));
    for (i, raw) in prior_raw.iter().enumerate() {
        let err = prior_errors.get(i).map(|s| s.as_str()).unwrap_or("(no error recorded)");
        out.push_str(&format!(
            "\n--- Attempt {} ---\nError: {}\nYour output was:\n{}\n",
            i + 1,
            err,
            raw.chars().take(500).collect::<String>(),
        ));
    }
    out.push_str("\n---\nNow emit the corrected JSON delta:\n<turn|>\n");
    out.push_str("<|turn>model\n");
    out
}

/// Cheap content gate for whether the schema delta pass should fire this turn.
///
/// The delta pass is a FULL 12B forward pass (tokenize + prefill + greedy
/// decode up to 256 tokens). Firing it unconditionally on every turn -
/// including "ok", "thanks", "lol", "yes": burns ~1-4s of dedicated GPU time
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
/// - No assistant content (empty/error reply: nothing to record).
///
/// # What does NOT skip (deliberately)
///
/// - Short roleplay actions ("I nod", "I draw": 2 words but world-moving).
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
            return true; // ambiguous: fire to be safe
        }
        return false; // short, compact, no pronoun → filler, skip
    }
    // Everything else: fire. Long or substantive exchanges always get a pass.
    true
}

const DELTA_SYSTEM_INSTRUCTION: &str = "\
You are a world-state tracker. Given the current schema and the last exchange, emit ONLY the keys that changed as a JSON delta. Do NOT rewrite unchanged keys.

Output format (raw JSON only: no markdown fences, no prose):
{
  \"summary\": \"<updated summary string, or omit if unchanged>\",
  \"recent_events\": [\"<new event>\", ...],
  \"entities\": {\"<key>\": \"<new value>\", \"<key_to_delete>\": null}
}

Rules:
- Emit ONLY changed keys. Omit unchanged sections entirely. If nothing tracked changed this turn, emit {}.
- entities: a null value means DELETE the key. A non-null string means SET/overwrite.
- Keep the delta minimal: a few keys at most per turn.
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
            &[], // no deferred attempts in the common case
        );
        assert!(prompt.contains("world-state tracker"));
        assert!(prompt.contains("I pick up the sword"));
        assert!(prompt.contains("You grab it."));
        assert!(prompt.starts_with("<|turn>system\n"));
        assert!(prompt.ends_with("<|turn>model\n"));
    }

    #[test]
    fn delta_prompt_folds_deferred_attempts_when_present() {
        // The fail-proof contract layer 3: a prior turn's failed delta must
        // surface in the next turn's prompt so the model gets a fresh shot.
        let deferred = vec![FailedAttempt {
            exchange: Some(("prior user text".to_string(), "prior model text".to_string())),
            trigger: None,
            errors: "pass 1 JSON parse: ... | pass 2 validation: ...".to_string(),
            passes_used: MAX_DELTA_PASSES,
        }];
        let prompt = render_delta_prompt(
            "{\"summary\":\"\"}",
            &("new user text".to_string(), "new model text".to_string()),
            &deferred,
        );
        assert!(prompt.contains("Previously deferred"));
        assert!(prompt.contains("prior user text"));
        assert!(prompt.contains("prior model text"));
        assert!(prompt.contains("pass 1 JSON parse"));
        // The new exchange is still the primary anchor.
        assert!(prompt.contains("new user text"));
    }

    #[test]
    fn accumulated_repair_prompt_shows_every_prior_pass() {
        // Pass 3 sees both pass 1 and pass 2's outputs + errors (load-bearing
        // for the accumulating-context shape — vs the old single-shot repair).
        let prior_raw = vec![
            "first bad output".to_string(),
            "second bad output".to_string(),
        ];
        let prior_errors = vec![
            "pass 1 JSON parse: unexpected token".to_string(),
            "pass 2 validation: invalid entity key".to_string(),
        ];
        let prompt = render_accumulated_repair_prompt(&prior_raw, &prior_errors);
        assert!(prompt.contains("2 prior attempt(s) failed"));
        assert!(prompt.contains("first bad output"));
        assert!(prompt.contains("second bad output"));
        assert!(prompt.contains("pass 1 JSON parse"));
        assert!(prompt.contains("pass 2 validation"));
    }

    #[test]
    fn accumulated_repair_prompt_truncates_long_raw_outputs() {
        // Defense against prompt bloat: a 10KB garbage raw output shouldn't
        // eat the entire context. Capped at 500 chars per pass.
        let huge = "x".repeat(10_000);
        let prompt = render_accumulated_repair_prompt(&[huge.clone()], &["err".to_string()]);
        // The full 10KB should not appear; only a 500-char preview.
        assert!(!prompt.contains(&huge));
        assert!(prompt.contains(&"x".repeat(500)));
    }

    // The gate is the M2 overhead fix: skip the full 12B forward pass on
    // clearly non-substantive turns. The contract is conservative: when in
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
        // 5+ words clears the word ceiling regardless of content: fires.
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
        // "I'm" / "I'll": pronoun check covers contractions too.
        assert!(should_fire_delta("I'm going north", "The path narrows."));
        assert!(should_fire_delta("I'll attack", "You strike."));
    }

    #[test]
    fn gate_skips_short_message_without_pronoun_or_verb_shape() {
        // 3 words, no pronoun, not action-shaped: filler, skip.
        assert!(!should_fire_delta("lol that's funny", "Glad you enjoyed it."));
    }
}
