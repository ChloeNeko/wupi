//! Production `Embedder` backed by `Embed.gguf` via `llama-cpp-2`.
//!
//! This is the real embedding backend for the Memory engine, replacing
//! [`crate::memory_embedder::StubEmbedder`] in production. It mirrors the chat
//! engine's architecture ([`crate::llm::LlamaCppBackend`] + [`crate::engine::ChatEngine`])
//! with BERT-specific deltas: a dedicated `wupi-embedder` thread owns a long-lived
//! `LlamaContext<'static>` configured for embedding extraction (CLS pooling,
//! embeddings enabled, no KV quantization).
//!
//! # Why a separate thread (not shared with `wupi-engine`)
//!
//! The chat context and the embedder context have **incompatible** params:
//!
//! | Param                | Chat (`engine.rs`)     | Embedder (here)             |
//! |----------------------|------------------------|-----------------------------|
//! | `with_embeddings`    | `false`                | `true`                      |
//! | `with_pooling_type`  | unset                  | `Cls`                       |
//! | `with_type_k/v`      | Q8_0                   | default F16                 |
//! | `with_n_ctx`         | 4000                   | 512 (BERT context_length)   |
//!
//! A `LlamaContext` is configured once at creation and cannot be reconfigured.
//! Sharing one thread would force either context recreation per call (expensive)
//! or hand-rolled request multiplexing (complex + reintroduces mutual blocking -
//! generation stalls while embedding runs on the same thread). Two threads +
//! two contexts coordinate only through the shared [`crate::memory::MemoryEngine`]
//! handle in `AppState`; neither blocks the other.
//!
//! # Cross-chat memory is NOT a thread concern
//!
//! Whether chats can see each other's memories is determined by *which SQLite
//! database the MemoryEngine points at*, not by *which thread embeds the query*.
//! One `MemoryEngine` instance in `AppState`, shared by all `chat_send` calls,
//! is what makes memory global. Thread affinity is invisible to that property.
//!
//! # What this does NOT do (Phase 2.6+)
//!
//! This module is unwired scaffolding: it builds and type-checks but nothing
//! calls it. AppState wiring + eager load at startup + the §2F cache-invalidation
//! decision all land in the next phase. Until then the `StubEmbedder` stands in.

use std::path::{Path, PathBuf};
use std::sync::mpsc;

use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};

use crate::llm::shared_backend;
use crate::memory_embedder::{EmbedFuture, Embedder, EMBED_DIM};

/// BERT context length (hard input limit). Inputs longer than this are
/// truncated after tokenization: long inputs silently degrade into garbage
/// embeddings otherwise. Sourced from `bert.context_length` in `Embed.gguf`.
const BERT_CONTEXT_LENGTH: u32 = 512;

/// BERT n_batch. Matched to the context length so a max-length input decodes
/// in a single batch. Same value as the chat engine's `PREFILL_BATCH_TOKENS`.
const BERT_N_BATCH: u32 = 512;

/// Post-tokenization truncation. With `AddBos::Always` (the bug-#4 fix) the
/// tokenizer always inserts `[CLS]` at position 0 and `[SEP]` at the end, so a
/// full-length input is exactly `[CLS] + 510 content + [SEP] = 512`. Truncating
/// to 512 keeps both special tokens (drops the 510th content token); the old
/// 511 value was a stale guard from the pre-fix era that silently dropped
/// `[SEP]` on long inputs. BERT tolerates a missing `[SEP]` but it's a small
/// quality hit on exactly the long inputs that already suffer from no chunking.
const BERT_TRUNCATE_TOKENS: usize = 512;

/// Query instruction for `bge-small-en-v1.5`. This is an ASYMMETRIC retrieval
/// model: per its model card, queries MUST be prefixed with this instruction
/// before embedding; documents (archived memories) are embedded raw.
///
/// Without it, query embeddings collapse toward the document centroid and the
/// cosine range compresses: irrelevant matches score too high to be floored
/// out. Runtime-measured 2026-07-14 (post embedder-fix verification, healthy
/// encoder): querying "gold" against "the weather is nice today" scored cosine
/// 0.53 with no prefix: well above the dense floor, so it would have surfaced
/// as a false positive. The asymmetric prefix is what separates the relevant
/// from the irrelevant. (Real-data calibration later showed the floor belongs
/// at 0.25, not 0.40: see [`crate::memory_rrf::DENSE_COSINE_FLOOR`].)
///
/// The leading space is intentional: bge's instruction is `instruction + " " + text`.
/// The trailing newline in the model-card example is not load-bearing; the
/// tokenizer treats `: ` then text the same as `:\n` then text.
const BGE_QUERY_INSTRUCTION: &str =
    "Represent this sentence for searching relevant passages: ";

// ---------------------------------------------------------------------------
// Control plane: channel types
// ---------------------------------------------------------------------------

/// A request posted to the embedder thread by [`LlamaCppEmbedder::embed`].
struct EmbedRequest {
    text: String,
    /// One-shot reply channel. Using a separate mpsc (not the Embedder trait's
    /// boxed future) keeps the embedder thread decoupled from async-runtime
    /// types: same shape as `EngineRequest::reply` in `engine.rs`.
    reply: mpsc::Sender<EmbedReply>,
}

/// What the embedder sends back when an embed completes.
enum EmbedReply {
    Ok(Vec<f32>),
    Err(String),
}

/// Control messages for the embedder thread's main loop.
enum EmbedMsg {
    Request(Box<EmbedRequest>),
}

// ---------------------------------------------------------------------------
// Handle (held by callers; fully Send + Sync)
// ---------------------------------------------------------------------------

/// The handle callers hold. Fully `Send + Sync`: it's just a channel sender,
/// no `LlamaContext` or `!Send` type crosses out of the embedder thread.
/// Mirrors [`crate::engine::ChatEngine`].
pub struct LlamaCppEmbedder {
    tx: mpsc::Sender<EmbedMsg>,
}

// SAFETY: `mpsc::Sender<EmbedMsg>` is Send (EmbedMsg owns only Send data).
// No `LlamaContext` or `!Send` type crosses out of the thread.
unsafe impl Send for LlamaCppEmbedder {}
unsafe impl Sync for LlamaCppEmbedder {}

impl LlamaCppEmbedder {
    /// Load `Embed.gguf` off-thread, then spawn the persistent embedder.
    /// Returns immediately with a handle; the receiver yields `Ok(())` once
    /// the embedding context is live (or `Err` if init failed).
    ///
    /// The caller MUST `recv()` from the returned receiver before treating
    /// the embedder as ready: context creation happens on the embedder
    /// thread, and if it fails the caller must not report "ready". Same
    /// readiness contract as `ChatEngine::spawn` (Bug #6).
    ///
    /// `n_gpu_layers` controls GPU offload. Embeddings are cheap to compute;
    /// a small model like bge-small (33M params) fits entirely in VRAM, so
    /// passing a large value (e.g. 9999) is correct and means every layer
    /// runs on CUDA.
    pub fn spawn_load(
        path: PathBuf,
        n_gpu_layers: u32,
    ) -> (Self, mpsc::Receiver<Result<(), String>>) {
        let (tx, rx) = mpsc::channel::<EmbedMsg>();
        let (init_tx, init_rx) = mpsc::channel::<Result<(), String>>();

        std::thread::Builder::new()
            .name("wupi-embedder".into())
            .spawn(move || {
                let mut runtime = match Self::init_runtime(&path, n_gpu_layers) {
                    Ok(mut rt) => {
                        // Self-test BEFORE signaling ready: the probe runs on the
                        // embedder thread (owns ctx + model), computes cosine in
                        // Rust (bypasses vec0), and logs the result. If the
                        // embedder is broken, we want to know at startup, not
                        // after the first chat turn produces garbage retrieval.
                        // Runs once, ~ms on GPU, never blocks the readiness
                        // signal long (the probe is 4 short embeds).
                        run_self_test(&mut rt);
                        let _ = init_tx.send(Ok(()));
                        rt
                    }
                    Err(e) => {
                        let msg = format!("embedder init failed: {e}");
                        tracing::error!(error = %msg, "embedder init failed; thread exiting");
                        let _ = init_tx.send(Err(msg.clone()));
                        Self::drain_failed(&rx, msg);
                        return;
                    }
                };
                tracing::info!(
                    path = %path.display(),
                    "wupi-embedder thread ready"
                );

                loop {
                    match rx.recv() {
                        Ok(EmbedMsg::Request(req)) => {
                            // Self-healing: isolate each embed so one panic
                            // doesn't kill the thread. Mirrors the chat
                            // engine's `catch_unwind` in `ChatEngine::spawn`.
                            let outcome = std::panic::catch_unwind(
                                std::panic::AssertUnwindSafe(|| {
                                    runtime.embed_one(&req.text)
                                }),
                            );
                            let reply_msg = match outcome {
                                Ok(Ok(vec)) => EmbedReply::Ok(vec),
                                Ok(Err(e)) => {
                                    tracing::warn!(error = %e, "embed failed");
                                    EmbedReply::Err(format!("{e:#}"))
                                }
                                Err(payload) => {
                                    let msg = payload
                                        .downcast_ref::<String>()
                                        .map(|s| s.clone())
                                        .or_else(|| {
                                            payload
                                                .downcast_ref::<&str>()
                                                .map(|s| s.to_string())
                                        })
                                        .unwrap_or_else(|| {
                                            "embedder panic (unknown cause)".to_string()
                                        });
                                    tracing::error!(panic = %msg, "embed panicked");
                                    EmbedReply::Err(format!("embedder panic: {msg}"))
                                }
                            };
                            // Send can fail only if the caller dropped the
                            // receiver (gave up): ignore.
                            let _ = req.reply.send(reply_msg);
                        }
                        Err(mpsc::RecvError) => {
                            tracing::info!(
                                "wupi-embedder: all senders dropped, exiting"
                            );
                            break;
                        }
                    }
                }
            })
            .expect("failed to spawn wupi-embedder thread");

        (LlamaCppEmbedder { tx }, init_rx)
    }

    /// Drain + fail any early requests queued before init failed, then exit.
    /// Mirrors `ChatEngine::drain_failed`.
    fn drain_failed(rx: &mpsc::Receiver<EmbedMsg>, why: String) {
        while let Ok(EmbedMsg::Request(req)) = rx.recv_timeout(std::time::Duration::from_millis(50)) {
            let _ = req.reply.send(EmbedReply::Err(why.clone()));
        }
    }

    /// Initialize the embedder runtime: load the model, leak it to `&'static`,
    /// create the embedding context. Runs on the embedder thread.
    fn init_runtime(path: &Path, n_gpu_layers: u32) -> anyhow::Result<EmbedderRuntime> {
        let backend = shared_backend();

        let params = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers);
        let model = LlamaModel::load_from_file(backend, path, &params)
            .map_err(|e| anyhow::anyhow!("embed model load: {e:?}"))?;

        // Sanity check: the loaded model's embedding width MUST match EMBED_DIM.
        // A wrong-file-loaded mistake (e.g. someone dropped a 768-dim model in
        // as Embed.gguf) fails HERE with a clear message, not later as a
        // confusing vec0 insert crash.
        let n_embd = usize::try_from(model.n_embd())
            .map_err(|e| anyhow::anyhow!("n_embd doesn't fit usize: {e}"))?;
        anyhow::ensure!(
            n_embd == EMBED_DIM,
            "Embed.gguf n_embd={n_embd} but EMBED_DIM={EMBED_DIM}: wrong model file? \
             (expected bge-small-en-v1.5, 384-dim)"
        );
        tracing::info!(n_embd, "embed model loaded");

        // Leak the model to &'static so the context can borrow it for its
        // whole life. Same rationale as llm.rs::into_static: LlamaContext<'a>
        // borrows &'a LlamaModel, and storing both is self-referential.
        // backend is already &'static (from shared_backend).
        let model_ref: &'static LlamaModel = Box::leak(Box::new(model));

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(BERT_CONTEXT_LENGTH))
            .with_n_batch(BERT_N_BATCH)
            // REQUIRED: without this, embeddings_seq_ith returns NotEnabled
            // and every embed fails.
            .with_embeddings(true)
            // bge-small-en-v1.5 is designed for CLS pooling (pooling_type=2
            // in its GGUF header). CLS = take the [CLS] token's representation
            // as the sequence embedding. The alternative (Mean) averages all
            // token positions and gives different (worse) rankings on bge.
            .with_pooling_type(LlamaPoolingType::Cls);
        // NO with_type_k / with_type_v: KV quantization degrades encoder
        // activations and embeddings aren't cached between calls anyway. F16
        // default is correct.

        let ctx = model_ref
            .new_context(backend, ctx_params)
            .map_err(|e| anyhow::anyhow!("embed context init: {e:?}"))?;
        tracing::info!(
            n_ctx = BERT_CONTEXT_LENGTH,
            pooling = "Cls",
            "embed context created"
        );

        Ok(EmbedderRuntime {
            ctx,
            model: model_ref,
        })
    }
}

/// Embedder self-test (2026-07-14). Embeds three known-similarity string pairs
/// via the real embedder and logs their Rust-computed cosine. Isolates "the
/// embedder produces bad vectors" from "vec0 storage is broken".
///
/// Uses the ASYMMETRIC retrieval path the data plane actually uses: the probe
/// is embedded AS A QUERY (with [`BGE_QUERY_INSTRUCTION`]) and each target is
/// embedded AS A DOCUMENT (raw). This is what the dense cosine floor will see
/// in production, so the numbers here are the calibration reference for
/// [`crate::memory_rrf::DENSE_COSINE_FLOOR`]: not a theoretical doc-doc score.
///
/// Expected ordering for a healthy bge-small-en-v1.5 with the query prefix:
///   - "gold" vs "gold is a precious metal"     → HIGH   (0.6-0.9)
///   - "gold" vs "silver is a precious metal"   → MEDIUM (0.4-0.7)
///   - "gold" vs "the weather is nice today"    → LOW    (~0.40-0.45: synthetic
///     worst case; real multi-topic data separates far more cleanly, with
///     irrelevant matches landing ≤0.10: see DENSE_COSINE_FLOOR's doc)
///
/// If all three are ~0.05 (random unit vectors), the EMBEDDER is broken and
/// vec0 is exonerated. If they make sense here but the 🧠 panel shows garbage,
/// vec0 storage is the suspect. Runs ONCE at startup; costs 4 embeds (~ms).
fn run_self_test(runtime: &mut EmbedderRuntime) {
    let pairs: &[(&str, &str, &str)] = &[
        ("gold", "gold is a precious metal", "HIGH"),
        ("gold", "silver is a precious metal", "MEDIUM"),
        ("gold", "the weather is nice today", "LOW"),
    ];
    // Embed "gold" AS A QUERY: this is the path search() takes, prefix and all.
    // The targets are embedded AS DOCUMENTS (raw): the archival path. Matching
    // the data plane's asymmetry is what makes these numbers the floor's true
    // calibration reference.
    let probe = match runtime.embed_one(&format!("{BGE_QUERY_INSTRUCTION}gold")) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(error = %format!("{e:#}"), "self-test: failed to embed probe 'gold'");
            return;
        }
    };
    for &(q, target, label) in pairs {
        // q is always "gold" in this set, but keep the pattern general.
        let _ = q;
        match runtime.embed_one(target) {
            Ok(v) => {
                let cos = cosine(&probe, &v);
                tracing::info!(
                    target,
                    expected = label,
                    cosine = %format!("{cos:.4}"),
                    "embedder self-test pair"
                );
            }
            Err(e) => {
                tracing::warn!(error = %format!("{e:#}"), "self-test: failed to embed '{target}'");
            }
        }
    }
}

impl Embedder for LlamaCppEmbedder {
    fn embed(&self, text: String) -> EmbedFuture {
        // Clone the sender so the boxed future is 'static. Routes through
        // `request()` so the channel-send error mapping lives in one place.
        let this_tx = self.tx.clone();
        Box::pin(async move {
            // Fresh oneshot per request: exactly the shape
            // `LlamaCppBackend::stream` uses to await an engine reply.
            let (reply_tx, reply_rx) = mpsc::channel::<EmbedReply>();
            let req = EmbedRequest {
                text,
                reply: reply_tx,
            };
            // Send via the same path request() uses. We inline the send here
            // (rather than call a &self method) because the future owns the
            // cloned sender and has no `&self` to borrow.
            this_tx
                .send(EmbedMsg::Request(Box::new(req)))
                .map_err(|_| anyhow::anyhow!("embedder thread closed"))?;

            // Await off the async runtime: embedding takes milliseconds on
            // GPU but we must not block a tokio worker for that duration.
            let reply = tokio::task::spawn_blocking(move || reply_rx.recv())
                .await
                .map_err(|e| anyhow::anyhow!("embed join: {e}"))?
                .map_err(|_| anyhow::anyhow!("embedder reply channel closed"))?;

            match reply {
                EmbedReply::Ok(vec) => Ok(vec),
                EmbedReply::Err(msg) => Err(anyhow::anyhow!(msg)),
            }
        })
    }

    /// bge-small is asymmetric: queries get the [`BGE_QUERY_INSTRUCTION`] prefix,
    /// documents are embedded raw (via [`embed`](Embedder::embed)). Without the
    /// prefix the cosine range compresses and irrelevant matches clear the dense
    /// floor: see the const's doc for the measured failure.
    fn embed_query(&self, text: String) -> EmbedFuture {
        let prefixed = format!("{BGE_QUERY_INSTRUCTION}{text}");
        self.embed(prefixed)
    }

    fn dim(&self) -> usize {
        EMBED_DIM
    }
}

/// Compute cosine similarity directly in Rust, bypassing vec0. Used ONLY by
/// the startup self-test diagnostic to isolate "embedder produces bad vectors"
/// from "vec0 stores/retrieves vectors wrong". If two obviously-similar texts
/// score high here but low through the 🧠 panel, the bug is in storage; if they
/// score low here too, the embeddings themselves are broken.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag_a <= 0.0 || mag_b <= 0.0 {
        return 0.0;
    }
    dot / (mag_a * mag_b)
}

// ---------------------------------------------------------------------------
// Runtime (owned by the embedder thread; never crosses thread boundaries)
// ---------------------------------------------------------------------------

/// The mutable runtime state owned by the embedder thread. Lives for the
/// thread's lifetime; never sent across a thread boundary (LlamaContext is
/// `!Send`). Mirrors `engine.rs::EngineRuntime`.
struct EmbedderRuntime {
    ctx: llama_cpp_2::context::LlamaContext<'static>,
    model: &'static LlamaModel,
}

impl EmbedderRuntime {
    /// Tokenize → encode → read CLS embedding → L2-normalize.
    ///
    /// # The encode vs decode distinction (2026-07-14 bug fix)
    ///
    /// `bge-small-en-v1.5` is a BERT encoder: bidirectional (non-causal).
    /// llama.cpp has two entry points:
    /// - `llama_decode`: causal mask (each token attends only to EARLIER
    ///   tokens). For autoregressive chat models (Gemma, Llama).
    /// - `llama_encode`: full bidirectional mask (every token attends to
    ///   every other token). For encoder/embedding models (BERT).
    ///
    /// The earlier revision called `ctx.decode(...)`. That's wrong for BERT.
    /// With a causal mask, position 0 ([CLS]) attends ONLY to itself: it
    /// never sees the rest of the sequence. Its hidden state is nearly
    /// context-free. The result: healthy-looking L2 norm (~9) but inverted /
    /// length-dependent semantics: short inputs scored high cosine to
    /// everything (CLS self-attention dominated), long inputs scored near
    /// zero (no bidirectional context to build meaning). Runtime-observed:
    /// "continue" scored cos 0.969 to "cow"; "Write me a short story about
    /// cows" scored cos 0.008 to "cow". Inverted and length-correlated -
    /// the signature of causal attention on a bidirectional model.
    ///
    /// The fix: `ctx.encode(...)`. The C++ `encode()` path also clears the
    /// pooled-embedding buffer (`embd_seq.clear()`, llama-context.cpp:1376)
    /// at the start of every call, so no manual KV/embd clear is needed
    /// between embeds: the encoder is self-cleaning.
    ///
    /// # The logits flag (also fixed 2026-07-14)
    ///
    /// `batch.add(..., logits=true)` for EVERY position: the pooling layer
    /// reads position 0's hidden state from the output buffer, so every
    /// position's embedding must be stored. The chat engine's `is_last`
    /// optimization (logits only on the final token) is wrong for encoders.
    fn embed_one(&mut self, text: &str) -> anyhow::Result<Vec<f32>> {
        // Tokenize WITH special tokens. Despite the misleading name, the
        // `AddBos` enum in llama-cpp-2 maps to the C API's `add_special`
        // parameter (see llama-cpp-2's str_to_token: it passes add_bos as
        // the 6th arg to llama_tokenize, which llama.h documents as
        // `add_special`). For BERT, `add_special=true` is what inserts
        // [CLS] at position 0 and [SEP] at the end: WITHOUT them, CLS
        // pooling reads position 0 = the first content token, producing
        // embeddings with healthy magnitude but inverted/garbled semantics
        // (runtime-observed 2026-07-14: "gold" scored HIGHER cosine to
        // "the weather is nice today" than to "gold is a precious metal").
        // The earlier "AddBos::Never" comment was wrong: it confused this
        // flag with the chat engine's Gemma BOS, which is a different
        // mechanism entirely.
        let mut tokens = self
            .model
            .str_to_token(text, AddBos::Always)
            .map_err(|e| anyhow::anyhow!("embed tokenize: {e:?}"))?;

        if tokens.is_empty() {
            anyhow::bail!("tokenized text is empty");
        }

        // Enforce BERT's 512-token limit. Inputs longer than this silently
        // truncate into garbage embeddings at the C++ level; truncating here
        // is the documented, expected behavior.
        //
        // Visibility (Phase 1 chunking): when the input exceeds the budget,
        // log at debug so silent truncation becomes observable in the live
        // exe's tracing output. The caller (`add_memory`) is supposed to chunk
        // first (see `memory::chunk_text`) so this should never fire on the
        // archival path post-chunking: but queries, codex entries, or future
        // callers can still exceed it. If this fires frequently, the chunk
        // budget may need lowering OR a caller is bypassing chunking.
        let pre_truncate_len = tokens.len();
        if pre_truncate_len > BERT_TRUNCATE_TOKENS {
            tracing::debug!(
                pre_truncate_tokens = pre_truncate_len,
                budget_tokens = BERT_TRUNCATE_TOKENS,
                "embedder input exceeds BERT context window; truncating (caller should chunk first)"
            );
        }
        tokens.truncate(BERT_TRUNCATE_TOKENS);

        // Build the batch. logits=true for EVERY position: the pooling layer
        // (CLS) reads position 0's hidden state from the output buffer, so
        // every position's embedding must be stored. The chat engine's
        // `is_last` optimization (logits only on the final token) is WRONG
        // for encoder/embedding models.
        let n_tokens = tokens.len();
        let mut batch = LlamaBatch::new(n_tokens, 1);
        for (i, tok) in tokens.iter().enumerate() {
            batch
                .add(*tok, i as i32, &[0], true)
                .map_err(|e| anyhow::anyhow!("embed batch add: {e:?}"))?;
        }

        // ENCODE, not decode. bge-small is a BERT encoder (bidirectional).
        // decode() applies a causal mask and produces semantically wrong
        // embeddings for non-causal models. See the method doc for the full
        // reasoning + the runtime signature that exposed it.
        self.ctx
            .encode(&mut batch)
            .map_err(|e| anyhow::anyhow!("embed encode: {e:?}"))?;

        // Read the pooled sequence embedding. embeddings_seq_ith borrows from
        // &self: to_vec before returning so the borrow ends with this fn.
        // Failure variants: NotEnabled (forgot with_embeddings(true): can't
        // happen here), NonePoolType (pooling misconfig).
        let slice = self
            .ctx
            .embeddings_seq_ith(0)
            .map_err(|e| anyhow::anyhow!("read embedding: {e:?}"))?;

        // L2-normalize. bge-small's raw CLS output needs normalization for
        // vec0's cosine MATCH to produce correct rankings (cosine = dot
        // product only when both vectors are unit-length). Doing it here means
        // every stored vector is already unit-length: single responsibility,
        // the embedder owns vector quality.
        let mut vec = slice.to_vec();
        // Diagnostic: log the pre-normalization L2 norm on the first embed
        // only. A healthy bge-small embedding has norm ~5-20; if it's ~0 or
        // NaN, the embedder is still broken. One-shot via a static flag so
        // the log isn't spammed on every embed.
        static NORM_LOGGED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        if !NORM_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            let pre_norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            tracing::info!(
                pre_norm,
                dim = vec.len(),
                first_3 = ?&vec[..3.min(vec.len())],
                "embedder diagnostic: first embed pre-normalization via encode() (expect norm ~5-20 for healthy bge-small)"
            );
        }
        l2_normalize(&mut vec);
        Ok(vec)
    }
}

/// In-place L2 normalization. Empty/zero vectors are left as-is (their norm
/// is 0; dividing would NaN): callers should not embed empty text, and the
/// `tokenized text is empty` guard above prevents it.
fn l2_normalize(v: &mut [f32]) {
    let sum_sq: f32 = v.iter().map(|x| x * x).sum();
    if sum_sq <= 0.0 {
        return;
    }
    let norm = sum_sq.sqrt();
    for x in v.iter_mut() {
        *x /= norm;
    }
}

// ---------------------------------------------------------------------------
// Model discovery: sibling to lib.rs::pick_main_model
// ---------------------------------------------------------------------------

/// Resolve the embed model file from the same search paths as the chat model.
///
/// Locked naming convention (AGENTS.md §2): the embeddings model is ALWAYS
/// `Embed.gguf`. Exact-match (case-insensitive), no size fallback: only one
/// file will ever have that name. Returns `None` if no embed model is present,
/// in which case Memory should fall back to `StubEmbedder` (graceful
/// degradation, not a crash).
pub fn pick_embed_model(dir: &Path) -> Option<PathBuf> {
    std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .find(|e| {
            e.path().extension().and_then(|x| x.to_str()) == Some("gguf")
                && e.file_name().to_string_lossy().to_lowercase() == "embed.gguf"
        })
        .map(|e| e.path())
}

/// Walk the same candidate dirs `resolve_model_path` uses for the chat model.
/// Kept here (not in lib.rs) so the embedder is self-contained: its loader
/// lives with its discovery. Callers (the future AppState wiring) pass the
/// list of dirs to check.
pub fn resolve_embed_model(dirs: &[PathBuf]) -> Option<PathBuf> {
    for dir in dirs {
        if dir.exists() {
            if let Some(p) = pick_embed_model(dir) {
                tracing::info!("resolved embed model: {}", p.display());
                return Some(p);
            }
        }
    }
    tracing::warn!("no Embed.gguf found: memory engine will fall back to StubEmbedder");
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_normalize_unit_vector() {
        let mut v = vec![3.0, 4.0]; // norm = 5
        l2_normalize(&mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "expected unit norm, got {norm}");
        // Direction preserved: 3/5, 4/5
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn l2_normalize_zero_vector_is_noop() {
        let mut v = vec![0.0f32, 0.0, 0.0];
        l2_normalize(&mut v);
        // Must NOT NaN; should stay zero.
        assert!(v.iter().all(|x| *x == 0.0));
    }

    #[test]
    fn l2_normalize_preserves_dimension() {
        let mut v = vec![1.0f32; 384];
        l2_normalize(&mut v);
        assert_eq!(v.len(), 384);
    }

    #[test]
    fn embed_dim_const_matches_gguf_header() {
        // Regression guard mirroring memory_embedder.rs: if the const drifts
        // from the model's actual dimension, this fails loudly.
        assert_eq!(EMBED_DIM, 384, "Embed.gguf is bge-small-en-v1.5 (384-dim)");
        assert_eq!(BERT_CONTEXT_LENGTH, 512, "bert.context_length from GGUF header");
    }
}
