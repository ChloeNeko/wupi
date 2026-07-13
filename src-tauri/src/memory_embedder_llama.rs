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
//! or hand-rolled request multiplexing (complex + reintroduces mutual blocking —
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
//! This module is unwired scaffolding — it builds and type-checks but nothing
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
/// truncated after tokenization — long inputs silently degrade into garbage
/// embeddings otherwise. Sourced from `bert.context_length` in `Embed.gguf`.
const BERT_CONTEXT_LENGTH: u32 = 512;

/// BERT n_batch. Matched to the context length so a max-length input decodes
/// in a single batch. Same value as the chat engine's `PREFILL_BATCH_TOKENS`.
const BERT_N_BATCH: u32 = 512;

/// Post-tokenization truncation. 511 not 512: leaves headroom in case the
/// tokenizer didn't auto-add [CLS]/[SEP] — defensive, costs nothing.
const BERT_TRUNCATE_TOKENS: usize = 511;

// ---------------------------------------------------------------------------
// Control plane — channel types
// ---------------------------------------------------------------------------

/// A request posted to the embedder thread by [`LlamaCppEmbedder::embed`].
struct EmbedRequest {
    text: String,
    /// One-shot reply channel. Using a separate mpsc (not the Embedder trait's
    /// boxed future) keeps the embedder thread decoupled from async-runtime
    /// types — same shape as `EngineRequest::reply` in `engine.rs`.
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
    Shutdown,
}

// ---------------------------------------------------------------------------
// Handle (held by callers; fully Send + Sync)
// ---------------------------------------------------------------------------

/// The handle callers hold. Fully `Send + Sync` — it's just a channel sender,
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
    /// the embedder as ready — context creation happens on the embedder
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
                    Ok(rt) => {
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
                            // receiver (gave up) — ignore.
                            let _ = req.reply.send(reply_msg);
                        }
                        Ok(EmbedMsg::Shutdown) => {
                            tracing::info!("wupi-embedder shutting down");
                            break;
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

    /// Signal the embedder to shut down. Best-effort — kept for future
    /// hot-swap, not currently called.
    #[allow(dead_code)]
    pub fn shutdown(&self) {
        let _ = self.tx.send(EmbedMsg::Shutdown);
    }

    /// Drain + fail any early requests queued before init failed, then exit.
    /// Mirrors `ChatEngine::drain_failed`.
    fn drain_failed(rx: &mpsc::Receiver<EmbedMsg>, why: String) {
        while let Ok(msg) = rx.recv_timeout(std::time::Duration::from_millis(50)) {
            if let EmbedMsg::Request(req) = msg {
                let _ = req.reply.send(EmbedReply::Err(why.clone()));
            }
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
            "Embed.gguf n_embd={n_embd} but EMBED_DIM={EMBED_DIM} — wrong model file? \
             (expected bge-small-en-v1.5, 384-dim)"
        );
        tracing::info!(n_embd, "embed model loaded");

        // Leak the model to &'static so the context can borrow it for its
        // whole life. Same rationale as llm.rs::into_static — LlamaContext<'a>
        // borrows &'a LlamaModel, and storing both is self-referential.
        // backend is already &'static (from shared_backend).
        let model_ref: &'static LlamaModel = Box::leak(Box::new(model));

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(BERT_CONTEXT_LENGTH))
            .with_n_batch(BERT_N_BATCH)
            // REQUIRED — without this, embeddings_seq_ith returns NotEnabled
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

impl Embedder for LlamaCppEmbedder {
    fn embed(&self, text: String) -> EmbedFuture {
        // Clone the sender so the boxed future is 'static. Routes through
        // `request()` so the channel-send error mapping lives in one place.
        let this_tx = self.tx.clone();
        Box::pin(async move {
            // Fresh oneshot per request — exactly the shape
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

            // Await off the async runtime — embedding takes milliseconds on
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

    fn dim(&self) -> usize {
        EMBED_DIM
    }
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
    /// Tokenize → decode → read CLS embedding → L2-normalize.
    ///
    /// One context is reused across calls — each decode overwrites the
    /// relevant state, so no `clear_kv_cache` is needed between embeds.
    /// (If this proves wrong at runtime — e.g. results drift — adding a
    /// clear is a one-line fix, but the contract says each decode is
    /// independent for embedding extraction.)
    fn embed_one(&mut self, text: &str) -> anyhow::Result<Vec<f32>> {
        // Tokenize WITHOUT BOS. BERT's [CLS]/[SEP] are inserted by the C++
        // tokenizer; a forced BOS prepends an unrelated token that pollutes
        // the embedding. Precedent: the chat engine uses AddBos::Never for
        // literal-only marker tokenization in `ChatEngine::init_runtime`.
        let mut tokens = self
            .model
            .str_to_token(text, AddBos::Never)
            .map_err(|e| anyhow::anyhow!("embed tokenize: {e:?}"))?;

        if tokens.is_empty() {
            anyhow::bail!("tokenized text is empty");
        }

        // Enforce BERT's 512-token limit. Inputs longer than this silently
        // truncate into garbage embeddings at the C++ level; truncating here
        // is the documented, expected behavior.
        tokens.truncate(BERT_TRUNCATE_TOKENS);

        // Build the batch. Mark only the last token as is_last=true — CLS
        // pooling produces one sequence-level embedding regardless of which
        // tokens have logits enabled, but matching the chat prefill pattern
        // is cheapest and correct.
        let n_tokens = tokens.len();
        let mut batch = LlamaBatch::new(n_tokens, 1);
        for (i, tok) in tokens.iter().enumerate() {
            let is_last = i == n_tokens - 1;
            batch
                .add(*tok, i as i32, &[0], is_last)
                .map_err(|e| anyhow::anyhow!("embed batch add: {e:?}"))?;
        }

        self.ctx
            .decode(&mut batch)
            .map_err(|e| anyhow::anyhow!("embed decode: {e:?}"))?;

        // Read the pooled sequence embedding. embeddings_seq_ith borrows from
        // &self — to_vec before returning so the borrow ends with this fn.
        // Failure variants: NotEnabled (forgot with_embeddings(true) — can't
        // happen here), NonePoolType (pooling misconfig).
        let slice = self
            .ctx
            .embeddings_seq_ith(0)
            .map_err(|e| anyhow::anyhow!("read embedding: {e:?}"))?;

        // L2-normalize. bge-small's raw CLS output needs normalization for
        // vec0's cosine MATCH to produce correct rankings (cosine = dot
        // product only when both vectors are unit-length). Doing it here means
        // every stored vector is already unit-length — single responsibility,
        // the embedder owns vector quality.
        let mut vec = slice.to_vec();
        l2_normalize(&mut vec);
        Ok(vec)
    }
}

/// In-place L2 normalization. Empty/zero vectors are left as-is (their norm
/// is 0; dividing would NaN) — callers should not embed empty text, and the
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
// Model discovery — sibling to lib.rs::pick_main_model
// ---------------------------------------------------------------------------

/// Resolve the embed model file from the same search paths as the chat model.
///
/// Locked naming convention (AGENTS.md §2): the embeddings model is ALWAYS
/// `Embed.gguf`. Exact-match (case-insensitive), no size fallback — only one
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
/// Kept here (not in lib.rs) so the embedder is self-contained — its loader
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
    tracing::warn!("no Embed.gguf found — memory engine will fall back to StubEmbedder");
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
        // Regression guard mirroring memory_embedder.rs — if the const drifts
        // from the model's actual dimension, this fails loudly.
        assert_eq!(EMBED_DIM, 384, "Embed.gguf is bge-small-en-v1.5 (384-dim)");
        assert_eq!(BERT_CONTEXT_LENGTH, 512, "bert.context_length from GGUF header");
    }
}
