//! The persistent chat engine — a dedicated generation thread.
//!
//! `LlamaContext` is `!Send + !Sync` (it owns a raw `NonNull` pointer into
//! llama.cpp's heap) and it borrows `&'a LlamaModel` for its whole life. Both
//! facts rule out storing it behind a cross-thread mutex in `AppState`. So the
//! context lives on **one dedicated thread** for the entire OS session, and
//! `chat_send` posts work to it over a channel.
//!
//! This is the stateful half of the Prime Directive §1 split:
//! the engine (stateful state machine) vs the stream filters (stateless
//! processors, already correct in `chat_format.rs` / `stream_filter.rs`).
//!
//! # What lives on the engine thread
//!
//! - The leaked `&'static LlamaModel` (leaked at load time — see `llm.rs` for
//!   why this is the idiomatic choice for a process-lifetime singleton).
//! - A single persistent `LlamaContext<'static>` with **Q8_0 KV cache** on
//!   both keys and values (~50% VRAM cut vs F16, near-zero quality loss).
//! - A `KvBuffer` tracking the token IDs currently resident in the cache, so
//!   each turn only prefills the **delta** since the last turn.
//!
//! # Self-healing
//!
//! The main loop wraps each request in `catch_unwind`. If a single generation
//! panics (bad token, OOM spike, model quirk), the thread survives, the KV
//! buffer is reset to cold, and the next request starts fresh. One bad turn
//! must not kill the engine.

use crate::chat_format::{ChatFormat, ModelFamily, ParsedOutput, ToolSpec};
use crate::kv_buffer::{scan_turn_boundaries, truncate_to_fit, KvBuffer};
use crate::session::ApiMessage;
use crate::llm::ChunkFn;
use llama_cpp_2::context::params::{KvCacheType, LlamaContextParams};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Duration;

/// A request posted to the engine thread by `chat_send`.
pub struct EngineRequest {
    pub messages: Vec<ApiMessage>,
    pub on_chunk: ChunkFn,
    /// Cancellation flag. Set to true by `chat_stop` to break the decode loop
    /// at the next token boundary. The engine checks this between tokens —
    /// never mid-decode — so the KV cache stays in a consistent state.
    pub cancel: Arc<AtomicBool>,
    /// One-shot reply: the engine fills this with the generation result (or an
    /// error). Using a separate channel (not the Tauri Channel) keeps the
    /// engine decoupled from Tauri's IPC types and lets `stream()` await it.
    pub reply: std::sync::mpsc::Sender<EngineReply>,
}

/// What the engine sends back when a generation completes.
pub enum EngineReply {
    Ok(ParsedOutput),
    Err(String),
}

/// Control messages for the engine thread's main loop.
enum EngineMsg {
    Request(Box<EngineRequest>),
    Shutdown,
}

/// The handle held by `AppState` (via `LlamaCppBackend`). Fully `Send` — it's
/// just a channel sender + a marker that the engine started OK.
pub struct ChatEngine {
    tx: mpsc::Sender<EngineMsg>,
    /// The model family, retained so the backend can report it without
    /// crossing thread boundaries to read the model.
    family: ModelFamily,
}

// SAFETY: mpsc::Sender<EngineMsg> is Send (EngineMsg owns only Send data).
// ModelFamily is Copy+Send. No `LlamaContext` or `!Send` type crosses out.
unsafe impl Send for ChatEngine {}
unsafe impl Sync for ChatEngine {}

impl ChatEngine {
    /// Spawn the engine thread. The caller has already loaded the model and
    /// leaked it to `&'static` (see `llm.rs::spawn_engine`). We take the
    /// static model ref + backend ref and own them for the thread's life.
    ///
    /// Returns the engine handle AND a receiver that yields `Ok(())` once
    /// the persistent context has been created (or `Err` if init failed).
    /// The caller MUST `recv()` from it before treating the engine as ready
    /// — context creation happens on the engine thread, and if it fails the
    /// UI must not report "ready" (Bug #6).
    pub fn spawn(
        backend: &'static llama_cpp_2::llama_backend::LlamaBackend,
        model: &'static LlamaModel,
        family: ModelFamily,
        context_size: u32,
    ) -> (Self, std::sync::mpsc::Receiver<Result<(), String>>) {
        let (tx, rx) = mpsc::channel::<EngineMsg>();
        let (init_tx, init_rx) = std::sync::mpsc::channel::<Result<(), String>>();

        std::thread::Builder::new()
            .name("wupi-engine".into())
            .spawn(move || {
                let mut engine = match Self::init_runtime(backend, model, family, context_size) {
                    Ok(rt) => {
                        // Signal readiness BEFORE entering the main loop so the
                        // caller doesn't report "ready" until the context is live.
                        let _ = init_tx.send(Ok(()));
                        rt
                    }
                    Err(e) => {
                        // Context init failed. Report it, then drain + fail
                        // any early requests before exiting the thread.
                        let msg = format!("engine init failed: {e}");
                        tracing::error!(error = %msg, "engine init failed; thread exiting");
                        let _ = init_tx.send(Err(msg.clone()));
                        Self::drain_failed(&rx, msg);
                        return;
                    }
                };
                tracing::info!("wupi-engine thread ready (family={:?})", family);

                loop {
                    match rx.recv() {
                        Ok(EngineMsg::Request(req)) => {
                            // Destructure once so the closure can move the
                            // generation inputs (messages, on_chunk, cancel)
                            // while `reply` stays borrowable afterward. Avoids
                            // E0382 move-then-borrow.
                            let EngineRequest {
                                messages,
                                on_chunk,
                                cancel,
                                reply,
                            } = *req;

                            // Self-healing: isolate each generation so one
                            // panic doesn't kill the thread.
                            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                                || engine.generate(messages, &on_chunk, &cancel),
                            ));
                            let reply_msg = match outcome {
                                Ok(Ok(parsed)) => EngineReply::Ok(parsed),
                                Ok(Err(e)) => {
                                    // Bug B fix (2026-07-12): clear the LIVE KV
                                    // cache too, not just the bookkeeping. The
                                    // old code only called buffer.reset(), which
                                    // left stale tokens in the llama.cpp cache
                                    // while token_log reported cold → next send's
                                    // cold-path prefill wrote over stale state →
                                    // the "model goes offline" infinite failure.
                                    tracing::warn!(error = %e, "generation failed; cold-resetting KV cache + buffer");
                                    engine.ctx.clear_kv_cache();
                                    engine.buffer.reset();
                                    EngineReply::Err(format!("{e:#}"))
                                }
                                Err(panic_payload) => {
                                    // catch_unwind captured the panic. Log it,
                                    // reset the buffer (cache state may be
                                    // inconsistent), and report a clean error.
                                    let msg = panic_payload
                                        .downcast_ref::<String>()
                                        .map(|s| s.clone())
                                        .or_else(|| {
                                            panic_payload.downcast_ref::<&str>().map(|s| s.to_string())
                                        })
                                        .unwrap_or_else(|| "engine panic (unknown cause)".to_string());
                                    tracing::error!(panic = %msg, "engine generation panicked; cold-resetting");
                                    engine.ctx.clear_kv_cache();
                                    engine.buffer.reset();
                                    EngineReply::Err(format!("engine panic: {msg}"))
                                }
                            };
                            // Sending can fail only if the caller gave up — ignore.
                            let _ = reply.send(reply_msg);
                        }
                        Ok(EngineMsg::Shutdown) => {
                            tracing::info!("wupi-engine shutting down");
                            break;
                        }
                        Err(mpsc::RecvError) => {
                            // All senders dropped — shut down.
                            tracing::info!("wupi-engine: all senders dropped, exiting");
                            break;
                        }
                    }
                }
            })
            .expect("failed to spawn wupi-engine thread");

        (ChatEngine { tx, family }, init_rx)
    }

    /// Post a generation request and return a receiver the caller awaits.
    pub fn request(&self, req: EngineRequest) -> Result<(), String> {
        self.tx
            .send(EngineMsg::Request(Box::new(req)))
            .map_err(|_| "engine thread closed".to_string())
    }

    /// Signal the engine to shut down. Best-effort — the thread exits on next
    /// recv. Not currently called (the engine lives for the process), but
    /// kept for future hot-swap / settings-reload flows.
    pub fn shutdown(&self) {
        let _ = self.tx.send(EngineMsg::Shutdown);
    }

    /// The model family this engine was built for.
    pub fn family(&self) -> ModelFamily {
        self.family
    }

    fn drain_failed(rx: &mpsc::Receiver<EngineMsg>, why: String) {
        // Give late callers a moment to queue, then fail them all.
        while let Ok(msg) = rx.recv_timeout(Duration::from_millis(50)) {
            if let EngineMsg::Request(req) = msg {
                let _ = req.reply.send(EngineReply::Err(why.clone()));
            }
        }
    }

    /// Initialize the persistent runtime: the LlamaContext (Q8_0 KV cache) +
    /// the token-ID log + the cached `<|turn>` marker tokens.
    fn init_runtime(
        backend: &'static llama_cpp_2::llama_backend::LlamaBackend,
        model: &'static LlamaModel,
        family: ModelFamily,
        context_size: u32,
    ) -> anyhow::Result<EngineRuntime> {
        let n_ctx = context_size.max(1024);
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(n_ctx))
            .with_n_batch(512)
            .with_embeddings(false)
            // Asymmetric-ish KV quantization: Q8_0 on both K and V is the
            // community-standard "free win" — ~50% VRAM cut vs F16, near-zero
            // quality loss on Gemma. K=Q8_0 keeps attention routing precise;
            // V=Q8_0 avoids the long-context drift that Q4_0 V would cause.
            .with_type_k(KvCacheType::Q8_0)
            .with_type_v(KvCacheType::Q8_0);
        let ctx = model
            .new_context(backend, ctx_params)
            .map_err(|e| anyhow::anyhow!("context init: {e:?}"))?;
        tracing::info!(n_ctx, kv_k = ?KvCacheType::Q8_0, kv_v = ?KvCacheType::Q8_0, "persistent context created");

        // Cache the `<|turn>` marker tokens for boundary scanning during
        // eviction. Tokenize the literal once (no BOS). For the Plain family
        // there's no `<|turn>` marker, so eviction falls back to the
        // last-boundary path — fine, since Plain is a fallback.
        let turn_marker = family
            .turn_marker_literal()
            .and_then(|lit| model.str_to_token(lit, AddBos::Never).ok())
            .unwrap_or_default();

        Ok(EngineRuntime {
            ctx,
            model,
            formatter: family.formatter(),
            buffer: KvBuffer::new(),
            turn_marker,
            n_ctx,
        })
    }
}

/// The mutable runtime state owned by the engine thread. Lives for the
/// thread's lifetime; never crosses a thread boundary.
struct EngineRuntime {
    ctx: LlamaContext<'static>,
    model: &'static LlamaModel,
    formatter: Box<dyn ChatFormat>,
    buffer: KvBuffer,
    /// Tokenized `<|turn>` literal (empty for families without one).
    turn_marker: Vec<LlamaToken>,
    n_ctx: u32,
}

/// Telemetry captured during the decode loop, returned to `generate()` for the
/// structured ENGINE PERFORMANCE TELEMETRY block.
struct DecodeTelemetry {
    parsed: ParsedOutput,
    tokens_generated: usize,
    /// Time from the start of the decode loop to the first token sampled.
    /// `None` if no tokens were generated (e.g. immediate EOS or cancel).
    time_to_first_token: Option<Duration>,
    /// Wall-clock time of the entire decode loop (first sample → loop exit).
    generation_elapsed: Duration,
    /// True if the loop broke because the `cancel` flag was set.
    cancelled: bool,
}

impl EngineRuntime {
    /// Run one generation: render → tokenize → delta-prefill → decode loop →
    /// append generated tokens to the log. Captures telemetry for the
    /// structured performance block emitted at the end.
    fn generate(
        &mut self,
        messages: Vec<ApiMessage>,
        on_chunk: &ChunkFn,
        cancel: &Arc<AtomicBool>,
    ) -> anyhow::Result<ParsedOutput> {
        // Split system vs conversation (same logic as the old generate_blocking).
        let mut system = String::new();
        let mut conv: Vec<ApiMessage> = Vec::with_capacity(messages.len());
        for m in &messages {
            if m.role == "system" && system.is_empty() {
                system.push_str(&m.content);
            } else {
                conv.push(m.clone());
            }
        }
        let tools: Vec<ToolSpec> = Vec::new();
        let prompt = self.formatter.render_prompt(&system, &conv, &tools, true);
        tracing::debug!(prompt_len = prompt.len(), "rendered prompt");

        let full_tokens = self
            .model
            .str_to_token(&prompt, AddBos::Always)
            .map_err(|e| anyhow::anyhow!("tokenize: {e:?}"))?;
        if full_tokens.is_empty() {
            anyhow::bail!("tokenized prompt is empty");
        }

        // --- Prompt truncation (Bug A fix, 2026-07-12): the OLD path tried
        // to evict from the KV cache via reconstruction, but the next turn's
        // rendered prompt still contained the FULL history → common_prefix_len
        // diverged right after the system prefix → the delta was ~the entire
        // conversation → NoKvCacheSlot. The fix inverts the relationship:
        // truncate the PROMPT to fit the cache, instead of rebuilding the
        // cache to match an ever-growing prompt. After this, full_tokens is
        // guaranteed to fit within (n_ctx - generation_reserve). ---
        let generation_reserve = (self.n_ctx / 4).max(512) as usize;
        let max_prompt_len = (self.n_ctx as usize).saturating_sub(generation_reserve);
        let full_tokens = if full_tokens.len() > max_prompt_len {
            if self.turn_marker.is_empty() {
                // Plain family has no turn markers — can't truncate safely.
                anyhow::bail!(
                    "context too long: {} tokens, max {}, and no truncation strategy for this model family",
                    full_tokens.len(), max_prompt_len
                );
            }
            let boundaries = scan_turn_boundaries(&full_tokens, &self.turn_marker);
            let system_prefix_len = self.estimate_system_prefix_len(&full_tokens);
            match truncate_to_fit(&full_tokens, max_prompt_len, system_prefix_len, &boundaries) {
                Some(truncated) => {
                    tracing::info!(
                        before = full_tokens.len(),
                        after = truncated.len(),
                        max = max_prompt_len,
                        dropped_turns = full_tokens.len().saturating_sub(truncated.len()),
                        "truncated prompt to fit context window"
                    );
                    truncated
                }
                None => anyhow::bail!(
                    "context too long even after truncation: {} tokens, system prefix {}, max {}",
                    full_tokens.len(), system_prefix_len, max_prompt_len
                ),
            }
        } else {
            full_tokens
        };

        // --- Delta prefill (the speedup): only decode tokens not already in
        // the cache. Track cached vs. prefilled for telemetry. ---
        let prefill_start = std::time::Instant::now();
        let (cached_tokens, prefilled_tokens) = if self.buffer.is_cold() {
            // Cold start — nothing was cached. Prefill everything.
            self.prefill(&full_tokens, 0)?;
            let system_prefix_len = self.estimate_system_prefix_len(&full_tokens);
            self.buffer.commit_cold(&full_tokens, system_prefix_len);
            (0usize, full_tokens.len())
        } else {
            let common = self.buffer.common_prefix_len(&full_tokens);
            let delta = &full_tokens[common..];
            if !delta.is_empty() {
                self.prefill(delta, common as i32)?;
            }
            self.buffer.commit_delta(common, delta);
            (common, delta.len())
        };
        let prefill_elapsed = prefill_start.elapsed();

        // --- Space guard (Bug #1 Part A, retained): if the cache is so full
        // after prefilling the delta that even a minimum generation window
        // won't fit, force a cold reset (clear live KV + bookkeeping) and
        // re-prefill the truncated prompt from scratch. This is safe now
        // because truncation guarantees `full_tokens.len() <= max_prompt_len`,
        // so a cold re-prefill always fits. The OLD reconstruct-based eviction
        // is gone — it was the source of the self-defeating-eviction bug. ---
        let min_gen_window = 128i32;
        let remaining = self.n_ctx as i32 - self.buffer.committed_len() as i32;
        if remaining < min_gen_window && !self.buffer.is_cold() {
            tracing::info!(
                remaining,
                "cache too tight after prefill; cold-resetting and re-prefilling truncated prompt"
            );
            self.ctx.clear_kv_cache();
            self.buffer.reset();
            self.prefill(&full_tokens, 0)?;
            let system_prefix_len = self.estimate_system_prefix_len(&full_tokens);
            self.buffer.commit_cold(&full_tokens, system_prefix_len);
        }
        let remaining_after = self.n_ctx as i32 - self.buffer.committed_len() as i32;
        if remaining_after < min_gen_window {
            anyhow::bail!(
                "context full: {} tokens resident, n_ctx={}, no room to generate",
                self.buffer.committed_len(),
                self.n_ctx
            );
        }

        // --- Generation loop (sampler + two-layer filter). Captures TTFT +
        // generation speed for telemetry. Checks `cancel` each token. ---
        let gen_result = self.decode_loop(on_chunk, cancel)?;

        // --- Structured telemetry block to stdout/terminal ---
        let prefill_ms = prefill_elapsed.as_secs_f64() * 1000.0;
        let ttft_ms = gen_result
            .time_to_first_token
            .map(|d| d.as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        let gen_elapsed_s = gen_result
            .generation_elapsed
            .as_secs_f64()
            .max(0.0001); // avoid divide-by-zero on sub-token runs
        let gen_speed = gen_result.tokens_generated as f64 / gen_elapsed_s;
        let cancelled_note = if gen_result.cancelled { " [CANCELLED]" } else { "" };

        eprintln!();
        eprintln!("[DEBUG] ─── ENGINE PERFORMANCE TELEMETRY ───{cancelled_note}");
        eprintln!("[DEBUG] Cached Tokens (Prefix Reuse): {cached_tokens} tokens");
        eprintln!("[DEBUG] Prefilled Tokens (New Input):  {prefilled_tokens} tokens");
        eprintln!("[DEBUG] Prefill Processing Time:       {prefill_ms:.1} ms");
        eprintln!("[DEBUG] Time to First Token (TTFT):    {ttft_ms:.1} ms");
        eprintln!("[DEBUG] Generation Speed:              {gen_speed:.1} tokens/sec");
        eprintln!("[DEBUG] ──────────────────────────────────────────");

        tracing::info!(
            cached_tokens,
            prefilled_tokens,
            prefill_ms,
            ttft_ms,
            gen_speed,
            generated_tokens = gen_result.tokens_generated,
            cancelled = gen_result.cancelled,
            cache_total = self.buffer.committed_len(),
            "telemetry"
        );

        Ok(gen_result.parsed)
    }

    /// The token-by-token generation loop. Sampler + ThoughtGate + StreamFilter
    /// — identical config to the previous `generate_blocking`. Captures TTFT +
    /// generation timing, and checks `cancel` between tokens so `chat_stop`
    /// can break out cleanly at a token boundary.
    fn decode_loop(
        &mut self,
        on_chunk: &ChunkFn,
        cancel: &Arc<AtomicBool>,
    ) -> anyhow::Result<DecodeTelemetry> {
        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::temp(1.0),
            LlamaSampler::top_p(0.95, 1),
            LlamaSampler::min_p(0.1, 1),
            LlamaSampler::greedy(),
        ]);

        let mut raw_out = String::new();
        let mut thought_gate = crate::chat_format::ThoughtGate::new();
        let mut marker_filter = crate::stream_filter::StreamFilter::new(&[
            "<|turn>",
            "<turn|>",
            "<|think|>",
            "<|channel>thought",
            "<channel|>",
            "<|tool_call>",
            "<tool_call|>",
            "<|tool_response>",
            "<tool_response|>",
            "<|tool>",
            "<tool|>",
        ]);

        let eos = self.model.token_eos();
        // n_cur is the position of the NEXT token to decode — one past the
        // last prefilled/generated token.
        let mut n_cur = self.buffer.committed_len() as i32;
        // Bug #1 Part B: clamp max_tokens to the remaining cache space. Leave
        // a 64-token safety margin so the final decode never overshoots n_ctx.
        // Floor at 64 so a nearly-full cache still gets a short reply rather
        // than bailing with nothing.
        let remaining = self.n_ctx as i32 - n_cur;
        let max_tokens = remaining.saturating_sub(64).clamp(64, 2048) as usize;
        let mut tokens_generated: Vec<LlamaToken> = Vec::with_capacity(256);

        // Bug #2: allocate the step batch ONCE outside the loop and reuse it
        // via .clear(). The old code allocated LlamaBatch::new(1,1) inside the
        // loop — one alloc per token (~500 for a typical reply). prefill()
        // already shows the correct pattern.
        let mut step_batch = LlamaBatch::new(1, 1);

        let loop_start = std::time::Instant::now();
        let mut time_to_first_token: Option<Duration> = None;
        let mut cancelled = false;

        loop {
            // --- Cancellation check: between tokens, never mid-decode. ---
            if cancel.load(Ordering::Relaxed) {
                cancelled = true;
                tracing::info!("decode loop cancelled by chat_stop");
                break;
            }

            if tokens_generated.len() >= max_tokens {
                break;
            }
            let new_token = sampler.sample(&self.ctx, -1);
            sampler.accept(new_token);

            if self.model.is_eog_token(new_token) || new_token == eos {
                break;
            }

            // TTFT is measured from loop start to the first sampled token.
            if time_to_first_token.is_none() {
                time_to_first_token = Some(loop_start.elapsed());
            }

            let mut decoder = encoding_rs::UTF_8.new_decoder();
            let piece = self
                .model
                .token_to_piece(new_token, &mut decoder, true, None)
                .map_err(|e| anyhow::anyhow!("token to piece: {e:?}"))?;

            if !piece.is_empty() {
                raw_out.push_str(&piece);
                let (gate_output, _is_thinking) = thought_gate.feed(&piece);
                if !gate_output.is_empty() {
                    let cleaned = marker_filter.feed(&gate_output);
                    if !cleaned.is_empty() {
                        on_chunk(&cleaned);
                    }
                }
            }

            n_cur += 1;
            tokens_generated.push(new_token);

            step_batch.clear();
            step_batch
                .add(new_token, n_cur - 1, &[0], true)
                .map_err(|e| anyhow::anyhow!("batch add: {e:?}"))?;
            self.ctx
                .decode(&mut step_batch)
                .map_err(|e| anyhow::anyhow!("decode: {e:?}"))?;
        }

        let generation_elapsed = loop_start.elapsed();

        // Flush both filters.
        let gate_tail = thought_gate.flush();
        if !gate_tail.is_empty() {
            let cleaned = marker_filter.feed(&gate_tail);
            if !cleaned.is_empty() {
                on_chunk(&cleaned);
            }
        }
        let filter_tail = marker_filter.flush();
        if !filter_tail.is_empty() {
            on_chunk(&filter_tail);
        }

        // Record the generated tokens in the buffer — they're now resident in
        // the KV cache at [committed_len .. committed_len + generated.len()).
        let gen_count = tokens_generated.len();
        self.buffer.append_generated(&tokens_generated);

        // Bug #3: attach the raw model output so it can be persisted onto the
        // assistant Message. The formatter re-renders from this next turn so
        // the rendered tokens match the KV cache exactly (no re-prefill).
        let mut parsed = self.formatter.parse_output(&raw_out);
        parsed.raw = raw_out;
        tracing::info!(
            tokens = gen_count,
            content_len = parsed.content.len(),
            reasoning_len = parsed.reasoning.len(),
            cancelled,
            cache_total = self.buffer.committed_len(),
            "generation complete"
        );

        Ok(DecodeTelemetry {
            parsed,
            tokens_generated: gen_count,
            time_to_first_token,
            generation_elapsed,
            cancelled,
        })
    }

    /// Decode `tokens` into the context starting at absolute position
    /// `start_pos`, in 512-token batches. The last token of the last batch
    /// gets logits so sampling can read it.
    fn prefill(&mut self, tokens: &[LlamaToken], start_pos: i32) -> anyhow::Result<()> {
        if tokens.is_empty() {
            return Ok(());
        }
        let n_batch = 512i32;
        let mut batch = LlamaBatch::new(512, 1);
        let mut consumed = 0usize;
        while consumed < tokens.len() {
            let take = std::cmp::min(n_batch as usize, tokens.len() - consumed);
            let is_last_chunk = consumed + take == tokens.len();
            batch.clear();
            for (i, tok) in tokens[consumed..consumed + take].iter().enumerate() {
                let is_final_token = is_last_chunk && i == take - 1;
                batch
                    .add(*tok, start_pos + (consumed + i) as i32, &[0], is_final_token)
                    .map_err(|e| anyhow::anyhow!("batch add: {e:?}"))?;
            }
            self.ctx
                .decode(&mut batch)
                .map_err(|e| anyhow::anyhow!("prefill decode: {e:?}"))?;
            consumed += take;
        }
        Ok(())
    }

    /// Phase 2 reconstruction: clear the KV cache entirely, then re-decode the
    /// system prefix + the surviving tail (everything after `cut`) from
    /// position 0. This is a clean rebuild — no RoPE surgery, no position
    /// shifting. The cost is one full prefill, paid rarely (only when history
    /// approaches `n_ctx`).
    ///
    /// NOTE (2026-07-12): this method is currently UNUSED in the hot path.
    /// The reconstruct-based eviction was the source of the self-defeating-
    /// eviction bug (Bug A) and has been replaced by prompt truncation
    /// (`truncate_to_fit` in kv_buffer.rs). Retained because the underlying
    /// `KvBuffer::should_evict` / `reconstruct_tokens` / `reconstruct_finish`
    /// machinery is tested and may be useful for the Memory (M) engine's
    /// summarization/rollup path when that lands.
    #[allow(dead_code)]
    fn reconstruct_cache(&mut self, cut: usize) -> anyhow::Result<()> {
        // Compose the re-decode sequence: pinned system prefix ++ surviving tail.
        let prefix = self.buffer.system_prefix().to_vec();
        let tail = self.buffer.reconstruct_tokens(cut).to_vec();
        let mut to_redecode = Vec::with_capacity(prefix.len() + tail.len());
        to_redecode.extend_from_slice(&prefix);
        to_redecode.extend_from_slice(&tail);

        // Wipe the cache. Clearing seq 0 with unbounded range drops everything.
        self.ctx.clear_kv_cache();

        // Re-decode the rebuilt sequence from position 0.
        self.prefill(&to_redecode, 0)?;

        // Update bookkeeping to reflect the new cache contents.
        self.buffer.reconstruct_finish(cut);
        tracing::info!(
            rebuilt_tokens = to_redecode.len(),
            cache_total = self.buffer.committed_len(),
            "cache reconstructed"
        );
        Ok(())
    }

    /// Estimate where the system prefix ends in the cold-start token stream.
    /// For Gemma4, the system block is `<|turn>system\n...<turn|>\n` and the
    /// next `<|turn>` token marks the start of the first user turn. If we
    /// can't find a boundary, pin nothing (0) — the prefix is still correct,
    /// just less protected from eviction.
    fn estimate_system_prefix_len(&self, tokens: &[LlamaToken]) -> usize {
        if self.turn_marker.is_empty() || tokens.len() < self.turn_marker.len() {
            return 0;
        }
        // Find the SECOND occurrence of the turn marker — the first opens the
        // system turn, the second opens the first conversation turn. Everything
        // before the second is the system prefix.
        let limit = tokens.len() - self.turn_marker.len();
        let mut found = 0;
        for i in 0..=limit {
            if tokens[i..i + self.turn_marker.len()] == self.turn_marker[..] {
                found += 1;
                if found == 2 {
                    return i;
                }
            }
        }
        // Only one turn marker (system only, no conversation yet). Pin nothing
        // extra — the whole prompt is effectively the prefix.
        0
    }
}

// `turn_marker_literal` lives on `ModelFamily` in `chat_format.rs` — the
// family is the authority on its own turn protocol, so the marker literal
// belongs there rather than dispatching through the trait by name string.
