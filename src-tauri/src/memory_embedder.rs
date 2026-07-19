//! The embedding seam for the Memory engine.
//!
//! Memory retrieval logic (RRF fusion, FTS5 queries, vec0 cosine) is written
//! against the [`Embedder`] trait, NOT against any concrete embedding backend.
//! This is the §3A mitigation: retrieval math stays unit-testable without
//! linking llama-cpp-2's CUDA layer. Tests construct a [`crate::memory::MemoryEngine`]
//! with a [`StubEmbedder`]; production will construct it with a real
//! `LlamaCppEmbedder` once the BERT load path lands (Phase 2.5).
//!
//! This file is deliberately dependency-free: no `llama-cpp-2`, no `rusqlite`,
//! no `sqlite-vec`. Pure trait + alias + const + one trivial impl. If a future
//! change adds a CUDA import here, the §3A invariant has been broken.
//!
//! # The dimension contract
//!
//! [`EMBED_DIM`] is the vector length every embedder MUST produce and the
//! width declared in the `vec0` DDL in `memory.rs`. A mismatch crashes `vec0`
//! at insert time with a confusing size error. The value is read directly from
//! `Embed.gguf`'s GGUF metadata header (`bert.embedding_length`): see the
//! citation on the const.

use std::future::Future;
use std::pin::Pin;

/// Vector width produced by `Embed.gguf`.
///
/// `Embed.gguf` is `bge-small-en-v1.5`: a BERT-architecture encoder, NOT
/// Gemma-family. Its GGUF header declares `bert.embedding_length = 384`
/// (parsed 2026-07-13). A 768 guess would have crashed `vec0` at first insert:
/// the virtual table is declared `float[384]` and the dimension is checked at
/// insert time, not compile time.
///
/// If `Embed.gguf` is ever replaced by a different embedding model, this
/// constant must change and the `vec0` DDL must be re-issued (the schema does
/// not migrate live). See AGENTS.md §2 (models on disk) for the naming
/// convention.
pub const EMBED_DIM: usize = 384;

/// Owned, boxed future returned by [`Embedder::embed`].
///
/// Mirrors the `StreamFuture` pattern in `llm.rs`: the codebase convention is
/// boxed futures returned from trait methods, NOT `async fn` in traits (neither
/// existing trait: `GenerationClient` nor `ChatFormat` - uses `async fn`).
/// Owned `String` input → `'static` future, avoiding lifetime gymnastics on
/// the borrow held across the await.
pub type EmbedFuture = Pin<Box<dyn Future<Output = anyhow::Result<Vec<f32>>> + Send>>;

/// Produces a dense vector embedding for a text.
///
/// Implementations MUST return a `Vec<f32>` of length [`EMBED_DIM`]. The
/// `vec0` insert path in `memory.rs` does not re-check the length: it would
/// be a redundant per-insert scan that the embedder contract already covers.
///
/// Receivers are `&self` (not `&mut self`), matching both existing traits.
/// Real backends that need interior mutability (a dedicated embedding thread +
/// channel, like `LlamaCppBackend` does for chat) provide it themselves behind
/// an `Arc<Mutex<...>>` or a channel handle.
pub trait Embedder: Send + Sync {
    /// Embed `text` as a DOCUMENT: the archived/storage side of retrieval.
    /// Asymmetric models (e.g. bge-small) embed documents raw.
    fn embed(&self, text: String) -> EmbedFuture;

    /// Embed `text` as a QUERY: the search side of retrieval. Asymmetric
    /// models apply a query instruction prefix here (see the concrete impl's
    /// doc); symmetric models (and [`StubEmbedder`]) default to [`embed`].
    ///
    /// The data plane calls this for search and [`embed`](Self::embed) for
    /// archival, so the model-specific query/document asymmetry lives entirely
    /// in the embedder: not in the retrieval math (AGENTS.md §3A). "This text
    /// is a query" is retrieval logic; "the query gets a prefix" is model
    /// behavior, and only the impl knows the latter.
    fn embed_query(&self, text: String) -> EmbedFuture {
        self.embed(text)
    }

    /// Reports this embedder's output dimensionality. Must equal [`EMBED_DIM`].
    /// Exists so callers can assert the contract at construction time without
    /// running an embed first.
    fn dim(&self) -> usize;
}

/// Blanket forward `Embedder` through `Box<E>`. Without this, `Box<dyn Embedder>`
/// does NOT auto-implement `Embedder` (trait objects don't self-forward), so
/// `MemoryEngine<Box<dyn Embedder>>` would not satisfy the `E: Embedder` bound.
/// This is the standard Rust pattern: the same blanket impl `std` provides for
/// `Box<dyn Read>`/`Box<dyn Write>`. `?Sized` is required so `dyn Embedder`
/// (which is `!Sized`) is covered, and it additionally covers `Box<ConcreteType>`.
///
/// Lets `AppState` hold one concrete `MemoryEngine<Box<dyn Embedder + Send + Sync>>`
/// regardless of which backend (`LlamaCppEmbedder` or `StubEmbedder`) was chosen
/// at startup: decided once in `setup()`, dispatched through one virtual call
/// per embed (negligible next to multi-ms GPU work).
impl<E: Embedder + ?Sized> Embedder for Box<E> {
    fn embed(&self, text: String) -> EmbedFuture {
        (**self).embed(text)
    }

    fn embed_query(&self, text: String) -> EmbedFuture {
        (**self).embed_query(text)
    }

    fn dim(&self) -> usize {
        (**self).dim()
    }
}

/// Deterministic, dependency-free embedder for tests.
///
/// NOT a real embedding. Produces a bag-of-characters histogram (byte value
/// modulo `dim` → bucket) so that two texts sharing characters get
/// non-orthogonal (cosine > 0) vectors. Good enough to verify the store/RRF
/// plumbing end-to-end; says nothing about semantic quality.
///
/// Cheaper than any real embedder by orders of magnitude, and crucially does
/// not link `llama-cpp-2`: so unit tests against this stay CUDA-free.
pub struct StubEmbedder {
    /// Vector width this stub will produce. Tests should pass [`EMBED_DIM`]
    /// so they exercise the same code path as production.
    pub dim: usize,
}

impl Embedder for StubEmbedder {
    fn embed(&self, text: String) -> EmbedFuture {
        let dim = self.dim;
        Box::pin(async move {
            // Byte-bucket histogram. ASCII letters collapse into buckets that
            // overlap across similar English text: enough for cosine > 0.
            let mut v = vec![0.0f32; dim];
            for b in text.as_bytes() {
                v[(*b as usize) % dim] += 1.0;
            }
            Ok(v)
        })
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_returns_requested_dim() {
        let s = StubEmbedder { dim: 384 };
        // Runtime doesn't matter; block on a futures executor by polling in a
        // single-threaded context. Use tokio since the crate already depends
        // on it: no new dep.
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("test rt");
        let v = rt.block_on(s.embed("hello".into())).expect("embed");
        assert_eq!(v.len(), 384, "stub must produce EMBED_DIM elements");
    }

    #[test]
    fn stub_is_deterministic() {
        let s = StubEmbedder { dim: 64 };
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("test rt");
        let a = rt.block_on(s.embed("the quick brown fox".into())).expect("embed a");
        let b = rt.block_on(s.embed("the quick brown fox".into())).expect("embed b");
        assert_eq!(a, b, "stub must be deterministic for the same input");
    }

    #[test]
    fn stub_overlapping_text_is_non_orthogonal() {
        // Two texts sharing characters must produce vectors with cosine > 0.
        // This is the property RRF + vec0 rely on to surface "similar" texts.
        let s = StubEmbedder { dim: 64 };
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("test rt");
        let a = rt.block_on(s.embed("hello world".into())).expect("embed a");
        let b = rt.block_on(s.embed("world hello".into())).expect("embed b");
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        assert!(dot > 0.0, "overlapping text must be non-orthogonal, dot={dot}");
    }

    #[test]
    fn embed_dim_const_matches_gguf_header() {
        // Regression guard: if someone swaps Embed.gguf and forgets to bump
        // this const, a unit test failure here is far cheaper than a vec0
        // insert crash at runtime.
        assert_eq!(EMBED_DIM, 384, "Embed.gguf is bge-small-en-v1.5 (384-dim)");
    }
}
