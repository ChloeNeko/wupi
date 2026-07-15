//! Score-aware Reciprocal Rank Fusion (RRF) — the merge step of hybrid search.
//!
//! Given two ranked lists — sparse (FTS5) and dense (vec0) — each carrying its
//! raw score, this module:
//!
//! 1. Floors the dense list on an absolute cosine threshold (the rejection
//!    authority — see [`DENSE_COSINE_FLOOR`]).
//! 2. Ranks the survivors within each list (position = 1-based rank).
//! 3. Fuses via weighted RRF:
//!
//! ```text
//! score(id) = w_sparse / (k + rank_sparse(id))
//!           + w_dense  / (k + rank_dense(id))
//! ```
//!
//! Pure, allocation-light, and the single most testable piece of the Memory
//! engine. No SQLite, no embedder, no async, no CUDA. The whole point of
//! keeping this in its own file is so the §3A promise ("retrieval math is
//! unit-testable without the embedding backend") is enforced by construction.
//!
//! # Why dense-only flooring (AGENTS.md §2M)
//!
//! The original v1 RRF (2026-07-13) fused on RANK ALONE and discarded the raw
//! scores. That meant a near-random dense hit at cosine 0.25 would still
//! contribute `1/(k+rank)` and could fuse-promote into the prompt. The
//! cross-topic bleed that surfaced in the schema-engine full-system test
//! (a dungeon query retrieving an unrelated cyberpunk memory — §2L) was a
//! direct consequence: rank-based RRF has no rejection signal.
//!
//! The fix is an ABSOLUTE cosine floor on the dense path. Dense (semantic)
//! similarity is the signal that catches "Alex in dungeon" vs "Alex in
//! cyberpunk" — the shared name gives weak lexical overlap but the scenes are
//! semantically distant, landing cosine around 0.25-0.35, below the floor.
//!
//! The sparse (BM25) path is deliberately NOT floored. Two reasons:
//! 1. BM25's absolute scale is model-dependent (document-length normalization,
//!    IDF behavior) and unreliable as a universal threshold — a floor that
//!    works for one corpus mis-tunes for another.
//! 2. The dense floor is already the rejection authority. Sparse only adds
//!    precision-boost on memories that PASSED the dense floor. Flooring sparse
//!    too would be a second rejection gate with no calibration story, and
//!    min-max on it would be RELATIVE to the retrieved set (it maps the
//!    best-of-the-batch to 1.0 regardless of whether the batch is all garbage)
//!    — exactly the failure mode that defeated v1.
//!
//! # Rank indexing is 1-BASED (not 0)
//!
//! The top result in each input list is rank `1` → contributes `1 / (60 + 1)`.
//! A common bug in RRF implementations uses 0-based indexing, which quietly
//! inflates the top result's contribution to `1 / 60` and distorts every
//! downstream comparison. The Cormack et al. (2009) formulation is 1-based;
//! we honor it and lock it with a unit test.
//!
//! # Why `k = 60`
//!
//! `k` is the smoothing constant. Small `k` heavily rewards top ranks (the
//! top result dominates); large `k` flattens the curve (deep results get more
//! weight). `60` is the value from the original paper and is the de-facto
//! standard across hybrid-search systems; it gives the long tail enough pull
//! to surface a memory that ranks e.g. dense-30 but sparse-1, without letting
//! noise overwhelm the top. It is NOT the final `limit` — those are
//! independent knobs.

use std::collections::{HashMap, HashSet};

use crate::memory::{DebugScores, MemoryId, RankedMemory};

/// RRF smoothing constant. Standard value from Cormack et al. (2009).
///
/// Pinned as a `const` so call sites read `RRF_K` instead of a magic `60`.
/// Distinct from the final `limit` passed to [`fuse_scored_rrf`].
pub const RRF_K: u32 = 60;

/// Default hard cosine floor for the dense path. Memories whose query→memory
/// cosine similarity falls below this are REJECTED before fusion — they never
/// contribute to the prompt. This is the rejection authority for cross-topic
/// bleed (AGENTS.md §2L problem #1, §2M fix).
///
/// Calibrated against real retrieval data 2026-07-14 (post embedder-fix +
/// asymmetric query-prefix verification). A multi-topic seed conversation
/// (butter, platinum, diamonds, tiramisu, carbon) queried with the single
/// word "butter" showed a clean ~4× gap between relevant and irrelevant:
///   - relevant (butter Q&A)      : cosine 0.317 – 0.376
///   - irrelevant (all 7 others)  : cosine 0.006 – 0.094
/// A floor of 0.25 sits in the wide gap — keeps the relevant matches with
/// ~0.07 margin and rejects every off-topic result with ~0.16 margin.
///
/// The earlier 0.40 const was a guess from the synthetic self-test probe;
/// it sat ABOVE the real relevant matches (0.32–0.38) and would have
/// rejected the very memories it should keep. Single-word queries score
/// lower than full-sentence queries because there's little context to build
/// meaning from, so the floor must accommodate the weak end of legitimate
/// matches, not the strong end.
///
/// This is a PROVISIONAL value — recalibrate as more query shapes come in.
/// The 🧠 debug panel's `dense_floor` override (`cos ≥`) lets you test
/// alternatives live without a rebuild (AGENTS.md §2M Checkpoint E).
pub const DENSE_COSINE_FLOOR: f32 = 0.25;

/// The dense cosine floor for Codex (authored reference lore) entries.
///
/// Codex entries are static, declarative technical documents. Their embedding
/// style is fundamentally different from conversational episodic memory:
/// reference prose doesn't use the same vocabulary, filler, or sentence
/// structure as chat turns. bge-small (and asymmetric retrieval models in
/// general) score these declarative docs lower on dense cosine even when they
/// are 100% relevant — a textbook passage about "Simulation Card XML format"
/// doesn't embed like a chat turn asking "how do I write a sim card?", even
/// though they're the same topic. This is standard domain asymmetry in RAG
/// retrieval (Codex v1, 2026-07-14, §2P).
///
/// Expecting technical reference prose to clear the same 0.25 hurdle as casual
/// chat history is like expecting a textbook to read like a DM. The lower
/// floor for Codex is the principled fix: per-domain flooring is best-
/// practice RAG design, not a hack. The 🧠 panel's `cos ≥` override does NOT
/// affect this — it overrides the EPISODIC floor only; the Codex floor is
/// applied independently in `fuse_scored_rrf` via the `codex_ids` set.
pub const CODEX_DENSE_FLOOR: f32 = 0.10;

/// Per-list weights for weighted RRF. Both default to 0.5 (standard RRF —
/// equal contribution). Tilting `dense` higher biases toward semantic
/// relevance (the rejection authority); tilting `sparse` higher biases toward
/// keyword precision. The weights need not sum to 1 — RRF is rank-based, so
/// only the RATIO between them matters.
#[derive(Debug, Clone, Copy)]
pub struct FusionWeights {
    pub sparse: f32,
    pub dense: f32,
}

impl Default for FusionWeights {
    fn default() -> Self {
        Self { sparse: 0.5, dense: 0.5 }
    }
}

/// Fuse two ranked, scored lists into one sorted ranking with a hard dense
/// floor and weighted RRF.
///
/// # Inputs
///
/// - `sparse`: `(id, bm25_raw)` best-first. Lower (more-negative) BM25 is a
///   better match. UNFLOORED — see module docs for why.
/// - `dense`: `(id, distance)` best-first. Lower distance is better;
///   cosine = `1 - distance`. Floored on `dense_cosine_floor`.
/// - `dense_cosine_floor`: drop dense candidates whose cosine < floor.
/// - `weights`: per-list RRF weights ([`FusionWeights::default`] = equal).
/// - `limit`: truncate the fused output to this many entries.
///
/// # Output
///
/// Up to `limit` [`RankedMemory`] entries, highest fused score first. Each
/// carries its fused `score` AND a populated [`DebugScores`] (raw cosine +
/// per-list ranks) so the 🧠 panel can show why each memory was pulled.
///
/// An id appearing in BOTH (floored) lists gets both score contributions
/// summed — this is the whole point of fusion: a memory that matches on both
/// axes deserves to outrank one that matches on only one.
///
/// # Per-class dense floor (Codex v1, §2P)
///
/// `codex_ids` carries the ids of Codex (authored reference lore) entries.
/// Codex entries are declarative technical documents whose embedding style
/// differs fundamentally from conversational episodic memory — bge-small
/// scores them lower on dense cosine even at 100% relevance (domain
/// asymmetry). Entries in `codex_ids` are floored on `codex_floor` instead
/// of `dense_cosine_floor`. Pass an empty set when no Codex entries exist
/// (the common case before Codex v1, or when the query is purely episodic).
pub fn fuse_scored_rrf(
    sparse: &[(MemoryId, f32)],
    dense: &[(MemoryId, f32)],
    dense_cosine_floor: f32,
    codex_ids: &HashSet<MemoryId>,
    codex_floor: f32,
    weights: FusionWeights,
    limit: usize,
) -> Vec<RankedMemory> {
    // ── Floor the dense list on absolute cosine ──────────────────────────
    // distance = 1 - cosine  →  cosine = 1 - distance. Keep cosine >= floor.
    // The rejected candidates never enter the fusion map, so they contribute
    // nothing to any id's score. This is the cross-topic rejection gate.
    //
    // PER-CLASS FLOOR: Codex entries (in `codex_ids`) use the lower
    // `codex_floor`; everything else uses `dense_cosine_floor`. This is the
    // domain-asymmetry fix (§2P) — declarative reference docs embed lower
    // than conversational turns at equal relevance, so they get a lower bar.
    let dense_survivors: Vec<(MemoryId, f32)> = dense
        .iter()
        .filter(|(id, distance)| {
            let cosine = 1.0 - distance;
            let floor = if codex_ids.contains(id) {
                codex_floor
            } else {
                dense_cosine_floor
            };
            cosine >= floor
        })
        .cloned()
        .collect();

    // ── Accumulate fused scores + record per-list ranks ──────────────────
    // Each entry in the map carries: accumulated weighted score, dense rank
    // (if present), sparse rank (if present), and the raw dense cosine (for
    // the debug panel — read off borderline hits to calibrate the floor).
    #[derive(Default, Clone)]
    struct Accum {
        score: f32,
        dense_rank: Option<u32>,
        sparse_rank: Option<u32>,
        dense_cosine: Option<f32>,
    }

    let mut acc: HashMap<MemoryId, Accum> = HashMap::with_capacity(
        sparse.len() + dense_survivors.len(),
    );

    // Sparse contributions (unfloored — rank order is the input order).
    for (i, (id, _bm25)) in sparse.iter().enumerate() {
        let rank = (i as u32) + 1; // 1-based
        let contribution = weights.sparse / (RRF_K as f32 + rank as f32);
        let a = acc.entry(*id).or_default();
        a.score += contribution;
        a.sparse_rank = Some(rank);
    }

    // Dense contributions (post-floor survivors only).
    for (i, (id, distance)) in dense_survivors.iter().enumerate() {
        let rank = (i as u32) + 1; // 1-based, among survivors
        let cosine = 1.0 - distance;
        let contribution = weights.dense / (RRF_K as f32 + rank as f32);
        let a = acc.entry(*id).or_default();
        a.score += contribution;
        a.dense_rank = Some(rank);
        // Record the raw cosine for the debug panel. Only dense survivors
        // have a cosine; memories surfaced via sparse-only have None here.
        a.dense_cosine = Some(cosine);
    }

    // ── Sort by fused score descending ───────────────────────────────────
    // Tie-break by id ascending so output is deterministic (tests + the debug
    // panel both rely on stable ordering across runs).
    let mut ranked: Vec<(MemoryId, Accum)> = acc.into_iter().collect();
    ranked.sort_by(|a, b| {
        b.1.score
            .partial_cmp(&a.1.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    ranked
        .into_iter()
        .take(limit)
        .map(|(id, a)| RankedMemory {
            entry: crate::memory::MemoryEntry {
                id,
                text_content: String::new(), // hydrated by fetch_entries
                timestamp: 0,
                role: crate::memory::Role::System,
                chunk_index: 0,
                salience: 0.0,
                metadata_json: None,
                card_id: String::new(),
                session_id: None,
            },
            score: a.score,
            debug: DebugScores {
                dense_cosine: a.dense_cosine,
                dense_rank: a.dense_rank,
                sparse_rank: a.sparse_rank,
            },
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: extract just the id order from a fused result.
    fn ids(out: &[RankedMemory]) -> Vec<MemoryId> {
        out.iter().map(|r| r.entry.id).collect()
    }

    /// Dense list helper: `(id, distance)`. distance = 1 - cosine, so cosine
    /// 0.9 → distance 0.1. Lower distance = better.
    fn dense(id: MemoryId, cosine: f32) -> (MemoryId, f32) {
        (id, 1.0 - cosine)
    }

    /// Sparse list helper: `(id, bm25)`. More-negative = better; the exact
    /// value is irrelevant to fusion (only rank matters), pick negatives.
    fn sparse(id: MemoryId, rank_quality: f32) -> (MemoryId, f32) {
        (id, -rank_quality)
    }

    #[test]
    fn empty_inputs_return_empty() {
        let out = fuse_scored_rrf(&[], &[], 0.40, &HashSet::new(), CODEX_DENSE_FLOOR, FusionWeights::default(), 10);
        assert!(out.is_empty(), "no inputs → no results");
    }

    #[test]
    fn one_empty_list_passes_the_other_through() {
        let s = vec![sparse(10, 1.0), sparse(20, 0.9), sparse(30, 0.8)];
        let out = fuse_scored_rrf(&s, &[], 0.40, &HashSet::new(), CODEX_DENSE_FLOOR, FusionWeights::default(), 10);
        assert_eq!(ids(&out), vec![10, 20, 30]);
    }

    #[test]
    fn dense_floor_drops_below_threshold() {
        // Three dense candidates: cosine 0.9 (keep), 0.5 (keep), 0.2 (drop).
        // Floor 0.40 → only the first two survive.
        let d = vec![dense(1, 0.9), dense(2, 0.5), dense(3, 0.2)];
        let out = fuse_scored_rrf(&[], &d, 0.40, &HashSet::new(), CODEX_DENSE_FLOOR, FusionWeights::default(), 10);
        assert_eq!(ids(&out), vec![1, 2], "below-floor candidate must be rejected");
        // The dropped one (id 3) must not appear anywhere.
        assert!(!ids(&out).contains(&3));
    }

    #[test]
    fn dense_floor_records_cosine_on_survivors_only() {
        let d = vec![dense(1, 0.9), dense(2, 0.2)];
        let out = fuse_scored_rrf(&[], &d, 0.40, &HashSet::new(), CODEX_DENSE_FLOOR, FusionWeights::default(), 10);
        // Survivor carries its raw cosine; the rejected one isn't in the output.
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].entry.id, 1);
        assert!(
            (out[0].debug.dense_cosine.unwrap() - 0.9).abs() < 1e-5,
            "survivor cosine should be recorded"
        );
    }

    #[test]
    fn codex_floor_lets_codex_survive_below_episodic_floor() {
        // Two dense candidates at cosine 0.15 (below episodic floor 0.25, above
        // codex floor 0.10). id=1 is Codex; id=2 is episodic.
        // - id=1 (Codex) → floor 0.10 → 0.15 >= 0.10 → SURVIVES.
        // - id=2 (episodic) → floor 0.25 → 0.15 < 0.25 → REJECTED.
        let d = vec![dense(1, 0.15), dense(2, 0.15)];
        let codex_ids: HashSet<MemoryId> = [1].into_iter().collect();
        let out = fuse_scored_rrf(&[], &d, 0.25, &codex_ids, 0.10, FusionWeights::default(), 10);
        assert_eq!(ids(&out), vec![1], "only the Codex entry survives");
    }

    #[test]
    fn codex_floor_still_rejects_garbage() {
        // A Codex entry at cosine 0.05 — below even the Codex floor (0.10).
        // It must still be rejected; the lower floor is not zero.
        let d = vec![dense(1, 0.05)];
        let codex_ids: HashSet<MemoryId> = [1].into_iter().collect();
        let out = fuse_scored_rrf(&[], &d, 0.25, &codex_ids, 0.10, FusionWeights::default(), 10);
        assert!(out.is_empty(), "Codex entry below codex floor must be rejected");
    }

    #[test]
    fn id_in_both_lists_outranks_id_in_one() {
        // id=5 is rank-1 in BOTH lists → score = (0.5+0.5)/(60+1) ≈ 0.0164.
        // Every other id is in only one list. With equal weights the overlap
        // must dominate.
        let s = vec![sparse(5, 1.0), sparse(1, 0.9), sparse(2, 0.8)];
        let d = vec![dense(5, 0.9), dense(9, 0.8), dense(8, 0.7)];
        let out = fuse_scored_rrf(&s, &d, 0.40, &HashSet::new(), CODEX_DENSE_FLOOR, FusionWeights::default(), 10);
        assert_eq!(out[0].entry.id, 5, "overlap must dominate");
        assert!(
            out[0].score > out[1].score,
            "fused score {} must beat single-list {}",
            out[0].score,
            out[1].score
        );
    }

    #[test]
    fn dense_weight_tilts_toward_semantic() {
        // id=1 is sparse-only (rank 1); id=2 is dense-only (rank 1, cosine 0.9).
        // Equal weights → tie broken by id → [1, 2].
        // Dense-heavy weights (sparse 0.1, dense 0.9) → id=2 dominates.
        let s = vec![sparse(1, 1.0)];
        let d = vec![dense(2, 0.9)];

        let equal = fuse_scored_rrf(&s, &d, 0.40, &HashSet::new(), CODEX_DENSE_FLOOR, FusionWeights::default(), 10);
        assert_eq!(ids(&equal), vec![1, 2], "equal weights → tie-break by id");

        let dense_heavy = fuse_scored_rrf(
            &s,
            &d,
            0.40,
            &HashSet::new(),
            CODEX_DENSE_FLOOR,
            FusionWeights { sparse: 0.1, dense: 0.9 },
            10,
        );
        assert_eq!(ids(&dense_heavy), vec![2, 1], "dense-heavy → dense id first");
    }

    #[test]
    fn limit_truncates() {
        let s: Vec<_> = (1..=5).map(|i| sparse(i, 1.0 / i as f32)).collect();
        let d: Vec<_> = (6..=10).map(|i| dense(i, 0.9 - i as f32 * 0.01)).collect();
        let out = fuse_scored_rrf(&s, &d, 0.40, &HashSet::new(), CODEX_DENSE_FLOOR, FusionWeights::default(), 3);
        assert_eq!(out.len(), 3, "limit must truncate the result");
    }

    #[test]
    fn rank_indexing_is_1_based_not_0() {
        // Regression guard: top rank contributes 1/(k+1), NOT 1/k. With equal
        // weights (0.5 each), a single sparse-list rank-1 id contributes
        // 0.5 / 61.
        let s = vec![sparse(1, 1.0)];
        let out = fuse_scored_rrf(&s, &[], 0.40, &HashSet::new(), CODEX_DENSE_FLOOR, FusionWeights::default(), 10);
        let expected = 0.5 / (RRF_K as f32 + 1.0);
        let got = out[0].score;
        assert!(
            (got - expected).abs() < 1e-6,
            "top score {got} should be w/(k+1) = {expected} (1-based)"
        );
        let wrong = 0.5 / RRF_K as f32;
        assert!(
            (got - wrong).abs() > 1e-6,
            "must not equal w/k (0-based bug); got {got}, w/k = {wrong}"
        );
    }

    #[test]
    fn scores_descend_monotonically() {
        let s = vec![sparse(1, 1.0), sparse(2, 0.9), sparse(3, 0.8), sparse(4, 0.7)];
        let d = vec![dense(5, 0.9), dense(1, 0.85), dense(6, 0.8), dense(2, 0.75)];
        let out = fuse_scored_rrf(&s, &d, 0.40, &HashSet::new(), CODEX_DENSE_FLOOR, FusionWeights::default(), 10);
        for w in out.windows(2) {
            assert!(
                w[0].score >= w[1].score,
                "scores must descend: {} then {}",
                w[0].score,
                w[1].score
            );
        }
    }

    #[test]
    fn sparse_only_id_has_no_dense_debug() {
        // An id surfaced only via sparse must have None dense_cosine/rank.
        let s = vec![sparse(7, 1.0)];
        let d = vec![dense(8, 0.9)];
        let out = fuse_scored_rrf(&s, &d, 0.40, &HashSet::new(), CODEX_DENSE_FLOOR, FusionWeights::default(), 10);
        let seven = out.iter().find(|r| r.entry.id == 7).unwrap();
        assert!(seven.debug.dense_cosine.is_none());
        assert!(seven.debug.dense_rank.is_none());
        assert_eq!(seven.debug.sparse_rank, Some(1));
    }
}
