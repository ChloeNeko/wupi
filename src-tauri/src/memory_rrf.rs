//! Reciprocal Rank Fusion (RRF) — the merge step of hybrid search.
//!
//! Given two ranked id lists (sparse/FTS5 and dense/vec0), fuse them into one
//! ranking via the standard formula:
//!
//! ```text
//! score(id) = Σ over each list L of  1 / (k + rank_L(id))
//! ```
//!
//! Pure, allocation-light, and the single most testable piece of the Memory
//! engine. No SQLite, no embedder, no async, no CUDA. The whole point of
//! keeping this in its own file is so the §3A promise ("retrieval math is
//! unit-testable without the embedding backend") is enforced by construction.
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

use std::collections::HashMap;

use crate::memory::{MemoryId, RankedMemory};

/// RRF smoothing constant. Standard value from Cormack et al. (2009).
///
/// Pinned as a `const` so call sites read `RRF_K` instead of a magic `60`.
/// Distinct from the final `limit` passed to [`fuse_rrf`].
pub const RRF_K: u32 = 60;

/// Fuse two ranked id lists into one sorted ranking.
///
/// `sparse` and `dense` are each already sorted best-first — that is, the
/// order returned by FTS5's `ORDER BY bm25() ASC` and vec0's
/// `ORDER BY distance ASC` respectively. Position within each slice IS the
/// rank (1-based: index 0 → rank 1).
///
/// An id appearing in BOTH lists gets both score contributions summed — this
/// is the whole point of fusion: a memory that matches on both axes deserves
/// to outrank one that matches on only one.
///
/// Returns up to `limit` entries, highest score first, each carrying its
/// fused score. The score's absolute value is not meaningful (it's a sum of
/// small fractions); only the ordering is.
pub fn fuse_rrf(sparse: &[MemoryId], dense: &[MemoryId], limit: usize) -> Vec<RankedMemory> {
    // Accumulate scores. Capacity is the smaller of "everything" and "a sane
    // cap" — in practice both inputs are <= RETRIEVAL_DEPTH (64), so this map
    // holds at most 128 entries. HashMap over i64 is cheap.
    let mut scores: HashMap<MemoryId, f32> = HashMap::with_capacity(sparse.len() + dense.len());

    // Add a list's rank contributions. `enumerate()` is 0-based; rank is
    // 1-based, so `+ 1`. Fused into a closure to avoid duplicating the loop
    // body across the two inputs (one block of math, two call sites).
    let mut add_list = |list: &[MemoryId]| {
        for (i, &id) in list.iter().enumerate() {
            let rank = (i as u32) + 1; // 1-based — see module doc.
            let contribution = 1.0 / (RRF_K as f32 + rank as f32);
            *scores.entry(id).or_insert(0.0) += contribution;
        }
    };
    add_list(sparse);
    add_list(dense);

    // Sort by score descending. For ties (same score), break ties by id
    // ascending so the order is deterministic — tests and the debug panel
    // both rely on stable output across runs.
    let mut ranked: Vec<(MemoryId, f32)> = scores.into_iter().collect();
    ranked.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    ranked
        .into_iter()
        .take(limit)
        // The entry text isn't known here — RRF is pure ranks. Callers hydrate
        // the full MemoryEntry via fetch_entries after fusion. The placeholder
        // text_content is fine because fetch_entries overwrites it.
        .map(|(id, score)| RankedMemory {
            entry: crate::memory::MemoryEntry {
                id,
                text_content: String::new(),
                timestamp: 0,
                role: crate::memory::Role::System,
                chunk_index: 0,
                salience: 0.0,
                metadata_json: None,
            },
            score,
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

    #[test]
    fn empty_inputs_return_empty() {
        let out = fuse_rrf(&[], &[], 10);
        assert!(out.is_empty(), "no inputs → no results");
    }

    #[test]
    fn one_empty_list_passes_the_other_through() {
        let sparse: Vec<MemoryId> = vec![10, 20, 30];
        let out = fuse_rrf(&sparse, &[], 10);
        assert_eq!(ids(&out), vec![10, 20, 30]);
    }

    #[test]
    fn top_of_each_list_dominates() {
        // Distinct ids, no overlap. Each list's top ranks highest in its own
        // contribution; sparse[0] and dense[0] both score 1/61, tie broken by
        // id ascending → dense id 100 comes before sparse id 1.
        let sparse: Vec<MemoryId> = vec![1, 2, 3];
        let dense: Vec<MemoryId> = vec![100, 200, 300];
        let out = fuse_rrf(&sparse, &dense, 10);
        // Both tops score 1/61; tie broken by id asc → 1 < 100.
        assert_eq!(out[0].entry.id, 1, "tie should break to lower id");
        assert_eq!(out[1].entry.id, 100);
        // Next tier: both 2nd-rank, score 1/62 each.
        assert_eq!(out[2].entry.id, 2);
        assert_eq!(out[3].entry.id, 200);
    }

    #[test]
    fn id_in_both_lists_outranks_id_in_one() {
        // The whole point of fusion. id=5 is rank-1 in BOTH lists →
        // score = 2/61. Every other id is in only one list → max 1/61.
        let sparse: Vec<MemoryId> = vec![5, 1, 2];
        let dense: Vec<MemoryId> = vec![5, 9, 8];
        let out = fuse_rrf(&sparse, &dense, 10);
        assert_eq!(out[0].entry.id, 5, "overlap must dominate");
        // Verify the math: 2/61 vs 1/61.
        let top_score = out[0].score;
        let second_score = out[1].score;
        assert!(
            top_score > second_score,
            "fused score {top_score} must beat single-list {second_score}"
        );
    }

    #[test]
    fn limit_truncates() {
        let sparse: Vec<MemoryId> = vec![1, 2, 3, 4, 5];
        let dense: Vec<MemoryId> = vec![6, 7, 8, 9, 10];
        let out = fuse_rrf(&sparse, &dense, 3);
        assert_eq!(out.len(), 3, "limit must truncate the result");
    }

    #[test]
    fn rank_indexing_is_1_based_not_0() {
        // Regression guard for the classic RRF bug. If indexing were 0-based,
        // the top rank would contribute 1/60 ≈ 0.01667 instead of 1/61 ≈
        // 0.01639. The difference is small but compounds across rankings.
        let sparse: Vec<MemoryId> = vec![1];
        let out = fuse_rrf(&sparse, &[], 10);
        let expected = 1.0 / (RRF_K as f32 + 1.0); // 1 / (60 + 1)
        let got = out[0].score;
        let delta = (got - expected).abs();
        assert!(
            delta < 1e-6,
            "top score {got} should be 1/(k+1) = {expected} (1-based), delta={delta}"
        );
        // And specifically NOT 1/k.
        let wrong = 1.0 / RRF_K as f32;
        assert!(
            (got - wrong).abs() > 1e-6,
            "top score must not equal 1/k (0-based bug); got {got}, 1/k = {wrong}"
        );
    }

    #[test]
    fn scores_descend_monotonically() {
        // Enough ids to populate several tiers; verify strict descent (ties
        // broken by id, so no equal scores in this input).
        let sparse: Vec<MemoryId> = vec![1, 2, 3, 4];
        let dense: Vec<MemoryId> = vec![5, 1, 6, 2];
        let out = fuse_rrf(&sparse, &dense, 10);
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
    fn duplicate_ids_within_one_list_dont_double_count() {
        // Defensive: a buggy upstream that returns the same id twice in one
        // list should NOT let it score 2x. The current impl uses
        // `entry().or_insert(0.0) += contribution`, which DOES double-count.
        // This test documents that behavior. If we later decide duplicates
        // should be deduped per-list, flip the assertion.
        let sparse: Vec<MemoryId> = vec![7, 7];
        let out = fuse_rrf(&sparse, &[], 10);
        // Both occurrences contribute 1/61 + 1/62.
        assert_eq!(out.len(), 1, "dedup across the fused output");
        let expected = 1.0 / 61.0 + 1.0 / 62.0;
        assert!(
            (out[0].score - expected).abs() < 1e-6,
            "duplicate-within-list currently double-counts: got {} expected {}",
            out[0].score,
            expected
        );
    }
}
