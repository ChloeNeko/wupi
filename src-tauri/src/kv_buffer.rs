//! Ring-buffer bookkeeping for the persistent chat context.
//!
//! This is the **Phase 1 / Phase 2** token-tracking layer described in the
//! engine design. It holds NO llama.cpp handles: it is pure bookkeeping over
//! a `Vec` of token IDs, which makes it fully `Send`/`Sync` and unit-testable
//! in isolation.
//!
//! # The two phases
//!
//! **Phase 1: Ring-Buffer Pointer Rotation.** We track where the system
//! prefix ends (`system_prefix_len`) and use `should_evict` to decide when the
//! cache is near capacity. The system prefix is pinned: it is never counted
//! toward eviction.
//!
//! **Phase 2: Token-ID Slicing Reconstruction.** Eviction is implemented as
//! a clean rebuild, NOT live RoPE surgery. We always hold the ground-truth
//! token IDs in `token_log`. To evict, we slice out the dropped prefix,
//! `clear_kv_cache()` the live context, and re-decode the surviving tokens
//! from position 0. This is one full prefill: rare, bounded, and never
//! risks corrupting position indices mid-generation.
//!
//! # Token integrity
//!
//! We slice token-ID vectors, never strings. A multi-byte char that straddles
//! a render boundary can never be split here because we're below the
//! tokenizer: every entry in `token_log` is an atomic token the model itself
//! produced. (Prime Directive §2: respect the silicon.)

use llama_cpp_2::token::LlamaToken;

/// The ring-buffer state tracking the live KV cache's logical contents.
///
/// Invariants:
/// - `token_log` always reflects exactly the tokens currently held in the
///   live KV cache (after any reconstruction). It is the single source of
///   truth for what the cache "contains."
/// - `system_prefix_len` is set once on the first (cold-start) generation and
///   never changes for the lifetime of the engine: the system prompt is
///   pinned at the front and excluded from eviction.
#[derive(Debug)]
pub struct KvBuffer {
    /// The full sequence of token IDs currently resident in the live KV cache,
    /// in submission order, starting at position 0.
    token_log: Vec<LlamaToken>,
    /// Number of leading tokens that constitute the (pinned) system prompt.
    /// `token_log[..system_prefix_len]` is the system turn; never evicted.
    system_prefix_len: usize,
}

impl KvBuffer {
    /// Create an empty buffer. The system prefix length is captured on the
    /// first `commit` (cold start).
    pub fn new() -> Self {
        KvBuffer {
            token_log: Vec::new(),
            system_prefix_len: 0,
        }
    }

    /// True if no generation has committed tokens yet (cold-start pending).
    pub fn is_cold(&self) -> bool {
        self.token_log.is_empty()
    }

    /// The number of tokens currently resident in the live KV cache.
    pub fn committed_len(&self) -> usize {
        self.token_log.len()
    }

    /// The pinned system-prefix length (0 until the first commit).
    pub fn system_prefix_len(&self) -> usize {
        self.system_prefix_len
    }

    /// The surviving (post-eviction) tokens, as a slice into `token_log`.
    /// This is the logical view the next reconstruction would re-decode.
    pub fn resident(&self) -> &[LlamaToken] {
        &self.token_log
    }

    /// Length of the longest common prefix between `incoming` and the tokens
    /// currently resident in the cache. This is the delta-diff: tokens before
    /// this index are already in the KV cache and need no re-prefill; tokens
    /// at/after this index are new (or divergent) and must be decoded.
    ///
    /// O(min(n, m)): a single pointer walk. Cheaper than one decode step.
    /// Returns 0 on a cold cache (everything must be prefilled).
    pub fn common_prefix_len(&self, incoming: &[LlamaToken]) -> usize {
        let max = std::cmp::min(self.token_log.len(), incoming.len());
        let mut i = 0;
        while i < max && self.token_log[i] == incoming[i] {
            i += 1;
        }
        i
    }

    /// Cold-start commit: the cache was empty, we just prefilled `tokens` from
    /// position 0. Record the system-prefix boundary so the pin is honored on
    /// future evictions. The caller (which rendered the prompt and knows where
    /// the system turn ends) controls the pin boundary precisely.
    pub fn commit_cold(&mut self, tokens: &[LlamaToken], system_prefix_len: usize) {
        debug_assert!(
            self.token_log.is_empty(),
            "commit_cold called on a non-empty buffer"
        );
        debug_assert!(
            system_prefix_len <= tokens.len(),
            "system prefix longer than the cold-start tokens"
        );
        self.token_log.clear();
        self.token_log.extend_from_slice(tokens);
        self.system_prefix_len = system_prefix_len;
    }

    /// Delta commit: prefilled `delta` tokens starting at position
    /// `common_prefix_len`, after truncating the log back to that length
    /// (anything diverging from the incoming prompt is invalidated: this
    /// handles mid-stream history edits correctly). The surviving tail is the
    /// common prefix; the delta extends it.
    ///
    /// After this call, `token_log` == common_prefix ++ delta, and
    /// `committed_len` reflects the new total.
    pub fn commit_delta(&mut self, common_prefix_len: usize, delta: &[LlamaToken]) {
        debug_assert!(
            common_prefix_len <= self.token_log.len(),
            "common prefix longer than the current log"
        );
        // Drop any divergent tail (history was edited, or the model's prior
        // generation is being superseded).
        self.token_log.truncate(common_prefix_len);
        self.token_log.extend_from_slice(delta);
    }

    /// Append the tokens generated during the just-completed decode loop to
    /// the log without touching the prefix. These tokens are now in the KV
    /// cache at positions `[committed_len .. committed_len + generated.len())`.
    pub fn append_generated(&mut self, generated: &[LlamaToken]) {
        self.token_log.extend_from_slice(generated);
    }

    /// Decide whether the cache is near capacity and should be evicted before
    /// the next prefill. We leave a `reserve` buffer (e.g. 25% of `n_ctx`)
    /// so the upcoming prefill + generation don't overflow.
    ///
    /// Returns the index (into `token_log`) at which to cut for a clean
    /// turn-boundary eviction, or `None` if eviction isn't needed yet.
    ///
    /// The cut index is always > `system_prefix_len` (the prefix is pinned).
    /// It advances to the first turn boundary at/after the drop threshold,
    /// where the target is to keep `n_ctx - reserve` tokens.
    /// Turn boundaries are passed in by the caller as sorted positions in
    /// `turn_boundaries` (e.g. the start offset of each `<|turn>...` block).
    pub fn should_evict(
        &self,
        n_ctx: u32,
        reserve: u32,
        turn_boundaries: &[usize],
    ) -> Option<usize> {
        let committed = self.committed_len();
        let target_kept = (n_ctx as usize).saturating_sub(reserve as usize);
        if committed <= target_kept {
            return None;
        }
        // We need to drop at least `committed - target_kept` tokens. Find the
        // first turn boundary that clears that threshold while staying past
        // the pinned system prefix.
        let min_cut = self.system_prefix_len + (committed - target_kept);
        let cut = turn_boundaries
            .iter()
            .copied()
            .find(|&b| b >= min_cut && b > self.system_prefix_len)
            // No boundary found past the threshold: evict up to the last
            // boundary that still keeps the system prefix pinned.
            .or_else(|| {
                turn_boundaries
                    .iter()
                    .copied()
                    .filter(|&b| b > self.system_prefix_len)
                    .max()
            })?;
        if cut >= committed {
            None // nothing meaningful to evict
        } else {
            Some(cut)
        }
    }

    /// Phase 2 reconstruction: produce the token slice to re-decode after
    /// clearing the KV cache. The caller is responsible for prepending the
    /// pinned system prefix (`token_log[..system_prefix_len]`) to this slice
    /// before decoding, since the rebuilt cache must still contain the system
    /// turn at the front. Returns `token_log[cut..]` (the post-eviction tail).
    pub fn reconstruct_tokens(&self, cut: usize) -> &[LlamaToken] {
        debug_assert!(cut >= self.system_prefix_len, "cut into system prefix");
        &self.token_log[cut..]
    }

    /// The pinned system-prefix tokens: what the caller must prepend to the
    /// `reconstruct_tokens` tail when rebuilding the cache after eviction.
    pub fn system_prefix(&self) -> &[LlamaToken] {
        &self.token_log[..self.system_prefix_len]
    }

    /// Finish a Phase 2 reconstruction: the live cache now holds the system
    /// prefix followed by `token_log[cut..]`, re-decoded from position 0.
    /// Reset the log to match: the new log is `prefix ++ tail`.
    pub fn reconstruct_finish(&mut self, cut: usize) {
        debug_assert!(cut <= self.token_log.len(), "cut past end of log");
        let prefix: Vec<LlamaToken> = self.token_log[..self.system_prefix_len].to_vec();
        let tail: Vec<LlamaToken> = self.token_log[cut..].to_vec();
        let mut rebuilt = Vec::with_capacity(prefix.len() + tail.len());
        rebuilt.extend_from_slice(&prefix);
        rebuilt.extend_from_slice(&tail);
        self.token_log = rebuilt;
    }

    /// Reset the entire buffer (e.g. on a hard context reset). Token log is
    /// cleared; the next commit is treated as a fresh cold start.
    pub fn reset(&mut self) {
        self.token_log.clear();
        self.system_prefix_len = 0;
    }
}

impl Default for KvBuffer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Turn-boundary scanning
// ---------------------------------------------------------------------------

/// Scan a rendered token stream for turn boundaries: positions where a new
/// `<|turn>` token sequence begins. Used by eviction to cut at clean turn
/// boundaries rather than mid-message.
///
/// `tokens` is the full tokenized prompt; `turn_marker` is the model's
/// `<|turn>` sequence as token IDs (caller tokenizes the literal once).
/// Returns sorted, deduplicated start positions (the index of the first token
/// of each match).
pub fn scan_turn_boundaries(tokens: &[LlamaToken], turn_marker: &[LlamaToken]) -> Vec<usize> {
    if turn_marker.is_empty() || tokens.len() < turn_marker.len() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let limit = tokens.len() - turn_marker.len();
    for i in 0..=limit {
        if tokens[i..i + turn_marker.len()] == *turn_marker {
            out.push(i);
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Truncate a rendered token stream so it fits within `max_len` tokens by
/// dropping the OLDEST conversation turns (those right after the system
/// prefix), keeping the system prefix and the most recent turns.
///
/// This is the fix for the self-defeating eviction bug (Bug A, 2026-07-12):
/// delta-prefill assumes the rendered prompt is a superset of what's in the
/// cache, but reconstruction drops middle tokens while the prompt retains
/// full history → the next prefill's delta is ~the entire conversation →
/// `NoKvCacheSlot`. Truncating the PROMPT to fit means the cache never needs
/// to hold more than `max_len` tokens, so eviction (when it runs) frees real
/// space the next delta can actually use.
///
/// `system_prefix_len` is the index where conversation turns begin (caller
/// computes this: for Gemma it's the position of the SECOND `<|turn>` marker,
/// since the first opens the system turn). Everything before it is pinned and
/// never dropped. `turn_boundaries` are the sorted start positions of each
/// `<|turn>` marker in `tokens` (from `scan_turn_boundaries`).
///
/// Returns the truncated token slice as a owned `Vec`. If the prompt already
/// fits, returns it unchanged. If it can't fit even with everything but the
/// system prefix + last turn dropped, returns `None` (caller bails cleanly).
///
/// Only call this for marker-bearing families (Gemma). Plain has no markers
/// and truncating mid-token would corrupt the prompt: the caller checks
/// `turn_marker` presence first.
pub fn truncate_to_fit(
    tokens: &[LlamaToken],
    max_len: usize,
    system_prefix_len: usize,
    turn_boundaries: &[usize],
) -> Option<Vec<LlamaToken>> {
    // Already fits: no work.
    if tokens.len() <= max_len {
        return Some(tokens.to_vec());
    }
    // Need at least: the system prefix + one turn marker to start the first
    // kept turn. The conversation turns begin at `system_prefix_len`; we only
    // consider boundaries that open a turn AT OR AFTER that point.
    let conv_boundaries: Vec<usize> = turn_boundaries
        .iter()
        .copied()
        .filter(|&b| b >= system_prefix_len)
        .collect();
    if conv_boundaries.is_empty() {
        // Over budget but no conversation turns to drop: the system prefix
        // alone exceeds max_len. Nothing safe to do.
        return None;
    }
    // Greedily drop the OLDEST conversation turn and re-check. Each boundary
    // in conv_boundaries opens a turn; dropping turns [0..keep_from] removes
    // those turns entirely. We want the SMALLEST keep_from such that the
    // surviving slice fits.
    //
    // The surviving slice is always: tokens[..system_prefix_len] ++ tokens[b..]
    // where b is the start of the earliest kept conversation turn.
    for &b in conv_boundaries.iter() {
        let kept_len = system_prefix_len + (tokens.len() - b);
        if kept_len <= max_len {
            let mut out =
                Vec::with_capacity(system_prefix_len + (tokens.len() - b));
            out.extend_from_slice(&tokens[..system_prefix_len]);
            out.extend_from_slice(&tokens[b..]);
            return Some(out);
        }
    }
    // Even keeping only the last turn exceeds max_len. The single most recent
    // turn + system prefix is too long.
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Convenience: synthesize a LlamaToken from a raw id. LlamaToken is
    /// repr(transparent) over i32, so this is safe for test data.
    fn tok(id: i32) -> LlamaToken {
        LlamaToken(id)
    }

    fn seq(ids: &[i32]) -> Vec<LlamaToken> {
        ids.iter().map(|&i| tok(i)).collect()
    }

    #[test]
    fn cold_start_commits_all_tokens_and_pins_prefix() {
        let mut buf = KvBuffer::new();
        assert!(buf.is_cold());
        let tokens = seq(&[1, 2, 3, 4, 5]);
        buf.commit_cold(&tokens, 2); // first 2 are system prefix
        assert_eq!(buf.committed_len(), 5);
        assert_eq!(buf.system_prefix_len(), 2);
        assert_eq!(buf.resident(), &tokens[..]);
        assert!(!buf.is_cold());
    }

    #[test]
    fn common_prefix_matches_identical_leading_tokens() {
        let mut buf = KvBuffer::new();
        buf.commit_cold(&seq(&[10, 20, 30, 40]), 0);
        // Same prefix, new tail.
        let incoming = seq(&[10, 20, 30, 99, 100]);
        assert_eq!(buf.common_prefix_len(&incoming), 3);
    }

    #[test]
    fn common_prefix_zero_on_cold_cache() {
        let buf = KvBuffer::new();
        assert_eq!(buf.common_prefix_len(&seq(&[1, 2, 3])), 0);
    }

    #[test]
    fn common_prefix_full_when_incoming_is_subset() {
        let mut buf = KvBuffer::new();
        buf.commit_cold(&seq(&[1, 2, 3, 4, 5]), 0);
        // Incoming is a prefix of the resident log.
        assert_eq!(buf.common_prefix_len(&seq(&[1, 2, 3])), 3);
    }

    #[test]
    fn delta_commit_extends_log_and_drops_divergent_tail() {
        let mut buf = KvBuffer::new();
        buf.commit_cold(&seq(&[1, 2, 3, 4, 5]), 0);
        // Simulate an edit: incoming shares only the first 2 tokens.
        // commit_delta(common=2, delta=[8,9]) should yield [1,2,8,9].
        buf.commit_delta(2, &seq(&[8, 9]));
        assert_eq!(buf.resident(), &seq(&[1, 2, 8, 9])[..]);
        assert_eq!(buf.committed_len(), 4);
    }

    #[test]
    fn delta_commit_with_empty_delta_just_truncates() {
        let mut buf = KvBuffer::new();
        buf.commit_cold(&seq(&[1, 2, 3, 4, 5]), 0);
        buf.commit_delta(2, &[]);
        assert_eq!(buf.resident(), &seq(&[1, 2])[..]);
    }

    #[test]
    fn append_generated_grows_log_in_place() {
        let mut buf = KvBuffer::new();
        buf.commit_cold(&seq(&[1, 2, 3]), 0);
        buf.append_generated(&seq(&[77, 78, 79]));
        assert_eq!(buf.resident(), &seq(&[1, 2, 3, 77, 78, 79])[..]);
    }

    // --- Eviction decision logic ---

    #[test]
    fn no_eviction_when_under_capacity() {
        let mut buf = KvBuffer::new();
        buf.commit_cold(&seq(&[1, 2, 3, 4]), 1); // prefix=1
        let boundaries = vec![1usize, 3]; // turns start at idx 1 and 3
        assert_eq!(buf.should_evict(100, 25, &boundaries), None);
    }

    #[test]
    fn eviction_cuts_past_system_prefix_at_turn_boundary() {
        let mut buf = KvBuffer::new();
        // 10 tokens, system prefix = 2 (pinned). n_ctx=8, reserve=2 →
        // target_kept = 6. committed(10) > 6 → must evict ≥4 tokens.
        // min_cut = prefix(2) + (10-6) = 6. First boundary ≥6 and >2 is 7.
        buf.commit_cold(&seq(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 2);
        let boundaries = vec![2, 5, 7];
        let cut = buf.should_evict(8, 2, &boundaries);
        assert_eq!(cut, Some(7));
    }

    #[test]
    fn eviction_falls_back_to_last_boundary_when_none_past_threshold() {
        let mut buf = KvBuffer::new();
        // 10 tokens, prefix=2. n_ctx=8, reserve=2 → target_kept=6, min_cut=6.
        // But the only boundary past the prefix is at 4 (< min_cut). Fallback
        // picks the max boundary past the prefix.
        buf.commit_cold(&seq(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 2);
        let boundaries = vec![2, 4];
        let cut = buf.should_evict(8, 2, &boundaries);
        assert_eq!(cut, Some(4));
    }

    #[test]
    fn eviction_returns_none_if_cut_would_drop_everything() {
        let mut buf = KvBuffer::new();
        // prefix occupies the whole log; only boundary is the prefix itself.
        buf.commit_cold(&seq(&[1, 2, 3, 4]), 4);
        let boundaries = vec![0];
        assert_eq!(buf.should_evict(2, 1, &boundaries), None);
    }

    // --- Phase 2 reconstruction: the token-integrity regression ---

    #[test]
    fn reconstruct_tokens_returns_tail_after_cut() {
        let mut buf = KvBuffer::new();
        buf.commit_cold(&seq(&[10, 20, 30, 40, 50, 60]), 2);
        let tail = buf.reconstruct_tokens(2);
        assert_eq!(tail, &seq(&[30, 40, 50, 60])[..]);
    }

    #[test]
    fn reconstruct_finish_keeps_prefix_and_tail_drops_middle() {
        let mut buf = KvBuffer::new();
        // prefix=[10,20], tail=[50,60], middle=[30,40] gets evicted.
        buf.commit_cold(&seq(&[10, 20, 30, 40, 50, 60]), 2);
        buf.reconstruct_finish(4); // cut at 4: drop [30,40]
        assert_eq!(buf.resident(), &seq(&[10, 20, 50, 60])[..]);
        assert_eq!(buf.committed_len(), 4);
    }

    /// The full eviction round-trip: a buffer that needs eviction should,
    /// after reconstruct_tokens + reconstruct_finish, contain exactly the
    /// system prefix plus the surviving tail: nothing more, nothing less,
    /// in order. This is the token-boundary integrity guarantee.
    #[test]
    fn full_eviction_roundtrip_preserves_token_integrity() {
        let mut buf = KvBuffer::new();
        let original = seq(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        buf.commit_cold(&original, 2); // prefix=2 pinned
        let boundaries = vec![2, 5, 7];
        let cut = buf.should_evict(8, 2, &boundaries).expect("should evict");
        assert_eq!(cut, 7);
        // The caller rebuilds: prefix ++ tail.
        let prefix = buf.system_prefix().to_vec();
        let tail = buf.reconstruct_tokens(cut).to_vec();
        let mut to_redecode = prefix;
        to_redecode.extend_from_slice(&tail);
        buf.reconstruct_finish(cut);
        // After reconstruction: resident == prefix ++ tail == [1,2,8,9,10]
        assert_eq!(buf.resident(), &seq(&[1, 2, 8, 9, 10])[..]);
        assert_eq!(buf.resident(), &to_redecode[..]);
        assert_eq!(buf.committed_len(), 5);
    }

    // --- Turn-boundary scanning ---

    #[test]
    fn scan_finds_all_turn_marker_positions() {
        let tokens = seq(&[1, 2, 99, 99, 3, 4, 99, 99, 5]);
        let marker = seq(&[99, 99]);
        let boundaries = scan_turn_boundaries(&tokens, &marker);
        assert_eq!(boundaries, vec![2, 6]);
    }

    #[test]
    fn scan_returns_empty_when_no_marker_present() {
        let tokens = seq(&[1, 2, 3]);
        let marker = seq(&[99]);
        assert!(scan_turn_boundaries(&tokens, &marker).is_empty());
    }

    #[test]
    fn scan_returns_empty_when_marker_longer_than_tokens() {
        let tokens = seq(&[1, 2]);
        let marker = seq(&[9, 9, 9]);
        assert!(scan_turn_boundaries(&tokens, &marker).is_empty());
    }

    #[test]
    fn scan_finds_adjacent_markers() {
        // marker = [9]; tokens = [9,9,9] → boundaries at 0,1,2
        let tokens = seq(&[9, 9, 9]);
        let marker = seq(&[9]);
        assert_eq!(scan_turn_boundaries(&tokens, &marker), vec![0, 1, 2]);
    }

    // --- truncate_to_fit (Bug A fix, 2026-07-12) ---

    #[test]
    fn truncate_returns_unchanged_when_under_limit() {
        // marker [9]; tokens: sys(2) + turn1(3) + turn2(3) = 8 tokens.
        // boundaries at 0 (system turn) and 2 (first conv turn) and 5.
        // system_prefix_len = 2 (second marker). Under max_len=10 → unchanged.
        let tokens = seq(&[9, 1, 9, 2, 3, 9, 4, 5]);
        let boundaries = vec![0usize, 2, 5];
        let out = truncate_to_fit(&tokens, 10, 2, &boundaries);
        assert_eq!(out.as_deref(), Some(&tokens[..]));
    }

    #[test]
    fn truncate_drops_oldest_conversation_turns_to_fit() {
        // 8 tokens, max_len=6, system_prefix_len=2. Need to shed 2+ tokens.
        // Dropping turn1 (starts at boundary 2) leaves: sys(2) + turn2(3) = 5. ✓
        let tokens = seq(&[9, 1, 9, 2, 3, 9, 4, 5]);
        let boundaries = vec![0usize, 2, 5];
        let out = truncate_to_fit(&tokens, 6, 2, &boundaries).unwrap();
        // Expected: tokens[..2] ++ tokens[5..] = [9,1,9,4,5] (5 tokens)
        assert_eq!(out, seq(&[9, 1, 9, 4, 5]));
        assert!(out.len() <= 6);
    }

    #[test]
    fn truncate_keeps_system_prefix_pinned() {
        // Whatever we drop, the system prefix [9,1] must survive at the front.
        let tokens = seq(&[9, 1, 9, 2, 3, 9, 4, 5]);
        let boundaries = vec![0usize, 2, 5];
        let out = truncate_to_fit(&tokens, 5, 2, &boundaries).unwrap();
        assert_eq!(&out[..2], &[tok(9), tok(1)]); // system prefix intact
    }

    #[test]
    fn truncate_returns_none_when_system_prefix_alone_exceeds_max() {
        // system_prefix_len=5 but only 4 tokens before first conv boundary.
        // Actually simpler: no conv boundaries past system prefix → None.
        let tokens = seq(&[9, 1, 9, 2, 3, 4, 5, 6]);
        let boundaries = vec![0usize]; // only the system-turn marker
        // system_prefix_len = 0 here (caller decides); with no conv boundaries,
        // there's nothing to drop.
        assert_eq!(truncate_to_fit(&tokens, 3, 0, &boundaries), None);
    }

    #[test]
    fn truncate_returns_none_when_single_turn_too_long() {
        // 8 tokens, max_len=4, system_prefix_len=2. Even keeping only the last
        // turn (3 tokens) + prefix (2) = 5 > 4. → None.
        let tokens = seq(&[9, 1, 9, 2, 3, 9, 4, 5]);
        let boundaries = vec![0usize, 2, 5];
        assert_eq!(truncate_to_fit(&tokens, 4, 2, &boundaries), None);
    }

    #[test]
    fn truncate_drops_multiple_turns_when_needed() {
        // 3 conv turns, need to drop 2 to fit.
        // tokens: sys[9,1] + t1[9,10,11] + t2[9,12,13] + t3[9,14,15] = 11 tokens
        // boundaries: 0, 2, 5, 8. system_prefix_len=2.
        // max_len=6: keep sys(2) + t3(3) = 5 ✓ (drop t1 and t2).
        let tokens = seq(&[9, 1, 9, 10, 11, 9, 12, 13, 9, 14, 15]);
        let boundaries = vec![0usize, 2, 5, 8];
        let out = truncate_to_fit(&tokens, 6, 2, &boundaries).unwrap();
        assert_eq!(out, seq(&[9, 1, 9, 14, 15]));
        assert!(out.len() <= 6);
    }
}
