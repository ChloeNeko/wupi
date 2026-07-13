//! The Memory engine — hybrid (FTS5 + sqlite-vec) retrieval fused via RRF.
//!
//! This module is the data-plane of Phase 2 (Memory). It owns a single SQLite
//! connection holding three tables that share one primary key:
//!
//! | Table          | Role                              | Key column |
//! |----------------|-----------------------------------|------------|
//! | `memories`     | Core metadata (id, text, role...) | `id` (PK)  |
//! | `memories_fts` | BM25 keyword search (FTS5 mirror) | `rowid`    |
//! | `memories_vec` | Dense cosine search (vec0)        | `rowid`    |
//!
//! The same `id` flows through all three so a single INSERT transaction makes
//! a memory fully searchable by both axes atomically, and RRF can refer to
//! a unified id space.
//!
//! # Async + spawn_blocking
//!
//! All SQLite work is blocking (`rusqlite::Connection` is `!Sync`). Async
//! methods here wrap every query in `tokio::task::spawn_blocking`, matching
//! the pattern established by `save_session` in `lib.rs` (AGENTS.md §2E).
//! The [`MemoryEngine::conn`] lives behind its own `Arc<std::sync::Mutex<...>>`
//! so the blocking closure can take ownership of a cheap `Arc` clone — `&self`
//! receivers, NOT `&mut self`, because `spawn_blocking`'s closure requires
//! `'static` and `&mut self` isn't `'static`. (The original spec had
//! `&mut self`; verdict E on spawn_blocking supersedes it.)
//!
//! # What's NOT here yet
//!
//! - Wiring into `AppState` / `chat_send`. Blocked on the §2F cache-invalidation
//!   decision (tail-injection vs accept-cold-reset).
//! - The real `LlamaCppEmbedder` (`Embed.gguf` is BERT, not Gemma — a new load
//!   path, not a chat-engine reuse).
//! - `debug_memory_query` IPC.
//! - Chunking (the 512-token BERT context limit is documented but unenforced).
//!
//! This module compiles as dead code in v1 — it is the foundation Phase 2.5
//! builds on. Items are `pub` for the future wiring; unused warnings are
//! suppressed via the `#![allow(dead_code)]` at the bottom of the module.

use std::path::Path;
use std::sync::{Arc, Mutex, Once};

use rusqlite::{params, Connection};

use crate::memory_embedder::{Embedder, EMBED_DIM};
use crate::memory_rrf::fuse_rrf;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Canonical memory identifier. Reused as the primary key across `memories`
/// (INTEGER PK), `memories_fts` (rowid), and `memories_vec` (rowid). This
/// reuse is what makes the 3-table insert atomic and the RRF fusion referable.
pub type MemoryId = i64;

/// Origin of a memory, mirroring the chat-turn roles plus a `Summary` slot.
///
/// `Summary` is reserved for the deferred `reconstruct_cache` rollup path
/// (AGENTS.md §2D) — defined now so the schema doesn't need a migration when
/// summarization lands. `System` covers any future system-injected memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
    System,
    Summary,
}

impl Role {
    /// SQLite stores role as TEXT; round-trips through this. Kept as
    /// `&'static str` (not the serde-lowercased form) so reads never depend
    /// on serde attributes.
    pub fn as_str(self) -> &'static str {
        match self {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::System => "system",
            Role::Summary => "summary",
        }
    }

    /// Inverse of [`Self::as_str`]. Unknown strings error rather than guess.
    pub fn parse(s: &str) -> anyhow::Result<Self> {
        Ok(match s {
            "user" => Role::User,
            "assistant" => Role::Assistant,
            "system" => Role::System,
            "summary" => Role::Summary,
            other => anyhow::bail!("unknown role: {other:?}"),
        })
    }
}

/// One stored memory, MINUS the embedding.
///
/// The vector lives in `memories_vec` keyed by `id` — it does NOT travel with
/// this struct. Carrying ~384 floats (`1.5 KB`) on every entry would bloat
/// every serialization, every RRF fusion, and every debug-IPC payload for no
/// reason: callers that need the vector can fetch it by id; callers that don't
/// (which is all of them in v1) pay nothing.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MemoryEntry {
    pub id: MemoryId,
    pub text_content: String,
    /// Unix epoch seconds at insert time.
    pub timestamp: i64,
    pub role: Role,
    /// 0 = whole-message memory. Positive values index into a chunked message
    /// once Phase 3 chunking lands.
    pub chunk_index: i32,
    /// Caller-supplied importance in `[0, 1]`. Stored but not yet used by
    /// retrieval (Phase 2.5 may weight RRF by this).
    pub salience: f32,
    /// Free-form JSON the caller wants associated with the memory
    /// (character, scene, tags...). Opaque to Memory; verbatim round-trip.
    pub metadata_json: Option<String>,
}

/// A search result carrying its fused RRF score.
///
/// Returned by [`MemoryEngine::search`] so the debug IPC can show *why* a
/// memory was pulled (verdict C, 2026-07-13: observability wins). Callers who
/// don't care about the score map to `.entry`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RankedMemory {
    pub entry: MemoryEntry,
    /// Fused RRF score. Higher is better. The scale is `1/61..~2/61` (one or
    /// both lists, top rank); absolute value is not meaningful, only ordering.
    pub score: f32,
}

/// Render a ranked hit list as a compact injection block for the system prompt.
///
/// One line per memory, prefixed with `[role]`, ordered by fused score (the
/// slice's natural order from `search`). No scores or metadata in the block —
/// keep it token-cheap (Prime Directive §1B.3: serialize strictly, no bloat).
/// The observability panel (pillar 4) is where scores go; the prompt only needs
/// the text the model should attend to.
pub fn render_memory_block(hits: &[RankedMemory]) -> String {
    hits.iter()
        .map(|h| format!("[{}] {}", h.entry.role.as_str(), h.entry.text_content))
        .collect::<Vec<_>>()
        .join("\n")
}

// ---------------------------------------------------------------------------
// sqlite-vec registration
// ---------------------------------------------------------------------------

// sqlite-vec registers itself via `sqlite3_auto_extension`, a one-time global
// hook that makes the `vec0` module available to every subsequently-opened
// Connection. We must run it exactly once per process; `Once` enforces that.
// The transmute is the documented registration pattern (see sqlite-vec's
// examples/simple-rust/demo.rs). SAFETY: the init fn signature matches what
// sqlite expects; the `Once` guard prevents double-registration.
static VEC_REGISTERED: Once = Once::new();

/// Register sqlite-vec globally. Safe to call any number of times.
///
/// # Panics
/// Panics if the registration itself fails (the underlying
/// `sqlite3_auto_extension` returns non-zero). This indicates a build or ABI
/// mismatch and is not recoverable at runtime.
fn ensure_vec_loaded() {
    VEC_REGISTERED.call_once(|| {
        // SAFETY: `sqlite3_vec_init` is the entry point the sqlite-vec crate
        // exports precisely for this use. The transmute matches the function
        // pointer type sqlite expects (`sqlite3_init_routine`). This is the
        // pattern from the official sqlite-vec Rust demo.
        unsafe {
            use rusqlite::ffi::sqlite3_auto_extension;
            use sqlite_vec::sqlite3_vec_init;
            let rc = sqlite3_auto_extension(Some(std::mem::transmute(
                sqlite3_vec_init as *const (),
            )));
            if rc != 0 {
                panic!("sqlite3_auto_extension failed with rc={rc}");
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// Hybrid search engine. Owns one SQLite connection (behind a Mutex) and an
/// embedder. Generic over `E` so tests inject a [`crate::memory_embedder::StubEmbedder`]
/// and production injects a future `LlamaCppEmbedder` — same retrieval code,
/// different embedding backend, no dyn-dispatch overhead.
///
/// Construct via [`MemoryEngine::open`]. Hold behind `Arc<tokio::sync::Mutex<...>>`
/// in `AppState` once wired (Phase 2.5).
pub struct MemoryEngine<E: Embedder> {
    /// Behind its own `Arc<Mutex>` so blocking SQLite work can move onto
    /// `spawn_blocking` WITHOUT holding a `&mut` borrow of `MemoryEngine`
    /// (the closure needs `'static`; `&mut self` isn't `'static`). Mirrors
    /// the double-`Arc<Mutex<...>>` pattern used by `AppState::active_cancel`.
    conn: Arc<Mutex<Connection>>,
    embedder: E,
}

impl<E: Embedder> MemoryEngine<E> {
    /// Open (or create) the SQLite database at `path`, run the schema, and
    /// register sqlite-vec. Returns an engine ready for `add_memory` / `search`.
    ///
    /// The connection is opened with `create_if_missing`; first-open creates
    /// the file + all three tables. Subsequent opens skip table creation
    /// (`CREATE ... IF NOT EXISTS` is idempotent).
    pub fn open(path: &Path, embedder: E) -> anyhow::Result<Self> {
        // Embedder contract: must agree with EMBED_DIM (and therefore the
        // vec0 DDL). Check at construction so a wrong embedder fails here,
        // not at the first insert.
        anyhow::ensure!(
            embedder.dim() == EMBED_DIM,
            "embedder dim {} != EMBED_DIM {} (vec0 DDL is float[{}])",
            embedder.dim(),
            EMBED_DIM,
            EMBED_DIM
        );

        ensure_vec_loaded();

        let conn = Connection::open(path)
            .map_err(|e| anyhow::anyhow!("open memory db: {e:?}"))?;

        // WAL: concurrent readers (the future debug IPC) + one writer
        // (add_memory) without blocking each other. Cheap on SSD, big win
        // once the observability panel lands.
        conn.pragma_update(None, "journal_mode", "WAL")
            .map_err(|e| anyhow::anyhow!("set WAL: {e:?}"))?;

        init_schema(&conn)?;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            embedder,
        })
    }

    /// Embed the text, then insert into all three tables in one transaction.
    ///
    /// Returns the new memory's id. `chunk_index` defaults to 0 (whole-message,
    /// no chunking yet — verdict I) and `metadata_json` to `None`.
    pub async fn add_memory(
        &self,
        text: String,
        role: Role,
        salience: f32,
    ) -> anyhow::Result<MemoryId> {
        // Embed on the Tokio worker. Real backends will spend milliseconds
        // here on GPU work; the StubEmbedder is microseconds. Either way the
        // embedder owns its own threading story (a dedicated thread + channel
        // for llama-cpp-2, since the context is !Send — same pattern as the
        // chat engine).
        let embedding = self.embedder.embed(text.clone()).await?;

        // Clone the Arc (cheap — one atomic increment), move into the closure.
        // The Mutex guard is acquired INSIDE the blocking closure, never held
        // across an await. Same shape as save_session in lib.rs §2E.
        let conn = self.conn.clone();
        let id = tokio::task::spawn_blocking(move || -> anyhow::Result<MemoryId> {
            let c = conn.lock().expect("memory conn mutex");
            insert_in_transaction(&c, &text, role, salience, 0, None, &embedding)
        })
        .await
        .map_err(|e| anyhow::anyhow!("add_memory join: {e}"))??;
        Ok(id)
    }

    /// Hybrid search: embed the query, pull top-N from each backend, fuse
    /// via RRF, hydrate the top `limit` into full [`MemoryEntry`] records.
    ///
    /// `N` (per-list retrieval depth) is intentionally larger than `limit`
    /// so RRF has overlap to work with — a memory at dense-rank 30 may still
    /// be a strong sparse match and deserve promotion.
    pub async fn search(&self, query: &str, limit: usize) -> anyhow::Result<Vec<RankedMemory>> {
        const RETRIEVAL_DEPTH: usize = 64; // verdict B, 2026-07-13.

        let embedding = self.embedder.embed(query.to_owned()).await?;

        // query is borrowed; the closure needs 'static, so take an owned copy.
        let query_owned = query.to_owned();
        let conn = self.conn.clone();
        // Wrap in Ok(...) to match add_memory's shape: the inner ?? unwraps
        // both the JoinError layer (.map_err + ?) AND the closure's own
        // Result layer (?), yielding Vec<RankedMemory>; Ok wraps it back to
        // match the return type. Single ? would also work (inner Result
        // passes through as the Ok value) but this keeps the two methods
        // structurally parallel.
        Ok(tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<RankedMemory>> {
            let c = conn.lock().expect("memory conn mutex");
            // Degrade to dense-only if FTS5 fails. The sparse and dense paths
            // are independent backends — a syntax error in one (e.g. an FTS5
            // operator char that slipped past sanitization) must not kill the
            // other. fuse_rrf handles an empty sparse list cleanly (dense
            // results keep their 1-based ranks). Logged at warn so a
            // recurrence is visible without breaking the turn.
            let sparse = match fts5_top_k(&c, &query_owned, RETRIEVAL_DEPTH) {
                Ok(ids) => ids,
                Err(e) => {
                    tracing::warn!(error = %format!("{e:#}"), "fts5 search failed; dense-only this turn");
                    Vec::new()
                }
            };
            let dense = vec0_top_k(&c, &embedding, RETRIEVAL_DEPTH)?;
            let fused = fuse_rrf(&sparse, &dense, limit);
            fetch_entries(&c, &fused)
        })
        .await
        .map_err(|e| anyhow::anyhow!("search join: {e}"))??)
    }
}

// ---------------------------------------------------------------------------
// Schema + private sync helpers (all run on the blocking thread)
// ---------------------------------------------------------------------------

/// Create the three tables if they don't exist. Idempotent.
///
/// The `vec0` dimension interpolates [`EMBED_DIM`] so the DDL can't drift from
/// the embedder contract — a swap to a different `Embed.gguf` fails at open
/// time (the const changes, the schema is re-issued against the new file),
/// not at first insert.
fn init_schema(conn: &Connection) -> anyhow::Result<()> {
    conn.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS memories (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            text_content   TEXT NOT NULL,
            timestamp      INTEGER NOT NULL,
            role           TEXT NOT NULL,
            chunk_index    INTEGER NOT NULL DEFAULT 0,
            salience       REAL NOT NULL DEFAULT 0.5,
            metadata_json  TEXT
        );

        -- FTS5 mirror. text_content is duplicated here (also in `memories`) —
        -- disk is cheap; external-content tables add trigger complexity not
        -- worth it for v1 (verdict G, 2026-07-13).
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(text_content);
        "#,
    )
    .map_err(|e| anyhow::anyhow!("create core+fts tables: {e:?}"))?;

    // vec0 DDL separately — its dimension comes from a const, so build the
    // statement with format!. (vec0's parser is picky; keep the literal clean.)
    let vec_ddl = format!(
        "CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(embedding float[{dim}]);",
        dim = EMBED_DIM
    );
    conn.execute_batch(&vec_ddl)
        .map_err(|e| anyhow::anyhow!("create vec0 table: {e:?}"))?;

    Ok(())
}

/// Insert one memory into all three tables inside a single transaction.
///
/// `memories` is written first to mint the id via `last_insert_rowid()`; that
/// id is then reused as the `rowid` for `memories_fts` and `memories_vec`.
/// If any step fails, `execute_batch`'s implicit transaction rolls back —
/// no orphaned keyword-searchable row missing its vector (or vice versa).
///
/// The embedding bytes are little-endian f32 — the wire format vec0 expects.
#[allow(clippy::too_many_arguments)]
fn insert_in_transaction(
    conn: &Connection,
    text: &str,
    role: Role,
    salience: f32,
    chunk_index: i32,
    metadata_json: Option<&str>,
    embedding: &[f32],
) -> anyhow::Result<MemoryId> {
    // Defensive: vec0 will reject a wrong-length blob with an opaque error;
    // catch it here with a clear message.
    anyhow::ensure!(
        embedding.len() == EMBED_DIM,
        "embedding length {} != EMBED_DIM {}",
        embedding.len(),
        EMBED_DIM
    );

    let ts = unix_now();

    let tx = conn
        .unchecked_transaction()
        .map_err(|e| anyhow::anyhow!("begin txn: {e:?}"))?;

    // 1. Mint the id from the core table. `Option<&str>` implements ToSql
    // directly (None → SQL NULL, Some → TEXT) — no intermediate dyn indirection
    // needed (which would borrow a local pattern binding and fail E0597).
    tx.execute(
        "INSERT INTO memories (text_content, timestamp, role, chunk_index, salience, metadata_json)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![text, ts, role.as_str(), chunk_index, salience, metadata_json],
    )
    .map_err(|e| anyhow::anyhow!("insert memories: {e:?}"))?;

    let id = tx.last_insert_rowid();

    // 2. FTS5 mirror, same rowid.
    tx.execute(
        "INSERT INTO memories_fts (rowid, text_content) VALUES (?1, ?2)",
        params![id, text],
    )
    .map_err(|e| anyhow::anyhow!("insert memories_fts: {e:?}"))?;

    // 3. vec0, same rowid. Embedding as raw LE f32 bytes.
    let emb_bytes = embed_to_bytes(embedding);
    tx.execute(
        "INSERT INTO memories_vec (rowid, embedding) VALUES (?1, ?2)",
        params![id, emb_bytes],
    )
    .map_err(|e| anyhow::anyhow!("insert memories_vec: {e:?}"))?;

    tx.commit()
        .map_err(|e| anyhow::anyhow!("commit txn: {e:?}"))?;

    Ok(id)
}

/// BM25 keyword search. Returns rowids best-first, up to `limit`.
///
/// The raw query is sanitized via [`sanitize_fts5_query`] before being passed
/// to FTS5's MATCH operator — FTS5 interprets `!`, `*`, `"`, `(`, `)`, `:` as
/// query-syntax operators, so unsanitized user input trips a syntax error on
/// the first punctuation mark (verified at runtime 2026-07-13: "Hello there
/// Wupi!" → `fts5: syntax error near "!"`). Phrase-quoting each whitespace
/// token neutralizes every operator char; FTS5's tokenizer then strips
/// punctuation inside the quotes, so `"Wupi!"` matches the indexed token
/// `wupi`. Empty/whitespace-only input short-circuits to an empty result
/// (no sparse contribution — dense-only retrieval).
fn fts5_top_k(conn: &Connection, query: &str, limit: usize) -> anyhow::Result<Vec<MemoryId>> {
    let sanitized = sanitize_fts5_query(query);
    if sanitized.is_empty() {
        return Ok(Vec::new());
    }
    let mut stmt = conn
        .prepare(
            "SELECT rowid FROM memories_fts
             WHERE memories_fts MATCH ?1
             ORDER BY bm25(memories_fts) ASC
             LIMIT ?2",
        )
        .map_err(|e| anyhow::anyhow!("prepare fts5: {e:?}"))?;

    let rows = stmt
        .query_map(params![&sanitized, limit as i64], |r| r.get::<_, MemoryId>(0))
        .map_err(|e| anyhow::anyhow!("query fts5: {e:?}"))?;

    let mut out = Vec::new();
    for r in rows {
        out.push(r.map_err(|e| anyhow::anyhow!("fts5 row: {e:?}"))?);
    }
    Ok(out)
}

/// Turn raw user text into a safe FTS5 MATCH query.
///
/// Splits on ASCII whitespace and wraps each token as a double-quoted FTS5
/// phrase. Phrase-quoted tokens are re-tokenized by FTS5's own tokenizer
/// (unicode61 strips punctuation), so operator characters like `!`, `*`, `"`
/// lose their special meaning. Internal double-quotes are escaped by doubling
/// (`""`), per FTS5's phrase-escape rule. Multiple quoted tokens form an
/// implicit-AND query.
///
/// Returns an empty string for empty/whitespace-only input — callers should
/// treat that as "no sparse query" (the dense path still runs).
fn sanitize_fts5_query(input: &str) -> String {
    input
        .split_whitespace()
        .map(|tok| {
            // Escape any literal `"` inside the token by doubling it.
            let escaped = tok.replace('"', "\"\"");
            format!("\"{escaped}\"")
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Cosine (dense) search. Returns rowids best-first (smallest distance), up
/// to `limit`. vec0's `distance` for cosine is `1 - cos_sim`, so ASC order
/// already puts the most-similar first.
fn vec0_top_k(
    conn: &Connection,
    query_embedding: &[f32],
    limit: usize,
) -> anyhow::Result<Vec<MemoryId>> {
    let emb_bytes = embed_to_bytes(query_embedding);
    let mut stmt = conn
        .prepare(
            "SELECT rowid FROM memories_vec
             WHERE embedding MATCH ?1
             ORDER BY distance
             LIMIT ?2",
        )
        .map_err(|e| anyhow::anyhow!("prepare vec0: {e:?}"))?;

    let rows = stmt
        .query_map(params![emb_bytes, limit as i64], |r| r.get::<_, MemoryId>(0))
        .map_err(|e| anyhow::anyhow!("query vec0: {e:?}"))?;

    let mut out = Vec::new();
    for r in rows {
        out.push(r.map_err(|e| anyhow::anyhow!("vec0 row: {e:?}"))?);
    }
    Ok(out)
}

/// Hydrate fused ids into full entries, preserving fused order + score.
///
/// Issues a single `SELECT ... WHERE id IN (...)`. For the small `limit`s
/// this engine serves (<=64), binding N params is cheaper than a JOIN against
/// a values-list and avoids SQLite's per-statement prepare overhead.
fn fetch_entries(conn: &Connection, fused: &[RankedMemory]) -> anyhow::Result<Vec<RankedMemory>> {
    if fused.is_empty() {
        return Ok(Vec::new());
    }

    // Build `id IN (?1, ?2, ...)` with one placeholder per id.
    let placeholders: String = (0..fused.len())
        .map(|i| format!("?{}", i + 1))
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!(
        "SELECT id, text_content, timestamp, role, chunk_index, salience, metadata_json
         FROM memories
         WHERE id IN ({placeholders})"
    );

    let mut stmt = conn
        .prepare(&sql)
        .map_err(|e| anyhow::anyhow!("prepare fetch_entries: {e:?}"))?;

    // Bind each id.
    let mut params_slice: Vec<&dyn rusqlite::ToSql> =
        Vec::with_capacity(fused.len());
    for r in fused {
        params_slice.push(&r.entry.id);
    }

    let rows = stmt
        .query_map(params_slice.as_slice(), |r| {
            let metadata_json: Option<String> = r.get(6)?;
            let role_str: String = r.get(3)?;
            Ok(MemoryEntry {
                id: r.get(0)?,
                text_content: r.get(1)?,
                timestamp: r.get(2)?,
                role: Role::parse(&role_str)
                    .map_err(|e| rusqlite::Error::ToSqlConversionFailure(e.into()))?,
                chunk_index: r.get(4)?,
                salience: r.get(5)?,
                metadata_json,
            })
        })
        .map_err(|e| anyhow::anyhow!("query fetch_entries: {e:?}"))?;

    // Collect into a map for order-preserving reassembly.
    let mut by_id: std::collections::HashMap<MemoryId, MemoryEntry> = std::collections::HashMap::new();
    for r in rows {
        let entry = r.map_err(|e| anyhow::anyhow!("fetch_entries row: {e:?}"))?;
        by_id.insert(entry.id, entry);
    }

    // Walk `fused` in score order, attaching the hydrated entry. If an id is
    // missing from the map (row deleted between query and fetch — a narrow
    // race), drop it silently rather than return a partial entry.
    let mut out = Vec::with_capacity(fused.len());
    for r in fused {
        if let Some(entry) = by_id.remove(&r.entry.id) {
            out.push(RankedMemory {
                entry,
                score: r.score,
            });
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Small utilities
// ---------------------------------------------------------------------------

/// Serialize an embedding as raw little-endian f32 bytes — vec0's wire format.
///
/// One alloc per embed. A `zerocopy::AsBytes` cast would be zero-alloc but
/// adds a dependency for a single call site; the cost (~1.5 KB alloc per
/// embed, amortized over a multi-millisecond GPU embed) is noise.
fn embed_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for f in v {
        out.extend_from_slice(&f.to_le_bytes());
    }
    out
}

/// Unix epoch seconds. Centralized so tests can inject a fixed value when
/// they need deterministic timestamps (none do yet — v1 tests check RRF, not
/// timestamps — but the indirection is here for when they do).
fn unix_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

// v1 scaffolding: this module compiles but is not yet wired into AppState.
// Suppress the unused warnings rather than gut the public surface — the whole
// point of this phase is to land the foundation Phase 2.5 calls.
#[allow(dead_code)]
fn _silence_unused_marker() {
    let _ = (Role::User, MemoryId::MIN, MemoryEntry {
        id: 0,
        text_content: String::new(),
        timestamp: 0,
        role: Role::User,
        chunk_index: 0,
        salience: 0.0,
        metadata_json: None,
    });
}
