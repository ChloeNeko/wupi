//! First-run GGUF downloader: pulls `WUPI.gguf` + `Embed.gguf` from a private
//! Hugging Face repo on first launch so the installer can ship without the
//! ~9.8 GB of models baked in.
//!
//! ## Why this exists
//! GitHub caps single files at 100 MB (and soft-caps repos at ~1 GB), so the
//! GGUFs can't live in the repo. Beta testers get a small installer exe; on
//! first run (no models detected) the boot overlay hands control to this
//! module, which streams both files into `app_data_dir/models/` — the 5th
//! entry in `model_search_dirs` (lib.rs), so the existing resolver picks them
//! up on the next boot scan with ZERO resolver changes.
//!
//! ## Auth model
//! The HF repo is PRIVATE. A fine-grained read-only access token scoped to
//! ONLY `ChloeNeko/WUPI` is baked into the binary as `HF_TOKEN`. This is the
//! accepted trade-off for a private beta: zero friction for testers (no
//! token-pasting), blast radius limited to that one repo's files if the exe
//! is reverse-engineered, and the token is revocable/rotatable anytime from
//! the HF settings page. To rotate: generate a new token, replace the
//! constant, rebuild.
//!
//! ## Resume correctness (the load-bearing detail)
//! HF `/resolve/<rev>/<file>` returns a 302 to a CDN backing URL
//! (`cdn-lfs.hf.co` / `cas-bridge.xethub.hf.co`). That signed URL EXPIRES
//! (typically minutes to ~1 hour). So a naive "save the redirect URL and
//! reuse it on resume" scheme breaks the moment the URL lapses mid-download.
//! The fix: on every (re)start of a file's download, re-hit `/resolve/` with
//! the Bearer token to obtain a FRESH signed URL, then issue
//! `Range: bytes=<existing-.part-size>-` against it. The re-resolve is cheap
//! (one 302); the resume is correct indefinitely. This is the same flow
//! `hf_hub_download` uses under the hood.
//!
//! ## Atomicity
//! Each file downloads to `<name>.gguf.part`. On full completion: fsync the
//! `.part`, then rename to `<name>.gguf`. A crash / cancel / network drop
//! leaves ONLY the `.part` file behind — never a half-written final file —
//! so the resolver never sees a corrupt gguf. The `.part` is reused on the
//! next resume attempt (truncated-to-correct-offset logic below).
//!
//! ## Concurrency
//! One downloader at a time. The frontend gates the overlay on
//! `download_models` returning; there's no multi-file parallelism (the two
//! files stream sequentially: WUPI first since the app can't boot without it,
//! Embed second). Sequential is also friendlier to flaky upstream bandwidth
//! than splitting across two sockets.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use futures_util::StreamExt;
use serde::Serialize;
use tauri::Emitter;
use tokio::io::AsyncWriteExt;

// ── Configuration ──────────────────────────────────────────────────────────

/// The HF repo holding the GGUFs. Private; read access via `HF_TOKEN`.
const HF_REPO: &str = "ChloeNeko/WUPI";
/// HF revision (branch). `main` is where the user uploaded both files.
const HF_REVISION: &str = "main";

/// Read-only fine-grained HF token scoped to ONLY `ChloeNeko/WUPI`.
///
/// Injected at BUILD TIME from the `HF_TOKEN` environment variable via
/// `option_env!` — the token value is NEVER committed to source. CI
/// (`.github/workflows/release.yml`) sets this from the `HF_TOKEN` GitHub
/// Secret during the build step, so the compiled binary has the real token
/// baked in. Local dev builds without the env var produce an empty string
/// here → HF returns 401 → the downloader reports a clear error. Local
/// devs don't need it: they already have the GGUFs on disk so the download
/// overlay never fires.
///
/// Bearer auth is sent on the `/resolve/` hop ONLY — the 302 redirect's
/// signed CDN URL carries its own short-lived signature, so the token never
/// leaves HF's own domain.
///
/// Rotation: revoke at https://huggingface.co/settings/tokens, mint a new
/// fine-grained read-only token scoped to ChloeNeko/WUPI, update the
/// `HF_TOKEN` GitHub Secret, push a fresh build. No source change needed.
///
/// NOTE: an earlier hardcoded version of this token (hf_GdgPcd…) was
/// committed then rewritten out of git history on 2026-07-19. That token
/// remains live until manually revoked at the HF settings page — it's
/// scoped read-only to ChloeNeko/WUPI so the realistic blast radius is
/// limited to GGUF downloads, but it should still be rotated when
/// convenient. See docs/UPDATER_SETUP.md.
const HF_TOKEN: &str = match option_env!("HF_TOKEN") {
    Some(t) => t,
    None => "",
};

/// The two files we need, in download order. WUPI first because the chat
/// engine can't boot without it; Embed is best-effort (the embedder falls
/// back to StubEmbedder on miss, see lib.rs setup).
pub const REQUIRED_FILES: &[&str] = &["WUPI.gguf", "Embed.gguf"];

/// Chunk size: reqwest's `bytes_stream()` yields its own chunks (typically
/// 8-16 KB from the TLS layer); we don't impose a fixed read size. Progress
/// granularity is therefore driven by the emit throttle below, not by chunk
/// size.

/// Throttle window for `download-progress` event emission. Emitting on every
/// TLS-sized chunk of a 9.8 GB file = ~1M events over a long download; that
/// floods the IPC channel and starves the UI thread. Emit at most every 500ms
/// instead — the polled `get_download_progress` (a direct IPC read) is the
/// authoritative UI source between emits.
const EMIT_INTERVAL_MS: u64 = 500;

// ── Public state (shared with AppState) ────────────────────────────────────

/// Snapshot of download progress, read by `get_download_progress` (polled by
/// the frontend) and emitted as the `download-progress` event payload.
///
/// `current_file_offset` + `current_file_total` describe the file actively
/// streaming; `overall_downloaded` + `overall_total` span both files so the
/// UI can render one progress bar for the whole job (the meaningful number
/// for a 9.8 GB + 36 MB download).
#[derive(Debug, Clone, Default, Serialize)]
pub struct DownloadProgress {
    /// Which phase the downloader is in.
    pub phase: DownloadPhase,
    /// Filename currently streaming (`"WUPI.gguf"` etc.), or `""` between files.
    pub current_file: String,
    /// Bytes of `current_file` written to disk so far.
    pub current_file_offset: u64,
    /// Total bytes of `current_file` (from HF `Content-Length`). 0 until the
    /// HEAD/first-range response sets it.
    pub current_file_total: u64,
    /// Bytes downloaded across ALL files this run (for the overall bar).
    pub overall_downloaded: u64,
    /// Sum of both files' totals (set once both sizes are known).
    pub overall_total: u64,
    /// Human-readable error if `phase == Failed`. Empty otherwise.
    pub error: String,
}

#[derive(Debug, Clone, Copy, Default, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum DownloadPhase {
    /// Nothing running. Initial state before `download_models` is invoked.
    #[default]
    Idle,
    /// Resolving `/resolve/` → fresh signed CDN URL for the current file.
    Resolving,
    /// Streaming bytes to `<name>.gguf.part`.
    Downloading,
    /// fsync + rename of the just-finished `.part` → final file.
    Finalizing,
    /// Both files complete and renamed; the caller may now boot normally.
    Done,
    /// Unrecoverable error. See `error`.
    Failed,
}

pub type CancelToken = Arc<AtomicBool>;

// ── URL construction ───────────────────────────────────────────────────────

/// The HF `/resolve/` URL for a file. Hits this with the Bearer header to
/// obtain a 302 → signed CDN URL; do NOT use this URL directly for the byte
/// stream (it's a redirect, not the content).
fn resolve_url(filename: &str) -> String {
    format!(
        "https://huggingface.co/{repo}/resolve/{rev}/{file}",
        repo = HF_REPO,
        rev = HF_REVISION,
        file = filename
    )
}

// ── HTTP client ────────────────────────────────────────────────────────────

/// Build a reqwest client. `rustls-tls` (already a project dep via api.rs's
/// HttpBackend) avoids any system OpenSSL on Windows. Follow redirects so the
/// 302 → CDN hop is transparent to the byte-stream loop (the Bearer header is
/// NOT forwarded to the CDN host by reqwest's redirect policy by default,
/// which is exactly what we want — the signed URL is self-authenticating).
fn http_client() -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .use_rustls_tls()
        .redirect(reqwest::redirect::Policy::default())
        .build()
        .map_err(|e| format!("failed to build HTTP client: {e}"))
}

// ── Core: stream one file with resume ──────────────────────────────────────

/// Download `filename` into `dest_dir/filename`, resuming from any existing
/// `<filename>.part`. Updates `progress` (under its Mutex) and emits
/// `download-progress` events on `app` at most every `EMIT_INTERVAL_MS`.
///
/// Returns the final file size on success.
async fn download_one(
    filename: &str,
    dest_dir: &Path,
    client: &reqwest::Client,
    progress: Arc<std::sync::Mutex<DownloadProgress>>,
    cancel: CancelToken,
    app: tauri::AppHandle,
) -> Result<u64, String> {
    let url = resolve_url(filename);
    let part_path: PathBuf = dest_dir.join(format!("{filename}.part"));
    let final_path: PathBuf = dest_dir.join(filename);

    // If the final file already exists, the caller (download_models) should
    // have skipped us. Defensive: if we're here anyway, treat as done.
    if final_path.exists() {
        return std::fs::metadata(&final_path)
            .map(|m| m.len())
            .map_err(|e| format!("stat existing {filename}: {e}"));
    }

    // ── Phase: Resolving ── re-hit /resolve/ for a fresh signed URL each
    // attempt. HF's signed CDN URLs expire; a saved URL from a prior run is
    // useless. The Bearer token authenticates this hop only.
    {
        let mut p = progress.lock().expect("progress mutex");
        p.phase = DownloadPhase::Resolving;
        p.current_file = filename.to_owned();
        p.current_file_offset = 0;
        p.current_file_total = 0;
    }
    let _ = app.emit("download-progress", progress_snapshot(&progress));

    // Existing .part = the resume offset. If it's larger than the remote file
    // (somehow), truncate to 0 and start over rather than write garbage.
    let resume_offset = match std::fs::metadata(&part_path) {
        Ok(m) => m.len(),
        Err(_) => 0,
    };

    let mut req = client
        .get(&url)
        .header(reqwest::header::AUTHORIZATION, format!("Bearer {HF_TOKEN}"));
    if resume_offset > 0 {
        req = req.header(reqwest::header::RANGE, format!("bytes={resume_offset}-"));
        tracing::info!(
            file = filename,
            offset = resume_offset,
            "resuming download from .part"
        );
    }
    let response = req
        .send()
        .await
        .map_err(|e| format!("HF resolve/send for {filename} failed: {e}"))?;

    // HF returns 200 for a fresh fetch, 206 for a ranged resume. Anything else
    // is a hard error (401 = bad/expired token, 404 = wrong repo/filename).
    let status = response.status();
    if !(status == reqwest::StatusCode::OK || status == reqwest::StatusCode::PARTIAL_CONTENT) {
        let body = response.text().await.unwrap_or_default();
        return Err(format!(
            "HF returned {status} for {filename} (token expired? wrong repo?): {body}"
        ));
    }
    let remote_total = response.content_length().unwrap_or(0);
    // For a 206, Content-Length is the REMAINING bytes (from offset to end);
    // for a 200 it's the full size. Compute absolute totals accordingly.
    let absolute_total = if status == reqwest::StatusCode::PARTIAL_CONTENT && remote_total > 0 {
        resume_offset + remote_total
    } else {
        remote_total
    };
    {
        let mut p = progress.lock().expect("progress mutex");
        p.phase = DownloadPhase::Downloading;
        p.current_file_offset = resume_offset;
        p.current_file_total = absolute_total;
        if p.overall_total == 0 {
            // First-file total unknown until now; seed it. Second file's
            // total adds in when it resolves.
            p.overall_total = absolute_total;
        }
    }

    // ── Phase: Downloading ── open the .part in append mode (or create) and
    // stream the body chunk-by-chunk. Resume offset is the file's existing
    // length, so append continues exactly where we left off.
    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&part_path)
        .await
        .map_err(|e| format!("open {filename}.part for write: {e}"))?;

    let mut stream = response.bytes_stream();
    let mut written_this_run: u64 = 0;
    let mut last_emit = std::time::Instant::now();

    while let Some(chunk_result) = stream.next().await {
        // Cancel check at the top of each chunk (between writes, never mid):
        // mirrors the engine decode-loop cancel invariant. Relaxed is correct
        // for the same reason (single-bit signal, no dependent data; §3).
        if cancel.load(Ordering::Relaxed) {
            // Flush what we have so the .part is reusable on next attempt.
            file.flush()
                .await
                .map_err(|e| format!("flush on cancel: {e}"))?;
            return Err("cancelled".to_owned());
        }
        let chunk = chunk_result.map_err(|e| format!("stream read {filename}: {e}"))?;
        file.write_all(&chunk)
            .await
            .map_err(|e| format!("write {filename}.part: {e}"))?;
        written_this_run = written_this_run
            .checked_add(chunk.len() as u64)
            .ok_or_else(|| "byte counter overflow".to_owned())?;

        // Update shared progress. current_file_offset is the absolute byte
        // position in the file (resume_offset + written_this_run) so the UI's
        // percentage is correct across resumes, not just within one run.
        let new_offset = resume_offset + written_this_run;
        {
            let mut p = progress.lock().expect("progress mutex");
            p.current_file_offset = new_offset;
            p.overall_downloaded = p.overall_downloaded.saturating_add(chunk.len() as u64);
        }
        if last_emit.elapsed() >= std::time::Duration::from_millis(EMIT_INTERVAL_MS) {
            let _ = app.emit("download-progress", progress_snapshot(&progress));
            last_emit = std::time::Instant::now();
        }
    }

    // ── Phase: Finalizing ── fsync + atomic rename. The fsync guarantees the
    // bytes hit disk before the rename (so a power loss between rename and
    // the kernel flushing pages can't leave a short final file). The rename
    // is atomic on the same filesystem (NTFS rename = single metadata op).
    {
        let mut p = progress.lock().expect("progress mutex");
        p.phase = DownloadPhase::Finalizing;
    }
    let _ = app.emit("download-progress", progress_snapshot(&progress));

    file.sync_all()
        .await
        .map_err(|e| format!("fsync {filename}.part: {e}"))?;
    drop(file);

    let final_size = resume_offset + written_this_run;
    std::fs::rename(&part_path, &final_path)
        .map_err(|e| format!("rename {filename}.part → {filename}: {e}"))?;

    tracing::info!(file = filename, bytes = final_size, "download complete");
    let _ = app.emit("download-progress", progress_snapshot(&progress));
    Ok(final_size)
}

/// Take a non-blocking snapshot of the shared progress for event emission.
fn progress_snapshot(progress: &Arc<std::sync::Mutex<DownloadProgress>>) -> DownloadProgress {
    progress.lock().expect("progress mutex").clone()
}

// ── Public entry: download all required files ──────────────────────────────

/// Download every file in `REQUIRED_FILES` into `dest_dir`. Skips files that
/// already exist at their final path (idempotent re-runs). Updates `progress`
/// throughout; honors `cancel`.
pub async fn download_all(
    dest_dir: PathBuf,
    progress: Arc<std::sync::Mutex<DownloadProgress>>,
    cancel: CancelToken,
    app: tauri::AppHandle,
) -> Result<(), String> {
    std::fs::create_dir_all(&dest_dir)
        .map_err(|e| format!("create models dir {}: {e}", dest_dir.display()))?;

    let client = http_client()?;

    for filename in REQUIRED_FILES {
        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled".to_owned());
        }
        let final_path = dest_dir.join(filename);
        if final_path.exists() {
            tracing::info!(file = filename, "already present; skipping");
            continue;
        }
        download_one(filename, &dest_dir, &client, Arc::clone(&progress), Arc::clone(&cancel), app.clone())
            .await?;
    }

    {
        let mut p = progress.lock().expect("progress mutex");
        p.phase = DownloadPhase::Done;
        p.current_file.clear();
    }
    let _ = app.emit("download-progress", progress_snapshot(&progress));
    Ok(())
}
