//! Portable self-updater: file-level replacement with user-data preservation.
//!
//! Replaces the Tauri updater plugin (which is installer-only and can't update
//! a portable exe). WUPI ships as a portable zip; this module downloads a new
//! zip from the GitHub release, extracts it in place, and replaces engine
//! files while preserving all four user-data top-level dirs (§8C):
//!   - `data/`   (user.xml, theme.json, api_config.json, docs/, _update/)
//!   - `memory/` (memory.sqlite)
//!   - `models/` (WUPI.gguf, Embed.gguf)
//!   - `apps/`   (per-card sessions, schemas, scenario cards, profiles)
//!
//! ## The preserve rule
//!
//! The portable zip ships engine content + the empty `data/` seed (wupi.sim +
//! user.xml only). It never ships `memory/`, `models/`, or `apps/` (release.cjs
//! excludes them — fresh extracts have no runtime state). So the rule is:
//!
//! ```text
//! for each file in the zip:
//!     // Preserved user data (the four top-level dirs). Within data/, only
//!     // wupi.sim is engine content and gets overwritten on update; user.xml
//!     // is preserved so the user's identity survives.
//!     if rel starts with "data/" AND rel != "data/wupi.sim": skip
//!     if rel starts with "memory/": skip (defensive; zip shouldn't have it)
//!     if rel starts with "models/": skip (defensive; zip shouldn't have it)
//!     if rel starts with "apps/":   skip (defensive; zip shouldn't have it)
//!     else if the file is wupi.exe: apply the rename-and-relaunch dance
//!     else: overwrite the destination in place
//! ```
//!
//! The four-dir carve-out is the entire preservation contract. No per-file
//! classification list.
//!
//! ## The Windows locked-exe dance
//!
//! Windows locks the running `wupi.exe`: it can be renamed but not deleted or
//! overwritten while the process is alive. The update sequence is therefore:
//!
//! 1. Download `portable.zip` to `<exe_dir>/data/_update/portable.zip.part`.
//! 2. Verify (deferred for the beta — HTTPS + GitHub release auth is the
//!    trust boundary; signature verification can be layered on later without
//!    changing this flow).
//! 3. Extract the zip into `<exe_dir>/data/_update/extracted/`.
//! 4. For each file in the extract: apply the preserve rule (above).
//!    - For `wupi.exe`: rename the current exe to `wupi.exe.old` (Windows
//!      permits renaming a running binary), then move the new exe into place.
//!      `wupi.exe.old` is deleted on the next boot (see `cleanup_old_exe`).
//! 5. Emit `update-applied`; the frontend prompts the user to restart. On
//!    restart, `app.restart()` relaunches the new exe and the old process
//!    exits (the OS releases its lock, allowing `wupi.exe.old` cleanup).
//!
//! ## Why not auto-restart
//!
//! The user always clicks "Restart now." A silent restart mid-session would
//! discard any in-flight generation; the click is the consent gate.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// The URL of the static manifest the updater polls. Published to gh-pages by
/// `scripts/release.cjs`. Same endpoint the old Tauri updater used.
const MANIFEST_URL: &str = "https://chloeneko.github.io/WUPI/updater/latest.json";

/// The result of [`check_for_updates`]: a new version is available, with its
/// version string, the portable-zip URL, and the release notes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateInfo {
    pub version: String,
    pub url: String,
    pub notes: String,
}

/// The manifest shape published by `release.cjs`. Mirrors the old Tauri
/// manifest fields (version/notes/pub_date) so the publish side barely
/// changes; we ignore the per-platform signature block (Tauri-specific).
#[derive(Debug, Deserialize)]
struct Manifest {
    version: String,
    notes: Option<String>,
    platforms: std::collections::HashMap<String, PlatformEntry>,
}

#[derive(Debug, Deserialize)]
struct PlatformEntry {
    url: Option<String>,
    // `signature` ignored — we don't verify minisig (deferred).
}

/// Poll the manifest; return `Some(UpdateInfo)` if `remote_version > current`.
/// Returns `None` when the manifest is unreachable, malformed, or already
/// up-to-date. Errors are logged-and-swallowed: the updater never blocks boot
/// and never surfaces fetch failures as user-visible errors (best-effort).
pub async fn check_for_updates(current_version: &str) -> Option<UpdateInfo> {
    let bytes = match fetch_manifest().await {
        Ok(b) => b,
        Err(e) => {
            tracing::info!(?e, "updater: manifest fetch failed (offline?)");
            return None;
        }
    };
    let manifest: Manifest = match serde_json::from_slice(&bytes) {
        Ok(m) => m,
        Err(e) => {
            tracing::warn!(?e, "updater: manifest malformed");
            return None;
        }
    };
    // Semver-ish compare: equal or older = no update. We don't pull in a semver
    // crate for one comparison; the publish side bumps version monotonically,
    // so a string compare is sufficient in practice. (If versioning ever goes
    // non-monotonic, swap in semver here.)
    if !is_newer(&manifest.version, current_version) {
        tracing::info!(
            new = %manifest.version,
            current = %current_version,
            "updater: on latest"
        );
        return None;
    }
    // Find the windows-x86_64 platform entry (the only one we publish).
    let entry = manifest.platforms.get("windows-x86_64")?;
    let url = entry.url.clone()?;
    Some(UpdateInfo {
        version: manifest.version,
        url,
        notes: manifest.notes.unwrap_or_default(),
    })
}

/// Apply a pending update: download → extract → swap files → emit event.
///
/// `app_handle` is used for path resolution (finds `<exe_dir>`) and event
/// emission. `update` is the [`UpdateInfo`] from [`check_for_updates`].
/// Progress events (`update-progress`, 0..=100) fire as the download streams.
/// The `update-applied` event fires once when the swap is complete + safe.
pub async fn perform_update(
    app_handle: &tauri::AppHandle,
    update: UpdateInfo,
) -> Result<(), String> {
    use tauri::Emitter;

    let exe_dir = exe_dir(app_handle).ok_or("could not resolve exe dir")?;
    let staging = exe_dir.join("data").join("_update");
    std::fs::create_dir_all(&staging).map_err(|e| format!("create staging: {e}"))?;

    let zip_part = staging.join("portable.zip.part");
    let zip_final = staging.join("portable.zip");

    // ── Phase 1: download with progress events ────────────────────────────
    download_with_progress(&update.url, &zip_part, app_handle).await?;

    // Atomic rename: .part → final. A crash leaves only the .part; the next
    // attempt re-downloads (correct: the zip is small enough that resuming
    // adds complexity for no real win).
    std::fs::rename(&zip_part, &zip_final).map_err(|e| format!("rename .part: {e}"))?;

    // ── Phase 2: extract ──────────────────────────────────────────────────
    let extracted = staging.join("extracted");
    // Clean slate: remove any leftover from a prior attempt.
    if extracted.exists() {
        std::fs::remove_dir_all(&extracted)
            .map_err(|e| format!("clean extracted/: {e}"))?;
    }
    std::fs::create_dir_all(&extracted).map_err(|e| format!("create extracted/: {e}"))?;
    extract_zip(&zip_final, &extracted)?;

    // ── Phase 3: apply with the preserve rule ─────────────────────────────
    apply_extracted(&extracted, &exe_dir)?;

    // ── Phase 4: cleanup staging ──────────────────────────────────────────
    // Keep the staging dir (cheap) but drop the bulky zip + extracted tree.
    let _ = std::fs::remove_dir_all(&extracted);
    let _ = std::fs::remove_file(&zip_final);

    let _ = app_handle.emit("update-applied", &update);
    Ok(())
}

/// Delete a leftover `wupi.exe.old` from a prior update. Called from `setup()`
/// on every boot — by the time the new exe runs, the old one's lock is gone.
/// Best-effort: a failure (file in use, perms) is logged and the file is
/// retried next boot.
pub fn cleanup_old_exe(app_handle: &tauri::AppHandle) {
    let Some(exe_dir) = exe_dir(app_handle) else {
        return;
    };
    let old = exe_dir.join("wupi.exe.old");
    if old.exists() {
        match std::fs::remove_file(&old) {
            Ok(()) => tracing::info!("cleaned up wupi.exe.old from prior update"),
            Err(e) => tracing::warn!(?e, "could not remove wupi.exe.old; will retry next boot"),
        }
    }
}

// ── Internals ──────────────────────────────────────────────────────────────

/// Resolve `<exe_dir>` — the directory containing `wupi.exe`.
fn exe_dir(app_handle: &tauri::AppHandle) -> Option<PathBuf> {
    let _ = app_handle; // unused on the happy path; kept for symmetry/future use
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf))
}

/// Fetch the manifest bytes. No streaming — the manifest is tiny (~1KB).
async fn fetch_manifest() -> Result<Vec<u8>, String> {
    let client = reqwest::Client::builder()
        .user_agent("wupi-updater")
        .build()
        .map_err(|e| format!("build client: {e}"))?;
    let resp = client
        .get(MANIFEST_URL)
        .send()
        .await
        .map_err(|e| format!("manifest GET: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("manifest status: {}", resp.status()));
    }
    resp.bytes()
        .await
        .map(|b| b.to_vec())
        .map_err(|e| format!("manifest body: {e}"))
}

/// New-version check. Treats the version strings as monotonic dotted numbers:
/// compares component-by-component as integers, falling back to string compare
/// on parse failure. Sufficient because `release.cjs` bumps monotonically
/// (patch/minor/major). Not a full semver impl (no prerelease tags).
fn is_newer(remote: &str, current: &str) -> bool {
    let cmp = compare_dotted(remote, current);
    matches!(cmp, std::cmp::Ordering::Greater)
}

fn compare_dotted(a: &str, b: &str) -> std::cmp::Ordering {
    let mut ait = a.split('.').fuse();
    let mut bit = b.split('.').fuse();
    loop {
        match (ait.next(), bit.next()) {
            (None, None) => return std::cmp::Ordering::Equal,
            (None, Some(_)) => return std::cmp::Ordering::Less,
            (Some(_), None) => return std::cmp::Ordering::Greater,
            (Some(a_part), Some(b_part)) => {
                let an: Option<u64> = a_part.parse().ok();
                let bn: Option<u64> = b_part.parse().ok();
                let ord = match (an, bn) {
                    (Some(an), Some(bn)) => an.cmp(&bn),
                    _ => a_part.cmp(b_part),
                };
                if ord != std::cmp::Ordering::Equal {
                    return ord;
                }
            }
        }
    }
}

/// Stream-download `url` into `dest`, emitting `update-progress` events with
/// the percentage complete. The total size is taken from the Content-Length
/// header; if absent, no progress events fire (the download just completes).
async fn download_with_progress(
    url: &str,
    dest: &Path,
    app_handle: &tauri::AppHandle,
) -> Result<(), String> {
    use futures_util::StreamExt;
    use tauri::Emitter;
    use tokio::io::AsyncWriteExt;

    let client = reqwest::Client::builder()
        .user_agent("wupi-updater")
        .build()
        .map_err(|e| format!("build client: {e}"))?;
    let resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| format!("zip GET: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("zip status: {}", resp.status()));
    }
    let total = resp.content_length();
    let mut file = tokio::fs::File::create(dest)
        .await
        .map_err(|e| format!("create dest: {e}"))?;
    let mut stream = resp.bytes_stream();
    let mut written: u64 = 0;
    let mut last_pct: u64 = 0;
    while let Some(chunk) = stream
        .next()
        .await
        .transpose()
        .map_err(|e| format!("zip stream: {e}"))?
    {
        file.write_all(&chunk)
            .await
            .map_err(|e| format!("write chunk: {e}"))?;
        written += chunk.len() as u64;
        if let Some(total) = total {
            let pct = (written * 100) / total.max(1);
            // Throttle: only emit on whole-percent change (2/sec cap at full
            // speed). Keeps the IPC channel quiet.
            if pct > last_pct {
                last_pct = pct;
                let _ = app_handle.emit(
                    "update-progress",
                    serde_json::json!({ "percent": pct, "downloaded": written, "total": total }),
                );
            }
        }
    }
    file.flush().await.map_err(|e| format!("flush: {e}"))?;
    Ok(())
}

/// Extract `zip_path` into `dest`. Uses the `zip` crate (pure Rust, no system
/// deps). Preserves directory structure; skips entries that would escape
/// `dest` (path-traversal defense: reject any entry whose canonicalized path
/// isn't under `dest`).
fn extract_zip(zip_path: &Path, dest: &Path) -> Result<(), String> {
    let file = std::fs::File::open(zip_path).map_err(|e| format!("open zip: {e}"))?;
    let mut archive = zip::ZipArchive::new(file).map_err(|e| format!("open archive: {e}"))?;
    let dest_canon = dest
        .canonicalize()
        .map_err(|e| format!("canonicalize dest: {e}"))?;
    for i in 0..archive.len() {
        let mut entry = archive
            .by_index(i)
            .map_err(|e| format!("read entry {i}: {e}"))?;
        let entry_path = match entry.enclosed_name() {
            Some(p) => p,
            None => continue, // skip unsafe paths (the zip crate's own guard)
        };
        let out = dest.join(&entry_path);
        // Belt-and-suspenders: re-check the joined path is under dest.
        let parent = out.parent().unwrap_or(dest);
        if let Ok(parent_canon) = parent.canonicalize().or_else(|_| std::fs::create_dir_all(parent).and_then(|_| parent.canonicalize())) {
            if !parent_canon.starts_with(&dest_canon) {
                tracing::warn!(?out, "zip entry escapes dest; skipping");
                continue;
            }
        }
        if entry.is_dir() {
            std::fs::create_dir_all(&out).map_err(|e| format!("mkdir {}: {e}", out.display()))?;
        } else {
            std::fs::create_dir_all(parent).map_err(|e| format!("mkdir parent: {e}"))?;
            let mut out_file = std::fs::File::create(&out)
                .map_err(|e| format!("create {}: {e}", out.display()))?;
            std::io::copy(&mut entry, &mut out_file)
                .map_err(|e| format!("write {}: {e}", out.display()))?;
        }
    }
    Ok(())
}

/// Walk `extracted/` and copy files into `exe_dir` with the preserve rule
/// (§8C). Carve-outs:
/// - `data/` is preserved EXCEPT `data/wupi.sim` (engine content; persona
///   updates ship in the zip and overwrite the local copy on update — Chloe's
///   call on the §8C internal contradiction).
/// - `memory/`, `models/`, `apps/` are fully preserved (defensive — the zip
///   shouldn't ship these, but the rule is total).
/// - `wupi.exe` is swapped via the rename-and-relaunch dance.
/// - Everything else is overwritten in place.
fn apply_extracted(extracted: &Path, exe_dir: &Path) -> Result<(), String> {
    let entries = walk_files(extracted)?;
    let exe_name = exe_basename();
    for src in entries {
        // Relative path from the extract root → the install root.
        let rel = match src.strip_prefix(extracted) {
            Ok(r) => r,
            Err(_) => continue,
        };
        // The preserve rule (§8C): the four user-data top-level dirs are
        // preserved. Within data/, wupi.sim is engine content and gets
        // overwritten; everything else in data/ (user.xml, theme.json,
        // api_config.json, docs/) is preserved.
        if is_preserved(rel) {
            tracing::info!(?rel, "preserve: user-data entry skipped");
            continue;
        }
        let dst = exe_dir.join(rel);
        if rel == Path::new(&exe_name) {
            // The running exe: rename + move. Windows permits renaming a
            // running binary but not overwriting it.
            swap_running_exe(&src, &dst)?;
            continue;
        }
        // Plain file overwrite. Create parent dirs (e.g. data/, assets/).
        if let Some(parent) = dst.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("mkdir parent: {e}"))?;
        }
        std::fs::copy(&src, &dst)
            .map_err(|e| format!("copy {} -> {}: {e}", src.display(), dst.display()))?;
        tracing::info!(?rel, "updated");
    }
    Ok(())
}

/// The §8C preserve rule as a predicate. Returns true for paths that must NOT
/// be overwritten by an update (user data). `data/wupi.sim` is the single
/// exception: it's engine content shipped in the zip and replaced on update.
///
/// `rel` is the file's path relative to the extract root (e.g. `data/user.xml`,
/// `memory/memory.sqlite`, `wupi.exe`).
fn is_preserved(rel: &Path) -> bool {
    // data/: preserved EXCEPT data/wupi.sim (engine content).
    if rel.starts_with("data") {
        return rel != Path::new("data/wupi.sim");
    }
    // memory/, models/, apps/: fully preserved.
    rel.starts_with("memory") || rel.starts_with("models") || rel.starts_with("apps")
}

/// The exe basename on this platform (`wupi.exe` on Windows). Stubbed for
/// non-Windows to keep the module portable for tests.
fn exe_basename() -> String {
    if cfg!(windows) {
        "wupi.exe".to_string()
    } else {
        "wupi".to_string()
    }
}

/// Swap the running exe (`dst`, currently in use) with the new one (`src`).
///
/// Windows locks the running exe: it can be renamed but not overwritten while
/// the process is alive. So:
///   1. Rename `dst` (`wupi.exe`) → `wupi.exe.old`. Windows permits this.
///   2. Move `src` (the extracted new exe) → `dst`.
///   3. `wupi.exe.old` is deleted on the NEXT boot by [`cleanup_old_exe`]
///      (the old process still holds its lock until exit; deletion here
///      would fail with "file in use" and serve no purpose).
fn swap_running_exe(src: &Path, dst: &Path) -> Result<(), String> {
    let old = dst.with_extension("exe.old");
    std::fs::rename(dst, &old)
        .map_err(|e| format!("rename {} to {}: {e}", dst.display(), old.display()))?;
    std::fs::rename(src, dst)
        .map_err(|e| format!("move new exe into place: {e}"))?;
    tracing::info!("swapped running exe; .old will be cleaned up on next boot");
    Ok(())
}

/// Recursively collect all files under `root` (depth-first). Used by
/// [`apply_extracted`] to walk the extracted tree.
fn walk_files(root: &Path) -> Result<Vec<PathBuf>, String> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = std::fs::read_dir(&dir).map_err(|e| format!("read_dir: {e}"))?;
        for entry in entries.flatten() {
            let path = entry.path();
            let ft = match entry.file_type() {
                Ok(ft) => ft,
                Err(_) => continue,
            };
            if ft.is_dir() {
                stack.push(path);
            } else if ft.is_file() {
                out.push(path);
            }
        }
    }
    Ok(out)
}
