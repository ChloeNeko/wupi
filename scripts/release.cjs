// Local release publisher for WUPI.
//
// WHY THIS EXISTS: WUPI's `llama-cpp-2` crate requires the CUDA Toolkit to
// compile (the chat model runs on GPU). GitHub's standard `windows-latest`
// runners DON'T have CUDA installed, so CI can't build WUPI. The release
// flow is therefore: YOU build locally (your PC has CUDA + a warm build
// cache → ~3 min builds), this script packages + publishes the result.
//
// WHAT IT DOES (one command, end-to-end):
//   1. Reads the Tauri signing key + password (same logic as build-signed.cjs)
//   2. Auto-bumps the version (patch by default; --minor / --major)
//   3. Runs `npx tauri build` with signing env vars
//   4. Creates a GitHub Release + uploads the signed artifacts
//   5. Writes latest.json to the gh-pages branch (the manifest the Tauri
//      updater polls at https://chloeneko.github.io/WUPI/updater/latest.json)
//
// USAGE:
//   npm run release                    # bump patch (0.1.0 → 0.1.1) + release
//   npm run release -- --minor         # bump minor (0.1.0 → 0.2.0)
//   npm run release -- --major         # bump major (0.1.0 → 1.0.0)
//   npm run release -- --no-bump       # release the current version as-is
//   npm run release -- --dry-run       # build + print what would happen, no upload
//
// PRE-REQS (one-time, see docs/UPDATER_SETUP.md):
//   - ~/.tauri/wupi.key generated (`npx @tauri-apps/cli signer generate`)
//   - gh CLI authenticated (`gh auth login`)
//   - gh-pages branch exists (this script creates it on first run if missing)
//   - GitHub Pages enabled on the gh-pages branch (one-time web UI step)
//
// SECURITY: same model as build-signed.cjs — the private key + password are
// read into memory and passed via child-process env only. NEVER logged,
// NEVER written to disk, NEVER set in the parent shell.

const { readFileSync, existsSync, mkdirSync, rmSync, readdirSync, copyFileSync } = require('fs');
const { join } = require('path');
const { homedir } = require('os');
const { spawn, spawnSync } = require('child_process');

// Windows-tolerant recursive delete. Node's `fs.rmSync` returns EPERM when
// Defender (MsMpEng.exe) or Search Indexer (SearchIndexer.exe) briefly holds
// a handle on a freshly-written .exe or its containing dir — even an EMPTY
// dir can be "busy" for several seconds after the build. `force:true` only
// swallows ENOENT, not EPERM. Retry with exponential backoff so a transient
// OS lock doesn't crash the release AFTER the build already succeeded.
//
// Returns true on success, false if it gave up (caller decides whether to
// fall back to a fresh versioned stage dir).
const rmSyncRetry = (target, { retries = 6, baseDelayMs = 500 } = {}) => {
  for (let i = 0; i < retries; i++) {
    try {
      rmSync(target, { recursive: true, force: true });
      return true;
    } catch (e) {
      if (e.code === 'ENOENT') return true;  // already gone — fine
      // EPERM/EBUSY: wait and retry. Anything else (e.g. ENOTDIR): throw.
      if (e.code !== 'EPERM' && e.code !== 'EBUSY') throw e;
      const delay = baseDelayMs * Math.pow(2, i);
      console.warn(`[release] rmSync ${e.code} on ${target}; retry ${i + 1}/${retries} in ${delay}ms…`);
      spawnSync('sleep', [String(Math.ceil(delay / 1000))]);
    }
  }
  return false;
};

// ── Parse args ──
const argv = process.argv.slice(2);
let bumpKind = 'patch';   // 'patch' | 'minor' | 'major' | null
let dryRun = false;
for (const a of argv) {
  if (a === '--minor') bumpKind = 'minor';
  else if (a === '--major') bumpKind = 'major';
  else if (a === '--no-bump') bumpKind = null;
  else if (a === '--dry-run') dryRun = true;
}

// ──────────────────────────────────────────────────────────────────────────
// Step 0: Resolve the signing key + password (mirrors build-signed.cjs).
//
// Key location: repo-relative `keys/` (gitignored — see .gitignore:2). The
// historical default of `~/.tauri/wupi.key` is kept as a fallback so we
// don't break anyone who set things up the old way, but `keys/` wins.
// Override with WUPI_SIGNING_KEY_PATH if you want to point elsewhere.
// ──────────────────────────────────────────────────────────────────────────
// __dirname is available immediately in CJS; `repoRoot` (defined in Step 1
// below) isn't yet. Use __dirname to compute the key path here.
const repoRootKeyPath = join(__dirname, '..', 'keys', 'wupi.key');
const homeKeyPath = join(homedir(), '.tauri', 'wupi.key');
const keyPath = process.env.WUPI_SIGNING_KEY_PATH
  || (existsSync(repoRootKeyPath) ? repoRootKeyPath : homeKeyPath);
if (!existsSync(keyPath)) {
  console.error(`[release] private key not found at: ${keyPath}`);
  console.error('[release] generate one with:');
  console.error('  npx @tauri-apps/cli signer generate -w keys/wupi.key');
  console.error('  (or set WUPI_SIGNING_KEY_PATH to point elsewhere)');
  process.exit(1);
}
let privateKey;
try {
  privateKey = readFileSync(keyPath, 'utf8').trim();
} catch (e) {
  console.error(`[release] failed to read private key: ${e.message}`);
  process.exit(1);
}

// Password resolution (priority: TAURI_SIGNING_PRIVATE_KEY_PASSWORD env >
// keys/wupi.key.pw > empty). Looks in the repo-relative `keys/` dir first
// (same convention as the private key above), falls back to ~/.tauri/.
//
// IMPORTANT: Tauri's signer reads `TAURI_SIGNING_PRIVATE_KEY_PASSWORD`
// (NOT `TAURI_KEY_PASSWORD` — that's an older name the signer no longer
// recognizes; setting it silently falls through to the interactive rpassword
// prompt and hangs the build). Verified against `tauri signer sign --help`.
const repoRootPwPath = join(__dirname, '..', 'keys', 'wupi.key.pw');
const homePwPath = join(homedir(), '.tauri', 'wupi.key.pw');
const pwFilePath = existsSync(repoRootPwPath) ? repoRootPwPath : homePwPath;
let password = '';
if (process.env.TAURI_SIGNING_PRIVATE_KEY_PASSWORD) {
  password = process.env.TAURI_SIGNING_PRIVATE_KEY_PASSWORD;
} else if (existsSync(pwFilePath)) {
  try {
    password = readFileSync(pwFilePath, 'utf8').replace(/\r?\n$/, '');
    console.log(`[release] password loaded from ${pwFilePath}`);
  } catch (e) {
    console.error(`[release] failed to read password file: ${e.message}`);
    process.exit(1);
  }
} else {
  console.warn('[release] no password source (no env var, no ~/.tauri/wupi.key.pw).');
  console.warn('                 If the key was generated WITH a password, this will hang.');
}

// ──────────────────────────────────────────────────────────────────────────
// Step 1: Read + bump the version.
// ──────────────────────────────────────────────────────────────────────────
const repoRoot = join(__dirname, '..');
const tauriConfPath = join(repoRoot, 'src-tauri', 'tauri.conf.json');
const tauriConfRaw = readFileSync(tauriConfPath, 'utf8');
const tauriConf = JSON.parse(tauriConfRaw);
const currentVersion = tauriConf.version;

let newVersion = currentVersion;
if (bumpKind) {
  const [major, minor, patch] = currentVersion.split('.').map(Number);
  if (bumpKind === 'patch') newVersion = `${major}.${minor}.${patch + 1}`;
  else if (bumpKind === 'minor') newVersion = `${major}.${minor + 1}.0`;
  else if (bumpKind === 'major') newVersion = `${major + 1}.0.0`;

  // Write the bumped version back. Preserve 2-space indentation (matches the
  // existing file's formatting). Don't touch anything else in the file.
  tauriConf.version = newVersion;
  const updatedRaw = JSON.stringify(tauriConf, null, 2);
  if (!dryRun) {
    require('fs').writeFileSync(tauriConfPath, updatedRaw + '\n');
    console.log(`[release] version bumped: ${currentVersion} → ${newVersion}`);
  } else {
    console.log(`[release] (dry-run) would bump version: ${currentVersion} → ${newVersion}`);
  }
} else {
  console.log(`[release] --no-bump: re-releasing version ${currentVersion}`);
}
const tag = `v${newVersion}`;

// ──────────────────────────────────────────────────────────────────────────
// Step 2: Commit the version bump (if any) so the built binary's reported
// version matches the git tag we publish. Skip in dry-run.
// ──────────────────────────────────────────────────────────────────────────
if (bumpKind && !dryRun) {
  console.log(`[release] committing version bump for ${tag}…`);
  const commit = spawnSync('git', ['add', tauriConfPath], { stdio: 'inherit' });
  if (commit.status !== 0) { console.error('[release] git add failed'); process.exit(1); }
  const cm = spawnSync('git', ['commit', '-m', `release: ${tag}`], { stdio: 'inherit' });
  if (cm.status !== 0) { console.error('[release] git commit failed'); process.exit(1); }
  // Push the bump commit IMMEDIATELY so the tag `gh release create` later
  // attaches actually exists on the remote. Without this, if `--target
  // ui-shell` resolves to the unpushed local HEAD, GitHub stores a dangling
  // tag pointing at a commit nobody can fetch. (Bite we hit on v0.2.1/v0.2.2.)
  console.log(`[release] pushing ui-shell to origin…`);
  const push = spawnSync('git', ['push', 'origin', 'ui-shell'], { stdio: 'inherit', cwd: repoRoot });
  if (push.status !== 0) {
    console.error('[release] git push failed. The version-bump commit is local-only; aborting');
    console.error('           before tagging a remote-pointing release. Push manually and re-run');
    console.error('           with `--no-bump`.');
    process.exit(1);
  }
}

// ──────────────────────────────────────────────────────────────────────────
// Step 2.5: HF_TOKEN gate. The compiled binary bakes HF_TOKEN in at COMPILE
// TIME via `option_env!("HF_TOKEN")` in src-tauri/src/model_downloader.rs —
// first-run GGUF download uses it as a Bearer against the PRIVATE
// ChloeNeko/WUPI HF repo. If unset here, the constant compiles to "" and
// every fresh install 403s on the download overlay. Warn loudly; don't
// refuse (you may legitimately be re-releasing an existing version whose
// first-run path is already cached for all users).
// ──────────────────────────────────────────────────────────────────────────
// HF_TOKEN discovery. The compiled binary bakes HF_TOKEN in via
// `option_env!("HF_TOKEN")` in model_downloader.rs at COMPILE TIME — first-run
// GGUF download uses it as a Bearer against the PRIVATE ChloeNeko/WUPI HF
// repo. If it's "" at compile time, every fresh install 403s on the download
// overlay.
//
// Discovery order (process env is unreliable on Windows: Git Bash background
// tasks / CI runners don't always source ~/.bashrc before spawning node, so
// process.env.HF_TOKEN can be empty even when the user "set it"):
//   1. keys/hf.key                       (PRIMARY — repo-relative, gitignored,
//                                        bare token, same convention as
//                                        keys/wupi.key)
//   2. process.env.HF_TOKEN              (explicit export in the parent shell)
//   3. ~/.bashrc `export HF_TOKEN=hf_…`  (legacy fallback; not preferred)
//
// Once found, we EXPORT it back to process.env so the childEnv spread below
// picks it up — `npx tauri build` and the cargo subprocesses will see it.
//
// HARD FAIL by default if not found: shipping a binary with no token is a
// silent footgun (looks fine, breaks every fresh install). Override with
// --allow-missing-hf-token ONLY for the rare case of re-releasing an existing
// version whose first-run path is already cached for all users.
const findHfToken = () => {
  // 1. keys/hf.key (preferred — keeps all release secrets in one gitignored dir)
  const keyFilePath = join(__dirname, '..', 'keys', 'hf.key');
  if (existsSync(keyFilePath)) {
    const raw = readFileSync(keyFilePath, 'utf8').trim();
    // Accept either a bare token (preferred) or `export HF_TOKEN=hf_…` (legacy)
    const m = raw.match(/(hf_[A-Za-z0-9]+)/);
    if (m) return m[1];
  }
  // 2. process.env.HF_TOKEN (explicit shell export)
  if (process.env.HF_TOKEN) return process.env.HF_TOKEN;
  // 3. ~/.bashrc fallback (legacy — we prefer keys/hf.key now)
  const bashrcPath = join(homedir(), '.bashrc');
  if (existsSync(bashrcPath)) {
    const bashrc = readFileSync(bashrcPath, 'utf8');
    const m = bashrc.match(/^\s*export\s+HF_TOKEN\s*=\s*(hf_[A-Za-z0-9]+)/m);
    if (m) return m[1];
  }
  return null;
};
const hfToken = findHfToken();
const allowMissing = argv.includes('--allow-missing-hf-token');
if (hfToken) {
  process.env.HF_TOKEN = hfToken;  // re-export so childEnv spread sees it
  console.log(`[release] HF_TOKEN resolved (len=${hfToken.length}, prefix=${hfToken.slice(0, 7)}…).`);
} else if (allowMissing) {
  console.warn('[release] !! HF_TOKEN not found and --allow-missing-hf-token passed.');
  console.warn('              The compiled binary will have HF_TOKEN="" — fresh installs will');
  console.warn('              403 on the first-run GGUF download.');
} else {
  console.error('[release] !! HF_TOKEN not found in keys/hf.key, env, or ~/.bashrc.');
  console.error('              The compiled binary would have HF_TOKEN="" → every fresh install');
  console.error('              403s on first-run GGUF download. REFUSING to ship a broken build.');
  console.error('');
  console.error('              Fix: put a fine-grained read-only HF token in keys/hf.key:');
  console.error('                echo hf_<token> > keys/hf.key   (gitignored, bare token only)');
  console.error('              Then re-run. To override (NOT recommended):');
  console.error('                npm run release -- --allow-missing-hf-token');
  process.exit(1);
}

// ──────────────────────────────────────────────────────────────────────────
// Step 3: Run `npx tauri build` with the signing env. Inherits stdio so
// build progress streams live to the console (CUDA recompile is the long
// part — needs visible output to know it's not hung).
// ──────────────────────────────────────────────────────────────────────────
const childEnv = {
  ...process.env,
  TAURI_SIGNING_PRIVATE_KEY: privateKey,
  // Correct env var name (see password-resolution note above). Setting both
  // names so the legacy build-signed.cjs script and any external tools that
  // still expect TAURI_KEY_PASSWORD keep working.
  TAURI_SIGNING_PRIVATE_KEY_PASSWORD: password,
  TAURI_KEY_PASSWORD: password,
  // HF_TOKEN forwarded explicitly (process.env spread above already includes
  // it, but listing it here makes the compile-time dependency visible at
  // the call site — see model_downloader.rs:88).
  HF_TOKEN: process.env.HF_TOKEN || '',
  CMAKE_BUILD_PARALLEL_LEVEL: process.env.CMAKE_BUILD_PARALLEL_LEVEL || '8',
};

console.log(`[release] running: npx tauri build`);
if (dryRun) {
  console.log('[release] (dry-run) skipping actual build');
} else {
  const buildResult = spawnSync('npx', ['tauri', 'build'], {
    env: childEnv,
    stdio: 'inherit',
    shell: true,  // npx is a .cmd on Windows; shell:true to invoke
  });
  if (buildResult.status !== 0) {
    console.error(`[release] tauri build failed (exit ${buildResult.status}).`);
    process.exit(buildResult.status ?? 1);
  }
  console.log('[release] build complete.');
}

// ──────────────────────────────────────────────────────────────────────────
// Step 4: Collect the artifacts (signed installer + signature + zips).
//
// Staging dir is `target/release/` — one folder, overwritten each release.
// We don't keep historical stage dirs lying around; the GitHub Release is
// the persistent artifact store. The retry-on-rmSync handles Defender /
// Search Indexer briefly locking freshly-written .exe files.
// ──────────────────────────────────────────────────────────────────────────
const nsisDir = join(repoRoot, 'src-tauri', 'target', 'release', 'bundle', 'nsis');
const msiDir = join(repoRoot, 'src-tauri', 'target', 'release', 'bundle', 'msi');
const stageDir = join(repoRoot, 'target', 'release');
if (!rmSyncRetry(stageDir)) {
  console.error(`[release] could not clear ${stageDir} after retries (OS lock).`);
  console.error('[release] Aborting rather than staging into a half-cleaned dir.');
  console.error('[release] Reboot clears the OS handle, then re-run with --no-bump.');
  process.exit(1);
}
mkdirSync(stageDir, { recursive: true });

const copyIf = (srcFile) => {
  if (existsSync(srcFile)) {
    copyFileSync(srcFile, join(stageDir, require('path').basename(srcFile)));
    return true;
  }
  return false;
};

// Renames a Tauri-bundled filename to a cleaner public-facing name.
// Tauri's defaults bake in `_x64` (the only arch WUPI ships — Windows 64-bit
// only, there will never be an x86 build) and `_en-US` (the only locale —
// WUPI is English-only for the beta). Both add visual noise without info.
//   WUPI_0.1.0_x64-setup.exe           → WUPI_0.1.0-setup.exe
//   WUPI_0.1.0_x64_en-US.msi           → WUPI_0.1.0.msi
const cleanName = (f) => f.replace(/_x64/, '').replace(/_en-US/, '');

// Only stage files for the CURRENT version. Tauri doesn't clean the nsis/msi
// bundle dirs between builds, so older version builds (e.g. WUPI_0.1.0-*
// lingering when releasing 0.1.1) would otherwise get staged too. Filtering
// by version prefix guarantees a clean stage dir with only this release's
// artifacts.
const versionPrefix = `WUPI_${newVersion}_`;
const isCurrentVersion = (f) => f.startsWith(versionPrefix);

const copyClean = (srcFile) => {
  if (!existsSync(srcFile)) return null;
  const original = require('path').basename(srcFile);
  if (!isCurrentVersion(original)) return null;
  const cleaned = cleanName(original);
  copyFileSync(srcFile, join(stageDir, cleaned));
  return cleaned;
};

// Primary: NSIS installer + its .sig (Tauri signs the installer directly;
// the .nsis.zip wrapper form is no longer produced by default in Tauri 2).
// Secondary: MSI for users who prefer it (not used by the updater).
let primaryName = null;       // CLEANED name of the file the updater downloads
let sigContent = null;        // contents of the matching .sig file
if (existsSync(nsisDir)) {
  for (const f of readdirSync(nsisDir)) {
    if (!isCurrentVersion(f)) continue;
    const cleaned = copyClean(join(nsisDir, f));
    if (!cleaned) continue;
    // Pick the NSIS setup exe as the updater payload (by CLEANED name).
    if (!primaryName && cleaned.endsWith('-setup.exe')) primaryName = cleaned;
    // Signature content goes into the manifest's `signature` field. The
    // minisig bytes are what Tauri's updater verifies; the original filename
    // baked into the minisig comment doesn't matter for verification.
    if (f.endsWith('-setup.exe.sig')) {
      sigContent = readFileSync(join(nsisDir, f), 'utf8').trim();
    }
  }
}
if (existsSync(msiDir)) {
  for (const f of readdirSync(msiDir)) {
    if (!isCurrentVersion(f)) continue;
    copyClean(join(msiDir, f));
  }
}

if (!primaryName || !sigContent) {
  console.error('[release] could not find NSIS setup exe or its .sig in build output.');
  console.error('[release] was the build actually signed? Check ~/.tauri/wupi.key exists,');
  console.error('           ~/.tauri/wupi.key.pw has the password, and');
  console.error('           createUpdaterArtifacts is true in tauri.conf.json.');
  process.exit(1);
}
console.log(`[release] staged ${readdirSync(stageDir).length} files (primary: ${primaryName})`);

if (dryRun) {
  console.log('\n[release] === DRY RUN SUMMARY ===');
  console.log(`  version: ${newVersion}`);
  console.log(`  tag:     ${tag}`);
  console.log(`  staged:  ${readdirSync(stageDir).join(', ')}`);
  console.log(`  would:   gh release create ${tag} <files> + push latest.json to gh-pages`);
  process.exit(0);
}

// ──────────────────────────────────────────────────────────────────────────
// Step 5: Create the GitHub Release + upload artifacts.
//
// GitHub's upload endpoint (uploads.github.com) intermittently returns
// 502/503 under load — empirically one of the flakier parts of their API.
// We retry up to 3 times with a 15s backoff. If a partial-failure left a
// draft release behind, we delete it before retrying (otherwise the second
// attempt fails on "tag already exists").
// ──────────────────────────────────────────────────────────────────────────
const repo = 'ChloeNeko/WUPI';
const assetUrl = `https://github.com/${repo}/releases/download/${tag}/${primaryName}`;
const notes = `WUPI ${tag}\n\nAuto-built + signed release from local machine.`;

console.log(`[release] creating GitHub Release ${tag}…`);
// No shell:true here — on Git Bash for Windows, shell:true glob-expands the
// tag (e.g. v0.1.0) and the file paths (forward slashes) and aborts with
// "no matches found". gh accepts absolute paths directly; we pass them as
// explicit argv to bypass shell interpretation entirely.
const stagedFiles = readdirSync(stageDir).map(f => join(stageDir, f));

let ghRelease = null;
let releaseOk = false;
for (let attempt = 1; attempt <= 3; attempt++) {
  if (attempt > 1) {
    // Clean up any partial-release leftover from the previous attempt so
    // `gh release create` doesn't fail on "tag already exists."
    console.log(`[release] retry ${attempt}/3: cleaning up partial release…`);
    spawnSync('gh', ['release', 'delete', tag, '--repo', repo, '--yes'],
              { stdio: 'ignore', cwd: repoRoot });
    spawnSync('git', ['push', 'origin', `:refs/tags/${tag}`],
              { stdio: 'ignore', cwd: repoRoot });
    spawnSync('git', ['tag', '-d', tag], { stdio: 'ignore', cwd: repoRoot });
    console.log('[release] waiting 15s before retry (GitHub upload 50X backoff)…');
    spawnSync('sleep', ['15']);
  }
  ghRelease = spawnSync('gh', [
    'release', 'create', tag,
    '--repo', repo,
    '--title', `WUPI ${tag}`,
    '--notes', notes,
    '--target', 'ui-shell',
    ...stagedFiles,
  ], { stdio: 'inherit', cwd: repoRoot });
  if (ghRelease.status === 0) { releaseOk = true; break; }
  console.warn(`[release] attempt ${attempt} failed (exit ${ghRelease.status}).`);
}

if (!releaseOk) {
  console.error(`[release] gh release create failed after 3 attempts.`);
  console.error('[release] Last resort — delete + retry manually:');
  console.error(`           gh release delete ${tag} --repo ${repo} --yes`);
  console.error(`           git push origin :refs/tags/${tag}`);
  console.error(`           npm run release -- --no-bump`);
  process.exit(ghRelease.status ?? 1);
}
console.log(`[release] GitHub Release ${tag} published.`);

// ──────────────────────────────────────────────────────────────────────────
// Step 6: Write latest.json to the gh-pages branch via the GitHub API.
//
// Why API (not `git checkout gh-pages`): the working tree has untracked
// build outputs (target/, dist/) and source files that conflict with a
// branch switch — `git checkout gh-pages` fails with "Please commit your
// changes or stash them" and the manifest then accidentally lands on
// ui-shell. Using the GitHub Contents API writes the file atomically with
// no working-tree interaction at all. gh CLI handles the auth + base64.
//
// Tauri's updater polls: https://chloeneko.github.io/WUPI/updater/latest.json
// ──────────────────────────────────────────────────────────────────────────
console.log('[release] publishing latest.json to gh-pages via GitHub API…');
const pubDate = new Date().toISOString();
const manifest = {
  version: newVersion,
  notes,
  pub_date: pubDate,
  platforms: {
    'windows-x86_64': {
      signature: sigContent,
      url: assetUrl,
    },
  },
};
const manifestJson = JSON.stringify(manifest, null, 2);

// The GitHub Contents API requires `content` as base64. We pass the base64
// string DIRECTLY as a gh api arg value (NOT via `-f content=@<file>` — the
// `@path` expansion is unreliable on Git Bash for Windows, where it sent the
// path string itself instead of the file contents → "content is not valid
// Base64" 422. Inline string args bypass path resolution entirely).
const manifestB64 = Buffer.from(manifestJson, 'utf8').toString('base64');

// Look up the existing file's SHA (so we can update rather than create).
// 404 = file doesn't exist yet (first release); any other error = real problem.
const existingSha = spawnSync('gh', [
  'api', `repos/${repo}/contents/updater/latest.json?ref=gh-pages`,
  '--jq', '.sha',
], { cwd: repoRoot, stdio: 'pipe', encoding: 'utf8' });
let shaArg = [];
if (existingSha.status === 0 && existingSha.stdout.trim()) {
  shaArg = ['-F', `sha=${existingSha.stdout.trim()}`];
}

// PUT the new content via the Contents API. The `content` field is the
// base64 string, passed as a direct arg value (no @file indirection).
// -f = raw string (no type conversion — keeps the base64 verbatim).
const putManifest = spawnSync('gh', [
  'api', `-XPUT`, `repos/${repo}/contents/updater/latest.json`,
  '-F', `message=release manifest: ${tag} [skip ci]`,
  '-F', `branch=gh-pages`,
  '-f', `content=${manifestB64}`,  // direct value: no @path resolution
  ...shaArg,
], { cwd: repoRoot, stdio: 'inherit' });

if (putManifest.status !== 0) {
  console.error(`[release] gh api PUT to gh-pages failed (exit ${putManifest.status}).`);
  console.error('[release] The manifest may need a manual push. The release itself');
  console.error('           is already published; only the updater manifest is at risk.');
  // Don't exit fatally — the Release is up; testers can still manually download.
  // The next successful release will fix the manifest.
}

// ──────────────────────────────────────────────────────────────────────────
// Step 7: working tree is untouched (we never checked out gh-pages). Nothing
// to restore. Done.
// ──────────────────────────────────────────────────────────────────────────

console.log('\n[release] ========================================');
console.log(`[release]  RELEASE ${tag} PUBLISHED`);
console.log('[release] ========================================');
console.log(`[release]  Release:   https://github.com/${repo}/releases/tag/${tag}`);
console.log(`[release]  Manifest:  https://chloeneko.github.io/WUPI/updater/latest.json`);
console.log(`[release]  Asset URL: ${assetUrl}`);
console.log('[release]  Next tester launch → auto-update fires.');
console.log('[release] ========================================');
