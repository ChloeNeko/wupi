// Local release publisher for WUPI (PORTABLE distribution).
//
// WHY THIS EXISTS: WUPI's `llama-cpp-2` crate requires the CUDA Toolkit to
// compile (the chat model runs on GPU). GitHub's standard `windows-latest`
// runners DON'T have CUDA installed, so CI can't build WUPI. The release
// flow is therefore: YOU build locally (your PC has CUDA + a warm build
// cache → ~3 min builds), this script packages + publishes the result.
//
// PORTABLE MODEL: WUPI ships as a portable zip — no installer, no uninstaller,
// nothing installs outside the folder the user extracts to. Updates are
// file-level (src-tauri/src/updater.rs): the app downloads a new portable
// zip, extracts it in place, and replaces engine files while preserving
// everything under `data/` (memory, models, sessions, schemas, theme, api
// config, user Operator.xml, user docs). The Tauri updater plugin (installer-
// only) was removed; this script publishes the manifest the custom updater
// polls.
//
// WHAT IT DOES (one command, end-to-end):
//   1. Auto-bumps the version (patch by default; --minor / --major)
//   2. Runs `npx tauri build` (produces target/release/wupi.exe)
//   3. Stages wupi.exe + dist/ + cards/ into a portable-zip layout
//   4. Zips the staged tree → WUPI_<version>_portable.zip
//   5. Creates a GitHub Release + uploads the zip
//   6. Writes latest.json to the gh-pages branch (the manifest the custom
//      updater polls at https://chloeneko.github.io/WUPI/updater/latest.json)
//
// USAGE:
//   npm run release                    # bump patch (0.2.2 → 0.2.3) + release
//   npm run release -- --minor         # bump minor (0.2.2 → 0.3.0)
//   npm run release -- --major         # bump major (0.2.2 → 1.0.0)
//   npm run release -- --no-bump       # release the current version as-is
//   npm run release -- --dry-run       # build + print what would happen, no upload
//
// PRE-REQS (one-time, see docs/UPDATER_SETUP.md):
//   - gh CLI authenticated (`gh auth login`)
//   - gh-pages branch exists (this script creates it on first run if missing)
//   - GitHub Pages enabled on the gh-pages branch (one-time web UI step)
//
// NO SIGNING KEY NEEDED: the portable updater trusts HTTPS + GitHub release
// auth for the beta (no minisig verification). Signature verification can be
// layered on later without changing the publish flow. Compare to the old
// NSIS flow which required keys/wupi.key + keys/wupi.key.pw.

const { readFileSync, existsSync, mkdirSync, rmSync, readdirSync, copyFileSync, cpSync, writeFileSync, statSync } = require('fs');
const { join } = require('path');
const { spawnSync } = require('child_process');

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
    writeFileSync(tauriConfPath, updatedRaw + '\n');
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
//
// Discovery order:
//   1. keys/hf.key                       (PRIMARY — repo-relative, gitignored,
//                                        bare token, same convention as the
//                                        historical signing key)
//   2. process.env.HF_TOKEN              (explicit export in the parent shell)
//   3. ~/.bashrc `export HF_TOKEN=hf_…`  (legacy fallback; not preferred)
//
// HARD FAIL by default if not found: shipping a binary with no token is a
// silent footgun (looks fine, breaks every fresh install). Override with
// --allow-missing-hf-token ONLY for the rare case of re-releasing an existing
// version whose first-run path is already cached for all users.
// ──────────────────────────────────────────────────────────────────────────
const { homedir } = require('os');
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
// Step 3: Run `npx tauri build`. Inherits stdio so build progress streams
// live to the console (CUDA recompile is the long part — needs visible output
// to know it's not hung).
//
// No signing env vars needed anymore: bundle.active=false means Tauri just
// compiles + emits target/release/wupi.exe, no installer, no minisig.
// ──────────────────────────────────────────────────────────────────────────
const childEnv = {
  ...process.env,
  // HF_TOKEN forwarded explicitly (process.env spread above already includes
  // it, but listing it here makes the compile-time dependency visible at
  // the call site — see model_downloader.rs).
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
// Step 4: Stage the portable-zip layout.
//
// The zip mirrors what a tester extracts to their Desktop:
//   WUPI/
//   ├── wupi.exe                      (from target/release/)
//   ├── index.html, script.js, …      (from dist/)
//   ├── cards/
//   │   ├── Wupi.sim                  (shipped engine file)
//   │   ├── Operator.xml              (shipped TEMPLATE — copied to data/ on
//   │   │                               first run; the user's live copy is
//   │   │                               never overwritten by updates)
//   │   ├── wupi_knowledge/           (shipped, may be empty)
//   │   └── game_cards/               (shipped scenario .sim files)
//   └── (no data/ — created on first run, preserved on update)
//
// We deliberately do NOT ship a docs/ dir: docs/ is user-owned (their codex
// library), and the engine-only dev docs (UPDATER_SETUP.md) have no place in
// a user install. The app creates data/docs/ lazily when the user authors
// their first codex entry.
// ──────────────────────────────────────────────────────────────────────────
const builtExe = join(repoRoot, 'src-tauri', 'target', 'release', 'wupi.exe');
if (!existsSync(builtExe)) {
  console.error(`[release] built exe not found at: ${builtExe}`);
  console.error('[release] did the build actually succeed? bundle.active=false in tauri.conf.json');
  console.error('           should produce target/release/wupi.exe directly.');
  process.exit(1);
}
const distDir = join(repoRoot, 'dist');
if (!existsSync(distDir)) {
  console.error(`[release] dist/ not found at: ${distDir}`);
  console.error('[release] the build should have run vite → dist/. Check beforeBuildCommand.');
  process.exit(1);
}
const cardsDir = join(repoRoot, 'cards');
if (!existsSync(join(cardsDir, 'Wupi.sim'))) {
  console.error(`[release] cards/Wupi.sim not found at: ${cardsDir}`);
  console.error('[release] Wupi.sim MUST ship in the portable zip — without it the persona');
  console.error('           loader falls back to a stub and the whole app is wrong.');
  process.exit(1);
}

// Stage under src-tauri/target/ (already gitignored) so the zip never shows
// up in `git status`. A top-level target/ would NOT be ignored by the current
// .gitignore (which only lists src-tauri/target/), so we re-use the cargo
// target dir for symmetry + zero new gitignore surface.
const stageRoot = join(repoRoot, 'src-tauri', 'target', 'release-portable');
const stageWupiDir = join(stageRoot, 'WUPI');
if (!rmSyncRetry(stageRoot)) {
  console.error(`[release] could not clear ${stageRoot} after retries (OS lock).`);
  console.error('[release] Aborting rather than staging into a half-cleaned dir.');
  console.error('[release] Reboot clears the OS handle, then re-run with --no-bump.');
  process.exit(1);
}
mkdirSync(stageWupiDir, { recursive: true });

// wupi.exe at the zip root.
copyFileSync(builtExe, join(stageWupiDir, 'wupi.exe'));
// dist/ contents (index.html, script.js, styles.css, paw.png, assets/) at the
// zip root — flat, NOT under a dist/ subdir. This matches the resolve_*_path
// walkers which expect assets next to wupi.exe.
for (const f of readdirSync(distDir)) {
  const src = join(distDir, f);
  const dst = join(stageWupiDir, f);
  const stat = statSync(src);
  if (stat.isDirectory()) {
    cpSync(src, dst, { recursive: true });
  } else {
    copyFileSync(src, dst);
  }
}
// cards/ as a subdir (preserves the cards/game_cards/ and cards/wupi_knowledge/
// structure the resolvers walk).
cpSync(cardsDir, join(stageWupiDir, 'cards'), { recursive: true });

console.log(`[release] staged portable layout at ${stageWupiDir}`);

// ──────────────────────────────────────────────────────────────────────────
// Step 4.5: Zip the staged tree using adm-zip (pure-JS, devDependency).
//
// Previously used PowerShell's Compress-Archive, but `Microsoft.PowerShell.
// Archive` fails to load on some Windows setups (ExecutionPolicy, module-
// load races, PSModulePath issues) — empirically flaky. adm-zip is pure JS,
// ~50KB, no native build, and runs in the same Node process as this script
// (zero cross-process surface). The Rust `zip` crate already handles the
// extraction side in the updater; this is the symmetric publish side.
//
// Named WUPI_<version>_portable.zip so it's unambiguous in the Release list.
// ──────────────────────────────────────────────────────────────────────────
const AdmZip = require('adm-zip');
const zipName = `WUPI_${newVersion}_portable.zip`;
const zipPath = join(stageRoot, zipName);
if (!dryRun) {
  console.log(`[release] zipping → ${zipName}…`);
  try {
    const zip = new AdmZip();
    // addLocalFolder takes the contents of the staged WUPI/ dir and places
    // them at the zip root (so the extract lands as a flat WUPI/ folder,
    // not WUPI/WUPI/). Empty dirs are included for structure (wupi_knowledge/,
    // game_cards/ if empty).
    zip.addLocalFolder(stageWupiDir);
    zip.writeZip(zipPath);
  } catch (e) {
    console.error(`[release] adm-zip failed: ${e.message}`);
    process.exit(1);
  }
  if (!existsSync(zipPath)) {
    console.error(`[release] zip not created at ${zipPath} despite no error — unknown failure.`);
    process.exit(1);
  }
  const zipSize = statSync(zipPath).size;
  console.log(`[release] zip ready: ${zipName} (${(zipSize / 1024 / 1024).toFixed(1)} MB)`);
}

if (dryRun) {
  console.log('\n[release] === DRY RUN SUMMARY ===');
  console.log(`  version: ${newVersion}`);
  console.log(`  tag:     ${tag}`);
  console.log(`  staged:  ${stageWupiDir}`);
  console.log(`  zip:     ${zipName}`);
  console.log(`  would:   gh release create ${tag} ${zipName} + push latest.json to gh-pages`);
  process.exit(0);
}

// ──────────────────────────────────────────────────────────────────────────
// Step 5: Create the GitHub Release + upload the portable zip.
//
// GitHub's upload endpoint (uploads.github.com) intermittently returns
// 502/503 under load — empirically one of the flakier parts of their API.
// We retry up to 3 times with a 15s backoff. If a partial-failure left a
// draft release behind, we delete it before retrying (otherwise the second
// attempt fails on "tag already exists").
// ──────────────────────────────────────────────────────────────────────────
const repo = 'ChloeNeko/WUPI';
const assetUrl = `https://github.com/${repo}/releases/download/${tag}/${zipName}`;
const notes = `WUPI ${tag} (portable)\n\nExtract anywhere and run wupi.exe. Everything stays in the folder.`;

console.log(`[release] creating GitHub Release ${tag}…`);
// No shell:true here — on Git Bash for Windows, shell:true glob-expands the
// tag (e.g. v0.1.0) and the file paths (forward slashes) and aborts with
// "no matches found". gh accepts absolute paths directly; we pass them as
// explicit argv to bypass shell interpretation entirely.

// If the release for this tag already exists (common with --no-bump
// re-releases, or a retry of a partially-successful previous run), delete
// it UP FRONT so the first gh-release-create attempt succeeds. Without
// this, attempt 1 always fails on "tag already exists" and we waste a
// 15s backoff cycle before the retry loop cleans it up.
const existingRelease = spawnSync('gh',
  ['release', 'view', tag, '--repo', repo, '--json', 'tagName'],
  { cwd: repoRoot, stdio: 'pipe', encoding: 'utf8' });
if (existingRelease.status === 0) {
  console.log(`[release] existing ${tag} release found; replacing it…`);
  spawnSync('gh', ['release', 'delete', tag, '--repo', repo, '--yes'],
            { stdio: 'ignore', cwd: repoRoot });
  spawnSync('git', ['push', 'origin', `:refs/tags/${tag}`],
            { stdio: 'ignore', cwd: repoRoot });
  spawnSync('git', ['tag', '-d', tag], { stdio: 'ignore', cwd: repoRoot });
}

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
    zipPath,
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
// The custom updater (src-tauri/src/updater.rs) polls:
//   https://chloeneko.github.io/WUPI/updater/latest.json
//
// Manifest shape: same `version`/`notes`/`pub_date`/`platforms` fields as
// the old Tauri manifest (the Rust Manifest struct ignores signature). The
// `signature` field is omitted — the portable updater doesn't verify
// minisig for the beta (HTTPS + GitHub release auth is the trust boundary).
// ──────────────────────────────────────────────────────────────────────────
console.log('[release] publishing latest.json to gh-pages via GitHub API…');
const pubDate = new Date().toISOString();
const manifest = {
  version: newVersion,
  notes,
  pub_date: pubDate,
  platforms: {
    'windows-x86_64': {
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
// Step 7: working tree is untouched (we never checked out gh-pages). The
// staged portable zip under src-tauri/target/release-portable/ stays on disk
// (handy for local testing of the updater; gitignored under src-tauri/target/).
// ──────────────────────────────────────────────────────────────────────────

console.log('\n[release] ========================================');
console.log(`[release]  RELEASE ${tag} PUBLISHED (portable)`);
console.log('[release] ========================================');
console.log(`[release]  Release:   https://github.com/${repo}/releases/tag/${tag}`);
console.log(`[release]  Manifest:  https://chloeneko.github.io/WUPI/updater/latest.json`);
console.log(`[release]  Asset URL: ${assetUrl}`);
console.log('[release]  Next tester launch → updater_check fires (DARK, devtools-only).');
console.log('[release] ========================================');
