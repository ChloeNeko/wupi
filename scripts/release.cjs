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
// ──────────────────────────────────────────────────────────────────────────
const keyPath = join(homedir(), '.tauri', 'wupi.key');
if (!existsSync(keyPath)) {
  console.error(`[release] private key not found at: ${keyPath}`);
  console.error('[release] generate one with:');
  console.error('  npx @tauri-apps/cli signer generate -w ~/.tauri/wupi.key');
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
// ~/.tauri/wupi.key.pw > empty).
//
// IMPORTANT: Tauri's signer reads `TAURI_SIGNING_PRIVATE_KEY_PASSWORD`
// (NOT `TAURI_KEY_PASSWORD` — that's an older name the signer no longer
// recognizes; setting it silently falls through to the interactive rpassword
// prompt and hangs the build). Verified against `tauri signer sign --help`.
const pwFilePath = join(homedir(), '.tauri', 'wupi.key.pw');
let password = '';
if (process.env.TAURI_SIGNING_PRIVATE_KEY_PASSWORD) {
  password = process.env.TAURI_SIGNING_PRIVATE_KEY_PASSWORD;
} else if (existsSync(pwFilePath)) {
  try {
    password = readFileSync(pwFilePath, 'utf8').replace(/\r?\n$/, '');
    console.log('[release] password loaded from ~/.tauri/wupi.key.pw');
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
// ──────────────────────────────────────────────────────────────────────────
const nsisDir = join(repoRoot, 'src-tauri', 'target', 'release', 'bundle', 'nsis');
const msiDir = join(repoRoot, 'src-tauri', 'target', 'release', 'bundle', 'msi');
const stageDir = join(repoRoot, 'target', 'release-stage');
rmSync(stageDir, { recursive: true, force: true });
mkdirSync(stageDir, { recursive: true });

const copyIf = (srcFile) => {
  if (existsSync(srcFile)) {
    copyFileSync(srcFile, join(stageDir, require('path').basename(srcFile)));
    return true;
  }
  return false;
};

// Primary: NSIS installer + its .sig (Tauri signs the installer directly;
// the .nsis.zip wrapper form is no longer produced by default in Tauri 2).
// Secondary: MSI for users who prefer it (not used by the updater).
let primaryName = null;       // the file the updater downloads + installs
let sigContent = null;        // contents of the matching .sig file
if (existsSync(nsisDir)) {
  for (const f of readdirSync(nsisDir)) {
    const copied = copyIf(join(nsisDir, f));
    if (!copied) continue;
    // Pick the NSIS setup exe as the updater payload.
    if (!primaryName && f.endsWith('-setup.exe')) primaryName = f;
    // Read the signature content from the .sig sitting next to the setup exe.
    if (f === `${primaryName}.sig`) {
      sigContent = readFileSync(join(nsisDir, f), 'utf8').trim();
    }
  }
  // The loop above captures primaryName before seeing its .sig in some
  // readdir orderings; re-read the .sig now that we know the name.
  if (primaryName && !sigContent) {
    const sigPath = join(nsisDir, `${primaryName}.sig`);
    if (existsSync(sigPath)) {
      sigContent = readFileSync(sigPath, 'utf8').trim();
    }
  }
}
if (existsSync(msiDir)) {
  for (const f of readdirSync(msiDir)) copyIf(join(msiDir, f));
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
// Uses `gh release create` (gh CLI must be authenticated). If the release
// already exists (re-releasing same version), --target gh-pages-draft flag
// is unnecessary; gh handles overwrite via `gh release upload --clobber`.
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
const ghRelease = spawnSync('gh', [
  'release', 'create', tag,
  '--repo', repo,
  '--title', `WUPI ${tag}`,
  '--notes', notes,
  '--target', 'ui-shell',
  ...stagedFiles,
], { stdio: 'inherit', cwd: repoRoot });

if (ghRelease.status !== 0) {
  console.error(`[release] gh release create failed (exit ${ghRelease.status}).`);
  console.error('[release] if the release already exists, delete it first:');
  console.error(`           gh release delete ${tag} --repo ${repo} --yes`);
  process.exit(ghRelease.status ?? 1);
}
console.log(`[release] GitHub Release ${tag} published.`);

// ──────────────────────────────────────────────────────────────────────────
// Step 6: Write latest.json to the gh-pages branch.
// Tauri's updater polls this URL:
//   https://chloeneko.github.io/WUPI/updater/latest.json
// The manifest format (Tauri v2): version + notes + pub_date + per-platform
// { signature, url }.
// ──────────────────────────────────────────────────────────────────────────
console.log('[release] publishing latest.json to gh-pages…');
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

// Write the manifest to a temp file, then push it to gh-pages via a sparse
// checkout (so we don't pull the whole gh-pages history every release).
const tmpManifest = join(require('os').tmpdir(), 'wupi-latest.json');
require('fs').writeFileSync(tmpManifest, manifestJson);

const ghPagesOps = [
  // Branch off fresh if gh-pages doesn't exist; otherwise check it out.
  ['git', ['fetch', 'origin', 'gh-pages:gh-pages'], { okIfNonZero: true }],
  ['git', ['checkout', 'gh-pages'], { okIfNonZero: false }],  // success: existing
  // If checkout failed, create the branch orphan from main
  ['git', ['checkout', '--orphan', 'gh-pages'], { okIfNonZero: true, onlyIfPrevFailed: true }],
];
// Run them as a sequence; simpler than encoding conditional logic into ops.
const gitRes = {};
gitRes.fetch = spawnSync('git', ['fetch', 'origin', 'gh-pages:gh-pages'], {
  cwd: repoRoot, stdio: 'pipe',
});
const hadGhPages = gitRes.fetch.status === 0;
if (hadGhPages) {
  spawnSync('git', ['checkout', 'gh-pages'], { cwd: repoRoot, stdio: 'inherit' });
} else {
  console.log('[release] gh-pages does not exist yet; creating orphan branch…');
  spawnSync('git', ['checkout', '--orphan', 'gh-pages'], { cwd: repoRoot, stdio: 'inherit' });
  // Wipe the index (orphan branch starts with everything staged from HEAD)
  spawnSync('git', ['rm', '-rf', '--cached', '.'], { cwd: repoRoot, stdio: 'ignore' });
}

// Copy the manifest into place + commit + push.
mkdirSync(join(repoRoot, 'updater'), { recursive: true });
copyFileSync(tmpManifest, join(repoRoot, 'updater', 'latest.json'));

spawnSync('git', ['add', 'updater/latest.json'], { cwd: repoRoot, stdio: 'inherit' });
const commitManifest = spawnSync('git', [
  'commit', '-m', `release manifest: ${tag} [skip ci]`,
], { cwd: repoRoot, stdio: 'inherit' });
if (commitManifest.status !== 0) {
  console.warn('[release] no manifest changes to commit (already up to date?)');
}
const pushManifest = spawnSync('git', ['push', 'origin', 'gh-pages'], {
  cwd: repoRoot, stdio: 'inherit',
});
if (pushManifest.status !== 0) {
  console.error(`[release] failed to push gh-pages (exit ${pushManifest.status}).`);
  console.error('[release] manifest written locally; manual push needed.');
}

// ──────────────────────────────────────────────────────────────────────────
// Step 7: Switch back to ui-shell so the user's working tree is restored.
// ──────────────────────────────────────────────────────────────────────────
spawnSync('git', ['checkout', 'ui-shell'], { cwd: repoRoot, stdio: 'inherit' });

console.log('\n[release] ========================================');
console.log(`[release]  RELEASE ${tag} PUBLISHED`);
console.log('[release] ========================================');
console.log(`[release]  Release:   https://github.com/${repo}/releases/tag/${tag}`);
console.log(`[release]  Manifest:  https://chloeneko.github.io/WUPI/updater/latest.json`);
console.log(`[release]  Asset URL: ${assetUrl}`);
console.log('[release]  Next tester launch → auto-update fires.');
console.log('[release] ========================================');
