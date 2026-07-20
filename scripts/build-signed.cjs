// Signed-release build wrapper for WUPI.
//
// Reads the Tauri updater private key + (optional) password, exports them as
// the TAURI_SIGNING_PRIVATE_KEY + TAURI_KEY_PASSWORD env vars for the child
// process ONLY (never the global env, never written to disk), then runs
// `npx tauri build`. Required so tauri-plugin-updater can sign the .nsis.zip
// / .msi.zip artifacts + the latest.json signature — without this,
// `createUpdaterArtifacts` in tauri.conf.json errors out post-build with
// "A public key has been found, but no private key."
//
// Usage:
//   npm run build:signed                          # password read from ~/.tauri/wupi.key.pw if present
//   npm run build:signed -- --password SECRET     # password inline (skips the .pw file)
//   TAURI_KEY_PASSWORD=SECRET npm run build:signed  # same, via env
//
// Files (all OUTSIDE the repo, never tracked):
//   ~/.tauri/wupi.key       REQUIRED — the private key (generated with `tauri signer generate`)
//   ~/.tauri/wupi.key.pw    OPTIONAL — single-line file containing just the password.
//                          Create with: echo 'your-password' > ~/.tauri/wupi.key.pw
//                          The --password flag + TAURI_KEY_PASSWORD env override this file.
//
// The private key path can be overridden with --key-path or the
// WUPI_SIGNING_KEY_PATH env var.
//
// All args after the script name are forwarded to `tauri build` (e.g.
// `npm run build:signed -- --debug` would pass --debug to tauri).
//
// Security notes:
// - The private key + password are read into memory + passed via the child
//   process env. They are NEVER logged, NEVER written to a file by this
//   script, NEVER set in the parent shell. When the build exits, the
//   secrets die with the child process.
// - The pubkey (safe to commit) lives in tauri.conf.json. This script only
//   handles the private half.
// - TAURI_KEY_PASSWORD is the env var Tauri's signer reads to skip the
//   stdin prompt — by setting it here, the build runs non-interactively.

const { readFileSync, existsSync } = require('fs');
const { join } = require('path');
const { homedir } = require('os');
const { spawn } = require('child_process');

// ── Parse args: pull --password + --key-path out, forward the rest. ──
const argv = process.argv.slice(2);
const forwardedArgs = [];
let passwordFromArg = null;
let keyPathOverride = null;
for (let i = 0; i < argv.length; i++) {
  const a = argv[i];
  if (a === '--password' || a === '-p') {
    passwordFromArg = argv[++i];
  } else if (a.startsWith('--password=')) {
    passwordFromArg = a.slice('--password='.length);
  } else if (a === '--key-path') {
    keyPathOverride = argv[++i];
  } else if (a.startsWith('--key-path=')) {
    keyPathOverride = a.slice('--key-path='.length);
  } else {
    forwardedArgs.push(a);
  }
}

// ── Resolve the private key path. ──
// Repo-relative `keys/wupi.key` wins (matches the project's gitignored
// `keys/` convention); falls back to `~/.tauri/wupi.key` for legacy setups.
// Override with --key-path or WUPI_SIGNING_KEY_PATH.
const repoRoot = join(__dirname, '..');
const repoRootKeyPath = join(repoRoot, 'keys', 'wupi.key');
const homeKeyPath = join(homedir(), '.tauri', 'wupi.key');
const keyPath = keyPathOverride
  || process.env.WUPI_SIGNING_KEY_PATH
  || (existsSync(repoRootKeyPath) ? repoRootKeyPath : homeKeyPath);

if (!existsSync(keyPath)) {
  console.error(`[build-signed] private key not found at: ${keyPath}`);
  console.error('[build-signed] generate one with:');
  console.error('  npx @tauri-apps/cli signer generate -w keys/wupi.key');
  console.error('  (or set WUPI_SIGNING_KEY_PATH to point elsewhere)');
  process.exit(1);
}

// ── Read the private key into memory. NEVER log it. ──
let privateKey;
try {
  privateKey = readFileSync(keyPath, 'utf8').trim();
} catch (e) {
  console.error(`[build-signed] failed to read private key: ${e.message}`);
  process.exit(1);
}
if (!privateKey) {
  console.error('[build-signed] private key file is empty.');
  process.exit(1);
}

// ── Password resolution (in priority order):
//    1. --password flag
//    2. TAURI_SIGNING_PRIVATE_KEY_PASSWORD env var (Tauri's current name)
//    3. TAURI_KEY_PASSWORD env var (legacy name, kept for backward compat)
//    4. ~/.tauri/wupi.key.pw file
//    5. empty string (key was generated without a password)
//
//    IMPORTANT: Tauri's signer reads `TAURI_SIGNING_PRIVATE_KEY_PASSWORD`
//    as of Tauri 2.x. The older `TAURI_KEY_PASSWORD` name is no longer
//    recognized by the signer itself (it silently falls through to the
//    interactive rpassword prompt and hangs the build). We accept both
//    env vars as INPUT here, but pass only the correct one to the child
//    process below.
// ──
const pwFilePath = existsSync(join(repoRoot, 'keys', 'wupi.key.pw'))
  ? join(repoRoot, 'keys', 'wupi.key.pw')
  : join(homedir(), '.tauri', 'wupi.key.pw');
let password = '';
if (passwordFromArg) {
  password = passwordFromArg;
} else if (process.env.TAURI_SIGNING_PRIVATE_KEY_PASSWORD) {
  password = process.env.TAURI_SIGNING_PRIVATE_KEY_PASSWORD;
} else if (process.env.TAURI_KEY_PASSWORD) {
  password = process.env.TAURI_KEY_PASSWORD;
} else if (existsSync(pwFilePath)) {
  try {
    password = readFileSync(pwFilePath, 'utf8').replace(/\r?\n$/, '');
    console.log(`[build-signed] password loaded from ${pwFilePath}`);
  } catch (e) {
    console.error(`[build-signed] failed to read password file: ${e.message}`);
    process.exit(1);
  }
} else {
  console.warn('[build-signed] no password source found (no --password flag,');
  console.warn('                 no TAURI_SIGNING_PRIVATE_KEY_PASSWORD env, no ~/.tauri/wupi.key.pw file).');
  console.warn('                 If the key was generated WITH a password, this will hang.');
}

// ── HF_TOKEN check (compile-time dependency, see release.cjs for full rationale). ──
//    The compiled binary bakes HF_TOKEN in via `option_env!` in
//    src-tauri/src/model_downloader.rs. If it's not in the env at build
//    time, the constant is "" and first-run GGUF downloads 403 against the
//    private ChloeNeko/WUPI HF repo. Warn loudly so this never silently
//    ships broken.
if (!process.env.HF_TOKEN) {
  console.warn('[build-signed] !! HF_TOKEN not set in environment.');
  console.warn('                 The first-run GGUF download (model_downloader.rs) bakes this');
  console.warn('                 token into the binary at compile time. Without it, fresh');
  console.warn('                 installs will get 403 from HuggingFace on the download overlay.');
  console.warn('                 To fix: export HF_TOKEN=hf_<fine-grained-read-only> before');
  console.warn('                 running this script. See docs/UPDATER_SETUP.md.');
}

// ── Build the child env: parent env + the two signing vars. ──
//    The vars are scoped to this.spawn call — they do NOT leak into the
//    parent shell (Node never mutates process.env here). Both Tauri-era
//    names are set so any external tool reading either one finds the value.
const childEnv = {
  ...process.env,
  TAURI_SIGNING_PRIVATE_KEY: privateKey,
  TAURI_SIGNING_PRIVATE_KEY_PASSWORD: password,
  TAURI_KEY_PASSWORD: password,
  // HF_TOKEN forwarded explicitly (process.env spread above already includes
  // it, but listing it here makes the compile-time dependency visible at
  // the call site — see model_downloader.rs:88).
  HF_TOKEN: process.env.HF_TOKEN || '',
  // Preserve the CUDA build parallelism used by the regular build path.
  CMAKE_BUILD_PARALLEL_LEVEL: process.env.CMAKE_BUILD_PARALLEL_LEVEL || '8',
};

console.log(`[build-signed] using key: ${keyPath}`);
console.log('[build-signed] TAURI_SIGNING_PRIVATE_KEY + TAURI_KEY_PASSWORD set');
console.log('                 for child process only.');
console.log(`[build-signed] running: npx tauri build ${forwardedArgs.join(' ')}`);

// ── Spawn tauri build with stdio inherited so output streams live. ──
//    shell:true on Windows because npx is a .cmd shim — Node's spawn with
//    shell:false throws EINVAL on .cmd files. The args are static + ours,
//    so the shell-injection surface is nil.
const child = spawn(
  `npx tauri build ${forwardedArgs.join(' ')}`,
  [],
  { env: childEnv, stdio: 'inherit', shell: true }
);

child.on('close', (code) => {
  // Exit with the child's code so npm reports success/failure correctly.
  // Don't log the key, don't dump env — just the exit status.
  if (code !== 0) {
    console.error(`[build-signed] tauri build exited with code ${code}.`);
  } else {
    console.log('[build-signed] signed build complete.');
  }
  process.exit(code ?? 0);
});

child.on('error', (err) => {
  console.error(`[build-signed] failed to spawn tauri build: ${err.message}`);
  process.exit(1);
});
