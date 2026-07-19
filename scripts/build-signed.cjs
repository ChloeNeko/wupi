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
const keyPath = keyPathOverride
  || process.env.WUPI_SIGNING_KEY_PATH
  || join(homedir(), '.tauri', 'wupi.key');

if (!existsSync(keyPath)) {
  console.error(`[build-signed] private key not found at: ${keyPath}`);
  console.error('[build-signed] generate one with:');
  console.error('  npx @tauri-apps/cli signer generate -w ~/.tauri/wupi.key');
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
//    2. TAURI_KEY_PASSWORD env var
//    3. ~/.tauri/wupi.key.pw file
//    4. empty string (key was generated without a password)
// ──
const pwFilePath = join(homedir(), '.tauri', 'wupi.key.pw');
let password = '';
if (passwordFromArg) {
  password = passwordFromArg;
} else if (process.env.TAURI_KEY_PASSWORD) {
  password = process.env.TAURI_KEY_PASSWORD;
} else if (existsSync(pwFilePath)) {
  try {
    password = readFileSync(pwFilePath, 'utf8').replace(/\r?\n$/, '');
    console.log('[build-signed] password loaded from ~/.tauri/wupi.key.pw');
  } catch (e) {
    console.error(`[build-signed] failed to read password file: ${e.message}`);
    process.exit(1);
  }
} else {
  console.warn('[build-signed] no password source found (no --password flag,');
  console.warn('                 no TAURI_KEY_PASSWORD env, no ~/.tauri/wupi.key.pw file).');
  console.warn('                 If the key was generated WITH a password, this will fail.');
}

// ── Build the child env: parent env + the two signing vars. ──
//    The vars are scoped to this.spawn call — they do NOT leak into the
//    parent shell (Node never mutates process.env here).
const childEnv = {
  ...process.env,
  TAURI_SIGNING_PRIVATE_KEY: privateKey,
  TAURI_KEY_PASSWORD: password,
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
