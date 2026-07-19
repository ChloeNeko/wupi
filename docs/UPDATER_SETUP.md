# Auto-Updater Setup

The in-app updater is fully wired (boot-screen auto-check + paw-menu "Check
for Updates" button). Releases are **built locally** by the maintainer (WUPI
requires CUDA to compile, which GitHub's CI runners don't have) and published
by the `npm run release` script. This doc walks through the one-time setup
and the per-release flow.

## Why signing is required

Tauri's updater refuses to install any payload that isn't signed with a
minisign key whose public half is baked into `tauri.conf.json`. This prevents
a compromised host from serving a malicious build to your beta testers. The
public key is already in the config; the private key is yours alone and
**never goes into the repo** — only into `~/.tauri/wupi.key` on your machine.

## Why builds happen locally (not CI)

WUPI's `llama-cpp-2` crate requires the CUDA Toolkit to compile (the chat
model runs on GPU). GitHub's standard `windows-latest` runners don't have
CUDA installed. Building locally uses your warm cache (~3 min per release)
and avoids the 30+ minute cold-compile CI would need even if it had CUDA.
Beta testers never need CUDA — they just download the built `.exe`.

---

## One-time setup

### Step 1: Generate the signing keypair (on your dev machine)

```bash
npx @tauri-apps/cli signer generate -w ~/.tauri/wupi.key
```

Pick a password (or leave blank for an unencrypted key — both work with the
release script). This creates:

- `~/.tauri/wupi.key` — **the private key. Never commit this. Never share it.**
- `~/.tauri/wupi.key.pub` — the public key.

Verify the public key in `src-tauri/tauri.conf.json` (`plugins.updater.pubkey`)
matches the `.pub` you just generated. The current value in the config is
base64-encoded; `signer generate` prints the base64 string to paste. Replace
the `pubkey` field if it doesn't match, then commit.

If you used a password, save it so the release script can read it:
```bash
echo 'your-password' > ~/.tauri/wupi.key.pw
```
(Skip this if you used an empty password — the script handles that case.)

### Step 2: Authenticate the `gh` CLI (if not already done)

The release script uses `gh release create` to publish. Authenticate once:
```bash
gh auth login
```
Follow the prompts (HTTPS + browser auth is easiest).

### Step 3: Enable GitHub Pages on the `gh-pages` branch

The Tauri updater polls `https://chloeneko.github.io/WUPI/updater/latest.json`.
GitHub Pages serves that URL from a `gh-pages` branch. The `npm run release`
script creates the branch on first run; enable Pages after your first release:

1. Run `npm run release` once (see "Per-release flow" below) — this creates
   the `gh-pages` branch with `updater/latest.json`.
2. Go to **https://github.com/ChloeNeko/WUPI/settings/pages**
3. **Build and deployment → Source**: "Deploy from a branch"
4. **Branch**: `gh-pages` → `/ (root)` → **Save**

Pages goes live in 1-2 min. Verify by loading the manifest URL in a browser
— it should return a JSON object with `version`, `platforms`, etc.

### Step 4: Set `HF_TOKEN` env var (for distributed GGUF downloader)

The first-run GGUF downloader in `model_downloader.rs` pulls `HF_TOKEN` from
the build environment at compile time and bakes the value into the binary.
On your dev machine, set it before running `npm run release`:

**Git Bash (one-time, then restart your shell):**
```bash
echo 'export HF_TOKEN=hf_your_token_here' >> ~/.bashrc
source ~/.bashrc
```

**Or per-release (prefix the command):**
```bash
HF_TOKEN=hf_your_token_here npm run release
```

Get the token from https://huggingface.co/settings/tokens → New token →
Fine-grained → Read access to contents of `ChloeNeko/WUPI`. Without this,
distributed builds compile fine but their in-app downloader 401s on first
run (local devs don't need it — they have the GGUFs on disk).

---

## Per-release flow (the part you do each time)

When you're ready to ship an update to beta testers:

```bash
npm run release
```

That's it. The script:

1. **Bumps the version** in `tauri.conf.json` (patch by default; `--minor`
   or `--major` for bigger bumps; `--no-bump` to re-release current).
2. **Commits the version bump** with message `release: vX.Y.Z`.
3. **Runs `npx tauri build`** with the signing env vars (this is the long
   step — ~3 min on a warm cache, 30+ min cold).
4. **Creates a GitHub Release** at tag `vX.Y.Z` and uploads the signed
   installer + signature + zip payloads.
5. **Writes `latest.json`** to the `gh-pages` branch and pushes it.
6. **Switches back to `ui-shell`** so your working tree is restored.

On each beta tester's next WUPI launch:
- The boot-screen check reads the new `latest.json`, announces the update in
  the LOADING OS terminal stream.
- They click the paw → **Check for Updates** to install immediately (or it
  installs silently on the next restart).

### Useful variants

```bash
npm run release -- --dry-run       # build + print what would happen, no upload
npm run release -- --minor         # bump minor version (0.1.5 → 0.2.0)
npm run release -- --no-bump       # re-release current version (e.g. after a broken release)
```

---

## Notes / gotchas

- **Models are NOT re-downloaded on update.** The GGUFs live in
  `%APPDATA%\com.wupi.os\models\` (the app data dir), which is separate from
  the install dir the NSIS updater overwrites. Fresh installs get the
  download overlay; updates never do.
- **Testers never need CUDA.** The compiled `.exe` has the CUDA runtime
  statically linked in. They just need an NVIDIA driver (which gamers
  already have).
- **Re-releasing the same version.** If `npm run release` fails because the
  tag already exists, delete the old release first:
  ```bash
  gh release delete vX.Y.Z --repo ChloeNeko/WUPI --yes
  git tag -d vX.Y.Z
  ```
  Then re-run. (Or use `--no-bump` to keep the same version.)
- **Rotating the signing key.** If the private key ever leaks: generate a
  new keypair, replace `pubkey` in `tauri.conf.json`, commit, run
  `npm run release`. Old builds won't be able to auto-update (signature
  mismatch) — testers reinstall once, then they're on the new key.
- **Rotating the HF token.** Revoke the old one at HF settings, generate a
  new fine-grained read-only token, update your `HF_TOKEN` env var (or
  `~/.bashrc`), and re-run `npm run release`. No source change needed.

> **Live-token note (2026-07-19):** an earlier version of the downloader
> hardcoded a token (`hf_GdgPcd…`) directly in source. The token was
> committed briefly when the repo flipped public, then purged from git
> history. **The token itself is still live** at Hugging Face until manually
> revoked — scoped read-only to `ChloeNeko/WUPI`, so the realistic blast
> radius is limited to GGUF downloads. Revoke it at
> https://huggingface.co/settings/tokens when convenient.
