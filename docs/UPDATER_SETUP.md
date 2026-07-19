# Auto-Updater Setup

The in-app updater is fully wired (boot-screen auto-check + paw-menu "Check
for Updates" button), and `.github/workflows/release.yml` builds + signs +
publishes on every push to `ui-shell`. **It will not work until you do this
3-step signer setup once.** Until then, the build CI fails on the signing
step and the in-app check silently no-ops (the manifest doesn't exist yet).

## Why signing is required

Tauri's updater refuses to install any payload that isn't signed with a
minisign key whose public half is baked into `tauri.conf.json`. This prevents
a compromised host from serving a malicious build to your beta testers. The
public key is already in the config; the private key is yours alone and
**never goes into the repo** — only into GitHub Secrets.

---

## Step 1: Generate the signing keypair (one time, on your dev machine)

From the WUPI repo root:

```bash
npx @tauri-apps/cli signer generate -w ~/.tauri/wupi.key
```

It will prompt for a password (pick one; you'll reuse it). This creates:

- `~/.tauri/wupi.key` — **the private key. Never commit this. Never share it.**
- `~/.tauri/wupi.key.pub` — the public key. This is what goes in the repo.

Then verify the public key in `src-tauri/tauri.conf.json` matches the `.pub`
you just generated. The current value in the config is base64-encoded; the
`signer generate` command prints the base64 string to paste. **Replace the
`pubkey` field with your real value** if it doesn't match:

```json
"plugins": {
  "updater": {
    "pubkey": "<PASTE THE BASE64 FROM `signer generate` HERE>",
    ...
  }
}
```

Commit that change.

## Step 2: Add the secrets to GitHub

In your browser, go to:

```
https://github.com/ChloeNeko/WUPI/settings/secrets/actions
```

Click **"New repository secret"** and add three:

| Name | Value |
|------|-------|
| `TAURI_SIGNING_PRIVATE_KEY` | The entire contents of `~/.tauri/wupi.key` (a multi-line base64 block starting `untrusted comment:`) |
| `TAURI_KEY_PASSWORD` | The password you chose in Step 1 |
| `HF_TOKEN` | A Hugging Face fine-grained read-only access token scoped to `ChloeNeko/WUPI` (the repo holding `WUPI.gguf` + `Embed.gguf`). Create at https://huggingface.co/settings/tokens → New token → Fine-grained → Read access to contents of selected repos → pick `ChloeNeko/WUPI`. Starts with `hf_…`. |

To get the private key contents on your dev machine (Git Bash):

```bash
cat ~/.tauri/wupi.key
# Copy the entire output, including the "untrusted comment:" line, and
# paste it into the TAURI_SIGNING_PRIVATE_KEY secret value.
```

**Why `HF_TOKEN`:** the first-run GGUF downloader in `model_downloader.rs`
pulls this at *build time* via `option_env!("HF_TOKEN")` and bakes the value
into the binary. The source never contains the token itself — only the macro
call. Distributed builds (from CI) have the real token; local dev builds
without the env var produce an empty string (the downloader 401s, but local
devs already have the GGUFs on disk so the overlay never fires).

**Rotation:** to rotate `HF_TOKEN`, revoke the old one at HF settings,
generate a new fine-grained read-only token, update the GitHub Secret, and
push a fresh build. No source change needed.

> **Live-token note (2026-07-19):** an earlier version of the downloader
> hardcoded a token (`hf_GdgPcd…`) directly in source. The token was
> committed briefly when the repo flipped public, then purged from git
> history. **The token itself is still live** at Hugging Face until manually
> revoked — scoped read-only to `ChloeNeko/WUPI`, so the realistic blast
> radius is limited to GGUF downloads. Revoke it at
> https://huggingface.co/settings/tokens when convenient; the `HF_TOKEN`
> secret + a fresh CI build is the rotation path.

## Step 3: Enable GitHub Pages on the `gh-pages` branch

The updater polls `https://chloeneko.github.io/WUPI/updater/latest.json`. That
URL is served by GitHub Pages from a `gh-pages` branch the workflow creates on
first run. Enable it once:

1. Push any commit to `ui-shell` so the workflow runs and creates the
   `gh-pages` branch (it'll fail at the signing step until Secrets are set,
   but the branch may still get created — if not, Step 3's branch selector
   won't show it until the workflow succeeds once).
2. Go to **https://github.com/ChloeNeko/WUPI/settings/pages**
3. Under **"Build and deployment" → "Source"**, pick **"Deploy from a branch"**
4. Under **"Branch"**, select **`gh-pages`** → folder **`/ (root)`** → **Save**

Pages can take 1-2 minutes to go live. Verify the manifest URL loads:

```
https://chloeneko.github.io/WUPI/updater/latest.json
```

It should return a JSON object with `version`, `platforms`, etc.

---

## After setup: how it flows

1. You `git push origin ui-shell` with any change.
2. GitHub Actions builds the NSIS installer + MSI, signs both with your key.
3. A prerelease GitHub Release is published at the version stamp.
4. `latest.json` on `gh-pages` updates to point at the new signed artifact.
5. On each beta tester's next WUPI launch, the boot-screen check finds the
   new version and announces it in the LOADING OS terminal stream; on next
   restart it installs + relaunches. They can also click the paw →
   **Check for Updates** to install immediately.

## Notes / gotchas

- **Models are NOT re-downloaded on update.** The GGUFs live in
  `%APPDATA%\com.wupi.os\models\` (the app data dir), which is separate from
  the install dir the NSIS updater overwrites. A fresh install needs the
  download overlay; an update never does.
- **Private repo caveat.** GitHub Release asset URLs require auth on private
  repos. The `latest.json` manifest is served publicly via Pages even on
  private repos, but the download URLs it points to may 401 for the updater
  client. If beta testers see "update download failed," the fix is to either
  flip the repo to public (simplest) or proxy the binaries through a host
  you control (VPS / Cloudflare R2). The manifest itself is fine either way.
- **Rotating the key.** If the private key ever leaks, generate a new
  keypair, replace the `pubkey` in `tauri.conf.json`, update both Secrets,
  and push. Old builds won't be able to auto-update (signature mismatch) —
  testers will need to manually reinstall once, then they're back on the
  new key.
- **Versioning.** Each push produces a `v0.1.0-dev.<timestamp>` prerelease
  tag. To ship a "real" numbered release (e.g. `v0.2.0`), edit the `version`
  field in `src-tauri/tauri.conf.json` and the workflow will use it for the
  release tag automatically.
