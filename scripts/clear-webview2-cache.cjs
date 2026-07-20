// Clears the WebView2 persistent cache for WUPI so a freshly-built exe
// doesn't serve stale frontend from the previous run. Run before every build.
// Safe to run when the app is closed; no-op if the cache dir doesn't exist.
const { rmSync, existsSync } = require('fs');
const { join } = require('path');

const cacheDir = join(
  process.env.LOCALAPPDATA || join(require('os').homedir(), 'AppData', 'Local'),
  'com.wupi.desktop',
  'EBWebView'
);

if (existsSync(cacheDir)) {
  rmSync(cacheDir, { recursive: true, force: true });
  console.log('[clear-webview2-cache] cleared:', cacheDir);
} else {
  console.log('[clear-webview2-cache] no cache to clear');
}
