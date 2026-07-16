// WUPI Terminal frontend — xterm.js wired to the Rust PTY (terminal.rs).
// Bytes flow PTY → Tauri Channel → xterm.write(); keystrokes flow
// xterm.onData → terminal_input. Resize is forwarded to terminal_resize.

import { Terminal } from '@xterm/xterm';
import { FitAddon } from '@xterm/addon-fit';
import { WebLinksAddon } from '@xterm/addon-web-links';
import { invoke, Channel } from '@tauri-apps/api/core';
import { getCurrentWebviewWindow } from '@tauri-apps/api/webviewWindow';

import '@xterm/xterm/css/xterm.css';

// Decode base64 (matches the Rust-side encoder in terminal.rs).
function b64decode(s) {
  const bin = atob(s);
  const len = bin.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}
function b64encode(bytes) {
  let bin = '';
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin);
}

// The window label is the terminal id used by the backend registry.
const label = getCurrentWebviewWindow().label;

const term = new Terminal({
  fontFamily: '"Cascadia Code", "Consolas", "SF Mono", Menlo, monospace',
  fontSize: 14,
  cursorBlink: true,
  cursorStyle: 'bar',
  allowProposedApi: true,
  theme: {
    background: 'rgba(0,0,0,0)',
    foreground: '#f3e9ff',
    cursor: '#ff66b2',
    cursorAccent: 'rgba(0,0,0,0)',
    selectionBackground: 'rgba(181, 52, 250, 0.35)',
    black:   '#1a0f2e',
    red:     '#ff6b9d',
    green:   '#b5f0c8',
    yellow:  '#ffe08a',
    blue:    '#9d8bff',
    magenta: '#ff66b2',
    cyan:    '#8de8ff',
    white:   '#e8d8ff',
    brightBlack:   '#3a2a52',
    brightRed:     '#ff9dc0',
    brightGreen:   '#c8f5d8',
    brightYellow:  '#ffecb3',
    brightBlue:    '#b3a8ff',
    brightMagenta: '#ff9dca',
    brightCyan:    '#b3efff',
    brightWhite:   '#ffffff',
  },
});

const fit = new FitAddon();
term.loadAddon(fit);
term.loadAddon(new WebLinksAddon());

const mount = document.getElementById('term');
term.open(mount);

// Fit to the window, then open the PTY with the initial size so the shell
// gets the right dimensions immediately.
const fitNow = () => { try { fit.fit(); } catch (_) { /* not laid out yet */ } };
fitNow();

// Keystrokes → PTY stdin.
term.onData((data) => {
  const bytes = new TextEncoder().encode(data);
  invoke('terminal_input', { label, data: b64encode(bytes) })
    .catch((e) => console.error('[term] input failed', e));
});

// Resize → PTY resize.
term.onResize(({ cols, rows }) => {
  invoke('terminal_resize', { label, cols, rows })
    .catch((e) => console.error('[term] resize failed', e));
});

// Channel delivers PTY stdout. { kind: "data", data: <base64>, label }.
const chan = new Channel();
chan.onmessage = (msg) => {
  if (!msg) return;
  if (msg.kind === 'data') {
    term.write(b64decode(msg.data));
  } else if (msg.kind === 'exit') {
    term.write('\r\n\x1b[90m[process exited]\x1b[0m\r\n');
  }
};

// Initialize the PTY for this window; this returns the label the backend
// registered under. The window itself was already created by the paw-menu
// `terminal_create_window` call.
invoke('terminal_init', { onEvent: chan })
  .then(() => {
    // After the backend has the label, push the initial cols/rows so the
    // shell prompt aligns to the window.
    invoke('terminal_resize', { label, cols: term.cols, rows: term.rows })
      .catch(() => {});
  })
  .catch((e) => {
    term.write(`\x1b[31mFailed to init terminal: ${e}\x1b[0m\r\n`);
    console.error('[term] init failed', e);
  });

// Magenta X hides the window — the PTY keeps running.
document.getElementById('termClose').addEventListener('click', (e) => {
  e.stopPropagation();
  invoke('terminal_close', { label }).catch((err) =>
    console.error('[term] close failed', err)
  );
});

// Keep xterm sized to the window on every OS resize.
let resizeTimer = null;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(fitNow, 30);
});

// Refit shortly after load in case the webview wasn't sized yet at first fit.
setTimeout(fitNow, 80);
