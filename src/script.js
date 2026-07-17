// Tauri 2 IPC + event APIs. Imported as ES modules now that script.js is
// `type="module"` (Vite bundles these; withGlobalTauri is off so the
// `window.__TAURI__` global is NOT injected — the import is the source of truth).
import { invoke, Channel } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';

const canvas = document.getElementById('aurora-canvas');
const ctx = canvas.getContext('2d');

// ── Theme palettes ─────────────────────────────────────────────────────────
// Each color code defines the aurora's sky gradient (top→bottom CSS color
// stops) and the curtain hue generator (base hue ± range). The animate() loop
// reads `currentPalette` — switching color codes re-paints on the next frame.
//
// "Vibrant" reproduces the original hardcoded values (the project default).
// New color codes = add entries here + a matching swatch in styles.css.
const COLOR_CODES = {
  Vibrant: {
    skyGradient: ['#02040a', '#060a17', '#150524', '#2b0b36', '#4a173d'],
    hueBase: 305,
    hueRange: 45,
  },
};

// The live palette; initialized from the persisted theme at boot (see
// `applyTheme` below). Defaults to Vibrant so the canvas paints immediately
// even before the IPC round-trip completes.
let currentPalette = COLOR_CODES.Vibrant;

// CSS-pixel dimensions (all drawing math uses these). The backing store is
// scaled by devicePixelRatio so stars/curtains render at physical-pixel
// resolution on high-DPI / 4K / ultrawide displays instead of being upscaled
// and blurry. This is the "resolution loss" fix.
let width, height;

function resize() {
  const dpr = window.devicePixelRatio || 1;
  width = window.innerWidth;
  height = window.innerHeight;
  canvas.width = Math.floor(width * dpr);
  canvas.height = Math.floor(height * dpr);
  canvas.style.width = width + 'px';
  canvas.style.height = height + 'px';
  // Reset transform then re-apply — resize() can fire repeatedly, and the
  // scale accumulates if not reset first.
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.scale(dpr, dpr);
}
window.addEventListener('resize', resize);
resize();

let mouseX = 0;
let mouseY = 0;
let currentX = 0;
let currentY = 0;

window.addEventListener('mousemove', (e) => {
  mouseX = (e.clientX / width) * 2 - 1;
  mouseY = (e.clientY / height) * 2 - 1;
});

const starCount = 1000;
const stars = Array.from({ length: starCount }, () => {
  const isTwinkling = Math.random() > 0.98;
  // colorIdx indexes STAR_COLORS — drawing buckets stars by color so the
  // context's fillStyle changes ~4×/frame instead of ~1000×.
  const colorIdx = Math.floor(Math.random() * 4);

  return {
    x: Math.random() * width,
    y: Math.random() * height,
    size: Math.random() * 0.9 + 0.4,
    alpha: Math.random() * 0.7 + 0.3,
    isTwinkling: isTwinkling,
    speed: isTwinkling ? (0.0005 + Math.random() * 0.0012) : 0,
    drift: Math.random() * 0.01 + 0.008 + 0.004,
    colorIdx: colorIdx,
  };
});

let time = 0;

// ── Cached sky gradient (perf: was recreated every frame) ─────────────────
// The gradient depends only on the palette + canvas height, both of which
// change rarely (theme switch / resize). Recreating it 60×/sec was pure waste
// — createLinearGradient + 5 addColorStop calls per frame. Rebuilt only when
// `currentPalette` or `height` changes.
let cachedSkyGrad = null;
let cachedSkyHeight = -1;
function skyGradient() {
  if (cachedSkyGrad && cachedSkyHeight === height) return cachedSkyGrad;
  const g = ctx.createLinearGradient(0, 0, 0, height);
  const stops = currentPalette.skyGradient;
  for (let i = 0; i < stops.length; i++) {
    g.addColorStop(i / (stops.length - 1), stops[i]);
  }
  cachedSkyGrad = g;
  cachedSkyHeight = height;
  return g;
}
// Invalidate the cache on resize (height changes → gradient must rebuild).
window.addEventListener('resize', () => { cachedSkyGrad = null; });

// ── Stars bucketed by color (perf: was 1000 fillStyle/globalAlpha toggles) ─
// Batching same-color stars into one fillStyle set + grouping alpha into a
// few bands collapses ~1000 state changes/frame into a handful. The visual
// difference is imperceptible (alpha quantized to 8 bands of 0.1).
const STAR_COLORS = ['#ffffff', '#e8f0ff', '#fff4e6', '#ffe6ee'];

function animate() {
  currentX += (mouseX - currentX) * 0.25;
  currentY += (mouseY - currentY) * 0.25;

  // Sky (cached gradient — see skyGradient()).
  ctx.globalCompositeOperation = 'source-over';
  ctx.globalAlpha = 1.0;
  ctx.fillStyle = skyGradient();
  ctx.fillRect(0, 0, width, height);

  // Stars: update positions/twinkle, then draw bucketed by color+alpha-band
  // so the context state changes once per bucket, not once per star.
  const px = currentX * 16;
  const py = currentY * 16;
  // buckets[colorIdx][alphaBand] = [{x,y,size}, ...]
  const buckets = [[[],[],[],[],[],[],[],[]],[[],[],[],[],[],[],[],[]],[[],[],[],[],[],[],[],[]],[[],[],[],[],[],[],[],[]]];
  for (let i = 0; i < stars.length; i++) {
    const s = stars[i];
    if (s.isTwinkling) {
      s.alpha += s.speed;
      if (s.alpha > 1 || s.alpha < 0.15) s.speed = -s.speed;
    }
    s.y -= s.drift;
    if (s.y < 0) s.y = height;
    const band = Math.min(7, Math.max(0, Math.floor(Math.abs(s.alpha) * 8)));
    buckets[s.colorIdx][band].push(s.x + px * s.size, s.y + py * s.size, s.size);
  }
  for (let c = 0; c < STAR_COLORS.length; c++) {
    ctx.fillStyle = STAR_COLORS[c];
    for (let b = 0; b < 8; b++) {
      const pts = buckets[c][b];
      if (pts.length === 0) continue;
      ctx.globalAlpha = (b + 0.5) / 8;
      for (let k = 0; k < pts.length; k += 3) {
        ctx.fillRect(pts[k], pts[k + 1], pts[k + 2], pts[k + 2]);
      }
    }
  }
  ctx.globalAlpha = 1.0;

  // Aurora borealis — 5 layered, independently-hued curtains. Each curtain
  // gets its own hue oscillation + its own blurred fill, which is what
  // produces the multi-color ribbon effect. The blur is expensive (a full
  // Gaussian pass per fill) but it IS the look — the soft bloom is the whole
  // point. Kept as-is per Chloe: do NOT collapse into one fill.
  // The perf gains live elsewhere (visibility pause, cached sky, star
  // batching) so the aurora can stay visually rich.
  ctx.globalCompositeOperation = 'screen';
  ctx.filter = 'blur(30px)';

  const curtains = 5;
  const baseCenterY = height * 0.42;

  for (let i = 0; i < curtains; i++) {
    const speed = time * (0.1 + i * 0.04);
    const thickness = 45 + i * 15;

    const yOffset = (i - (curtains / 2)) * 12;
    const activeCenterY = baseCenterY + yOffset;

    ctx.beginPath();

    for (let x = -150; x <= width + 150; x += 40) {
      const y = activeCenterY
              + Math.sin(x * 0.0015 + speed + i * 2.3) * 85
              + Math.cos(x * 0.0008 - speed) * 45
              - thickness;
      if (x === -150) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }

    for (let x = width + 150; x >= -150; x -= 40) {
      const y = activeCenterY
              + Math.sin(x * 0.0015 + speed + i * 2.3) * 85
              + Math.cos(x * 0.0008 - speed) * 45
              + thickness;
      ctx.lineTo(x, y);
    }
    ctx.closePath();

    const hue = currentPalette.hueBase + Math.sin(time * 1.0 + i) * currentPalette.hueRange;

    ctx.fillStyle = `hsla(${hue}, 100%, 65%, 0.18)`;
    ctx.fill();
  }

  ctx.filter = 'none';
  time += 0.0025;
  // Don't schedule the next frame while paused — see `paused` + the
  // visibility/focus handlers below. The canvas RAF is the app's dominant
  // idle CPU/GPU cost; pausing it is what makes Sleep "barely noticeable"
  // AND what stops the lag when the window is covered/minimized.
  if (!paused) requestAnimationFrame(animate);
}

// Render loop control. `paused` is set by THREE independent signals so the
// expensive RAF loop stops the moment the canvas isn't visible to the user:
//   1. `canvas-pause` event from Rust (system_menu power_sleep).
//   2. `document.visibilitychange` → hidden (alt-tab, minimize, another app
//      fully covering the window). The standard browser RAF throttle isn't
//      enough — WebView2 still fires RAF in some hidden states, and even a
//      throttled RAF re-runs the full animate() body.
//   3. `window.blur` (focus lost to another app) as a belt-and-suspenders
//      fallback when visibilitychange doesn't fire (e.g. another window
//      dragged over this one without minimizing).
// Resume mirrors all three. The animate() loop self-gates on `paused`.
let paused = false;

function startLoop() {
  if (paused) { paused = false; requestAnimationFrame(animate); }
}

// Tauri emits these from system_menu power_sleep / power_wake. Guard with
// .catch so a dev preview outside Tauri doesn't throw on the listener.
listen('canvas-pause', () => { paused = true; }).catch(() => {});
listen('canvas-resume', () => { startLoop(); }).catch(() => {});

// Pause when the page is hidden (alt-tab / minimize / tab switch). This is
// THE fix for "lag when another app covers the window" — without it the RAF
// keeps running the full animate() body at full speed even when nothing's
// visible. Resume on visible.
document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    paused = true;
  } else {
    startLoop();
  }
});

// Pause when the window loses focus (another app comes to the foreground).
// Belt-and-suspenders: visibilitychange covers most cases, but blur fires
// for "another window dragged over this one" where the page isn't technically
// hidden. Resume only if also visible + not manually paused via power_sleep.
window.addEventListener('blur', () => { paused = true; });
window.addEventListener('focus', () => {
  if (!document.hidden) startLoop();
});

animate();

// NOTE: this file is loaded as type="module", which defers execution until
// after the DOM is parsed — so DOMContentLoaded has ALREADY fired by the time
// we run. Do NOT wrap the wiring in a DOMContentLoaded listener (it would
// never execute). The elements below all exist at module-eval time.
const pawBtn = document.getElementById('pawBtn');
const dropdownMenu = document.getElementById('dropdownMenu');
  const clockBtn = document.getElementById('clockBtn');
  const clockDropdownMenu = document.getElementById('clockDropdownMenu');
  const digitalTimeEl = document.getElementById('digitalTime');
  const calendarBtn = document.getElementById('calendarBtn');
  const calendarDropdownMenu = document.getElementById('calendarDropdownMenu');
  const dateDisplayEl = document.getElementById('dateDisplay');
  const gridContainer = document.getElementById('calendarGrid');
  
  // New UI Elements
  const wifiBtn = document.getElementById('wifiBtn');
  const wifiDropdownMenu = document.getElementById('wifiDropdownMenu');
  const bluetoothBtn = document.getElementById('bluetoothBtn');
  const bluetoothDropdownMenu = document.getElementById('bluetoothDropdownMenu');
  const audioBtn = document.getElementById('audioBtn');
  const audioDropdownMenu = document.getElementById('audioDropdownMenu');
  
  const hourHand = document.querySelector('.hour-hand');
  const minuteHand = document.querySelector('.minute-hand');

  function toggleDropdown(menu, event) {
    event.stopPropagation();
    const isOpen = menu.classList.contains('show');
    
    // Clear all open menus
    dropdownMenu.classList.remove('show');
    clockDropdownMenu.classList.remove('show');
    calendarDropdownMenu.classList.remove('show');
    wifiDropdownMenu.classList.remove('show');
    bluetoothDropdownMenu.classList.remove('show');
    audioDropdownMenu.classList.remove('show');
    
    if (!isOpen) {
      menu.classList.add('show');
    }
  }

  pawBtn.addEventListener('click', (e) => toggleDropdown(dropdownMenu, e));
  clockBtn.addEventListener('click', (e) => toggleDropdown(clockDropdownMenu, e));
  calendarBtn.addEventListener('click', (e) => toggleDropdown(calendarDropdownMenu, e));
  wifiBtn.addEventListener('click', (e) => toggleDropdown(wifiDropdownMenu, e));
  bluetoothBtn.addEventListener('click', (e) => toggleDropdown(bluetoothDropdownMenu, e));
  audioBtn.addEventListener('click', (e) => toggleDropdown(audioDropdownMenu, e));

  // ── Paw menu: power actions (Shutdown / Restart / Sleep) ────────────────
  // The three power commands exposed by system_menu.rs. Each closes the
  // dropdown first so it doesn't flash on the next launch.
  const closePawMenu = () => dropdownMenu.classList.remove('show');

  document.getElementById('shutdownBtn')?.addEventListener('click', () => {
    closePawMenu();
    invoke('power_shutdown_cmd');
  });
  document.getElementById('restartBtn')?.addEventListener('click', () => {
    closePawMenu();
    invoke('power_restart_cmd');
  });
  document.getElementById('sleepBtn')?.addEventListener('click', () => {
    closePawMenu();
    invoke('power_sleep_cmd');
  });

  // ── Theme cascade (paw → theme → color code) ────────────────────────────
  // Three aligned panels. Clicking Theme opens panel 2; clicking a theme opens
  // panel 3 (color codes); clicking a color code persists + applies live. The
  // document-click dismiss handler (below) closes all three on outside click.
  const themePanel = document.getElementById('themePanel');
  const colorCodePanel = document.getElementById('colorCodePanel');

  // Apply a theme + color code to the running canvas. Unknown color codes
  // silently fall back to Vibrant so a stale theme.json can't break the loop.
  function applyTheme(theme, colorCode) {
    currentPalette = COLOR_CODES[colorCode] || COLOR_CODES.Vibrant;
    // Mark the selected option in each panel (the `.selected` highlight).
    document.querySelectorAll('.theme-option').forEach((el) => {
      el.classList.toggle('selected', el.dataset.theme === theme);
    });
    document.querySelectorAll('.colorcode-option').forEach((el) => {
      el.classList.toggle('selected', el.dataset.colorcode === colorCode);
    });
  }

  // Load the persisted theme on boot and paint the cascade selection state.
  invoke('theme_get')
    .then((t) => { if (t) applyTheme(t.theme, t.colorCode); })
    .catch((e) => console.warn('[Wupi] theme_get failed', e));

  document.getElementById('themeBtn')?.addEventListener('click', (e) => {
    e.stopPropagation();
    // Toggle the theme panel; keep the paw menu open so the cascade reads as
    // an extension of it.
    const open = themePanel.classList.toggle('show');
    if (!open) colorCodePanel.classList.remove('show');
  });

  document.querySelectorAll('.theme-option').forEach((el) => {
    el.addEventListener('click', (e) => {
      e.stopPropagation();
      // Selecting a theme opens the color-code panel (cascade level 3).
      applyTheme(el.dataset.theme,
        document.querySelector('.colorcode-option.selected')?.dataset.colorcode || 'Vibrant');
      colorCodePanel.classList.add('show');
    });
  });

  document.querySelectorAll('.colorcode-option').forEach((el) => {
    el.addEventListener('click', (e) => {
      e.stopPropagation();
      const themeName = document.querySelector('.theme-option.selected')?.dataset.theme || 'Aurora';
      const cc = el.dataset.colorcode;
      applyTheme(themeName, cc);
      invoke('theme_set', { themeName, colorCode: cc }).catch((err) =>
        console.warn('[Wupi] theme_set failed', err)
      );
    });
  });

  document.addEventListener('click', () => {
    dropdownMenu.classList.remove('show');
    clockDropdownMenu.classList.remove('show');
    calendarDropdownMenu.classList.remove('show');
    wifiDropdownMenu.classList.remove('show');
    bluetoothDropdownMenu.classList.remove('show');
    audioDropdownMenu.classList.remove('show');
    themePanel?.classList.remove('show');
    colorCodePanel?.classList.remove('show');
  });

  const wifiToggle = document.querySelector('.wifi-toggle-row');
  const wifiIcon = wifiBtn.querySelector('.status-icon');

  // ── Wi-Fi dropdown: real current network + scan list ────────────────────
  function refreshWifi() {
    // Current connection.
    invoke('wifi_get_current')
      .then((s) => {
        const dot = wifiDropdownMenu.querySelector('.wifi-toggle-row .status-dot');
        const toggleText = wifiToggle.querySelector('.toggle-text');
        if (s && s.connected) {
          dot?.classList.add('connected');
          wifiIcon.classList.remove('disabled');
          toggleText.textContent = `Connected: ${s.ssid || '(unnamed)'}`;
        } else {
          dot?.classList.remove('connected');
          toggleText.textContent = 'Turn Wi-Fi On';
        }
      })
      .catch((e) => console.warn('[Wupi] wifi_get_current failed', e));

    // Network list (deduped backend-side by SSID now). Rebuild only if absent
    // to avoid flicker; the toggle row above updates independently.
    const existingList = wifiDropdownMenu.querySelector('.scan-list');
    if (existingList) existingList.remove();
    invoke('wifi_scan')
      .then((nets) => {
        if (!nets || !nets.length) return;
        const list = document.createElement('div');
        list.className = 'scan-list';
        const header = document.createElement('div');
        header.className = 'dropdown-status-title';
        header.textContent = 'Available';
        list.appendChild(header);
        for (const n of nets) {
          const btn = document.createElement('button');
          btn.className = 'dropdown-item wifi-network';
          const lock = n.secure ? '🔒 ' : '';
          // No signal % — it was noisy and the same network appeared multiple
          // times at different strengths. SSID-only now (backend dedups).
          btn.innerHTML = `<span class="status-dot"></span>${lock}${n.ssid}`;
          btn.addEventListener('click', (ev) => {
            ev.stopPropagation();
            invoke('wifi_connect', { ssid: n.ssid, password: n.secure ? prompt(`Password for ${n.ssid}:`) || null : null })
              .then(() => refreshWifi())
              .catch((err) => console.error('[Wupi] wifi_connect failed', err));
          });
          list.appendChild(btn);
        }
        wifiDropdownMenu.appendChild(list);
      })
      .catch((e) => console.warn('[Wupi] wifi_scan failed', e));
  }

  // The Wi-Fi toggle row: disconnects when connected, connects (toggles radio)
  // when off. Windows exposes Wi-Fi radio via the WinRT Radio API (same as
  // Bluetooth), so we route through wifi_toggle_radio.
  wifiToggle.addEventListener('click', (e) => {
    e.stopPropagation();
    const dot = wifiToggle.querySelector('.status-dot');
    const isOn = dot?.classList.contains('connected');
    invoke('wifi_toggle_radio', { on: !isOn })
      .then(() => refreshWifi())
      .catch((err) => console.error('[Wupi] wifi_toggle_radio failed', err));
  });

  wifiBtn.addEventListener('click', () => {
    setTimeout(() => {
      if (wifiDropdownMenu.classList.contains('show')) refreshWifi();
    }, 0);
  });

  const btToggle = document.querySelector('.bt-toggle-row');
  const btIcon = bluetoothBtn.querySelector('.status-icon');

  // ── Bluetooth dropdown: real radio state + device list ──────────────────
  function refreshBluetooth() {
    invoke('bluetooth_get_state')
      .then((s) => {
        const dot = bluetoothDropdownMenu.querySelector('.bt-toggle-row .status-dot');
        const toggleText = btToggle.querySelector('.toggle-text');
        if (s && s.radio_on) {
          dot?.classList.add('connected');
          btIcon.classList.remove('disabled');
          toggleText.textContent = 'Turn Bluetooth Off';
        } else {
          dot?.classList.remove('connected');
          btIcon.classList.add('disabled');
          toggleText.textContent = 'Turn Bluetooth On';
        }
      })
      .catch((e) => console.warn('[Wupi] bluetooth_get_state failed', e));

    const existingList = bluetoothDropdownMenu.querySelector('.bt-device-list');
    if (existingList) existingList.remove();
    invoke('bluetooth_list_devices')
      .then((devs) => {
        if (!devs || !devs.length) return;
        const list = document.createElement('div');
        list.className = 'bt-device-list';
        const header = document.createElement('div');
        header.className = 'dropdown-status-title devices-header';
        header.textContent = 'My Devices';
        list.appendChild(header);
        for (const d of devs) {
          const btn = document.createElement('button');
          btn.className = 'dropdown-item device-opt';
          const state = d.connected ? '🟢 ' : (d.paired ? '⚪ ' : '');
          btn.innerHTML = `<span class="status-dot ${d.paired ? 'connected' : ''}"></span>${state}${d.name}`;
          if (!d.paired) {
            btn.addEventListener('click', (ev) => {
              ev.stopPropagation();
              invoke('bluetooth_pair', { deviceId: d.id })
                .then((ok) => { if (ok) refreshBluetooth(); })
                .catch((err) => console.error('[Wupi] bluetooth_pair failed', err));
            });
          }
          list.appendChild(btn);
        }
        bluetoothDropdownMenu.appendChild(list);
      })
      .catch((e) => console.warn('[Wupi] bluetooth_list_devices failed', e));
  }

  // The toggle row now actually flips the radio.
  btToggle.addEventListener('click', (e) => {
    e.stopPropagation();
    const isOff = btIcon.classList.contains('disabled');
    invoke('bluetooth_toggle_radio', { on: isOff })
      .then(() => refreshBluetooth())
      .catch((err) => console.error('[Wupi] bluetooth_toggle_radio failed', err));
  });

  bluetoothBtn.addEventListener('click', () => {
    setTimeout(() => {
      if (bluetoothDropdownMenu.classList.contains('show')) refreshBluetooth();
    }, 0);
  });

  // "Add Device" — discover in-range unpaired BT devices and list them under
  // the button. Clicking one calls bluetooth_pair (Windows shows the native
  // PIN/confirmation UI for devices that need it).
  document.getElementById('btAddBtn')?.addEventListener('click', (e) => {
    e.stopPropagation();
    const existing = bluetoothDropdownMenu.querySelector('.bt-discover-list');
    if (existing) {
      existing.remove();
      return;
    }
    const list = document.createElement('div');
    list.className = 'bt-discover-list';
    const loading = document.createElement('div');
    loading.className = 'dropdown-status-title';
    loading.textContent = 'Searching…';
    list.appendChild(loading);
    bluetoothDropdownMenu.appendChild(list);
    invoke('bluetooth_discover')
      .then((devs) => {
        list.innerHTML = '';
        if (!devs || !devs.length) {
          const empty = document.createElement('div');
          empty.className = 'dropdown-status-title';
          empty.textContent = 'No devices found';
          list.appendChild(empty);
          return;
        }
        const header = document.createElement('div');
        header.className = 'dropdown-status-title';
        header.textContent = 'Available Devices';
        list.appendChild(header);
        for (const d of devs) {
          const btn = document.createElement('button');
          btn.className = 'dropdown-item';
          btn.innerHTML = `<span class="status-dot"></span>${d.name}`;
          btn.addEventListener('click', (ev) => {
            ev.stopPropagation();
            btn.textContent = 'Pairing…';
            invoke('bluetooth_pair', { deviceId: d.id })
              .then((ok) => {
                if (ok) refreshBluetooth();
                else btn.textContent = `${d.name} (failed)`;
              })
              .catch((err) => {
                console.error('[Wupi] bluetooth_pair failed', err);
                btn.textContent = `${d.name} (error)`;
              });
          });
          list.appendChild(btn);
        }
      })
      .catch((err) => {
        console.warn('[Wupi] bluetooth_discover failed', err);
        list.remove();
      });
  });

  const volumeSlider = document.getElementById('volumeSlider');
  const volumePercent = document.getElementById('volumePercent');
  const audioIcon = audioBtn.querySelector('.status-icon');

  // Set the audio icon based on a volume level (0 / low / high).
  function setAudioIcon(val) {
    if (val == 0) {
      audioIcon.innerHTML = `
        <svg class="status-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
            <line x1="23" y1="9" x2="17" y2="15"></line>
            <line x1="17" y1="9" x2="23" y2="15"></line>
        </svg>`;
    } else if (val < 50) {
      audioIcon.innerHTML = `
        <svg class="status-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
            <path d="M15.54 8.46a5 5 0 0 1 0 7.07"></path>
        </svg>`;
    } else {
      audioIcon.innerHTML = `
        <svg class="status-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
            <path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path>
        </svg>`;
    }
  }

  // ── Audio dropdown: live volume + output list ───────────────────────────
  // Debounced volume set so dragging the slider doesn't spam IPC calls.
  let volTimer = null;
  volumeSlider.addEventListener('input', (e) => {
    const val = Number(e.target.value);
    volumePercent.textContent = `${val}%`;
    setAudioIcon(val);
    clearTimeout(volTimer);
    volTimer = setTimeout(() => {
      invoke('audio_set_volume', { volume: val }).catch((err) =>
        console.error('[Wupi] audio_set_volume failed', err)
      );
    }, 60);
  });

  // ── Audio dropdown: live volume + output list ───────────────────────────
  // Split into two pieces to kill the flicker: the volume/mute is polled every
  // 1s (slider/percent/icon only — no DOM rebuild), and the output-device list
  // is built ONCE when the dropdown opens (it almost never changes mid-session).
  // The previous version rebuilt the whole list each tick → flicker.

  function refreshAudioVolume() {
    invoke('audio_get_state')
      .then((s) => {
        if (!s) return;
        // Only touch the slider/percent/icon — never rebuild the device list.
        volumeSlider.value = s.volume;
        volumePercent.textContent = `${s.volume}%`;
        setAudioIcon(s.muted ? 0 : s.volume);
      })
      .catch((e) => console.warn('[Wupi] audio_get_state failed', e));
  }

  function buildAudioOutputs() {
    const existingList = audioDropdownMenu.querySelector('.output-list');
    if (existingList) existingList.remove();
    invoke('audio_list_outputs')
      .then((outs) => {
        if (!outs || !outs.length) return;
        const list = document.createElement('div');
        list.className = 'output-list';
        const header = document.createElement('div');
        header.className = 'dropdown-status-title';
        header.textContent = 'Output';
        list.appendChild(header);
        for (const o of outs) {
          const btn = document.createElement('button');
          btn.className = 'dropdown-item output-option' + (o.is_default ? ' selected' : '');
          btn.innerHTML = `<span class="status-dot ${o.is_default ? 'connected' : ''}"></span>${o.name}`;
          if (!o.is_default) {
            btn.addEventListener('click', (ev) => {
              ev.stopPropagation();
              invoke('audio_set_default_output', { id: o.id })
                .then(() => buildAudioOutputs())
                .catch((err) => console.error('[Wupi] audio_set_default_output failed', err));
            });
          }
          list.appendChild(btn);
        }
        audioDropdownMenu.appendChild(list);
      })
      .catch((e) => console.warn('[Wupi] audio_list_outputs failed', e));
  }

  let audioPollTimer = null;
  audioBtn.addEventListener('click', () => {
    setTimeout(() => {
      if (audioDropdownMenu.classList.contains('show')) {
        // Opened: build the device list once + load volume, then poll volume only.
        buildAudioOutputs();
        refreshAudioVolume();
        clearInterval(audioPollTimer);
        audioPollTimer = setInterval(refreshAudioVolume, 1000);
      } else {
        clearInterval(audioPollTimer);
        audioPollTimer = null;
      }
    }, 0);
  });

  function updateClocks() {
    const now = new Date();
    const seconds = now.getSeconds();
    const minutes = now.getMinutes();
    const hours = now.getHours();

    const minuteDegrees = ((minutes / 60) * 360) + ((seconds / 60) * 6);
    const hourDegrees = ((hours / 12) * 360) + ((minutes / 60) * 30);

    hourHand.style.transform = `translate(-50%) rotate(${hourDegrees}deg)`;
    minuteHand.style.transform = `translate(-50%) rotate(${minuteDegrees}deg)`;

    let displayHours = hours;
    const displayMinutes = String(minutes).padStart(2, '0');
    const displaySeconds = String(seconds).padStart(2, '0');
    const ampm = displayHours >= 12 ? 'PM' : 'AM';
    
    displayHours = displayHours % 12;
    displayHours = displayHours ? displayHours : 12; 
    const formattedHours = String(displayHours).padStart(2, '0');

    digitalTimeEl.textContent = `${formattedHours}:${displayMinutes}:${displaySeconds} ${ampm}`;

    const options = { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' };
    dateDisplayEl.textContent = now.toLocaleDateString('en-US', options);

    const dayOfWeek = now.getDay();
    const week = Math.floor((now.getDate() - 1) / 7);
    const activeIndex = (dayOfWeek + week * 7) % 28;

    gridContainer.innerHTML = '';
    for (let r = 0; r < 4; r++) {
      for (let c = 0; c < 7; c++) {
        const index = r * 7 + c;
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('x', 17 + c * 10);
        rect.setAttribute('y', 40 + r * 12);
        rect.setAttribute('width', 6);
        rect.setAttribute('height', 6);
        rect.setAttribute('rx', 1);
        
        if (index === activeIndex) {
          rect.setAttribute('fill', '#b534fa');
          rect.style.filter = 'drop-shadow(0 0 3px #ff66b2)';
        } else {
          rect.setAttribute('fill', '#ffffff');
        }
        
        gridContainer.appendChild(rect);
      }
    }
  }

  updateClocks();
  setInterval(updateClocks, 1000);

  // ════════════════════════════════════════════════════════════════════════
  // APP WINDOW MANAGER
  // ════════════════════════════════════════════════════════════════════════
  // The surfaces (Chat, Profile Editor, Codex, Docks) are DOM overlays in
  // the ONE Tauri window. Background rules (per Chloe's spec):
  //   - WUPI Chat (chat): the ONLY window that pauses the canvas (stars +
  //     aurora OFF). Its own background is ~80% opaque so the paused backdrop
  //     doesn't show through. Closing it resumes the canvas.
  //   - Everything else (Codex, Profile, Docks home): canvas keeps running —
  //     stars/aurora animate behind the translucent glass.
  //
  // The previous version painted a frozen gradient into the framebuffer while
  // paused, which caused the compositor to tear/glitch and froze the loop on
  // close. The fix: NEVER manually paint the canvas here. Only flip the
  // `paused` flag; the RAF loop (animate) already handles start/stop cleanly
  // via its `if (!paused) requestAnimationFrame(animate)` guard, and when
  // un-paused it repaints fresh on the next frame. No half-painted frames.

  const openWindows = new Set();
  let zCounter = 1000;
  // No window pauses the canvas anymore — the background stays active behind
  // every surface (Chat is now translucent enough that stars show through).
  // Kept as a hook in case a future surface wants to freeze the background.
  function syncCanvasForWindows() {
    /* no-op: background always active */
  }

  function openWindow(id) {
    const el = document.getElementById(id);
    if (!el) return;
    if (openWindows.has(id)) {
      // Already open — just raise it to the top.
      el.style.zIndex = ++zCounter;
      return;
    }
    openWindows.add(id);
    el.style.zIndex = ++zCounter;
    el.classList.add('show');
    el.setAttribute('aria-hidden', 'false');
    syncCanvasForWindows();
    // Fire an onOpen hook if the surface registered one (e.g. Profile loads
    // its fields, Codex loads its list, Chat may show intro).
    const hook = windowOpenHooks.get(id);
    if (hook) hook();
  }

  function closeWindow(id) {
    const el = document.getElementById(id);
    if (!el) return;
    if (!openWindows.has(id)) return;
    openWindows.delete(id);
    el.classList.remove('show');
    el.setAttribute('aria-hidden', 'true');
    syncCanvasForWindows();
  }

  // Surfaces register an async onOpen hook (load data when first shown).
  const windowOpenHooks = new Map();

  // ✕ close buttons (data-close="winId").
  document.querySelectorAll('.app-window-close[data-close]').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      closeWindow(btn.dataset.close);
    });
  });

  // Esc closes the topmost open window.
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape' || openWindows.size === 0) return;
    // Close the highest-z open window (last added to the set isn't strictly
    // topmost, but in practice users Esc the one they just opened). Find by
    // max z-index for correctness.
    let topId = null;
    let topZ = -1;
    for (const id of openWindows) {
      const el = document.getElementById(id);
      const z = parseInt(el?.style.zIndex || '0', 10);
      if (z > topZ) { topZ = z; topId = id; }
    }
    if (topId) closeWindow(topId);
  });

  // Clicks inside a window must NOT bubble to the document-level handler that
  // closes the top-bar dropdowns (that handler also doesn't close windows, but
  // stopping propagation keeps the dropdown logic from running needlessly and
  // prevents a window-open dock click from immediately re-closing dropdowns).
  document.querySelectorAll('.app-window').forEach((win) => {
    win.addEventListener('click', (e) => e.stopPropagation());
  });

  // ── Draggable windows (Profile, Codex) ───────────────────────────────────
  // Header is the drag handle. The window is absolutely positioned; dragging
  // updates `left`/`top`. Only windows with `.draggable` get this — Chat is
  // fixed (immovable per spec), Docks-home is full-screen (no drag).
  function makeDraggable(winEl) {
    const handle = winEl.querySelector('.app-window-header');
    if (!handle) return;
    handle.style.cursor = 'grab';
    let dragging = false;
    let startX = 0, startY = 0, startLeft = 0, startTop = 0;

    handle.addEventListener('mousedown', (e) => {
      // Don't drag when clicking the close button or interactive header el.
      if (e.target.closest('.app-window-close')) return;
      dragging = true;
      handle.style.cursor = 'grabbing';
      // Switch from transform-center to absolute left/top so we can move it.
      const rect = winEl.getBoundingClientRect();
      winEl.style.left = rect.left + 'px';
      winEl.style.top = rect.top + 'px';
      winEl.style.transform = 'none';
      winEl.classList.add('dragged'); // CSS: drop the centering transform
      startX = e.clientX;
      startY = e.clientY;
      startLeft = rect.left;
      startTop = rect.top;
      e.preventDefault();
    });
    window.addEventListener('mousemove', (e) => {
      if (!dragging) return;
      const dx = e.clientX - startX;
      const dy = e.clientY - startY;
      // Keep the title bar on-screen (don't let it vanish off an edge).
      const maxX = window.innerWidth - 80;
      const maxY = window.innerHeight - 48;
      const nl = Math.min(Math.max(startLeft + dx, 0), maxX);
      const nt = Math.min(Math.max(startTop + dy, 0), maxY);
      winEl.style.left = nl + 'px';
      winEl.style.top = nt + 'px';
    });
    window.addEventListener('mouseup', () => {
      if (!dragging) return;
      dragging = false;
      handle.style.cursor = 'grab';
    });
  }
  document.querySelectorAll('.app-window.draggable').forEach(makeDraggable);

  // ── Dock wiring ──────────────────────────────────────────────────────────
  // Click an open app's dock item again → closes it (toggle behavior). The
  // quick-access dock order is fixed: Chat → Profile → Codex (NOT alphabetical
  // — that's the Docks home grid). Apps (Docks launcher) is special: it closes
  // any open surface windows then shows the home grid.
  function dockToggle(id) {
    if (openWindows.has(id)) closeWindow(id);
    else openWindow(id);
  }

  document.getElementById('dockChat')?.addEventListener('click', (e) => {
    e.stopPropagation();
    dockToggle('chat');
  });
  document.getElementById('dockProfile')?.addEventListener('click', (e) => {
    e.stopPropagation();
    dockToggle('profile');
  });
  document.getElementById('dockCodex')?.addEventListener('click', (e) => {
    e.stopPropagation();
    dockToggle('codex');
  });
  document.getElementById('dockApps')?.addEventListener('click', (e) => {
    e.stopPropagation();
    // Docks = "home": close any open surface windows and show the launcher
    // grid. (apps itself is the full-screen home overlay.) Not a toggle —
    // clicking Docks while home is open is a no-op (it's already home).
    if (openWindows.has('apps')) return;
    closeWindow('chat');
    closeWindow('profile');
    closeWindow('codex');
    openWindow('apps');
  });

  // Home-grid launcher icons (inside apps): open the matching surface.
  document.querySelectorAll('.home-app[data-open]').forEach((icon) => {
    icon.addEventListener('click', (e) => {
      e.stopPropagation();
      const target = icon.dataset.open;
      closeWindow('apps'); // leave home, open the app
      openWindow(target);
    });
  });

  // ════════════════════════════════════════════════════════════════════════
  // PROFILE EDITOR
  // ════════════════════════════════════════════════════════════════════════
  (function profileEditor() {
    const nameEl = document.getElementById('profName');
    const descEl = document.getElementById('profDescription');
    const saveBtn = document.getElementById('profSaveBtn');
    const statusEl = document.getElementById('profStatus');
    if (!nameEl) return;

    function setStatus(msg, kind) {
      statusEl.textContent = msg || '';
      statusEl.className = 'profile-status' + (kind ? ' ' + kind : '');
    }

    // Load fresh every time the window opens — cheap, and guarantees the editor
    // reflects disk state (someone could have hand-edited Operator.xml).
    windowOpenHooks.set('profile', () => {
      setStatus('Loading…');
      invoke('operator_profile_get')
        .then((profile) => {
          if (profile) {
            nameEl.value = profile.name || '';
            descEl.value = profile.description || '';
          } else {
            nameEl.value = ''; descEl.value = '';
          }
          setStatus('');
        })
        .catch((err) => setStatus('Load failed: ' + err, 'err'));
    });

    saveBtn?.addEventListener('click', () => {
      saveBtn.disabled = true;
      setStatus('Saving…');
      invoke('operator_profile_set', {
        name: nameEl.value,
        description: descEl.value,
      })
        .then(() => setStatus('Saved — applies next message', 'ok'))
        .catch((err) => setStatus('Save failed: ' + err, 'err'))
        .finally(() => { saveBtn.disabled = false; });
    });
  })();

  // ════════════════════════════════════════════════════════════════════════
  // THE CODEX — authored lore library (NOT a memory browser)
  // ════════════════════════════════════════════════════════════════════════
  // Codex is a library of authored reference "books" — world lore, TV-show
  // facts, worldbuilding. Source of truth = .md files in codex/ (re-seeded to
  // the retrieval index at boot + after each edit). It has NOTHING to do with
  // chat history or Wupi's persona — just the lore you author.
  //
  // UI: two panes. Left = searchable list of entries (title + tags). Right =
  // reader for the selected entry, with an Edit mode and a New-entry mode.
  (function codex() {
    const listEl = document.getElementById('codexList');
    const statusEl = document.getElementById('codexStatus');
    const searchEl = document.getElementById('codexSearch');
    const addBtn = document.getElementById('codexAddBtn');
    const readerEl = document.getElementById('codexReader');
    if (!listEl || !readerEl) return;

    let allFiles = []; // cached for client-side search filter

    function setStatus(msg, kind) {
      statusEl.textContent = msg || '';
      statusEl.className = 'codex-status' + (kind ? ' ' + kind : '');
    }
    function escapeHtml(s) {
      return String(s).replace(/[&<>"']/g, (c) => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
      }[c]));
    }

    // ── List rendering ───────────────────────────────────────────────────
    function renderList(filter) {
      const q = (filter || '').trim().toLowerCase();
      const files = q
        ? allFiles.filter((f) =>
            f.title.toLowerCase().includes(q) ||
            f.body.toLowerCase().includes(q) ||
            f.tags.some((t) => t.toLowerCase().includes(q)))
        : allFiles;
      if (!files.length) {
        listEl.innerHTML = `<div class="codex-empty">${q ? 'No matches.' : 'No lore yet. Click “+ New” to add your first entry.'}</div>`;
        return;
      }
      listEl.innerHTML = files.map((f) => `
        <div class="codex-row" data-filename="${escapeHtml(f.filename)}">
          <div class="codex-row-title">${escapeHtml(f.title || f.filename)}</div>
          <div class="codex-row-tags">${f.tags.map((t) => `<span class="codex-tag">${escapeHtml(t)}</span>`).join('')}</div>
        </div>`).join('');
    }

    function loadAll() {
      setStatus('Loading…');
      return invoke('codex_list')
        .then((files) => {
          allFiles = Array.isArray(files) ? files : [];
          renderList(searchEl.value);
          setStatus(allFiles.length ? `${allFiles.length} entr${allFiles.length === 1 ? 'y' : 'ies'}` : '');
        })
        .catch((err) => {
          listEl.innerHTML = `<div class="codex-empty">Failed to load.</div>`;
          setStatus('Load failed: ' + err, 'err');
        });
    }

    // ── Reader pane ──────────────────────────────────────────────────────
    function showReader(file) {
      readerEl.innerHTML = `
        <div class="codex-reader-head">
          <div>
            <div class="codex-reader-title">${escapeHtml(file.title || file.filename)}</div>
            <div class="codex-reader-tags">${file.tags.map((t) => `<span class="codex-tag">${escapeHtml(t)}</span>`).join('')}</div>
          </div>
          <div class="codex-reader-actions">
            <button class="codex-mini-btn" id="crEdit">Edit</button>
            <button class="codex-mini-btn del" id="crDelete">Delete</button>
          </div>
        </div>
        <div class="codex-reader-body">${escapeHtml(file.body)}</div>`;
      document.getElementById('crEdit').addEventListener('click', () => showEditor(file));
      document.getElementById('crDelete').addEventListener('click', () => deleteEntry(file.filename, file.title || file.filename));
    }

    function showEmptyReader(msg) {
      readerEl.innerHTML = `<div class="codex-reader-empty">${escapeHtml(msg || 'Select an entry to read, or add new lore.')}</div>`;
    }

    // ── Editor pane (edit existing or create new) ───────────────────────
    function showEditor(file) {
      const isNew = !file;
      readerEl.innerHTML = `
        <div class="codex-editor">
          <div class="codex-editor-row">
            <label class="field-label">Title</label>
            <input type="text" id="ceTitle" class="field-input" value="${escapeHtml(file?.title || '')}" placeholder="e.g. Neo-Kyoto" />
          </div>
          <div class="codex-editor-row">
            <label class="field-label">Tags (comma-separated)</label>
            <input type="text" id="ceTags" class="field-input" value="${escapeHtml((file?.tags || []).join(', '))}" placeholder="lore, location, setting" />
          </div>
          <div class="codex-editor-row">
            <label class="field-label">Body</label>
            <textarea id="ceBody" class="field-textarea codex-editor-body" placeholder="The factual lore…">${escapeHtml(file?.body || '')}</textarea>
          </div>
          <div class="codex-editor-actions">
            <button class="field-btn" id="ceCancel">Cancel</button>
            <button class="field-btn primary" id="ceSave">${isNew ? 'Create' : 'Save'}</button>
          </div>
        </div>`;
      const originalFilename = file?.filename || '';
      document.getElementById('ceCancel').addEventListener('click', () => {
        if (file) showReader(file); else showEmptyReader();
      });
      document.getElementById('ceSave').addEventListener('click', () => {
        const title = document.getElementById('ceTitle').value.trim();
        const tags = document.getElementById('ceTags').value.split(',').map((t) => t.trim()).filter(Boolean);
        const body = document.getElementById('ceBody').value;
        if (!title) { document.getElementById('ceTitle').focus(); return; }
        // Filename derives from the title for new entries; stays stable for edits.
        const filename = isNew ? title : originalFilename;
        document.getElementById('ceSave').disabled = true;
        setStatus('Saving…');
        invoke('codex_save', { filename, title, tags, body })
          .then((savedName) => {
            setStatus(isNew ? 'Created.' : 'Saved.', 'ok');
            // Re-list then open the saved entry in the reader.
            return loadAll().then(() => {
              const updated = allFiles.find((f) => f.filename === savedName);
              if (updated) showReader(updated); else showEmptyReader();
            });
          })
          .catch((err) => { setStatus('Save failed: ' + err, 'err'); document.getElementById('ceSave').disabled = false; });
      });
    }

    function deleteEntry(filename, label) {
      if (!confirm(`Delete "${label}"? This removes the lore file. This cannot be undone.`)) return;
      setStatus('Deleting…');
      invoke('codex_delete', { filename })
        .then(() => { setStatus('Deleted.', 'ok'); showEmptyReader(); loadAll(); })
        .catch((err) => setStatus('Delete failed: ' + err, 'err'));
    }

    // ── Wiring ───────────────────────────────────────────────────────────
    // Clicking a list row opens it in the reader.
    listEl.addEventListener('click', (e) => {
      const row = e.target.closest('.codex-row[data-filename]');
      if (!row) return;
      const filename = row.dataset.filename;
      const file = allFiles.find((f) => f.filename === filename);
      if (file) showReader(file);
    });

    // Search filters the list client-side (the corpus is small).
    searchEl?.addEventListener('input', () => renderList(searchEl.value));

    // + New opens a blank editor.
    addBtn?.addEventListener('click', () => showEditor(null));

    windowOpenHooks.set('codex', () => { loadAll(); showEmptyReader(); });
  })();

  // ════════════════════════════════════════════════════════════════════════
  // WUPI CHAT — full streaming chat surface
  // ════════════════════════════════════════════════════════════════════════
  (function wupiChat() {
    const msgsEl = document.getElementById('chatMessages');
    const inputEl = document.getElementById('chatInput');
    const sendBtn = document.getElementById('chatSendBtn');
    const stopBtn = document.getElementById('chatStopBtn');
    if (!msgsEl) return;

    // Tauri v2 Channel for streaming — imported statically at the top of the
    // module, so it's always available (no race with a dynamic import).
    let generating = false;
    let emptyShown = true;

    function showEmpty() {
      if (!emptyShown) return;
      msgsEl.innerHTML = `<div class="chat-empty">Say hello to Wupi.</div>`;
    }
    function clearEmpty() {
      if (!emptyShown) return;
      emptyShown = false;
      msgsEl.innerHTML = '';
    }

    function scrollBottom() {
      msgsEl.scrollTop = msgsEl.scrollHeight;
    }

    function addUserBubble(text) {
      clearEmpty();
      const div = document.createElement('div');
      div.className = 'msg user';
      div.textContent = text;
      msgsEl.appendChild(div);
      scrollBottom();
    }

    function addErrorBubble(msg) {
      const div = document.createElement('div');
      div.className = 'msg-error';
      div.textContent = msg;
      msgsEl.appendChild(div);
      scrollBottom();
    }

    // A static (non-streaming) Wupi message — used for the randomized intro
    // shown when Chat first opens. Mirrors the finalized bubble shape.
    function addWupiBubble(text) {
      clearEmpty();
      const div = document.createElement('div');
      div.className = 'msg wupi';
      div.textContent = text;
      msgsEl.appendChild(div);
      scrollBottom();
    }

    // Returns the wupi bubble element + a text setter.
    function startWupiBubble() {
      clearEmpty();
      const div = document.createElement('div');
      div.className = 'msg wupi streaming';
      msgsEl.appendChild(div);
      scrollBottom();
      return div;
    }

    function finalizeWupiBubble(div, finalText, reasoning) {
      div.classList.remove('streaming');
      div.textContent = finalText || '(no response)';
      if (reasoning && reasoning.trim()) {
        const det = document.createElement('details');
        det.className = 'msg-reasoning';
        const sum = document.createElement('summary');
        sum.textContent = 'Reasoning';
        const body = document.createElement('div');
        body.className = 'msg-reasoning-body';
        body.textContent = reasoning;
        det.appendChild(sum);
        det.appendChild(body);
        div.appendChild(det);
      }
      scrollBottom();
    }

    function setGenerating(on) {
      generating = on;
      inputEl.disabled = on;
      sendBtn.disabled = on;
      stopBtn.disabled = !on;
    }

    async function send() {
      if (generating) return;
      const text = inputEl.value.trim();
      if (!text) return;

      inputEl.value = '';
      addUserBubble(text);

      const bubble = startWupiBubble();
      let streamed = '';
      setGenerating(true);

      const channel = new Channel();
      channel.onmessage = (e) => {
        if (!e) return;
        if (e.type === 'chunk') {
          streamed += e.text || '';
          bubble.textContent = streamed;
          scrollBottom();
        } else if (e.type === 'error') {
          setGenerating(false);
          // Replace the partial bubble with an error notice.
          bubble.remove();
          addErrorBubble(e.message || 'Generation failed.');
        } else if (e.type === 'done') {
          setGenerating(false);
          finalizeWupiBubble(bubble, e.final_text != null ? e.final_text : streamed, e.reasoning || '');
        }
      };

      invoke('chat_send', { text, onEvent: channel })
        .catch((err) => {
          if (generating) {
            setGenerating(false);
            bubble.remove();
            addErrorBubble('Failed to send: ' + err);
          }
        });
    }

    sendBtn?.addEventListener('click', send);
    stopBtn?.addEventListener('click', () => {
      invoke('chat_stop').catch((e) => console.warn('[Wupi] chat_stop failed', e));
    });

    // Enter sends, Shift+Enter for newline.
    inputEl?.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        send();
      }
    });

    // On each open: reset to a fresh conversation view + show Wupi's randomized
    // intro (one per open, from the SIM card's introductions list via the
    // get_intro IPC). The intro is UI-only — never sent to the model or archived.
    function loadIntro() {
      emptyShown = true;
      msgsEl.innerHTML = '';
      invoke('get_intro')
        .then((intro) => {
          if (intro) {
            addWupiBubble(intro);
          } else {
            showEmpty();
          }
        })
        .catch((e) => {
          console.warn('[Wupi] get_intro failed', e);
          showEmpty();
        });
    }
    windowOpenHooks.set('chat', loadIntro);
    loadIntro();
  })();