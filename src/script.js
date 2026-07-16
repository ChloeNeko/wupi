// Tauri 2 IPC + event APIs. Imported as ES modules now that script.js is
// `type="module"` (Vite bundles these; withGlobalTauri is off so the
// `window.__TAURI__` global is NOT injected — the import is the source of truth).
import { invoke } from '@tauri-apps/api/core';
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
  
  const colorPalette = ['#ffffff', '#e8f0ff', '#fff4e6', '#ffe6ee'];
  const color = colorPalette[Math.floor(Math.random() * colorPalette.length)];

  return {
    x: Math.random() * width,
    y: Math.random() * height, 
    size: Math.random() * 0.9 + 0.4,
    alpha: Math.random() * 0.7 + 0.3, 
    isTwinkling: isTwinkling,
    speed: isTwinkling ? (0.0005 + Math.random() * 0.0012) : 0,
    drift: Math.random() * 0.01 + 0.008 + 0.004,
    color: color
  };
});

let time = 0;

function animate() {
  currentX += (mouseX - currentX) * 0.25;
  currentY += (mouseY - currentY) * 0.25;

  const skyGrad = ctx.createLinearGradient(0, 0, 0, height);

  const stops = currentPalette.skyGradient;
  // Evenly distribute the stops across [0, 1].
  for (let i = 0; i < stops.length; i++) {
    skyGrad.addColorStop(i / (stops.length - 1), stops[i]);
  }

  ctx.globalCompositeOperation = 'source-over';
  ctx.globalAlpha = 1.0;
  ctx.fillStyle = skyGrad;
  ctx.fillRect(0, 0, width, height);

  stars.forEach(s => {
    if (s.isTwinkling) {
      s.alpha += s.speed;
      if (s.alpha > 1 || s.alpha < 0.15) s.speed = -s.speed; 
    }
    
    s.y -= s.drift;
    if (s.y < 0) s.y = height; 

    ctx.fillStyle = s.color;
    ctx.globalAlpha = Math.abs(s.alpha);
    ctx.fillRect(
        s.x + (currentX * s.size * 16), 
        s.y + (currentY * s.size * 16), 
        s.size, 
        s.size
    ); 
  });

  ctx.globalAlpha = 1.0; 
  
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
  // Don't schedule the next frame while sleeping — the canvas RAF is the
  // app's dominant idle CPU/GPU cost, and pausing it is what makes Sleep
  // "barely noticeable." Wake (canvas-resume event) restarts the loop.
  if (!paused) requestAnimationFrame(animate);
}

// Render loop control: starts running, suspended on `canvas-pause`,
// resumed on `canvas-resume`. Both events come from the Rust side
// (system_menu power_sleep / power_wake).
let paused = false;

function startLoop() {
  if (paused) { paused = false; requestAnimationFrame(animate); }
}

// Tauri emits these from system_menu power_sleep / power_wake. Guard with
// .catch so a dev preview outside Tauri doesn't throw on the listener.
listen('canvas-pause', () => { paused = true; }).catch(() => {});
listen('canvas-resume', () => { startLoop(); }).catch(() => {});

animate();

document.addEventListener('DOMContentLoaded', () => {
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
  // Theme + Terminal items are wired in later phases; here we only hook the
  // three power commands exposed by system_menu.rs. Each closes the dropdown
  // first so it doesn't flash on the next launch.
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

  // ── Paw menu: Terminal ──────────────────────────────────────────────────
  // Opens (or focuses) the borderless glassmorphism terminal window. The
  // window's own terminal.js then spawns the PTY via terminal_init.
  document.querySelector('.terminal-item')?.addEventListener('click', () => {
    closePawMenu();
    invoke('terminal_create_window').catch((e) =>
      console.error('[Wupi] terminal_create_window failed', e)
    );
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
});