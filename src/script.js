// Tauri 2 IPC + event APIs. Imported as ES modules now that script.js is
// `type="module"` (Vite bundles these; withGlobalTauri is off so the
// `window.__TAURI__` global is NOT injected — the import is the source of truth).
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';

const canvas = document.getElementById('aurora-canvas');
const ctx = canvas.getContext('2d');

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
  
  skyGrad.addColorStop(0, '#02040a');
  skyGrad.addColorStop(0.35, '#060a17'); 
  skyGrad.addColorStop(0.65, '#150524'); 
  skyGrad.addColorStop(0.85, '#2b0b36'); 
  skyGrad.addColorStop(1, '#4a173d'); 

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

    const hue = 305 + Math.sin(time * 1.0 + i) * 45; 
    
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

  document.addEventListener('click', () => {
    dropdownMenu.classList.remove('show');
    clockDropdownMenu.classList.remove('show');
    calendarDropdownMenu.classList.remove('show');
    wifiDropdownMenu.classList.remove('show');
    bluetoothDropdownMenu.classList.remove('show');
    audioDropdownMenu.classList.remove('show');
  });

  const wifiToggle = document.querySelector('.wifi-toggle-row');
  const wifiIcon = wifiBtn.querySelector('.status-icon');
  wifiToggle.addEventListener('click', (e) => {
    e.stopPropagation();
    wifiIcon.classList.toggle('disabled');
    const isOff = wifiIcon.classList.contains('disabled');
    wifiToggle.querySelector('.toggle-text').textContent = isOff ? "Connect WUPI-NET" : "Disconnect WUPI-NET";
    const dot = wifiDropdownMenu.querySelector('.status-dot');
    const netOpt = wifiDropdownMenu.querySelector('.network-opt');
    if (isOff) {
      dot.classList.remove('connected');
      netOpt.style.opacity = '0.3';
    } else {
      dot.classList.add('connected');
      netOpt.style.opacity = '1';
    }
  });

  const btToggle = document.querySelector('.bt-toggle-row');
  const btIcon = bluetoothBtn.querySelector('.status-icon');
  btToggle.addEventListener('click', (e) => {
    e.stopPropagation();
    btIcon.classList.toggle('disabled');
    const isOff = btIcon.classList.contains('disabled');
    btToggle.querySelector('.toggle-text').textContent = isOff ? "Turn Bluetooth On" : "Turn Bluetooth Off";
    const dot = bluetoothDropdownMenu.querySelector('.status-dot');
    const devOpt = bluetoothDropdownMenu.querySelector('.device-opt');
    const devHeader = bluetoothDropdownMenu.querySelector('.devices-header');
    if (isOff) {
      dot.classList.remove('connected');
      devOpt.style.display = 'none';
      devHeader.style.display = 'none';
    } else {
      dot.classList.add('connected');
      devOpt.style.display = 'flex';
      devHeader.style.display = 'block';
    }
  });

  const volumeSlider = document.getElementById('volumeSlider');
  const volumePercent = document.getElementById('volumePercent');
  const audioIcon = audioBtn.querySelector('.status-icon');

  volumeSlider.addEventListener('input', (e) => {
    const val = e.target.value;
    volumePercent.textContent = `${val}%`;

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