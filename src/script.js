import { invoke, Channel } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';

const canvas = document.getElementById('aurora-canvas');
const ctx = canvas.getContext('2d');

// Each color code defines the aurora's sky gradient (top→bottom CSS color
// stops) and the curtain hue generator (base hue ± range). The animate() loop
// reads `currentPalette`: switching color codes re-paints on the next frame.
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

// Aurora offscreen buffer. The 5 curtains are rendered here ONCE per frame
// with NO blur (cheap fills, no Gaussian pass), then the whole composite is
// blitted to the main canvas through a SINGLE blur(30px). This collapses the
// expensive op from 5x per frame → 1x, which is what fixed the boot-wipe
// stutter (at-rest was OK because the pipeline was warm; the wipe hit cold
// and 5 cold blur passes/frame stuttered). The buffer is DPR-scaled so the
// blur resolves at physical-pixel resolution (no softness on high-DPI).
// Lazily (re)allocated in resize() to match viewport + DPR.
let auroraBuf = null;
let auroraBufCtx = null;

function resize() {
  const dpr = window.devicePixelRatio || 1;
  width = window.innerWidth;
  height = window.innerHeight;
  canvas.width = Math.floor(width * dpr);
  canvas.height = Math.floor(height * dpr);
  canvas.style.width = width + 'px';
  canvas.style.height = height + 'px';
  // Reset transform then re-apply: resize() can fire repeatedly, and the
  // scale accumulates if not reset first.
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.scale(dpr, dpr);
  // (Re)allocate the aurora offscreen buffer at physical-pixel resolution.
  // On first call (boot), this is what makes the boot wipe cheap.
  if (!auroraBuf) {
    auroraBuf = document.createElement('canvas');
    auroraBufCtx = auroraBuf.getContext('2d');
  }
  auroraBuf.width = canvas.width;
  auroraBuf.height = canvas.height;
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
  // colorIdx indexes STAR_COLORS: drawing buckets stars by color so the
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

// The gradient depends only on the palette + canvas height, both of which
// change rarely (theme switch / resize). Recreating it 60×/sec was pure waste
//: createLinearGradient + 5 addColorStop calls per frame. Rebuilt only when
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

// Batching same-color stars into one fillStyle set + grouping alpha into a
// few bands collapses ~1000 state changes/frame into a handful. The visual
// difference is imperceptible (alpha quantized to 8 bands of 0.1).
const STAR_COLORS = ['#ffffff', '#e8f0ff', '#fff4e6', '#ffe6ee'];

// Boot reveal: aurora curtains reveal LEFT-TO-RIGHT (a "wipe" rather than a
// global opacity ramp). The wipe is BOTH an aesthetic choice AND a perf fix.
//
// Three gates work together:
//   auroraIntensity   — the overall fade-in (0 → 1 over AURORA_RAMP_MS).
//   auroraRevealX     — the left-to-right wipe position (px).
//   auroraBufFrozen   — when true, the offscreen buffer holds a snapshot
//                       and isn't redrawn (the wipe only changes the blit
//                       slice, not the curtain shapes). At rest this is
//                       false so the aurora animates live.
//
// The boot-wipe stutter fix (3 layered wins):
// 1. Offscreen buffer: 5 curtains rendered with NO blur, then ONE blurred
//    blit to main → 5x fewer Gaussian passes per frame.
// 2. Dirty flag: during the wipe the buffer is frozen (curtain shapes barely
//    change in 0.5s, and any frozen-frame artifacts are masked by the blur
//    + the wipe's own motion). Zero buffer redraw cost during the wipe.
// 3. Interpolated blur radius (10px → 30px with intensity): Gaussian cost
//    scales roughly with radius², so blur(10px) at wipe-start is ~9x cheaper
//    than blur(30px). The visual blooms as it reveals.
// All three fire only during the wipe. At rest the buffer redraws live with
// the full 30px blur, identical to the locked aesthetic.
let auroraIntensity = 0;
let auroraRampStart = 0;
const AURORA_RAMP_MS = 600;
// The wipe runs concurrently with the intensity ramp. Shorter than RAMP_MS
// so the wipe front finishes ahead of the full-opacity settle.
let auroraRevealX = 0;          // current wipe x (px, CSS px)
let auroraRevealStart = 0;      // 0 = not yet armed
const AURORA_WIPE_MS = 500;
// Buffer dirty flag: false = redraw curtains every frame (live at rest);
// true = hold the snapshot (during the wipe, set by revealAfterLand).
// auroraBufDirty: when frozen, render exactly once at full intensity, then
// hold. Cleared on the first frozen render, re-armed when the freeze engages.
let auroraBufFrozen = false;
let auroraBufDirty = false;
// Blur radius floor/ceiling (CSS px). Gaussian cost ~ radius².
const AURORA_BLUR_FLOOR = 10;
const AURORA_BLUR_CEIL = 30;

function animate() {
  if (auroraRampStart && auroraIntensity < 1) {
    auroraIntensity = Math.min(1, (performance.now() - auroraRampStart) / AURORA_RAMP_MS);
  }
  if (auroraRevealStart && auroraRevealX < width + 300) {
    // Ease-in-out so the wipe starts slow, accelerates, settles — reads as
    // "fluid" rather than a constant mechanical sweep.
    const t = Math.min(1, (performance.now() - auroraRevealStart) / AURORA_WIPE_MS);
    const eased = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
    auroraRevealX = -150 + eased * (width + 600);
  }
  currentX += (mouseX - currentX) * 0.25;
  currentY += (mouseY - currentY) * 0.25;

  // Sky (cached gradient: see skyGradient()).
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

  // Aurora borealis: 5 layered, independently-hued curtains. Each curtain
  // gets its own hue oscillation. The soft bloom (blur 30px) IS the look —
  // Chloe's call: do NOT collapse the visual into one fill.
  //
  // PERF ARCHITECTURE (the boot-wipe stutter fix):
  // The OLD code set ctx.filter='blur(30px)' and called ctx.fill() 5 times
  // per frame — 5 separate Gaussian blur passes. At rest that was tolerable
  // (warm pipeline), but the boot wipe fired into a cold pipeline and 5 cold
  // blur passes/frame stuttered visibly.
  //
  // The NEW code renders all 5 curtains to an offscreen buffer (auroraBuf)
  // with NO blur (cheap path fills), then blits the composite to the main
  // canvas through a SINGLE blur(30px). 5x fewer Gaussian passes per frame,
  // constant cost whether booting or at rest. The boot wipe is then a cheap
  // source-crop on the drawImage (only the revealed x-range is sampled),
  // so the blur also processes less data during the wipe — doubly cheap.
  if (auroraIntensity > 0.001 && auroraBufCtx) {
    const dpr = window.devicePixelRatio || 1;
    const curtains = 5;
    const baseCenterY = height * 0.42;

    // ── Pass 1: render curtains to offscreen buffer (NO blur), UNLESS the
    //    buffer is frozen AND already rendered this freeze. The freeze is set
    //    during the boot wipe (curtain shapes barely move in 0.5s, and any
    //    frozen-frame drift is masked by the blur + the wipe's own motion).
    //    At rest the freeze is cleared so the aurora animates live.
    if (!auroraBufFrozen || auroraBufDirty) {
      auroraBufCtx.setTransform(1, 0, 0, 1, 0, 0);
      auroraBufCtx.clearRect(0, 0, auroraBuf.width, auroraBuf.height);
      auroraBufCtx.scale(dpr, dpr);
      auroraBufCtx.globalCompositeOperation = 'source-over';

      // When frozen, paint at FULL intensity — the fade-in is driven by the
      // interpolated blur radius (10→30) + the wipe, not per-curtain alpha.
      // This snapshot is what the wipe reveals left-to-right.
      const a = auroraBufFrozen ? 0.18 : (0.18 * auroraIntensity);
      for (let i = 0; i < curtains; i++) {
        const speed = time * (0.1 + i * 0.04);
        const thickness = 45 + i * 15;
        const yOffset = (i - (curtains / 2)) * 12;
        const activeCenterY = baseCenterY + yOffset;

        auroraBufCtx.beginPath();
        for (let x = -150; x <= width + 150; x += 40) {
          const y = activeCenterY
                  + Math.sin(x * 0.0015 + speed + i * 2.3) * 85
                  + Math.cos(x * 0.0008 - speed) * 45
                  - thickness;
          if (x === -150) auroraBufCtx.moveTo(x, y);
          else auroraBufCtx.lineTo(x, y);
        }
        for (let x = width + 150; x >= -150; x -= 40) {
          const y = activeCenterY
                  + Math.sin(x * 0.0015 + speed + i * 2.3) * 85
                  + Math.cos(x * 0.0008 - speed) * 45
                  + thickness;
          auroraBufCtx.lineTo(x, y);
        }
        auroraBufCtx.closePath();

        const hue = currentPalette.hueBase + Math.sin(time * 1.0 + i) * currentPalette.hueRange;
        auroraBufCtx.fillStyle = `hsla(${hue}, 100%, 65%, ${a})`;
        auroraBufCtx.fill();
      }
      // If we just rendered the frozen snapshot, clear dirty so subsequent
      // frozen frames skip the redraw.
      if (auroraBufFrozen) auroraBufDirty = false;
    }

    // ── Pass 2: blit the composite with ONE interpolated blur pass.
    // Gaussian cost ~ radius², so scaling the radius 10→30 with intensity
    // makes the early wipe frames ~9x cheaper than the locked 30px. The
    // visual blooms as it reveals. At rest (intensity=1) the full 30px
    // returns and the look is identical to the locked aesthetic.
    ctx.globalCompositeOperation = 'screen';
    const blurPx = AURORA_BLUR_FLOOR +
      (AURORA_BLUR_CEIL - AURORA_BLUR_FLOOR) * auroraIntensity;
    ctx.filter = `blur(${blurPx.toFixed(1)}px)`;

    const wipeXCss = Math.min(Math.max(auroraRevealX, 0), width);
    const srcW = Math.floor(wipeXCss * dpr);
    if (srcW > 0) {
      ctx.drawImage(auroraBuf, 0, 0, srcW, auroraBuf.height, 0, 0, wipeXCss, height);
    }

    ctx.filter = 'none';
    ctx.globalCompositeOperation = 'source-over';
  } // end auroraIntensity > 0.001 cost gate
  time += 0.0025;
  // Don't schedule the next frame while paused: see `paused` + the
  // visibility/focus handlers below. The canvas RAF is the app's dominant
  // idle CPU/GPU cost; pausing it is what makes Sleep "barely noticeable"
  // AND what stops the lag when the window is covered/minimized.
  if (!paused) requestAnimationFrame(animate);
}

// Render loop control. `paused` is set by FOUR independent signals so the
// expensive RAF loop stops the moment the canvas isn't visible to the user:
//   0. BOOT GATE: `bootDone` is false until setupBootSplash()'s
//      revealAfterLand() runs (~0.5s after the paw lands). startLoop()
//      refuses to start while it's false, so no early focus/visibility event
//      can paint stars behind the boot paw. The canvas stays dormant while
//      the paw is hopping so the desktop is the only thing behind it.
//   1. `canvas-pause` event from Rust (system_menu power_sleep).
//   2. `document.visibilitychange` → hidden (alt-tab, minimize, another app
//      fully covering the window). The standard browser RAF throttle isn't
//      enough: WebView2 still fires RAF in some hidden states, and even a
//      throttled RAF re-runs the full animate() body.
//   3. `window.blur` (focus lost to another app) as a belt-and-suspenders
//      fallback when visibilitychange doesn't fire (e.g. another window
//      dragged over this one without minimizing).
// Resume mirrors all three. The animate() loop self-gates on `paused`.
let paused = true;
let bootDone = false;

function startLoop() {
  // Boot dormancy: refuse to start until setupBootSplash()'s revealAfterLand()
  // opens the gate. Without this, an early focus/visibility event during the
  // 5s paw hop would un-pause and paint stars behind the boot paw.
  if (!bootDone) return;
  if (paused) { paused = false; requestAnimationFrame(animate); }
}

// Tauri emits these from system_menu power_sleep / power_wake. Guard with
// .catch so a dev preview outside Tauri doesn't throw on the listener.
listen('canvas-pause', () => { paused = true; }).catch(() => {});
listen('canvas-resume', () => { startLoop(); }).catch(() => {});

// Pause when the page is hidden (alt-tab / minimize / tab switch). This is
// THE fix for "lag when another app covers the window": without it the RAF
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

// NOTE: animate() is NOT kicked off here at module-load time. The canvas is
// dormant during the boot paw phase (paused = true). setupBootSplash()'s
// revealAfterLand() opens bootDone + calls startLoop() ~0.5s after the paw
// lands — the first animate() frame paints sky + stars only, then the aurora
// blooms in over AURORA_RAMP_MS once the ramp is armed. Calling animate() at
// module load would paint stars behind the boot paw (the "background shows
// with the circle" bug) AND fight the boot gate.

// "WUPI OS" title: live AI-status indicator
// The title reflects the live state of the MAIN chat model (local 12B OR the
// connected API model: NOT Agent.gguf, which runs on its own thread and
// never drives chat_send). Three states:
//   - 'idle'    : connected, not generating → steady medium white glow
//   - 'offline' : no AI connected (boot pre-load, ONLINE w/ no profile,
//                 connect error, or a mode swap in progress) → fast red flash
//   - 'typing'  : main model actively generating tokens → subtle random
//                 white pulse spurts driven by a jittered setTimeout loop
//                 (CSS can't do random timing).
//
// State inputs:
//   1. The `model-status` Tauri event: Rust already emits ready/error/
//      no_model at boot + on api_disconnect reload; this is the offline/idle
//      authority. We just never listened before.
//   2. The chat IIFE's setGenerating() flag: bridges to 'typing'/'idle'.
//   3. The AI panel's mode swaps: calls 'offline' while a swap is pending.
const osTitleEl = document.querySelector('.os-title');
let titleState = 'idle';      // 'idle' | 'offline' | 'typing'
let titleFlickerTimer = null;  // the setTimeout handle for the typing pulse

function applyTitleClass() {
  if (!osTitleEl) return;
  osTitleEl.classList.remove('is-offline', 'is-typing');
  if (titleState === 'offline') osTitleEl.classList.add('is-offline');
  else if (titleState === 'typing') osTitleEl.classList.add('is-typing');
}

// The random "typing" pulse: toggles .title-flicker on a jittered timer so
// the glow bursts feel organic (like someone actually typing). ON 80-200ms,
// OFF 120-500ms, re-rolled each cycle. Stops when state leaves 'typing'.
function scheduleNextFlicker() {
  if (titleState !== 'typing' || !osTitleEl) return;
  const isOn = osTitleEl.classList.contains('title-flicker');
  const delay = isOn
    ? 80 + Math.random() * 120   // ON duration: 80-200ms
    : 120 + Math.random() * 380; // OFF duration: 120-500ms
  titleFlickerTimer = setTimeout(() => {
    if (titleState !== 'typing') return;
    osTitleEl.classList.toggle('title-flicker');
    scheduleNextFlicker();
  }, delay);
}

function stopFlicker() {
  if (titleFlickerTimer) { clearTimeout(titleFlickerTimer); titleFlickerTimer = null; }
  if (osTitleEl) osTitleEl.classList.remove('title-flicker');
}

function setTitleState(state) {
  if (!osTitleEl || state === titleState) return;
  const wasTyping = titleState === 'typing';
  titleState = state;
  applyTitleClass();
  if (state === 'typing') {
    scheduleNextFlicker();
  } else if (wasTyping) {
    stopFlicker();
  }
}

// Subscribe to Rust's model-status events (already emitted, previously
// unobserved). Boot starts at 'idle' (steady white) per Chloe's call: the
// pulse only fires for actual typing, and the red alarm only for confirmed
// offline/error states. The first model-status event then corrects to the
// real state.
(async () => {
  try {
    await listen('model-status', (e) => {
      const status = e?.payload?.status;
      // typing state is owned by the chat flag; don't clobber it here. Only
      // model-status transitions affect idle/offline.
      if (titleState === 'typing') return;
      if (status === 'ready') setTitleState('idle');
      else if (status === 'error' || status === 'no_model') setTitleState('offline');
    });
  } catch (err) {
    console.warn('[Wupi] model-status listen failed', err);
  }
})();

// ─── Boot paw → fly home → staged reveal ────────────────────────────────────
// The OS window boots transparent + always-on-top (tauri.conf.json) and STAYS
// transparent for its lifetime. What controls desktop bleed-through is the
// BODY background-color:
//   - body.booting         → transparent (CSS) → desktop shows through.
//   - body:not(.booting)   → #02040a (CSS)     → solid black covers desktop.
//
// CHOREOGRAPHY (per spec, refined):
//   0.0s  Blank screen (paw parked below the bottom edge, off-screen).
//   1.0s  Paw ENTERS from the bottom, jumping up to viewport center.
//         Two hops total; sparkles burst at each hop apex.
//   ~3.4s After hop 2 lands, paw FLIES to its home spot in the top-left
//         (the real .paw-img rect, read via getBoundingClientRect), shrinking
//         from ~120px → 45px. easeOutQuint, no overshoot.
//   land  Staged reveal (see revealAfterLand): body opaque → top-bar fades in
//         → canvas paints sky+stars → aurora LEFT-TO-RIGHT wipe (cheap during
//         reveal) → boot-paw removed → dock fades in last.
//
// STAGING NOTE: the top-bar's backdrop-filter:blur and the aurora's blur(30px)
// are the two heavy GPU costs. They are now staged so they DON'T overlap —
// the top-bar finishes its 0.6s fade BEFORE the aurora wipe arms. That's the
// real fix for "aurora load-in looks laggy": it's not the aurora alone, it's
// the aurora + top-bar blur running concurrently.
//
// Gate: chat `model-status: ready` (the 12B load — Rust's single source of
// truth, Rust is untouched) AND a minimum dwell timer. Both must resolve
// before the flight begins (the entry + hops always run regardless — they're
// the loading animation that hides the model load). The existing model-status
// listener above keeps its title-indicator job; this is a SEPARATE listener
// so the title's `typing` no-op guard can't swallow the wake signal.
(function setupBootSplash() {
  // Timing constants (ms).
  const ENTRY_DELAY = 3000;       // blank screen before paw enters (+2s per spec)
  const ENTRY_DURATION = 700;     // paw rises from bottom → center
  const HOP_DURATION = 450;       // each hop (up + down) — faster per spec
  const HOP_APEX = HOP_DURATION / 2;
  const HOP_HEIGHT = 80;          // px the inner img rises per hop
  const PAUSE_BETWEEN_HOPS = 120; // tiny rest between hop 1 and hop 2
  // Paw display size at center. The resting paw-img is 45px; ~2.67x makes
  // it ~120px, prominent in the middle of the screen during the hops.
  const PAW_BOOT_SCALE = 2.67;
  const PAW_REST_SIZE = 45;
  // Staged-reveal delays (ms) measured from flight-land (transitionend).
  // Top-bar fade is 0.6s in CSS; aurora wipe arms AFTER it finishes so the
  // two blur costs never overlap.
  const DELAY_SKY = 200;          // canvas RAF starts (sky + stars only)
  const DELAY_PAW_REMOVE = 400;   // boot-paw fades → real paw revealed
  const DELAY_AURORA = 800;       // aurora wipe arms (after top-bar's 0.6s fade)
  // Min-dwell: gate the FLIGHT on the model being ready + a floor so the
  // hops always play out even if the model loads instantly.
  const MIN_DWELL_MS = ENTRY_DELAY + ENTRY_DURATION + 2 * HOP_DURATION + PAUSE_BETWEEN_HOPS + 200;

  let modelReady = false;
  let dwellDone = false;
  let flightApproved = false;
  let hopsDone = false;

  const bootPaw = document.getElementById('boot-paw');
  const bootPawImg = bootPaw ? bootPaw.querySelector('.boot-paw-img') : null;
  const realPaw = document.querySelector('.paw-img');

  // <body> already carries .booting from index.html; this is belt-and-suspenders
  // for the dev-preview case where the HTML might not have it.
  document.body.classList.add('booting');

  // ── Phase 0: park the paw off-screen below center, scaled up, BEFORE any
  //    transition can fire. Setting transform inline at module-eval time
  //    means the first paint already has it hidden — no flash at center.
  //
  //    CRITICAL: the CSS rule on #boot-paw has `transition: transform 1.2s`
  //    always active. If we just set style.transform, the browser animates
  //    from the default (translate(0,0) scale(1) = top-left) to the parked
  //    position over 1.2s — making the paw visibly slide across the screen
  //    during the 1s "blank" phase. Fix: disable the transition, set the
  //    transform, force a reflow so the browser commits the un-transitioned
  //    state, then re-enable the transition. Subsequent transform changes
  //    (the entry WAAPI animation + the flight) animate correctly.
  if (bootPaw) {
    const restCx = (window.innerWidth - PAW_REST_SIZE) / 2;
    const parkCy = window.innerHeight + 50; // just below the bottom edge
    bootPaw.style.transition = 'none';      // suppress the slide-to-park
    bootPaw.style.width = PAW_REST_SIZE + 'px';
    bootPaw.style.height = PAW_REST_SIZE + 'px';
    bootPaw.style.transform =
      `translate(${restCx}px, ${parkCy}px) scale(${PAW_BOOT_SCALE})`;
    // Force reflow: reading offsetHeight synchronously flushes the style
    // queue so the browser commits the parked transform BEFORE we restore
    // the transition. Without this the two style writes batch together and
    // the transition still fires on the park.
    void bootPaw.offsetHeight;
    // Restore the CSS transition for the flight (entry uses WAAPI which
    // overrides inline style anyway, but the flight relies on this).
    bootPaw.style.transition = '';
  }

  // ── Sparkle burst. Spawns N .boot-sparkle children of #boot-paw, each
  //    flying outward in a random direction via the --burst CSS var. They
  //    self-clean via the animation's `forwards` + a transitionend remover.
  function spawnSparkles(count = 8) {
    if (!bootPaw) return;
    for (let i = 0; i < count; i++) {
      const s = document.createElement('div');
      s.className = 'boot-sparkle';
      const angle = (Math.PI * 2 * i) / count + Math.random() * 0.4;
      const dist = 30 + Math.random() * 40;
      const dx = Math.cos(angle) * dist;
      const dy = Math.sin(angle) * dist - 10; // bias upward slightly
      s.style.setProperty('--burst', `translate(${dx.toFixed(1)}px, ${dy.toFixed(1)}px)`);
      bootPaw.appendChild(s);
      s.addEventListener('animationend', () => s.remove(), { once: true });
    }
  }

  // ── Entry + hops. Uses the Web Animations API so we can dispatch sparkle
  //    bursts at exact hop apexes (timings.onFrame). The inner img's
  //    translateY animates; #boot-paw's translate/scale are reserved for the
  //    entry rise + the later flight.
  function startEntryAndHops() {
    if (!bootPaw || !bootPawImg) { hopsDone = true; maybeFly(); return; }

    // Entry: animate #boot-paw's transform from "parked below" → "center".
    // Done via WAAPI so the entry and the hops compose cleanly, and so we
    // get a precise entry-end callback for starting hop 1.
    const restCx = (window.innerWidth - PAW_REST_SIZE) / 2;
    const restCy = (window.innerHeight - PAW_REST_SIZE) / 2;
    const parkCy = window.innerHeight + 50;

    const entryAnim = bootPaw.animate(
      [
        { transform: `translate(${restCx}px, ${parkCy}px) scale(${PAW_BOOT_SCALE})` },
        { transform: `translate(${restCx}px, ${restCy}px) scale(${PAW_BOOT_SCALE})` },
      ],
      { duration: ENTRY_DURATION, easing: 'cubic-bezier(0.22, 1, 0.36, 1)', fill: 'forwards' }
    );
    // Commit the entry's final state to the inline style so subsequent CSS
    // transitions (the flight) animate FROM this exact matrix.
    entryAnim.onfinish = () => {
      entryAnim.commitStyles();
      entryAnim.cancel();
      runHops();
    };
  }

  function runHops() {
    if (!bootPawImg) { hopsDone = true; maybeFly(); return; }
    let hop = 0;
    const doHop = () => {
      hop++;
      const a = bootPawImg.animate(
        [
          { transform: 'translateY(0)' },
          { transform: `translateY(-${HOP_HEIGHT}px)` },
          { transform: 'translateY(0)' },
        ],
        { duration: HOP_DURATION, easing: 'ease-in-out', fill: 'forwards' }
      );
      // Sparkle at the apex (HOP_APEX ms in).
      setTimeout(spawnSparkles, HOP_APEX);
      a.onfinish = () => {
        if (hop < 2) {
          setTimeout(doHop, PAUSE_BETWEEN_HOPS);
        } else {
          // Lock the inner img at translateY(0) so the flight transform is clean.
          a.commitStyles();
          a.cancel();
          hopsDone = true;
          maybeFly();
        }
      };
    };
    doHop();
  }

  // Kick off the entry + hops on a timer (the 1s blank pause).
  setTimeout(startEntryAndHops, ENTRY_DELAY);

  // ── Flight gate. Both model-ready AND hops-done AND min-dwell must hold.
  function maybeFly() {
    if (flightApproved) return;
    if (!(modelReady && hopsDone && dwellDone)) return;
    flyPawHome();
  }

  // Min-dwell floor so the choreography always plays out.
  setTimeout(() => { dwellDone = true; maybeFly(); }, MIN_DWELL_MS);

  // Primary gate: the chat 12B model finished loading.
  listen('model-status', (e) => {
    if (e?.payload?.status === 'ready') {
      modelReady = true;
      maybeFly();
    }
  }).catch(() => {});
  // Safety net: no_model (echo mode) / error also resolve the gate so the user
  // is never trapped behind the boot phase if the model path fails. The title
  // indicator still shows 'offline' via the listener above.
  listen('model-status', (e) => {
    const s = e?.payload?.status;
    if (s === 'no_model' || s === 'error') {
      modelReady = true;
      maybeFly();
    }
  }).catch(() => {});

  // ── Phase 2: fly the paw from center → home. Reads the real .paw-img's
  //    current rect so the landing is pixel-accurate regardless of layout.
  function flyPawHome() {
    flightApproved = true;
    if (!bootPaw) { revealAfterLand(); return; }

    // Read the real paw's resting rect. During boot the top-bar is at
    // opacity:0 but still laid out (NOT display:none), so getBoundingClientRect
    // returns the true home coordinates.
    let targetX = 0, targetY = 0;
    if (realPaw) {
      const r = realPaw.getBoundingClientRect();
      targetX = r.left;
      targetY = r.top;
    }

    // One-shot: when the flight transition ends, run the staged reveal.
    const onLand = (e) => {
      if (e.propertyName !== 'transform') return;
      bootPaw.removeEventListener('transitionend', onLand);
      revealAfterLand();
    };
    bootPaw.addEventListener('transitionend', onLand);

    // Set the target transform. The CSS transition on #boot-paw
    // (transform 1.2s easeOutQuint, no overshoot) animates the flight.
    // rAF double-buffer so the browser commits the start transform before
    // we set the target, guaranteeing the transition runs.
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        bootPaw.style.transform =
          `translate(${targetX}px, ${targetY}px) scale(1)`;
      });
    });
  }

  // ── Phase 3: staged reveal. Called when the paw lands. Each step is a
  //    setTimeout off the landing moment.
  function revealAfterLand() {
    // +0.0s: drop .booting. Body goes opaque #02040a (CSS), AND the top-bar
    // + dock opacity transitions arm (their CSS rules key off :not(.booting)).
    // The top-bar starts fading in immediately (0.1s CSS delay, 0.6s fade).
    document.body.classList.remove('booting');

    // +0.2s: start the canvas RAF. First frame paints sky + stars only
    // (curtain block gated on auroraIntensity > 0.001, still 0 here, AND
    // auroraRevealX still ~0 so even if it weren't, nothing would draw).
    setTimeout(() => {
      bootDone = true;
      startLoop();
    }, DELAY_SKY);

    // +0.4s: fade + remove the boot paw. The top-bar is well into its fade
    // by now, so the real .paw-img reads as a continuous handoff.
    setTimeout(() => {
      if (!bootPaw) return;
      bootPaw.classList.add('fade-out');
      bootPaw.addEventListener('transitionend', () => bootPaw.remove(), { once: true });
    }, DELAY_PAW_REMOVE);

    // +0.8s: arm the aurora ramp + the left-to-right wipe. Staged AFTER the
    // top-bar's 0.6s fade (which started at +0.1s) so the two blur costs
    // don't overlap — this is the real fix for "aurora load-in looks laggy".
    // The buffer FREEZE is set here so the wipe frames don't redraw the
    // curtain paths (they barely move in 0.5s); cleared AURORA_WIPE_MS later
    // so the aurora animates live at rest.
    setTimeout(() => {
      auroraRampStart = performance.now();
      auroraRevealStart = performance.now();
      auroraBufFrozen = true;
      auroraBufDirty = true;  // render the snapshot once on the next frame
      setTimeout(() => { auroraBufFrozen = false; }, AURORA_WIPE_MS + 100);
    }, DELAY_AURORA);
  }
})();

// NOTE: this file is loaded as type="module", which defers execution until
// after the DOM is parsed: so DOMContentLoaded has ALREADY fired by the time
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
          // No signal %: it was noisy and the same network appeared multiple
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

  // "Add Device": discover in-range unpaired BT devices and list them under
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

  // Split into two pieces to kill the flicker: the volume/mute is polled every
  // 1s (slider/percent/icon only: no DOM rebuild), and the output-device list
  // is built ONCE when the dropdown opens (it almost never changes mid-session).
  // The previous version rebuilt the whole list each tick → flicker.

  function refreshAudioVolume() {
    invoke('audio_get_state')
      .then((s) => {
        if (!s) return;
        // Only touch the slider/percent/icon: never rebuild the device list.
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

  // APP WINDOW MANAGER
  // The surfaces (Chat, Profile Editor, Codex, Docks) are DOM overlays in
  // the ONE Tauri window. Background rules (per Chloe's spec):
  //   - WUPI Chat (chat): the ONLY window that pauses the canvas (stars +
  //     aurora OFF). Its own background is ~80% opaque so the paused backdrop
  //     doesn't show through. Closing it resumes the canvas.
  //   - Everything else (Codex, Profile, Docks home): canvas keeps running -
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
  // No window pauses the canvas anymore: the background stays active behind
  // every surface (Chat is now translucent enough that stars show through).
  // Kept as a hook in case a future surface wants to freeze the background.
  function syncCanvasForWindows() {
    /* no-op: background always active */
  }

  function openWindow(id) {
    const el = document.getElementById(id);
    if (!el) return;
    if (openWindows.has(id)) {
      // Already open: just raise it to the top.
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
    // Fire an onClose hook if the surface registered one (e.g. Chat tears
    // down an active ink-reveal timer so it doesn't tick against a detached
    // node).
    const closeHook = windowCloseHooks.get(id);
    if (closeHook) closeHook();
  }

  // Surfaces register an async onOpen hook (load data when first shown).
  const windowOpenHooks = new Map();
  // Surfaces register an onClose hook (tear down timers, etc.).
  const windowCloseHooks = new Map();

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

  // Header is the drag handle. The window is absolutely positioned; dragging
  // updates `left`/`top`. Only windows with `.draggable` get this: Chat is
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

  // Click an open app's dock item again → closes it (toggle behavior). The
  // quick-access dock order is fixed: API → Chat → Profile → Codex (NOT
  // alphabetical: that's the Docks home grid). Apps (Docks launcher) is
  // special: it closes any open surface windows then shows the home grid.
  function dockToggle(id) {
    if (openWindows.has(id)) closeWindow(id);
    else openWindow(id);
  }

  document.getElementById('dockApi')?.addEventListener('click', (e) => {
    e.stopPropagation();
    dockToggle('api');
  });
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
    // grid. (apps itself is the full-screen home overlay.) Not a toggle -
    // clicking Docks while home is open is a no-op (it's already home).
    if (openWindows.has('apps')) return;
    closeWindow('api');
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

  // PROFILE EDITOR
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

    // Load fresh every time the window opens: cheap, and guarantees the editor
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
        .then(() => setStatus('Saved: applies next message', 'ok'))
        .catch((err) => setStatus('Save failed: ' + err, 'err'))
        .finally(() => { saveBtn.disabled = false; });
    });
  })();

  // AI: Connection Profile panel (LOCAL | ONLINE mode selector + profile CRUD)
  // Source of truth = api_config.json (loaded at boot into AppState). The
  // panel shows two large mode boxes: LOCAL (the single WUPI 12B bubble) or
  // ONLINE (saved endpoint profiles + an editor). Selecting ONLINE triggers
  // the model swap (12B unloads, Agent.gguf spins up for schema/memory);
  // selecting LOCAL reverts it. Temperature is fixed at 1.0 (no UI field).
  // The model field is a dropdown populated from the endpoint's /models
  // list after a successful connect: never free text.
  (function apiPanel() {
    const root = document.getElementById('api');
    if (!root) return;
    const panel = document.getElementById('aiPanel');
    const editorEl = document.getElementById('apiEditor');
    const nameEl = document.getElementById('apiName');
    const endpointEl = document.getElementById('apiEndpoint');
    const keyEl = document.getElementById('apiKey');
    const addBtn = document.getElementById('aiAddBtn');
    const editProfileBtn = document.getElementById('aiEditProfileBtn');
    const deleteProfileBtn = document.getElementById('aiDeleteProfileBtn');
    const localConnectBtn = document.getElementById('aiLocalConnectBtn');
    const statusEl = document.getElementById('apiStatus');
    const modeLocalBtn = document.getElementById('aiModeLocal');
    const modeOnlineBtn = document.getElementById('aiModeOnline');
    const localSection = document.getElementById('aiLocalSection');
    const onlineSection = document.getElementById('aiOnlineSection');
    const profileSelect = document.getElementById('aiProfileSelect');
    const modelSelect = document.getElementById('apiModel');
    const onlineBubble = document.getElementById('aiOnlineBubble');
    const connectBtn = document.getElementById('aiConnectBtn');

    let editingId = null; // null = creating; string = editing existing
    let lastConfig = null; // cached for rendering
    let currentMode = 'local'; // UI view: 'local' | 'online'
    let runtimeSource = 'local'; // actual backend source
    let activeProfileId = null; // currently-connected profile (mirror of backend)
    // Model cache: profileId → { ids: [..], selected: str }. Avoids refetching
    // /models when toggling between already-loaded profiles.
    const modelCache = new Map();
    let modeInitialized = false;

    function setStatus(msg, kind) {
      statusEl.textContent = msg || '';
      statusEl.className = 'profile-status' + (kind ? ' ' + kind : '');
    }

    function escapeHtml(s) {
      return String(s || '').replace(/[&<>"']/g, (c) => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
      }[c]));
    }

    function findProfile(id) {
      return lastConfig?.profiles.find((p) => p.id === id) || null;
    }

    // Render the profile dropdown from the cached config. Sorted alphabetically
    // by name. Active profile is flagged with a ● prefix. The "Create a New
    // Profile" placeholder option is ONLY shown when there are zero saved
    // profiles: once any exist it disappears (the + button is the create
    // affordance then). Selecting the placeholder focuses the editor.
    function renderProfileSelect(config) {
      lastConfig = config;
      const profiles = [...(config.profiles || [])].sort((a, b) =>
        (a.name || a.id).localeCompare(b.name || b.id)
      );
      // Capture the selection BEFORE we rebuild the DOM. After innerHTML
      // rebuilds the options, the .value reverts to "": so we must remember
      // it now and re-apply it after.
      const prevValue = profileSelect.value;

      if (profiles.length === 0) {
        // No saved profiles yet: the dropdown IS the "create" affordance.
        profileSelect.innerHTML = '<option value="">Create a New Profile</option>';
        profileSelect.disabled = false;
        editProfileBtn.disabled = true;
        deleteProfileBtn.disabled = true;
        return;
      }
      profileSelect.disabled = false;
      // Once profiles exist, drop the "Create a New Profile" placeholder -
      // the + button below handles creation.
      profileSelect.innerHTML = profiles.map((p) => {
        const isActive = p.id === config.active_profile_id;
        return `<option value="${escapeHtml(p.id)}">${isActive ? '● ' : ''}${escapeHtml(p.name || p.id)}</option>`;
      }).join('');
      // Re-apply the previous selection if that profile still exists.
      // Otherwise default to the active profile (or the first one).
      const stillExists = (id) => id && [...profileSelect.options].some((o) => o.value === id);
      const target = stillExists(prevValue) ? prevValue
                   : stillExists(config.active_profile_id) ? config.active_profile_id
                   : profiles[0].id;
      profileSelect.value = target;
      // Edit/trash are enabled whenever a real profile is selected.
      // Per Chloe: even a single profile must be editable/deletable.
      const hasRealSelection = !!profileSelect.value;
      editProfileBtn.disabled = !hasRealSelection;
      deleteProfileBtn.disabled = !hasRealSelection;
    }

    // Update the online bubble. Three states:
    //   - connected (runtime on API): magenta glow + "Name: model"
    //   - selection pending (profile+model picked, not yet Connect'd): subdued
    //     preview of what Connect will activate, no glow
    //   - nothing picked: muted "No profile connected"
    function renderOnlineBubble() {
      // Connected: runtime actually on API with an active profile.
      if (runtimeSource === 'api' && activeProfileId) {
        const p = findProfile(activeProfileId);
        if (p) {
          onlineBubble.classList.add('active');
          onlineBubble.classList.remove('pending');
          onlineBubble.innerHTML =
            `<span class="ai-online-bubble-text">${escapeHtml(p.name || p.id)}</span>` +
            `<span class="ai-online-bubble-sep">-</span>` +
            `<span class="ai-online-bubble-model">${escapeHtml(p.model || '?')}</span>`;
          return;
        }
      }
      // Selection pending: profile + model both picked in the dropdowns but
      // not yet connected. Show a preview so the user sees what they're about
      // to activate. Uses the "pending" style (no glow, lighter text).
      const pickedProfileId = profileSelect?.value;
      const pickedModel = modelSelect?.value;
      if (pickedProfileId && pickedModel) {
        const p = findProfile(pickedProfileId);
        if (p) {
          onlineBubble.classList.remove('active');
          onlineBubble.classList.add('pending');
          onlineBubble.innerHTML =
            `<span class="ai-online-bubble-text">${escapeHtml(p.name || p.id)}</span>` +
            `<span class="ai-online-bubble-sep">-</span>` +
            `<span class="ai-online-bubble-model">${escapeHtml(pickedModel)}</span>`;
          return;
        }
      }
      // Nothing useful to show.
      onlineBubble.classList.remove('active', 'pending');
      onlineBubble.innerHTML = '<span class="ai-online-bubble-text">No profile connected</span>';
    }

    // Fetch /models for a profile + populate the model dropdown. Cached per
    // profile so switching back doesn't refetch. Default-selects the saved
    // model if present in the list, else the first alphabetically. The list
    // is sorted alphabetically (case-insensitive): NanoGPT's /models returns
    // 100+ models in provider-defined order (a chaotic mix of org/name), so
    // alphabetical is the only sane default. There's no membership/free-vs-
    // paid field in the OpenAI-standard /models response, so we can't group
    // by tier without custom metadata: just alphabetize for now.
    async function populateModelDropdown(profile) {
      if (!profile) {
        modelSelect.innerHTML = '<option value="">Pick a profile to load models…</option>';
        modelSelect.disabled = true;
        return;
      }
      // Cache hit. But HONOR the user's current in-UI selection: if the
      // dropdown already has a value and it's still in the cached list,
      // keep it selected. Otherwise a refresh() after Connect would fling
      // the selection back to the cache's stale `selected` field.
      const cached = modelCache.get(profile.id);
      if (cached) {
        const currentPick = modelSelect.value;
        const honored = (currentPick && cached.ids.includes(currentPick))
          ? currentPick
          : cached.selected;
        renderModelOptions(cached.ids, honored);
        return;
      }
      modelSelect.disabled = true;
      modelSelect.innerHTML = '<option value="">Loading models…</option>';
      try {
        const v = await invoke('api_profile_test', { profile });
        const rawIds = (v && Array.isArray(v.data))
          ? v.data.map((m) => (typeof m === 'string' ? m : m?.id)).filter(Boolean)
          : [];
        if (rawIds.length === 0) {
          modelSelect.innerHTML = '<option value="">No models returned</option>';
          return;
        }
        // Sort alphabetically, case-insensitive, deterministic for equal keys.
        const ids = [...rawIds].sort((a, b) =>
          a.toLowerCase().localeCompare(b.toLowerCase()) || a.localeCompare(b)
        );
        // Default to the profile's saved model if it's in the list; else the
        // first alphabetically. The user's in-UI pick (if any) takes priority
        // on cache hit (handled above).
        const preferred = (profile.model && ids.includes(profile.model)) ? profile.model : ids[0];
        modelCache.set(profile.id, { ids, selected: preferred });
        renderModelOptions(ids, preferred);
      } catch (err) {
        modelSelect.innerHTML = '<option value="">Failed to load models</option>';
        setStatus('Model list fetch failed: ' + err, 'err');
      }
    }

    function renderModelOptions(ids, selected) {
      modelSelect.innerHTML = ids.map((id) =>
        `<option value="${escapeHtml(id)}"${id === selected ? ' selected' : ''}>${escapeHtml(id)}</option>`
      ).join('');
      modelSelect.disabled = false;
    }

    // Update the Connect button's enabled state. Requires a profile + model.
    function updateConnectEnabled() {
      const ready = !!profileSelect.value && !!modelSelect.value;
      connectBtn.disabled = !ready;
    }

    // Apply the current UI mode to the DOM. Pure view; no backend call.
    function applyMode() {
      panel.dataset.mode = currentMode;
      modeLocalBtn.classList.toggle('active', currentMode === 'local');
      modeOnlineBtn.classList.toggle('active', currentMode === 'online');
      localSection.classList.toggle('hidden', currentMode !== 'local');
      onlineSection.classList.toggle('visible', currentMode === 'online');
    }

    async function refresh() {
      try {
        const config = await invoke('api_profiles_list');
        const extra = await invoke('model_source_get');
        lastConfig = config;
        runtimeSource = (extra?.source || config.model_source) === 'api' ? 'api' : 'local';
        activeProfileId = config.active_profile_id || null;
        renderProfileSelect(config);
        renderOnlineBubble();
        // Seed the mode from the backend ONCE on first refresh. After that
        // the mode is the user's click: refresh() must never clobber it.
        if (!modeInitialized) {
          currentMode = runtimeSource === 'api' ? 'online' : 'local';
          modeInitialized = true;
          applyMode();
        }
        // ALWAYS populate the model dropdown for the currently-selected
        // profile (if any). Programmatic .value = ... doesn't fire the
        // change event, so this is the only reliable way to keep the model
        // list in sync after a refresh.
        if (currentMode === 'online' && profileSelect.value) {
          await populateModelDropdown(findProfile(profileSelect.value));
        }
        updateConnectEnabled();
        setStatus('');
      } catch (err) {
        setStatus('Load failed: ' + err, 'err');
      }
    }

    // Clicking LOCAL: disconnect API (if on API), reload the 12B. Clicking
    // ONLINE: reconnect the last-used API profile + model. No separate
    // disconnect affordance: you pick one or the other.
    modeLocalBtn?.addEventListener('click', async () => {
      if (currentMode === 'local' && runtimeSource === 'local') return;
      currentMode = 'local';
      applyMode();
      if (runtimeSource === 'api') {
        setTitleState('offline'); // red while the 12B reloads
        setStatus('Disconnecting API: reloading WUPI 12B…', '');
        try {
          await invoke('api_disconnect');
          setStatus('Back on local WUPI 12B.', 'ok');
        } catch (err) {
          setStatus('Disconnect failed: ' + err + '.', 'err');
        }
        await refresh();
      }
    });

    modeOnlineBtn?.addEventListener('click', async () => {
      currentMode = 'online';
      applyMode();
      // If a profile is already connected, nothing to do: just show it.
      if (runtimeSource === 'api' && activeProfileId) return;
      // Reconnect the last-used profile if one exists.
      if (activeProfileId) {
        setTitleState('offline');
        setStatus('Connecting last-used profile…', '');
        try {
          await invoke('api_connect', { profileId: activeProfileId });
          setStatus('Connected.', 'ok');
        } catch (err) {
          setStatus('Connect failed: ' + err + ': still on local.', 'err');
          setTitleState('idle');
        }
        await refresh();
      } else {
        // No active profile: ONLINE view is up, user picks one + hits Connect.
        setStatus('');
      }
    });

    // When the dropdown has no real selection (zero-profile state: the
    // "Create a New Profile" placeholder is selected), focus the editor so
    // the user can start typing their first profile.
    profileSelect?.addEventListener('change', async () => {
      const selectedId = profileSelect.value;
      if (!selectedId) {
        // "Create a New Profile" (or no selection): prep the editor.
        clearEditor();
        nameEl?.focus();
        // Edit/trash aren't meaningful without a real profile.
        editProfileBtn.disabled = true;
        deleteProfileBtn.disabled = true;
        updateConnectEnabled();
        renderOnlineBubble();
        return;
      }
      const p = findProfile(selectedId);
      await populateModelDropdown(p);
      updateConnectEnabled();
      renderOnlineBubble();
      // Real profile selected: enable edit/trash.
      editProfileBtn.disabled = false;
      deleteProfileBtn.disabled = false;
    });

    // Also writes the new pick back into the cache so a subsequent refresh()
    // (which hits the cache) honors it instead of flinging back to the old
    // default: the cause of the "dropdown flings to first after Connect" bug.
    modelSelect?.addEventListener('change', () => {
      const pickedProfileId = profileSelect.value;
      const pickedModel = modelSelect.value;
      if (pickedProfileId && pickedModel) {
        const cached = modelCache.get(pickedProfileId);
        if (cached && cached.selected !== pickedModel) {
          modelCache.set(pickedProfileId, { ...cached, selected: pickedModel });
        }
      }
      updateConnectEnabled();
      renderOnlineBubble();
    });

    connectBtn?.addEventListener('click', async () => {
      const profileId = profileSelect.value;
      const modelId = modelSelect.value;
      if (!profileId || !modelId) return;
      // Persist the chosen model into the profile before connecting: the
      // backend's api_connect validates non-empty model.
      const p = findProfile(profileId);
      if (p && p.model !== modelId) {
        const updated = { ...p, model: modelId, temperature: 1.0 };
        try {
          await invoke('api_profile_save', { profile: updated });
        } catch (err) {
          setStatus('Could not save model choice: ' + err, 'err');
          return;
        }
      }
      setTitleState('offline'); // red while swapping
      setStatus('Connecting…', '');
      connectBtn.disabled = true;
      try {
        await invoke('api_connect', { profileId });
        setStatus('Connected: chat via API now.', 'ok');
      } catch (err) {
        setStatus('Connect failed: ' + err + ': still on local.', 'err');
        setTitleState('idle');
      }
      await refresh();
    });

    function clearEditor() {
      editingId = null;
      nameEl.value = '';
      endpointEl.value = '';
      keyEl.value = '';
      editorEl.classList.remove('editing');
      setStatus('');
    }

    function loadEditor(profile) {
      editingId = profile?.id || null;
      nameEl.value = profile?.name || '';
      endpointEl.value = profile?.endpoint || '';
      keyEl.value = profile?.api_key || '';
      editorEl.classList.add('editing');
      setStatus('Editing "' + (profile?.name || '') + '". + overwrites.');
      nameEl.focus();
    }

    // Errors via the status line if any field is empty. Does NOT auto-connect
    //: just lands the profile in the dropdown and auto-selects it.
    addBtn?.addEventListener('click', async () => {
      const name = nameEl.value.trim();
      if (!name) { setStatus('Name is required.', 'err'); nameEl.focus(); return; }
      if (!endpointEl.value.trim()) { setStatus('API URL is required.', 'err'); endpointEl.focus(); return; }
      if (!keyEl.value.trim()) { setStatus('API key is required.', 'err'); keyEl.focus(); return; }
      // Preserve the existing model if editing; new profiles start empty and
      // get their model from the dropdown after selection.
      const existing = editingId ? findProfile(editingId) : null;
      const profile = {
        id: editingId || '',
        name,
        endpoint: endpointEl.value.trim(),
        api_key: keyEl.value,
        model: existing?.model || '',
        temperature: 1.0,
      };
      addBtn.disabled = true;
      setStatus(editingId ? 'Saving…' : 'Adding…');
      try {
        const saved = await invoke('api_profile_save', { profile });
        const savedId = saved?.id || editingId || name;
        clearEditor();
        await refresh();
        // Auto-select the just-saved profile + populate its models.
        profileSelect.value = savedId;
        if (profileSelect.value === savedId) {
          profileSelect.dispatchEvent(new Event('change'));
          setStatus('Saved. Pick a model, then Connect.', 'ok');
        } else {
          setStatus('Saved.', 'ok');
        }
      } catch (err) {
        setStatus('Save failed: ' + err, 'err');
      } finally {
        addBtn.disabled = false;
      }
    });

    editProfileBtn?.addEventListener('click', () => {
      const p = findProfile(profileSelect.value);
      if (!p) { setStatus('Pick a profile to edit first.', 'err'); return; }
      loadEditor(p);
    });

    deleteProfileBtn?.addEventListener('click', async () => {
      const id = profileSelect.value;
      const p = findProfile(id);
      if (!p) { setStatus('Pick a profile to delete first.', 'err'); return; }
      if (!confirm(`Delete profile "${p.name || p.id}"?\nThis removes the saved API URL + key.`)) return;
      setStatus('Deleting…');
      try {
        await invoke('api_profile_delete', { profileId: id });
        // If we were editing this profile, clear the editor.
        if (editingId === id) clearEditor();
        setStatus('Deleted.', 'ok');
        await refresh();
      } catch (err) {
        setStatus('Delete failed: ' + err, 'err');
      }
    });

    // LOCAL is always "connected" to the 12B by design: clicking Connect
    // here is visual parity with ONLINE. If the runtime is somehow on API
    // (user connected then peeked at LOCAL), this triggers the disconnect +
    // 12B reload. Otherwise it's a no-op confirmation.
    localConnectBtn?.addEventListener('click', async () => {
      if (runtimeSource === 'api') {
        setTitleState('offline');
        localConnectBtn.disabled = true;
        setStatus('Disconnecting API: reloading WUPI 12B…', '');
        try {
          await invoke('api_disconnect');
          setStatus('Back on local WUPI 12B.', 'ok');
        } catch (err) {
          setStatus('Disconnect failed: ' + err, 'err');
        }
        await refresh();
      } else {
        setStatus('Already on local WUPI 12B.', 'ok');
      }
    });

    // Load fresh every time the window opens.
    windowOpenHooks.set('api', () => { refresh(); });
  })();

  // THE CODEX: authored lore library (NOT a memory browser)
  // Codex is a library of authored reference "books": world lore, TV-show
  // facts, worldbuilding. Source of truth = .md files in codex/ (re-seeded to
  // the retrieval index at boot + after each edit). It has NOTHING to do with
  // chat history or Wupi's persona: just the lore you author.
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

  // Ink reveal: paces streamed text to a smooth 10 chars/sec on the DOM,
  // independent of how fast the backend generates. The backend finishes in
  // ~1-2s for a typical turn but the user reads at ~10 cps, so the UI drips
  // the text out of a buffer on a timer. The blinking caret (`.streaming`
  // class on the bubble) stays on until the reveal catches up to the full
  // target. Shared by the chat path now and the game/narrator path when it
  // ships; the helper is agnostic to the source of the text.
  const REVEAL_TICK_MS = 100;       // 10 ticks/sec
  const REVEAL_CHARS_PER_TICK = 1;  // 10 chars/sec total

  // The currently-active reveal, if any. Module-level so a new send() can
  // flush a still-draining previous reveal before starting its own.
  let activeReveal = null;

  // Start an ink reveal on `bubble`. Returns a handle with push/flush/destroy.
  // The caller pushes the full accumulated text on every chunk; the helper
  // advances a visible cursor on a timer and writes `target.slice(0, shown)`
  // to `bubble.textContent`.
  //
  // `onTick` (optional) fires after every visible write (e.g. to scroll).
  // `onComplete` (optional) fires once when the caller signals the target is
  // final (push with isFinal=true) AND the reveal has shown all of it. This
  // is the hook for finalizing the bubble (removing the caret, showing the
  // reasoning panel). If the reveal already caught up when the final push
  // arrives, onComplete fires synchronously; otherwise it fires from the
  // tick loop when shown reaches target.length.
  function startInkReveal(bubble, onTick, onComplete) {
    let target = '';
    let shown = 0;
    let timer = null;
    let finalArmed = false;   // caller pushed isFinal=true
    let completed = false;    // onComplete has fired
    const clearActive = () => {
      if (activeReveal === api) activeReveal = null;
    };
    const stop = () => {
      if (timer !== null) { clearInterval(timer); timer = null; }
    };
    const write = () => {
      bubble.textContent = target.slice(0, shown);
      if (typeof onTick === 'function') onTick();
    };
    const maybeComplete = () => {
      if (finalArmed && !completed && shown >= target.length) {
        completed = true;
        stop();
        clearActive();
        if (typeof onComplete === 'function') onComplete();
      }
    };
    const tick = () => {
      if (shown < target.length) {
        shown = Math.min(shown + REVEAL_CHARS_PER_TICK, target.length);
        write();
      } else {
        // Nothing left to drip. Stop the timer; restart on next push.
        stop();
      }
      maybeComplete();
    };
    const api = {
      // Set the full target text. isFinal=true marks it as the last push
      // (backend sent `done`): onComplete fires when the reveal catches up.
      // A new push always re-arms completion (a previously-completed reveal
      // can fire onComplete again if more final text arrives, e.g. the user
      // clicked stop mid-reveal, the reveal flushed, then `done` arrived).
      push(fullText, isFinal) {
        target = fullText;
        if (isFinal) {
          finalArmed = true;
          completed = false;
        }
        if (shown < target.length && timer === null) {
          timer = setInterval(tick, REVEAL_TICK_MS);
        }
        maybeComplete();
      },
      // Reveal everything immediately and stop. Used when the caller wants
      // to skip ahead (user clicked stop, or a new send preempts this one).
      flush(fullText) {
        if (fullText != null) target = fullText;
        shown = target.length;
        stop();
        write();
        maybeComplete();
        // flush is not necessarily "final" — if the caller wants onComplete
        // to fire, push(finalText, true) before flushing, or the caller's
        // own done-handling runs after flush returns.
      },
      // Tear down without a final write or completion fire (used on error:
      // the bubble is removed anyway).
      destroy() {
        stop();
        clearActive();
      },
    };
    activeReveal = api;
    return api;
  }

  // WUPI CHAT: full streaming chat surface
  (function wupiChat() {
    const msgsEl = document.getElementById('chatMessages');
    const inputEl = document.getElementById('chatInput');
    const sendBtn = document.getElementById('chatSendBtn');
    const stopBtn = document.getElementById('chatStopBtn');
    if (!msgsEl) return;

    // Tauri v2 Channel for streaming: imported statically at the top of the
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

    // A static (non-streaming) Wupi message: used for the randomized intro
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
      // Bridge to the title status indicator: the main model is "typing"
      // during a chat_send. This flag is the authoritative source: it only
      // flips on user-driven chat sends, so Agent.gguf (schema engine, own
      // thread, never drives chat_send) is excluded by construction.
      setTitleState(on ? 'typing' : 'idle');
    }

    async function send() {
      if (generating) return;
      const text = inputEl.value.trim();
      if (!text) return;

      // If a previous reveal is still dripping, flush it to completion so the
      // user sees the full previous reply before the new one starts.
      if (activeReveal) activeReveal.flush();

      inputEl.value = '';
      addUserBubble(text);

      const bubble = startWupiBubble();
      let streamed = '';
      setGenerating(true);

      // The reveal's onComplete finalizes the bubble: removes the caret,
      // sets the final text, appends the reasoning panel. Fires when the
      // backend has sent `done` AND the 10 cps timer has shown all the text.
      let pendingFinalize = null;  // {finalText, reasoning}
      const reveal = startInkReveal(
        bubble,
        () => scrollBottom(),
        () => {
          if (pendingFinalize) {
            const { finalText, reasoning } = pendingFinalize;
            pendingFinalize = null;
            finalizeWupiBubble(bubble, finalText, reasoning);
          }
        },
      );

      const channel = new Channel();
      channel.onmessage = (e) => {
        if (!e) return;
        if (e.type === 'chunk') {
          streamed += e.text || '';
          reveal.push(streamed);
        } else if (e.type === 'error') {
          reveal.destroy();
          setGenerating(false);
          // Replace the partial bubble with an error notice.
          bubble.remove();
          addErrorBubble(e.message || 'Generation failed.');
        } else if (e.type === 'done') {
          // Backend finished. Arm the finalize; the reveal fires onComplete
          // either synchronously (short reply, already drained) or when the
          // timer catches up to the final text.
          setGenerating(false);
          const finalText = e.final_text != null ? e.final_text : streamed;
          pendingFinalize = { finalText, reasoning: e.reasoning || '' };
          reveal.push(finalText, true);
        }
      };

      invoke('chat_send', { text, onEvent: channel })
        .catch((err) => {
          if (generating) {
            reveal.destroy();
            setGenerating(false);
            bubble.remove();
            addErrorBubble('Failed to send: ' + err);
          }
        });
    }

    sendBtn?.addEventListener('click', send);
    stopBtn?.addEventListener('click', () => {
      // If a reveal is still dripping, flush it so the user sees the full
      // text immediately on stop. The backend will then send done/error
      // which finalizes the bubble.
      if (activeReveal) activeReveal.flush();
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
    // get_intro IPC). The intro is UI-only: never sent to the model or archived.
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
    // Tear down an active ink-reveal timer when the chat window closes so it
    // doesn't keep ticking against a detached bubble node.
    windowCloseHooks.set('chat', () => {
      if (activeReveal) activeReveal.destroy();
    });
    loadIntro();
  })();