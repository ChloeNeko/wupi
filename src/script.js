/* ════════════════════════════════════════════════════════════════
   WUPI OS — "Aurora Core" shell logic
   Pure, import-free classic script (runs from file:// and Vite alike).
   ─── Prime Directive §1B: peak efficiency, pristine data ───
     · rAF + delta-clamped stepping
     · object-pooled petals (no per-frame allocation)
     · rAF paused on visibilitychange / reduced-motion
     · GPU-only transforms (no layout-thrashing)
     · single-write clock, debounced resize
   No Tauri dependency in this shell. The preserved main.ts (chat +
   memory + schema glue) stays dormant, unreferenced, for re-wiring.
   ════════════════════════════════════════════════════════════════ */
(function () {
  "use strict";

  const prefersReducedMotion = window.matchMedia(
    "(prefers-reduced-motion: reduce)"
  ).matches;

  /* ──────────────────────────────────────────────────────────────
     1. Sakura petals — drifting canvas particle system
     ────────────────────────────────────────────────────────────── */
  const PETAL_COUNT = prefersReducedMotion ? 0 : 32;
  const PETAL_COLORS = [
    "rgba(255, 194, 220, 0.78)", // pink
    "rgba(255, 175, 210, 0.70)", // deeper pink
    "rgba(240, 210, 245, 0.66)", // lilac-white
    "rgba(217, 170, 230, 0.62)", // soft magenta
  ];

  const canvas = document.getElementById("petals");
  const ctx = canvas.getContext("2d", { alpha: true });

  let dpr = Math.min(window.devicePixelRatio || 1, 2); // cap at 2x for perf
  let vw = 0,
    vh = 0;

  // Object pool: petals are reused (wrap-to-top), never GC'd mid-loop.
  const petals = [];

  function makePetal(top) {
    return {
      x: Math.random() * vw,
      y: top ? Math.random() * vh : -20 - Math.random() * vh * 0.5,
      size: 7 + Math.random() * 11,
      speed: 0.35 + Math.random() * 0.7, // px per frame at 60fps baseline
      sway: 0.6 + Math.random() * 1.4, // horizontal sway amplitude
      swaySpeed: 0.0008 + Math.random() * 0.0018,
      phase: Math.random() * Math.PI * 2,
      rot: Math.random() * Math.PI * 2,
      rotSpeed: (Math.random() - 0.5) * 0.02,
      color: PETAL_COLORS[(Math.random() * PETAL_COLORS.length) | 0],
    };
  }

  function resize() {
    vw = window.innerWidth;
    vh = window.innerHeight;
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.floor(vw * dpr);
    canvas.height = Math.floor(vh * dpr);
    canvas.style.width = vw + "px";
    canvas.style.height = vh + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  // A single petal shape: two arcs forming the cherry-blossom notch.
  function drawPetal(p) {
    const s = p.size;
    ctx.save();
    ctx.translate(p.x, p.y);
    ctx.rotate(p.rot);
    ctx.globalAlpha = 1;
    ctx.fillStyle = p.color;
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.bezierCurveTo(s * 0.5, -s * 0.2, s * 0.5, -s, 0, -s);
    ctx.bezierCurveTo(-s * 0.5, -s, -s * 0.5, -s * 0.2, 0, 0);
    ctx.fill();
    // subtle notch at the tip
    ctx.fillStyle = "rgba(0,0,0,0.10)";
    ctx.beginPath();
    ctx.ellipse(0, -s * 0.82, s * 0.08, s * 0.18, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  let lastT = performance.now();
  let running = !prefersReducedMotion;

  function frame(now) {
    if (!running) return;
    // delta-clamp: huge gaps (tab switch) don't fast-forward the swarm
    const dt = Math.min((now - lastT) / (1000 / 60), 3);
    lastT = now;

    ctx.clearRect(0, 0, vw, vh);

    for (let i = 0; i < petals.length; i++) {
      const p = petals[i];
      p.y += p.speed * dt;
      p.phase += p.swaySpeed * dt * 16;
      p.x += Math.sin(p.phase) * p.sway * dt;
      p.rot += p.rotSpeed * dt;

      if (p.y > vh + 20) {
        // recycle to top — keep the pool stable, no allocation
        p.y = -20;
        p.x = Math.random() * vw;
      }
      drawPetal(p);
    }
    requestAnimationFrame(frame);
  }

  function initPetals() {
    resize();
    for (let i = 0; i < PETAL_COUNT; i++) petals.push(makePetal(true));
    if (running) requestAnimationFrame(frame);
  }

  // Pause rAF when the tab/window is hidden — no GPU burn while idle.
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      running = false;
    } else if (!prefersReducedMotion) {
      running = true;
      lastT = performance.now();
      requestAnimationFrame(frame);
    }
  });

  // Debounced resize: trailing-edge so a drag-resize doesn't spam reallocation.
  let resizeTimer = 0;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(resize, 150);
  });

  /* ──────────────────────────────────────────────────────────────
     2. Clock — single DOM write per second, stacked time + date
     ────────────────────────────────────────────────────────────── */
  const timeEl = document.getElementById("clock-time");
  const dateEl = document.getElementById("clock-date");

  function tick() {
    const now = new Date();
    timeEl.textContent = now.toLocaleTimeString([], {
      hour: "numeric",
      minute: "2-digit",
    });
    dateEl.textContent = now.toLocaleDateString([], {
      weekday: "long",
      month: "long",
      day: "numeric",
    });
  }

  /* ──────────────────────────────────────────────────────────────
     3. Dock — auto-hide reveal + macOS fish-eye magnification
     ────────────────────────────────────────────────────────────── */
  const dock = document.getElementById("dock");
  const dockZone = document.getElementById("dock-zone");
  const dockApps = Array.from(document.querySelectorAll(".dock-app"));

  const MAG_MAX = 1.5; // peak scale at cursor
  const MAG_RANGE = 110; // px radius of influence
  let hideTimer = null;

  function reveal() {
    clearTimeout(hideTimer);
    dock.classList.add("revealed");
  }
  function scheduleHide() {
    clearTimeout(hideTimer);
    hideTimer = setTimeout(() => dock.classList.remove("revealed"), 350);
  }

  dockZone.addEventListener("mouseenter", reveal);
  dockZone.addEventListener("mouseleave", scheduleHide);
  dock.addEventListener("mouseenter", reveal);
  dock.addEventListener("mouseleave", scheduleHide);

  // Fish-eye magnifier: set per-icon --scale from cursor distance.
  // Runs on rAF-coalesced mousemove to avoid layout thrash.
  let pendingMouseX = null;
  let magnifyScheduled = false;

  function applyMagnify() {
    magnifyScheduled = false;
    const mx = pendingMouseX;
    if (mx === null) return;
    for (const app of dockApps) {
      const r = app.getBoundingClientRect();
      const center = r.left + r.width / 2;
      const dist = Math.abs(mx - center);
      let scale = 1;
      if (dist < MAG_RANGE) {
        // smooth falloff (cosine) so the fish-eye is continuous
        const t = 1 - dist / MAG_RANGE;
        scale = 1 + (MAG_MAX - 1) * (0.5 - 0.5 * Math.cos(Math.PI * t));
      }
      app.style.setProperty("--scale", scale.toFixed(3));
    }
  }

  dock.addEventListener("mousemove", (e) => {
    pendingMouseX = e.clientX;
    if (!magnifyScheduled) {
      magnifyScheduled = true;
      requestAnimationFrame(applyMagnify);
    }
  });
  // reset scales when the cursor leaves the dock
  dock.addEventListener("mouseleave", () => {
    for (const app of dockApps) app.style.setProperty("--scale", "1");
  });

  /* ──────────────────────────────────────────────────────────────
     4. Floating window manager — drag + persist position
     ────────────────────────────────────────────────────────────── */
  const windowsEl = document.getElementById("windows");
  const tmpl = document.getElementById("window-template");
  const STORAGE_KEY = "wupi.windowPositions";
  const PLACEHOLDER_BODY =
    '<p class="window-placeholder">This is a placeholder window. The ' +
    "app's real interface will be wired in here.</p>";

  let positionStore = loadPositions();
  // cascade offset so successive new windows don't stack perfectly
  let cascadeIndex = 0;

  function loadPositions() {
    try {
      return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {};
    } catch {
      return {};
    }
  }
  function savePositions() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(positionStore));
    } catch {
      /* quota / private mode — non-fatal */
    }
  }

  /**
   * Spawn a window from the hidden template.
   * @param {object} opts
   * @param {string} opts.app   - app id (used as the persistence key)
   * @param {string} [opts.title]
   * @param {string} [opts.bodyHTML]
   */
  function openWindow({ app, title, bodyHTML }) {
    const el = tmpl.content.firstElementChild.cloneNode(true);
    const saved = positionStore[app];

    // title
    const titleEl = el.querySelector(".window-title");
    if (title) titleEl.textContent = title;

    // body
    el.querySelector(".window-body").innerHTML = bodyHTML || PLACEHOLDER_BODY;

    // position: restore saved coords, else cascade-default in viewport
    let top, left;
    if (saved) {
      top = saved.top;
      left = saved.left;
    } else {
      const offset = (cascadeIndex % 6) * 32;
      left = Math.max(40, window.innerWidth / 2 - 230 + offset);
      top = Math.max(70, window.innerHeight / 2 - 160 + offset);
      cascadeIndex++;
    }
    // clamp so a restored window never spawns fully off-screen
    const rect0 = clampPosition(left, top, 460, 320);
    el.style.left = rect0.left + "px";
    el.style.top = rect0.top + "px";

    windowsEl.appendChild(el);
    makeDraggable(el, app);
    el.querySelector(".window-close").addEventListener("click", () => el.remove());
    return el;
  }

  function clampPosition(left, top, w, h) {
    const maxX = window.innerWidth - w;
    const maxY = window.innerHeight - h;
    return {
      left: Math.min(Math.max(8, left), Math.max(8, maxX)),
      top: Math.min(Math.max(54, top), Math.max(54, maxY)), // keep below topbar
    };
  }

  /**
   * Pointer-event dragging by the title bar. Clamps to the viewport so the
   * title bar can never be dragged off-screen (always grabbable). Persists
   * the final {top,left} to localStorage on pointerup.
   */
  function makeDraggable(el, app) {
    const handle = el.querySelector(".window-titlebar");
    let dragging = false;
    let startX = 0,
      startY = 0,
      originLeft = 0,
      originTop = 0;

    handle.addEventListener("pointerdown", (e) => {
      // only primary button / touch
      if (e.button !== undefined && e.button !== 0) return;
      dragging = true;
      startX = e.clientX;
      startY = e.clientY;
      originLeft = el.offsetLeft;
      originTop = el.offsetTop;
      el.classList.add("dragging");
      handle.setPointerCapture(e.pointerId);
      e.preventDefault();
    });

    handle.addEventListener("pointermove", (e) => {
      if (!dragging) return;
      const dx = e.clientX - startX;
      const dy = e.clientY - startY;
      const rect = el.getBoundingClientRect();
      const pos = clampPosition(
        originLeft + dx,
        originTop + dy,
        rect.width,
        rect.height
      );
      el.style.left = pos.left + "px";
      el.style.top = pos.top + "px";
    });

    function endDrag(e) {
      if (!dragging) return;
      dragging = false;
      el.classList.remove("dragging");
      try {
        handle.releasePointerCapture(e.pointerId);
      } catch {
        /* pointerId may already be released */
      }
      // persist final position
      positionStore[app] = {
        top: el.offsetTop,
        left: el.offsetLeft,
      };
      savePositions();
    }
    handle.addEventListener("pointerup", endDrag);
    handle.addEventListener("pointercancel", endDrag);
  }

  // Dock clicks spawn placeholder windows so the manager is testable now.
  // (The real app surfaces get wired in later — this is scaffolding.)
  document.querySelectorAll(".dock-app").forEach((btn) => {
    btn.addEventListener("click", () => {
      const app = btn.dataset.app;
      const labels = {
        wupi: "WUPI",
        terminal: "Terminal",
        codex: "Codex",
        allapps: "All Apps",
      };
      openWindow({ app, title: labels[app] || app });
    });
  });

  /* ──────────────────────────────────────────────────────────────
     Boot
     ────────────────────────────────────────────────────────────── */
  initPetals();
  tick();
  // align the clock to the wall-clock second boundary so it never drifts
  setInterval(tick, 1000);
})();
