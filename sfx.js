/* =========================================================
   sfx.js  —  সিন্থেসাইজ করা খুব নরম সাউন্ড + ঘন কুয়াশা (fog) ব্যাকগ্রাউন্ড (কোনো অডিও ফাইল লাগে না)
   (নতুন ফাইল: app.py এর শেষে থাকা ছোট লোডার এটাকে ব্রাউজারের মূল পেজে বসায়)

   কনসোল থেকে টিউন করা যায়:
     __psSfx.play('success')   // যেকোনো সাউন্ড টেস্ট
     __psSfx.setLevel(1)       // 0=বন্ধ, 1=নিচু, 2=মাঝারি, 3=জোরে
   ========================================================= */
(function () {
  'use strict';
  var W = window, D = document;
  if (W.__psSfx) return;

  /* ---------- সেটিংস সংরক্ষণ ---------- */
  function lsGet(k, d) {
    try { var v = W.localStorage.getItem(k); return v === null ? d : v; } catch (e) { return d; }
  }
  function lsSet(k, v) { try { W.localStorage.setItem(k, v); } catch (e) {} }

  var LEVELS = [0, 0.35, 0.6, 1.0];
  var LEVEL_NAMES = ['বন্ধ', 'নিচু', 'মাঝারি', 'জোরে'];
  var level = parseInt(lsGet('ps_sfx_level_v2', '2'), 10);
  if (isNaN(level) || level < 0 || level > 3) level = 2;
  var ambOn = lsGet('ps_amb_on', '1') === '1';
  var lastSaveAt = 0;

  /* ---------- অডিও ইঞ্জিন ---------- */
  var ctx = null, master = null;

  function ensureCtx() {
    if (!ctx) {
      var AC = W.AudioContext || W.webkitAudioContext;
      if (!AC) return null;
      try { ctx = new AC(); } catch (e) { return null; }
      master = ctx.createGain();
      master.gain.value = LEVELS[level];
      /* উঁচু ফ্রিকোয়েন্সি কেটে দেওয়া: শব্দ গোল ও নরম থাকে */
      var lp = ctx.createBiquadFilter();
      lp.type = 'lowpass'; lp.frequency.value = 1900; lp.Q.value = 0.3;
      var comp = ctx.createDynamicsCompressor();
      /* খুব হালকা, অন্ধকার প্রতিধ্বনি (reverb-এর মতো) */
      var dly = ctx.createDelay(0.8); dly.delayTime.value = 0.21;
      var fbLp = ctx.createBiquadFilter(); fbLp.type = 'lowpass'; fbLp.frequency.value = 1100;
      var fb = ctx.createGain(); fb.gain.value = 0.28;
      var wet = ctx.createGain(); wet.gain.value = 0.14;
      master.connect(lp);
      lp.connect(comp);
      lp.connect(dly); dly.connect(fbLp); fbLp.connect(fb); fb.connect(dly); dly.connect(wet); wet.connect(comp);
      comp.connect(ctx.destination);
    }
    if (ctx.state === 'suspended') { try { ctx.resume(); } catch (e) {} }
    return ctx;
  }

  /* একটি নরম নোট: ধীরে ফুটে ওঠে, ধীরে মিলিয়ে যায়; সঙ্গে খুব ক্ষীণ উপরের সুর */
  function tone(f0, t0, dur, o) {
    o = o || {};
    var t = ctx.currentTime + t0;
    var v = o.vol || 0.05;
    var a = o.attack || 0.025;
    var parts = [[1, 1, dur], [2, 0.10, dur * 0.55]];
    for (var i = 0; i < parts.length; i++) {
      var osc = ctx.createOscillator();
      var g = ctx.createGain();
      var pv = v * parts[i][1], pd = parts[i][2];
      osc.type = 'sine';
      osc.frequency.setValueAtTime(f0 * parts[i][0], t);
      if (o.f1) osc.frequency.linearRampToValueAtTime(o.f1 * parts[i][0], t + pd);
      g.gain.setValueAtTime(0.0001, t);
      g.gain.exponentialRampToValueAtTime(pv, t + a);
      g.gain.exponentialRampToValueAtTime(0.0001, t + pd);
      osc.connect(g); g.connect(master);
      osc.start(t); osc.stop(t + pd + 0.05);
    }
  }

  /* ---------- সাউন্ড তালিকা (সবই খুব নরম, ছোট, শান্ত সুরের) ---------- */
  var SOUNDS = {
    tick:    function () { tone(523.25, 0, 0.16, { vol: 0.040 }); },
    soft:    function () { tone(659.25, 0, 0.12, { vol: 0.022 }); },
    on:      function () { tone(523.25, 0, 0.20, { vol: 0.040 }); tone(659.25, 0.10, 0.30, { vol: 0.040 }); },
    pop:     function () { tone(587.33, 0, 0.26, { vol: 0.045 }); },
    remove:  function () { tone(440, 0, 0.30, { vol: 0.040, f1: 392 }); },
    prev:    function () { tone(493.88, 0, 0.16, { vol: 0.035 }); },
    next:    function () { tone(587.33, 0, 0.16, { vol: 0.035 }); },
    zin:     function () { tone(523.25, 0, 0.14, { vol: 0.030 }); },
    zout:    function () { tone(440, 0, 0.14, { vol: 0.030 }); },
    save:    function () { tone(523.25, 0, 0.30, { vol: 0.045 }); tone(659.25, 0.12, 0.45, { vol: 0.045 }); },
    start:   function () { tone(392, 0, 0.30, { vol: 0.040 }); tone(523.25, 0.12, 0.40, { vol: 0.035 }); },
    upload:  function () { tone(440, 0, 0.26, { vol: 0.040 }); tone(587.33, 0.12, 0.40, { vol: 0.038 }); },
    success: function () {
      tone(523.25, 0.00, 0.50, { vol: 0.040 });
      tone(659.25, 0.16, 0.50, { vol: 0.040 });
      tone(783.99, 0.32, 0.80, { vol: 0.038 });
    },
    error:   function () {
      tone(349.23, 0, 0.40, { vol: 0.045 });
      tone(293.66, 0.20, 0.55, { vol: 0.045 });
    },
    warn:    function () {
      tone(392, 0, 0.26, { vol: 0.040 });
      tone(392, 0.24, 0.36, { vol: 0.040 });
    }
  };

  var lastPlayed = {};
  function play(name) {
    if (!LEVELS[level]) return;
    var fn = SOUNDS[name];
    if (!fn) return;
    var now = Date.now();
    if (now - (lastPlayed[name] || 0) < 45) return;
    lastPlayed[name] = now;
    if (!ensureCtx()) return;
    try { fn(); } catch (e) {}
  }

  function setLevel(n) {
    level = Math.max(0, Math.min(3, n | 0));
    lsSet('ps_sfx_level_v2', String(level));
    if (master) master.gain.value = LEVELS[level];
    updateControls();
  }

  /* প্রথম ক্লিক/কী-তে অডিও "আনলক" (ব্রাউজারের অটোপ্লে নিয়ম) */
  function unlock() { if (LEVELS[level]) ensureCtx(); }

  /* ---------- মূল পেজের ইভেন্ট ---------- */
  function onParentClick(e) {
    var t = e.target;
    if (!t || !t.closest) return;
    if (t.closest('.ps-ctl-btn')) return;
    if (t.closest('#ps-choose-pc, #ps-choose-phone')) { play('pop'); return; }
    if (t.closest('#ps-mode-switcher-fab')) { play('soft'); return; }
    if (t.closest('[data-testid="stDownloadButton"] button')) { play('save'); return; }
    var btn = t.closest('button');
    if (btn) {
      if (btn.disabled) return;
      var txt = (btn.innerText || '').trim();
      if (txt.indexOf('স্ক্যান') > -1) { play('start'); return; }
      play('tick');
      return;
    }
    if (t.closest('summary')) { play('soft'); return; }
    if (t.closest('[data-testid="stRadio"] label, [data-testid="stCheckbox"] label, [data-testid="stToggle"] label')) { play('soft'); return; }
    if (t.closest('[role="option"]')) { play('soft'); return; }
  }

  function onParentPointerUp(e) {
    var t = e.target;
    if (t && t.closest && t.closest('[data-testid="stSlider"]')) play('soft');
  }

  D.addEventListener('pointerdown', unlock, true);
  D.addEventListener('keydown', unlock, true);
  D.addEventListener('click', onParentClick, true);
  D.addEventListener('pointerup', onParentPointerUp, true);

  /* ---------- অবস্থা দেখে সাউন্ড: ফল, সতর্কবার্তা, আপলোড ---------- */
  var seenAlerts = (typeof WeakSet !== 'undefined') ? new WeakSet() : null;
  var lastBanner = null, lastUpload = null, ready = false;

  function norm(s) { return (s || '').replace(/\s+/g, ' ').trim(); }

  function scanState() {
    try {
      var alerts = D.querySelectorAll('[data-testid="stAlert"]');
      for (var i = 0; i < alerts.length; i++) {
        var el = alerts[i];
        if (seenAlerts) { if (seenAlerts.has(el)) continue; seenAlerts.add(el); }
        if (!ready) continue;
        var kEl = el.querySelector('[data-testid^="stAlertContent"]');
        var k = ((kEl && kEl.getAttribute('data-testid')) || '').toLowerCase();
        if (k.indexOf('error') > -1) play('error');
        else if (k.indexOf('warning') > -1) play('warn');
        else if (k.indexOf('success') > -1) play('success');
        else play('soft');
      }

      var banner = D.querySelector('.result-banner');
      var bKey = banner ? norm(banner.textContent) : '';
      if (bKey !== lastBanner) {
        if (ready && bKey && Date.now() - lastSaveAt > 1500) play('success');
        lastBanner = bKey;
      }

      var up = D.querySelector('.uploaded-file-card');
      var uKey = up ? norm(up.textContent) : '';
      if (uKey !== lastUpload) {
        if (ready && uKey) play('upload');
        lastUpload = uKey;
      }
    } catch (e) {}
  }

  var scanTimer = null;
  function scheduleScan() {
    if (scanTimer) return;
    scanTimer = setTimeout(function () { scanTimer = null; scanState(); }, 120);
  }
  try {
    new MutationObserver(scheduleScan).observe(D.body, { childList: true, subtree: true });
  } catch (e) {}
  setInterval(scanState, 1000);
  scanState();
  setTimeout(function () { ready = true; }, 900);

  /* ---------- বক্স এডিটর (iframe) এর ভেতরের সাউন্ড ---------- */
  function bindEditor(fr) {
    var d;
    try { d = fr.contentDocument; } catch (e) { return; }
    if (!d || !d.body || d.__psSfxBound) return;
    var ov = d.getElementById('overlay');
    var shell = d.getElementById('editorShell');
    if (!ov || !shell) return;
    if (shell.style.display === 'none') return;   /* গ্লোবাল-UI শেল, আসল এডিটর নয় */
    d.__psSfxBound = true;

    d.addEventListener('pointerdown', unlock, true);
    d.addEventListener('click', function (e) {
      var t = e.target;
      if (!t || !t.closest) return;
      var b = t.closest('button');
      if (!b) return;
      var id = b.id;
      if (id === 'fabAddBtn') return;              /* বক্স যোগ হলে নিচের observer 'pop' বাজাবে */
      if (id === 'saveBtn') { lastSaveAt = Date.now(); play('save'); return; }
      if (id === 'prevBtn') { play('prev'); return; }
      if (id === 'nextBtn') { play('next'); return; }
      if (id === 'zInBtn') { play('zin'); return; }
      if (id === 'zOutBtn') { play('zout'); return; }
      play('tick');
    }, true);

    d.addEventListener('pointerdown', function (e) {
      var t = e.target;
      if (!t || !t.closest) return;
      if (t.closest('.qbox') && !t.closest('.qbox-del')) play('soft');
    }, true);

    var pg = d.getElementById('pageIndicator');
    var lastN = ov.querySelectorAll('.qbox').length;
    var lastPg = pg ? pg.textContent : '';
    try {
      new MutationObserver(function () {
        var n = ov.querySelectorAll('.qbox').length;
        var p = pg ? pg.textContent : '';
        if (p === lastPg) {
          if (n > lastN) play('pop');
          else if (n < lastN) play('remove');
        }
        lastN = n; lastPg = p;
      }).observe(ov, { childList: true });
    } catch (e) {}
  }

  function bindAllEditors() {
    try {
      var frames = D.querySelectorAll('iframe');
      for (var i = 0; i < frames.length; i++) bindEditor(frames[i]);
    } catch (e) {}
  }
  setInterval(bindAllEditors, 700);
  bindAllEditors();

  /* ---------- ছোট কন্ট্রোল বাটন (নিচের বাম কোণ) ---------- */
  var SVG_OPEN = '<svg viewBox="0 0 24 24">';
  function sfxIcon() {
    var spk = '<polygon points="11 5 6 9 3 9 3 15 6 15 11 19 11 5"></polygon>';
    if (level === 0) return SVG_OPEN + spk + '<line x1="22" y1="9" x2="16" y2="15"></line><line x1="16" y1="9" x2="22" y2="15"></line></svg>';
    var w1 = '<path d="M15.5 9.5a4 4 0 0 1 0 5"></path>';
    var w2 = '<path d="M18.5 7a8 8 0 0 1 0 10"></path>';
    var w3 = '<path d="M21.5 4.5a12 12 0 0 1 0 15"></path>';
    return SVG_OPEN + spk + w1 + (level >= 2 ? w2 : '') + (level >= 3 ? w3 : '') + '</svg>';
  }
  var AMB_ICON = SVG_OPEN + '<polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>';

  function updateControls() {
    var s = D.getElementById('ps-sfx-ctl');
    if (s) {
      s.innerHTML = sfxIcon();
      s.classList.toggle('ps-off', level === 0);
      s.title = 'সাউন্ড: ' + LEVEL_NAMES[level] + ' (ক্লিক করে বদলান)';
    }
    var a = D.getElementById('ps-amb-ctl');
    if (a) {
      a.classList.toggle('ps-off', !ambOn);
      a.title = 'ব্যাকগ্রাউন্ড অ্যানিমেশন: ' + (ambOn ? 'চালু' : 'বন্ধ');
    }
    if (D.body) D.body.classList.toggle('ps-ambient-off', !ambOn);
  }

  function buildControls() {
    if (!D.body) return;
    if (!D.getElementById('ps-sfx-ctl')) {
      var s = D.createElement('button');
      s.id = 'ps-sfx-ctl'; s.className = 'ps-ctl-btn'; s.type = 'button';
      s.addEventListener('click', function () {
        setLevel((level + 1) % 4);
        if (level > 0) play('on');
      });
      D.body.appendChild(s);
    }
    if (!D.getElementById('ps-amb-ctl')) {
      var a = D.createElement('button');
      a.id = 'ps-amb-ctl'; a.className = 'ps-ctl-btn'; a.type = 'button';
      a.innerHTML = AMB_ICON;
      a.addEventListener('click', function () {
        ambOn = !ambOn;
        lsSet('ps_amb_on', ambOn ? '1' : '0');
        updateControls();
        play(ambOn ? 'on' : 'soft');
      });
      D.body.appendChild(a);
    }
    updateControls();
  }
  buildControls();
  setInterval(buildControls, 2000);   /* Streamlit রি-রেন্ডারে হারিয়ে গেলে ফিরিয়ে আনে */


  /* =======================================================
     ঘন কুয়াশা (fog) ব্যাকগ্রাউন্ড — canvas
     • অনেকগুলো বড় নরম মেঘ ধীরে ভাসে
     • আঙুল/মাউস নাড়ালে কাছের কুয়াশা সরে যায়, আর পিছনে হালকা কুয়াশার রেখা পড়ে
     ======================================================= */
  (function fogEngine() {
    if (D.getElementById('ps-fog')) return;
    var cv = D.createElement('canvas');
    cv.id = 'ps-fog';
    D.body.insertBefore(cv, D.body.firstChild);
    var c = cv.getContext('2d');
    if (!c) return;

    var SC = 0.25;                          /* কম রেজোলিউশনে আঁকা: দ্রুত ও আরও ঝাপসা */
    var w = 0, h = 0, phone = false;
    var reduce = false;
    try { reduce = W.matchMedia('(prefers-reduced-motion: reduce)').matches; } catch (e) {}

    var COLS = [[94, 158, 139], [70, 112, 168], [120, 96, 170], [94, 158, 139], [86, 140, 170]];
    var puffs = [], wisps = [];
    var N = 16;

    function rnd(a, b) { return a + Math.random() * (b - a); }

    function build() {
      puffs = [];
      for (var i = 0; i < N; i++) {
        puffs.push({
          hx: rnd(0, 1), hy: rnd(0, 1),            /* ঘরের অবস্থান (0..1) */
          ax: rnd(0.06, 0.16), ay: rnd(0.05, 0.14),/* ভেসে বেড়ানোর পরিসর */
          sx: rnd(0.025, 0.06), sy: rnd(0.02, 0.05),/* গতি */
          px: rnd(0, 6.28), py: rnd(0, 6.28),
          r: rnd(0.28, 0.5),                       /* ব্যাসার্ধ (পর্দার বড় দিকের অনুপাতে) */
          a: rnd(0.22, 0.36),                      /* ঘনত্ব */
          col: COLS[i % COLS.length],
          ox: 0, oy: 0, vx: 0, vy: 0               /* স্পর্শে সরে যাওয়া */
        });
      }
    }

    function resize() {
      phone = (D.body.getAttribute('data-device-mode') === 'phone');
      var sc = phone ? 0.2 : SC;
      w = Math.max(60, Math.round(W.innerWidth * sc));
      h = Math.max(60, Math.round(W.innerHeight * sc));
      cv.width = w; cv.height = h;
    }
    resize(); build();
    W.addEventListener('resize', resize);

    /* ---- স্পর্শ / মাউস ---- */
    var lastX = null, lastY = null;
    function move(x, y) {
      if (!ambOn) return;
      var nx = x / W.innerWidth, ny = y / W.innerHeight;
      if (lastX === null) { lastX = nx; lastY = ny; return; }
      var dx = nx - lastX, dy = ny - lastY;
      lastX = nx; lastY = ny;
      var sp = Math.sqrt(dx * dx + dy * dy);
      if (sp < 0.0005) return;
      var asp = W.innerWidth / W.innerHeight;
      var mx = Math.max(W.innerWidth, W.innerHeight);
      for (var i = 0; i < puffs.length; i++) {
        var p = puffs[i];
        var cx = p.hx + p.ax * Math.sin(p.px) + p.ox;
        var cy = p.hy + p.ay * Math.sin(p.py) + p.oy;
        var ddx = (nx - cx) * asp, ddy = (ny - cy);
        var d = Math.sqrt(ddx * ddx + ddy * ddy);
        var reach = p.r * mx / W.innerHeight * 0.9;
        if (d < reach) {
          var k = (1 - d / reach);
          p.vx += dx * k * 1.1 + (-ddx / (d + 0.05)) * k * sp * 0.25 / asp;
          p.vy += dy * k * 1.1 + (-ddy / (d + 0.05)) * k * sp * 0.25;
        }
      }
      /* আঙুলের পিছনে হালকা কুয়াশার রেখা */
      if (wisps.length < 36) {
        wisps.push({
          x: nx, y: ny, vx: dx * 0.4, vy: dy * 0.4,
          r: rnd(0.10, 0.18), life: 1,
          col: COLS[(Math.random() * COLS.length) | 0]
        });
      }
    }
    function onPtr(e) { move(e.clientX, e.clientY); }
    function onTouch(e) { var t = e.touches && e.touches[0]; if (t) move(t.clientX, t.clientY); }
    function onEnd() { lastX = null; lastY = null; }
    D.addEventListener('pointermove', onPtr, { passive: true, capture: true });
    D.addEventListener('touchstart', function (e) { lastX = null; onTouch(e); }, { passive: true, capture: true });
    D.addEventListener('touchmove', onTouch, { passive: true, capture: true });
    D.addEventListener('touchend', onEnd, { passive: true, capture: true });
    D.addEventListener('pointerleave', onEnd, true);

    /* ---- আঁকা ---- */
    function blob(x, y, r, col, a) {
      var g = c.createRadialGradient(x, y, 0, x, y, r);
      g.addColorStop(0, 'rgba(' + col[0] + ',' + col[1] + ',' + col[2] + ',' + a + ')');
      g.addColorStop(0.55, 'rgba(' + col[0] + ',' + col[1] + ',' + col[2] + ',' + (a * 0.45) + ')');
      g.addColorStop(1, 'rgba(' + col[0] + ',' + col[1] + ',' + col[2] + ',0)');
      c.fillStyle = g;
      c.fillRect(x - r, y - r, r * 2, r * 2);
    }

    var t0 = Date.now(), lastDraw = 0;
    function frame(now) {
      W.requestAnimationFrame(frame);
      if (!ambOn || D.hidden) return;
      var gap = phone ? 33 : 16;
      if (now - lastDraw < gap) return;
      lastDraw = now;

      var t = reduce ? 0 : (Date.now() - t0) / 1000;
      var mx = Math.max(w, h);
      c.clearRect(0, 0, w, h);

      var cnt = phone ? Math.min(11, puffs.length) : puffs.length;
      for (var i = 0; i < cnt; i++) {
        var p = puffs[i];
        /* স্প্রিং: সরে গিয়ে ধীরে ফিরে আসে (নরম, দোলাহীন) */
        p.vx += -p.ox * 0.0032; p.vy += -p.oy * 0.0032;
        p.vx *= 0.955; p.vy *= 0.955;
        p.ox += p.vx; p.oy += p.vy;
        var lim = 0.35;
        if (p.ox > lim) p.ox = lim; else if (p.ox < -lim) p.ox = -lim;
        if (p.oy > lim) p.oy = lim; else if (p.oy < -lim) p.oy = -lim;

        var x = (p.hx + p.ax * Math.sin(t * p.sx * 6.283 + p.px) + p.ox) * w;
        var y = (p.hy + p.ay * Math.sin(t * p.sy * 6.283 + p.py) + p.oy) * h;
        var br = 1 + 0.07 * Math.sin(t * 0.18 + p.px);          /* ধীরে "শ্বাস" */
        blob(x, y, p.r * mx * br, p.col, p.a);
      }

      for (var j = wisps.length - 1; j >= 0; j--) {
        var q = wisps[j];
        q.x += q.vx; q.y += q.vy;
        q.vx *= 0.97; q.vy *= 0.97;
        q.life -= 0.010;
        if (q.life <= 0) { wisps.splice(j, 1); continue; }
        blob(q.x * w, q.y * h, q.r * mx * (1.4 - q.life * 0.4), q.col, 0.20 * q.life);
      }
    }
    W.requestAnimationFrame(frame);
  })();

  W.__psSfx = { play: play, setLevel: setLevel, sounds: Object.keys(SOUNDS) };
})();
