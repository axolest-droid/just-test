/* =========================================================
   sfx.js  —  সিন্থেসাইজ করা নরম সাউন্ড এফেক্ট (কোনো অডিও ফাইল লাগে না)
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

  var LEVELS = [0, 0.3, 0.6, 1.0];
  var LEVEL_NAMES = ['বন্ধ', 'নিচু', 'মাঝারি', 'জোরে'];
  var level = parseInt(lsGet('ps_sfx_level', '2'), 10);
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
      var lp = ctx.createBiquadFilter();
      lp.type = 'lowpass'; lp.frequency.value = 3400; lp.Q.value = 0.5;
      var comp = ctx.createDynamicsCompressor();
      var dly = ctx.createDelay(0.5); dly.delayTime.value = 0.13;
      var fb = ctx.createGain(); fb.gain.value = 0.26;
      var wet = ctx.createGain(); wet.gain.value = 0.20;
      master.connect(lp);
      lp.connect(comp);
      lp.connect(dly); dly.connect(fb); fb.connect(dly); dly.connect(wet); wet.connect(comp);
      comp.connect(ctx.destination);
    }
    if (ctx.state === 'suspended') { try { ctx.resume(); } catch (e) {} }
    return ctx;
  }

  function tone(f0, t0, dur, o) {
    o = o || {};
    var osc = ctx.createOscillator();
    var g = ctx.createGain();
    var t = ctx.currentTime + t0;
    var v = o.vol || 0.1;
    var a = o.attack || 0.008;
    osc.type = o.type || 'sine';
    osc.frequency.setValueAtTime(f0, t);
    if (o.f1) osc.frequency.exponentialRampToValueAtTime(o.f1, t + dur * 0.9);
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(v, t + a);
    g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    osc.connect(g); g.connect(master);
    osc.start(t); osc.stop(t + dur + 0.03);
  }

  /* ---------- সাউন্ড তালিকা (সবই নরম, ছোট) ---------- */
  var SOUNDS = {
    tick:    function () { tone(560, 0, 0.07, { vol: 0.10, f1: 520 }); tone(1120, 0, 0.05, { vol: 0.025 }); },
    soft:    function () { tone(700, 0, 0.045, { vol: 0.055 }); },
    on:      function () { tone(523.25, 0, 0.09, { vol: 0.08 }); tone(784, 0.07, 0.14, { vol: 0.08 }); },
    pop:     function () { tone(330, 0, 0.12, { vol: 0.13, f1: 640 }); tone(990, 0.02, 0.06, { vol: 0.02 }); },
    remove:  function () { tone(480, 0, 0.16, { vol: 0.10, f1: 210 }); },
    prev:    function () { tone(500, 0, 0.08, { vol: 0.09, f1: 440 }); },
    next:    function () { tone(560, 0, 0.08, { vol: 0.09, f1: 640 }); },
    zin:     function () { tone(520, 0, 0.07, { vol: 0.08, f1: 620 }); },
    zout:    function () { tone(620, 0, 0.07, { vol: 0.08, f1: 520 }); },
    save:    function () { tone(659.25, 0, 0.16, { vol: 0.10 }); tone(880, 0.09, 0.26, { vol: 0.10 }); },
    start:   function () { tone(392, 0, 0.18, { vol: 0.08, f1: 523 }); tone(523.25, 0.06, 0.22, { vol: 0.06 }); },
    upload:  function () { tone(440, 0, 0.14, { vol: 0.09, f1: 587 }); tone(587.33, 0.10, 0.22, { vol: 0.08 }); },
    success: function () {
      tone(523.25, 0.00, 0.30, { vol: 0.09 });
      tone(659.25, 0.11, 0.30, { vol: 0.09 });
      tone(783.99, 0.22, 0.34, { vol: 0.09 });
      tone(1046.5, 0.33, 0.50, { vol: 0.07 });
    },
    error:   function () {
      tone(330, 0, 0.20, { type: 'triangle', vol: 0.12, f1: 300 });
      tone(247, 0.14, 0.32, { type: 'triangle', vol: 0.12, f1: 220 });
    },
    warn:    function () {
      tone(392, 0, 0.14, { type: 'triangle', vol: 0.09 });
      tone(392, 0.18, 0.20, { type: 'triangle', vol: 0.09 });
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
    lsSet('ps_sfx_level', String(level));
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

  W.__psSfx = { play: play, setLevel: setLevel, sounds: Object.keys(SOUNDS) };
})();
