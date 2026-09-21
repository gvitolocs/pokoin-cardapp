// Scan Connect phone mode for scan.pokoin.com/connect.
// The phone is a camera: it pairs with a 4-digit code (or QR), turns each
// physical card into one scan event, and uploads it to the Pokoin dashboard.
// It never edits metadata. Spec: pokoin-web docs/SCAN_CONNECT.md.
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.ScanConnect = api;
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  // Same thresholds as CardVault api/_scan_connect.js (server decides; the phone
  // only uses them to know when a card is "done" and whether to attach a photo).
  const MATCH_SCORE = 0.80;
  const MATCH_MARGIN = 0.08;
  const DETECT_HOLD_MS = 1200;
  const CLEAR_FRAMES = 2;
  const SWAP_FRAMES = 2;
  const HEARTBEAT_MS = 3000;
  const STORE_KEY = 'pokoin.scanConnect.v1';
  const OUTBOX_KEY = 'pokoin.scanConnect.outbox.v1';

  function topAndMargin(data) {
    const hits = Array.isArray(data && data.hits) ? data.hits.filter((h) => h && h.public_id) : [];
    const top = hits[0] || null;
    const score = top ? Number(top.score) || 0 : 0;
    const second = hits.find((h) => h && top && String(h.public_id) !== String(top.public_id));
    const margin = second ? score - (Number(second.score) || 0) : 1;
    return { hits, top, score, margin };
  }

  function confident(summary) {
    return Boolean(summary.top) && summary.score >= MATCH_SCORE && summary.margin >= MATCH_MARGIN;
  }

  /**
   * One physical card → one event. Fires on a confident match, or after a
   * card has been in view ~1.2 s without one. Re-arms when the card leaves
   * the frame, or when a *different* confident card replaces it for two frames.
   */
  function createCaptureGate(now = () => Date.now()) {
    let armed = true;
    let seenSince = null;
    let best = null;
    let emittedId = '';
    let clearFrames = 0;
    let swapId = '';
    let swapFrames = 0;

    function emit(result) {
      armed = false;
      seenSince = null;
      best = null;
      clearFrames = 0;
      swapFrames = 0;
      emittedId = result.summary.top ? String(result.summary.top.public_id) : '';
      return result;
    }

    return {
      /** Feed one identify response; returns the result to send, or null. */
      push(data, context) {
        const summary = topAndMargin(data);
        const hasCard = Array.isArray(data && data.boxes) && data.boxes.length > 0;
        const t = now();
        const result = { data, summary, context };
        if (armed) {
          if (confident(summary)) return emit(result);
          if (!hasCard) {
            seenSince = null;
            best = null;
            return null;
          }
          if (seenSince === null) seenSince = t;
          if (!best || summary.score > best.summary.score) best = result;
          if (t - seenSince >= DETECT_HOLD_MS) return emit(best);
          return null;
        }
        if (!hasCard) {
          clearFrames += 1;
          if (clearFrames >= CLEAR_FRAMES) {
            armed = true;
            seenSince = null;
          }
          swapFrames = 0;
          return null;
        }
        clearFrames = 0;
        if (confident(summary) && String(summary.top.public_id) !== emittedId) {
          const id = String(summary.top.public_id);
          swapFrames = id === swapId ? swapFrames + 1 : 1;
          swapId = id;
          if (swapFrames >= SWAP_FRAMES) return emit(result);
        } else {
          swapFrames = 0;
        }
        return null;
      },
      /** Manual shutter: send what is in view now. */
      force(data, context) {
        return emit({ data, summary: topAndMargin(data), context });
      },
      get armed() {
        return armed;
      },
    };
  }

  /** Keep the lowest-RTT sample: offset = server − phone at the RTT midpoint. */
  function createClock() {
    let best = null;
    return {
      sample(sentAt, receivedAt, serverTime) {
        const rtt = receivedAt - sentAt;
        if (!Number.isFinite(rtt) || rtt < 0 || !Number.isFinite(Number(serverTime))) return;
        const offset = Math.round(Number(serverTime) - (sentAt + rtt / 2));
        // Prefer the tightest round trip; refresh at least once a minute for drift.
        if (!best || rtt <= best.rtt || receivedAt - best.at > 60_000) best = { rtt, offset, at: receivedAt };
      },
      get offset() {
        return best ? best.offset : null;
      },
    };
  }

  /** `#c=4827&k=<secret>` (or the same as query) from the dashboard QR → { pin, qr }. */
  function pairingFromHash(hash) {
    const params = new URLSearchParams(String(hash || '').replace(/^#/, ''));
    const pin = /^[0-9]{4}$/.test(params.get('c') || '') ? params.get('c') : '';
    const qr = /^[A-Za-z0-9_-]{20,64}$/.test(params.get('k') || '') ? params.get('k') : '';
    return { pin, qr };
  }

  /** Prefer the fragment (never logged); fall back to query for "Open in Safari" hops. */
  function pairingFromLocation(loc) {
    const fromHash = pairingFromHash(loc && loc.hash);
    if (fromHash.pin || fromHash.qr) return fromHash;
    return pairingFromHash(loc && loc.search);
  }

  function connectUrlWithPair(originPath, pair) {
    const params = new URLSearchParams();
    if (pair && pair.pin) params.set('c', pair.pin);
    if (pair && pair.qr) params.set('k', pair.qr);
    const q = params.toString();
    return q ? `${originPath}?${q}` : originPath;
  }

  /**
   * Camera / in-app browsers keep same-origin <a href> navigations inside the
   * mini browser. Hop into Chrome (same /connect URL) when we can. A future
   * Pokoin scan app can replace this with its own URL scheme.
   */
  /** Chrome deep-link for the same https URL (Camera / in-app browsers). */
  function chromeUrlFor(absUrl, ua) {
    const agent = String(ua || '');
    if (/iPhone|iPad|iPod/i.test(agent)) {
      if (/^https:\/\//i.test(absUrl)) return absUrl.replace(/^https:\/\//i, 'googlechromes://');
      if (/^http:\/\//i.test(absUrl)) return absUrl.replace(/^http:\/\//i, 'googlechrome://');
    }
    if (/Android/i.test(agent) && /^https?:\/\//i.test(absUrl)) {
      const hostPath = absUrl.replace(/^https?:\/\//i, '');
      return `intent://${hostPath}#Intent;scheme=https;package=com.android.chrome;action=android.intent.action.VIEW;end`;
    }
    return '';
  }

  function openInChrome(win, absUrl) {
    const ua = String((win.navigator && win.navigator.userAgent) || '');
    const chrome = chromeUrlFor(absUrl, ua);
    try {
      if (chrome) {
        // Prefer a real navigation — Camera's mini browser often ignores
        // location.href custom schemes even inside a click handler.
        win.location.assign(chrome);
        return;
      }
    } catch (_) {
      /* fall through */
    }
    try {
      const opened = win.open(absUrl, '_blank', 'noopener,noreferrer');
      if (opened) return;
    } catch (_) {
      /* fall through */
    }
    win.location.assign(absUrl);
  }

  /** @deprecated use openInChrome — kept as the exported name for older callers. */
  function openInSystemBrowser(win, absUrl) {
    return openInChrome(win, absUrl);
  }

  function uuid() {
    if (typeof crypto !== 'undefined' && crypto.randomUUID) return crypto.randomUUID();
    const b = new Uint8Array(16);
    crypto.getRandomValues(b);
    b[6] = (b[6] & 0x0f) | 0x40;
    b[8] = (b[8] & 0x3f) | 0x80;
    const h = Array.from(b, (x) => x.toString(16).padStart(2, '0')).join('');
    return `${h.slice(0, 8)}-${h.slice(8, 12)}-${h.slice(12, 16)}-${h.slice(16, 20)}-${h.slice(20)}`;
  }

  /** Build the immutable event once; retries resend the same object. */
  function buildEvent(result, { sequence, clockOffsetMs, image }) {
    const ctx = result.context || {};
    const hits = result.summary.hits.slice(0, 8).map((h) => ({
      public_id: String(h.public_id),
      score: Number(h.score) || 0,
      name: String(h.name || '').slice(0, 120),
    }));
    return {
      scanEventId: uuid(),
      clientSequence: sequence,
      capturedAt: ctx.capturedAt || Date.now(),
      clockOffsetMs: clockOffsetMs == null ? null : clockOffsetMs,
      recognition: { catalog: String((result.data && result.data.catalog) || ''), hits },
      image: image || undefined,
      timings: {
        captureToRequestMs: ctx.requestAt && ctx.capturedAt ? Math.max(0, ctx.requestAt - ctx.capturedAt) : undefined,
        identifyMs: ctx.respondedAt && ctx.requestAt ? Math.max(0, ctx.respondedAt - ctx.requestAt) : undefined,
      },
    };
  }

  /**
   * FIFO outbox, one request in flight. Events survive reloads (without the
   * photo) so a dropped connection never loses a physical scan.
   */
  function createOutbox({ send, storage, onChange = () => {}, sleep = (ms) => new Promise((r) => setTimeout(r, ms)) }) {
    let queue = [];
    let running = false;
    let halted = false;
    try {
      queue = JSON.parse((storage && storage.getItem(OUTBOX_KEY)) || '[]');
    } catch (_) {
      queue = [];
    }
    const persist = () => {
      try {
        storage && storage.setItem(OUTBOX_KEY, JSON.stringify(queue.map((e) => ({ ...e, image: undefined }))));
      } catch (_) {
        // storage full / private mode: in-memory queue still works
      }
      onChange(queue.length);
    };
    async function pump() {
      if (running) return;
      running = true;
      let attempt = 0;
      try {
        while (queue.length && !halted) {
          const event = queue[0];
          let outcome;
          try {
            outcome = await send(event, attempt);
          } catch (_) {
            outcome = { retry: true };
          }
          if (outcome && outcome.ok) {
            queue.shift();
            attempt = 0;
            persist();
          } else if (outcome && outcome.drop) {
            queue.shift();
            persist();
          } else if (outcome && outcome.halt) {
            halted = true;
          } else {
            attempt += 1;
            await sleep(Math.min(8000, 300 * 2 ** Math.min(attempt, 5)));
          }
        }
      } finally {
        running = false;
      }
    }
    return {
      add(event) {
        queue.push(event);
        persist();
        pump();
      },
      resume() {
        halted = false;
        pump();
      },
      clear() {
        queue = [];
        persist();
      },
      get size() {
        return queue.length;
      },
    };
  }

  // ------------------------------------------------------------------ browser UI

  function start(win) {
    const doc = win.document;
    const API = String(win.SCAN_CONNECT_API || 'https://api.pokoin.com').replace(/\/$/, '');
    const storage = win.localStorage;
    const clock = createClock();
    const gate = createCaptureGate();
    let state = { token: '', sessionId: '' };
    try {
      state = { ...state, ...JSON.parse(storage.getItem(STORE_KEY) || '{}') };
    } catch (_) {
      // ignore
    }
    let sequence = Number(storage.getItem(`${STORE_KEY}.seq`) || 0);
    let paused = false;
    let sent = 0;
    let lastPreview = null;
    let heartbeatTimer = 0;
    let cameraStarted = false;

    const el = (tag, attrs = {}, children = []) => {
      const node = doc.createElement(tag);
      for (const [k, v] of Object.entries(attrs)) {
        if (k === 'class') node.className = v;
        else if (k === 'text') node.textContent = v;
        else node.setAttribute(k, v);
      }
      for (const child of children) node.append(child);
      return node;
    };

    // Keypad panel
    const slots = [0, 1, 2, 3].map(() => el('span', { class: 'sc-slot', text: '_' }));
    const status = el('p', { class: 'sc-status', role: 'status', 'aria-live': 'polite' });
    const keys = el('div', { class: 'sc-keys' });
    let digits = '';
    const panel = el('div', { class: 'sc-panel', id: 'scanConnectPanel' }, [
      el('img', { src: '/static/pokoin-icon.png', alt: '', class: 'sc-logo' }),
      el('h1', { text: 'Connect to dashboard' }),
      el('p', { class: 'sc-hint', text: 'Type the 4 digits shown on pokoin.com → Inventory → Scan' }),
      el('div', { class: 'sc-slots', 'aria-label': 'Pairing code' }, slots),
      status,
      keys,
    ]);
    const hidden = el('input', { class: 'sc-hidden', inputmode: 'numeric', autocomplete: 'one-time-code', 'aria-label': 'Pairing code', maxlength: '4' });
    panel.append(hidden);

    // Connected bar + shutter
    const barText = el('span', { class: 'sc-bar-text', text: 'Connected' });
    const barPile = el('span', { class: 'sc-bar-pile' });
    const leaveBtn = el('button', { type: 'button', class: 'sc-leave', text: 'Disconnect' });
    const bar = el('div', { class: 'sc-bar', hidden: '' }, [el('span', { class: 'sc-dot' }), barText, barPile, leaveBtn]);
    if (!doc.getElementById('sc-stack-full-style')) {
      const style = doc.createElement('style');
      style.id = 'sc-stack-full-style';
      style.textContent = `
.sc-bar.is-stack-full {
  animation: sc-stack-pulse 0.9s ease-in-out 2;
  box-shadow: 0 0 0 2px rgba(240, 180, 41, 0.85);
}
.sc-bar.is-stack-full .sc-dot { background: #f0b429; }
.sc-bar.is-stack-full .sc-bar-pile { color: #f0b429; font-weight: 600; }
@keyframes sc-stack-pulse {
  0%, 100% { filter: brightness(1); }
  50% { filter: brightness(1.25); }
}`;
      doc.head.appendChild(style);
    }
    let stackFullTimer = null;
    function flashStackFull() {
      bar.classList.add('is-stack-full');
      if (stackFullTimer) clearTimeout(stackFullTimer);
      stackFullTimer = setTimeout(() => bar.classList.remove('is-stack-full'), 2800);
      try {
        if (navigator.vibrate) navigator.vibrate([40, 40, 40]);
      } catch (_) {}
    }
    const flash = el('div', { class: 'sc-flash', role: 'status', 'aria-live': 'polite', hidden: '' });
    const shutter = el('button', { type: 'button', class: 'sc-shutter', hidden: '', 'aria-label': 'Add the card in view' }, [el('span', { text: 'Add card' })]);

    function renderSlots() {
      slots.forEach((slot, i) => {
        slot.textContent = digits[i] || '_';
        slot.classList.toggle('filled', Boolean(digits[i]));
      });
    }

    function keypress(k) {
      if (k === 'del') digits = digits.slice(0, -1);
      else if (/^[0-9]$/.test(k) && digits.length < 4) digits += k;
      renderSlots();
      if (digits.length === 4) pair({ pin: digits });
    }

    ['1', '2', '3', '4', '5', '6', '7', '8', '9', '', '0', 'del'].forEach((k) => {
      const b = el('button', { type: 'button', class: 'sc-key', text: k === 'del' ? '⌫' : k });
      if (!k) b.disabled = true;
      b.addEventListener('click', () => keypress(k));
      keys.append(b);
    });
    hidden.addEventListener('input', () => {
      digits = hidden.value.replace(/\D/g, '').slice(0, 4);
      hidden.value = digits;
      renderSlots();
      if (digits.length === 4) pair({ pin: digits });
    });
    doc.addEventListener('keydown', (event) => {
      if (panel.hidden) return;
      if (/^[0-9]$/.test(event.key)) keypress(event.key);
      else if (event.key === 'Backspace') keypress('del');
    });

    async function request(path, body, token) {
      const sentAt = Date.now();
      const res = await win.fetch(`${API}${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...(token ? { Authorization: `Scan ${token}` } : {}) },
        body: JSON.stringify(body || {}),
      });
      const receivedAt = Date.now();
      const data = await res.json().catch(() => ({}));
      if (data && data.serverTime) clock.sample(sentAt, receivedAt, data.serverTime);
      return { status: res.status, data };
    }

    function saveState() {
      try {
        storage.setItem(STORE_KEY, JSON.stringify(state));
      } catch (_) {
        // private mode
      }
    }

    async function pair(body) {
      status.textContent = 'Connecting…';
      keys.classList.add('busy');
      let ok = false;
      try {
        const { status: code, data } = await request('/api/scan-pair', { ...body, device: '' });
        if (code === 200 && data.phoneToken) {
          state = { token: data.phoneToken, sessionId: data.sessionId };
          saveState();
          status.textContent = 'Connected to Pokoin Dashboard';
          panel.classList.add('ok');
          const tip = doc.getElementById('scOpenBrowser');
          if (tip) tip.remove();
          if (data.defaultsLabel) barPile.textContent = data.defaultsLabel;
          setTimeout(openScanner, 700);
          ok = true;
          return ok;
        }
        status.textContent = code === 429
          ? 'Too many tries. Wait a minute.'
          : body.qr
            ? 'This QR code has expired. Type the new code from the dashboard.'
            : (data.error || 'Code not valid or expired.');
      } catch (_) {
        status.textContent = 'No connection. Check the network and try again.';
      } finally {
        keys.classList.remove('busy');
        if (!state.token) {
          digits = '';
          hidden.value = '';
          renderSlots();
        }
      }
      return ok;
    }

    function showKeypad(message) {
      clearInterval(heartbeatTimer);
      panel.hidden = false;
      panel.classList.remove('ok');
      bar.hidden = true;
      shutter.hidden = true;
      digits = '';
      renderSlots();
      status.textContent = message || '';
      if (typeof win.stopLive === 'function') win.stopLive();
    }

    function disconnected(message) {
      state = { token: '', sessionId: '' };
      saveState();
      outbox.clear();
      showKeypad(message);
    }

    const outbox = createOutbox({
      storage,
      onChange: (n) => {
        barText.textContent = paused ? 'Paused on dashboard' : n ? `Sending ${n}…` : `Connected · ${sent} sent`;
      },
      send: async (event, attempt) => {
        if (!state.token) return { halt: true };
        const { status: code, data } = await request('/api/scan-phone?action=scan', { ...event, timings: { ...event.timings, attempt } }, state.token);
        if (code === 200) {
          sent = data.received || sent + (data.duplicate ? 0 : 1);
          if (data.stackFull) flashStackFull();
          if (data.defaultsLabel) barPile.textContent = data.defaultsLabel;
          return { ok: true };
        }
        if (code === 401) {
          disconnected('Disconnected from the dashboard. Enter a new code.');
          return { halt: true };
        }
        if (code === 409 && data.code === 'paused') {
          paused = true;
          return { retry: true };
        }
        if (code === 409 || code === 400 || code === 413) return { drop: true };
        return { retry: true };
      },
    });

    async function heartbeat() {
      if (!state.token) return;
      try {
        const { status: code, data } = await request('/api/scan-phone?action=heartbeat', {}, state.token);
        if (code === 401) {
          disconnected('Session ended on the dashboard. Enter a new code.');
          return;
        }
        if (code === 200) {
          const wasPaused = paused;
          paused = data.paused === true;
          barPile.textContent = data.defaultsLabel || '';
          barText.textContent = paused ? 'Paused on dashboard' : `Connected · ${data.received} sent`;
          bar.classList.toggle('paused', paused);
          if (wasPaused && !paused) outbox.resume();
        }
      } catch (_) {
        barText.textContent = 'Reconnecting…';
      }
    }

    function openScanner() {
      panel.hidden = true;
      bar.hidden = false;
      shutter.hidden = false;
      clearInterval(heartbeatTimer);
      heartbeat();
      heartbeatTimer = setInterval(heartbeat, HEARTBEAT_MS);
      outbox.resume();
      if (typeof win.selectCatalog === 'function') win.selectCatalog('pokemon', 'generic');
      if (typeof win.setMode === 'function') win.setMode('single');
      if (typeof win.startCam === 'function' && !cameraStarted) {
        cameraStarted = true;
        win.startCam();
      } else if (typeof win.scheduleLive === 'function') {
        win.scheduleLive(100);
      }
    }

    async function thumbnail(blob, data) {
      try {
        const box = data && data.boxes && data.boxes[0] && data.boxes[0].xyxy;
        const bitmap = await win.createImageBitmap(blob);
        const sx = box ? Math.max(0, box[0] * (bitmap.width / (data.img_w || bitmap.width))) : 0;
        const sy = box ? Math.max(0, box[1] * (bitmap.height / (data.img_h || bitmap.height))) : 0;
        const sw = box ? Math.min(bitmap.width - sx, (box[2] - box[0]) * (bitmap.width / (data.img_w || bitmap.width))) : bitmap.width;
        const sh = box ? Math.min(bitmap.height - sy, (box[3] - box[1]) * (bitmap.height / (data.img_h || bitmap.height))) : bitmap.height;
        const scale = Math.min(1, 320 / Math.max(sw, sh));
        const canvas = doc.createElement('canvas');
        canvas.width = Math.max(1, Math.round(sw * scale));
        canvas.height = Math.max(1, Math.round(sh * scale));
        canvas.getContext('2d').drawImage(bitmap, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);
        for (const quality of [0.7, 0.5, 0.35]) {
          const url = canvas.toDataURL('image/jpeg', quality);
          if (url.length * 0.75 < 44_000) return url.slice(url.indexOf(',') + 1);
        }
      } catch (_) {
        // no thumbnail is fine
      }
      return '';
    }

    function showFlash(result) {
      const s = result.summary;
      const ok = confident(s);
      flash.textContent = ok ? `✓ ${s.top.name || 'Card'}` : s.top ? '? Check on dashboard' : '? Not identified — check on dashboard';
      flash.className = `sc-flash ${ok ? 'ok' : 'check'}`;
      flash.hidden = false;
      clearTimeout(showFlash.t);
      showFlash.t = setTimeout(() => {
        flash.hidden = true;
      }, 1100);
      if (win.navigator && typeof win.navigator.vibrate === 'function') win.navigator.vibrate(ok ? 20 : [20, 60, 20]);
    }

    async function send(result) {
      showFlash(result);
      sequence += 1;
      try {
        storage.setItem(`${STORE_KEY}.seq`, String(sequence));
      } catch (_) {
        // ignore
      }
      const image = !confident(result.summary) && result.context && result.context.blob
        ? await thumbnail(result.context.blob, result.data)
        : '';
      outbox.add(buildEvent(result, { sequence, clockOffsetMs: clock.offset, image }));
    }

    shutter.addEventListener('click', () => {
      if (lastPreview) send(gate.force(lastPreview.data, lastPreview.context));
    });
    leaveBtn.addEventListener('click', async () => {
      try {
        await request('/api/scan-phone?action=leave', {}, state.token);
      } catch (_) {
        // offline: the dashboard can disconnect it too
      }
      disconnected('Disconnected.');
    });

    doc.body.append(panel, bar, flash, shutter);

    const api = {
      /** Called by index.html for every live identify response. */
      onResult(data, context) {
        if (!state.token || panel.hidden === false || paused) return;
        lastPreview = { data, context };
        const result = gate.push(data, context);
        if (result) send(result);
      },
    };

    function clearPairFromUrl() {
      try {
        win.history.replaceState(null, '', win.location.pathname);
      } catch (_) {
        /* ignore */
      }
    }

    function showOpenInBrowser(pair) {
      if (doc.getElementById('scOpenBrowser')) return;
      const path = connectUrlWithPair('/connect', pair);
      const href = new URL(path, win.location.origin).href;
      const ua = String((win.navigator && win.navigator.userAgent) || '');
      // Put googlechromes:// / intent:// on the <a href> so iOS Camera's
      // native tap opens Chrome; JS location.assign alone often no-ops there.
      const chromeHref = chromeUrlFor(href, ua) || href;
      const btn = el('a', {
        class: 'sc-open-browser-btn',
        href: chromeHref,
        rel: 'noopener noreferrer',
        text: 'Open in Chrome',
      });
      btn.addEventListener('click', (ev) => {
        // If the scheme href already works, let the browser follow it.
        if (chromeHref !== href) return;
        ev.preventDefault();
        openInChrome(win, href);
      });
      panel.append(el('div', { class: 'sc-open-browser', id: 'scOpenBrowser' }, [
        btn,
        el('p', {
          class: 'sc-open-browser-note',
          text: 'Camera mini browser cannot keep the scan session. Opens this same page in Chrome.',
        }),
      ]));
    }

    // Reloading or opening this page in Safari/Chrome must NOT drop the pairing,
    // so there is no leave-on-unload handler. An abandoned phone is covered:
    // the desk shows "Connection lost" after 12 s of missed heartbeats and the
    // session expires after 30 min idle. Disconnect (here or on the desk) is the
    // deliberate way to free a session.

    const link = pairingFromLocation(win.location);
    if (link.qr) {
      // Opened from the dashboard QR. Keep c/k in the URL until pair succeeds so
      // "Open in Safari" from the Camera app still carries the secret.
      state = { token: '', sessionId: '' };
      outbox.clear();
      showKeypad('');
      digits = link.pin;
      renderSlots();
      showOpenInBrowser(link);
      pair(link.pin ? { qr: link.qr, pin: link.pin } : { qr: link.qr }).then((ok) => {
        if (ok) clearPairFromUrl();
      });
    } else if (state.token) {
      openScanner();
    } else {
      showKeypad('');
      setTimeout(() => hidden.focus(), 50);
    }
    return api;
  }

  return {
    MATCH_SCORE,
    MATCH_MARGIN,
    topAndMargin,
    confident,
    createCaptureGate,
    pairingFromLocation,
    connectUrlWithPair,
    chromeUrlFor,
    openInChrome,
    openInSystemBrowser,
    createClock,
    createOutbox,
    buildEvent,
    pairingFromHash,
    /** True on /connect: index.html defers the camera and the card-page redirect. */
    isConnectPath(loc) {
      return /^\/connect\/?$/.test(String((loc && loc.pathname) || ''));
    },
    start,
  };
});
