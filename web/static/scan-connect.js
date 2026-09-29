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
  // Past this the scan goes out unchosen (desk review) instead of waiting.
  const PRINTINGS_TIMEOUT_MS = 2500;
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

  const nameKey = (hit) => String((hit && hit.name) || '').trim().toLowerCase();

  /**
   * The picture is settled even though printings tie: top ≥ 0.80 and every
   * hit within the margin carries the top's name (HGSS vs Call of Legends
   * energy). BattleScan's own `_immediate` accepts that frame. Only *when* to
   * send is decided here; the server groups hits by artwork, never by name.
   */
  function settled(summary) {
    if (!summary.top || summary.score < MATCH_SCORE || !nameKey(summary.top)) return false;
    const rival = summary.hits.find((h) => nameKey(h) !== nameKey(summary.top));
    return !rival || summary.score - (Number(rival.score) || 0) >= MATCH_MARGIN;
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
          if (confident(summary) || settled(summary)) return emit(result);
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
   *
   * An event can be *held* while the seller picks its printing: it is already
   * persisted (a reload sends it as-is, never loses it) but the pump stops in
   * front of it until `release`, so order is kept and it is sent exactly once.
   */
  function createOutbox({ send, storage, onChange = () => {}, sleep = (ms) => new Promise((r) => setTimeout(r, ms)) }) {
    let queue = [];
    let running = false;
    let halted = false;
    const held = new Set();
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
      onChange(queue.length - held.size);
    };
    async function pump() {
      if (running) return;
      running = true;
      let attempt = 0;
      try {
        while (queue.length && !halted) {
          const event = queue[0];
          if (held.has(event.scanEventId)) break;
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
      add(event, { hold = false } = {}) {
        if (hold) held.add(event.scanEventId);
        queue.push(event);
        persist();
        pump();
      },
      /** Send a held event, optionally with fields added (the printing choice). Once only. */
      release(scanEventId, patch) {
        if (!held.delete(scanEventId)) return false;
        const event = queue.find((e) => e.scanEventId === scanEventId);
        if (event && patch) Object.assign(event, patch);
        persist();
        pump();
        return true;
      },
      resume() {
        halted = false;
        pump();
      },
      clear() {
        queue = [];
        held.clear();
        persist();
      },
      get size() {
        return queue.length;
      },
    };
  }

  /**
   * One pending printing choice at a time (pokoin-web docs/SCAN_CONNECT.md#printing-choice).
   * Its event is already in the outbox, held. Every way out releases it exactly
   * once: with the tapped printing, or without one (Skip, a different card,
   * disconnect) so the dashboard reviews it. The offered printings are frozen
   * when the tiles open; later camera frames never change them.
   */
  function createPrintingPick({ release }) {
    let current = null;
    return {
      open(event, printings) {
        const hits = (event.recognition && event.recognition.hits) || [];
        current = {
          event,
          printings: printings.slice(),
          ids: new Set([...hits.map((h) => String(h.public_id)), ...printings.map((p) => String(p.cardId))]),
        };
        return current;
      },
      /** The pending card back in view (or re-sent by the gate) keeps its tiles. */
      sameCard(result) {
        const top = result && result.summary && result.summary.top;
        return Boolean(current && top && current.ids.has(String(top.public_id)));
      },
      /** A tile tap. Only the first tap of this card counts. */
      choose(cardId) {
        if (!current) return null;
        const printing = current.printings.find((p) => String(p.cardId) === String(cardId));
        if (!printing) return null;
        const pick = current;
        current = null;
        release(pick.event.scanEventId, { printing: { cardId: String(printing.cardId) } });
        return printing;
      },
      /** Send unchosen: the dashboard shows it for review with the same printings. */
      cancel() {
        if (!current) return false;
        const pick = current;
        current = null;
        release(pick.event.scanEventId);
        return true;
      },
      get active() {
        return current;
      },
    };
  }

  /** Worth asking the server: only a confident top hit can be a confident artwork. */
  function asksPrintings(event) {
    const hits = (event && event.recognition && event.recognition.hits) || [];
    return hits.length > 0 && Number(hits[0].score) >= MATCH_SCORE;
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

    // Printing choice tiles, docked at the bottom of the camera frame.
    const pickTitle = el('span', { class: 'sc-pick-title', id: 'scPickTitle', 'aria-live': 'polite' });
    const pickSkip = el('button', {
      type: 'button',
      class: 'sc-pick-skip',
      text: 'Skip',
      'aria-label': 'Skip: choose the printing on the dashboard',
    });
    const pickRow = el('div', { class: 'sc-pick-row' });
    const tray = el('div', { class: 'sc-pick', id: 'scPick', role: 'group', 'aria-labelledby': 'scPickTitle', tabindex: '-1', hidden: '' }, [
      el('div', { class: 'sc-pick-head' }, [pickTitle, pickSkip]),
      pickRow,
    ]);
    const picker = createPrintingPick({ release: (id, patch) => outbox.release(id, patch) });
    let trayVersion = 0;

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
          if (data.scanCatalog) applyScanCatalog(data.scanCatalog);
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
      picker.cancel();
      closeTray();
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
          if (data.scanCatalog) applyScanCatalog(data.scanCatalog);
          barText.textContent = paused ? 'Paused on dashboard' : `Connected · ${data.received} sent`;
          bar.classList.toggle('paused', paused);
          if (wasPaused && !paused) outbox.resume();
        }
      } catch (_) {
        barText.textContent = 'Reconnecting…';
      }
    }

    let phoneCatalog = { family: 'pokemon', variant: 'generic' };
    function applyScanCatalog(info) {
      if (info && info.family) phoneCatalog = { family: info.family, variant: info.variant || 'generic' };
      if (typeof win.selectCatalog === 'function') {
        win.selectCatalog(phoneCatalog.family, phoneCatalog.variant);
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
      applyScanCatalog(phoneCatalog);
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

    function vibrate(pattern) {
      if (win.navigator && typeof win.navigator.vibrate === 'function') win.navigator.vibrate(pattern);
    }

    function flashText(text, tone) {
      flash.textContent = text;
      flash.className = `sc-flash ${tone}`;
      flash.hidden = false;
      clearTimeout(flashText.t);
      flashText.t = setTimeout(() => {
        flash.hidden = true;
      }, 1100);
    }

    /** `answer` (server printing check) wins over the phone's own margin rule. */
    function showFlash(result, answer) {
      const s = result.summary;
      const ok = answer ? answer.state === 'matched' : confident(s);
      const name = (answer && answer.name) || (s.top && s.top.name) || 'Card';
      flashText(ok ? `✓ ${name}` : s.top ? '? Check on dashboard' : '? Not identified — check on dashboard', ok ? 'ok' : 'check');
      vibrate(ok ? 20 : [20, 60, 20]);
    }

    function closeTray() {
      trayVersion += 1;
      tray.hidden = true;
      pickRow.replaceChildren();
      doc.body.classList.remove('sc-picking');
      if (state.token && panel.hidden) shutter.hidden = false;
    }

    function tile(printing) {
      const mark = el('span', { class: 'sc-tile-mark' });
      // Expansion / program mark first; the catalog set code only when no image loads.
      const showCode = () => mark.replaceChildren(el('span', { class: 'sc-tile-code', text: printing.setCode || printing.setName || '?' }));
      const sources = [printing.symbolUrl, printing.symbolAltUrl].filter(Boolean);
      if (sources.length) {
        const img = el('img', { alt: '', src: sources.shift(), decoding: 'async', draggable: 'false' });
        img.addEventListener('error', () => {
          if (sources.length) img.src = sources.shift();
          else showCode();
        });
        mark.append(img);
      } else {
        showCode();
      }
      const label = printing.label || [printing.setName, printing.number && `card ${printing.number}`].filter(Boolean).join(', ');
      const button = el('button', { type: 'button', class: 'sc-tile', 'aria-label': label, title: label }, [mark]);
      if (printing.number) button.append(el('span', { class: 'sc-tile-num', text: printing.number }));
      if (printing.detail) button.append(el('span', { class: 'sc-tile-detail', text: printing.detail }));
      button.addEventListener('click', () => pickPrinting(printing, button));
      return button;
    }

    function openTray(event, answer) {
      picker.open(event, answer.printings);
      trayVersion += 1;
      const name = String(answer.printings[0].name || '');
      pickTitle.textContent = name ? `Which printing? · ${name}` : 'Which printing?';
      pickRow.replaceChildren(...answer.printings.map(tile));
      pickRow.scrollLeft = 0;
      clearTimeout(flashText.t);
      flash.hidden = true;
      shutter.hidden = true;
      tray.hidden = false;
      doc.body.classList.add('sc-picking');
      vibrate([15, 40, 15]);
      // Focus the group, not a tile: a ringed first tile would read as a preselected printing.
      tray.focus({ preventScroll: true });
    }

    function pickPrinting(printing, button) {
      const chosen = picker.choose(printing.cardId);
      if (!chosen) return; // a second tap on this card, or it was already sent
      for (const b of pickRow.querySelectorAll('.sc-tile')) b.disabled = true;
      button.classList.add('picked');
      flashText(`✓ ${chosen.name || 'Card'} · ${[chosen.setCode || chosen.setName, chosen.number].filter(Boolean).join(' ')}`, 'ok');
      vibrate(20);
      const version = trayVersion;
      setTimeout(() => {
        if (version === trayVersion) closeTray();
      }, 220);
    }

    function skipPrinting() {
      if (!picker.cancel()) return;
      closeTray();
      flashText('? Check on dashboard', 'check');
    }

    pickSkip.addEventListener('click', skipPrinting);
    tray.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        skipPrinting();
        return;
      }
      if (event.key !== 'ArrowRight' && event.key !== 'ArrowLeft') return;
      const tiles = Array.from(pickRow.querySelectorAll('.sc-tile:not([disabled])'));
      if (!tiles.length) return;
      const at = tiles.indexOf(doc.activeElement);
      event.preventDefault();
      const next = at < 0
        ? tiles[event.key === 'ArrowRight' ? 0 : tiles.length - 1]
        : tiles[(at + (event.key === 'ArrowRight' ? 1 : tiles.length - 1)) % tiles.length];
      next.focus();
      if (typeof next.scrollIntoView === 'function') next.scrollIntoView({ block: 'nearest', inline: 'nearest' });
    });

    /** Read-only server check: which printings of this artwork the batch language allows. */
    async function askPrintings(event) {
      if (!asksPrintings(event) || !state.token) return null;
      let timer = 0;
      try {
        const res = await Promise.race([
          request('/api/scan-phone?action=printings', {
            recognition: event.recognition,
            capturedAt: event.capturedAt,
            clockOffsetMs: event.clockOffsetMs,
          }, state.token),
          new Promise((resolve) => {
            timer = setTimeout(() => resolve(null), PRINTINGS_TIMEOUT_MS);
          }),
        ]);
        return res && res.status === 200 && res.data ? res.data : null;
      } catch (_) {
        return null;
      } finally {
        clearTimeout(timer);
      }
    }

    // Gate emissions run one at a time so a printing question never interleaves.
    let captures = Promise.resolve();
    function send(result) {
      captures = captures.then(() => capture(result)).catch(() => {});
      return captures;
    }

    async function capture(result) {
      if (picker.active) {
        if (picker.sameCard(result)) return; // the pending card again: keep its tiles
        picker.cancel(); // moved on without choosing: the dashboard reviews it
        closeTray();
      }
      const quick = confident(result.summary);
      if (quick) showFlash(result);
      sequence += 1;
      try {
        storage.setItem(`${STORE_KEY}.seq`, String(sequence));
      } catch (_) {
        // ignore
      }
      const summary = result.summary || {};
      try {
        console.info('[scan-connect] queue-for-dashboard', JSON.stringify({
          sequence,
          confident: confident(summary),
          score: summary.score || 0,
          margin: summary.margin || 0,
          name: summary.top ? summary.top.name : null,
          publicId: summary.top ? summary.top.public_id : null,
          boxes: Array.isArray(result.data && result.data.boxes) ? result.data.boxes.length : 0,
        }));
      } catch (_) {
        // ignore
      }
      const image = !quick && result.context && result.context.blob
        ? await thumbnail(result.context.blob, result.data)
        : '';
      // Persisted now (a reload sends it as-is), sent once the printing is known.
      const event = buildEvent(result, { sequence, clockOffsetMs: clock.offset, image });
      outbox.add(event, { hold: true });
      const answer = await askPrintings(event);
      if (answer && answer.choose && Array.isArray(answer.printings) && answer.printings.length > 1 && state.token) {
        openTray(event, answer);
        return;
      }
      if (!quick) showFlash(result, answer);
      outbox.release(event.scanEventId);
    }

    leaveBtn.addEventListener('click', async () => {
      try {
        await request('/api/scan-phone?action=leave', {}, state.token);
      } catch (_) {
        // offline: the dashboard can disconnect it too
      }
      disconnected('Disconnected.');
    });

    doc.body.append(panel, bar, flash, shutter);
    // Inside the camera dock so it rides above the safe area and camera select.
    (doc.querySelector('.dock') || doc.querySelector('.stage') || doc.body).append(tray);

    const api = {
      /** Called by index.html for every live identify response. */
      onResult(data, context) {
        if (!state.token || panel.hidden === false || paused) return;
        lastPreview = { data, context };
        const result = gate.push(data, context);
        if (result) send(result);
      },
      /**
       * Optional: index.html sets this so Add card runs a fresh identify
       * instead of only reusing the last live preview.
       * @type {null | (() => Promise<{data: object, context: object} | null>)}
       */
      captureNow: null,
    };

    shutter.addEventListener('click', () => {
      const run = async () => {
        shutter.disabled = true;
        try {
          let preview = lastPreview;
          if (typeof api.captureNow === 'function') {
            const fresh = await api.captureNow();
            if (fresh && fresh.data) preview = fresh;
          }
          if (preview) send(gate.force(preview.data, preview.context));
        } finally {
          shutter.disabled = false;
        }
      };
      run();
    });

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
    settled,
    createCaptureGate,
    createPrintingPick,
    asksPrintings,
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
