// node --test scripts/scan-connect.test.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { test } = require('node:test');
const sc = require(path.join(__dirname, '..', 'web', 'static', 'scan-connect.js'));

const frame = (hits, boxes = 1) => ({
  catalog: 'pokemon_generic',
  boxes: Array.from({ length: boxes }, () => ({ xyxy: [10, 10, 200, 280] })),
  img_w: 960,
  img_h: 720,
  hits: hits.map(([id, score]) => ({ public_id: id, score, name: `c${id}` })),
});
const empty = () => frame([], 0);

function clockAt(start = 0) {
  let t = start;
  const now = () => t;
  now.advance = (ms) => {
    t += ms;
  };
  return now;
}

test('confident match fires once; the same card staying in view does not fire again', () => {
  const now = clockAt();
  const gate = sc.createCaptureGate(now);
  assert.ok(gate.push(frame([['220962', 0.93]])));
  for (let i = 0; i < 20; i += 1) {
    now.advance(280);
    assert.equal(gate.push(frame([['220962', 0.93]])), null);
  }
});

test('four identical copies: card leaves the frame between copies → four events', () => {
  const now = clockAt();
  const gate = sc.createCaptureGate(now);
  let events = 0;
  for (let copy = 0; copy < 4; copy += 1) {
    for (let f = 0; f < 3; f += 1) {
      now.advance(280);
      if (gate.push(frame([['220962', 0.94]]))) events += 1;
    }
    for (let f = 0; f < 2; f += 1) {
      now.advance(280);
      gate.push(empty());
    }
  }
  assert.equal(events, 4);
});

test('a single empty frame (motion blur) does not re-arm', () => {
  const now = clockAt();
  const gate = sc.createCaptureGate(now);
  assert.ok(gate.push(frame([['1', 0.95]])));
  gate.push(empty());
  assert.equal(gate.push(frame([['1', 0.95]])), null);
});

test('swap without clearing the frame: a different confident card for two frames fires', () => {
  const now = clockAt();
  const gate = sc.createCaptureGate(now);
  assert.ok(gate.push(frame([['1', 0.95]])));
  assert.equal(gate.push(frame([['2', 0.95]])), null);
  const fired = gate.push(frame([['2', 0.95]]));
  assert.equal(fired.summary.top.public_id, '2');
  // One stray different frame is not a swap.
  assert.equal(gate.push(frame([['3', 0.95]])), null);
  assert.equal(gate.push(frame([['2', 0.95]])), null);
});

test('ambiguous or unidentified card fires after ~1.2 s with the best frame, so scanning never stalls', () => {
  const now = clockAt();
  const gate = sc.createCaptureGate(now);
  assert.equal(gate.push(frame([['7', 0.70], ['8', 0.68]])), null);
  now.advance(500);
  assert.equal(gate.push(frame([['7', 0.76], ['8', 0.70]])), null);
  now.advance(500);
  assert.equal(gate.push(frame([['7', 0.72]])), null);
  now.advance(300);
  const fired = gate.push(frame([['7', 0.71]]));
  assert.ok(fired);
  assert.equal(fired.summary.score, 0.76, 'best frame in the window');
  // Detected but zero hits also resolves.
  gate.push(empty());
  gate.push(empty());
  gate.push(frame([], 1));
  now.advance(1300);
  const none = gate.push(frame([], 1));
  assert.ok(none);
  assert.equal(none.summary.top, null);
});

test('high score but close runner-up is not confident (0.08 margin)', () => {
  assert.equal(sc.confident(sc.topAndMargin(frame([['1', 0.9], ['2', 0.85]]))), false);
  assert.equal(sc.confident(sc.topAndMargin(frame([['1', 0.9], ['1', 0.89], ['2', 0.8]]))), true, 'same id twice is not a runner-up');
});

test('manual shutter forces an event even when disarmed', () => {
  const gate = sc.createCaptureGate(clockAt());
  gate.push(frame([['1', 0.95]]));
  assert.ok(gate.force(frame([['1', 0.95]])));
});

test('clock offset keeps the tightest round trip', () => {
  const clock = sc.createClock();
  clock.sample(1000, 1400, 5200); // rtt 400 → offset 4000
  assert.equal(clock.offset, 4000);
  clock.sample(2000, 2040, 6030); // rtt 40 → offset 4010
  assert.equal(clock.offset, 4010);
  clock.sample(3000, 3900, 9999); // worse rtt ignored
  assert.equal(clock.offset, 4010);
  clock.sample(3000, 2000, 1); // negative rtt ignored
  assert.equal(clock.offset, 4010);
});

test('event carries immutable id, sequence, capture time, offset and phone timings', () => {
  const gate = sc.createCaptureGate(clockAt());
  const result = gate.push(frame([['220962', 0.93], ['220964', 0.2]]), { capturedAt: 1000, requestAt: 1040, respondedAt: 1290 });
  const event = sc.buildEvent(result, { sequence: 7, clockOffsetMs: -30, image: '' });
  assert.match(event.scanEventId, /^[0-9a-f-]{36}$/);
  assert.equal(event.clientSequence, 7);
  assert.equal(event.capturedAt, 1000);
  assert.equal(event.clockOffsetMs, -30);
  assert.deepEqual(event.timings, { captureToRequestMs: 40, identifyMs: 250 });
  assert.deepEqual(event.recognition.hits.map((h) => h.public_id), ['220962', '220964']);
  assert.equal(event.image, undefined);
  assert.notEqual(sc.buildEvent(result, { sequence: 8 }).scanEventId, event.scanEventId);
});

function memoryStorage() {
  const data = new Map();
  return { getItem: (k) => (data.has(k) ? data.get(k) : null), setItem: (k, v) => data.set(k, String(v)), data };
}

test('outbox: FIFO, retries the same event on network loss, survives reload without photos', async () => {
  const storage = memoryStorage();
  const sent = [];
  let fail = 2;
  let release;
  const drained = new Promise((resolve) => {
    release = resolve;
  });
  const outbox = sc.createOutbox({
    storage,
    sleep: async () => {},
    onChange: (n) => {
      if (n === 0 && sent.length) release();
    },
    send: async (event) => {
      sent.push(event.scanEventId);
      if (event.scanEventId === 'a' && fail > 0) {
        fail -= 1;
        throw new TypeError('Failed to fetch');
      }
      return { ok: true };
    },
  });
  outbox.add({ scanEventId: 'a', image: 'BIG' });
  outbox.add({ scanEventId: 'b' });
  await drained;
  assert.deepEqual(sent, ['a', 'a', 'a', 'b']);

  const halted = sc.createOutbox({ storage, sleep: async () => {}, send: async () => ({ halt: true }) });
  halted.add({ scanEventId: 'c', image: 'BIG' });
  await new Promise((r) => setTimeout(r, 10));
  const stored = JSON.parse(storage.getItem('pokoin.scanConnect.outbox.v1'));
  assert.deepEqual(stored, [{ scanEventId: 'c' }], 'photo dropped from storage');
  const reloaded = sc.createOutbox({ storage, sleep: async () => {}, send: async () => ({ ok: true }) });
  assert.equal(reloaded.size, 1);
});

test('only /connect enables connect mode; / keeps the card-page redirect', () => {
  assert.equal(sc.isConnectPath({ pathname: '/connect' }), true);
  assert.equal(sc.isConnectPath({ pathname: '/connect/' }), true);
  assert.equal(sc.isConnectPath({ pathname: '/' }), false);
  assert.equal(sc.isConnectPath({ pathname: '/connected' }), false);
});

test('dashboard QR link carries the 4-digit code and the secret; junk is ignored', () => {
  assert.deepEqual(sc.pairingFromHash('#c=4827&k=AbCdEfGhIjKlMnOpQrStUvWxYz012345'), { pin: '4827', qr: 'AbCdEfGhIjKlMnOpQrStUvWxYz012345' });
  assert.deepEqual(sc.pairingFromHash('#k=AbCdEfGhIjKlMnOpQrStUvWxYz012345'), { pin: '', qr: 'AbCdEfGhIjKlMnOpQrStUvWxYz012345' });
  assert.deepEqual(sc.pairingFromHash('#c=48271&k=short'), { pin: '', qr: '' });
  assert.deepEqual(sc.pairingFromHash('#c=<script>&k=%3Cx%3E'), { pin: '', qr: '' });
  assert.deepEqual(sc.pairingFromHash(''), { pin: '', qr: '' });
});

test('pairingFromLocation prefers hash, falls back to query', () => {
  assert.deepEqual(
    sc.pairingFromLocation({ hash: '#c=4827&k=AbCdEfGhIjKlMnOpQrStUvWxYz012345', search: '' }),
    { pin: '4827', qr: 'AbCdEfGhIjKlMnOpQrStUvWxYz012345' },
  );
  assert.deepEqual(
    sc.pairingFromLocation({ hash: '', search: '?c=4827&k=AbCdEfGhIjKlMnOpQrStUvWxYz012345' }),
    { pin: '4827', qr: 'AbCdEfGhIjKlMnOpQrStUvWxYz012345' },
  );
  assert.deepEqual(sc.pairingFromLocation({ hash: '', search: '' }), { pin: '', qr: '' });
});

test('connectUrlWithPair keeps c/k for Open in Safari hops', () => {
  assert.equal(
    sc.connectUrlWithPair('/connect', { pin: '4827', qr: 'AbCdEfGhIjKlMnOpQrStUvWxYz012345' }),
    '/connect?c=4827&k=AbCdEfGhIjKlMnOpQrStUvWxYz012345',
  );
  assert.equal(sc.connectUrlWithPair('/connect', { pin: '', qr: '' }), '/connect');
});

test('chromeUrlFor builds iOS googlechromes and Android intent URLs', () => {
  assert.equal(
    sc.chromeUrlFor('https://scan.pokoin.com/connect?c=4827&k=AbCdEfGhIjKlMnOpQrStUvWxYz012345', 'iPhone'),
    'googlechromes://scan.pokoin.com/connect?c=4827&k=AbCdEfGhIjKlMnOpQrStUvWxYz012345',
  );
  assert.equal(
    sc.chromeUrlFor('https://scan.pokoin.com/connect?c=4827&k=secret', 'Mozilla/5.0 (Linux; Android 14)'),
    'intent://scan.pokoin.com/connect?c=4827&k=secret#Intent;scheme=https;package=com.android.chrome;action=android.intent.action.VIEW;end',
  );
  assert.equal(sc.chromeUrlFor('https://scan.pokoin.com/connect', 'Macintosh'), '');
});

test('openInChrome hops iOS Camera mini browser into Chrome (same /connect URL)', () => {
  const hrefs = [];
  const win = {
    navigator: { userAgent: 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X)' },
    location: {
      assign(v) { hrefs.push(v); },
      href: '',
    },
    open() { return null; },
  };
  sc.openInChrome(win, 'https://scan.pokoin.com/connect?c=4827&k=AbCdEfGhIjKlMnOpQrStUvWxYz012345');
  assert.equal(
    hrefs[0],
    'googlechromes://scan.pokoin.com/connect?c=4827&k=AbCdEfGhIjKlMnOpQrStUvWxYz012345',
  );
});

test('openInChrome hops Android in-app browser into Chrome via intent', () => {
  const hrefs = [];
  const win = {
    navigator: { userAgent: 'Mozilla/5.0 (Linux; Android 14)' },
    location: {
      assign(v) { hrefs.push(v); },
      href: '',
    },
    open() { return null; },
  };
  sc.openInChrome(win, 'https://scan.pokoin.com/connect?c=4827&k=secret');
  assert.equal(
    hrefs[0],
    'intent://scan.pokoin.com/connect?c=4827&k=secret#Intent;scheme=https;package=com.android.chrome;action=android.intent.action.VIEW;end',
  );
});

test('no unload handler frees the pairing: reloading the phone keeps the session', () => {
  // pagehide/beforeunload/visibilitychange also fire on reload and when the Camera
  // mini browser hands off to Safari. Freeing the session there broke the documented
  // "phone reload keeps the pairing" behaviour (pokoin-web docs/SCAN_CONNECT.md and
  // the E2E step of the same name). Disconnect is the deliberate way to free it;
  // an abandoned phone is covered by the 12 s "Connection lost" indication on the
  // desk and by the 30-minute inactivity expiry.
  const src = fs.readFileSync(path.join(__dirname, '..', 'web', 'static', 'scan-connect.js'), 'utf8');
  for (const evt of ['pagehide', 'beforeunload', 'unload', 'visibilitychange']) {
    assert.equal(src.includes(`addEventListener('${evt}'`), false, `${evt} listener must not exist here`);
    assert.equal(src.includes(`addEventListener("${evt}"`), false, `${evt} listener must not exist here`);
  }
  assert.match(src, /action=leave/);
});
