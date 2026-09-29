#!/usr/bin/env node
/**
 * Phone viewport layout, run on nezopt (playwright-core available):
 *
 *   node scripts/scan-connect-phone-layout.cjs
 *
 * 1. Add card shutter must not cover the camera select on common phones.
 * 2. Printing choice tiles (real flow: tied recognition → printings API →
 *    tray) stay inside the camera frame, clear the camera select, are real
 *    touch targets at most ~1/5 of the viewport tall, centre when there are
 *    two, and scroll sideways at full size when there are more.
 */
const assert = require('node:assert/strict');
const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');

const ROOT = path.resolve(__dirname, '..');
const WEB = path.join(ROOT, 'web');
const PLAYWRIGHT = process.env.PLAYWRIGHT_CORE
  || path.join(ROOT, '..', 'pokemon-card-extension', 'node_modules', 'playwright-core');

const { chromium, devices } = require(PLAYWRIGHT);

const PHONES = [
  { name: 'iPhone SE', ...devices['iPhone SE'] },
  { name: 'iPhone 14', ...devices['iPhone 14'] },
  { name: 'iPhone 14 Pro Max', ...devices['iPhone 14 Pro Max'] },
  { name: 'Pixel 7', ...devices['Pixel 7'] },
];

// Printing tray also on landscape phones and a desktop browser.
const TRAY_VIEWPORTS = [
  ...PHONES,
  { name: 'iPhone 14 landscape', ...devices['iPhone 14 landscape'] },
  { name: 'Desktop', viewport: { width: 1280, height: 800 } },
];

const API = 'https://api.pokoin.com';
const tile = (cardId, setName, setCode, number, detail = '') => ({
  cardId, name: 'Grass Energy', setName, setCode, number, detail,
  symbolUrl: `https://cdn.pokoin.com/expansions/symbols/${setCode.toLowerCase()}.png?v=cm1`,
  symbolAltUrl: '', label: `${setName}, card ${number}`,
});
const TWO = [tile('224950', 'Call of Legends', 'CLO', '88/95'), tile('242396', 'HeartGold & SoulSilver', 'HGS', '115/123')];
const SIX = [
  ...TWO,
  tile('525054', 'League Promos', 'LPR', '88/95', 'Play! Pokemon · Holo Promo'),
  tile('448040', 'Trick or Trade', 'TRICKORTRADE', '076/203'),
  tile('609516', 'World Championship Decks 2022', 'WCD2022', '076/203', 'WCD 2022 · Ondrej Skubal'),
  tile('641442', 'World Championship Decks 2023', 'WCD2023', '076/203', 'WCD 2023 · Gabriel Fernandez'),
];

function overlaps(a, b) {
  return !(a.x + a.width <= b.x || b.x + b.width <= a.x || a.y + a.height <= b.y || b.y + b.height <= a.y);
}

function serve() {
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      const url = new URL(req.url, 'http://127.0.0.1');
      let file = url.pathname === '/' || url.pathname === '/connect'
        ? path.join(WEB, 'index.html')
        : path.join(WEB, url.pathname.replace(/^\//, ''));
      if (!file.startsWith(WEB) || !fs.existsSync(file)) {
        res.writeHead(404); res.end('missing'); return;
      }
      const ext = path.extname(file);
      const type = ext === '.js' ? 'text/javascript' : ext === '.css' ? 'text/css' : ext === '.png' ? 'image/png' : 'text/html';
      res.writeHead(200, { 'Content-Type': type });
      fs.createReadStream(file).pipe(res);
    });
    server.listen(0, '127.0.0.1', () => resolve(server));
  });
}

async function main() {
  const server = await serve();
  const { port } = server.address();
  const browser = await chromium.launch({ headless: true });
  const failures = [];

  for (const phone of PHONES) {
    const context = await browser.newContext({
      ...phone,
      // Camera permission not needed; we only check geometry.
    });
    const page = await context.newPage();
    await page.goto(`http://127.0.0.1:${port}/connect`, { waitUntil: 'domcontentloaded' });
    // Force connected UI: shutter + camera select visible (skip pairing).
    await page.evaluate(() => {
      document.body.classList.add('scan-connect');
      const panel = document.getElementById('scanConnectPanel');
      if (panel) panel.hidden = true;
      const cam = document.getElementById('cameraControls');
      if (cam) cam.hidden = false;
      // Inject shutter if connect script has not (no auto-pair in this harness).
      let shutter = document.querySelector('.sc-shutter');
      if (!shutter) {
        shutter = document.createElement('button');
        shutter.className = 'sc-shutter';
        shutter.textContent = 'Add card';
        shutter.removeAttribute('hidden');
        document.body.appendChild(shutter);
      } else {
        shutter.hidden = false;
      }
      const select = document.getElementById('cameraSelect');
      if (select && !select.options.length) {
        const opt = document.createElement('option');
        opt.textContent = 'Back Camera';
        select.append(opt);
      }
    });
    const shutterBox = await page.locator('.sc-shutter').boundingBox();
    const camBox = await page.locator('#cameraControls').boundingBox();
    assert.ok(shutterBox, `${phone.name}: shutter missing`);
    assert.ok(camBox, `${phone.name}: camera controls missing`);
    if (overlaps(shutterBox, camBox)) {
      failures.push(`${phone.name}: shutter ${JSON.stringify(shutterBox)} overlaps camera ${JSON.stringify(camBox)}`);
    }
    // Camera select must sit above the shutter (smaller y = higher on screen).
    if (camBox.y + camBox.height > shutterBox.y + 2) {
      failures.push(`${phone.name}: camera bottom ${camBox.y + camBox.height} not above shutter top ${shutterBox.y}`);
    }
    await context.close();
  }

  for (const view of TRAY_VIEWPORTS) {
    for (const offered of [TWO, SIX]) {
      const problems = await trayLayout(browser, port, view, offered);
      failures.push(...problems.map((p) => `${view.name} (${offered.length} tiles): ${p}`));
    }
  }

  await browser.close();
  server.close();
  if (failures.length) {
    console.error(failures.join('\n'));
    process.exit(1);
  }
  console.log(`ok ${PHONES.length} phone viewports — shutter clears camera select; printing tiles fit ${TRAY_VIEWPORTS.length} viewports`);
}

async function trayLayout(browser, port, view, offered) {
  const problems = [];
  const context = await browser.newContext({ ...view });
  // Already paired: the page opens the scanner straight away.
  await context.addInitScript(() => {
    localStorage.setItem('pokoin.scanConnect.v1', JSON.stringify({ token: 't'.repeat(43), sessionId: 's1' }));
  });
  const page = await context.newPage();
  const asked = [];
  await page.route(`${API}/**`, async (route) => {
    const url = new URL(route.request().url());
    const action = url.searchParams.get('action');
    if (action === 'printings') {
      asked.push(JSON.parse(route.request().postData() || '{}'));
      return route.fulfill({ json: { state: 'ambiguous', choose: true, printings: offered, serverTime: Date.now() } });
    }
    return route.fulfill({ json: { status: 'connected', paused: false, received: 0, serverTime: Date.now(), defaultsLabel: 'EN · NM' } });
  });
  // One real mark, the rest 404 → set-code fallback.
  await page.route('https://cdn.pokoin.com/**', (route) => (route.request().url().includes('/clo.png')
    ? route.fulfill({ path: path.join(WEB, 'static', 'pokoin-icon.png') })
    : route.fulfill({ status: 404, body: '' })));
  await page.goto(`http://127.0.0.1:${port}/connect`, { waitUntil: 'domcontentloaded' });
  await page.evaluate(() => {
    const cam = document.getElementById('cameraControls');
    cam.hidden = false;
    const select = document.getElementById('cameraSelect');
    if (!select.options.length) select.append(new Option('Back Camera'));
    // HGSS / Call of Legends energy tie: settled artwork, printing unknown.
    scanConnect.onResult({
      catalog: 'pokemon_generic', boxes: [{ xyxy: [10, 10, 200, 280] }], img_w: 960, img_h: 720,
      hits: [
        { public_id: '242396', score: 0.95, name: 'Grass Energy' },
        { public_id: '224950', score: 0.949, name: 'Grass Energy' },
      ],
    }, { capturedAt: Date.now() });
  });
  try {
    await page.locator('.sc-pick').waitFor({ state: 'visible', timeout: 4000 });
  } catch (_) {
    await context.close();
    return ['tray never opened'];
  }
  if (asked.length !== 1) problems.push(`printings asked ${asked.length}×`);
  const m = await page.evaluate(() => {
    const box = (el) => { const r = el.getBoundingClientRect(); return { x: r.x, y: r.y, width: r.width, height: r.height }; };
    const row = document.querySelector('.sc-pick-row');
    const tiles = Array.from(document.querySelectorAll('.sc-tile'));
    return {
      vw: innerWidth,
      vh: innerHeight,
      pageScrollW: document.documentElement.scrollWidth,
      stage: box(document.querySelector('.stage')),
      tray: box(document.querySelector('.sc-pick')),
      camera: box(document.getElementById('cameraControls')),
      shutterHidden: document.querySelector('.sc-shutter').hidden,
      row: { ...box(row), scrollWidth: row.scrollWidth, clientWidth: row.clientWidth },
      tiles: tiles.map((t) => ({ ...box(t), label: t.getAttribute('aria-label'), tag: t.tagName, img: Boolean(t.querySelector('img')), code: t.querySelector('.sc-tile-code')?.textContent || '' })),
      tileFocused: Boolean(document.activeElement && document.activeElement.classList.contains('sc-tile')),
      trayFocused: document.activeElement === document.querySelector('.sc-pick'),
      markOverflow: tiles.some((t) => { const img = t.querySelector('.sc-tile-mark img'); if (!img) return false; const a = img.getBoundingClientRect(); const b = t.querySelector('.sc-tile-mark').getBoundingClientRect(); return a.bottom > b.bottom + 1 || a.top < b.top - 1; }),
    };
  });
  const inside = (b, o) => b.x >= o.x - 1 && b.y >= o.y - 1 && b.x + b.width <= o.x + o.width + 1 && b.y + b.height <= o.y + o.height + 1;
  if (m.pageScrollW > m.vw + 1) problems.push(`page scrolls sideways (${m.pageScrollW} > ${m.vw})`);
  if (!inside(m.tray, m.stage)) problems.push(`tray ${JSON.stringify(m.tray)} leaves the camera frame`);
  if (overlaps(m.tray, m.camera)) problems.push('tray covers the camera select');
  if (!m.shutterHidden) problems.push('shutter still shown under the tray');
  if (m.tiles.length !== offered.length) problems.push(`${m.tiles.length} tiles for ${offered.length} printings`);
  for (const t of m.tiles) {
    if (t.tag !== 'BUTTON' || !t.label) problems.push(`tile without button/label: ${JSON.stringify(t)}`);
    if (t.height < 44 || t.width < 44) problems.push(`tile too small ${t.width}×${t.height}`);
    if (t.height > m.vh * 0.2 + 1) problems.push(`tile ${t.height}px taller than 1/5 of ${m.vh}`);
    const ratio = t.width / t.height;
    if (ratio < 1.1 || ratio > 1.2) problems.push(`tile ratio ${ratio.toFixed(2)} not ~1.15`);
    if (t.y < m.row.y - 1 || t.y + t.height > m.row.y + m.row.height + 1) problems.push('tile leaves its row vertically');
  }
  const first = m.tiles[0];
  if (!first.img) problems.push('loaded symbol not shown');
  if (m.tiles[1] && (m.tiles[1].img || m.tiles[1].code !== 'HGS')) problems.push(`missing symbol did not fall back to the set code (${m.tiles[1].code})`);
  if (m.tileFocused) problems.push('a tile starts focused, which reads as a preselected printing');
  if (!m.trayFocused) problems.push('tray group not focused for keyboard / screen reader users');
  if (m.markOverflow) problems.push('a symbol overflows its mark area');
  // Keyboard: ArrowRight enters the tiles, Escape skips.
  await page.keyboard.press('ArrowRight');
  if (!(await page.evaluate(() => document.activeElement === document.querySelector('.sc-tile')))) problems.push('ArrowRight does not reach the first tile');
  if (offered.length === 2) {
    const left = first.x - m.row.x;
    const right = m.row.x + m.row.width - (m.tiles[1].x + m.tiles[1].width);
    if (Math.abs(left - right) > 4) problems.push(`two tiles not centred (left ${left.toFixed(1)}, right ${right.toFixed(1)})`);
  } else if (m.row.scrollWidth > m.row.clientWidth + 1) {
    // More tiles than fit: the row scrolls, the tiles keep their size.
    const sizes = new Set(m.tiles.map((t) => Math.round(t.width)));
    if (sizes.size !== 1) problems.push(`scrolling tiles changed size: ${[...sizes]}`);
  }
  // A tap sends the scan with that printing and closes the tray; a second tap is ignored.
  const scans = [];
  await page.unroute(`${API}/**`);
  await page.route(`${API}/**`, (route) => {
    const url = new URL(route.request().url());
    if (url.searchParams.get('action') === 'scan') scans.push(JSON.parse(route.request().postData() || '{}'));
    return route.fulfill({ json: { ok: true, received: scans.length, serverTime: Date.now() } });
  });
  const target = page.locator('.sc-tile').nth(1);
  await target.click();
  await target.click({ force: true }).catch(() => {});
  await page.locator('.sc-pick').waitFor({ state: 'hidden', timeout: 3000 }).catch(() => problems.push('tray stayed open after a tap'));
  await page.waitForTimeout(150);
  if (scans.length !== 1) problems.push(`${scans.length} scan uploads after a double tap`);
  else if (!scans[0].printing || scans[0].printing.cardId !== offered[1].cardId) problems.push(`scan sent without the tapped printing: ${JSON.stringify(scans[0].printing)}`);
  if (await page.locator('.sc-shutter').isHidden()) problems.push('shutter not back after the choice');
  await context.close();
  return problems;
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
