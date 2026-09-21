#!/usr/bin/env node
/**
 * Phone viewport layout: Add card shutter must not cover the camera select
 * on common iPhone CSS sizes. Run on nezopt (playwright-core available):
 *
 *   node scripts/scan-connect-phone-layout.mjs
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

  await browser.close();
  server.close();
  if (failures.length) {
    console.error(failures.join('\n'));
    process.exit(1);
  }
  console.log(`ok ${PHONES.length} phone viewports — shutter clears camera select`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
