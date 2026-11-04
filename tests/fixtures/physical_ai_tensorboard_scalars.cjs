'use strict';

const fs = require('node:fs/promises');
const path = require('node:path');
const { createHash } = require('node:crypto');
const { constants } = require('node:fs');
const assert = require('node:assert/strict');

const ORIGIN = 'http://127.0.0.1:6007';
const LIMIT = 1024 * 1024;
const SCOPE = 'native chart CSV/history only; MP4 retrieval and full V/task6.1 remain UNVERIFIED';
const METRICS = [
  {
    tag: 'train/rl/frames', csv: 'train-rl-frames.csv', png: 'train-rl-frames.png',
    axes: { x_axis: 'native logger step (not source physical frames)', value: 'physical training frames' },
  },
  {
    tag: 'eval/rl/return_mean', csv: 'eval-rl-return-mean.csv', png: 'eval-rl-return-mean.png',
    axes: { x_axis: 'evaluation source physical frames', value: 'mean evaluation episode return' },
  },
];

function expectedHistories(manifest, mode = 'tb-only') {
  assert(mode === 'tb-only' || mode === 'both', 'TensorBoard mode must be tb-only or both');
  const main = manifest?.modes?.[mode]?.native?.tensorboard?.main;
  const base = manifest?.modes?.[mode]?.logging?.tensorboard?.log_dir;
  for (const directory of [base, main?.log_dir]) {
    assert.equal(typeof directory, 'string', 'missing TensorBoard log_dir');
    assert(path.isAbsolute(directory) && path.normalize(directory) === directory, 'invalid log_dir');
  }
  const relativeRun = path.relative(base, main.log_dir);
  assert.equal(relativeRun, `physical-ai-logging-${mode}/cartpole-ppo_v1`, 'unexpected native relative run');
  const histories = {};
  for (const { tag } of METRICS) {
    const history = main.scalar_history?.[tag];
    assert(Array.isArray(history) && history.length >= 2, `missing native history: ${tag}`);
    let previousStep = -1;
    let previousFrames = -1;
    histories[tag] = history.map(({ step, value }) => {
      assert(Number.isSafeInteger(step) && step >= 0 && step > previousStep, `invalid native step: ${tag}`);
      assert(Number.isFinite(value), `invalid native value: ${tag}`);
      if (tag === 'train/rl/frames') {
        assert(Number.isSafeInteger(value) && value >= 0 && value > previousFrames, 'physical frames must increase');
        previousFrames = value;
      }
      previousStep = step;
      return { step, value };
    });
  }
  return { relativeRun, histories };
}

function nativeCSVURL(href, tag, relativeRun) {
  assert.equal(typeof href, 'string', 'missing native CSV href');
  assert(href.length > 0, 'empty native CSV href');
  const url = new URL(href, `${ORIGIN}/`);
  assert.equal(url.origin, ORIGIN, 'foreign CSV origin');
  assert(!url.username && !url.password && !url.hash, 'CSV credentials/fragment forbidden');
  assert.equal(url.pathname, '/data/plugin/scalars/scalars', 'unexpected CSV endpoint');
  assert.deepEqual([...url.searchParams.keys()].sort(), ['format', 'run', 'tag'], 'unexpected CSV query keys');
  for (const [key, value] of [['run', relativeRun], ['tag', tag], ['format', 'csv']]) {
    assert.deepEqual(url.searchParams.getAll(key), [value], `unexpected CSV ${key}`);
  }
  return url.href;
}

function parseCSV(bytes) {
  const lines = bytes.toString('utf8').split(/\r?\n/);
  if (lines.at(-1) === '') lines.pop();
  assert.equal(lines.shift(), 'Wall time,Step,Value', 'unexpected CSV header');
  assert(lines.length > 0, 'empty CSV history');
  const numeric = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/;
  let previousStep = -1;
  return lines.map((line) => {
    const columns = line.split(',');
    assert(columns.length === 3 && columns.every((column) => numeric.test(column)), 'invalid CSV numeric row');
    const [wall_time, step, value] = columns.map(Number);
    assert([wall_time, step, value].every(Number.isFinite) && wall_time >= 0, 'nonfinite/invalid CSV row');
    assert(Number.isSafeInteger(step) && step >= 0 && step > previousStep, 'invalid/duplicate CSV step');
    previousStep = step;
    return { wall_time, step, value };
  });
}

async function save(output, filename, bytes) {
  await fs.writeFile(path.join(output, filename), bytes, { flag: 'wx', mode: 0o600 });
  return { file: filename, bytes: bytes.length, sha256: createHash('sha256').update(bytes).digest('hex') };
}

async function bounded(promise, milliseconds, message) {
  let timer;
  try {
    return await Promise.race([promise, new Promise((_, reject) => { timer = setTimeout(() => reject(new Error(message)), milliseconds); })]);
  } finally { clearTimeout(timer); }
}

async function readNativeDownload(page, url, click, output) {
  const scratch = await fs.mkdtemp(path.join(output, '.download-'));
  let browser, target, contextId, frameId, guid, completed = false, primary, bytes, received = 0, timer, observing = false, transferFailure;
  let finish, reject;
  const finished = new Promise(resolve => { finish = resolve; });
  const failed = new Promise((_, value) => { reject = value; });
  failed.catch(() => {}); // Setup may fail before the transfer race is attached.
  const fail = error => { transferFailure ??= error; reject(transferFailure); };
  const begin = event => {
    if (event.frameId !== frameId) { fail(new Error('download did not originate in the owned frame')); return; }
    if (guid || !/^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$/.test(event.guid)) {
      fail(new Error('unexpected native download identity')); return;
    }
    guid = event.guid;
    if (event.url !== url) fail(new Error('download URL differs from native CSV href'));
  };
  const progress = event => {
    if (event.guid !== guid) return;
    received = Math.max(received, event.receivedBytes);
    if (![event.receivedBytes, event.totalBytes].every(value => Number.isFinite(value) && value >= 0)) {
      fail(new Error('invalid native download progress')); return;
    }
    if (event.receivedBytes > LIMIT || event.totalBytes > LIMIT) {
      fail(new Error('native CSV exceeds 1 MiB during download')); return;
    }
    if (event.state === 'canceled') fail(new Error('native CSV download failed: canceled'));
    else if (event.state === 'completed') { completed = true; finish(); }
  };
  const request = current => {
    if (!current.redirectedFrom()) return;
    for (let item = current; item; item = item.redirectedFrom()) {
      if (item.url() === url) { fail(new Error('native CSV redirects forbidden')); return; }
    }
  };
  try {
    const context = page.context();
    browser = await bounded(context.browser().newBrowserCDPSession(), 30000, 'download protocol setup deadline');
    target = await bounded(context.newCDPSession(page), 30000, 'download target setup deadline');
    const info = await bounded(target.send('Target.getTargetInfo'), 30000, 'download identity deadline');
    assert.equal(info.targetInfo.type, 'page', 'owned download target must be a page');
    contextId = info.targetInfo.browserContextId;
    assert(typeof contextId === 'string' && contextId, 'private browser context identity required');
    const tree = await bounded(target.send('Page.getFrameTree'), 30000, 'download frame identity deadline');
    frameId = tree.frameTree.frame.id;
    assert(typeof frameId === 'string' && frameId, 'owned download frame required');
    browser.on('Browser.downloadWillBegin', begin);
    browser.on('Browser.downloadProgress', progress);
    page.on('request', request);
    observing = true;
    await bounded(browser.send('Browser.setDownloadBehavior', {
      browserContextId: contextId, behavior: 'allowAndName', downloadPath: scratch, eventsEnabled: true,
    }), 30000, 'download admission deadline');
    if (transferFailure) throw transferFailure;
    timer = setTimeout(() => fail(new Error('native CSV download completion deadline exceeded')), 30000);
    await Promise.race([Promise.all([click(), finished]), failed]);
    const file = await fs.open(path.join(scratch, guid), constants.O_RDONLY | constants.O_NOFOLLOW);
    try {
      const stat = await file.stat();
      assert(stat.isFile() && stat.uid === process.getuid(), 'native download must be an owned regular file');
      assert(stat.size <= LIMIT, 'native CSV exceeds 1 MiB');
      const buffer = Buffer.alloc(LIMIT + 1);
      const result = await file.read(buffer, 0, buffer.length, 0);
      assert(result.bytesRead <= LIMIT && result.bytesRead === stat.size, 'native CSV size changed');
      bytes = buffer.subarray(0, result.bytesRead);
    } finally { await file.close(); }
  } catch (error) { primary = error; }
  finally {
    clearTimeout(timer);
    const cleanup = async action => {
      try { await bounded(action(), 5000, 'native download cleanup deadline'); }
      catch (error) { if (!primary) primary = error; else (primary.cleanupErrors ??= []).push(String(error)); }
    };
    if (browser && contextId) {
      if (guid && !completed) await cleanup(() => browser.send('Browser.cancelDownload', { browserContextId: contextId, guid }));
      await cleanup(() => browser.send('Browser.setDownloadBehavior', { browserContextId: contextId, behavior: 'deny', eventsEnabled: false }));
      browser.off('Browser.downloadWillBegin', begin);
      browser.off('Browser.downloadProgress', progress);
    }
    if (observing) page.off('request', request);
    if (target) await cleanup(() => target.detach());
    if (browser) await cleanup(() => browser.detach());
    if (guid) for (const suffix of ['', '.crdownload']) await cleanup(async () => {
      try { await fs.unlink(path.join(scratch, guid + suffix)); }
      catch (error) { if (error.code !== 'ENOENT') throw error; }
    });
    await cleanup(() => fs.rmdir(scratch));
  }
  if (primary) throw primary;
  return { bytes, budget: { limit_bytes: LIMIT, completion_seconds: 30, received_bytes: received, status: 'completed', scratch_cleanup: 'PASS' } };
}

async function runTensorBoardScalars(page, manifest, output, mode = 'tb-only') {
  const { relativeRun, histories } = expectedHistories(manifest, mode);
  assert.equal(typeof output, 'string', 'output must be an absolute directory');
  assert(path.isAbsolute(output) && path.resolve(output) === output, 'invalid output path');
  const stat = await fs.lstat(output);
  assert(stat.isDirectory() && !stat.isSymbolicLink(), 'output must be a real directory');
  assert.equal(stat.uid, process.getuid(), 'output must be owned by this user');
  assert.equal(stat.mode & 0o7777, 0o700, 'output must have mode 0700');
  assert.equal(await fs.realpath(output), output, 'output must not contain symlinks');
  assert.deepEqual(await fs.readdir(output), [], 'output must be empty');
  assert.equal(page.url(), `${ORIGIN}/#scalars`, 'unexpected owned Scalars page');

  // Native percentile scaling discards both distinct points in a two-point history.
  const smoothing = page.locator('tf-scalar-dashboard:visible tf-smoothing-input input[type="number"]');
  await smoothing.fill('0');
  await smoothing.press('Tab');
  assert.equal(await smoothing.inputValue(), '0', 'raw chart smoothing must be disabled');
  const outliers = page.getByRole('checkbox', { name: 'Ignore outliers in chart scaling', exact: true });
  if (await outliers.getAttribute('aria-checked') === 'true') await outliers.click({ timeout: 30000 });
  assert.equal(await outliers.getAttribute('aria-checked'), 'false', 'raw chart must include all scalar values');
  const summary = { scope: SCOPE, mode, relative_run: relativeRun,
    chart_settings: { ignore_outliers: false, smoothing: 0, role: 'raw native scalar inspection; event values unchanged' }, metrics: [] };
  for (const metric of METRICS) {
    // Tag search adds duplicate cards; use the ordinary named category instead.
    const prefix = metric.tag.split('/')[0];
    const category = page.locator('tf-scalar-dashboard:visible tf-category-paginated-view').filter({
      has: page.locator(`span.category-name[title="${prefix}"]`),
    });
    assert.equal(await category.count(), 1, `expected one native category: ${prefix}`);
    await category.waitFor({ state: 'visible', timeout: 30000 });
    const card = category.locator('tf-scalar-card:visible').filter({ has: page.getByText(metric.tag, { exact: true }) });
    const advance = await card.count() === 0;
    if (advance) {
      const next = category.getByRole('button', { name: 'Next page', exact: true });
      assert.equal(await next.count(), 2, 'expected paired native pagination controls');
      await next.nth(0).click({ timeout: 30000 });
    }
    await card.waitFor({ state: 'visible', timeout: 30000 });
    assert.equal(await card.count(), 1, `expected one native card: ${metric.tag}`);
    await card.scrollIntoViewIfNeeded({ timeout: 30000 });
    await card.locator('vz-line-chart2').waitFor({ state: 'visible', timeout: 30000 });
    await card.locator('vz-line-chart2 .ghost').waitFor({ state: 'hidden', timeout: 30000 });
    await card.locator('paper-icon-button[title="Fit domain to data"]').click({ timeout: 30000 });
    const curve = card.locator('vz-line-chart2 path.line[d*="L"]');
    // A painted horizontal SVG stroke has zero geometric height, unlike a widget.
    await curve.waitFor({ state: 'attached', timeout: 30000 });
    assert.equal(await curve.count(), 1, 'expected one native multi-point data curve');
    const menu = card.locator('tf-downloader paper-dropdown-menu');
    await menu.waitFor({ state: 'visible', timeout: 30000 });
    assert.equal(await menu.count(), 1, 'expected one native run dropdown');
    await menu.click();
    await page.getByRole('option', { name: relativeRun, exact: true }).click();
    const link = card.getByRole('link', { name: 'CSV', exact: true });
    await link.waitFor({ state: 'visible', timeout: 30000 });
    assert.notEqual(await link.getAttribute('download'), null, 'native CSV must be a download link');
    const href = await link.getAttribute('href');
    const url = nativeCSVURL(href, metric.tag, relativeRun);
    const { bytes, budget } = await readNativeDownload(page, url, () => link.click(), output);
    const csv = await save(output, metric.csv, bytes);
    const rows = parseCSV(bytes);
    assert.deepEqual(rows.map(({ step, value }) => ({ step, value })), histories[metric.tag], `native CSV/history mismatch: ${metric.tag}`);
    const plot = await save(output, metric.png, await card.screenshot({ type: 'png' }));
    const curvePath = await curve.getAttribute('d');
    const geometry = await curve.evaluate(path => {
      const box = path.getBoundingClientRect(), svg = path.ownerSVGElement.getBoundingClientRect();
      const style = getComputedStyle(path);
      let painted = path.isConnected && style.stroke !== 'none' && Number(style.strokeOpacity) > 0 && parseFloat(style.strokeWidth) > 0;
      for (let element = path; element; element = element.parentElement || element.getRootNode().host) {
        const css = getComputedStyle(element);
        painted &&= css.display !== 'none' && css.visibility === 'visible' && Number(css.opacity) > 0;
      }
      return { width: box.width, height: box.height, painted,
        within_svg: box.left >= svg.left && box.right <= svg.right && box.top >= svg.top && box.bottom <= svg.bottom,
        curve_box: [box.left, box.top, box.right, box.bottom], svg_box: [svg.left, svg.top, svg.right, svg.bottom] };
    });
    assert(geometry.painted && geometry.within_svg && geometry.width > 0 && geometry.height >= 0 &&
      [...geometry.curve_box, ...geometry.svg_box].every(Number.isFinite),
      `expected painted native curve within chart viewport: ${JSON.stringify(geometry)}`);
    summary.metrics.push({
      tag: metric.tag, category: prefix, pagination_advanced: advance,
      axes: metric.axes, native_curve_path: curvePath, native_curve_geometry: geometry,
      native_csv_link: href, native_csv_url: url,
      csv, rows, expected_native_history: histories[metric.tag], plot_screenshot: plot, download_budget: budget,
    });
  }
  await save(output, 'tensorboard-scalars-summary.json', Buffer.from(`${JSON.stringify(summary, null, 2)}\n`));
  return summary;
}

module.exports = { runTensorBoardScalars };
