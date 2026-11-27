'use strict';

const fs = require('node:fs/promises');
const assert = require('node:assert/strict');
const { test } = require('node:test');
const { EventEmitter } = require('node:events');
const { createHash } = require('node:crypto');
const { runTensorBoardScalars } = require('../fixtures/physical_ai_tensorboard_scalars.cjs');

const RUN = 'physical-ai-logging-tb-only/cartpole-ppo_v1';
const ORIGIN = 'http://127.0.0.1:6007';
const OUTPUT = '/tmp/opencode/mock-owned-scalars';
const TRAIN = 'train/rl/frames';
const EVAL = 'eval/rl/return_mean';
const GUID = '01234567-0123-4567-89ab-0123456789ab';
let activeBytes, nativeCompleted;

function manifest(mode = 'tb-only') {
  const run = `physical-ai-logging-${mode}/cartpole-ppo_v1`;
  return { modes: { [mode]: {
    logging: { tensorboard: { log_dir: '/source/tb' } },
    native: { tensorboard: { main: {
      log_dir: `/source/tb/${run}`,
      scalar_history: {
        [TRAIN]: [{ step: 0, value: 32 }, { step: 1, value: 64 }],
        [EVAL]: [{ step: 0, value: 9.5 }, { step: 32, value: 9.5 }, { step: 64, value: 9.5 }],
      },
    } } },
  } } };
}

// Memory-only filesystem: tests never create output or alter app/browser state.
async function withFS(options, check) {
  const originals = {};
  const writes = new Map();
  activeBytes = undefined;
  nativeCompleted = false;
  const fake = {
    lstat: async (directory) => {
      assert.equal(directory, OUTPUT);
      return {
        isDirectory: () => !options.notDirectory, isSymbolicLink: () => !!options.symlink,
        uid: options.uid ?? process.getuid(), mode: options.mode ?? 0o40700,
      };
    },
    realpath: async () => options.realpath ?? OUTPUT,
    readdir: async () => options.entries ?? [],
    mkdtemp: async prefix => { assert.equal(prefix, `${OUTPUT}/.download-`); return `${OUTPUT}/.download-owned`; },
    open: async filename => {
      assert.equal(filename, `${OUTPUT}/.download-owned/${GUID}`);
      assert(nativeCompleted, 'file cannot be opened before native completion');
      options.reads = (options.reads ?? 0) + 1;
      if (options.noFile) throw Object.assign(new Error('native CSV file missing'), { code: 'ENOENT' });
      return {
        stat: async () => ({ isFile: () => true, uid: process.getuid(), size: activeBytes.length }),
        read: async (buffer, offset, length) => ({ bytesRead: activeBytes.copy(buffer, offset, 0, length) }),
        close: async () => {},
      };
    },
    unlink: async filename => {
      assert([`${OUTPUT}/.download-owned/${GUID}`, `${OUTPUT}/.download-owned/${GUID}.crdownload`].includes(filename));
      (options.unlinked ??= []).push(filename);
    },
    rmdir: async directory => { assert.equal(directory, `${OUTPUT}/.download-owned`); options.removed = (options.removed ?? 0) + 1; },
    writeFile: async (filename, bytes, flags) => {
      assert.deepEqual(flags, { flag: 'wx', mode: 0o600 });
      assert([
        'train-rl-frames.csv', 'train-rl-frames.png', 'eval-rl-return-mean.csv',
        'eval-rl-return-mean.png', 'tensorboard-scalars-summary.json',
      ].some((name) => filename === `${OUTPUT}/${name}`), 'only fixed literal filenames');
      if (options.collision || writes.has(filename)) throw Object.assign(new Error('exclusive collision'), { code: 'EEXIST' });
      writes.set(filename, Buffer.from(bytes));
    },
  };
  for (const [key, value] of Object.entries(fake)) { originals[key] = fs[key]; fs[key] = value; }
  try { await check(writes); } finally { Object.assign(fs, originals); }
}

function mockPage(source, options = {}, mode = 'tb-only') {
  const run = `physical-ai-logging-${mode}/cartpole-ppo_v1`;
  const events = [];
  let activeTag;
  let selected = false;
  let ignoreOutliers = true;
  let smoothing = '0.6';
  let requestListener;
  let allowed = false;
  const browser = new EventEmitter();
  browser.send = async (method, params) => {
    assert.equal(params.browserContextId, 'owned-context');
    if (method === 'Browser.setDownloadBehavior') {
      if (params.behavior === 'allowAndName') {
        assert.equal(params.downloadPath, `${OUTPUT}/.download-owned`);
        assert.equal(params.eventsEnabled, true);
        allowed = true;
      } else { assert.equal(params.behavior, 'deny'); allowed = false; }
      events.push(`protocol:${params.behavior}`);
    } else {
      assert.equal(method, 'Browser.cancelDownload');
      assert.equal(params.guid, GUID);
      events.push('protocol:cancel');
      if (options.cancelFailure) throw new Error('native cancel cleanup failed');
    }
  };
  browser.detach = async () => events.push('protocol:browser-detach');
  const target = {
    send: async method => {
      if (method === 'Target.getTargetInfo') return { targetInfo: { type: 'page', browserContextId: 'owned-context' } };
      assert.equal(method, 'Page.getFrameTree'); return { frameTree: { frame: { id: 'owned-frame' } } };
    },
    detach: async () => events.push('protocol:target-detach'),
  };
  const nativeURL = (tag) => `${ORIGIN}/data/plugin/scalars/scalars?run=${encodeURIComponent(run)}&tag=${encodeURIComponent(tag)}&format=csv`;
  const page = {
    context: () => ({ browser: () => ({ newBrowserCDPSession: async () => browser }), newCDPSession: async () => target }),
    url: () => options.pageURL ?? `${ORIGIN}/#scalars`,
    getByText: (tag, config) => { assert.deepEqual(config, { exact: true }); return { tag }; },
    locator: (selector) => {
      events.push(`selector:${selector}`);
      const heading = /^span\.category-name\[title="(train|eval)"\]$/.exec(selector);
      if (heading) return { prefix: heading[1] };
      if (selector === 'tf-scalar-dashboard:visible tf-smoothing-input input[type="number"]') return {
        fill: async value => { assert.equal(value, '0'); if (!options.smoothingStuck) smoothing = value; events.push('raw-smoothing'); },
        press: async key => { assert.equal(key, 'Tab'); },
        inputValue: async () => smoothing,
      };
      assert.equal(selector, 'tf-scalar-dashboard:visible tf-category-paginated-view', 'native category scope must exclude additive search results');
      return { filter: ({ has: category }) => {
        let nextPage = false;
        return {
          count: async () => options.categoryCount ?? 1, waitFor: async () => {},
          getByRole: (role, config) => {
            assert.equal(role, 'button'); assert.deepEqual(config, { name: 'Next page', exact: true });
            return { count: async () => options.nextCount ?? 2, nth: index => {
              assert.equal(index, 0, 'paired native pagination buttons have the same target');
              return { click: async () => { nextPage = true; events.push(`${activeTag}:next-page`); } };
            } };
          },
          locator: child => {
            assert.equal(child, 'tf-scalar-card:visible');
            return { filter: ({ has }) => {
              activeTag = has.tag;
              selected = false;
              const tag = activeTag;
              let fitted = false;
              assert.equal(category.prefix, tag.split('/')[0]);
              const count = () => options.cardCount ?? (tag === EVAL && !nextPage ? 0 : 1);
              return {
                waitFor: async (config) => { assert.equal(config.state, 'visible'); assert.equal(count(), 1, 'expected one native card in the selected category/page'); events.push(`${tag}:card`); },
                count: async () => count(),
                scrollIntoViewIfNeeded: async () => { events.push(`${tag}:scroll-chart`); },
                locator: (child) => {
                  if (child === 'vz-line-chart2') return { waitFor: async () => { events.push(`${tag}:chart`); } };
                  if (child === 'vz-line-chart2 .ghost') return {
                    waitFor: async config => { assert.equal(config.state, 'hidden'); assert.equal(smoothing, '0'); events.push(`${tag}:raw-chart`); },
                  };
                  if (child === 'paper-icon-button[title="Fit domain to data"]') return {
                    click: async () => { fitted = true; events.push(`${tag}:fit-domain`); },
                  };
                  if (child === 'vz-line-chart2 path.line[d*="L"]') return {
                    count: async () => 1,
                    waitFor: async config => {
                      assert.equal(config.state, 'attached', 'horizontal SVG strokes have zero bounding-box height');
                      assert.equal(ignoreOutliers, false, 'two-point history is hidden by native percentile scaling');
                      assert.equal(smoothing, '0', 'raw scalar inspection must not substitute smoothed values');
                      if (options.emptyPlot) throw new Error('native chart has no rendered data curve');
                      events.push(`${tag}:chart-data`);
                    },
                    getAttribute: async name => { assert.equal(name, 'd'); return tag === EVAL ? 'M10,80L200,80' : 'M10,100L200,10'; },
                    evaluate: async fn => {
                      assert(fitted, 'native raw data domain must be fitted after smoothing finishes');
                      const old = globalThis.getComputedStyle;
                      globalThis.getComputedStyle = () => ({ display: options.hiddenCurve ? 'none' : 'inline', visibility: 'visible', opacity: '1', stroke: 'rgb(0, 119, 187)', strokeOpacity: '1', strokeWidth: '2px' });
                      try {
                        return fn({
                          isConnected: true, parentElement: null, getRootNode: () => ({}),
                          getBoundingClientRect: () => ({ left: 10, right: 200, top: options.clippedCurve ? -20 : 20, bottom: tag === EVAL ? 20 : 110, width: 190, height: tag === EVAL ? 0 : 90 }),
                          ownerSVGElement: { getBoundingClientRect: () => ({ left: 0, right: 330, top: 0, bottom: 200, width: 330, height: 200 }) },
                        });
                      } finally { if (old === undefined) delete globalThis.getComputedStyle; else globalThis.getComputedStyle = old; }
                    },
                  };
                  assert.equal(child, 'tf-downloader paper-dropdown-menu');
                  return {
                    waitFor: async () => {}, count: async () => 1,
                    click: async () => { events.push(`${tag}:menu`); },
                  };
                },
                getByRole: (role, config) => {
                  assert.equal(role, 'link');
                  assert.deepEqual(config, { name: 'CSV', exact: true });
                  assert(selected, 'CSV links cannot be read before native run selection');
                  return {
                    waitFor: async () => {},
                    getAttribute: async (name) => name === 'download' ? (options.noDownload ? null : '') : (options.href ?? nativeURL(tag)),
                    click: async () => {
                      assert(allowed && browser.listenerCount('Browser.downloadProgress') === 1, 'active budget must precede native CSV click');
                      events.push(`${tag}:csv-click`);
                      if (options.redirect) {
                        const initial = { url: () => nativeURL(tag), redirectedFrom: () => null };
                        requestListener({ url: () => 'https://foreign.invalid/file', redirectedFrom: () => initial });
                      }
                      const history = source.modes[mode].native.tensorboard.main.scalar_history[tag];
                      const csv = options.csv ?? `Wall time,Step,Value\n${history.map(({ step, value }) => `1792937017.25,${step},${value}`).join('\n')}\n`;
                      activeBytes = options.oversize ? Buffer.alloc(1024 * 1024 + 1) : Buffer.from(csv);
                      nativeCompleted = false;
                      browser.emit('Browser.downloadWillBegin', { frameId: 'owned-frame', guid: GUID, url: options.downloadURL ?? nativeURL(tag) });
                      if (options.neverComplete) return;
                      if (options.oversize || options.declaredOversize) {
                        browser.emit('Browser.downloadProgress', { guid: GUID, state: 'inProgress', receivedBytes: activeBytes.length, totalBytes: options.declaredOversize ? 1024 * 1024 + 1 : 0 });
                      } else {
                        nativeCompleted = !options.failure;
                        browser.emit('Browser.downloadProgress', { guid: GUID, state: options.failure ? 'canceled' : 'completed', receivedBytes: activeBytes.length, totalBytes: activeBytes.length });
                      }
                    },
                  };
                },
                screenshot: async (config) => {
                  assert.deepEqual(config, { type: 'png' });
                  events.push(`${tag}:screenshot`);
                  return Buffer.from('mock native card PNG');
                },
              };
            } };
          },
        };
      } };
    },
    getByRole: (role, config) => {
      if (role === 'checkbox') {
        assert.deepEqual(config, { name: 'Ignore outliers in chart scaling', exact: true });
        return {
          getAttribute: async key => { assert.equal(key, 'aria-checked'); return String(ignoreOutliers); },
          click: async () => {
            assert.equal(smoothing, '0', 'native bounds must reset after smoothing is disabled');
            if (!options.outliersStuck) ignoreOutliers = !ignoreOutliers;
            events.push('show-all-values');
          },
        };
      }
      assert.equal(role, 'option');
      assert.deepEqual(config, { name: run, exact: true });
      return { click: async () => { selected = true; events.push(`${activeTag}:run`); } };
    },
    waitForEvent: () => assert.fail('budget must not wait for Playwright download completion'),
    on: (event, listener) => { assert.equal(event, 'request'); requestListener = listener; },
    off: (event, listener) => { assert.equal(event, 'request'); assert.equal(listener, requestListener); requestListener = undefined; },
  };
  return { page, events, listenerRemoved: () => requestListener === undefined && browser.listenerCount('Browser.downloadProgress') === 0 && browser.listenerCount('Browser.downloadWillBegin') === 0 };
}

async function rejects(name, options, pattern, mutate = () => {}) {
  await test(name, async () => {
    const source = manifest();
    mutate(source);
    const mock = mockPage(source, options);
    await withFS(options, async () => {
      await assert.rejects(runTensorBoardScalars(mock.page, source, options.output ?? OUTPUT), pattern);
      assert(mock.listenerRemoved(), 'request observer must be removed after failure');
    });
  });
}

async function main() {
  await test('native selection/click order, exact histories, axes, hashes and private fixed outputs', async () => {
    const source = manifest();
    const mock = mockPage(source);
    await withFS({}, async (writes) => {
      const result = await runTensorBoardScalars(mock.page, source, OUTPUT);
      assert.equal(result.mode, 'tb-only');
      assert.equal(result.scope, 'native chart CSV/history only; MP4 retrieval and full V/task6.1 remain UNVERIFIED');
      assert.equal(result.relative_run, RUN);
      assert.deepEqual(result.chart_settings, { ignore_outliers: false, smoothing: 0, role: 'raw native scalar inspection; event values unchanged' });
      assert.equal(result.metrics[0].axes.x_axis, 'native logger step (not source physical frames)');
      assert.equal(result.metrics[0].axes.value, 'physical training frames');
      assert.equal(result.metrics[1].axes.x_axis, 'evaluation source physical frames');
      assert.equal(writes.size, 5);
      for (const metric of result.metrics) {
        assert.deepEqual(metric.rows.map(({ step, value }) => ({ step, value })), metric.expected_native_history);
        assert.equal(metric.native_curve_geometry.height, metric.tag === EVAL ? 0 : 90);
        assert(metric.native_curve_geometry.painted && metric.native_curve_geometry.within_svg);
        for (const artifact of [metric.csv, metric.plot_screenshot]) {
          const bytes = writes.get(`${OUTPUT}/${artifact.file}`);
          assert.equal(artifact.sha256, createHash('sha256').update(bytes).digest('hex'));
          assert.equal(artifact.bytes, bytes.length);
        }
        assert.deepEqual(mock.events.filter((event) => event.startsWith(`${metric.tag}:`)).map((event) => event.slice(metric.tag.length + 1)),
          [...(metric.tag === EVAL ? ['next-page'] : []), 'card', 'scroll-chart', 'chart', 'raw-chart', 'fit-domain', 'chart-data', 'menu', 'run', 'csv-click', 'screenshot']);
      }
      assert.deepEqual(JSON.parse(writes.get(`${OUTPUT}/tensorboard-scalars-summary.json`)), result);
      assert(mock.listenerRemoved());
    });
  });
  await test('active oversized downloads cancel before completion or filesystem reads', async () => {
    for (const options of [{ oversize: true }, { declaredOversize: true }]) {
      const mock = mockPage(manifest(), options);
      await withFS(options, async writes => {
        await assert.rejects(runTensorBoardScalars(mock.page, manifest(), OUTPUT), /exceeds 1 MiB during download/);
        assert(mock.events.includes('protocol:cancel'));
        assert(mock.events.includes('protocol:deny'));
        assert.equal(options.reads ?? 0, 0);
        assert.equal(options.removed, 1);
        assert.equal(options.unlinked.length, 2);
        assert.equal(writes.size, 0);
        assert(mock.listenerRemoved());
      });
    }
  });
  await test('named categories and native pagination exclude additive search duplicates', async () => {
    const options = {};
    const mock = mockPage(manifest(), options);
    await withFS(options, async () => {
      const result = await runTensorBoardScalars(mock.page, manifest(), OUTPUT);
      assert.equal(result.metrics.length, 2);
      assert(mock.events.includes(`${EVAL}:next-page`));
      assert(mock.events.includes('show-all-values') && mock.events.includes('raw-smoothing'));
      assert(!mock.events.some(event => event.includes('tf-tag-filterer')));
      assert.deepEqual(result.metrics.map(({ category, pagination_advanced }) => ({ category, pagination_advanced })),
        [{ category: 'train', pagination_advanced: false }, { category: 'eval', pagination_advanced: true }]);
    });
  });
  await test('stalled downloads have a completion deadline and owned cancellation', async t => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const options = { neverComplete: true };
    const mock = mockPage(manifest(), options);
    await withFS(options, async writes => {
      const pending = runTensorBoardScalars(mock.page, manifest(), OUTPUT);
      await new Promise(setImmediate);
      assert(mock.events.includes(`${TRAIN}:csv-click`));
      t.mock.timers.tick(30000);
      await assert.rejects(pending, /download completion deadline exceeded/);
      assert(mock.events.includes('protocol:cancel'));
      assert(mock.events.includes('protocol:deny'));
      assert.equal(options.reads ?? 0, 0);
      assert.equal(options.removed, 1);
      assert.equal(writes.size, 0);
      assert(mock.listenerRemoved());
    });
  });
  await test('cancel cleanup errors do not replace the original budget failure', async () => {
    const options = { oversize: true, cancelFailure: true };
    const mock = mockPage(manifest(), options);
    await withFS(options, async () => {
      await assert.rejects(runTensorBoardScalars(mock.page, manifest(), OUTPUT), error =>
        /exceeds 1 MiB during download/.test(error.message) && error.cleanupErrors.some(item => item.includes('native cancel cleanup failed')));
      assert(mock.events.includes('protocol:deny'));
      assert.equal(options.removed, 1);
    });
  });
  await rejects('wrong native run', {}, /unexpected native relative run/, (m) => { m.modes['tb-only'].native.tensorboard.main.log_dir = '/source/tb/wrong'; });
  await rejects('missing named native metric', {}, /missing native history/, (m) => { delete m.modes['tb-only'].native.tensorboard.main.scalar_history[EVAL]; });
  await rejects('nonfinite native value', {}, /invalid native value/, (m) => { m.modes['tb-only'].native.tensorboard.main.scalar_history[EVAL][0].value = Infinity; });
  await rejects('negative native step', {}, /invalid native step/, (m) => { m.modes['tb-only'].native.tensorboard.main.scalar_history[EVAL][0].step = -1; });
  await rejects('duplicate native step', {}, /invalid native step/, (m) => { m.modes['tb-only'].native.tensorboard.main.scalar_history[EVAL][1].step = 0; });
  await rejects('static physical training frames', {}, /physical frames must increase/, (m) => { m.modes['tb-only'].native.tensorboard.main.scalar_history[TRAIN][1].value = 32; });
  await rejects('fractional physical training frames', {}, /physical frames must increase/, (m) => { m.modes['tb-only'].native.tensorboard.main.scalar_history[TRAIN][0].value = 1.5; });
  for (const [name, options, pattern] of [
    ['relative output', { output: 'relative' }, /invalid output path/],
    ['output traversal', { output: `${OUTPUT}/../other` }, /invalid output path/],
    ['non-directory output', { notDirectory: true }, /real directory/],
    ['symlink output', { symlink: true }, /real directory/],
    ['wrong output owner', { uid: process.getuid() + 1 }, /owned by this user/],
    ['public output', { mode: 0o40755 }, /mode 0700/],
    ['symlink ancestor', { realpath: '/elsewhere' }, /must not contain symlinks/],
    ['nonempty output', { entries: ['existing'] }, /must be empty/],
    ['exclusive write collision', { collision: true }, /exclusive collision/],
    ['foreign page', { pageURL: 'https://foreign.invalid/#scalars' }, /owned Scalars page/],
    ['native outlier control failure', { outliersStuck: true }, /include all scalar values/],
    ['native smoothing control failure', { smoothingStuck: true }, /smoothing must be disabled/],
    ['missing named category', { categoryCount: 0 }, /one native category/],
    ['duplicate named category', { categoryCount: 2 }, /one native category/],
    ['unexpected native pagination controls', { nextCount: 1 }, /paired native pagination/],
    ['metric missing after native pagination', { cardCount: 0 }, /one native card/],
    ['empty native chart is not scalar visualization proof', { emptyPlot: true }, /native chart has no rendered data curve/],
    ['hidden SVG curve is not rendered evidence', { hiddenCurve: true }, /painted native curve/],
    ['clipped SVG curve is not full-history visualization', { clippedCurve: true }, /painted native curve/],
    ['duplicate native cards', { cardCount: 2 }, /one native card/],
    ['not a native download link', { noDownload: true }, /download link/],
    ['download URL changed', { downloadURL: 'https://foreign.invalid/file' }, /differs from native CSV href/],
    ['redirect chain', { redirect: true }, /redirects forbidden/],
    ['download failure', { failure: 'aborted' }, /download failed/],
    ['missing native downloaded file', { noFile: true }, /native CSV file missing/],
    ['download over 1 MiB', { oversize: true }, /exceeds 1 MiB/],
  ]) await rejects(name, options, pattern);

  const valid = `${ORIGIN}/data/plugin/scalars/scalars?run=${encodeURIComponent(RUN)}&tag=${encodeURIComponent(TRAIN)}&format=csv`;
  for (const [name, href, pattern] of [
    ['foreign CSV origin', valid.replace(ORIGIN, 'https://foreign.invalid'), /foreign CSV origin/],
    ['CSV credentials', valid.replace('http://', 'http://user:secret@'), /credentials/],
    ['wrong CSV endpoint', valid.replace('/scalars/scalars?', '/scalars/other?'), /endpoint/],
    ['wrong CSV run', valid.replace(encodeURIComponent(RUN), 'other'), /CSV run/],
    ['wrong CSV tag', valid.replace(encodeURIComponent(TRAIN), encodeURIComponent(EVAL)), /CSV tag/],
    ['wrong CSV format', valid.replace('format=csv', 'format=json'), /CSV format/],
    ['duplicate CSV tag', `${valid}&tag=extra`, /query keys/],
    ['missing CSV format', valid.replace('&format=csv', ''), /query keys/],
    ['arbitrary query', `${valid}&filename=evil`, /query keys/],
    ['CSV fragment', `${valid}#fragment`, /fragment/],
  ]) await rejects(name, { href }, pattern);

  for (const [name, csv, pattern] of [
    ['wrong header', 'Wall Time,Step,Value\n1,0,32\n1,1,64\n', /header/],
    ['empty history', 'Wall time,Step,Value\n', /empty CSV history/],
    ['extra column', 'Wall time,Step,Value\n1,0,32,1\n', /numeric row/],
    ['empty value', 'Wall time,Step,Value\n1,0,\n', /numeric row/],
    ['NaN value', 'Wall time,Step,Value\n1,0,NaN\n', /numeric row/],
    ['overflow value', 'Wall time,Step,Value\n1,0,1e999\n', /nonfinite/],
    ['negative wall time', 'Wall time,Step,Value\n-1,0,32\n', /nonfinite\/invalid/],
    ['fractional step', 'Wall time,Step,Value\n1,0.5,32\n', /CSV step/],
    ['negative step', 'Wall time,Step,Value\n1,-1,32\n', /CSV step/],
    ['duplicate step', 'Wall time,Step,Value\n1,0,32\n1,0,64\n', /CSV step/],
    ['descending steps', 'Wall time,Step,Value\n1,1,32\n1,0,64\n', /CSV step/],
    ['missing step', 'Wall time,Step,Value\n1,0,32\n', /history mismatch/],
    ['extra step', 'Wall time,Step,Value\n1,0,32\n1,1,64\n1,2,96\n', /history mismatch/],
    ['wrong step', 'Wall time,Step,Value\n1,0,32\n1,2,64\n', /history mismatch/],
    ['wrong value', 'Wall time,Step,Value\n1,0,32\n1,1,63\n', /history mismatch/],
    ['blank row', 'Wall time,Step,Value\n1,0,32\n\n1,1,64\n', /numeric row/],
  ]) await rejects(name, { csv }, pattern);

  await test('both fake histories and dropdown/CSV remain scoped mock proof', async () => {
    const source = { modes: { ...manifest().modes, ...manifest('both').modes } };
    const expected = {
      [TRAIN]: [{ step: 0, value: 32 }, { step: 1, value: 64 }],
      [EVAL]: [{ step: 0, value: 9.5 }, { step: 32, value: 9.5 }, { step: 64, value: 9.5 }],
    };
    const run = 'physical-ai-logging-both/cartpole-ppo_v1';
    assert.equal(source.modes.both.native.tensorboard.main.log_dir, `${source.modes.both.logging.tensorboard.log_dir}/${run}`);
    for (const tag of [TRAIN, EVAL]) {
      assert.deepEqual(source.modes.both.native.tensorboard.main.scalar_history[tag], expected[tag]);
      // Distinct in-memory decoy proves neither helper nor fake CSV falls back to tb-only.
      source.modes['tb-only'].native.tensorboard.main.scalar_history[tag] = expected[tag].map(({ step, value }) => ({ step, value: value + 1 }));
    }
    const options = {};
    const mock = mockPage(source, options, 'both');
    await withFS(options, async writes => {
      const result = await runTensorBoardScalars(mock.page, source, OUTPUT, 'both');
      assert.equal(result.mode, 'both');
      assert.equal(result.relative_run, run);
      assert.match(result.scope, /full V\/task6.1 remain UNVERIFIED/);
      assert.deepEqual(result.chart_settings, { ignore_outliers: false, smoothing: 0, role: 'raw native scalar inspection; event values unchanged' });
      assert.deepEqual(result.metrics.map(metric => metric.axes), [
        { x_axis: 'native logger step (not source physical frames)', value: 'physical training frames' },
        { x_axis: 'evaluation source physical frames', value: 'mean evaluation episode return' },
      ]);
      for (const metric of result.metrics) {
        assert.equal(new URL(metric.native_csv_url).searchParams.get('run'), run);
        assert.deepEqual(metric.expected_native_history, expected[metric.tag]);
        assert.deepEqual(metric.rows.map(({ step, value }) => ({ step, value })), expected[metric.tag]);
        assert.equal(writes.get(`${OUTPUT}/${metric.csv.file}`).toString(), `Wall time,Step,Value\n${expected[metric.tag].map(({ step, value }) => `1792937017.25,${step},${value}`).join('\n')}\n`);
        assert.equal(metric.native_curve_geometry.height, metric.tag === EVAL ? 0 : 90);
        assert(metric.native_curve_geometry.painted && metric.native_curve_geometry.within_svg);
        assert.deepEqual(metric.download_budget, { limit_bytes: 1024 * 1024, completion_seconds: 30, received_bytes: metric.csv.bytes, status: 'completed', scratch_cleanup: 'PASS' });
        assert.deepEqual(mock.events.filter(event => event.startsWith(`${metric.tag}:`)).map(event => event.slice(metric.tag.length + 1)),
          [...(metric.tag === EVAL ? ['next-page'] : []), 'card', 'scroll-chart', 'chart', 'raw-chart', 'fit-domain', 'chart-data', 'menu', 'run', 'csv-click', 'screenshot']);
      }
      assert.equal(writes.size, 5);
      assert.equal(options.removed, 2);
      assert(mock.listenerRemoved());
      assert.deepEqual(JSON.parse(writes.get(`${OUTPUT}/tensorboard-scalars-summary.json`)), result);
    });
  });
  await test('disabled/unknown scalar modes reject before manifest/output/browser access', async t => {
    const untouched = new Proxy({}, { get: () => assert.fail('invalid mode must be rejected first') });
    t.mock.method(fs, 'lstat', () => assert.fail('no output access'));
    t.mock.method(fs, 'writeFile', () => assert.fail('no output writes'));
    for (const mode of ['disabled', 'mlflow-only', 'unknown', '', '__proto__', null, false, 0, {}, ['both'], new String('both')]) {
      await assert.rejects(runTensorBoardScalars(untouched, untouched, OUTPUT, mode), /mode must be tb-only or both/);
    }
  });
  await test('scalar native run/base cannot cross modes or fall back before output access', async t => {
    t.mock.method(fs, 'lstat', () => assert.fail('no output access'));
    for (const [mode, other] of [['both', 'tb-only'], ['tb-only', 'both']]) {
      const source = manifest(mode);
      source.modes[mode].native.tensorboard.main.log_dir = `/source/tb/physical-ai-logging-${other}/cartpole-ppo_v1`;
      await assert.rejects(runTensorBoardScalars({}, source, OUTPUT, mode), /unexpected native relative run/);
      const wrongBase = manifest(mode);
      wrongBase.modes[mode].logging.tensorboard.log_dir = '/other/tb';
      await assert.rejects(runTensorBoardScalars({}, wrongBase, OUTPUT, mode), /unexpected native relative run/);
      await assert.rejects(runTensorBoardScalars({}, manifest(other), OUTPUT, mode), /missing TensorBoard log_dir/);
    }
  });
  await test('both scalar dropdown never admits a tb-only CSV run', async () => {
    const source = manifest('both');
    const options = { href: valid };
    const mock = mockPage(source, options, 'both');
    await withFS(options, async writes => {
      await assert.rejects(runTensorBoardScalars(mock.page, source, OUTPUT, 'both'), /unexpected CSV run/);
      assert(mock.events.includes(`${TRAIN}:run`));
      assert(!mock.events.includes(`${TRAIN}:csv-click`));
      assert.equal(writes.size, 0);
      assert(mock.listenerRemoved());
    });
  });
  console.log('Mocks only; real UI and full V/task6.1 UNVERIFIED');
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
