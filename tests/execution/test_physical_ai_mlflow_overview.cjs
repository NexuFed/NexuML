'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const crypto = require('node:crypto');
const { runMLflowOverview } = require('../fixtures/physical_ai_mlflow_overview.cjs');

const OUTPUT = '/tmp/opencode/mock-overview-output';
const SUMMARY = path.join(OUTPUT, 'mlflow-overview.json');
const FRAMES = 'train/rl/frames';
const RETURNS = 'eval/rl/return_mean';
const BUDGET = 'scenario/training/total_frames';
const ENVIRONMENT = 'scenario/interaction/environment/params/env_name';
const EXPERIMENT = 'scenario/logging/experiment_name';
const KEYS = [FRAMES, RETURNS, BUDGET, ENVIRONMENT, EXPERIMENT];
const STEMS = ['metric-train-frames', 'metric-return', 'param-frame-budget', 'param-environment', 'param-experiment'];
const sha256 = (bytes) => crypto.createHash('sha256').update(bytes).digest('hex');

function fixture() {
  const manifest = { modes: {} };
  for (const [index, mode] of ['mlflow-only', 'both'].entries()) {
    const media = { frame_step: 64, captured_frames: 10, fps: 50, provenance: {},
      metrics: { [RETURNS]: 9.5 }, gif: { frames: 9, size: [128, 85], changing_nonuniform_frames: true },
      mp4: { frames: 10, codec: 'h264', size: [128, 86] }, files: {} };
    for (const ext of ['gif', 'mp4', 'json']) {
      media.files[ext] = { path: `/producer/${mode}/frame_000000000064.${ext}`, sha256: 'c'.repeat(64) };
    }
    const main = { run_id: (index ? 'b' : 'a').repeat(32), experiment_id: String(index + 1), status: 'FINISHED',
      scalar_history: { [FRAMES]: [{ step: 0, value: 32 }, { step: 1, value: 64 }],
        [RETURNS]: [{ step: 0, value: 2.5 }, { step: 32, value: 5 }, { step: 64, value: 9.5 }] },
      media: { 64: media } };
    manifest.modes[mode] = { logging: { experiment_name: `physical-ai-logging-${mode}` },
      configs: { 'case.yaml': { path: `/producer/${mode}/case.yaml`, sha256: (index ? 'e' : 'd').repeat(64) } },
      native: { mlflow: { main } } };
    if (mode === 'both') manifest.modes[mode].native.tensorboard = { main: { media: { 64: structuredClone(media) } } };
  }
  return manifest;
}

function mockFilesystem(t, options = {}) {
  const saved = new Map(), calls = [];
  t.mock.method(fs, 'lstat', async (directory) => {
    calls.push(['lstat', directory]);
    if (options.statError) throw options.statError;
    const stat = { isDirectory: () => true, isSymbolicLink: () => false,
      uid: process.getuid(), mode: 0o40700, dev: 1, ino: 2 };
    return Object.assign(stat, options.stat?.(directory, calls) || {});
  });
  t.mock.method(fs, 'realpath', async (directory) => { calls.push(['realpath', directory]); return options.realpath || directory; });
  t.mock.method(fs, 'readdir', async (directory) => { calls.push(['readdir', directory]); return options.entries || []; });
  t.mock.method(fs, 'readFile', async () => { throw new Error('source YAML/files must not be read'); });
  t.mock.method(fs, 'open', async (filename, flags, mode) => {
    calls.push(['open', filename, flags, mode]);
    assert.equal(path.dirname(filename), OUTPUT);
    assert.equal(flags, 'wx');
    assert.equal(mode, 0o600);
    assert(!saved.has(filename), 'exclusive writes only');
    if (options.openError) throw options.openError;
    return {
      writeFile: async (bytes) => { calls.push(['write', filename]); saved.set(filename, bytes); },
      close: async () => { calls.push(['close', filename]); },
    };
  });
  return { saved, calls };
}

// Public locator doubles only: a row is invisible until its real scroll method is invoked.
function fakePage(manifest, mode = 'mlflow-only') {
  const main = manifest.modes[mode].native.mlflow.main;
  const state = { url: `http://127.0.0.1:5001/#/experiments/${main.experiment_id}/runs/${main.run_id}`,
    title: `Run ${main.run_id} - MLflow`, selected: 'true', calls: 0, actions: [], current: null,
    status: 'Finished', id: main.run_id, hiddenInitial: false, rows: {}, pngs: new Map() };
  const values = ['64', String(main.scalar_history[RETURNS].at(-1).value), '64', 'CartPole-v1', `physical-ai-logging-${mode}`];
  for (const [index, key] of KEYS.entries()) state.rows[key] = [[key, values[index]]];
  const visible = (key, what) => {
    assert.equal(state.current, key, 'actual public row scroll must precede visibility/value/screenshot');
    state.actions.push([what, key]);
  };
  const waitOptions = (options) => assert.deepEqual(options, { state: 'visible', timeout: 30000 });
  function png(key, kind, options) {
    assert.deepEqual(options, { type: 'png', timeout: 30000 });
    visible(key, `${kind}-png`);
    const bytes = Buffer.from(`mock unchanged native ${kind} PNG: ${key}`);
    state.pngs.set(`${key}:${kind}`, bytes);
    state.onScreenshot?.(kind, key);
    return bytes;
  }
  const page = {
    url: () => { state.calls++; return state.url; },
    title: async () => { state.calls++; return state.title; },
    getByText: (text, options) => {
      state.calls++;
      assert.deepEqual(options, { exact: true });
      assert.equal(state.current, null, 'UUID/status visibility is initial only, not required below fold');
      return {
        count: async () => Number(text === state.id || text === state.status),
        waitFor: async (options) => {
          waitOptions(options);
          if (state.hiddenInitial) throw new Error('initial text hidden');
          state.actions.push(['initial-visible', text]);
        },
      };
    },
    getByRole: (role, options) => {
      state.calls++;
      if (role === 'tab') {
        assert.deepEqual(options, { name: 'Overview', exact: true });
        return { count: async () => 1, getAttribute: async (name) => {
          assert.equal(name, 'aria-selected'); return state.selected;
        } };
      }
      if (role === 'cell') {
        assert.equal(options.exact, true);
        return { key: options.name, role };
      }
      assert.equal(role, 'row');
      assert.equal(options, undefined);
      return { filter: ({ has }) => {
        assert.equal(has.role, 'cell');
        const key = has.key;
        return {
          count: async () => state.rows[key]?.length || 0,
          scrollIntoViewIfNeeded: async (options) => {
            assert.deepEqual(options, { timeout: 30000 });
            state.current = key; state.actions.push(['scroll', key]);
          },
          waitFor: async (options) => { waitOptions(options); visible(key, 'row-visible'); },
          getByRole: (role) => {
            assert.equal(role, 'cell');
            return {
              count: async () => state.rows[key][0].length,
              nth: (index) => {
                assert.equal(index, 1, 'second cell is the actual VALUE column');
                return {
                  waitFor: async (options) => { waitOptions(options); visible(key, 'value-visible'); },
                  innerText: async () => { visible(key, 'read-value'); return state.rows[key][0][index]; },
                };
              },
            };
          },
          screenshot: async (options) => png(key, 'row', options),
        };
      } };
    },
    screenshot: async (options) => png(state.current, 'viewport', options),
  };
  return { page, state };
}

for (const mode of ['mlflow-only', 'both']) {
  test(`${mode}: real scroll order, exact VALUE column, unchanged row/viewport bytes, summary last`, async (t) => {
    const manifest = fixture();
    const { page, state } = fakePage(manifest, mode);
    const { saved, calls } = mockFilesystem(t);
    if (mode === 'both') delete manifest.modes['mlflow-only'];
    const result = mode === 'mlflow-only' ? await runMLflowOverview(page, manifest, OUTPUT) :
      await runMLflowOverview(page, manifest, OUTPUT, mode);
    assert.equal(result.mode, mode);
    assert.equal(result.run_id, manifest.modes[mode].native.mlflow.main.run_id);
    assert.equal(result.experiment_id, manifest.modes[mode].native.mlflow.main.experiment_id);
    assert.deepEqual(result.source_config, manifest.modes[mode].configs['case.yaml']);
    assert.deepEqual(result.native_histories, manifest.modes[mode].native.mlflow.main.scalar_history);
    assert.match(result.axis_units[FRAMES].step, /not physical frames/);
    assert.match(result.scope, /fixed G1.*UNVERIFIED/);
    assert.match(result.config_scope, /YAML not read; not all 119/);
    assert.equal(result.status.actual, 'Finished');
    assert.deepEqual(result.rows.map(({ key }) => key), KEYS);
    assert.deepEqual(result.rows.map(({ expected }) => expected), [64, 9.5, 64, 'CartPole-v1', `physical-ai-logging-${mode}`]);
    assert.deepEqual(state.actions.slice(2), KEYS.flatMap((key) =>
      ['scroll', 'row-visible', 'value-visible', 'read-value', 'row-png', 'viewport-png'].map((action) => [action, key])));
    assert.equal(saved.size, 11);
    for (const [index, row] of result.rows.entries()) {
      assert.equal(row.actual, String(row.expected));
      for (const [descriptor, kind, filename] of [[row.screenshot, 'row', `${STEMS[index]}-row.png`],
        [row.viewport_screenshot, 'viewport', `${STEMS[index]}.png`]]) {
        assert.equal(descriptor.filename, filename);
        const bytes = state.pngs.get(`${row.key}:${kind}`);
        assert.strictEqual(saved.get(path.join(OUTPUT, filename)), bytes, 'native bytes passed through unchanged');
        assert.equal(descriptor.bytes, bytes.length);
        assert.equal(descriptor.sha256, sha256(bytes));
      }
    }
    assert.deepEqual(JSON.parse(saved.get(SUMMARY)), result);
    assert.equal([...saved.keys()].at(-1), SUMMARY);
    assert.deepEqual(calls.filter(([action]) => action === 'open').map(([, filename]) => filename), [...saved.keys()]);
    assert.equal(calls.filter(([action]) => action === 'close').length, 11);
    for (const ancestor of [OUTPUT, '/tmp/opencode', '/tmp', '/']) assert(calls.some(([action, value]) => action === 'lstat' && value === ancestor));
  });
}

test('latest return expectation comes from selected native history, with only outer whitespace trimmed', async (t) => {
  const manifest = fixture();
  manifest.modes.both.native.mlflow.main.scalar_history[RETURNS].at(-1).value = 10.25;
  const { page, state } = fakePage(manifest, 'both');
  state.rows[RETURNS][0][1] = ' \n10.25\t';
  const { saved } = mockFilesystem(t);
  const result = await runMLflowOverview(page, manifest, OUTPUT, 'both');
  assert.equal(result.rows[1].expected, 10.25);
  assert.equal(result.rows[1].actual, '10.25');
  assert(saved.has(SUMMARY));
});

const manifestFailures = [
  ['wrong status', (m) => { m.native.mlflow.main.status = 'RUNNING'; }],
  ['wrong run ID', (m) => { m.native.mlflow.main.run_id = 'invalid'; }],
  ['missing main (no child fallback)', (m) => { m.native.mlflow.child = m.native.mlflow.main; delete m.native.mlflow.main; }],
  ['missing frame history', (m) => { delete m.native.mlflow.main.scalar_history[FRAMES]; }],
  ['empty history', (m) => { m.native.mlflow.main.scalar_history[RETURNS] = []; }],
  ['nonfinite value', (m) => { m.native.mlflow.main.scalar_history[RETURNS][0].value = NaN; }],
  ['nonfinite step', (m) => { m.native.mlflow.main.scalar_history[RETURNS][0].step = Infinity; }],
  ['string value', (m) => { m.native.mlflow.main.scalar_history[FRAMES][0].value = '32'; }],
  ['reversed history', (m) => { m.native.mlflow.main.scalar_history[RETURNS].reverse(); }],
  ['duplicate step', (m) => { m.native.mlflow.main.scalar_history[FRAMES][1].step = 0; }],
  ['wrong fixed G1 frames', (m) => { m.native.mlflow.main.scalar_history[FRAMES][1].value = 65; }],
  ['missing source config', (m) => { delete m.configs['case.yaml']; }],
  ['wrong source config basename', (m) => { m.configs['case.yaml'].path = '/producer/eval-case.yaml'; }],
  ['invalid source hash', (m) => { m.configs['case.yaml'].sha256 = 'bad'; }],
  ['wrong logger label', (m) => { m.logging.experiment_name = 'physical-ai-logging-mlflow-only'; }],
  ['missing logger label', (m) => { delete m.logging.experiment_name; }],
  ['both parity mismatch', (m) => { m.native.tensorboard.main.media[64].files.gif.sha256 = 'f'.repeat(64); }],
];
for (const [name, mutate] of manifestFailures) {
  test(`manifest admission rejects ${name} before FS/UI, never falls back to valid default mode`, async (t) => {
    const manifest = fixture();
    const { page, state } = fakePage(manifest, 'both');
    const { saved, calls } = mockFilesystem(t);
    mutate(manifest.modes.both);
    await assert.rejects(runMLflowOverview(page, manifest, OUTPUT, 'both'));
    assert.equal(calls.length, 0);
    assert.equal(state.calls, 0);
    assert(!saved.has(SUMMARY));
  });
}

test('missing selected both mode cannot fall back to mlflow-only', async (t) => {
  const manifest = fixture();
  const { page, state } = fakePage(manifest);
  const { calls } = mockFilesystem(t);
  delete manifest.modes.both;
  await assert.rejects(runMLflowOverview(page, manifest, OUTPUT, 'both'));
  assert.equal(calls.length, 0); assert.equal(state.calls, 0);
});

for (const mode of ['tb-only', 'disabled', '', null]) {
  test(`unsupported mode ${JSON.stringify(mode)} rejected before FS/UI`, async (t) => {
    const manifest = fixture();
    const { page, state } = fakePage(manifest);
    const { saved, calls } = mockFilesystem(t);
    await assert.rejects(runMLflowOverview(page, manifest, OUTPUT, mode), /unsupported mode/);
    assert.equal(calls.length, 0); assert.equal(state.calls, 0); assert.equal(saved.size, 0);
  });
}

for (const output of ['relative', '/tmp/opencode/../output', `${OUTPUT}/`, '', null]) {
  test(`noncanonical output ${JSON.stringify(output)} rejected before FS/UI`, async (t) => {
    const manifest = fixture();
    const { page, state } = fakePage(manifest);
    const { calls } = mockFilesystem(t);
    await assert.rejects(runMLflowOverview(page, manifest, output), /canonical absolute output/);
    assert.equal(calls.length, 0); assert.equal(state.calls, 0);
  });
}

test('invalid page input rejects before FS', async (t) => {
  const { calls } = mockFilesystem(t);
  await assert.rejects(runMLflowOverview({}, fixture(), OUTPUT), /public native page API/);
  assert.equal(calls.length, 0);
});

for (const [name, options] of [
  ['nonempty output', { entries: ['existing.png'] }],
  ['missing output', { statError: new Error('ENOENT caller directory absent') }],
  ['wrong permissions', { stat: (p) => p === OUTPUT ? { mode: 0o40755 } : {} }],
  ['wrong owner', { stat: (p) => p === OUTPUT ? { uid: process.getuid() + 1 } : {} }],
  ['symlink output', { stat: (p) => p === OUTPUT ? { isSymbolicLink: () => true } : {} }],
  ['symlink ancestor', { stat: (p) => p === '/tmp/opencode' ? { isSymbolicLink: () => true } : {} }],
  ['file output', { stat: (p) => p === OUTPUT ? { isDirectory: () => false } : {} }],
  ['different realpath', { realpath: '/elsewhere' }],
]) {
  test(`${name} rejected before UI and without summary`, async (t) => {
    const manifest = fixture();
    const { page, state } = fakePage(manifest);
    const { saved } = mockFilesystem(t, options);
    await assert.rejects(runMLflowOverview(page, manifest, OUTPUT));
    assert.equal(state.calls, 0); assert.equal(saved.size, 0);
  });
}

for (const [name, mutate] of [
  ['wrong metric', (s) => { s.rows[FRAMES][0][1] = '63'; }],
  ['formatted metric is not exact', (s) => { s.rows[FRAMES][0][1] = '64.0'; }],
  ['nearby return is not tolerated', (s) => { s.rows[RETURNS][0][1] = '9.5000001'; }],
  ['wrong budget config', (s) => { s.rows[BUDGET][0][1] = '128'; }],
  ['wrong environment config', (s) => { s.rows[ENVIRONMENT][0][1] = 'CartPole-v0'; }],
  ['wrong experiment config', (s) => { s.rows[EXPERIMENT][0][1] = 'physical-ai-logging-both'; }],
  ['duplicate row', (s) => { s.rows[RETURNS].push([...s.rows[RETURNS][0]]); }],
  ['missing row', (s) => { delete s.rows[ENVIRONMENT]; }],
  ['three cells', (s) => { s.rows[FRAMES][0].push('64'); }],
  ['one cell', (s) => { s.rows[FRAMES][0].pop(); }],
  ['wrong run URL', (s) => { s.url = s.url.replace('a'.repeat(32), 'c'.repeat(32)); }],
  ['artifact URL not root Overview', (s) => { s.url += '/artifacts'; }],
  ['wrong experiment URL', (s) => { s.url = s.url.replace('/experiments/1/', '/experiments/9/'); }],
  ['wrong run title', (s) => { s.title = 'Run other - MLflow'; }],
  ['unselected Overview', (s) => { s.selected = 'false'; }],
  ['wrong visible UUID', (s) => { s.id = 'c'.repeat(32); }],
  ['wrong visible status', (s) => { s.status = 'Running'; }],
  ['initial UUID/status hidden', (s) => { s.hiddenInitial = true; }],
  ['context changes during screenshot', (s) => { s.onScreenshot = () => { s.selected = 'false'; }; }],
]) {
  test(`${name} rejects, keeps partial files, and never writes success summary`, async (t) => {
    const manifest = fixture();
    const { page, state } = fakePage(manifest);
    const { saved } = mockFilesystem(t);
    mutate(state);
    await assert.rejects(runMLflowOverview(page, manifest, OUTPUT));
    assert(!saved.has(SUMMARY));
    if (name === 'wrong experiment config') assert.equal(saved.size, 8);
    if (name === 'context changes during screenshot') assert.equal(saved.size, 2);
  });
}

test('directory replacement before save rejects instead of writing through changed output', async (t) => {
  let checks = 0;
  const { saved } = mockFilesystem(t, { stat: (p) => p === OUTPUT ? { ino: ++checks === 1 ? 2 : 3 } : {} });
  const { page } = fakePage(fixture());
  await assert.rejects(runMLflowOverview(page, fixture(), OUTPUT), /output directory changed/);
  assert.equal(saved.size, 0);
});

test('exclusive-open error propagates unchanged, without summary', async (t) => {
  const error = new Error('EEXIST no overwrite');
  const { saved } = mockFilesystem(t, { openError: error });
  const { page } = fakePage(fixture());
  await assert.rejects(runMLflowOverview(page, fixture(), OUTPUT), (actual) => actual === error);
  assert.equal(saved.size, 0);
});

test('screenshot error propagates unchanged and retains preceding native row bytes', async (t) => {
  const error = new Error('native screenshot failed');
  const { saved } = mockFilesystem(t);
  const { page } = fakePage(fixture());
  page.screenshot = async () => { throw error; };
  await assert.rejects(runMLflowOverview(page, fixture(), OUTPUT), (actual) => actual === error);
  assert.equal(saved.size, 1);
  assert(!saved.has(SUMMARY));
});
