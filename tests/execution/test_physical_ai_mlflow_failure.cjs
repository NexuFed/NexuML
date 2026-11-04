'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const crypto = require('node:crypto');
const { runMLflowFailure } = require('../fixtures/physical_ai_mlflow_failure.cjs');

const OUTPUT = '/tmp/opencode/mock-failure-output';
const ANCHOR = '/proc/self/fd/42';
const SUMMARY = path.join(OUTPUT, 'mlflow-failure.json');
const hash = (bytes) => crypto.createHash('sha256').update(bytes).digest('hex');
function fixture() {
  return { run_id: 'a'.repeat(32), experiment_id: '1', status: 'FAILED',
    experiment_name: 'physical-ai-logging-partial-delivery',
    cli: { exit_code: 1, status: 'FAIL', log_path: '/producer/cli.log', log_sha256: 'b'.repeat(64) },
    tensorboard_gif: { path: '/producer/tb/artifacts/evaluation/frame_000000000064.gif', sha256: 'c'.repeat(64) },
    prior_artifacts: ['source_package.json', 'config.yaml'].map((filename) => ({ filename, sha256: 'd'.repeat(64) })) };
}
function filesystem(t, options = {}) {
  const saved = new Map(), calls = [];
  const directoryState = { opened: false, closed: false, replaced: false, anchoredWrites: [] };
  t.after(() => assert.equal(directoryState.opened, directoryState.closed, 'every acquired directory FD must be closed'));
  let checks = 0;
  const stat = () => ({ isDirectory: () => true, isSymbolicLink: () => false,
    uid: process.getuid(), mode: 0o40700, dev: 1, ino: 2 });
  t.mock.method(fs, 'lstat', async (directory) => {
    calls.push(['lstat', directory]);
    if (options.statError) throw options.statError;
    if (directory === OUTPUT) checks++;
    return Object.assign(stat(), directory === OUTPUT && directoryState.replaced ? { ino: 3 } : {}, options.stat?.(directory, checks));
  });
  t.mock.method(fs, 'realpath', async (directory) => { calls.push(['realpath', directory]); return options.realpath || directory; });
  t.mock.method(fs, 'readdir', async (directory) => {
    calls.push(['readdir', directory]); assert.equal(directory, ANCHOR); return options.entries || [];
  });
  t.mock.method(fs, 'readFile', async () => { throw new Error('parent-owned source must not be read'); });
  t.mock.method(fs, 'open', async (filename, flags, mode) => {
    calls.push(['open', filename, flags, mode]);
    if (typeof flags === 'number') {
      assert.equal(filename, OUTPUT);
      assert.equal(flags, fs.constants.O_RDONLY | fs.constants.O_DIRECTORY | fs.constants.O_NOFOLLOW);
      assert.equal(mode, undefined); assert(!directoryState.opened);
      if (options.directoryOpenError) throw options.directoryOpenError;
      directoryState.opened = true;
      return { fd: 42, stat: async () => {
        calls.push(['fstat']);
        if (options.directoryStatError) throw options.directoryStatError;
        return Object.assign(stat(), options.directoryStat);
      }, close: async () => {
        assert(!directoryState.closed); directoryState.closed = true; calls.push(['directory-close']);
        if (options.directoryCloseError) throw options.directoryCloseError;
      } };
    }
    assert.equal(path.dirname(filename), ANCHOR, 'writes must use held directory FD, never output pathname');
    assert(directoryState.opened && !directoryState.closed);
    assert(['failed-run.png', 'artifacts.png', 'mlflow-failure.json'].includes(path.basename(filename)));
    if (options.replaceAtFileOpen) directoryState.replaced = true;
    assert.equal(flags, 'wx'); assert.equal(mode, 0o600);
    const ownedFilename = path.join(OUTPUT, path.basename(filename));
    assert(!saved.has(ownedFilename));
    if (options.openError) throw options.openError;
    return { writeFile: async (bytes) => {
      if (options.writeError) throw options.writeError;
      directoryState.anchoredWrites.push({ ino: 2, pathnameReplaced: directoryState.replaced });
      saved.set(ownedFilename, bytes); calls.push(['write', filename]);
    }, close: async () => { calls.push(['close', filename]); if (options.closeError) throw options.closeError; } };
  });
  return { saved, calls, directoryState };
}
function nativePage(manifest) {
  const root = `http://127.0.0.1:5001/#/experiments/${manifest.experiment_id}/runs/${manifest.run_id}`;
  const state = { url: root, title: `Run ${manifest.run_id} - MLflow`, tab: 'Overview', calls: [], actions: [],
    id: manifest.run_id, status: 'Failed', counts: {}, hidden: new Set(), attrs: {}, forbidden: 0, pngs: [] };
  const wait = (name, options) => {
    assert.deepEqual(options, { state: 'visible', timeout: 30000 });
    if (state.hidden.has(name)) throw new Error(`hidden ${name}`);
    if (state.delayed?.has(name)) { state.counts[name] = 1; state.delayed.delete(name); }
    state.actions.push(['visible', name]);
  };
  function locator(name, attrs = {}) {
    return { count: async () => state.counts[name] ?? 1,
      waitFor: async (options) => wait(name, options),
      getAttribute: async (key) => state.attrs[name]?.[key] ?? attrs[key],
      innerText: async () => state.breadcrumbText ?? manifest.experiment_name };
  }
  const page = {
    url: () => { state.calls.push('url'); return state.url; },
    title: async () => { state.calls.push('title'); return state.title; },
    locator: (selector) => {
      state.calls.push(['locator', selector]);
      if (selector === '[data-testid="experiment-runs-link"]') return locator('breadcrumb', {
        href: `#/experiments/${manifest.experiment_id}/evaluation-runs` });
      if (selector === '[data-testid="key-value-Run ID"]') return {
        ...locator('Run ID field'), getByText: (text, options) => {
          state.calls.push(['scoped-UUID', text]); assert.equal(text, manifest.run_id);
          assert.deepEqual(options, { exact: true });
          return { ...locator(text), count: async () => state.counts[text] ?? Number(text === state.id) };
        },
      };
      if (selector === '[data-testid="key-value-Status"]') return {
        ...locator('Status field'), getByRole: (role) => {
          assert.equal(role, 'status'); state.calls.push(['scoped-status', role]);
          return { ...locator('Failed'), innerText: async () => state.status };
        },
      };
      assert.equal(selector, '[data-testid="artifact-tree-node"][artifact-name$=".gif" i], [data-testid="artifact-tree-node"][artifact-name$=".mp4" i], [data-testid="artifact-tree-node"][artifact-name="frame_000000000064.json"]');
      assert.equal(state.tab, 'Artifacts');
      return { count: async () => state.forbidden };
    },
    getByText: (text, options) => {
      state.calls.push(['text', text]); assert.deepEqual(options, { exact: true });
      const result = locator(text);
      result.count = async () => state.counts[text] ?? Number(text === state.id || text === state.status || text === state.unrelatedText);
      return result;
    },
    getByLabel: (name, options) => {
      state.calls.push(['label', name]); assert.deepEqual(options, { exact: true });
      assert(['source_package.json', 'config.yaml', 'evaluation'].includes(name));
      assert.equal(state.tab, 'Artifacts');
      return locator(name, { 'data-testid': 'artifact-tree-node', 'artifact-name': name });
    },
    getByRole: (role, options) => {
      state.calls.push(['role', role, options]); assert.equal(role, 'tab');
      assert(['Overview', 'Artifacts'].includes(options.name)); assert.equal(options.exact, true);
      const name = options.name;
      const result = locator(name, { 'aria-selected': state.tab === name ? 'true' : 'false' });
      if (options.selected !== undefined) {
        assert.equal(options.selected, true);
        result.waitFor = async (opts) => { wait(name, opts); assert.equal(state.tab, name); state.actions.push(['selected', name]); };
      }
      result.click = async (opts) => {
        assert.equal(name, 'Artifacts', 'only native tab click permitted');
        assert.deepEqual(opts, { timeout: 30000 });
        state.actions.push(['click', name]);
        state.tab = state.afterClickTab ?? name;
        state.url = state.afterClickURL ?? `${root}/artifacts`;
      };
      return result;
    },
    screenshot: async (options) => {
      assert.deepEqual(options, { type: 'png', timeout: 30000 });
      state.actions.push(['png', state.tab]);
      const bytes = Buffer.from(`unchanged native ${state.tab} PNG`);
      state.pngs.push(bytes); state.onScreenshot?.(state.tab);
      return bytes;
    },
  };
  return { page, state };
}

test('native status then Artifacts; unchanged viewport bytes; summary last, no source reads or blocker click', async (t) => {
  const manifest = fixture(), { page, state } = nativePage(manifest), { saved, calls } = filesystem(t);
  const result = await runMLflowFailure(page, manifest, OUTPUT);
  assert.equal(result.status.actual, 'Failed'); assert.equal(result.status.expected, 'FAILED');
  assert.deepEqual(result.cli, manifest.cli); assert.deepEqual(result.tensorboard_gif, manifest.tensorboard_gif);
  assert.deepEqual(result.prior_artifacts, manifest.prior_artifacts);
  assert.equal(result.run_id, manifest.run_id); assert.equal(result.experiment_id, manifest.experiment_id);
  assert.equal(result.scope, 'native FAILED status and retained-file visibility only; CLI/media delivery proofs parent-owned; whole6.2/V remain UNVERIFIED');
  assert.deepEqual(result.native.filenames, ['source_package.json', 'config.yaml', 'evaluation']);
  assert.match(result.native.evaluation, /not opened, not rollout media/);
  assert.deepEqual(state.actions.filter(([action]) => ['png', 'click', 'selected'].includes(action)),
    [['png', 'Overview'], ['click', 'Artifacts'], ['selected', 'Artifacts'], ['png', 'Artifacts']]);
  const firstPNG = state.actions.findIndex(([action]) => action === 'png');
  for (const text of [manifest.run_id, 'Failed']) assert(state.actions.slice(0, firstPNG).some(([a, n]) => a === 'visible' && n === text));
  for (const [index, descriptor] of [result.screenshots.failed_run, result.screenshots.artifacts].entries()) {
    assert.strictEqual(saved.get(path.join(OUTPUT, descriptor.filename)), state.pngs[index]);
    assert.equal(descriptor.sha256, hash(state.pngs[index])); assert.equal(descriptor.bytes, state.pngs[index].length);
  }
  assert.deepEqual([...saved.keys()], ['failed-run.png', 'artifacts.png', 'mlflow-failure.json'].map((name) => path.join(OUTPUT, name)));
  assert.deepEqual(JSON.parse(saved.get(SUMMARY)), result);
  assert.equal(calls.filter(([action]) => action === 'close').length, 3);
  assert.equal(calls.filter(([action]) => action === 'directory-close').length, 1);
  assert(!state.calls.some(([action]) => action === 'text'), 'whole-page text must not establish status or UUID');
  for (const ancestor of [OUTPUT, '/tmp/opencode', '/tmp', '/']) assert(calls.some(([a, p]) => a === 'lstat' && p === ancestor));
});

test('native artifact loading waits before strict unique tree admission', async (t) => {
  const manifest = fixture(), { page, state } = nativePage(manifest), { saved } = filesystem(t);
  state.delayed = new Set(['source_package.json', 'config.yaml', 'evaluation']);
  for (const name of state.delayed) state.counts[name] = 0;
  const result = await runMLflowFailure(page, manifest, OUTPUT);
  assert.equal(state.delayed.size, 0);
  assert.equal(result.native.filenames.length, 3);
  assert(saved.has(SUMMARY));
});

for (const [name, mutate] of [
  ['missing manifest', () => null], ['extra field', (m) => { m.mode = 'both'; }],
  ['invalid run', (m) => { m.run_id = 'bad'; }], ['numeric run', (m) => { m.run_id = 1; }],
  ['numeric experiment', (m) => { m.experiment_id = 1; }], ['invalid experiment', (m) => { m.experiment_id = '../1'; }],
  ['wrong status', (m) => { m.status = 'FINISHED'; }], ['wrong experiment name', (m) => { m.experiment_name = 'other'; }],
  ...[NaN, Infinity, -Infinity, 0, 2, '1', null].map((value) => [`exit ${value}`, (m) => { m.cli.exit_code = value; }]),
  ['wrong CLI status', (m) => { m.cli.status = 'PASS'; }], ['missing CLI', (m) => { delete m.cli; }],
  ['relative log', (m) => { m.cli.log_path = 'cli.log'; }], ['noncanonical log', (m) => { m.cli.log_path = '/producer/../cli.log'; }],
  ['NUL log', (m) => { m.cli.log_path = '/producer/\0'; }], ['bad log hash', (m) => { m.cli.log_sha256 = 'bad'; }],
  ['missing GIF', (m) => { delete m.tensorboard_gif; }], ['wrong GIF basename', (m) => { m.tensorboard_gif.path = '/artifacts/evaluation/frame_000000000065.gif'; }],
  ['wrong GIF directory', (m) => { m.tensorboard_gif.path = '/producer/frame_000000000064.gif'; }],
  ['relative GIF', (m) => { m.tensorboard_gif.path = 'artifacts/evaluation/frame_000000000064.gif'; }],
  ['bad GIF hash', (m) => { m.tensorboard_gif.sha256 = 'bad'; }], ['missing prior files', (m) => { m.prior_artifacts = []; }],
  ['wrong prior basename', (m) => { m.prior_artifacts[0].filename = '../source_package.json'; }],
  ['duplicate prior file', (m) => { m.prior_artifacts[1] = m.prior_artifacts[0]; }],
  ['bad prior hash', (m) => { m.prior_artifacts[1].sha256 = 'bad'; }],
]) test(`manifest admission: ${name} before any FS/UI`, async (t) => {
  let manifest = fixture(); const { page, state } = nativePage(manifest), { calls } = filesystem(t);
  manifest = mutate(manifest) === null ? null : manifest;
  await assert.rejects(runMLflowFailure(page, manifest, OUTPUT));
  assert.equal(calls.length, 0); assert.equal(state.calls.length, 0);
});

for (const output of ['relative', '/tmp/opencode/../bad', `${OUTPUT}/`, '/tmp/opencode/\0', null]) {
  test(`invalid output ${JSON.stringify(output)} before FS/UI`, async (t) => {
    const manifest = fixture(), { page, state } = nativePage(manifest), { calls } = filesystem(t);
    await assert.rejects(runMLflowFailure(page, manifest, output), /canonical absolute output/);
    assert.equal(calls.length, 0); assert.equal(state.calls.length, 0);
  });
}
test('invalid public page rejected before FS', async (t) => {
  const { calls } = filesystem(t);
  await assert.rejects(runMLflowFailure({}, fixture(), OUTPUT), /public native page/);
  assert.equal(calls.length, 0);
});

for (const [name, options] of [
  ['nonempty', { entries: ['old.png'] }], ['missing', { statError: new Error('ENOENT') }],
  ['permissions', { stat: (p) => p === OUTPUT ? { mode: 0o40755 } : {} }],
  ['special permissions', { stat: (p) => p === OUTPUT ? { mode: 0o41700 } : {} }],
  ['owner', { stat: (p) => p === OUTPUT ? { uid: process.getuid() + 1 } : {} }],
  ['file', { stat: (p) => p === OUTPUT ? { isDirectory: () => false } : {} }],
  ['symlink', { stat: (p) => p === OUTPUT ? { isSymbolicLink: () => true } : {} }],
  ['symlink ancestor', { stat: (p) => p === '/tmp' ? { isSymbolicLink: () => true } : {} }],
  ['realpath', { realpath: '/elsewhere' }],
]) test(`output ${name} rejected before UI`, async (t) => {
  const manifest = fixture(), { page, state } = nativePage(manifest), { saved } = filesystem(t, options);
  await assert.rejects(runMLflowFailure(page, manifest, OUTPUT));
  assert.equal(state.calls.length, 0); assert.equal(saved.size, 0);
});

for (const [name, mutate] of [
  ['wrong run', (s) => { s.url = s.url.replace('a'.repeat(32), 'e'.repeat(32)); }],
  ['wrong initial tab URL', (s) => { s.url += '/artifacts'; }],
  ['wrong origin', (s) => { s.url = s.url.replace('127.0.0.1', 'localhost'); }],
  ['wrong title', (s) => { s.title = 'Run other - MLflow'; }], ['wrong Overview tab', (s) => { s.tab = 'Artifacts'; }],
  ['duplicate Overview', (s) => { s.counts.Overview = 2; }], ['duplicate breadcrumb', (s) => { s.counts.breadcrumb = 2; }],
  ['wrong breadcrumb name', (s) => { s.breadcrumbText = 'other'; }],
  ['wrong breadcrumb href', (s) => { s.attrs.breadcrumb = { href: '#/experiments/2/evaluation-runs' }; }],
  ['wrong visible UUID', (s) => { s.id = 'e'.repeat(32); }], ['wrong visible status', (s) => { s.status = 'Running'; }],
  ['duplicate UUID', (s) => { s.counts['a'.repeat(32)] = 2; }], ['duplicate Failed', (s) => { s.counts.Failed = 2; }],
  ['hidden status', (s) => { s.hidden.add('Failed'); }], ['hidden UUID', (s) => { s.hidden.add('a'.repeat(32)); }],
  ['wrong Artifacts selected', (s) => { s.afterClickTab = 'Overview'; }],
  ['wrong Artifacts URL', (s) => { s.afterClickURL = s.url; }], ['duplicate Artifacts', (s) => { s.counts.Artifacts = 2; }],
  ...['source_package.json', 'config.yaml', 'evaluation'].flatMap((filename) => [
    [`missing ${filename}`, (s) => { s.counts[filename] = 0; }], [`duplicate ${filename}`, (s) => { s.counts[filename] = 2; }],
    [`hidden ${filename}`, (s) => { s.hidden.add(filename); }],
  ]),
  ['wrong tree testid', (s) => { s.attrs.evaluation = { 'data-testid': 'directory' }; }],
  ['wrong tree basename', (s) => { s.attrs.evaluation = { 'artifact-name': 'evaluation/' }; }],
  ['forbidden GIF/MP4/metadata', (s) => { s.forbidden = 1; }],
  ...['Overview', 'Artifacts'].flatMap((tab) => ['url', 'title', 'breadcrumb', 'tab'].map((what) => [
    `${tab} capture changes ${what}`, (s) => { s.onScreenshot = (current) => {
      if (current !== tab) return;
      if (what === 'url') s.url += '/other';
      if (what === 'title') s.title = 'other';
      if (what === 'breadcrumb') s.breadcrumbText = 'other';
      if (what === 'tab') s.tab = 'other';
    }; },
  ])),
]) test(`native rejection: ${name}, no summary`, async (t) => {
  const manifest = fixture(), { page, state } = nativePage(manifest), { saved } = filesystem(t);
  mutate(state);
  await assert.rejects(runMLflowFailure(page, manifest, OUTPUT));
  assert(!saved.has(SUMMARY));
  if (name.startsWith('Overview capture')) assert.equal(saved.size, 1);
  if (name.startsWith('Artifacts capture')) assert.equal(saved.size, 2);
});

for (const changedAt of [2, 3, 4]) test(`directory replacement before write ${changedAt - 1}`, async (t) => {
  const manifest = fixture(), { page } = nativePage(manifest);
  const { saved, directoryState } = filesystem(t, { stat: (p, count) => p === OUTPUT && count >= changedAt + 1 ? { ino: 3 } : {} });
  await assert.rejects(runMLflowFailure(page, manifest, OUTPUT), /output directory changed/);
  assert.equal(saved.size, changedAt - 2); assert(!saved.has(SUMMARY));
  assert(directoryState.closed);
});
test('original native screenshot failure preserved with earlier PNG retained', async (t) => {
  const manifest = fixture(), { page } = nativePage(manifest), { saved } = filesystem(t);
  const error = new Error('native screenshot failed'), original = page.screenshot;
  let count = 0;
  page.screenshot = async (options) => { if (++count === 2) throw error; return original(options); };
  await assert.rejects(runMLflowFailure(page, manifest, OUTPUT), (actual) => actual === error);
  assert.equal(saved.size, 1); assert(!saved.has(SUMMARY));
});
for (const operation of ['open', 'write', 'close']) test(`original ${operation} error preserved`, async (t) => {
  const manifest = fixture(), { page } = nativePage(manifest), error = new Error(operation);
  const options = { [`${operation}Error`]: error };
  if (operation === 'write') options.closeError = new Error('secondary close error');
  const { saved } = filesystem(t, options);
  await assert.rejects(runMLflowFailure(page, manifest, OUTPUT), (actual) => actual === error);
  assert(!saved.has(SUMMARY));
});

test('Running status with unrelated visible Failed text cannot pass the native Status field check', async (t) => {
  const manifest = fixture(), { page, state } = nativePage(manifest), { saved } = filesystem(t);
  state.status = 'Running'; state.unrelatedText = 'Failed';
  assert.equal(await page.getByText('Failed', { exact: true }).count(), 1);
  await assert.rejects(runMLflowFailure(page, manifest, OUTPUT), /native Status field must be Failed/);
  assert.equal(saved.size, 0);
});
for (const field of ['Status field', 'Run ID field', 'Failed']) {
  test(`unknown/missing native selector fails closed: ${field}`, async (t) => {
    const manifest = fixture(), { page, state } = nativePage(manifest), { saved } = filesystem(t);
    state.counts[field] = 0;
    await assert.rejects(runMLflowFailure(page, manifest, OUTPUT), /one exact native element/);
    assert.equal(saved.size, 0);
  });
}
test('native Status permits outer whitespace only', async (t) => {
  const manifest = fixture(), { page, state } = nativePage(manifest);
  filesystem(t); state.status = ' \nFailed\t';
  assert.equal((await runMLflowFailure(page, manifest, OUTPUT)).status.actual, 'Failed');
});
test('replacement after pathname check cannot redirect the anchored write', async (t) => {
  const manifest = fixture(), { page } = nativePage(manifest);
  const { saved, calls, directoryState } = filesystem(t, { replaceAtFileOpen: true });
  await assert.rejects(runMLflowFailure(page, manifest, OUTPUT), /output directory changed/);
  assert(directoryState.replaced && directoryState.closed);
  assert.deepEqual([...saved.keys()], [path.join(OUTPUT, 'failed-run.png')]);
  assert.deepEqual(directoryState.anchoredWrites, [{ ino: 2, pathnameReplaced: true }]);
  assert.deepEqual(calls.filter(([action, , flags]) => action === 'open' && flags === 'wx'),
    [['open', `${ANCHOR}/failed-run.png`, 'wx', 0o600]]);
});
for (const [name, options, pattern] of [
  ['wrong FD inode', { directoryStat: { ino: 9 } }, /anchor changed/],
  ['wrong FD device', { directoryStat: { dev: 9 } }, /anchor changed/],
  ['FD not directory', { directoryStat: { isDirectory: () => false } }, /anchor changed/],
  ['FD wrong owner', { directoryStat: { uid: process.getuid() + 1 } }, /anchor changed/],
  ['FD wrong permissions', { directoryStat: { mode: 0o40755 } }, /anchor changed/],
  ['pathname replaced before FD use', { stat: (p, count) => p === OUTPUT && count > 1 ? { ino: 9 } : {} }, /anchor changed/],
]) test(`directory anchor admission: ${name}`, async (t) => {
  const manifest = fixture(), { page, state } = nativePage(manifest);
  const { saved, directoryState } = filesystem(t, options);
  await assert.rejects(runMLflowFailure(page, manifest, OUTPUT), pattern);
  assert(directoryState.closed); assert.equal(state.calls.length, 0); assert.equal(saved.size, 0);
});
for (const operation of ['directoryOpen', 'directoryStat', 'directoryClose']) {
  test(`original ${operation} error preserved with correct FD ownership cleanup`, async (t) => {
    const manifest = fixture(), { page } = nativePage(manifest), error = new Error(operation);
    const options = { [`${operation}Error`]: error };
    if (operation === 'directoryStat') options.directoryCloseError = new Error('secondary directory close');
    const { directoryState } = filesystem(t, options);
    await assert.rejects(runMLflowFailure(page, manifest, OUTPUT), (actual) => actual === error);
    assert.equal(directoryState.opened, operation !== 'directoryOpen');
    assert.equal(directoryState.closed, operation !== 'directoryOpen');
  });
}
test('UI failure remains original when directory cleanup also fails', async (t) => {
  const manifest = fixture(), { page } = nativePage(manifest), error = new Error('native capture');
  const { saved, directoryState } = filesystem(t, { directoryCloseError: new Error('secondary close') });
  page.screenshot = async () => { throw error; };
  await assert.rejects(runMLflowFailure(page, manifest, OUTPUT), (actual) => actual === error);
  assert(directoryState.closed); assert.equal(saved.size, 0);
});
