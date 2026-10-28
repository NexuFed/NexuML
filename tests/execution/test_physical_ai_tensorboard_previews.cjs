'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const crypto = require('node:crypto');
const SCRIPT = '../fixtures/physical_ai_tensorboard_previews.cjs';
const { runTensorBoardPreviews, decodeTensorBoardImageSource, validateTensorBoardManifest } = require(SCRIPT);

const RUN = 'physical-ai-logging-tb-only/cartpole-ppo_v1';
const TAG = 'eval/rl/episode';
const OUTPUT = '/tmp/opencode/tb-preview-mocked-output';
const BASE = 'http://127.0.0.1:6007/';
const sha256 = (bytes) => crypto.createHash('sha256').update(bytes).digest('hex');
// These are synthetic byte fixtures, never native browser/codec acceptance evidence.
const GIF = Buffer.from('47494638396180005500000000', 'hex');
function png(frame) {
  const bytes = Buffer.alloc(25);
  Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]).copy(bytes);
  bytes.write('IHDR', 12);
  bytes.writeUInt32BE(128, 16);
  bytes.writeUInt32BE(85, 20);
  bytes[24] = frame;
  return bytes;
}
function source(step, run = RUN) {
  return `data/plugin/images/individualImage?blob_key=${Buffer.from(JSON.stringify(['', run, TAG, step, 2])).toString('base64url')}&ts=1792937018.8743327`;
}
function manifest(mode = 'tb-only') {
  const run = `physical-ai-logging-${mode}/cartpole-ppo_v1`;
  return { modes: { [mode]: { logging: { tensorboard: { log_dir: '/owned/events' } }, native: { tensorboard: { main: {
    log_dir: `/owned/events/${run}`, image_tag: TAG, scalar_history: {},
    previews: [0, 32, 64].map((step) => ({ step, sha256: sha256(GIF), decoded: { frames: 9, size: [128, 85], changing_nonuniform_frames: true } })),
    associated_policy_exports: Object.fromEntries(['untrained', 'intermediate', 'final'].map((key, index) => [key, {
      source_frames: [0, 32, 64][index], policy_files_sha256: { 'state_dict.pt': String(index + 1).repeat(64) },
    }])),
  } } } } } };
}
function mockOutput(t, overrides = {}) {
  const files = new Map();
  const stat = { isDirectory: () => true, isSymbolicLink: () => false, uid: process.getuid(), mode: 0o40700, dev: 1, ino: 7, ...overrides };
  t.mock.method(fs, 'lstat', async (name) => { assert.equal(name, OUTPUT); return stat; });
  t.mock.method(fs, 'realpath', async (name) => name);
  t.mock.method(fs, 'readdir', async () => []);
  t.mock.method(fs, 'open', async (name, flags, mode) => {
    assert(name.startsWith(`${OUTPUT}/`));
    assert(!name.slice(OUTPUT.length + 1).includes('/'));
    assert.equal(flags, 'wx');
    assert.equal(mode, 0o600);
    assert(!files.has(name), 'exclusive evidence names');
    files.set(name, null);
    return { writeFile: async (bytes) => { files.set(name, bytes); }, close: async () => {} };
  });
  return files;
}
function fakePage({ staticImage = false, body = GIF, run = RUN, imageRun = run, responseStatus = 200, contentType = 'image/gif' } = {}) {
  const actions = [];
  let ordinal = 2;
  let sample = 0;
  const src = () => source([0, 32, 64][ordinal], imageRun);
  const handle = {
    evaluate: async (fn) => fn({ isConnected: true, complete: true, currentSrc: new URL(src(), BASE).href, naturalWidth: 128, naturalHeight: 85 }),
    screenshot: async (options) => {
      assert.deepEqual(options, { type: 'png', animations: 'allow', scale: 'css', timeout: 30000 });
      actions.push(`sample:${[0, 32, 64][ordinal]}`);
      return png(staticImage ? 0 : sample++ % 2);
    },
    dispose: async () => {},
  };
  const image = {
    count: async () => 1, waitFor: async () => {}, getAttribute: async (name) => { assert.equal(name, 'src'); return src(); },
    elementHandle: async () => handle,
  };
  const slider = {
    count: async () => 1, waitFor: async () => {},
    getAttribute: async (name) => ({ role: 'slider', tabindex: '0', 'aria-valuemin': '0', 'aria-valuemax': '2', 'aria-valuenow': String(ordinal) })[name],
    press: async (key, options) => {
      assert.deepEqual(options, { timeout: 30000 });
      actions.push(key);
      if (key === 'Home') ordinal = 0;
      else { assert.equal(key, 'ArrowRight'); ordinal++; }
    },
  };
  const card = {
    waitFor: async () => {}, count: async () => 1, elementHandle: async () => ({ dispose: async () => {} }),
    locator: (selector) => { assert(['paper-slider#steps', 'img'].includes(selector)); return selector === 'img' ? image : slider; },
  };
  const page = {
    url: () => `${BASE}#images`,
    getByText: (text, options) => { assert.equal(text, run); assert.deepEqual(options, { exact: true }); return 'named-run'; },
    locator: (selector) => {
      assert.equal(selector, 'tf-image-loader');
      return { filter: (options) => { assert.deepEqual(options, { has: 'named-run' }); return card; } };
    },
    waitForFunction: async (fn, args, options) => {
      assert.equal(args.relativeRun, run);
      assert.equal(args.ordinal, ordinal);
      assert.equal(args.step, [0, 32, 64][ordinal]);
      assert.deepEqual(options, { timeout: 30000, polling: 35 });
      const nativeImage = { matches: (selector) => selector === 'img', complete: true,
        naturalWidth: 128, naturalHeight: 85, currentSrc: new URL(src(), BASE).href,
        getAttribute: () => src() };
      const nativeSlider = { matches: (selector) => selector === 'paper-slider#steps',
        getAttribute: () => String(ordinal) };
      const shadowHost = { matches: () => false, shadowRoot: { querySelectorAll: () => [nativeSlider, nativeImage] } };
      const nativeArgs = { ...args, card: {
        querySelectorAll: () => [], shadowRoot: { querySelectorAll: () => [shadowHost] },
      } };
      const previousLocation = global.location;
      let loaded;
      try {
        global.location = { href: `${BASE}#images` };
        assert.equal(fn({ ...nativeArgs, step: args.step + 1 }), false, 'stale native step must wait');
        assert.equal(fn({ ...nativeArgs, ordinal: args.ordinal + 1 }), false, 'stale aria value must wait');
        nativeImage.complete = false;
        assert.equal(fn(nativeArgs), false, 'unloaded native image must wait');
        nativeImage.complete = true;
        nativeImage.naturalWidth = 0;
        assert.equal(fn(nativeArgs), false, 'zero native dimensions must wait');
        nativeImage.naturalWidth = 128;
        const originalSource = nativeImage.currentSrc;
        for (const tuple of [['', run, TAG, args.step], ['', run, TAG, args.step, 1], ['', run, TAG, args.step, 2, 'extra']]) {
          nativeImage.currentSrc = `${BASE}data/plugin/images/individualImage?blob_key=${Buffer.from(JSON.stringify(tuple)).toString('base64url')}`;
          assert.equal(fn(nativeArgs), false, 'missing/wrong/extra native sample must wait');
        }
        nativeImage.currentSrc = originalSource;
        loaded = fn(nativeArgs);
        assert(loaded, 'recursive open-shadow selection must find native image');
      } finally {
        if (previousLocation === undefined) delete global.location;
        else global.location = previousLocation;
      }
      actions.push(`loaded:${args.step}`);
      return { jsonValue: async () => loaded, dispose: async () => {} };
    },
    waitForTimeout: async (ms) => { assert.equal(ms, 35); actions.push('wait:35'); },
    request: { get: async (url, options) => {
      assert.equal(url, new URL(src(), BASE).href);
      assert.deepEqual(options, { timeout: 30000, maxRedirects: 0 });
      actions.push(`fetch:${[0, 32, 64][ordinal]}`);
      return { status: () => responseStatus, headers: () => ({ 'content-type': contentType }), body: async () => body };
    } },
    screenshot: async (options) => {
      assert.deepEqual(options, { type: 'png', fullPage: true, animations: 'allow', timeout: 30000 });
      actions.push(`page:${[0, 32, 64][ordinal]}`);
      return png(0);
    },
    reload: async (options) => {
      assert.deepEqual(options, { waitUntil: 'domcontentloaded', timeout: 30000 });
      actions.push('reload');
      ordinal = 2;
    },
  };
  return { page, actions };
}

test('import has no filesystem/browser execution', (t) => {
  t.mock.method(fs, 'lstat', () => assert.fail('import must be inert'));
  t.mock.method(fs, 'open', () => assert.fail('import must be inert'));
  delete require.cache[require.resolve(SCRIPT)];
  assert.equal(typeof require(SCRIPT).runTensorBoardPreviews, 'function');
});

test('decodes literal actual DOM blob and exact identity', () => {
  const literal = 'data/plugin/images/individualImage?blob_key=WyIiLCJwaHlzaWNhbC1haS1sb2dnaW5nLXRiLW9ubHkvY2FydHBvbGUtcHBvX3YxIiwiZXZhbC9ybC9lcGlzb2RlIiw2NCwyXQ&ts=1792937018.8743327';
  const decoded = decodeTensorBoardImageSource(literal, RUN, TAG, 64);
  assert.deepEqual(decoded.tuple, ['', RUN, TAG, 64, 2]);
  assert.equal(decoded.url, `${BASE}${literal}`);
});

test('rejects wrong run/step/tag, credentials, foreign origins, routes and malformed blobs', () => {
  for (const src of [source(0, 'wrong-run'), source(32), source(0).replace('data/plugin', 'other/plugin'),
    `http://user:pass@127.0.0.1:6007/${source(0)}`, `http://foreign.invalid:6007/${source(0)}`,
    `https://127.0.0.1:6007/${source(0)}`, `http://127.0.0.1:6008/${source(0)}`,
    `${source(0)}&blob_key=abc`, 'data/plugin/images/individualImage?blob_key=!!!!',
    'data/plugin/images/individualImage?blob_key=e30']) {
    assert.throws(() => decodeTensorBoardImageSource(src, RUN, TAG, 0));
  }
  assert.throws(() => decodeTensorBoardImageSource(source(0), RUN, 'wrong-tag', 0));
  for (const tuple of [['', RUN, TAG, 0], ['', RUN, TAG, 0, 1], ['', RUN, TAG, 0, 2, 'extra']]) {
    const src = `data/plugin/images/individualImage?blob_key=${Buffer.from(JSON.stringify(tuple)).toString('base64url')}`;
    assert.throws(() => decodeTensorBoardImageSource(src, RUN, TAG, 0), /native image identity mismatch/);
  }
});

test('fixed manifest admission rejects steps/hash/dimensions/policy drift', () => {
  assert.equal(validateTensorBoardManifest(manifest()).relativeRun, RUN);
  const mutate = [
    (m) => { m.previews[1].step = 31; }, (m) => { m.previews.pop(); },
    (m) => { m.previews[0].sha256 = '../not-a-hash'; },
    (m) => { m.previews[0].decoded.size = [0, 85]; },
    (m) => { m.previews[0].decoded.size = [128, 4097]; },
    (m) => { m.previews[0].decoded.frames = 1; },
    (m) => { m.previews[0].decoded.changing_nonuniform_frames = false; },
    (m) => { m.associated_policy_exports.final.source_frames = 63; },
    (m) => { m.associated_policy_exports.final.policy_files_sha256['state_dict.pt'] = 'bad'; },
    (m) => { m.associated_policy_exports.checkpoint_final = m.associated_policy_exports.final; delete m.associated_policy_exports.final; },
  ];
  for (const change of mutate) {
    const m = manifest();
    change(m.modes['tb-only'].native.tensorboard.main);
    assert.throws(() => validateTensorBoardManifest(m));
  }
});

test('native keyboard selection and animation precede reopening; all files exclusive', async (t) => {
  const files = mockOutput(t);
  const { page, actions } = fakePage();
  const result = await runTensorBoardPreviews(page, manifest(), OUTPUT);
  assert.equal(result.mode, 'tb-only');
  assert.deepEqual(actions.filter((action) => !['sample:0', 'sample:32', 'sample:64', 'wait:35'].includes(action)), [
    'Home', 'loaded:0', 'fetch:0', 'page:0',
    'Home', 'ArrowRight', 'loaded:32', 'fetch:32', 'page:32',
    'Home', 'ArrowRight', 'ArrowRight', 'loaded:64', 'fetch:64', 'page:64',
    'reload', 'Home', 'loaded:0', 'fetch:0', 'page:0',
  ]);
  assert.deepEqual(result.stages.map((stage) => stage.step), [0, 32, 64]);
  assert.equal(result.stages[2].stage, 'checkpoint_final');
  assert.equal(result.stages[2].associated_policy_export.key, 'final');
  assert.equal(result.reopened.reopened, true);
  assert.equal(result.reopened.step, 0);
  for (const stage of [...result.stages, result.reopened]) {
    assert.equal(stage.animation.samples.length, 2);
    assert.equal(new Set(stage.animation.samples.map((sample) => sample.sha256)).size, 2);
    assert.equal(stage.native_gif.sha256, sha256(GIF));
  }
  assert.equal(files.size, 17);
  assert.deepEqual(JSON.parse(files.get(`${OUTPUT}/tensorboard-previews.json`)), result);
  assert.match(result.scope, /scalar UI, MP4 retrieval and V\/task6.1 remain UNVERIFIED/);
});

test('static screenshots fail after 20 samples, retain partial files, never reopen/promote', async (t) => {
  const files = mockOutput(t);
  const { page, actions } = fakePage({ staticImage: true });
  await assert.rejects(runTensorBoardPreviews(page, manifest(), OUTPUT), /stayed static/);
  assert.equal(actions.filter((action) => action === 'sample:0').length, 20);
  assert.equal(actions.filter((action) => action === 'wait:35').length, 19);
  assert.equal(files.size, 21);
  assert(!actions.includes('reload'));
  assert(!files.has(`${OUTPUT}/tensorboard-previews.json`));
});

test('native payload hash mismatch fails without fallback or screenshots', async (t) => {
  const files = mockOutput(t);
  const altered = Buffer.concat([GIF, Buffer.from('changed')]);
  const { page, actions } = fakePage({ body: altered });
  await assert.rejects(runTensorBoardPreviews(page, manifest(), OUTPUT), /hash mismatch/);
  assert.equal(files.size, 0);
  assert(!actions.some((action) => action.startsWith('sample:')));
});

test('output admission fails before browser interaction for mode, symlink, nonempty paths', async (t) => {
  mockOutput(t, { mode: 0o40755 });
  await assert.rejects(runTensorBoardPreviews({}, manifest(), OUTPUT), /owned 0700/);
  t.mock.restoreAll();
  mockOutput(t);
  t.mock.method(fs, 'realpath', async () => '/somewhere/else');
  await assert.rejects(runTensorBoardPreviews({}, manifest(), OUTPUT), /symlinks/);
  t.mock.restoreAll();
  mockOutput(t);
  t.mock.method(fs, 'readdir', async () => ['existing-file']);
  await assert.rejects(runTensorBoardPreviews({}, manifest(), OUTPUT), /must be caller-created and empty/);
});

test('both and default preview admission retain their exact mode-scoped identities', () => {
  const m = { modes: { ...manifest().modes, ...manifest('both').modes } };
  for (const mode of ['tb-only', 'both']) {
    const { main, relativeRun } = validateTensorBoardManifest(m, mode);
    assert.equal(main, m.modes[mode].native.tensorboard.main);
    assert.equal(relativeRun, `physical-ai-logging-${mode}/cartpole-ppo_v1`);
    assert.equal(main.log_dir, `${m.modes[mode].logging.tensorboard.log_dir}/${relativeRun}`);
    assert.deepEqual(main.previews.map(({ decoded }) => decoded), Array(3).fill({ frames: 9, size: [128, 85], changing_nonuniform_frames: true }));
  }
  assert.equal(validateTensorBoardManifest(m).main, m.modes['tb-only'].native.tensorboard.main);
});

test('both fake UI selects only both run/blob/policies through all stages and reopen', async (t) => {
  const m = manifest('both');
  m.modes['tb-only'] = manifest().modes['tb-only'];
  m.modes['tb-only'].native.tensorboard.main.previews[0].sha256 = '0'.repeat(64);
  m.modes.both.native.tensorboard.main.associated_policy_exports.final.policy_files_sha256['state_dict.pt'] = '8'.repeat(64);
  const run = 'physical-ai-logging-both/cartpole-ppo_v1';
  const files = mockOutput(t);
  const { page, actions } = fakePage({ run });
  const result = await runTensorBoardPreviews(page, m, OUTPUT, 'both');
  assert.equal(result.mode, 'both');
  assert.equal(result.relative_run, run);
  assert.deepEqual(result.stages.map(stage => stage.step), [0, 32, 64]);
  assert.deepEqual(result.stages.map(stage => stage.associated_policy_export.key), ['untrained', 'intermediate', 'final']);
  assert.equal(result.stages[2].associated_policy_export.actor_sha256, '8'.repeat(64));
  assert.equal(result.reopened.reopened, true);
  assert.equal(result.reopened.step, 0);
  for (const stage of [...result.stages, result.reopened]) {
    assert.deepEqual(stage.selected_blob_tuple, ['', run, TAG, stage.step, 2]);
    assert.equal(stage.native_gif.sha256, sha256(GIF));
    assert.equal(stage.animation.samples.length, 2);
    assert.equal(new Set(stage.animation.samples.map(sample => sample.sha256)).size, 2);
  }
  assert.equal(actions.filter(action => action === 'reload').length, 1);
  assert.equal(files.size, 17);
  assert.deepEqual(JSON.parse(files.get(`${OUTPUT}/tensorboard-previews.json`)), result);
  assert.match(result.scope, /V\/task6.1 remain UNVERIFIED/);
});

test('disabled/unknown preview modes reject before manifest/output/browser access', async (t) => {
  const untouched = new Proxy({}, { get: () => assert.fail('invalid mode must be rejected first') });
  t.mock.method(fs, 'lstat', () => assert.fail('no output access'));
  t.mock.method(fs, 'open', () => assert.fail('no output writes'));
  for (const mode of ['disabled', 'mlflow-only', 'unknown', '', '__proto__', null, false, 0, {}, ['both'], new String('both')]) {
    assert.throws(() => validateTensorBoardManifest(untouched, mode), /mode must be tb-only or both/);
    await assert.rejects(runTensorBoardPreviews(untouched, untouched, OUTPUT, mode), /mode must be tb-only or both/);
  }
});

test('preview run/base/blob identity cannot cross modes or fall back', () => {
  for (const [mode, other] of [['both', 'tb-only'], ['tb-only', 'both']]) {
    const m = manifest(mode);
    const run = `physical-ai-logging-${mode}/cartpole-ppo_v1`;
    const wrongRun = `physical-ai-logging-${other}/cartpole-ppo_v1`;
    m.modes[mode].native.tensorboard.main.log_dir = `/owned/events/${wrongRun}`;
    assert.throws(() => validateTensorBoardManifest(m, mode));
    assert.throws(() => decodeTensorBoardImageSource(source(0, wrongRun), run, TAG, 0), /native image identity mismatch/);
    const wrongBase = manifest(mode);
    wrongBase.modes[mode].logging.tensorboard.log_dir = '/other/events';
    assert.throws(() => validateTensorBoardManifest(wrongBase, mode));
    assert.throws(() => validateTensorBoardManifest(manifest(other), mode), /native main log_dir required/);
  }
});
