'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const http = require('node:http');
const fs = require('node:fs/promises');
const crypto = require('node:crypto');
const { runMLflowArtifacts, validateManifest, artifactURI, retrieveArtifact,
  playNativeVideo, sampleNativeGIF } = require('../fixtures/physical_ai_mlflow_artifacts.cjs');

const ORIGIN = 'http://127.0.0.1:5001';
const RUN = '05c57215623c49e0bab276bb1bf430db';
const STEM = 'frame_000000000064';
const sha = (b) => crypto.createHash('sha256').update(b).digest('hex');
const uri = (ext, run = RUN) => `${ORIGIN}/get-artifact?path=evaluation%2F${STEM}.${ext}&run_uuid=${run}`;
const expected = { origin: ORIGIN, size: [128, 86], duration: 0.2 };

function fixture() {
  const metrics = { 'eval/rl/episodes': 2, 'eval/rl/return_mean': 9.5 };
  const bytes = { mp4: Buffer.from('mock mp4'), gif: Buffer.from('mock gif'),
    json: Buffer.from(JSON.stringify({ frame_step: 64, metrics, provenance: {} })) };
  const media = { frame_step: 64, metrics, provenance: {}, captured_frames: 10, fps: 50,
    gif: { frames: 9, size: [128, 85], changing_nonuniform_frames: true },
    mp4: { frames: 10, codec: 'h264', size: [128, 86] }, files: {} };
  for (const ext of ['mp4', 'gif', 'json']) media.files[ext] = { path: `/source/${STEM}.${ext}`, sha256: sha(bytes[ext]) };
  const main = { run_id: RUN, experiment_id: '1', status: 'FINISHED', media: { 64: media } };
  return { bytes, main, media, manifest: { modes: { 'mlflow-only': { native: { mlflow: { main } } } } } };
}

// Small in-memory request/response doubles: never open a socket.
function emitter(properties = {}) {
  const listeners = {};
  return Object.assign(properties, {
    on(name, callback) { (listeners[name] ||= []).push(callback); return this; },
    emit(name, value) { for (const callback of listeners[name] || []) callback(value); },
    destroy() { this.destroyed = true; },
  });
}
function mockHTTP(t, script) {
  const calls = [];
  t.mock.method(http, 'get', (url, options, callback) => {
    const request = emitter(), response = emitter({ statusCode: 200, headers: {}, rawHeaders: [] });
    calls.push({ url, options, request, response });
    queueMicrotask(() => script({ request, response, callback, url }));
    return request;
  });
  return calls;
}

test('manifest uses fixed mlflow-only main, bounded decoded data, and source hashes', () => {
  const { manifest, main, media } = fixture();
  assert.equal(validateManifest(manifest).duration, 0.2);
  for (const mutate of [() => { main.run_id = 'wrong'; }, () => { main.experiment_id = '../1'; },
    () => { main.status = 'RUNNING'; }, () => { media.files.gif.sha256 = 'no'; },
    () => { media.files.json.path = '/source/other.json'; }, () => { media.mp4.frames = 100000; },
    () => { media.provenance = { policy_hash: 'invented' }; }]) {
    const copy = structuredClone(manifest);
    mutate();
    assert.throws(() => validateManifest(manifest));
    Object.assign(main, copy.modes['mlflow-only'].native.mlflow.main);
    Object.assign(media, main.media[64]);
    main.media[64] = media;
  }
});

test('artifact route rejects wrong origin/run/path, credentials, fragment, extras and duplicates', () => {
  assert.equal(artifactURI(uri('mp4'), RUN, 'mp4'), uri('mp4'));
  for (const bad of [uri('gif'), uri('mp4').replace(RUN, 'a'.repeat(32)),
    uri('mp4').replace('127.0.0.1', 'localhost'), uri('mp4').replace('http:', 'https:'),
    uri('mp4').replace(':5001', ':5002'), uri('mp4').replace('127.0.0.1', 'user@127.0.0.1'),
    `${uri('mp4')}#x`, `${uri('mp4')}#`, `${uri('mp4')}&extra=1`, `${uri('mp4')}&run_uuid=${RUN}`,
    uri('mp4').replace('/get-artifact', '/other'), uri('mp4').replace('127.0.0.1', '2130706433')]) {
    assert.throws(() => artifactURI(bad, RUN, 'mp4'));
  }
});

test('direct retrieval preserves exact bytes/headers and disables shared agent', async (t) => {
  const bytes = Buffer.from([0, 255, 1, 2]);
  const calls = mockHTTP(t, ({ response, callback }) => {
    response.headers = { 'content-length': '4', 'content-encoding': 'identity' };
    response.rawHeaders = ['Content-Length', '4'];
    callback(response);
    response.emit('data', bytes.subarray(0, 2)); response.emit('data', bytes.subarray(2)); response.emit('end');
  });
  const result = await retrieveArtifact(uri('mp4'), RUN, 'mp4');
  assert.deepEqual(result.bytes, bytes);
  assert.equal(result.receipt.sha256, sha(bytes));
  assert.deepEqual(result.receipt.raw_headers, ['Content-Length', '4']);
  assert.deepEqual(calls[0].options, { agent: false, headers: { 'Accept-Encoding': 'identity' } });
});

test('retrieval rejects status/redirect/encoding/declared size before body', async (t) => {
  for (const properties of [{ statusCode: 302 }, { statusCode: 404 },
    { headers: { 'content-encoding': 'gzip' } }, { headers: { 'content-length': '1048577' } },
    { headers: { 'content-length': 'Infinity' } }, { headers: { 'content-length': '-1' } }]) {
    const calls = mockHTTP(t, ({ response, callback }) => { Object.assign(response, properties); callback(response); });
    await assert.rejects(retrieveArtifact(uri('mp4'), RUN, 'mp4'));
    assert(calls[0].request.destroyed && calls[0].response.destroyed);
    t.mock.restoreAll();
  }
});

test('active unknown-length oversize destroys request and response before end', async (t) => {
  const calls = mockHTTP(t, ({ response, callback }) => {
    callback(response); response.emit('data', Buffer.alloc(1024 * 1024));
    response.emit('data', Buffer.alloc(1));
  });
  await assert.rejects(retrieveArtifact(uri('gif'), RUN, 'gif'), /active byte budget/);
  assert(calls[0].request.destroyed && calls[0].response.destroyed);
});

test('absolute deadline is not reset by trickling chunks; stalls are destroyed', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout', 'Date'] });
  const calls = mockHTTP(t, ({ response, callback }) => { callback(response); response.emit('data', Buffer.from('x')); });
  const pending = retrieveArtifact(uri('mp4'), RUN, 'mp4');
  const rejection = assert.rejects(pending, /absolute deadline/);
  await Promise.resolve();
  for (let n = 0; n < 3; n++) { t.mock.timers.tick(9999); calls[0].response.emit('data', Buffer.from('x')); }
  assert(!calls[0].request.destroyed);
  t.mock.timers.tick(3);
  await rejection;
  assert(calls[0].request.destroyed && calls[0].response.destroyed);
});

test('stream errors/abort/early close and Content-Length mismatch propagate', async (t) => {
  for (const event of ['error', 'aborted', 'close', 'length']) {
    const calls = mockHTTP(t, ({ response, callback }) => {
      response.headers = { 'content-length': '2' }; callback(response);
      response.emit('data', Buffer.from('x'));
      if (event === 'length') response.emit('end'); else response.emit(event, new Error('stream failed'));
    });
    await assert.rejects(retrieveArtifact(uri('json'), RUN, 'json'));
    assert(calls[0].request.destroyed && calls[0].response.destroyed);
    t.mock.restoreAll();
  }
});

function fakeVideo(behavior = 'advance') {
  const listeners = {};
  let callback;
  const video = { isConnected: true, currentSrc: `${ORIGIN.replace('http:', 'blob:http:')}/id`,
    readyState: 4, currentTime: 0, duration: 0.2, videoWidth: 128, videoHeight: 86,
    paused: true, error: null, loop: false, ended: false, cancelled: false,
    getAttribute: () => video.currentSrc,
    addEventListener: (name, fn) => { listeners[name] = fn; },
    removeEventListener: (name) => { delete listeners[name]; },
    requestVideoFrameCallback: (fn) => { callback = fn; return 1; },
    cancelVideoFrameCallback: () => { video.cancelled = true; },
    pause: () => { video.paused = true; },
    play: () => {
      assert(listeners.ended && listeners.error && callback, 'listeners and callback precede play');
      video.paused = false;
      if (behavior === 'pending') return new Promise(() => {});
      if (behavior === 'reject') return Promise.reject(new Error('play rejected'));
      queueMicrotask(() => {
        if (behavior === 'error') { video.error = { code: 3 }; listeners.error(); return; }
        callback(1, { mediaTime: 0.02, presentedFrames: 1, width: 128, height: 86 });
        callback(2, { mediaTime: behavior === 'static' ? 0.02 : 0.04,
          presentedFrames: behavior === 'static' ? 1 : 2, width: 128, height: 86 });
        video.currentTime = 0.2; video.ended = true; video.paused = true; listeners.ended();
      });
      return Promise.resolve();
    } };
  return { video, listeners };
}

test('actual media callback algorithm requires advancing native frames and ended', async () => {
  const { video, listeners } = fakeVideo();
  const result = await playNativeVideo(video, expected);
  assert(result.ended && result.current_time > 0 && result.frames.length === 2);
  assert(video.cancelled); assert.deepEqual(listeners, {});
});

test('native media error, play rejection, static/nonadvancing video reject with cleanup', async () => {
  for (const behavior of ['error', 'reject', 'static']) {
    const { video, listeners } = fakeVideo(behavior);
    await assert.rejects(playNativeVideo(video, expected));
    assert(video.paused && video.cancelled); assert.deepEqual(listeners, {});
  }
  const { video } = fakeVideo(); video.currentSrc = 'blob:http://other/id';
  await assert.rejects(playNativeVideo(video, expected), /unexpected native video/);
});

test('10s absolute media deadline includes unresolved play promise and cleans up', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout', 'Date'] });
  const { video, listeners } = fakeVideo('pending');
  const pending = playNativeVideo(video, expected);
  const rejection = assert.rejects(pending, /absolute deadline/);
  t.mock.timers.tick(10000); await rejection;
  assert(video.paused && video.cancelled); assert.deepEqual(listeners, {});
});

test('ended alone cannot bypass a pending play promise deadline', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout', 'Date'] });
  const { video, listeners } = fakeVideo();
  const originalPlay = video.play;
  video.play = () => { originalPlay(); return new Promise(() => {}); };
  const pending = playNativeVideo(video, expected);
  const rejection = assert.rejects(pending, /absolute deadline/);
  await Promise.resolve();
  assert(video.ended && listeners.ended);
  t.mock.timers.tick(10000); await rejection;
  assert(video.cancelled); assert.deepEqual(listeners, {});
});

function png(value) {
  const b = Buffer.alloc(25); Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]).copy(b);
  b.write('IHDR', 12); b.writeUInt32BE(128, 16); b.writeUInt32BE(85, 20); b[24] = value; return b;
}
function fakeImage(changing = true) {
  let n = 0;
  const state = { currentSrc: `blob:${ORIGIN}/image`, isConnected: true, complete: true,
    naturalWidth: 128, naturalHeight: 85, getAttribute: () => `blob:${ORIGIN}/image` };
  return { evaluate: async (fn, argument) => fn(state, argument), screenshot: async () => png(changing ? n++ : 0), dispose: async () => {} };
}

test('unchanged GIF element needs distinct PNG bytes; static images fail after 20 samples', async () => {
  let samples = 0, waits = 0;
  const page = { waitForTimeout: async (ms) => { assert.equal(ms, 35); waits++; } };
  const save = async () => { samples++; };
  assert.equal((await sampleNativeGIF(page, fakeImage(), [128, 85], save)).samples.length, 2);
  samples = 0; waits = 0;
  await assert.rejects(sampleNativeGIF(page, fakeImage(false), [128, 85], save), /stayed static/);
  assert.equal(samples, 20); assert.equal(waits, 19);
});

function mockFilesystem(t) {
  const saved = new Map();
  t.mock.method(fs, 'lstat', async () => ({ isDirectory: () => true, isSymbolicLink: () => false,
    uid: process.getuid(), mode: 0o40700, dev: 1, ino: 2 }));
  t.mock.method(fs, 'realpath', async (p) => p);
  t.mock.method(fs, 'readdir', async () => []);
  t.mock.method(fs, 'open', async (p, flags, mode) => {
    assert.equal(flags, 'wx'); assert.equal(mode, 0o600); assert(!saved.has(p));
    return { writeFile: async (bytes) => { saved.set(p, bytes); }, close: async () => {} };
  });
  return saved;
}

function fakePage(options = {}) {
  const clicks = [];
  const run = options.run || RUN, mode = options.mode || 'mlflow-only';
  let waiting, selected;
  const visible = { waitFor: async () => {}, count: async () => 1 };
  const page = {
    url: () => options.pageURL || `${ORIGIN}/#/experiments/1/runs/${run}/artifacts`,
    title: async () => `Run ${run} - MLflow`,
    getByRole: (role, config) => {
      if (role === 'img') {
        assert.deepEqual(config, { name: `evaluation/${STEM}.gif`, exact: true });
        return { ...visible, count: async () => options.imageCount ?? 1, elementHandle: async () => fakeImage() };
      }
      assert.equal(role, 'tab');
      return { getAttribute: async () => 'true' };
    },
    waitForResponse: (predicate, config) => {
      assert.equal(config.timeout, 30000);
      return new Promise((resolve, reject) => { waiting = { predicate, resolve, reject }; });
    },
    getByLabel: (name, exact) => ({ ...visible,
      getAttribute: async (key) => key === 'data-testid' ? 'artifact-tree-node' : name,
      click: async () => {
        assert.deepEqual(exact, { exact: true }); assert(waiting, 'wait must precede click');
        selected = name.split('.').at(-1); clicks.push(selected);
        if (options.noResponse) { waiting.reject(new Error('native UI response absent')); return; }
        const response = { url: () => uri(selected, run), status: () => options.status || 200,
          request: () => ({ redirectedFrom: () => options.redirect || null }), headersArray: async () => [] };
        assert(waiting.predicate(response)); waiting.resolve(response); waiting = null;
      } }),
    locator: (selector) => {
      if (selector === '[data-testid="experiment-runs-link"]') return {
        getAttribute: async () => '#/experiments/1/evaluation-runs', innerText: async () => `physical-ai-logging-${mode}` };
      if (selector.startsWith('span[')) { assert.equal(selector, `span[title="evaluation/${STEM}.${selected}"]`); return visible; }
      if (selector.startsWith('video')) return { ...visible, evaluate: async (fn, arg) => fn(fakeVideo().video, arg), screenshot: async () => png(0) };
      assert.fail('generic blob selector includes the hidden native full-screen image copy');
    },
    waitForTimeout: async () => {},
  };
  return { page, clicks };
}

function mockNativeBlobs(t, bytes, mismatch = false) {
  t.mock.method(globalThis, 'fetch', async (url) => {
    assert(url.startsWith(`blob:${ORIGIN}/`), 'only the existing native blob is read');
    const value = mismatch ? Buffer.from('different native blob') : bytes[url.endsWith('/image') ? 'gif' : 'mp4'];
    let read = false;
    return { ok: true, body: { getReader: () => ({
      read: async () => { if (read) return { done: true }; read = true; return { value, done: false }; },
      releaseLock() {},
    }) } };
  });
}

test('mocked end-to-end selects MP4/GIF/JSON once, saves wx0600 bytes and scoped summary', async (t) => {
  const f = fixture(), saved = mockFilesystem(t), { page, clicks } = fakePage();
  mockNativeBlobs(t, f.bytes);
  mockHTTP(t, ({ response, callback, url }) => {
    const ext = new URL(url).searchParams.get('path').split('.').at(-1);
    callback(response); response.emit('data', f.bytes[ext]); response.emit('end');
  });
  const summary = await runMLflowArtifacts(page, f.manifest, '/mock-output');
  assert.deepEqual(clicks, ['mp4', 'gif', 'json']);
  for (const ext of clicks) assert.deepEqual(saved.get(`/mock-output/${STEM}.${ext}`), f.bytes[ext]);
  assert.equal(summary.artifacts.length, 3);
  for (const ext of clicks) assert(saved.has(`/mock-output/${STEM}-${ext}-receipt.json`));
  assert(summary.scope.includes('full6.2/V remain UNVERIFIED'));
  assert(saved.has('/mock-output/mlflow-artifacts.json'));
  assert.equal(summary.source_frames, 64); assert.deepEqual(summary.provenance, {});
});

test('correct downloaded bytes cannot qualify a different native player blob', async (t) => {
  const f = fixture(), saved = mockFilesystem(t);
  mockNativeBlobs(t, f.bytes, true);
  mockHTTP(t, ({ response, callback, url }) => {
    const ext = new URL(url).searchParams.get('path').split('.').at(-1);
    callback(response); response.emit('data', f.bytes[ext]); response.emit('end');
  });
  await assert.rejects(runMLflowArtifacts(fakePage().page, f.manifest, '/mock-output'), /native blob.*mismatch/);
  assert(saved.has(`/mock-output/${STEM}.mp4`));
  assert(!saved.has('/mock-output/mlflow-artifacts.json'));
});

test('ambiguous named native GIF previews still fail rather than choose a first image', async (t) => {
  const f = fixture(), saved = mockFilesystem(t);
  mockNativeBlobs(t, f.bytes);
  mockHTTP(t, ({ response, callback, url }) => {
    const ext = new URL(url).searchParams.get('path').split('.').at(-1);
    callback(response); response.emit('data', f.bytes[ext]); response.emit('end');
  });
  await assert.rejects(runMLflowArtifacts(fakePage({ imageCount: 2 }).page, f.manifest, '/mock-output'), /one native blob image required/);
  assert(JSON.parse(saved.get(`/mock-output/${STEM}-mp4-receipt.json`)).playback.ended);
  assert(!saved.has('/mock-output/mlflow-artifacts.json'));
});

test('no fallback HTTP after absent UI response, wrong page, UI status or redirect', async (t) => {
  mockFilesystem(t);
  let calls = 0;
  t.mock.method(http, 'get', () => { calls++; throw new Error('must not fetch'); });
  for (const options of [{ noResponse: true }, { pageURL: `${ORIGIN}/#/wrong` }, { status: 500 }, { redirect: {} }]) {
    await assert.rejects(runMLflowArtifacts(fakePage(options).page, fixture().manifest, '/mock-output'));
  }
  assert.equal(calls, 0);
});

test('hash and JSON metric mismatch retain partial bytes but never write success summary', async (t) => {
  for (const fault of ['hash', 'metrics']) {
    const f = fixture(), saved = mockFilesystem(t);
    mockNativeBlobs(t, f.bytes);
    if (fault === 'hash') f.media.files.mp4.sha256 = '0'.repeat(64);
    else {
      f.bytes.json = Buffer.from(JSON.stringify({ frame_step: 64, metrics: {}, provenance: {} }));
      f.media.files.json.sha256 = sha(f.bytes.json);
    }
    mockHTTP(t, ({ response, callback, url }) => {
      callback(response); response.emit('data', f.bytes[new URL(url).searchParams.get('path').split('.').at(-1)]); response.emit('end');
    });
    await assert.rejects(runMLflowArtifacts(fakePage().page, f.manifest, '/mock-output'), /mismatch/);
    assert(saved.has(`/mock-output/${STEM}.mp4`));
    assert(!saved.has('/mock-output/mlflow-artifacts.json'));
    t.mock.restoreAll();
  }
});

test('nonempty, insecure, symlinked or replaced output fails closed', async (t) => {
  const f = fixture();
  for (const fault of ['nonempty', 'mode', 'symlink', 'replaced']) {
    const saved = mockFilesystem(t);
    if (fault === 'nonempty') t.mock.method(fs, 'readdir', async () => ['existing']);
    if (fault === 'symlink') t.mock.method(fs, 'realpath', async () => '/elsewhere');
    if (fault === 'mode' || fault === 'replaced') {
      let calls = 0;
      t.mock.method(fs, 'lstat', async () => ({ isDirectory: () => true, isSymbolicLink: () => false,
        uid: process.getuid(), mode: fault === 'mode' ? 0o40755 : 0o40700, dev: 1, ino: calls++ ? 3 : 2 }));
    }
    mockHTTP(t, ({ response, callback }) => { callback(response); response.emit('data', f.bytes.mp4); response.emit('end'); });
    await assert.rejects(runMLflowArtifacts(fakePage().page, f.manifest, '/mock-output'));
    assert.equal(saved.size, 0);
    t.mock.restoreAll();
  }
});

function bothFixture() {
  const f = fixture(), main = structuredClone(f.main);
  main.run_id = 'e58df20bab2e4b7c9fb121b36320b9a6';
  for (const ext of ['mp4', 'gif', 'json']) main.media[64].files[ext].path = `/both/mlflow/evaluation/${STEM}.${ext}`;
  const tb = structuredClone(main.media[64]);
  for (const ext of ['mp4', 'gif', 'json']) tb.files[ext].path = `/both/tb/artifacts/evaluation/${STEM}.${ext}`;
  f.manifest.modes.both = { native: { mlflow: { main }, tensorboard: { main: { media: { 64: tb } } } } };
  return { ...f, main, media: main.media[64], tb };
}

test('mocked both journey uses its own run, breadcrumb, payloads and scoped receipts', async (t) => {
  const f = bothFixture(), saved = mockFilesystem(t);
  const { page, clicks } = fakePage({ mode: 'both', run: f.main.run_id });
  // A different single-logger payload must not supply the both-mode expectations.
  f.manifest.modes['mlflow-only'].native.mlflow.main.media[64].files.mp4.sha256 = '0'.repeat(64);
  mockNativeBlobs(t, f.bytes);
  const calls = mockHTTP(t, ({ response, callback, url }) => {
    const ext = new URL(url).searchParams.get('path').split('.').at(-1);
    callback(response); response.emit('data', f.bytes[ext]); response.emit('end');
  });
  const summary = await runMLflowArtifacts(page, f.manifest, '/mock-output', 'both');
  assert.equal(summary.mode, 'both'); assert.equal(summary.run_id, f.main.run_id);
  assert.equal(summary.experiment_id, '1');
  assert(summary.scope.startsWith('both native UI/artifact-route proof'));
  assert(summary.scope.includes('both/disabled modes/partial-delivery and full6.2/V remain UNVERIFIED'));
  assert.deepEqual(clicks, ['mp4', 'gif', 'json']); assert.equal(calls.length, 3);
  for (const ext of clicks) {
    assert.deepEqual(saved.get(`/mock-output/${STEM}.${ext}`), f.bytes[ext]);
    const receipt = JSON.parse(saved.get(`/mock-output/${STEM}-${ext}-receipt.json`));
    assert.equal(receipt.artifact.uri, uri(ext, f.main.run_id));
    assert.equal(receipt.artifact.sha256, f.tb.files[ext].sha256);
  }
  assert(summary.mp4.ended); assert.equal(summary.gif.samples.length, 2);
  assert.deepEqual(summary.json.metrics, f.media.metrics);
  assert.equal(saved.get('/mock-output/mlflow-artifacts.json'), `${JSON.stringify(summary, null, 2)}\n`);
});

test('both cross-logger hash, metric or frame mismatch rejects before UI, HTTP or filesystem', async (t) => {
  const touched = [];
  const forbidden = (name) => () => { touched.push(name); throw new Error(`unexpected ${name}`); };
  const page = new Proxy({}, { get: forbidden('UI') });
  for (const name of ['lstat', 'realpath', 'readdir', 'open']) t.mock.method(fs, name, forbidden(name));
  t.mock.method(http, 'get', forbidden('HTTP'));
  for (const fault of ['mp4', 'gif', 'json', 'metrics', 'frame']) {
    const f = bothFixture();
    if (fault === 'metrics') f.tb.metrics = {};
    else if (fault === 'frame') f.tb.frame_step = 32;
    else f.tb.files[fault].sha256 = '0'.repeat(64);
    await assert.rejects(runMLflowArtifacts(page, f.manifest, '/mock-output', 'both'),
      fault === 'frame' ? /same-rollout TensorBoard frame/ : /cross-logger.*mismatch/);
  }
  assert.deepEqual(touched, []);
});

test('unsupported modes reject before manifest access, UI, HTTP or filesystem', async (t) => {
  const touched = [];
  const forbidden = (name) => () => { touched.push(name); throw new Error(`unexpected ${name}`); };
  const manifest = new Proxy({}, { get: forbidden('manifest') }), page = new Proxy({}, { get: forbidden('UI') });
  for (const name of ['lstat', 'realpath', 'readdir', 'open']) t.mock.method(fs, name, forbidden(name));
  t.mock.method(http, 'get', forbidden('HTTP'));
  for (const mode of ['tb-only', 'disabled', 'Both', '', null, 1, {}, new String('both')]) {
    assert.throws(() => validateManifest(manifest, mode), /unsupported mode/);
    await assert.rejects(runMLflowArtifacts(page, manifest, '/mock-output', mode), /unsupported mode/);
  }
  assert.deepEqual(touched, []);
});
