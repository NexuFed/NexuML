'use strict';

const fs = require('node:fs/promises');
const path = require('node:path');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');
const http = require('node:http');

const ORIGIN = 'http://127.0.0.1:5001';
const STEM = 'frame_000000000064';
const LIMIT = 1024 * 1024;
const SCOPE = 'native UI/artifact-route proof; both/disabled modes/partial-delivery and full6.2/V remain UNVERIFIED';
const sha256 = (bytes) => crypto.createHash('sha256').update(bytes).digest('hex');
const hashValid = (s) => typeof s === 'string' && /^[a-f0-9]{64}$/.test(s);

function validateManifest(manifest, mode = 'mlflow-only') {
  assert(mode === 'mlflow-only' || mode === 'both', 'unsupported mode');
  const main = manifest?.modes?.[mode]?.native?.mlflow?.main;
  assert(main && /^[a-f0-9]{32}$/.test(main.run_id), 'invalid main run_id');
  assert(typeof main.experiment_id === 'string' && /^\d+$/.test(main.experiment_id), 'invalid experiment_id');
  assert.equal(main.status, 'FINISHED');
  const media = main.media?.['64'];
  assert.equal(media?.frame_step, 64);
  assert.deepEqual(Object.keys(media.files).sort(), ['gif', 'json', 'mp4']);
  for (const ext of ['mp4', 'gif', 'json']) {
    assert.equal(path.basename(media.files[ext].path), `${STEM}.${ext}`);
    assert(hashValid(media.files[ext].sha256), 'invalid source sha256');
  }
  assert.deepEqual(media.gif, { frames: 9, size: [128, 85], changing_nonuniform_frames: true });
  assert.deepEqual(media.mp4, { frames: 10, codec: 'h264', size: [128, 86] });
  assert.equal(media.captured_frames, 10);
  assert.equal(media.fps, 50);
  assert.deepEqual(media.provenance, {}, 'in-fit policy file provenance is not supplied');
  assert(media.metrics && Object.keys(media.metrics).length > 0 &&
    Object.values(media.metrics).every(Number.isFinite), 'finite expected metrics required');
  if (mode === 'both') {
    const tb = manifest.modes.both.native.tensorboard?.main?.media?.['64'];
    assert.equal(tb?.frame_step, 64, 'same-rollout TensorBoard frame required');
    assert.deepEqual(tb.metrics, media.metrics, 'cross-logger metrics mismatch');
    for (const ext of ['mp4', 'gif', 'json']) {
      assert.equal(path.basename(tb.files[ext].path), `${STEM}.${ext}`);
      assert.equal(tb.files[ext].sha256, media.files[ext].sha256, `cross-logger ${ext} hash mismatch`);
    }
  }
  return { main, media, duration: media.mp4.frames / media.fps };
}

function artifactURI(uri, run, ext) {
  assert(typeof uri === 'string' && uri.length <= 4096, 'invalid artifact URI');
  const url = new URL(uri);
  assert(url.origin === ORIGIN && !url.username && !url.password && !uri.includes('#') &&
    url.pathname === '/get-artifact', 'unexpected artifact route');
  assert.deepEqual([...url.searchParams.keys()].sort(), ['path', 'run_uuid'], 'unexpected/duplicate query');
  assert.equal(url.searchParams.get('run_uuid'), run, 'artifact run mismatch');
  assert.equal(url.searchParams.get('path'), `evaluation/${STEM}.${ext}`, 'artifact path mismatch');
  // Reject URL-parser normalization (including alternate host spellings).
  assert.equal(url.href, uri, 'noncanonical artifact URI');
  return uri;
}

function retrieveArtifact(uri, run, ext) {
  artifactURI(uri, run, ext);
  return new Promise((resolve, reject) => {
    let request, response, settled = false, length = null, total = 0;
    const chunks = [];
    const started = Date.now();
    function fail(error) {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      response?.destroy();
      request?.destroy();
      reject(error);
    }
    const timer = setTimeout(() => fail(new Error('artifact absolute deadline exceeded')), 30000);
    function deadline() { assert(Date.now() - started < 30000, 'artifact absolute deadline exceeded'); }
    try {
      request = http.get(uri, { agent: false, headers: { 'Accept-Encoding': 'identity' } }, (res) => {
        response = res;
        if (settled) { res.destroy(); return; }
        res.on('error', fail);
        res.on('aborted', () => fail(new Error('artifact response aborted')));
        try {
          deadline();
          assert.equal(res.statusCode, 200, 'artifact HTTP status (redirects forbidden)');
          assert(!res.headers['content-encoding'] || res.headers['content-encoding'] === 'identity', 'encoded artifact forbidden');
          if (res.headers['content-length'] !== undefined) {
            assert(/^\d+$/.test(res.headers['content-length']), 'invalid Content-Length');
            length = Number(res.headers['content-length']);
            assert(Number.isSafeInteger(length) && length <= LIMIT, 'Content-Length exceeds budget');
          }
        } catch (error) { fail(error); return; }
        res.on('data', (chunk) => {
          if (settled) return;
          try {
            deadline();
            assert(Buffer.isBuffer(chunk), 'artifact must be raw bytes');
            total += chunk.length;
            assert(total <= LIMIT, 'artifact active byte budget exceeded');
            chunks.push(chunk);
          } catch (error) { fail(error); }
        });
        res.on('end', () => {
          if (settled) return;
          try {
            deadline();
            assert(total > 0, 'empty artifact');
            if (length !== null) assert.equal(total, length, 'Content-Length mismatch');
            settled = true;
            clearTimeout(timer);
            const bytes = Buffer.concat(chunks, total);
            resolve({ bytes, receipt: {
              uri, status: res.statusCode, raw_headers: res.rawHeaders,
              bytes: total, sha256: sha256(bytes),
              role: 'independent direct HTTP retrieval of native UI-observed artifact route',
              budget: '1MiB accumulated bytes; network chunk overshoot possible; absolute 30s deadline',
            } });
          } catch (error) { fail(error); }
        });
        res.on('close', () => { if (!settled) fail(new Error('artifact closed before completion')); });
      });
      request.on('error', fail);
    } catch (error) { fail(error); }
  });
}

// Read the existing player blob; a matching second HTTP response alone proves no association.
async function nativeBlobIdentity(element, expected) {
  const src = element.currentSrc;
  const url = new URL(src);
  if (url.protocol !== 'blob:' || url.origin !== expected.origin || url.hash || url.search ||
    element.getAttribute('src') !== src || !element.isConnected) throw new Error('unexpected native blob source');
  const controller = new AbortController();
  let reader, timer, primary;
  const work = (async () => {
    const response = await fetch(src, { signal: controller.signal });
    if (!response.ok || !response.body) throw new Error('native blob read failed');
    reader = response.body.getReader();
    let size = 0;
    const chunks = [];
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > expected.limit) throw new Error('native blob exceeds byte budget');
      chunks.push(value);
    }
    if (!size) throw new Error('native blob is empty');
    const bytes = new Uint8Array(size);
    let offset = 0;
    for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
    const digest = await crypto.subtle.digest('SHA-256', bytes);
    if (!element.isConnected || element.currentSrc !== src || element.getAttribute('src') !== src) throw new Error('native blob source changed during hashing');
    return { src, bytes: size, sha256: Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('') };
  })();
  try {
    return await Promise.race([work, new Promise((_, reject) => {
      timer = setTimeout(() => { controller.abort(); reject(new Error('native blob identity deadline exceeded')); }, 10000);
    })]);
  } catch (error) { primary = error; throw error; }
  finally {
    clearTimeout(timer);
    controller.abort();
    try { reader?.releaseLock(); }
    catch (error) { if (!primary) throw error; (primary.cleanupErrors ??= []).push(String(error)); }
  }
}

// Runs on the unchanged native HTMLMediaElement; never seeks or substitutes frames.
function playNativeVideo(video, expected) {
  return new Promise((resolve, reject) => {
    const startedAt = Date.now(), frames = [];
    let callback, timer, settled = false, src, playResolved = false, endedSeen = false;
    function finish(error) {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      video.removeEventListener('ended', ended);
      video.removeEventListener('error', failed);
      try { if (callback !== undefined) video.cancelVideoFrameCallback(callback); } catch (_) { /* cleanup only */ }
      if (error) {
        try { video.pause(); } catch (_) { /* preserve primary failure */ }
        reject(error);
      } else resolve({ src, started_at: startedAt, ended_at: Date.now(), ended: video.ended,
        current_time: video.currentTime, duration: video.duration,
        size: [video.videoWidth, video.videoHeight], frames });
    }
    function state() {
      if (Date.now() - startedAt >= 10000) throw new Error('native video absolute deadline exceeded');
      if (!video.isConnected || video.error || video.currentSrc !== src) throw new Error('native video error/detached/source changed');
    }
    function failed() { finish(new Error('native video media error')); }
    function ended() {
      try {
        state();
        if (!video.ended || video.currentTime <= 0 || frames.length < 2 ||
          !frames.every((frame, index) => index === 0 || (frame.mediaTime > frames[index - 1].mediaTime &&
            frame.presentedFrames > frames[index - 1].presentedFrames))) throw new Error('native video did not advance and end');
        endedSeen = true;
        if (playResolved) finish();
      } catch (error) { finish(error); }
    }
    function frame(now, metadata) {
      if (settled) return;
      try {
        state();
        if (!Number.isFinite(metadata.mediaTime) || !Number.isFinite(metadata.presentedFrames)) throw new Error('invalid native frame metadata');
        frames.push({ now, mediaTime: metadata.mediaTime, presentedFrames: metadata.presentedFrames,
          width: metadata.width, height: metadata.height, expectedDisplayTime: metadata.expectedDisplayTime });
        if (frames.length > 1024) throw new Error('native frame callback budget exceeded');
        callback = video.requestVideoFrameCallback(frame);
      } catch (error) { finish(error); }
    }
    try {
      src = video.currentSrc;
      const blob = new URL(src);
      if (blob.protocol !== 'blob:' || blob.origin !== expected.origin || (expected.src && src !== expected.src) ||
        video.getAttribute('src') !== src || video.readyState < 2 || video.error ||
        video.videoWidth !== expected.size[0] || video.videoHeight !== expected.size[1] ||
        !Number.isFinite(video.duration) || video.duration <= 0 ||
        Math.abs(video.duration - expected.duration) > 0.001 || !video.paused || video.currentTime !== 0 || video.loop ||
        typeof video.requestVideoFrameCallback !== 'function') throw new Error('unexpected native video state');
      state();
      timer = setTimeout(() => finish(new Error('native video absolute deadline exceeded')), 10000);
      video.addEventListener('ended', ended);
      video.addEventListener('error', failed);
      callback = video.requestVideoFrameCallback(frame);
      Promise.resolve(video.play()).then(() => {
        if (settled) return;
        playResolved = true;
        if (endedSeen) ended();
      }, (error) => finish(error));
    } catch (error) { finish(error); }
  });
}

function pngSize(bytes) {
  assert(Buffer.isBuffer(bytes) && bytes.length >= 24 &&
    bytes.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10])) &&
    bytes.toString('ascii', 12, 16) === 'IHDR', 'native screenshot must be PNG');
  const size = [bytes.readUInt32BE(16), bytes.readUInt32BE(20)];
  assert(size.every((n) => Number.isInteger(n) && n > 0 && n <= 4096), 'invalid PNG dimensions');
  return size;
}

async function sampleNativeGIF(page, image, expected, save) {
  const initial = await image.evaluate((img) => ({ src: img.currentSrc, attribute: img.getAttribute('src') }));
  const blob = new URL(initial.src);
  assert(blob.protocol === 'blob:' && blob.origin === ORIGIN && initial.src === initial.attribute, 'unexpected native GIF source');
  const samples = [];
  let first;
  async function check() {
    const state = await image.evaluate((img) => ({ src: img.currentSrc, connected: img.isConnected,
      complete: img.complete, size: [img.naturalWidth, img.naturalHeight] }));
    assert(state.connected && state.complete && state.src === initial.src, 'native GIF detached/source changed');
    assert.deepEqual(state.size, expected);
  }
  for (let n = 0; n < 20; n++) {
    if (n) await page.waitForTimeout(35);
    await check();
    const png = await image.screenshot({ type: 'png', animations: 'allow', scale: 'css', timeout: 30000 });
    await check();
    const size = pngSize(png);
    if (!n) first = png;
    else assert.deepEqual(size, samples[0].size, 'native GIF screenshot geometry changed');
    const filename = `${STEM}-gif-sample-${n}.png`;
    await save(filename, png);
    samples.push({ filename, sha256: sha256(png), size });
    if (!png.equals(first)) return { src: initial.src, natural_size: expected, samples,
      scope: 'unchanged native img PNG samples; no manual frame advancement' };
  }
  throw new Error('native GIF stayed static across 20 samples');
}

async function checkOutput(output) {
  assert(typeof output === 'string' && path.isAbsolute(output) && path.resolve(output) === output, 'canonical absolute output required');
  const stat = await fs.lstat(output);
  assert(stat.isDirectory() && !stat.isSymbolicLink() && stat.uid === process.getuid() &&
    (stat.mode & 0o7777) === 0o700, 'real owned 0700 output required');
  assert.equal(await fs.realpath(output), output, 'output symlinks forbidden');
  return stat;
}

async function runMLflowArtifacts(page, manifest, output, mode = 'mlflow-only') {
  const { main, media, duration } = validateManifest(manifest, mode);
  const initial = await checkOutput(output);
  assert.deepEqual(await fs.readdir(output), [], 'caller-created empty output required');
  async function save(filename, bytes) {
    const current = await checkOutput(output);
    assert(current.dev === initial.dev && current.ino === initial.ino, 'output directory changed');
    const file = await fs.open(path.join(output, filename), 'wx', 0o600);
    try { await file.writeFile(bytes); } finally { await file.close(); }
  }
  async function context() {
    assert.equal(page.url(), `${ORIGIN}/#/experiments/${main.experiment_id}/runs/${main.run_id}/artifacts`, 'unexpected run/artifacts page');
    assert.equal(await page.title(), `Run ${main.run_id} - MLflow`, 'wrong run header context');
    const breadcrumb = page.locator('[data-testid="experiment-runs-link"]');
    assert.equal(await breadcrumb.getAttribute('href'), `#/experiments/${main.experiment_id}/evaluation-runs`);
    assert.equal(await breadcrumb.innerText(), `physical-ai-logging-${mode}`);
    assert.equal(await page.getByRole('tab', { name: 'Artifacts', exact: true }).getAttribute('aria-selected'), 'true');
  }
  const artifacts = [];
  let mp4, gif, json;
  for (const ext of ['mp4', 'gif', 'json']) {
    await context();
    const filename = `${STEM}.${ext}`, fullpath = `evaluation/${filename}`;
    const file = page.getByLabel(filename, { exact: true });
    assert.equal(await file.count(), 1, 'one native tree file required');
    assert.equal(await file.getAttribute('data-testid'), 'artifact-tree-node');
    assert.equal(await file.getAttribute('artifact-name'), filename);
    const responsePromise = page.waitForResponse((response) => {
      try { artifactURI(response.url(), main.run_id, ext); return true; } catch (_) { return false; }
    }, { timeout: 30000 });
    // Both promises are observed even when the actual native click fails.
    const [, response] = await Promise.all([file.click({ timeout: 30000 }), responsePromise]);
    assert.equal(response.status(), 200, 'native UI response failed');
    assert.equal(response.request().redirectedFrom(), null, 'native UI redirect forbidden');
    const uri = artifactURI(response.url(), main.run_id, ext);
    await page.locator(`span[title="${fullpath}"]`).waitFor({ state: 'visible', timeout: 30000 });
    await context();
    const { bytes, receipt } = await retrieveArtifact(uri, main.run_id, ext);
    await save(filename, bytes);
    assert.equal(receipt.sha256, media.files[ext].sha256, 'artifact manifest hash mismatch');
    receipt.filename = filename;
    receipt.native_ui = { uri, status: response.status(), raw_headers: await response.headersArray(), header: fullpath };
    artifacts.push(receipt);
    if (ext === 'mp4') {
      const video = page.locator('video[aria-label="video"]');
      assert.equal(await video.count(), 1);
      await video.waitFor({ state: 'visible', timeout: 30000 });
      const native = await video.evaluate(nativeBlobIdentity, { origin: ORIGIN, limit: LIMIT });
      assert.equal(native.sha256, receipt.sha256, 'native blob/downloaded payload mismatch');
      mp4 = await video.evaluate(playNativeVideo, { origin: ORIGIN, size: media.mp4.size, duration, src: native.src });
      mp4.payload_association = { uri, sha256: receipt.sha256, native_blob: native, source_frames: 64, fullpath };
      mp4.screenshot = `${STEM}-native-video.png`;
      await save(mp4.screenshot, await video.screenshot({ type: 'png', animations: 'allow', timeout: 30000 }));
    } else if (ext === 'gif') {
      const image = page.getByRole('img', { name: fullpath, exact: true });
      assert.equal(await image.count(), 1, 'one native blob image required');
      await image.waitFor({ state: 'visible', timeout: 30000 });
      const handle = await image.elementHandle();
      assert(handle, 'native image detached');
      let primary;
      try {
        const native = await handle.evaluate(nativeBlobIdentity, { origin: ORIGIN, limit: LIMIT });
        assert.equal(native.sha256, receipt.sha256, 'native blob/downloaded payload mismatch');
        gif = await sampleNativeGIF(page, handle, media.gif.size, save);
        assert.equal(gif.src, native.src, 'native GIF source changed after hashing');
        gif.payload_association = { uri, sha256: receipt.sha256, native_blob: native, source_frames: 64, fullpath };
      } catch (error) { primary = error; throw error; }
      finally {
        try { await handle.dispose(); }
        catch (error) { if (!primary) throw error; (primary.cleanupErrors ??= []).push(String(error)); }
      }
    } else {
      const text = bytes.toString('utf8');
      assert(Buffer.from(text, 'utf8').equals(bytes), 'JSON must be valid UTF-8');
      const payload = JSON.parse(text);
      assert.equal(payload.frame_step, 64);
      assert.deepEqual(payload.metrics, media.metrics, 'JSON metrics mismatch');
      assert.deepEqual(payload.provenance, {});
      json = { frame_step: 64, metrics: payload.metrics,
        inline: 'not required/unsupported by this proof; no Monaco text assertion', downloaded: true };
    }
    await save(`${STEM}-${ext}-receipt.json`, `${JSON.stringify({
      artifact: receipt, playback: ext === 'mp4' ? mp4 : ext === 'gif' ? gif : json,
    }, null, 2)}\n`);
  }
  await context();
  const summary = { scope: `${mode} ${SCOPE}`, mode, run_id: main.run_id, experiment_id: main.experiment_id,
    source_frames: 64, artifacts, mp4, gif, json, producer_decoded: { gif: media.gif, mp4: media.mp4 },
    provenance: {}, interpretation: 'No in-fit policy-file hash or fresh actor-load claim; no overall C/V/L/R/D promotion.' };
  await save('mlflow-artifacts.json', `${JSON.stringify(summary, null, 2)}\n`);
  return summary;
}

module.exports = { runMLflowArtifacts, validateManifest, artifactURI, retrieveArtifact, playNativeVideo, sampleNativeGIF };
