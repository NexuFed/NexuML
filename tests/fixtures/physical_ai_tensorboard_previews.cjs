'use strict';

const fs = require('node:fs/promises');
const path = require('node:path');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');

const BASE = 'http://127.0.0.1:6007/';
const STEPS = [0, 32, 64];
const POLICY_KEYS = ['untrained', 'intermediate', 'final'];
const SCOPE = 'native TB preview interaction only; scalar UI, MP4 retrieval and V/task6.1 remain UNVERIFIED';
const sha256 = (bytes) => crypto.createHash('sha256').update(bytes).digest('hex');
const hashValid = (hash) => typeof hash === 'string' && /^[a-f0-9]{64}$/.test(hash);
const dimensionValid = (n) => Number.isInteger(n) && n > 0 && n <= 4096;

function validateTensorBoardManifest(manifest, mode = 'tb-only') {
  assert(mode === 'tb-only' || mode === 'both', 'TensorBoard mode must be tb-only or both');
  const main = manifest?.modes?.[mode]?.native?.tensorboard?.main;
  const base = manifest?.modes?.[mode]?.logging?.tensorboard?.log_dir;
  assert(main && typeof main.log_dir === 'string' && path.isAbsolute(main.log_dir), 'native main log_dir required');
  assert(typeof base === 'string' && path.isAbsolute(base) && path.normalize(base) === base, 'native base log_dir required');
  const relativeRun = path.relative(base, main.log_dir);
  assert.equal(relativeRun, `physical-ai-logging-${mode}/cartpole-ppo_v1`);
  assert.equal(main.image_tag, 'eval/rl/episode');
  assert(Array.isArray(main.previews), 'native previews required');
  assert.deepEqual(main.previews.map((preview) => preview.step), STEPS, 'native steps must be exactly [0,32,64]');
  main.previews.forEach((preview, index) => {
    assert(hashValid(preview.sha256), 'invalid native preview sha256');
    assert(Array.isArray(preview.decoded?.size) && preview.decoded.size.length === 2 &&
      preview.decoded.size.every(dimensionValid), 'invalid native preview dimensions');
    assert(Number.isInteger(preview.decoded.frames) && preview.decoded.frames > 1 &&
      preview.decoded.frames <= 1024 && preview.decoded.changing_nonuniform_frames === true,
    'producer must identify bounded animated nonuniform native frames');
    const policy = main.associated_policy_exports?.[POLICY_KEYS[index]];
    assert.equal(policy?.source_frames, STEPS[index], 'associated policy source frames mismatch');
    assert(hashValid(policy?.policy_files_sha256?.['state_dict.pt']), 'invalid associated actor sha256');
  });
  return { main, relativeRun };
}

function decodeTensorBoardImageSource(src, relativeRun, imageTag, step) {
  assert(typeof src === 'string' && src.length > 0 && src.length <= 8192, 'native image src required');
  const url = new URL(src, BASE);
  assert(url.origin === new URL(BASE).origin && !url.username && !url.password && !url.hash &&
    url.pathname === '/data/plugin/images/individualImage', 'unexpected native image URL');
  assert.equal(url.searchParams.getAll('blob_key').length, 1, 'one blob_key required');
  const encoded = url.searchParams.get('blob_key');
  assert(/^[A-Za-z0-9_-]+$/.test(encoded), 'invalid base64url blob_key');
  const bytes = Buffer.from(encoded, 'base64url');
  assert(bytes.length <= 4096 && bytes.toString('base64url') === encoded, 'noncanonical blob_key');
  const tuple = JSON.parse(bytes.toString('utf8'));
  assert.deepEqual(tuple, ['', relativeRun, imageTag, step, 2], 'native image identity mismatch');
  return { url: url.href, tuple };
}

async function checkOutput(output) {
  assert(typeof output === 'string' && path.isAbsolute(output) && path.resolve(output) === output,
    'output must be a canonical absolute path');
  const stat = await fs.lstat(output);
  assert(stat.isDirectory() && !stat.isSymbolicLink() && stat.uid === process.getuid() &&
    (stat.mode & 0o7777) === 0o700, 'output must be a real owned 0700 directory');
  assert.equal(await fs.realpath(output), output, 'output path must not contain symlinks');
  return stat;
}

// Read only: wait for the selected native image, including open shadow descendants.
function selectedNativeImage({ card, relativeRun, imageTag, step, ordinal, size }) {
  const nodes = [];
  function visit(root) {
    if (root.shadowRoot) visit(root.shadowRoot);
    for (const node of root.querySelectorAll('*')) {
      nodes.push(node);
      if (node.shadowRoot) visit(node.shadowRoot);
    }
  }
  visit(card);
  const sliders = nodes.filter((node) => node.matches('paper-slider#steps'));
  const images = nodes.filter((node) => node.matches('img'));
  if (sliders.length !== 1 || sliders[0].getAttribute('aria-valuenow') !== String(ordinal) || images.length !== 1) return false;
  const img = images[0];
  if (!img.complete || img.naturalWidth !== size[0] || img.naturalHeight !== size[1]) return false;
  const src = img.currentSrc || img.getAttribute('src');
  if (!src) return false;
  const url = new URL(src, location.href);
  const encoded = url.searchParams.get('blob_key');
  if (!encoded) return false;
  const tuple = JSON.parse(atob(encoded.replace(/-/g, '+').replace(/_/g, '/')));
  if (!Array.isArray(tuple) || tuple.length !== 5 || tuple[0] !== '' || tuple[1] !== relativeRun || tuple[2] !== imageTag || tuple[3] !== step || tuple[4] !== 2) return false;
  return { currentSrc: img.currentSrc, naturalWidth: img.naturalWidth, naturalHeight: img.naturalHeight };
}

function pngSize(bytes) {
  assert(Buffer.isBuffer(bytes) && bytes.length >= 24 &&
    bytes.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10])) &&
    bytes.toString('ascii', 12, 16) === 'IHDR', 'native screenshot must be PNG');
  const size = [bytes.readUInt32BE(16), bytes.readUInt32BE(20)];
  assert(size.every(dimensionValid), 'native screenshot dimensions out of bounds');
  return size;
}

async function runTensorBoardPreviews(page, manifest, output, mode = 'tb-only') {
  const { main, relativeRun } = validateTensorBoardManifest(manifest, mode);
  const initialDirectory = await checkOutput(output);
  assert.deepEqual(await fs.readdir(output), [], 'output must be caller-created and empty');
  function checkPage() {
    const url = new URL(page.url());
    assert.equal(url.href, `${BASE}#images`, 'page must already be the owned TB images page');
  }
  checkPage();
  async function save(filename, bytes) {
    const currentDirectory = await checkOutput(output);
    assert.equal(currentDirectory.dev, initialDirectory.dev, 'output directory changed');
    assert.equal(currentDirectory.ino, initialDirectory.ino, 'output directory changed');
    const file = await fs.open(path.join(output, filename), 'wx', 0o600);
    try { await file.writeFile(bytes); } finally { await file.close(); }
  }
  async function capture(index, reopened) {
    checkPage();
    const step = STEPS[index];
    const preview = main.previews[index];
    const prefix = `${reopened ? 'reopened-' : ''}step-${step}`;
    const card = page.locator('tf-image-loader').filter({ has: page.getByText(relativeRun, { exact: true }) });
    await card.waitFor({ state: 'visible', timeout: 30000 });
    assert.equal(await card.count(), 1, 'one native main-run card required');
    const slider = card.locator('paper-slider#steps');
    assert.equal(await slider.count(), 1, 'one native step slider required');
    await slider.waitFor({ state: 'visible', timeout: 30000 });
    for (const [attribute, value] of Object.entries({ role: 'slider', tabindex: '0', 'aria-valuemin': '0', 'aria-valuemax': '2' })) {
      assert.equal(await slider.getAttribute(attribute), value, `unexpected native slider ${attribute}`);
    }
    await slider.press('Home', { timeout: 30000 });
    for (let n = 0; n < index; n++) await slider.press('ArrowRight', { timeout: 30000 });
    const cardHandle = await card.elementHandle();
    assert(cardHandle, 'native card detached');
    const loaded = await page.waitForFunction(selectedNativeImage,
      { card: cardHandle, relativeRun, imageTag: main.image_tag, step, ordinal: index, size: preview.decoded.size },
      { timeout: 30000, polling: 35 });
    const selected = await loaded.jsonValue();
    await loaded.dispose();
    await cardHandle.dispose();
    assert.equal(await slider.getAttribute('aria-valuenow'), String(index));
    const img = card.locator('img');
    assert.equal(await img.count(), 1, 'one native image required');
    await img.waitFor({ state: 'visible', timeout: 30000 });
    const srcAttribute = await img.getAttribute('src');
    const source = decodeTensorBoardImageSource(srcAttribute, relativeRun, main.image_tag, step);
    assert.equal(decodeTensorBoardImageSource(selected.currentSrc, relativeRun, main.image_tag, step).url, source.url);
    const response = await page.request.get(source.url, { timeout: 30000, maxRedirects: 0 });
    assert.equal(response.status(), 200, 'native GIF retrieval failed');
    assert(/^image\/gif(?:\s*;|$)/i.test(response.headers()['content-type'] || ''), 'native image must be image/gif');
    const bytes = await response.body();
    assert(bytes.length >= 10 && bytes.length <= 1024 * 1024, 'native GIF size out of bounds');
    assert(/^GIF8[79]a$/.test(bytes.toString('ascii', 0, 6)), 'native GIF signature missing');
    assert.deepEqual([bytes.readUInt16LE(6), bytes.readUInt16LE(8)], preview.decoded.size, 'native GIF dimensions mismatch');
    assert.equal(sha256(bytes), preview.sha256, 'native preview hash mismatch');
    await save(`${prefix}.gif`, bytes);
    const imageHandle = await img.elementHandle();
    assert(imageHandle, 'native image detached');
    async function checkImage() {
      const state = await imageHandle.evaluate((image) => ({
        connected: image.isConnected, complete: image.complete, src: image.currentSrc,
        size: [image.naturalWidth, image.naturalHeight],
      }));
      assert(state.connected && state.complete, 'native image detached or unloaded');
      assert.equal(state.src, source.url, 'native image changed during animation sampling');
      assert.deepEqual(state.size, preview.decoded.size, 'native image dimensions mismatch');
    }
    const samples = [];
    let firstBytes;
    let changing = false;
    for (let n = 0; n < 20; n++) {
      if (n) await page.waitForTimeout(35);
      await checkImage();
      const png = await imageHandle.screenshot({ type: 'png', animations: 'allow', scale: 'css', timeout: 30000 });
      await checkImage();
      const size = pngSize(png);
      if (n) assert.deepEqual(size, samples[0].size, 'native image screenshot geometry changed');
      else firstBytes = png;
      const filename = `${prefix}-sample-${n}.png`;
      await save(filename, png);
      samples.push({ filename, sha256: sha256(png), size });
      if (!png.equals(firstBytes)) { changing = true; break; }
    }
    await imageHandle.dispose();
    assert(changing && new Set(samples.map((sample) => sample.sha256)).size >= 2,
      'native image screenshots stayed static across 20 samples');
    assert.equal(await slider.getAttribute('aria-valuenow'), String(index));
    const screenshot = `${prefix}-page.png`;
    await save(screenshot, await page.screenshot({ type: 'png', fullPage: true, animations: 'allow', timeout: 30000 }));
    const policyKey = POLICY_KEYS[index];
    const policy = main.associated_policy_exports[policyKey];
    return {
      step, ordinal: index, stage: index === 2 ? 'checkpoint_final' : policyKey, reopened,
      src_attribute: srcAttribute, current_src: source.url, selected_blob_tuple: source.tuple,
      native_gif: { filename: `${prefix}.gif`, sha256: sha256(bytes), size: preview.decoded.size },
      producer_decoded: preview.decoded,
      associated_policy_export: { key: policyKey, source_frames: policy.source_frames, actor_sha256: policy.policy_files_sha256['state_dict.pt'] },
      animation: { scope: 'unchanged native img element PNG screenshots only; no manual frame advancement', samples },
      screenshot,
    };
  }
  const stages = [];
  for (let index = 0; index < STEPS.length; index++) stages.push(await capture(index, false));
  await page.reload({ waitUntil: 'domcontentloaded', timeout: 30000 });
  const reopened = await capture(0, true);
  const summary = {
    scope: SCOPE, mode, relative_run: relativeRun, image_tag: main.image_tag, stages, reopened,
    interpretation: 'Associated export hashes are manifest provenance, not browser actor revalidation. Identical stage GIFs do not establish learning or convergence.',
  };
  await save('tensorboard-previews.json', `${JSON.stringify(summary, null, 2)}\n`);
  return summary;
}

module.exports = { runTensorBoardPreviews, decodeTensorBoardImageSource, validateTensorBoardManifest };
