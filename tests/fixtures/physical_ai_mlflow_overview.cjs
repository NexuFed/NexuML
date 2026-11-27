'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const crypto = require('node:crypto');
const { validateManifest } = require('./physical_ai_mlflow_artifacts.cjs');

const ORIGIN = 'http://127.0.0.1:5001';
const SCOPE = 'fixed G1 native Overview/status/latest metrics/selected config only; media/logger errors/whole6.2/V remain UNVERIFIED';
const sha256 = (bytes) => crypto.createHash('sha256').update(bytes).digest('hex');

async function checkOutput(output) {
  let directory = output;
  let leaf;
  for (;;) {
    const stat = await fs.lstat(directory);
    assert(stat.isDirectory() && !stat.isSymbolicLink(), 'real directory ancestors required');
    if (directory === output) leaf = stat;
    const parent = path.dirname(directory);
    if (parent === directory) break;
    directory = parent;
  }
  assert(leaf.uid === process.getuid() && (leaf.mode & 0o7777) === 0o700, 'owned 0700 output required');
  assert.equal(await fs.realpath(output), output, 'output symlinks forbidden');
  return leaf;
}

async function runMLflowOverview(page, manifest, output, mode = 'mlflow-only') {
  const { main } = validateManifest(manifest, mode);
  assert(page && ['url', 'title', 'getByRole', 'getByText', 'screenshot'].every(
    (key) => typeof page[key] === 'function'), 'public native page API required');
  assert(typeof output === 'string' && path.isAbsolute(output) && path.resolve(output) === output,
    'canonical absolute output required');
  const nativeHistories = {};
  for (const key of ['train/rl/frames', 'eval/rl/return_mean']) {
    const history = main.scalar_history?.[key];
    assert(Array.isArray(history) && history.length > 0, `native history required: ${key}`);
    assert(history.every((point, index) => point && Number.isFinite(point.value) &&
      Number.isSafeInteger(point.step) && point.step >= 0 &&
      (index === 0 || point.step > history[index - 1].step)), `finite ordered native history required: ${key}`);
    nativeHistories[key] = history.map(({ step, value }) => ({ step, value }));
  }
  const frames = nativeHistories['train/rl/frames'].at(-1).value;
  assert.equal(frames, 64, 'fixed G1 final physical frames required');
  const experimentName = manifest.modes[mode].logging?.experiment_name;
  assert.equal(experimentName, `physical-ai-logging-${mode}`, 'selected mode experiment label mismatch');
  const config = manifest.modes[mode].configs?.['case.yaml'];
  assert(config && typeof config.path === 'string' && path.basename(config.path) === 'case.yaml' &&
    typeof config.sha256 === 'string' && /^[a-f0-9]{64}$/.test(config.sha256), 'source case.yaml reference required');
  const sourceConfig = { path: config.path, sha256: config.sha256 };
  // Fixed G1 smoke: the budget equals final physical frames, not a generic budget derivation.
  const selected = [
    ['metric', 'train/rl/frames', frames, 'metric-train-frames'],
    ['metric', 'eval/rl/return_mean', nativeHistories['eval/rl/return_mean'].at(-1).value, 'metric-return'],
    ['param', 'scenario/training/total_frames', frames, 'param-frame-budget'],
    ['param', 'scenario/interaction/environment/params/env_name', 'CartPole-v1', 'param-environment'],
    ['param', 'scenario/logging/experiment_name', experimentName, 'param-experiment'],
  ];
  const initial = await checkOutput(output);
  assert.deepEqual(await fs.readdir(output), [], 'caller-created empty output required');
  async function save(filename, bytes) {
    const current = await checkOutput(output);
    assert(current.dev === initial.dev && current.ino === initial.ino, 'output directory changed');
    const file = await fs.open(path.join(output, filename), 'wx', 0o600);
    try { await file.writeFile(bytes); } finally { await file.close(); }
  }
  async function context() {
    assert.equal(page.url(), `${ORIGIN}/#/experiments/${main.experiment_id}/runs/${main.run_id}`, 'unexpected root Overview URL');
    assert.equal(await page.title(), `Run ${main.run_id} - MLflow`, 'wrong run title');
    const tab = page.getByRole('tab', { name: 'Overview', exact: true });
    assert.equal(await tab.count(), 1, 'one Overview tab required');
    assert.equal(await tab.getAttribute('aria-selected'), 'true', 'Overview tab must be selected');
  }
  async function screenshot(locator, filename) {
    const bytes = await locator.screenshot({ type: 'png', timeout: 30000 });
    assert(Buffer.isBuffer(bytes) && bytes.length > 0, 'native PNG bytes required');
    await save(filename, bytes);
    return { filename, bytes: bytes.length, sha256: sha256(bytes) };
  }
  await context();
  for (const text of [main.run_id, 'Finished']) {
    const value = page.getByText(text, { exact: true });
    assert.equal(await value.count(), 1, `one initial ${text} required`);
    await value.waitFor({ state: 'visible', timeout: 30000 });
  }
  const rows = [];
  for (const [kind, key, expected, stem] of selected) {
    await context();
    const row = page.getByRole('row').filter({ has: page.getByRole('cell', { name: key, exact: true }) });
    assert.equal(await row.count(), 1, `one native row required: ${key}`);
    await row.scrollIntoViewIfNeeded({ timeout: 30000 });
    await row.waitFor({ state: 'visible', timeout: 30000 });
    const cells = row.getByRole('cell');
    assert.equal(await cells.count(), 2, `two native cells required: ${key}`);
    const value = cells.nth(1);
    await value.waitFor({ state: 'visible', timeout: 30000 });
    const actual = (await value.innerText()).trim();
    assert.equal(actual, String(expected), `native value mismatch: ${key}`);
    const rowScreenshot = await screenshot(row, `${stem}-row.png`);
    const viewportScreenshot = await screenshot(page, `${stem}.png`);
    await context();
    rows.push({ kind, key, expected, actual, screenshot: rowScreenshot, viewport_screenshot: viewportScreenshot });
  }
  await context();
  const summary = { scope: SCOPE, mode, run_id: main.run_id, experiment_id: main.experiment_id,
    status: { expected: main.status, actual: 'Finished', observation: 'initial exact visible native text' },
    source_frames: frames, source_config: sourceConfig, native_histories: nativeHistories,
    axis_units: {
      'train/rl/frames': { step: 'native logger step (not physical frames)', value: 'physical environment frames' },
      'eval/rl/return_mean': { step: 'physical environment frames', value: 'mean episodic return' },
    }, rows,
    config_scope: 'three selected native rows only; source path/hash retained, YAML not read; not all 119 params validated',
  };
  await save('mlflow-overview.json', `${JSON.stringify(summary, null, 2)}\n`);
  return summary;
}

module.exports = { runMLflowOverview };
