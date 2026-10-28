'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const crypto = require('node:crypto');

const EXPERIMENT = 'physical-ai-logging-partial-delivery';
const SCOPE = 'native FAILED status and retained-file visibility only; CLI/media delivery proofs parent-owned; whole6.2/V remain UNVERIFIED';
const sha256 = (bytes) => crypto.createHash('sha256').update(bytes).digest('hex');
const canonical = (value) => typeof value === 'string' && !value.includes('\0') && path.isAbsolute(value) && path.resolve(value) === value;
const hashValid = (value) => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value);
function keys(value, expected) {
  assert(value && typeof value === 'object' && !Array.isArray(value), 'object required');
  assert.deepEqual(Object.keys(value).sort(), expected.split(' ').sort(), 'unexpected manifest fields');
}

async function runMLflowFailure(page, manifest, output) {
  keys(manifest, 'run_id experiment_id status experiment_name cli tensorboard_gif prior_artifacts');
  assert(typeof manifest.run_id === 'string' && /^[a-f0-9]{32}$/.test(manifest.run_id), 'invalid run_id');
  assert(typeof manifest.experiment_id === 'string' && /^\d+$/.test(manifest.experiment_id), 'invalid experiment_id');
  assert.equal(manifest.status, 'FAILED');
  assert.equal(manifest.experiment_name, EXPERIMENT);
  keys(manifest.cli, 'exit_code status log_path log_sha256');
  assert(Number.isFinite(manifest.cli.exit_code) && manifest.cli.exit_code === 1, 'original CLI exit 1 required');
  assert.equal(manifest.cli.status, 'FAIL');
  assert(canonical(manifest.cli.log_path) && hashValid(manifest.cli.log_sha256), 'invalid CLI log reference');
  keys(manifest.tensorboard_gif, 'path sha256');
  assert(canonical(manifest.tensorboard_gif.path) && manifest.tensorboard_gif.path.endsWith('/artifacts/evaluation/frame_000000000064.gif') &&
    hashValid(manifest.tensorboard_gif.sha256), 'invalid TensorBoard GIF reference');
  assert(Array.isArray(manifest.prior_artifacts) && manifest.prior_artifacts.length === 2, 'two prior artifacts required');
  for (const [index, filename] of ['source_package.json', 'config.yaml'].entries()) {
    const file = manifest.prior_artifacts[index];
    keys(file, 'filename sha256');
    assert.equal(file.filename, filename);
    assert(hashValid(file.sha256), 'invalid prior artifact hash');
  }
  assert(page && ['url', 'title', 'getByRole', 'getByText', 'getByLabel', 'locator', 'screenshot'].every(
    (key) => typeof page[key] === 'function'), 'public native page API required');
  assert(canonical(output), 'canonical absolute output required');
  assert.equal(process.platform, 'linux', 'Linux /proc directory anchoring required');
  const { run_id, experiment_id } = manifest;
  const cli = { ...manifest.cli }, tensorboard_gif = { ...manifest.tensorboard_gif };
  const prior_artifacts = manifest.prior_artifacts.map((file) => ({ ...file }));
  const root = `http://127.0.0.1:5001/#/experiments/${experiment_id}/runs/${run_id}`;
  async function checkOutput() {
    let directory = output, leaf;
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
  const initial = await checkOutput();
  const directory = await fs.open(output, fs.constants.O_RDONLY | fs.constants.O_DIRECTORY | fs.constants.O_NOFOLLOW);
  let primary;
  try {
    const anchored = await directory.stat(), admitted = await checkOutput();
    assert(anchored.isDirectory() && anchored.uid === process.getuid() && (anchored.mode & 0o7777) === 0o700 &&
      anchored.dev === initial.dev && anchored.ino === initial.ino &&
      anchored.dev === admitted.dev && anchored.ino === admitted.ino, 'output directory anchor changed');
    const anchor = `/proc/self/fd/${directory.fd}`;
    assert.deepEqual(await fs.readdir(anchor), [], 'caller-created empty output required');
    async function save(filename, bytes) {
      assert(['failed-run.png', 'artifacts.png', 'mlflow-failure.json'].includes(filename), 'fixed output filename required');
      const current = await checkOutput();
      assert(current.dev === initial.dev && current.ino === initial.ino, 'output directory changed');
      const file = await fs.open(`${anchor}/${filename}`, 'wx', 0o600);
      try { await file.writeFile(bytes); }
      catch (error) { await file.close().catch(() => {}); throw error; }
      await file.close();
    }
    async function visible(locator) {
      await locator.waitFor({ state: 'visible', timeout: 30000 });
      assert.equal(await locator.count(), 1, 'one exact native element required');
    }
    async function context(tab) {
      assert.equal(page.url(), root + (tab === 'Artifacts' ? '/artifacts' : ''), 'unexpected native run URL');
      assert.equal(await page.title(), `Run ${run_id} - MLflow`, 'wrong run title');
      const breadcrumb = page.locator('[data-testid="experiment-runs-link"]');
      await visible(breadcrumb);
      assert.equal(await breadcrumb.getAttribute('href'), `#/experiments/${experiment_id}/evaluation-runs`);
      assert.equal(await breadcrumb.innerText(), EXPERIMENT);
      const selected = page.getByRole('tab', { name: tab, exact: true });
      await visible(selected);
      assert.equal(await selected.getAttribute('aria-selected'), 'true', `${tab} tab must be selected`);
    }
    async function status() {
      // Supplied native HTML selectors; unknown/missing native fields must fail closed.
      const idField = page.locator('[data-testid="key-value-Run ID"]');
      await visible(idField);
      await visible(idField.getByText(run_id, { exact: true }));
      const statusField = page.locator('[data-testid="key-value-Status"]');
      await visible(statusField);
      const value = statusField.getByRole('status');
      await visible(value);
      assert.equal((await value.innerText()).trim(), 'Failed', 'native Status field must be Failed');
    }
    async function files() {
      for (const filename of ['source_package.json', 'config.yaml', 'evaluation']) {
        const file = page.getByLabel(filename, { exact: true });
        await visible(file);
        assert.equal(await file.getAttribute('data-testid'), 'artifact-tree-node');
        assert.equal(await file.getAttribute('artifact-name'), filename);
      }
      assert.equal(await page.locator('[data-testid="artifact-tree-node"][artifact-name$=".gif" i], [data-testid="artifact-tree-node"][artifact-name$=".mp4" i], [data-testid="artifact-tree-node"][artifact-name="frame_000000000064.json"]').count(), 0, 'unexpected rollout media tree node');
    }
    async function capture(filename) {
      const bytes = await page.screenshot({ type: 'png', timeout: 30000 });
      assert(Buffer.isBuffer(bytes) && bytes.length > 0, 'native PNG bytes required');
      await save(filename, bytes);
      return { filename, bytes: bytes.length, sha256: sha256(bytes) };
    }
    await context('Overview');
    await status();
    const failed_run = await capture('failed-run.png');
    await context('Overview');
    await status();
    await page.getByRole('tab', { name: 'Artifacts', exact: true }).click({ timeout: 30000 });
    await page.getByRole('tab', { name: 'Artifacts', exact: true, selected: true }).waitFor({ state: 'visible', timeout: 30000 });
    await context('Artifacts');
    await files();
    const artifacts = await capture('artifacts.png');
    await context('Artifacts');
    await files();
    const summary = { scope: SCOPE, run_id, experiment_id, status: { expected: 'FAILED', actual: 'Failed' },
      cli, tensorboard_gif, prior_artifacts, native: { url: page.url(), title: await page.title(),
        breadcrumb: { href: `#/experiments/${experiment_id}/evaluation-runs`, text: EXPERIMENT },
        filenames: ['source_package.json', 'config.yaml', 'evaluation'],
        evaluation: 'parent-described regular-file blocker; native filename visibility only; not opened, not rollout media' }, screenshots: { failed_run, artifacts } };
    await save('mlflow-failure.json', `${JSON.stringify(summary, null, 2)}\n`);
    return summary;
  } catch (error) { primary = error; throw error; }
  finally {
    try { await directory.close(); }
    catch (error) { if (!primary) throw error; }
  }
}

module.exports = { runMLflowFailure };
