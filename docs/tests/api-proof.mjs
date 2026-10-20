import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { readFileSync, existsSync, writeFileSync, unlinkSync, mkdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const docs = fileURLToPath(new URL('../', import.meta.url));
const root = fileURLToPath(new URL('../../', import.meta.url));
const report = [];
const version = process.env.DOCS_VERSION || 'dev';
const cli = spawnSync(`${docs}python/.venv/bin/python`, ['scripts/extract_cli.py', '--output', '.generated/cli.json'], { cwd: docs, encoding: 'utf8' });
assert.equal(cli.status, 0, cli.stdout + cli.stderr);
const probes = ['src/nexuml', 'library/src/nexuml_library'].flatMap(dir => [
  [`${root}${dir}/docs_api_probe.py`, 'def public_probe(value: int = 7) -> int:\n    """A public discovery check."""\n    return value\n'],
  [`${root}${dir}/_docs_private_probe.py`, 'def hidden_probe():\n    pass\n'],
]);
probes.push([`${docs}src/content/docs/zz-api-proof.mdx`, '---\ntitle: API test fixture\ndescription: Temporary API acceptance fixture\n---\n\nimport { Autodoc } from "starlight-pydocs/components";\n\n<Autodoc name="nexuml.core.types.ScenarioSpec" />\n']);
for (const [path] of probes) assert.ok(!existsSync(path), `Refusing to overwrite ${path}`);
try {
  for (const [path, text] of probes) writeFileSync(path, text);
  for (const mount of ['NexuML', '']) {
    const env = { ...process.env, DOCS_MOUNT: mount, DOCS_TARGET: 'versioned', ASTRO_TELEMETRY_DISABLED: '1' };
    const result = spawnSync(process.execPath, ['node_modules/astro/bin/astro.mjs', 'build'], { cwd: docs, env, encoding: 'utf8' });
    assert.equal(result.status, 0, result.stderr + result.stdout);
    const output = `${docs}.output/versioned/`;
    const home = readFileSync(`${output}zz-api-proof/index.html`, 'utf8');
    const types = readFileSync(`${output}reference/api/nexuml/core/types/index.html`, 'utf8');
    assert.ok(home.includes('ScenarioSpec'), 'Autodoc fixture missing');
    assert.ok(types.includes('id="nexuml.core.types.ScenarioSpec.training"'));
    assert.ok(types.includes('id="nexuml.core.types.TrainingSpec.max_epochs"'));
    assert.ok(types.includes('github.com/NexuFed/NexuML/blob/'));
    assert.ok(types.includes(`${mount ? `/${mount}` : ''}/${version}/reference/api/nexuml/`));
    for (const pkg of ['nexuml', 'nexuml_library']) {
      const prefix = `${output}reference/api/${pkg}`;
      const probe = readFileSync(`${prefix}/docs_api_probe/index.html`, 'utf8');
      assert.ok(probe.includes(`id="${pkg}.docs_api_probe.public_probe"`));
      assert.ok(!existsSync(`${prefix}/_docs_private_probe/index.html`));
      const symbols = readFileSync(`${prefix}/symbols.json`, 'utf8');
      assert.ok(symbols.includes(`${pkg}.docs_api_probe.public_probe`));
      assert.ok(!symbols.includes('hidden_probe'));
    }
    const exported = readFileSync(`${output}reference/api/nexuml/index.html`, 'utf8');
    assert.ok(exported.includes('id="nexuml.nn_module"'), 'Public re-export missing');
    const linear = readFileSync(`${output}reference/api/nexuml_library/layers/model/linear_encoder/index.html`, 'utf8');
    assert.ok(linear.includes('LinearEncoder.build'));
    const global = spawnSync(process.execPath, ['node_modules/astro/bin/astro.mjs', 'build'], {
      cwd: docs, env: { ...env, DOCS_TARGET: 'changelog' }, encoding: 'utf8',
    });
    assert.equal(global.status, 0, global.stderr + global.stdout);
    assert.ok(!existsSync(`${docs}.output/changelog/reference`), 'Global build leaked API routes');
    assert.ok(existsSync(`${docs}.output/changelog/changelog/index.html`));
    report.push({ mount: mount || '/', apiPackages: 2, autodoc: true, automaticModules: true, privateExcluded: true, globalIsolated: true });
  }
  mkdirSync(`${docs}.output`, { recursive: true });
  writeFileSync(`${docs}.output/api-proof.json`, JSON.stringify(report, null, 2) + '\n');
  console.log('API proof passed for project and root mounts.');
} finally {
  for (const [path] of probes) unlinkSync(path);
}
