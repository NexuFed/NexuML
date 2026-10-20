import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { site } from '../config.mjs';

const docs = fileURLToPath(new URL('../', import.meta.url));
const root = fileURLToPath(new URL('../../', import.meta.url));
function run(command, args, cwd = docs) {
  const result = spawnSync(command, args, { cwd, env: { ...process.env, ASTRO_TELEMETRY_DISABLED: '1' }, stdio: 'inherit' });
  if (result.error) throw result.error;
  if (result.status !== 0) process.exit(result.status ?? 1);
}
run(process.execPath, ['--test', 'tests/changelog.test.mjs']);
run(`${docs}python/.venv/bin/python`, ['scripts/check_content.py']);
run(`${docs}python/.venv/bin/python`, ['-m', 'pytest', '--noconftest', 'docs/tests', 'tests/test_cli_lightweight.py', '-q'], root);
run(`${docs}python/.venv/bin/ruff`, ['check', 'docs/scripts', 'docs/python/griffe_static.py', 'docs/tests'], root);
run(`${docs}python/.venv/bin/ty`, ['check', '--python', 'python/.venv/bin/python', 'scripts', 'python/griffe_static.py', 'tests']);
run(`${docs}python/.venv/bin/python`, ['scripts/extract_cli.py', '--output', '.generated/cli.json']);
run(process.execPath, ['node_modules/astro/bin/astro.mjs', 'check']);
run(process.execPath, ['scripts/build.mjs', 'all']);
run(`${docs}python/.venv/bin/python`, ['scripts/check_links.py', '.output/preview', '--site', site]);
