import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const project = fileURLToPath(new URL('../python/', import.meta.url));
const env = { ...process.env, UV_PROJECT_ENVIRONMENT: `${project}.venv` };
delete env.VIRTUAL_ENV;
const result = spawnSync('uv', ['sync', '--project', project, '--frozen'], { env, stdio: 'inherit' });
if (result.error) throw result.error;
if (result.status !== 0) process.exit(result.status ?? 1);
const install = spawnSync('uv', ['pip', 'install', '--python', `${project}.venv/bin/python`,
  '--no-deps', '--no-build-isolation', '--editable', fileURLToPath(new URL('../../', import.meta.url))],
  { env, stdio: 'inherit' });
if (install.error) throw install.error;
process.exit(install.status ?? 1);
