import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { writeFileSync, existsSync, mkdirSync, copyFileSync, unlinkSync, readFileSync, rmdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const docs = fileURLToPath(new URL('../', import.meta.url));
const entries = [
  ['2026-10-22-zz-new-proof', 'Newest acceptance fixture', ''],
  ['2026-10-21-aa-order-proof', 'Alphabetical acceptance fixture', ''],
  ['2026-10-21-zz-media-proof', 'Media acceptance fixture', `media:\n  - type: image\n    src: /changelog/media/zz-proof.webp\n    alt: NexuFed wordmark media fixture\n  - type: video\n    src: /changelog/media/zz-proof.webm\n    label: Video acceptance fixture\n    transcript: This is a generated media playback test, not a product announcement.\n`],
];
const files = entries.map(([id]) => `${docs}src/content/changelog/${id}.mdx`);
const media = `${docs}public/global/changelog/media`;
const createdDirectories = [media, `${docs}public/global/changelog`].filter(path => !existsSync(path));
files.push(`${media}/zz-proof.webp`, `${media}/zz-proof.webm`);
files.push(`${docs}src/content/docs/zz-generated-pipeline.mdx`);
for (const path of files) assert.ok(!existsSync(path), `Refusing to overwrite fixture path ${path}`);
function run(args) {
  const result = spawnSync(process.execPath, args, { cwd: docs, stdio: 'inherit', env: process.env });
  if (result.error) throw result.error;
  return result.status ?? 1;
}
let status = 1;
try {
  mkdirSync(media, { recursive: true });
  copyFileSync(`${docs}src/assets/brand/nexufed-wordmark.webp`, `${media}/zz-proof.webp`);
  copyFileSync(`${docs}tests/fixtures/media-proof.webm`, `${media}/zz-proof.webm`);
  writeFileSync(`${docs}src/content/docs/zz-generated-pipeline.mdx`, `---\ntitle: Generated pipeline acceptance fixture\ndescription: Actual Mermaid output from the synthetic recipe compiler\n---\n\n${readFileSync(`${docs}tests/fixtures/generated-pipeline.md`, 'utf8')}`);
  entries.forEach(([id, title, extra]) => writeFileSync(`${docs}src/content/changelog/${id}.mdx`, `---\ndate: '${id.slice(0, 10)}'\ntitle: ${title}\ncategory: improvement\nsummary: Excluded acceptance fixture\n${extra}---\n\nThis entry exists only during browser acceptance checks.\n`));
  status = run(['scripts/build.mjs', 'all']);
  if (status === 0) {
    const mount = `/${(process.env.DOCS_MOUNT ?? 'NexuML').replace(/^\/+|\/+$/g, '')}/`.replace('//', '/');
    const preview = `${docs}.output/preview${mount}`;
    const versions = JSON.parse(readFileSync(`${preview}versions.json`, 'utf8'));
    const fallbackVersion = process.env.DOCS_VERSION === 'v0.0.0' ? 'v0.0.1' : 'v0.0.0';
    versions.push({ version: fallbackVersion, title: `${fallbackVersion} (fallback test fixture)`, aliases: [] });
    const otherBranch = process.env.DOCS_VERSION === 'main' ? 'dev' : 'main';
    versions.push({ version: otherBranch, title: `${otherBranch} (unreleased)`, aliases: [] });
    writeFileSync(`${preview}versions.json`, JSON.stringify(versions));
    const publication = JSON.parse(readFileSync(`${preview}publication.json`, 'utf8'));
    publication.versions[fallbackVersion] = { revision: null, routes: [''] };
    publication.versions[otherBranch] = { revision: null, routes: [''] };
    writeFileSync(`${preview}publication.json`, JSON.stringify(publication));
    mkdirSync(`${preview}${fallbackVersion}`, { recursive: true });
    writeFileSync(`${preview}${fallbackVersion}/index.html`, '<!doctype html><title>Excluded fallback fixture</title><h1>Version fallback acceptance fixture</h1>');
    mkdirSync(`${preview}${otherBranch}`, { recursive: true });
    writeFileSync(`${preview}${otherBranch}/index.html`, '<!doctype html><title>Excluded branch fixture</title><h1>Other branch acceptance fixture</h1>');
  }
  if (status === 0) status = run(['node_modules/@playwright/test/cli.js', 'test']);
} finally {
  for (const path of files) if (existsSync(path)) unlinkSync(path);
  for (const directory of createdDirectories) {
    try { rmdirSync(directory); } catch (error) { if (!['ENOENT', 'ENOTEMPTY'].includes(error.code)) throw error; }
  }
  const cleanup = run(['scripts/build.mjs', 'all']);
  if (cleanup !== 0) status = cleanup;
}
process.exit(status);
