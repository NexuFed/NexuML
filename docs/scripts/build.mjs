import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { mkdirSync, writeFileSync, cpSync, rmSync, readdirSync, readFileSync } from 'node:fs';
import { site, mount, version, revision } from '../config.mjs';

const docs = fileURLToPath(new URL('../', import.meta.url));
const python = `${docs}python/.venv/bin/python`;
export function run(command, args, extra = {}) {
  const result = spawnSync(command, args, { cwd: docs, env: { ...process.env, ASTRO_TELEMETRY_DISABLED: '1', ...extra }, encoding: 'utf8' });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(result.stderr + result.stdout);
  return result.stdout + result.stderr;
}

function build(target) {
  mkdirSync(`${docs}.output/logs`, { recursive: true });
  if (target === 'versioned') {
    run(python, ['scripts/extract_cli.py', '--output', '.generated/cli.json']);
  }
  const log = run(process.execPath, ['node_modules/astro/bin/astro.mjs', 'build'], { DOCS_TARGET: target });
  writeFileSync(`${docs}.output/logs/${target}.log`, log);
  if (target === 'versioned') {
    cpSync(`${docs}.output/versioned/sitemap-0.xml`, `${docs}.output/versioned/sitemap.xml`);
  } else {
    const sitemap = `${docs}.output/changelog/sitemap-index.xml`;
    // Both frozen MkDocs and new Starlight snapshots expose this stable sitemap endpoint.
    writeFileSync(sitemap, readFileSync(sitemap, 'utf8').replace('</sitemapindex>', `<sitemap><loc>${site}${mount}latest/sitemap.xml</loc></sitemap></sitemapindex>`));
  }
  writeFileSync(`${docs}.output/${target}/build.json`, JSON.stringify({ schemaVersion: 1, target, version, revision, site, mount }, null, 2));
  console.log(`Built ${target} at ${target === 'versioned' ? `${mount}${version}/` : `${mount}changelog/`}`);
}

export function routes(directory, prefix = '') {
  return readdirSync(directory, { withFileTypes: true }).flatMap(entry => {
    const relative = `${prefix}${entry.name}`;
    return entry.isDirectory() ? routes(`${directory}/${entry.name}`, `${relative}/`)
      : entry.name === 'index.html' ? [prefix] : [];
  });
}

function previewTree() {
  const preview = `${docs}.output/preview`;
  rmSync(preview, { recursive: true, force: true });
  const root = `${preview}${mount}`;
  mkdirSync(root, { recursive: true });
  cpSync(`${docs}.output/changelog`, root, { recursive: true });
  cpSync(`${docs}.output/versioned`, `${root}${version}`, { recursive: true });
  writeFileSync(`${root}versions.json`, JSON.stringify([{ version, title: `${version} (local preview)`, aliases: [] }]));
  writeFileSync(`${root}publication.json`, JSON.stringify({ previewOnly: true, schemaVersion: 1,
    versions: { [version]: { revision, routes: routes(`${root}${version}`) } } }));
  writeFileSync(`${root}index.html`, `<!doctype html><meta charset="utf-8"><title>NexuML local preview</title><meta http-equiv="refresh" content="0;url=${mount}${version}/"><a href="${mount}${version}/">Open local documentation preview</a>`);
  // Global pages link to latest in production; local preview intentionally has no fabricated release.
  cpSync(`${root}${version}`, `${root}latest`, { recursive: true });
  console.log(`Local preview: ${mount}${version}/ and ${mount}changelog/ (not a publication archive)`);
}

const mode = process.argv[2] || 'all';
if (!['all', 'versioned', 'changelog'].includes(mode)) throw new Error(`Invalid build target: ${mode}`);
if (mode === 'all') { build('versioned'); build('changelog'); previewTree(); }
else build(mode);
