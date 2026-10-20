import assert from 'node:assert/strict';
import { writeFileSync, existsSync, unlinkSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const docs = fileURLToPath(new URL('../', import.meta.url));
const path = `${docs}src/content/docs/zz-invalid-proof.mdx`;
const collisions = ['zz', 'ZZ'].map(slug => `${docs}src/content/changelog/2026-10-22-${slug}-invalid-proof.mdx`);
for (const file of [path, ...collisions]) assert.ok(!existsSync(file), `Refusing to overwrite ${file}`);
try {
  for (const content of [
    '---\ndescription: Missing title\n---\n\nBroken frontmatter.\n',
    '---\ntitle: Syntax proof\ndescription: Invalid MDX\n---\n\n<Broken',
  ]) {
    writeFileSync(path, content);
    const result = spawnSync(process.execPath, ['node_modules/astro/bin/astro.mjs', 'build'], {
      cwd: docs, env: { ...process.env, DOCS_TARGET: 'versioned', ASTRO_TELEMETRY_DISABLED: '1' }, encoding: 'utf8',
    });
    assert.notEqual(result.status, 0, 'Invalid content unexpectedly built successfully');
    assert.match(result.stdout + result.stderr, /zz-invalid-proof/);
  }
  for (const file of collisions) writeFileSync(file, "---\ndate: '2026-10-22'\ntitle: Collision fixture\ncategory: improvement\nsummary: Must not disappear during ID normalization\n---\n\nExcluded test entry.\n");
  const collision = spawnSync(process.execPath, ['node_modules/astro/bin/astro.mjs', 'build'], {
    cwd: docs, env: { ...process.env, DOCS_TARGET: 'changelog', ASTRO_TELEMETRY_DISABLED: '1' }, encoding: 'utf8',
  });
  assert.notEqual(collision.status, 0, 'Colliding changelog filenames unexpectedly built successfully');
  assert.match(collision.stdout + collision.stderr, /ZZ-invalid-proof/);
  console.log('Invalid frontmatter, malformed MDX, and colliding changelog filenames are rejected by the actual build.');
} finally {
  for (const file of [path, ...collisions]) if (existsSync(file)) unlinkSync(file);
}
