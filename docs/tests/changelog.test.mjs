import { test } from 'node:test';
import assert from 'node:assert/strict';
import { changelogSchema, changelogRoute, orderedEntries } from '../src/changelog.mjs';

const entry = { date: '2026-10-22', title: 'Update', category: 'feature', summary: 'Useful changes' };
test('valid editorial and matched release metadata', () => {
  assert.equal(changelogSchema.parse(entry).version, undefined);
  assert.equal(changelogSchema.parse({ ...entry, category: 'release', version: 'v0.2.1', links: [{ label: 'Release', url: 'https://github.com/NexuFed/NexuML/releases/tag/v0.2.1' }] }).version, 'v0.2.1');
});
test('invalid metadata, links, media and releases fail closed', () => {
  for (const invalid of [
    { ...entry, date: '2026-02-30' }, { ...entry, title: '' }, { ...entry, category: 'other' },
    { ...entry, summary: undefined }, { ...entry, category: 'release' },
    { ...entry, category: 'release', version: 'v0.2.1', links: [{ label: 'Wrong release', url: 'https://github.com/NexuFed/NexuML/releases/tag/v0.1.0' }] },
    ...['javascript:alert(1)', '//evil.invalid', '/%2e%2e/escape'].map(url => ({ ...entry, links: [{ label: 'Unsafe', url }] })),
    { ...entry, media: [{ type: 'image', src: '/changelog/media/missing.webp', alt: 'Missing' }] },
    { ...entry, media: [{ type: 'image', src: 'https://example.com/a.webp', alt: '' }] },
    { ...entry, media: [{ type: 'video', src: 'https://example.com/a.webm', label: 'Demo' }] },
  ]) assert.equal(changelogSchema.safeParse(invalid).success, false, JSON.stringify(invalid));
});
test('stable routes, date matching, duplicates and deterministic order', () => {
  const entries = ['2026-10-22-zed', '2026-09-01-old', '2026-10-22-alpha'].map(id => ({ id, data: { ...entry, date: id.slice(0, 10) } }));
  assert.deepEqual(orderedEntries(entries).map(e => e.id), ['2026-10-22-alpha', '2026-10-22-zed', '2026-09-01-old']);
  assert.equal(changelogRoute(entries[0]), '2026-10-22/zed');
  assert.throws(() => orderedEntries([entries[0], entries[0]]), /Duplicate/);
  assert.throws(() => changelogRoute({ id: '2026-10-21-zed', data: entry }), /mismatch/);
});
