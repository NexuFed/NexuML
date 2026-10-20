import { z } from 'astro/zod';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const local = (s) => {
  try {
    return s.startsWith('/') && !s.startsWith('//') && !s.includes('\\')
      && !decodeURIComponent(s).split(/[/?#]/).includes('..');
  } catch { return false; }
};
const safeUrl = z.string().refine(value => {
  try { return local(value) || new URL(value).protocol === 'https:'; } catch { return false; }
}, 'Use an HTTPS URL or a site-local path');
const nonempty = z.string().trim().min(1);
export const changelogSchema = z.object({
  date: z.string().regex(/^\d{4}-\d{2}-\d{2}$/).refine(value => {
    const date = new Date(`${value}T00:00:00Z`);
    return !Number.isNaN(+date) && date.toISOString().slice(0, 10) === value;
  }, 'Invalid calendar date'),
  title: nonempty,
  category: z.enum(['release', 'feature', 'improvement', 'fix']),
  summary: nonempty,
  version: z.string().regex(/^v\d+\.\d+\.\d+$/).optional(),
  links: z.array(z.object({ label: nonempty, url: safeUrl }).strict()).default([]),
  media: z.array(z.discriminatedUnion('type', [
    z.object({ type: z.literal('image'), src: safeUrl, alt: nonempty }).strict(),
    z.object({ type: z.literal('video'), src: safeUrl, label: nonempty,
      transcript: nonempty.optional(),
      captions: z.object({ src: safeUrl, lang: nonempty, label: nonempty }).strict().optional(),
    }).strict(),
  ])).default([]),
}).strict().superRefine((entry, ctx) => {
  if (entry.category === 'release' && (!entry.version || !entry.links.some(({ url }) =>
    ['releases/tag', 'tree'].some(prefix => url === `https://github.com/NexuFed/NexuML/${prefix}/${entry.version}`)))) {
    ctx.addIssue({ code: 'custom', path: ['version'], message: 'Release entries require a version and matching GitHub release/tag link' });
  }
  entry.media.forEach((media, index) => {
    if (media.type === 'video' && !media.transcript && !media.captions) {
      ctx.addIssue({ code: 'custom', path: ['media', index], message: 'Video requires captions or a text alternative' });
    }
    for (const src of [media.src, media.type === 'video' ? media.captions?.src : undefined].filter(Boolean)) {
      if (local(src) && !existsSync(fileURLToPath(new URL(`../public/global${src.split(/[?#]/)[0]}`, import.meta.url)))) {
        ctx.addIssue({ code: 'custom', path: ['media', index], message: `Missing local asset: ${src}` });
      }
    }
  });
});

export function changelogRoute(entry) {
  const match = /^(\d{4}-\d{2}-\d{2})-([a-z0-9]+(?:-[a-z0-9]+)*)$/.exec(entry.id);
  if (!match || match[1] !== entry.data.date) throw new Error(`Changelog filename/date mismatch: ${entry.id}`);
  return `${match[1]}/${match[2]}`;
}

export function orderedEntries(entries) {
  const routes = new Set();
  for (const entry of entries) {
    const route = changelogRoute(entry);
    if (routes.has(route)) throw new Error(`Duplicate changelog route: ${route}`);
    routes.add(route);
  }
  return [...entries].sort((a, b) => b.data.date.localeCompare(a.data.date) || a.id.localeCompare(b.id));
}
