import { defineCollection } from 'astro:content';
import { docsLoader } from '@astrojs/starlight/loaders';
import { docsSchema } from '@astrojs/starlight/schema';
import { glob } from 'astro/loaders';
import { changelogSchema } from './changelog.mjs';
import { z } from 'astro/zod';

export const collections = {
  docs: defineCollection({
    loader: process.env.DOCS_TARGET === 'changelog'
      ? { name: 'empty-versioned-docs', load: async ({ store }) => { store.clear(); } }
      : docsLoader(),
    schema: docsSchema({ extend: z.object({ title: z.string().trim().min(1), description: z.string().trim().min(1).optional() }) }),
  }),
  changelog: defineCollection({
    loader: process.env.DOCS_TARGET === 'changelog'
      ? glob({
        pattern: '*.mdx', base: './src/content/changelog',
        // Keep raw filenames so route validation sees entries the default slugger would merge.
        generateId: ({ entry }) => entry.replace(/\.mdx$/, ''),
      })
      : { name: 'empty-global-changelog', load: async ({ store }) => { store.clear(); } },
    schema: changelogSchema,
  }),
};
