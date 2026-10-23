import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightPydocs, { pydocsSidebarGroup } from 'starlight-pydocs';
import { unified } from '@astrojs/markdown-remark';
import mermaid from 'astro-mermaid';
import { fileURLToPath } from 'node:url';
import { site, base, target, revision } from './config.mjs';

export default defineConfig({
  site,
  base,
  trailingSlash: 'always',
  cacheDir: `.astro/${target}`,
  outDir: `.output/${target}`,
  publicDir: target === 'versioned' ? './public/versioned' : './public/global',
  markdown: { shikiConfig: { themes: { light: 'github-light-high-contrast', dark: 'github-dark' } }, processor: unified({ remarkPlugins: [() => (tree) => {
    const scope = (node) => {
      if (['link', 'image', 'definition'].includes(node.type) && /^\/(?!\/)/.test(node.url)) {
        node.url = `${base}${node.url.slice(1)}`;
      }
      node.children?.forEach(scope);
    };
    scope(tree);
  }] }) },
  integrations: [
    mermaid({ autoTheme: true, enableLog: false, mermaidConfig: { securityLevel: 'strict' } }),
    ...(target === 'changelog' ? [{
      name: 'nexuml-changelog-routes',
      hooks: { 'astro:config:setup': ({ injectRoute }) => {
        injectRoute({ pattern: '/changelog/', entrypoint: './src/routes/changelog/index.astro' });
        injectRoute({ pattern: '/changelog/[...entry]', entrypoint: './src/routes/changelog/[...entry].astro' });
      } },
    }] : []),
    starlight({
      title: 'NexuML',
      favicon: '/favicon.png',
      logo: { src: './src/assets/brand/nexufed-wordmark.webp', alt: 'NexuFed AI' },
      customCss: ['./src/styles/brand.css'],
      expressiveCode: { styleOverrides: { frames: {
        tooltipSuccessBackground: '#007852', tooltipSuccessForeground: '#ffffff',
      } } },
      components: {
        Sidebar: './src/components/Sidebar.astro',
        ThemeProvider: './src/components/ThemeProvider.astro',
      },
      description: 'Compose reusable machine learning components into reproducible experiments.',
      social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/NexuFed/NexuML' }],
      editLink: { baseUrl: `https://github.com/NexuFed/NexuML/edit/${revision}/docs/` },
      sidebar: target === 'versioned' ? [
        { label: 'Start here', items: [
          { label: 'Overview', slug: '' }, { slug: 'start' }, { slug: 'start/install' },
          { slug: 'start/first-run' }, { slug: 'studio' }, { slug: 'tutorials' },
          { label: 'Optional CIFAR example', slug: 'start/cifar-resnet' },
        ] },
        { label: 'Use NexuML', collapsed: true, items: [
          { label: 'Overview', slug: 'how-to' },
          { label: 'Run experiments', items: [
            { slug: 'how-to/run-scenarios' }, { slug: 'how-to/data-loading' },
            { slug: 'how-to/tracking' }, { slug: 'how-to/checkpoints' },
            { slug: 'how-to/evaluate' }, { slug: 'how-to/tune' },
          ] },
          { label: 'Export and reuse', collapsed: true, items: [
            { slug: 'how-to/export-dataset' }, { slug: 'how-to/export' },
          ] },
          { label: 'Extend NexuML', collapsed: true, items: [
            { slug: 'how-to/custom-library' }, { slug: 'how-to/custom-layer' },
            { slug: 'how-to/custom-data-source' }, { slug: 'how-to/custom-eval-algorithm' },
            { slug: 'how-to/scenario-file' },
          ] },
          { label: 'Execution and scaling', collapsed: true, items: [
            { slug: 'how-to/training-backends' }, { slug: 'how-to/auto-batch-size' },
            { slug: 'how-to/training-backends/jobs' }, { slug: 'how-to/training-backends/ray' },
            { slug: 'how-to/deploy' },
          ] },
          { label: 'Studio hosting and API', collapsed: true, items: [
            { slug: 'how-to/local-api' }, { slug: 'how-to/studio-hosting' },
          ] },
          { label: 'Automate research', collapsed: true, items: [{ slug: 'how-to/autoresearch' }] },
        ] },
        { label: 'Understand NexuML', collapsed: true, items: [
          { slug: 'explanation/why-nexuml' }, { slug: 'explanation/architecture' },
          { slug: 'explanation/diagrams' },
        ] },
        { label: 'Reference', collapsed: true, items: [
          { slug: 'reference' }, { slug: 'reference/cli' }, { slug: 'reference/scenario-spec' },
          { slug: 'reference/registry' }, { slug: 'reference/backends' },
          { slug: 'reference/environment' }, { slug: 'reference/tuning-file' }, pydocsSidebarGroup,
        ] },
      ] : [],
      plugins: target === 'versioned' ? [starlightPydocs({
        runner: { command: [
          fileURLToPath(new URL('./python/.venv/bin/python', import.meta.url)),
          fileURLToPath(new URL('./python/griffe_static.py', import.meta.url)),
        ] },
        packages: [
          { name: 'nexuml', base: 'reference/api/nexuml', search: ['../src'] },
          { name: 'nexuml_library', base: 'reference/api/nexuml_library', search: ['../library/src'] },
        ].map(pkg => ({
          ...pkg,
          docstringStyle: 'google',
          extensions: ['griffe_pydantic'],
          members: { exclude: [`${pkg.name}._*`, `${pkg.name}.**._*`] },
          sourceLink: { host: 'github', repo: 'NexuFed/NexuML', ref: revision, root: '..' },
          sidebar: { label: pkg.name, collapsed: true },
        })),
      })] : [],
    }),
  ],
});
