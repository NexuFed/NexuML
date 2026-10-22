import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

const mount = `/${(process.env.DOCS_MOUNT ?? 'NexuML').replace(/^\/+|\/+$/g, '')}/`.replace('//', '/');
const version = process.env.DOCS_VERSION || 'dev';
const base = `${mount}${version}/`;
const fallbackVersion = version === 'v0.0.0' ? 'v0.0.1' : 'v0.0.0';

test('prose, API, CLI, diagrams, search, controls and accessible layout', async ({ page }, info) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('response', response => { if (response.status() >= 400) errors.push(`${response.status()} ${response.url()}`); });
  for (const route of ['start/first-run/', 'reference/api/nexuml/', 'reference/api/nexuml_library/layers/model/linear_encoder/', 'reference/cli/']) {
    await page.goto(`${base}${route}`);
    await page.waitForLoadState('networkidle');
    await expect(page.locator('h1')).toBeVisible();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBeTruthy();
    await page.screenshot({ path: info.outputPath(`${route.replaceAll('/', '-')}${info.project.name}.png`), fullPage: true });
    const accessibility = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze();
    await info.attach(`${route.replaceAll('/', '-')}-axe.json`, { body: JSON.stringify(accessibility.violations), contentType: 'application/json' });
    expect(accessibility.violations.map(v => ({ id: v.id, targets: v.nodes.map(n => n.target) }))).toEqual([]);
  }
  await expect(page.locator('#nexuml-train')).toBeVisible();
  await expect(page.locator('section[aria-labelledby="nexuml-train"]').getByText('--override, -O', { exact: true })).toBeVisible();
  await page.goto(`${base}explanation/architecture/`);
  await expect(page.locator('.mermaid svg').first()).toBeVisible();
  await page.goto(`${base}zz-generated-pipeline/`);
  await expect(page.locator('.mermaid svg').first()).toBeVisible();
  await page.getByRole('button', { name: 'Search', exact: true }).click();
  const dialog = page.getByRole('dialog', { name: 'Search', exact: true });
  await dialog.getByRole('textbox', { name: 'Search', exact: true }).fill('ScenarioSpec');
  await expect(dialog.locator('a[href*="/reference/api/"]').first()).toBeVisible();
  await page.keyboard.press('Escape');
  await page.goto(`${base}how-to/export/`);
  const tabs = page.getByRole('tab');
  expect(await tabs.count()).toBeGreaterThan(0);
  await tabs.last().click();
  await expect(tabs.last()).toHaveAttribute('aria-selected', 'true');
  expect(await page.locator('.expressive-code button').count()).toBeGreaterThan(0);
  await page.context().grantPermissions(['clipboard-read', 'clipboard-write']);
  const code = page.locator('.expressive-code').filter({ visible: true }).first();
  const copied = (await code.locator('code').innerText()).trim().replace(/\n{2,}/g, '\n');
  await code.locator('button').last().click();
  await expect.poll(() => page.evaluate(async () => (await navigator.clipboard.readText()).trim().replace(/\n{2,}/g, '\n'))).toBe(copied);
  if (info.project.name === 'mobile') await page.getByRole('button', { name: 'Menu' }).click();
  const themes = page.getByRole('combobox', { name: 'Select theme' });
  await themes.last().selectOption('light');
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
  const light = await new AxeBuilder({ page }).withTags(['wcag2aa']).analyze();
  expect(light.violations).toEqual([]);
  await themes.last().selectOption('auto');
  await page.goto(`${base}start/first-run/`);
  expect(await page.evaluate(() => localStorage.getItem('starlight-theme'))).toBe('');
  await page.keyboard.press('Tab');
  await expect(page.getByRole('link', { name: 'Skip to content' })).toBeFocused();
  expect(errors).toEqual([]);
});

test('global ordering, rich media and version fallback', async ({ page }, info) => {
  await page.goto(`${mount}changelog/`);
  const titles = await page.locator('article h3').allTextContents();
  expect(titles.slice(0, 3)).toEqual(['Newest acceptance fixture', 'Alphabetical acceptance fixture', 'Media acceptance fixture']);
  await page.getByRole('link', { name: 'Media acceptance fixture' }).click();
  const image = page.getByAltText('NexuFed wordmark media fixture');
  await expect(image).toBeVisible();
  expect(await image.evaluate((el: HTMLImageElement) => el.naturalWidth)).toBeGreaterThan(0);
  const video = page.getByLabel('Video acceptance fixture');
  await expect(video).toHaveAttribute('controls', '');
  await expect.poll(() => video.evaluate((el: HTMLVideoElement) => el.readyState)).toBeGreaterThanOrEqual(1);
  await expect(page.getByText('This is a generated media playback test, not a product announcement.')).toBeVisible();
  if (info.project.name === 'mobile') await page.getByRole('button', { name: 'Menu' }).click();
  await expect(page.getByRole('combobox', { name: 'Documentation version' })).toBeVisible();
  await expect(page.getByRole('link', { name: `Open ${version}`, exact: true })).toBeVisible();
  await page.screenshot({ path: info.outputPath(`changelog-${info.project.name}.png`), fullPage: true });
  const accessibility = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze();
  expect(accessibility.violations).toEqual([]);
  if (info.project.name === 'mobile') await page.getByRole('button', { name: 'Menu' }).click();
  await page.getByRole('button', { name: 'Search', exact: true }).click();
  const dialog = page.getByRole('dialog', { name: 'Search', exact: true });
  await dialog.getByRole('textbox', { name: 'Search', exact: true }).fill('Typed configuration');
  await expect(dialog.locator('a[href*="/changelog/2026-09-03/"]').first()).toBeVisible();
  expect(await dialog.locator('a[href*="/reference/api/"]').count()).toBe(0);
  await page.keyboard.press('Escape');
  await page.goto(`${base}start/first-run/`);
  if (info.project.name === 'mobile') await page.getByRole('button', { name: 'Menu' }).click();
  await page.getByRole('combobox', { name: 'Documentation version' }).selectOption(fallbackVersion);
  await expect(page.getByRole('status')).toContainText('unavailable');
  const fallback = page.getByRole('link', { name: `Open ${fallbackVersion} home` });
  await expect(fallback).toHaveAttribute('href', `${mount}${fallbackVersion}/`);
  await fallback.click();
  await expect(page.locator('h1')).toHaveText('Version fallback acceptance fixture');
});

test('light-mode API and CLI remain readable', async ({ page }, info) => {
  await page.addInitScript(() => localStorage.setItem('starlight-theme', 'light'));
  for (const route of ['reference/api/nexuml/', 'reference/api/nexuml_library/layers/model/linear_encoder/', 'reference/cli/']) {
    await page.goto(`${base}${route}`);
    await page.waitForLoadState('networkidle');
    await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBeTruthy();
    const accessibility = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze();
    await info.attach(`light-${route.replaceAll('/', '-')}-axe.json`, { body: JSON.stringify(accessibility.violations), contentType: 'application/json' });
    expect(accessibility.violations.map(v => ({ id: v.id, targets: v.nodes.map(n => n.target) }))).toEqual([]);
    await page.screenshot({ path: info.outputPath(`light-${route.replaceAll('/', '-')}${info.project.name}.png`), fullPage: true });
  }
});
