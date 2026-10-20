import { execFileSync } from 'node:child_process';

export const target = process.env.DOCS_TARGET || 'versioned';
if (!['versioned', 'changelog'].includes(target)) throw new Error(`Invalid DOCS_TARGET: ${target}`);
export const site = process.env.DOCS_SITE || 'https://nexufed.github.io';
if (!['https:', 'http:'].includes(new URL(site).protocol) || new URL(site).origin !== site) throw new Error('DOCS_SITE must be an HTTP(S) origin');
export const mount = `/${(process.env.DOCS_MOUNT ?? 'NexuML').replace(/^\/+|\/+$/g, '')}/`.replace('//', '/');
if (!/^\/(?:[a-zA-Z0-9_-]+\/)*$/.test(mount)) throw new Error('DOCS_MOUNT must be a plain deployment path');
export const version = process.env.DOCS_VERSION || 'dev';
if (!/^(main|dev|v\d+\.\d+\.\d+)$/.test(version)) throw new Error(`Invalid DOCS_VERSION: ${version}`);
export const revision = process.env.DOCS_REVISION || execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim();
if (!/^[a-f0-9]{40}$/.test(revision)) throw new Error('DOCS_REVISION must be a full commit SHA');
export const base = target === 'versioned' ? `${mount}${version}/` : mount;
export const globalUrl = (path) => `${mount}${path.replace(/^\//, '')}`;
