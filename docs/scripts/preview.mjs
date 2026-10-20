import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { resolve, extname, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { mount, version } from '../config.mjs';

const root = fileURLToPath(new URL('../.output/preview/', import.meta.url));
const mime = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json',
  '.svg': 'image/svg+xml', '.webp': 'image/webp', '.png': 'image/png', '.woff2': 'font/woff2', '.webm': 'video/webm', '.mp4': 'video/mp4', '.vtt': 'text/vtt', '.xml': 'application/xml', '.yaml': 'text/yaml', '.wasm': 'application/wasm' };
createServer(async (req, res) => {
  try {
    let file = resolve(root, `.${decodeURIComponent(new URL(req.url, 'http://localhost').pathname)}`);
    if (file !== resolve(root) && !file.startsWith(root.endsWith(sep) ? root : `${root}${sep}`)) { res.writeHead(403); res.end(); return; }
    if ((await stat(file)).isDirectory()) file = resolve(file, 'index.html');
    const body = await readFile(file);
    res.writeHead(200, { 'Content-Type': mime[extname(file)] || 'application/octet-stream' });
    res.end(body);
  } catch { res.writeHead(404); res.end('Not found'); }
}).listen(Number(process.env.PORT || 4321), '127.0.0.1', () => {
  console.log(`Preview: http://127.0.0.1:${process.env.PORT || 4321}${mount}${version}/`);
});
