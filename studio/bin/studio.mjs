#!/usr/bin/env node
import { spawn } from "node:child_process";
import { createRequire } from "node:module";
import { access, cp, mkdir, mkdtemp, readFile, realpath, rm, stat, symlink } from "node:fs/promises";
import { constants } from "node:fs";
import { basename, dirname, join, resolve, delimiter } from "node:path";
import { fileURLToPath } from "node:url";
import { tmpdir } from "node:os";
import { randomBytes } from "node:crypto";
import { createServer } from "node:net";
import { parseArgs } from "node:util";
import { listenHost, publicOrigin } from "./connection.mjs";
import { backendRequirement, compatibilityError, interfaceVersion } from "./compatibility.mjs";

const require = createRequire(import.meta.url);
const packageDirectory = resolve(dirname(fileURLToPath(import.meta.url)), "..");

/** @param {string[]} argv @param {Record<string, string | undefined>} env */
export function argumentsFor(argv, env = process.env) {
  const result = parseArgs({ args: argv, allowPositionals: true, options: {
    nexuml: {type:"string"}, python: {type:"string"}, api: {type:"string"},
    "api-token-file": {type:"string"}, port: {type:"string"}, "api-port": {type:"string"},
    host: {type:"string"}, "public-origin": {type:"string"}, "proxy-token-file": {type:"string"},
    "no-open": {type:"boolean"}, help: {type:"boolean", short:"h"},
  }});
  if (result.positionals.length > 1) throw new Error("Supply one working directory.");
  const { values } = result;
  if ([values.nexuml, values.python, values.api].filter(Boolean).length > 1) throw new Error("Choose one of --nexuml, --python, or --api.");
  if (values["api-token-file"] && !values.api) throw new Error("--api-token-file requires --api.");
  for (const key of ["port", "api-port"]) if (values[key] && !/^[0-9]+$/.test(values[key])) throw new Error(`${key} must be a port.`);
  values["public-origin"] ??= env.NEXUML_STUDIO_PUBLIC_ORIGIN;
  values["proxy-token-file"] ??= env.NEXUML_STUDIO_PROXY_TOKEN_FILE;
  if (values["public-origin"] && !values["proxy-token-file"]) throw new Error("Hosted mode requires --proxy-token-file and an authenticated HTTPS reverse proxy that overwrites X-NexuML-Studio-Proxy.");
  if (values["proxy-token-file"] && !values["public-origin"]) throw new Error("--proxy-token-file requires --public-origin.");
  values.host = listenHost(values.host ?? env.NEXUML_STUDIO_HOST, publicOrigin(values["public-origin"]));
  return { ...values, directory: resolve(result.positionals[0] ?? ".") };
}

async function privateToken(file) {
  if (process.platform !== "win32" && (await stat(file)).mode & 0o077) throw new Error("Token file must be private (chmod 600).");
  const token = (await readFile(file, "utf8")).trim();
  if (Buffer.byteLength(token) < 32) throw new Error("Token must be at least 32 bytes.");
  return token;
}

export async function executable(name, env = process.env) {
  const candidates = name.includes("/") || name.includes("\\") ? [resolve(name)] :
    (env.PATH ?? "").split(delimiter).flatMap(directory => process.platform === "win32" ?
      [join(directory, name), join(directory, `${name}.exe`)] : [join(directory, name)]);
  for (const path of candidates) {
    try { await access(path, process.platform === "win32" ? constants.F_OK : constants.X_OK); if ((await stat(path)).isFile()) return path; }
    catch { /* Try the next PATH entry, never install a missing executable. */ }
  }
  throw new Error(`Executable not found: ${name}. Install the backend with uv tool install "nexuml[api,library]" and ensure nexuml is on PATH (uv tool update-shell), or select an existing project: nexuml-studio --python /path/to/project/.venv/bin/python .`);
}

export async function freePort(value, host = "127.0.0.1") {
  const port = value ? Number(value) : 0;
  if (!Number.isInteger(port) || port < 0 || port > 65535 || (value && port === 0)) throw new Error("Port must be between 1 and 65535.");
  return await new Promise((done, reject) => {
    const server = createServer();
    server.once("error", reject);
    server.listen(port, host, () => { const selected = server.address().port; server.close(() => done(selected)); });
  });
}

async function ready(url, headers, child, token, service) {
  const deadline = Date.now() + 60000;
  while (Date.now() < deadline) {
    if (child && (!child.pid || child.exitCode !== null || child.signalCode !== null)) throw new Error(`${service} exited during startup (${child.signalCode ?? child.exitCode ?? "could not spawn"}). Read the server diagnostic above. Ensure the selected environment contains nexuml[api,library]; for a tool installation use uv tool install --reinstall "nexuml[api,library]", or select your project with --python /path/to/project/.venv/bin/python.`);
    try {
      const response = await fetch(url, { headers, signal: AbortSignal.timeout(1000) });
      if (response.ok) return response;
      if ([401,403,409].includes(response.status)) throw new Error(`Startup rejected (${response.status}) by ${service}. Studio requires ${backendRequirement}. Check the selected API's version, private token file, exact origin and directory. For a different project use nexuml-studio --python /path/to/project/.venv/bin/python .`);
    } catch (error) {
      if (error.message.startsWith("Startup rejected")) throw error;
    }
    await new Promise(done => setTimeout(done, 150));
  }
  throw new Error(`${service} did not become ready within 60 seconds: ${url.replaceAll(token, "[redacted]")}. Check the server diagnostic above and that this selected environment has the API extra. Stop Studio with Ctrl-C before restarting.`);
}

async function stop(child) {
  if (!child || !child.pid || child.exitCode !== null || child.signalCode !== null) return;
  const exited = new Promise(done => child.once("exit", done));
  if (process.platform === "win32") {
    // Windows SIGTERM is forceful. Stop the owned tree before losing its leader.
    await new Promise(done => spawn("taskkill", ["/PID", String(child.pid), "/T", "/F"], {stdio:"ignore"}).once("exit", done));
    await exited;return;
  }
  child.kill("SIGTERM");
  await Promise.race([exited, new Promise(done => setTimeout(done, 15000))]);
  if (child.exitCode !== null || child.signalCode !== null) return;
  try { process.kill(-child.pid, "SIGKILL"); } catch (error) { if (error.code !== "ESRCH") throw error; }
  await exited;
}

export async function launch(options) {
  const [major,minor]=process.versions.node.split(".").map(Number);
  if(major<22 || (major===22 && minor<13))throw new Error("Studio requires Node >=22.13. Select/install a supported Node version yourself.");
  const hostedOrigin = publicOrigin(options["public-origin"] ?? process.env.NEXUML_STUDIO_PUBLIC_ORIGIN);
  const host = listenHost(options.host ?? process.env.NEXUML_STUDIO_HOST, hostedOrigin);
  const proxyFile = options["proxy-token-file"] ?? process.env.NEXUML_STUDIO_PROXY_TOKEN_FILE;
  if (hostedOrigin && !proxyFile) throw new Error("Hosted mode requires --proxy-token-file and an authenticated HTTPS reverse proxy.");
  const proxyToken = hostedOrigin ? await privateToken(proxyFile) : "";
  if (host !== "127.0.0.1") console.warn("WARNING: non-loopback Studio listener. Firewall this port to the authenticated proxy only; never expose the Python API port.");
  let directory;
  try {
    directory = await realpath(options.directory);
    if (!(await stat(directory)).isDirectory()) throw new Error("Not a directory");
  } catch {
    throw new Error(`Working directory does not exist, is not a directory, or cannot be accessed: ${options.directory}. Select an existing directory: nexuml-studio /path/to/work`);
  }
  const port = await freePort(options.port, host);
  const origin = `http://127.0.0.1:${port}`;
  const browserOrigin = hostedOrigin ?? origin;
  let token = randomBytes(32).toString("base64url");
  let apiUrl;
  let apiChild;
  let apiService="Attached NexuML API";
  let uiChild;
  let runtimeDirectory;
  let closing;
  const shutdown = () => closing ??= (async () => {
    await stop(uiChild);
    await stop(apiChild); // Attach mode never adds a process handle here.
    if (runtimeDirectory) await rm(runtimeDirectory, {recursive:true, force:true});
  })();
  const onSignal = () => { void shutdown().then(() => process.exit(0)); };
  const onMessage = message => { if (message === "shutdown") onSignal(); };
  if (process.connected) process.on("message",onMessage);
  process.once("SIGINT", onSignal);
  process.once("SIGTERM", onSignal);
  try {
    if (options.api) {
      const url = new URL(options.api);
      if (url.protocol !== "http:" || !["127.0.0.1", "localhost", "[::1]"].includes(url.hostname) ||
          url.username || url.password || url.pathname !== "/" || url.search || url.hash) throw new Error("--api must be a loopback HTTP origin.");
      if (!options["api-token-file"]) throw new Error("Attach requires --api-token-file.");
      token = await privateToken(options["api-token-file"]);
      apiUrl = url.origin;
    } else {
      const apiPort = await freePort(options["api-port"]);
      if (apiPort === port) throw new Error("Studio and API need different ports.");
      apiUrl = `http://127.0.0.1:${apiPort}`;
      const command = await executable(options.python ?? options.nexuml ?? "nexuml");
      apiService=`NexuML API (${command})`;
      console.log(`Starting ${apiService}\nDirectory: ${directory}`);
      const args = [...(options.python ? ["-m", "nexuml.cli.main"] : []), "serve",
        "--directory", directory, "--port", String(apiPort), "--origin", origin];
      apiChild = spawn(command, args, {cwd:directory, env:{...process.env, NEXUML_API_TOKEN:token, NEXUML_STUDIO_PROXY_TOKEN:""},
        stdio:"inherit", detached:process.platform !== "win32"});
      apiChild.on("error", error => console.error(`NexuML startup: ${error.message}`));
    }
    if (hostedOrigin && proxyToken === token) throw new Error("Proxy and API credentials must be different.");
    const response = await ready(`${apiUrl}/api/v1/runtime`, {Authorization:`Bearer ${token}`, Origin:origin, "X-NexuML-Interface":String(interfaceVersion)}, apiChild, token, apiService);
    const identity = await response.json();
    const incompatible = compatibilityError(identity);
    if (incompatible) throw new Error(`${apiService}: ${incompatible}`);
    if (await realpath(identity.working_directory) !== directory) throw new Error(`API working directory ${identity.working_directory} differs from selected ${directory}. Select the API's directory or restart it with --directory "${directory}".`);
    console.log(`NexuML ${identity.nexuml_version} · Python ${identity.python_version ?? "unknown"} · ${identity.python_executable}\nDirectory: ${directory}`);
    // Next can write runtime/cache files: use a private disposable copy, not npm assets.
    await access(join(packageDirectory, ".next", "BUILD_ID"));
    runtimeDirectory = await mkdtemp(join(tmpdir(), "nexuml-studio-"));
    await cp(join(packageDirectory, ".next"), join(runtimeDirectory, ".next"), {recursive:true,
      filter: source => basename(source) !== "cache"});
    await cp(join(packageDirectory, "next.config.mjs"), join(runtimeDirectory, "next.config.mjs"));
    await mkdir(join(runtimeDirectory, "bin"));
    await cp(join(packageDirectory, "bin", "connection.mjs"), join(runtimeDirectory, "bin", "connection.mjs"));
    await mkdir(join(runtimeDirectory, "node_modules"));
    // Use npm's platform-correct runtime packages; standalone tracing includes build-host native binaries.
    for (const name of ["next", "react", "react-dom"]) {
      const location = dirname(require.resolve(`${name}/package.json`));
      const destination = join(runtimeDirectory, "node_modules", name);
      await mkdir(dirname(destination), {recursive:true});
      await symlink(location, destination, process.platform === "win32" ? "junction" : "dir");
    }
    uiChild = spawn(process.execPath, [require.resolve("next/dist/bin/next"), "start", runtimeDirectory,
      "--hostname", host, "--port", String(port)], {cwd:runtimeDirectory,
      env:{...process.env, NODE_ENV:"production", NEXT_TELEMETRY_DISABLED:"1", NEXUML_STUDIO_TOKEN:token,
        NEXUML_STUDIO_API:apiUrl, NEXUML_STUDIO_ORIGIN:origin,
        NEXUML_STUDIO_HOST:host, NEXUML_STUDIO_PUBLIC_ORIGIN:hostedOrigin ?? "", NEXUML_STUDIO_PROXY_TOKEN:proxyToken}, stdio:"inherit", detached:process.platform !== "win32"});
    await ready(`http://${host === "0.0.0.0" ? "127.0.0.1" : host}:${port}`, {}, uiChild, token, "Studio UI");
    console.log(`Studio ready: ${browserOrigin}`);
    if (!options["no-open"]) {
      const command = process.platform === "win32" ? "explorer.exe" : process.platform === "darwin" ? "open" : "xdg-open";
      const browser = spawn(command, [browserOrigin], {stdio:"ignore"});
      const openManually=()=>console.log(`Browser could not open automatically. Open ${browserOrigin} in your browser; keep this terminal running.`);
      browser.on("error", openManually);
      browser.on("exit", code => {if(code)openManually();});
      browser.unref();
    }
    const exited=await Promise.race([new Promise(done => uiChild.once("exit", (code,signal)=>done({service:"Studio UI",code,signal}))),
      ...(apiChild ? [new Promise(done => apiChild.once("exit", (code,signal)=>done({service:apiService,code,signal})))] : [])]);
    if(!closing)throw new Error(`${exited.service} stopped (${exited.signal ?? exited.code}). Check its diagnostic above, then restart nexuml-studio with the same selected runtime and directory. Run records remain on disk.`);
  } finally {
    await shutdown();
    process.removeListener("SIGINT", onSignal);
    process.removeListener("SIGTERM", onSignal);
    process.removeListener("message", onMessage);
  }
}

if (process.argv[1] && await realpath(process.argv[1]) === await realpath(fileURLToPath(import.meta.url))) {
  try {
    const options = argumentsFor(process.argv.slice(2));
    if (options.help) console.log("nexuml-studio [directory] [--nexuml executable | --python interpreter | --api origin --api-token-file file] [--port number] [--api-port number] [--host IPv4] [--public-origin https://host --proxy-token-file file] [--no-open]\nHost/public origin/proxy token file also use NEXUML_STUDIO_HOST/NEXUML_STUDIO_PUBLIC_ORIGIN/NEXUML_STUDIO_PROXY_TOKEN_FILE. Listener defaults to 127.0.0.1 even when hosted. Hosting requires an authenticated HTTPS reverse proxy overwriting X-NexuML-Studio-Proxy; firewall any explicitly non-loopback UI listener to that proxy only. The API stays loopback-only.\nUses an existing NexuML with its api extra. Never installs or syncs Python packages. Trusted libraries/scenarios execute local Python, not sandboxed code.");
    else await launch(options);
  } catch (error) { console.error(`Studio: ${error.message}`); process.exitCode = 1; }
}
