#!/usr/bin/env node
import { spawn } from "node:child_process";
import { createRequire } from "node:module";
import { cp, mkdir, mkdtemp, readFile, realpath, rm, stat, symlink } from "node:fs/promises";
import { constants } from "node:fs";
import { access } from "node:fs/promises";
import { dirname, join, resolve, delimiter } from "node:path";
import { fileURLToPath } from "node:url";
import { tmpdir } from "node:os";
import { randomBytes } from "node:crypto";
import { createServer } from "node:net";
import { parseArgs } from "node:util";

const require = createRequire(import.meta.url);
const packageDirectory = resolve(dirname(fileURLToPath(import.meta.url)), "..");

export function argumentsFor(argv) {
  const result = parseArgs({ args: argv, allowPositionals: true, options: {
    nexuml: {type:"string"}, python: {type:"string"}, api: {type:"string"},
    "api-token-file": {type:"string"}, port: {type:"string"}, "api-port": {type:"string"},
    "no-open": {type:"boolean"}, help: {type:"boolean", short:"h"},
  }});
  if (result.positionals.length > 1) throw new Error("Supply one working directory.");
  const { values } = result;
  if ([values.nexuml, values.python, values.api].filter(Boolean).length > 1) throw new Error("Choose one of --nexuml, --python, or --api.");
  if (values["api-token-file"] && !values.api) throw new Error("--api-token-file requires --api.");
  for (const key of ["port", "api-port"]) if (values[key] && !/^[0-9]+$/.test(values[key])) throw new Error(`${key} must be a port.`);
  return { ...values, directory: resolve(result.positionals[0] ?? ".") };
}

export async function executable(name, env = process.env) {
  const candidates = name.includes("/") || name.includes("\\") ? [resolve(name)] :
    (env.PATH ?? "").split(delimiter).flatMap(directory => process.platform === "win32" ?
      [join(directory, name), join(directory, `${name}.exe`)] : [join(directory, name)]);
  for (const path of candidates) {
    try { await access(path, process.platform === "win32" ? constants.F_OK : constants.X_OK); if ((await stat(path)).isFile()) return path; }
    catch { /* Try the next PATH entry, never install a missing executable. */ }
  }
  throw new Error(`Executable not found: ${name}. Select an existing installation with --nexuml or --python.`);
}

export async function freePort(value) {
  const port = value ? Number(value) : 0;
  if (!Number.isInteger(port) || port < 0 || port > 65535 || (value && port === 0)) throw new Error("Port must be between 1 and 65535.");
  return await new Promise((done, reject) => {
    const server = createServer();
    server.once("error", reject);
    server.listen(port, "127.0.0.1", () => { const selected = server.address().port; server.close(() => done(selected)); });
  });
}

async function ready(url, headers, child, token) {
  const deadline = Date.now() + 60000;
  while (Date.now() < deadline) {
    if (child && (child.exitCode !== null || child.signalCode !== null)) throw new Error("Owned service exited during startup. Check the selected NexuML API extra and server diagnostic above.");
    try {
      const response = await fetch(url, { headers, signal: AbortSignal.timeout(1000) });
      if (response.ok) return response;
      if ([401,403,409].includes(response.status)) throw new Error(`Startup rejected (${response.status}). Check token, exact origin, directory and interface version.`);
    } catch (error) {
      if (error.message.startsWith("Startup rejected")) throw error;
    }
    await new Promise(done => setTimeout(done, 150));
  }
  throw new Error(`Service readiness timed out: ${url.replaceAll(token, "[redacted]")}`);
}

async function stop(child) {
  if (!child || child.exitCode !== null || child.signalCode !== null) return;
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
  const directory = await realpath(options.directory);
  if (!(await stat(directory)).isDirectory()) throw new Error("Working directory must exist.");
  const port = await freePort(options.port);
  const origin = `http://127.0.0.1:${port}`;
  let token = randomBytes(32).toString("base64url");
  let apiUrl;
  let apiChild;
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
      const file = options["api-token-file"];
      if (process.platform !== "win32" && (await stat(file)).mode & 0o077) throw new Error("API token file must be private (chmod 600).");
      token = (await readFile(file, "utf8")).trim();
      if (Buffer.byteLength(token) < 32) throw new Error("API token must be at least 32 bytes.");
      apiUrl = url.origin;
    } else {
      const apiPort = await freePort(options["api-port"]);
      if (apiPort === port) throw new Error("Studio and API need different ports.");
      apiUrl = `http://127.0.0.1:${apiPort}`;
      const command = await executable(options.python ?? options.nexuml ?? "nexuml");
      const args = [...(options.python ? ["-m", "nexuml.cli.main"] : []), "serve",
        "--directory", directory, "--port", String(apiPort), "--origin", origin];
      apiChild = spawn(command, args, {cwd:directory, env:{...process.env, NEXUML_API_TOKEN:token},
        stdio:"inherit", detached:process.platform !== "win32"});
      apiChild.on("error", error => console.error(`NexuML startup: ${error.message}`));
    }
    const response = await ready(`${apiUrl}/api/v1/runtime`, {Authorization:`Bearer ${token}`, Origin:origin, "X-NexuML-Interface":"1"}, apiChild, token);
    const identity = await response.json();
    if (identity.interface_version !== 1) throw new Error("Studio/API interface mismatch (requires 1).");
    if (await realpath(identity.working_directory) !== directory) throw new Error("Attached API directory differs from the selected working directory.");
    console.log(`NexuML ${identity.nexuml_version} · ${identity.python_executable}\nDirectory: ${directory}`);
    // Next can write runtime/cache files: use a private disposable copy, not npm assets.
    await access(join(packageDirectory, ".next", "BUILD_ID"));
    runtimeDirectory = await mkdtemp(join(tmpdir(), "nexuml-studio-"));
    await cp(join(packageDirectory, ".next"), join(runtimeDirectory, ".next"), {recursive:true,
      filter: source => !source.includes(`${process.platform === "win32" ? "\\" : "/"}cache${process.platform === "win32" ? "\\" : "/"}`)});
    await cp(join(packageDirectory, "next.config.mjs"), join(runtimeDirectory, "next.config.mjs"));
    const packageJson = JSON.parse(await readFile(join(packageDirectory, "package.json"), "utf8"));
    await mkdir(join(runtimeDirectory, "node_modules"));
    for (const name of Object.keys(packageJson.dependencies)) {
      const location = dirname(require.resolve(`${name}/package.json`));
      const destination = join(runtimeDirectory, "node_modules", name);
      await mkdir(dirname(destination), {recursive:true});
      await symlink(location, destination, process.platform === "win32" ? "junction" : "dir");
    }
    uiChild = spawn(process.execPath, [require.resolve("next/dist/bin/next"), "start", runtimeDirectory,
      "--hostname", "127.0.0.1", "--port", String(port)], {cwd:runtimeDirectory,
      env:{...process.env, NEXT_TELEMETRY_DISABLED:"1", NEXUML_STUDIO_TOKEN:token,
        NEXUML_STUDIO_API:apiUrl, NEXUML_STUDIO_ORIGIN:origin}, stdio:"inherit", detached:process.platform !== "win32"});
    await ready(origin, {}, uiChild, token);
    console.log(`Studio ready: ${origin}`);
    if (!options["no-open"]) {
      const command = process.platform === "win32" ? "explorer.exe" : process.platform === "darwin" ? "open" : "xdg-open";
      const browser = spawn(command, [origin], {stdio:"ignore"});
      browser.on("error", () => console.log("Open the Studio address in your browser."));
      browser.unref();
    }
    await Promise.race([new Promise(done => uiChild.once("exit", done)),
      ...(apiChild ? [new Promise(done => apiChild.once("exit", done))] : [])]);
  } finally {
    await shutdown();
    process.removeListener("SIGINT", onSignal);
    process.removeListener("SIGTERM", onSignal);
    process.removeListener("message", onMessage);
  }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    const options = argumentsFor(process.argv.slice(2));
    if (options.help) console.log("nexuml-studio [directory] [--nexuml executable | --python interpreter | --api origin --api-token-file file] [--port number] [--api-port number] [--no-open]\nUses an existing NexuML with its api extra. Never installs or syncs Python packages. Trusted libraries/scenarios execute local Python, not sandboxed code.");
    else await launch(options);
  } catch (error) { console.error(`Studio: ${error.message}`); process.exitCode = 1; }
}
