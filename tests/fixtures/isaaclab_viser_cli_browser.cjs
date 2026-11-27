// Public-CLI browser proof in the prepared private profile; no dependency installs here.
const fs = require("node:fs");
const crypto = require("node:crypto");
const path = require("node:path");
const net = require("node:net");
const childProcess = require("node:child_process");
const [output, mode, portText] = process.argv.slice(2);
const port = Number(portText);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function allocateIpc(output, mode) {
  const root = process.env.TMPDIR;
  if (!root || !path.isAbsolute(root) || root !== "/tmp/ray" || fs.realpathSync(root) !== root) {
    throw new Error("Live IPC requires the real /tmp/ray TMPDIR");
  }
  const type = fs.statfsSync(root).type;
  if (type !== 0xEF53) throw new Error(`Live IPC requires worker-local ext4, got ${type}`);
  const ipcPath = fs.mkdtempSync(path.join(root, `nexuml-spectator-${mode}-`));
  const receiptPath = path.join(output, "live-ipc.json");
  const receipt = {
    path: ipcPath, type, storage: "worker-local ext4", mode,
    scope: "Transient live IPC only; not checkpoint or evidence storage",
    cleanup: { status: "pending" },
  };
  try {
    fs.writeFileSync(receiptPath, JSON.stringify(receipt), { flag: "wx", mode: 0o600 });
  } catch (error) {
    try { fs.rmSync(ipcPath, { recursive: true }); }
    catch (cleanupError) { error.cleanupErrors = [cleanupError]; }
    throw error;
  }
  return {
    path: ipcPath,
    cleanup(child) {
      const terminal = child === null || Number.isInteger(child.exitCode) || typeof child.signalCode === "string";
      receipt.cleanup = { status: "retained", reason: "Owned child exit is unconfirmed" };
      let failure;
      try {
        if (terminal) {
          receipt.cleanup = { status: "failed", reason: "Owned IPC removal did not complete" };
          fs.rmSync(ipcPath, { recursive: true });
          receipt.cleanup = child === null
            ? { status: "removed", reason: "No native process was created" }
            : { status: "removed", exit_code: child.exitCode, signal: child.signalCode };
        }
      } catch (error) { failure = error; throw error; }
      finally {
        try { fs.writeFileSync(receiptPath, JSON.stringify(receipt), { mode: 0o600 }); }
        catch (error) {
          if (!failure) throw error;
          (failure.cleanupErrors ??= []).push(error);
        }
      }
    },
  };
}

function readProgress(ipc) {
  return JSON.parse(fs.readFileSync(path.join(ipc, "progress.json"), "utf8"));
}

function spawnNative(launch, log, ipc) {
  const child = childProcess.spawn(launch.argv[0], launch.argv.slice(1), {
    stdio: ["ignore", log, log],
    env: { ...process.env, ...launch.env, NEXUML_SPECTATOR_PROBE_IPC: ipc.path },
  });
  const native = { child, error: null };
  native.completed = new Promise((resolve) => {
    child.once("exit", (code, signal) => resolve({ code, signal }));
    child.once("error", (error) => {
      native.error = error;
      if (child.pid === undefined) resolve({ error });
    });
  });
  return native;
}

async function cleanup(browser, child, waitExit, completed, log, ipc, primary) {
  let failure = primary;
  const failed = (error) => {
    if (failure && failure !== error) (failure.cleanupErrors ??= []).push(error);
    else failure = error;
  };
  try {
    if (browser) await browser.close();
  } catch (error) { failed(error); }
  try {
    if (child && child.exitCode === null && child.signalCode === null) {
      child.kill("SIGTERM");
      try { await waitExit(20000); }
      catch { child.kill("SIGKILL"); await completed; }
    }
  } catch (error) { failed(error); }
  try { if (ipc) ipc.cleanup(child); }
  catch (error) { failed(error); }
  try { fs.closeSync(log); }
  catch (error) { failed(error); }
  if (failure) throw failure;
}

function verifyCompletion(output, mode, result) {
  const expected = mode.startsWith("failure") ? 1 : mode.startsWith("cancel") ? 143 : 0;
  if (result.code !== expected) throw new Error(`Native result ${JSON.stringify(result)} expected ${expected}`);
  const read = (name) => JSON.parse(fs.readFileSync(path.join(output, name), "utf8"));
  const proof = read("native-cleanup.json");
  if (proof.status !== "PASS" || proof.mode !== mode || !(proof.steps > 0)) {
    throw new Error("Native cleanup proof is absent or incomplete");
  }
  const trace = read("trajectory-manifest.json");
  if (trace.steps !== proof.steps || !Array.isArray(trace.parts) || trace.parts.length === 0) {
    throw new Error("Native trajectory manifest is incomplete");
  }
  let steps = 0;
  for (const [index, part] of trace.parts.entries()) {
    const expected = index === 0 ? "trajectory.pt" : `trajectory-${String(index).padStart(4, "0")}.pt`;
    if (part.path !== expected || !Number.isInteger(part.steps) || part.steps <= 0 || part.steps > 64) {
      throw new Error("Native trajectory part is invalid");
    }
    const target = path.join(output, part.path);
    const stat = fs.statSync(target);
    if (!stat.isFile() || stat.size <= 0 || stat.size > 32 * 1024**2 || stat.size !== part.bytes ||
        crypto.createHash("sha256").update(fs.readFileSync(target)).digest("hex") !== part.sha256) {
      throw new Error("Native trajectory part size/hash differs");
    }
    steps += part.steps;
  }
  if (steps !== trace.steps) throw new Error("Native trajectory steps are missing");
  if (!mode.startsWith("replay")) {
    const finalizations = read("logger-finalization.json");
    if (!Array.isArray(finalizations) || finalizations.length === 0 ||
        finalizations.some((event) => event.completed !== true || event.error !== null || !event.job_scope_active)) {
      throw new Error("Requested logger finalization did not complete successfully");
    }
  }
  const log = fs.readFileSync(`${output}.log`, "utf8");
  if (!log.includes(`CLI_PROBE_NATIVE_CLOSED_AND_OWNED_THREADS_STOPPED ${mode}`)) {
    throw new Error("Native cleanup sentinel is absent");
  }
  if (mode.startsWith("failure") || mode.startsWith("cancel")) {
    const injection = read("injection.json");
    const identity = mode.startsWith("failure") ? "injected spectator training failure" : "SIGTERM";
    if (injection.mode !== mode || injection.batch_idx !== 2 || injection.identity !== identity) {
      throw new Error("Expected third-batch injection proof is absent");
    }
    if (mode.startsWith("failure") && !log.replace(/\s+/g, " ").includes(identity)) {
      throw new Error("Expected injected native failure was not observed");
    }
  } else if (!mode.startsWith("replay")) {
    for (const name of ["training-state.pt", "complete.ckpt"]) {
      if (!fs.statSync(path.join(output, name)).isFile()) throw new Error(`Missing ${name}`);
    }
  }
}

async function main() {
  const { chromium } = require(process.env.NEXUML_PLAYWRIGHT_MODULE);
  const python = process.env.NEXUML_SPECTATOR_PYTHON;
  if (!python || !path.isAbsolute(python)) throw new Error("NEXUML_SPECTATOR_PYTHON must be the caller's absolute sys.executable");
  const prepared = childProcess.spawnSync(python, [path.join(__dirname, "isaaclab_viser_cli.py"), "prepare", mode, output, portText], {
    encoding: "utf8", timeout: 60000, env: { ...process.env, PYTHONDONTWRITEBYTECODE: "1" },
  });
  if (prepared.error || prepared.status !== 0) throw new Error(`CLI preparation failed: ${prepared.error || prepared.stderr}`);
  const launch = JSON.parse(prepared.stdout);
  if (launch.argv[0] !== python || launch.argv[1] !== "-m" || launch.argv[2] !== "nexuml.cli.main") {
    throw new Error("Preparation did not return the supplied interpreter's public CLI");
  }
  fs.mkdirSync(path.dirname(output), { recursive: true });
  fs.writeFileSync(`${output}.launch.json`, JSON.stringify(launch), { flag: "wx", mode: 0o600 });
  const log = fs.openSync(`${output}.log`, "wx");
  let ipc;
  try { ipc = allocateIpc(output, mode); }
  catch (error) {
    try { fs.closeSync(log); }
    catch (cleanupError) { (error.cleanupErrors ??= []).push(cleanupError); }
    throw error;
  }
  console.log("PUBLIC_CLI_WITH_TEST_OWNED_INSTANCE_PROBE", JSON.stringify(launch.argv));
  const errors = [];
  let browser, launched, primary;
  async function waitExit(ms) {
    let timer;
    try {
      const result = await Promise.race([launched.completed, new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error("Native completion deadline")), ms);
      })]);
      if (launched.error) throw launched.error;
      return result;
    } finally { clearTimeout(timer); }
  }
  try {
    launched = spawnNative(launch, log, ipc);
    for (let i = 0; i < 240 && !fs.existsSync(path.join(ipc.path, "ready.json")); ++i) {
      if (launched.error) throw launched.error;
      if (launched.child.exitCode !== null) throw new Error(`Native startup failed: ${launched.child.exitCode}`);
      await sleep(250);
    }
    if (!fs.existsSync(path.join(ipc.path, "ready.json"))) throw new Error("Native startup deadline");
    const executablePath = process.env.NEXUML_CHROMIUM_EXECUTABLE;
    const icd = process.env.NEXUML_BROWSER_SWIFTSHADER_ICD;
    const venv = process.env.UV_PROJECT_ENVIRONMENT;
    if (!venv || !path.isAbsolute(venv)) throw new Error("Private worker environment required");
    const runtime = path.join(venv, "browser-runtime");
    const setup = JSON.parse(fs.readFileSync(path.join(runtime, "setup.json"), "utf8"));
    if (setup.status !== "success" || setup.paths.chromeBinary !== executablePath ||
        setup.paths.playwrightModulePath !== process.env.NEXUML_PLAYWRIGHT_MODULE ||
        fs.realpathSync(process.execPath) !== fs.realpathSync(setup.paths.node) ||
        icd !== path.join(path.dirname(executablePath), "vk_swiftshader_icd.json")) {
      throw new Error("Browser selection differs from the admitted private runtime");
    }
    // Only the browser inherits client libraries/fonts; native CLI environment is unchanged.
    const browserEnvironment = {
      ...process.env, DISPLAY: "",
      LD_LIBRARY_PATH: setup.paths.libraryPrefix + (process.env.LD_LIBRARY_PATH ? `:${process.env.LD_LIBRARY_PATH}` : ""),
      HOME: setup.paths.homePath, FONTCONFIG_FILE: setup.paths.fontsConf,
      XDG_CACHE_HOME: setup.paths.fontCache,
      VK_ICD_FILENAMES: icd, VK_DRIVER_FILES: icd,
    };
    browser = await chromium.launch({
      headless: true, chromiumSandbox: true, executablePath, env: browserEnvironment,
      args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
    });
    fs.writeFileSync(path.join(output, "browser-runtime.json"), JSON.stringify({
      node: process.version, chromium: browser.version(), sandbox_requested: true,
      graphics_scope: "Browser-side SwiftShader only; no server RTX/sensor qualification", port,
    }), { flag: "wx", mode: 0o600 });
    async function connectPage() {
      const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
      page.setDefaultTimeout(7000);
      page.on("pageerror", (error) => errors.push(String(error)));
      page.on("console", (message) => console.log("BROWSER_CONSOLE", message.type(), message.text()));
      await page.goto(`http://127.0.0.1:${port}/`);
      await page.waitForLoadState("networkidle");
      await page.getByRole("button", { name: "Pause Display", exact: true }).waitFor();
      return page;
    }
    let page = await connectPage();
    fs.writeFileSync(path.join(output, "browser-dom.html"), await page.content());
    const progress = () => readProgress(ipc.path);
    async function native(check, label) {
      for (let i = 0; i < 30; ++i) {
        if (check(progress())) return;
        await sleep(200);
      }
      throw new Error(`Native display did not update: ${label}: ${JSON.stringify(progress())}`);
    }
    await native((p) => p.plots > 0 && p.plot_samples >= 3 && p.marker_count > 1, "live plot samples and native mesh/frame markers");
    if (!await page.locator(".uplot").first().isVisible()) {
      await page.getByText("joint_effort", { exact: true }).first().click();
    }
    await page.locator(".uplot").first().waitFor({ state: "visible" });
    await page.waitForFunction(() => {
      const canvas = document.querySelector('.uplot canvas');
      if (!canvas) return false;
      const bounds = canvas.getBoundingClientRect();
      return canvas.width > 0 && canvas.height > 0 && bounds.width > 0 && bounds.height > 0;
    }, null, {timeout:7000});
    const chartGeometry = await page.locator('.uplot').first().evaluate((plot) => ({
      bounds: {width:plot.getBoundingClientRect().width,height:plot.getBoundingClientRect().height},
      html: plot.outerHTML,
      canvases: [...plot.querySelectorAll('canvas')].map(canvas=>({
        width:canvas.width,height:canvas.height,
        cssWidth:canvas.getBoundingClientRect().width,cssHeight:canvas.getBoundingClientRect().height,
      })),
    }));
    fs.writeFileSync(path.join(output,'browser-chart-geometry.json'),JSON.stringify(chartGeometry));
    if (!chartGeometry.canvases.some(canvas=>canvas.width>0 && canvas.height>0 && canvas.cssWidth>0 && canvas.cssHeight>0)) {
      throw new Error('Native plot has no nonempty rendered canvas');
    }
    await page.screenshot({ path: path.join(output, "browser-native-geometry-and-plot.png"), animations:"disabled" });
    const before = progress();
    await page.getByRole("button", { name: "Pause Display", exact: true }).click();
    await sleep(1200);
    if (!(progress().steps > before.steps && progress().paused)) throw new Error("Display pause stopped execution");
    await page.getByRole("button", { name: "Pause Display", exact: true }).click();
    await page.mouse.move(700, 400);
    await page.mouse.down();
    await page.mouse.move(900, 500, { steps: 12 });
    await page.mouse.up();
    await page.mouse.wheel(0, -300);
    await sleep(700);
    const after = progress();
    if (JSON.stringify(after.cameras) === JSON.stringify(before.cameras)) throw new Error("Browser camera did not move");
    fs.writeFileSync(path.join(output, "browser-camera.json"), JSON.stringify({ before: before.cameras, after: after.cameras }));
    await page.screenshot({ path: path.join(output, "browser-moved.png") });
    const select = page.getByLabel("Visible Environment", { exact: true });
    for (const value of (mode.startsWith("replay") ? ["0", "Configured"] : ["0", "1", "0", "Configured"])) {
      await select.click();
      await page.getByRole("option", { name: value, exact: true }).click();
      await native((p) => p.selection === value, `environment ${value}`);
      if (!Object.values(progress().scene_visible).some(Boolean)) throw new Error("Selection permanently hid geometry");
    }
    await page.getByLabel("Markers", { exact: true }).uncheck();
    await native((p) => Object.values(p.marker_visible).every((v) => !v), "markers off");
    await page.getByLabel("Markers", { exact: true }).check();
    await native((p) => p.marker_count > 1 && Object.values(p.marker_visible).every(Boolean), "mesh/frame markers restored");
    await page.getByLabel("Live Plots", { exact: true }).uncheck();
    await native((p) => !p.plot_visible, "plots off");
    await page.getByLabel("Live Plots", { exact: true }).check();
    await native((p) => p.plot_visible && p.plots > 0, "plots restored");
    await page.getByRole("button", { name: "Reset Camera", exact: true }).click();
    for (const name of ["Pause Simulation", "Reset Episode", "Apply Force"]) {
      if (await page.getByRole("button", { name, exact: true }).count()) throw new Error(`Native mutator exposed: ${name}`);
    }
    await page.request.get(`http://127.0.0.1:${port}/?reset=true&action=100&force=100&budget=0`);
    await page.screenshot({ path: path.join(output, "browser-controls-restored.png") });
    const disconnectedAt = progress().steps;
    await page.close();
    await sleep(1000);
    if (!(progress().steps > disconnectedAt && progress().clients === 0)) throw new Error("Disconnect stopped execution");
    page = await connectPage();
    await native((p) => p.clients > 0, "reconnect");
    await page.screenshot({ path: path.join(output, "browser-reconnected.png") });
    console.log("NATIVE_BROWSER_CAMERA_SELECTION_MARKERS_PLOTS_PAUSE_AND_RECONNECT_VERIFIED", mode);
    fs.writeFileSync(path.join(output, "browser-errors.json"), JSON.stringify(errors));
    if (errors.length) throw new Error(errors.join("\n"));
    await browser.close(); browser = null;
    const result = await waitExit(120000);
    verifyCompletion(output, mode, result);
    const released = net.createServer();
    await new Promise((resolve, reject) => released.once("error", reject).listen(port, "127.0.0.1", resolve));
    await new Promise((resolve) => released.close(resolve));
    console.log("NATIVE_BROWSER_JOB_STATUS_AND_LISTENER_CLEANUP_VERIFIED", mode, result.code);
  } catch (error) { primary = error; throw error; }
  finally {
    const child = !launched || (launched.error && launched.child.pid === undefined) ? null : launched.child;
    await cleanup(browser, child, waitExit, launched?.completed, log, ipc, primary);
  }
}
module.exports = { allocateIpc, readProgress, spawnNative, cleanup, verifyCompletion };
if (require.main === module) main().catch((error) => { console.error(error); process.exitCode = 1; });
