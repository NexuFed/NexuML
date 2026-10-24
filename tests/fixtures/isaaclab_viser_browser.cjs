// Opt-in browser proof. Both tools must already be installed; this never installs them.
const fs = require("node:fs");
const path = require("node:path");
const net = require("node:net");
const { spawn } = require("node:child_process");
const { chromium } = require(process.env.NEXUML_PLAYWRIGHT_MODULE);
const [output, mode, portText] = process.argv.slice(2);
const port = Number(portText);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function main() {
  fs.mkdirSync(path.dirname(output), { recursive: true });
  const log = fs.openSync(`${output}.log`, "wx");
  const child = spawn(".venv/bin/python", ["tests/fixtures/isaaclab_viser_job.py", mode, output, portText], {
    stdio: ["ignore", log, log],
    env: { ...process.env, OMP_NUM_THREADS: "1", MKL_NUM_THREADS: "1", NEXUML_LOGS_ROOT: path.join(output, "native-logs") },
  });
  const completed = new Promise((resolve) => child.on("exit", (code, signal) => resolve({ code, signal })));
  const errors = [];
  let browser;
  async function waitExit(ms) {
    let timer;
    try {
      return await Promise.race([completed, new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error("Native completion deadline")), ms);
      })]);
    } finally { clearTimeout(timer); }
  }
  try {
    for (let i = 0; i < 240 && !fs.existsSync(path.join(output, "ready.json")); ++i) {
      if (child.exitCode !== null) throw new Error(`Native startup failed: ${child.exitCode}`);
      await sleep(250);
    }
    if (!fs.existsSync(path.join(output, "ready.json"))) throw new Error("Native startup deadline");
    const executablePath = process.env.NEXUML_CHROMIUM_EXECUTABLE;
    const icd = process.env.NEXUML_BROWSER_SWIFTSHADER_ICD;
    const browserEnvironment = icd ? {
      ...process.env, DISPLAY: "", LD_LIBRARY_PATH: path.dirname(executablePath),
      VK_ICD_FILENAMES: icd, VK_DRIVER_FILES: icd,
    } : process.env;
    browser = await chromium.launch({
      headless: true, executablePath, env: browserEnvironment,
      args: icd ? ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"] : [],
    });
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
    const progress = () => JSON.parse(fs.readFileSync(path.join(output, "progress.json"), "utf8"));
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
    const expected = mode.startsWith("failure") ? 1 : mode.startsWith("cancel") ? 143 : 0;
    if (result.code !== expected) throw new Error(`Native result ${JSON.stringify(result)} expected ${expected}`);
    const released = net.createServer();
    await new Promise((resolve, reject) => released.once("error", reject).listen(port, "127.0.0.1", resolve));
    await new Promise((resolve) => released.close(resolve));
    console.log("NATIVE_BROWSER_JOB_STATUS_AND_LISTENER_CLEANUP_VERIFIED", mode, result.code);
  } finally {
    if (browser) await browser.close();
    if (child.exitCode === null && child.signalCode === null) {
      child.kill("SIGTERM");
      try { await waitExit(20000); }
      catch { child.kill("SIGKILL"); await completed; }
    }
    fs.closeSync(log);
  }
}
main().catch((error) => { console.error(error); process.exitCode = 1; });
