// Remote recon only: caller owns producer validation, servers, and final UI acceptance.
const fs = require("node:fs");
const path = require("node:path");

function allowedRequest(value) {
  if (typeof value !== "string" || !/^http:\/\/127\.0\.0\.1:(6007|5001)(?:[/?#]|$)/.test(value)) return false;
  try {
    const url = new URL(value);
    return url.protocol === "http:" && url.hostname === "127.0.0.1" &&
      !url.username && !url.password && ["6007", "5001"].includes(url.port);
  } catch { return false; }
}

function safeLogURL(value) {
  try {
    const url = new URL(value);
    url.username = ""; url.password = ""; url.search = ""; url.hash = "";
    return url.href;
  } catch { return "[invalid URL]"; }
}

function validateRunIdentity(manifest, mode = "separate") {
  if (!["separate", "both"].includes(mode)) throw new Error("browser mode must be separate or both");
  const tbMode = mode === "both" ? "both" : "tb-only";
  const mlMode = mode === "both" ? "both" : "mlflow-only";
  const tb = manifest?.modes?.[tbMode]?.native?.tensorboard?.main;
  const ml = manifest?.modes?.[mlMode]?.native?.mlflow?.main;
  if (typeof tb?.log_dir !== "string" || !path.isAbsolute(tb.log_dir) ||
      typeof tb.image_tag !== "string" || !tb.image_tag ||
      typeof ml?.experiment_id !== "string" || !ml.experiment_id || /[^0-9]/.test(ml.experiment_id) ||
      typeof ml.run_id !== "string" || ml.run_id.length !== 32 || /[^0-9a-f]/.test(ml.run_id)) {
    throw new Error(`Expected ${tbMode}.native.tensorboard.main and ${mlMode}.native.mlflow.main native identities`);
  }
  return { tensorboard: { log_dir: tb.log_dir, image_tag: tb.image_tag },
    mlflow: { experiment_id: ml.experiment_id, run_id: ml.run_id } };
}

function validateFailureIdentity(manifest) {
  if (manifest?.status !== "FAILED" || manifest?.cli?.status !== "FAIL" || manifest.cli.exit_code !== 1 ||
      manifest.experiment_name !== "physical-ai-logging-partial-delivery" ||
      typeof manifest.run_id !== "string" || !/^[a-f0-9]{32}$/.test(manifest.run_id) ||
      typeof manifest.experiment_id !== "string" || !/^\d+$/.test(manifest.experiment_id)) {
    throw new Error("Expected explicit failed-run identity and original CLI failure");
  }
  return { mlflow: { run_id: manifest.run_id, experiment_id: manifest.experiment_id, status: "Failed" } };
}

async function waitForNativePage(page, label, identity) {
  if (label === "MLflow") {
    await page.getByText(identity.mlflow.run_id, { exact: true }).waitFor({ state: "visible", timeout: 30000 });
    await page.getByText(identity.mlflow.status || "Finished", { exact: true }).waitFor({ state: "visible", timeout: 30000 });
  } else {
    await page.getByText(/TensorBoard/i).first().waitFor({ state: "visible", timeout: 30000 });
    const image = page.locator('img[src*="data/plugin/images/individualImage"]').first();
    await image.waitFor({ state: "visible", timeout: 30000 });
    await image.evaluate(async image => {
      let timer;
      try {
        await Promise.race([image.decode(), new Promise((_, reject) => { timer = setTimeout(() => reject(new Error("Native GIF decode deadline exceeded")), 30000); })]);
        if (!(image.naturalWidth > 0 && image.naturalHeight > 0)) throw new Error("Native GIF was not decoded");
      } finally { clearTimeout(timer); }
    });
  }
}

async function savePageSnapshot(page, label, files) {
  const discovery = await page.evaluate(() => {
    const roots = [document], elements = [];
    for (let i = 0; i < roots.length; ++i) {
      for (const element of roots[i].querySelectorAll("*")) {
        if (element.shadowRoot) roots.push(element.shadowRoot);
        if (element.matches('button,a,input,img,video,[role="button"],[role="slider"],input[type="range"]')) {
          elements.push({ tag: element.localName, id: element.id, role: element.getAttribute("role"),
            text: element.innerText || element.getAttribute("alt") || "", ariaLabel: element.getAttribute("aria-label"),
            type: element.getAttribute("type"), href: element.getAttribute("href"), src: element.getAttribute("src"),
            outerHTML: element.outerHTML, shadowRoot: i, visible: !!element.getClientRects().length,
            media: element.localName === "video" ? { currentSrc: element.currentSrc, readyState: element.readyState,
              currentTime: element.currentTime, duration: Number.isFinite(element.duration) ? element.duration : null,
              width: element.videoWidth, height: element.videoHeight, paused: element.paused,
              error: element.error ? { code: element.error.code, message: element.error.message } : null } : undefined });
        }
      }
    }
    return { bodyText: [document.body?.innerText || "", ...roots.slice(1).flatMap(root =>
      [...root.children].filter(child => !["STYLE", "SCRIPT"].includes(child.tagName)).map(child => child.innerText || ""))].join("\n"), elements, shadowRootCount: roots.length - 1 };
  });
  fs.writeFileSync(files.get(`${label}.html`), await page.content());
  fs.writeFileSync(files.get(`${label}.json`), JSON.stringify(discovery, null, 2));
  fs.writeFileSync(files.get(`${label}.png`), await page.screenshot({ fullPage: true, type: "png", timeout: 30000 }));
  if (!discovery.bodyText.trim()) throw new Error(`${label} expected page is empty`);
}

async function inspectNativePanels(tbPage, mlflowPage, snapshot) {
  for (const [page, label, open] of [
    [tbPage, "TB-scalars", async () => {
      await tbPage.getByRole("tab", { name: /^scalars$/i }).click({ timeout: 30000 });
      await tbPage.locator("tf-scalar-dashboard").waitFor({ state: "visible", timeout: 30000 });
      const train = tbPage.getByRole("button", { name: /^train\s+6$/ });
      if (await train.getAttribute("open-button") === null) await train.click({ timeout: 30000 });
      await tbPage.getByText("train/rl/frames", { exact: true }).first().waitFor({ state: "visible", timeout: 30000 });
      await tbPage.getByText("Show data download links", { exact: true }).click({ timeout: 30000 });
    }],
    [mlflowPage, "MLflow-artifacts", async () => {
      await mlflowPage.getByRole("tab", { name: "Artifacts", exact: true }).click({ timeout: 30000 });
      const folder = mlflowPage.getByText("evaluation", { exact: true }).first();
      await folder.click({ timeout: 30000 });
      await mlflowPage.getByText("frame_000000000064.mp4", { exact: true }).first().waitFor({ state: "visible", timeout: 30000 });
    }],
  ]) {
    await page.bringToFront();
    let primary;
    try { await open(); } catch (error) { primary = error; }
    try { await snapshot(page, label); } catch (error) { if (!primary) primary = error; }
    if (primary) throw primary;
  }
  return { scope: "native scalar/artifact panels opened only; scalar history and downloaded playback UNVERIFIED" };
}

async function main() {
  const [output, manifestPath, mode] = process.argv.slice(2);
  if (process.argv.length !== 4 && !(process.argv.length === 5 && ["--guard-forced-exit", "--both-loggers", "--partial-delivery"].includes(mode))) {
    throw new Error("Usage: physical_ai_logger_browser.cjs OWNED_OUTPUT_DIR LOGGER_MANIFEST.json [--guard-forced-exit|--both-loggers|--partial-delivery]");
  }
  const failureMode = mode === "--partial-delivery";
  const loggerMode = failureMode ? "partial-delivery" : mode === "--both-loggers" ? "both" : "separate";
  const tbMode = loggerMode === "both" ? "both" : "tb-only";
  const mlMode = loggerMode === "both" ? "both" : "mlflow-only";
  const directory = path.resolve(output);
  const stat = fs.lstatSync(directory);
  if (!stat.isDirectory() || fs.realpathSync(directory) !== directory || stat.uid !== process.getuid()) {
    throw new Error("Output must be a caller-created owned real directory");
  }
  const files = new Map();
  const errors = { pageErrors: [], requestFailures: [], blockedURLs: [], cleanupErrors: [], diagnosticErrors: [], consoleMessages: [], httpErrors: [] };
  const pageErrors = [];
  const probe = { status: "FAILED", claim: "UNVERIFIED", logger_mode: loggerMode, pages: [],
    preview_selection: "UNVERIFIED", playback: "UNVERIFIED", scalar_inspection: "UNVERIFIED",
    artifact_downloads: "UNVERIFIED", overview_inspection: "UNVERIFIED", V_gate: "UNVERIFIED", C_gate: "UNVERIFIED",
    L_gate: "UNVERIFIED", R_gate: "UNVERIFIED", D_gate: "UNVERIFIED",
    tasks_6_1_6_2: "UNVERIFIED; reconnaissance does not qualify either task",
    route_coverage: "Context HTTP request routing only; WebSockets not intercepted. Background flags are not universal egress proof.",
    graphics_scope: "Browser software graphics only; no server RTX/sensor qualification" };
  let browser, context, tbPage, mlflowPage, tracing = false, primary;
  try {
    for (const name of ["probe.json", "TB.html", "TB.json", "TB.png", "MLflow.html", "MLflow.json", "MLflow.png", "MLflow-overview.html", "MLflow-overview.json", "MLflow-overview.png", "TB-scalars.html", "TB-scalars.json", "TB-scalars.png", "MLflow-artifacts.html", "MLflow-artifacts.json", "MLflow-artifacts.png", "MLflow-media.html", "MLflow-media.json", "MLflow-media.png", "trace.zip", "errors.json"]) {
      files.set(name, fs.openSync(path.join(directory, name), "wx", 0o600));
    }
    const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
    const identity = failureMode ? validateFailureIdentity(manifest) : validateRunIdentity(manifest, loggerMode);
    probe.native_identities = identity;
    const venv = process.env.UV_PROJECT_ENVIRONMENT;
    if (!venv || !path.isAbsolute(venv)) throw new Error("UV_PROJECT_ENVIRONMENT is required");
    const runtime = path.join(venv, "browser-runtime");
    const setup = JSON.parse(fs.readFileSync(path.join(runtime, "setup.json"), "utf8"));
    if (setup.status !== "success") throw new Error("Successful browser-runtime/setup.json required");
    const paths = setup.paths;
    for (const key of ["node", "playwrightModulePath", "chromeBinary", "fontsConf", "fontCache", "homePath"]) {
      if (typeof paths?.[key] !== "string" || !paths[key].startsWith(`${runtime}/`) ||
          !fs.realpathSync(paths[key]).startsWith(`${runtime}/`)) throw new Error(`Non-private setup path: ${key}`);
    }
    if (fs.realpathSync(process.execPath) !== fs.realpathSync(paths.node)) throw new Error("Use receipt paths.node to run this script");
    const prefix = `${runtime}/clients/usr/lib/x86_64-linux-gnu:${runtime}/clients/lib/x86_64-linux-gnu:${venv}/native-runtime/usr/lib/x86_64-linux-gnu`;
    if (paths.libraryPrefix !== prefix) throw new Error("Unexpected private libraryPrefix");
    const { chromium } = require(paths.playwrightModulePath);
    browser = await chromium.launch({ headless: true, executablePath: paths.chromeBinary, chromiumSandbox: true,
      env: { ...process.env, LD_LIBRARY_PATH: prefix + (process.env.LD_LIBRARY_PATH ? `:${process.env.LD_LIBRARY_PATH}` : ""),
        HOME: paths.homePath, FONTCONFIG_FILE: paths.fontsConf, XDG_CACHE_HOME: paths.fontCache },
      args: ["--disable-background-networking", "--disable-component-update", "--disable-client-side-phishing-detection",
        "--disable-sync", "--disable-breakpad", "--no-first-run", "--disable-dev-shm-usage", "--disable-gpu", "--no-proxy-server"] });
    probe.browser_version = browser.version();
    if (mode === "--guard-forced-exit") {
      fs.writeSync(1, "NEXUML_BROWSER_LAUNCHED=" + JSON.stringify({ version: browser.version(), pid: process.pid }) + "\n");
      process.kill(process.pid, "SIGKILL");
      await new Promise(() => {}); // Only the dedicated guard may clean the detached browser here.
    }
    context = await browser.newContext({ viewport: { width: 1280, height: 900 }, serviceWorkers: "block", acceptDownloads: true });
    await context.route("**/*", async (route) => {
      const url = route.request().url();
      if (allowedRequest(url)) await route.continue();
      else { errors.blockedURLs.push(safeLogURL(url)); await route.abort("blockedbyclient"); }
    });
    // No WebSocket API is assumed: only HTTP route coverage is claimed above.
    await context.tracing.start({ screenshots: true, snapshots: true }); tracing = true;
    for (const [label, url] of [
      ...(!failureMode ? [["TB", "http://127.0.0.1:6007/#images"]] : []),
      ["MLflow", `http://127.0.0.1:5001/#/experiments/${identity.mlflow.experiment_id}/runs/${identity.mlflow.run_id}`],
    ]) {
      const page = await context.newPage();
      const outstanding = new Set();
      page.setDefaultTimeout(30000);
      page.on("pageerror", (error) => { pageErrors.push(error); errors.pageErrors.push(String(error.stack || error)); });
      page.on("request", request => outstanding.add(request));
      page.on("requestfinished", request => outstanding.delete(request));
      page.on("requestfailed", request => { outstanding.delete(request); errors.requestFailures.push({ url: safeLogURL(request.url()), error: request.failure()?.errorText }); });
      page.on("response", response => { if (response.status() >= 400) errors.httpErrors.push({ url: safeLogURL(response.url()), status: response.status() }); });
      page.on("console", message => { if (["error", "warning"].includes(message.type())) errors.consoleMessages.push({ label, type: message.type(), text: message.text() }); });
      let response;
      try {
        response = await page.goto(url, { waitUntil: "domcontentloaded", timeout: 30000 });
        if (!response || response.status() < 200 || response.status() >= 300) throw new Error(`${label} navigation non-2xx: ${response?.status()}`);
        await waitForNativePage(page, label, identity);
      } catch (error) {
        try { await savePageSnapshot(page, label, files); }
        catch (diagnostic) { errors.diagnosticErrors.push({ label, error: String(diagnostic.stack || diagnostic) }); }
        throw error;
      } finally {
        probe[`${label}_unfinished_requests`] = [...outstanding].map(request => ({ url: safeLogURL(request.url()), type: request.resourceType() }));
      }
      await savePageSnapshot(page, label, files);
      if (pageErrors.length) throw pageErrors[0];
      probe.pages.push({ label, url, http_status: response.status(), claim: "UI_OPENED_ONLY" });
      if (label === "TB") tbPage = page;
      else mlflowPage = page;
    }
    if (failureMode) {
      const failureDirectory = path.join(directory, "mlflow-failure");
      fs.mkdirSync(failureDirectory, { mode: 0o700 });
      const { runMLflowFailure } = require("./physical_ai_mlflow_failure.cjs");
      let failureError;
      try { probe.mlflow_failure = await runMLflowFailure(mlflowPage, manifest, failureDirectory); }
      catch (error) { failureError = error; }
      try { await savePageSnapshot(mlflowPage, "MLflow-media", files); }
      catch (error) {
        errors.diagnosticErrors.push({ label: "MLflow-failure", error: String(error.stack || error) });
        if (!failureError) failureError = error;
      }
      if (failureError) throw failureError;
    } else {
      await mlflowPage.bringToFront();
      const overviewDirectory = path.join(directory, "mlflow-overview");
      fs.mkdirSync(overviewDirectory, { mode: 0o700 });
      const { runMLflowOverview } = require("./physical_ai_mlflow_overview.cjs");
      let overviewError;
      try {
        probe.mlflow_overview = await runMLflowOverview(mlflowPage, manifest, overviewDirectory, mlMode);
        probe.overview_inspection = "PASS";
      } catch (error) { overviewError = error; }
      try { await savePageSnapshot(mlflowPage, "MLflow-overview", files); }
      catch (error) {
        errors.diagnosticErrors.push({ label: "MLflow-overview", error: String(error.stack || error) });
        if (!overviewError) overviewError = error;
      }
      if (overviewError) throw overviewError;
      await tbPage.bringToFront();
      const previewDirectory = path.join(directory, "tensorboard-previews");
      fs.mkdirSync(previewDirectory, { mode: 0o700 });
      const { runTensorBoardPreviews } = require("./physical_ai_tensorboard_previews.cjs");
      probe.tensorboard_previews = await runTensorBoardPreviews(tbPage, manifest, previewDirectory, tbMode);
      if (pageErrors.length) throw pageErrors[0];
      probe.preview_selection = "PASS";
      probe.playback = "PASS (native TB GIF only; MP4 UNVERIFIED)";
      probe.native_panels = await inspectNativePanels(tbPage, mlflowPage, async (page, label) => {
        await savePageSnapshot(page, label, files);
        if (label === "TB-scalars") {
          const output = path.join(directory, "tensorboard-scalars");
          fs.mkdirSync(output, { mode: 0o700 });
          const { runTensorBoardScalars } = require("./physical_ai_tensorboard_scalars.cjs");
          probe.tensorboard_scalars = await runTensorBoardScalars(page, manifest, output, tbMode);
          probe.scalar_inspection = "PASS";
        }
      });
      await mlflowPage.bringToFront();
      const artifactDirectory = path.join(directory, "mlflow-artifacts");
      fs.mkdirSync(artifactDirectory, { mode: 0o700 });
      const { runMLflowArtifacts } = require("./physical_ai_mlflow_artifacts.cjs");
      let artifactError;
      try {
        probe.mlflow_artifacts = await runMLflowArtifacts(mlflowPage, manifest, artifactDirectory, mlMode);
        probe.artifact_downloads = `PASS (${mlMode} native artifact route; other modes UNVERIFIED)`;
        probe.playback = `PASS (native TB GIF and ${mlMode} GIF/MP4; full V UNVERIFIED)`;
      } catch (error) { artifactError = error; }
      try { await savePageSnapshot(mlflowPage, "MLflow-media", files); }
      catch (error) {
        errors.diagnosticErrors.push({ label: "MLflow-media", error: String(error.stack || error) });
        if (!artifactError) artifactError = error;
      }
      if (artifactError) throw artifactError;
    }
    if (pageErrors.length) throw pageErrors[0];
    if (tbPage) await tbPage.close();
    await mlflowPage.close();
    probe.status = "success"; probe.claim = "UI_OPENED_ONLY";
  } catch (error) { primary = pageErrors[0] || error; }
  finally {
    for (const [label, cleanup] of [
      // trace.zip was exclusively reserved above; Playwright fills only our placeholder.
      ["trace", async () => { if (tracing) await context.tracing.stop({ path: path.join(directory, "trace.zip") }); }],
      ["context", async () => { if (context) await context.close(); }],
      ["browser", async () => { if (browser) await browser.close(); }],
    ]) {
      try { await cleanup(); }
      catch (error) { errors.cleanupErrors.push({ label, error: String(error.stack || error) }); if (!primary) primary = error; }
    }
    if (pageErrors.length && !primary) primary = pageErrors[0];
    if (primary) { probe.status = "FAILED"; probe.claim = "UNVERIFIED"; errors.primaryError = String(primary.stack || primary); }
    for (const [name, data] of [["probe.json", probe], ["errors.json", errors]]) {
      try { if (files.has(name)) fs.writeFileSync(files.get(name), JSON.stringify(data, null, 2)); }
      catch (error) { console.error("Secondary receipt error:", error); if (!primary) primary = error; }
    }
    for (const fd of files.values()) {
      try { fs.closeSync(fd); } catch (error) { console.error("Secondary close error:", error); if (!primary) primary = error; }
    }
  }
  if (primary) throw primary;
}

module.exports = { allowedRequest, safeLogURL, validateRunIdentity, validateFailureIdentity, waitForNativePage, inspectNativePanels };
if (require.main === module) main().catch((error) => { console.error(error); process.exitCode = 1; });
