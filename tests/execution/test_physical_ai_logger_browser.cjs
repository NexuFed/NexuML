const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");
const SCRIPT = path.join(__dirname, "../fixtures/physical_ai_logger_browser.cjs");
const { allowedRequest, safeLogURL, validateRunIdentity, validateFailureIdentity, waitForNativePage, inspectNativePanels } = require(SCRIPT);

test("only exact credential-free fixed HTTP loopback origins are allowed", () => {
  for (const url of ["http://127.0.0.1:6007/", "http://127.0.0.1:5001/api?x=1"]) assert.equal(allowedRequest(url), true);
  for (const url of ["http://127.0.0.1:6008/", "http://localhost:6007/", "http://127.1:6007/",
    "http://127.0.0.2:6007/", "http://127.0.0.1.evil:6007/", "http://[::1]:6007/",
    "https://127.0.0.1:6007/", "ws://127.0.0.1:5001/", "file:///tmp/secret", "data:text/plain,x",
    "http://user:password@127.0.0.1:6007/", "http://127.0.0.1:6007@evil/", "not a URL", null]) {
    assert.equal(allowedRequest(url), false, String(url));
  }
});

test("logged URLs remove credentials, queries and fragments", () => {
  assert.equal(safeLogURL("https://alice:secret@example.test/x?token=private#private"), "https://example.test/x");
  assert.equal(safeLogURL("http://127.0.0.1:6007/a?q=1"), "http://127.0.0.1:6007/a");
  assert.equal(safeLogURL("bad token=secret"), "[invalid URL]");
});

const manifest = () => ({ modes: {
  "tb-only": { native: { tensorboard: { main: { log_dir: "/owned/logs/tb/train", image_tag: "eval/rl/episode" } } } },
  "mlflow-only": { native: { mlflow: { main: { experiment_id: "42", run_id: "0123456789abcdef0123456789abcdef" } } } },
} });

test("identities come only from the two native reconnaissance manifest modes", () => {
  assert.deepEqual(validateRunIdentity(manifest()), {
    tensorboard: { log_dir: "/owned/logs/tb/train", image_tag: "eval/rl/episode" },
    mlflow: { experiment_id: "42", run_id: "0123456789abcdef0123456789abcdef" },
  });
  for (const [field, value] of [["experiment_id", 42], ["experiment_id", "../42"], ["experiment_id", "42\n"],
    ["run_id", "abc"], ["run_id", "G".repeat(32)], ["run_id", "a".repeat(32) + "/x"]]) {
    const input = manifest(); input.modes["mlflow-only"].native.mlflow.main[field] = value;
    assert.throws(() => validateRunIdentity(input), /native identities/);
  }
  const input = manifest(); input.modes.model_run = input.modes["tb-only"]; delete input.modes["tb-only"];
  assert.throws(() => validateRunIdentity(input), /native identities/);
  assert.throws(() => validateRunIdentity({}), /native identities/);
});

test("both mode selects its own native identities without single-logger fallback", () => {
  const input = manifest();
  input.modes.both = { native: {
    tensorboard: { main: { log_dir: "/owned/both/logs/tb/run", image_tag: "eval/rl/episode" } },
    mlflow: { main: { experiment_id: "1", run_id: "e58df20bab2e4b7c9fb121b36320b9a6" } },
  } };
  const value = validateRunIdentity(input, "both");
  assert.equal(value.tensorboard.log_dir, input.modes.both.native.tensorboard.main.log_dir);
  assert.equal(value.mlflow.run_id, input.modes.both.native.mlflow.main.run_id);
  delete input.modes.both;
  assert.throws(() => validateRunIdentity(input, "both"), /native identities/);
  const untouched = new Proxy({}, { get() { assert.fail("mode must reject before manifest access"); } });
  for (const mode of ["disabled", "tb-only", "unknown", null, {}, new String("both")]) {
    assert.throws(() => validateRunIdentity(untouched, mode), /browser mode/);
  }
});

test("static recon labels cannot claim UI controls or gate acceptance", () => {
  const source = fs.readFileSync(SCRIPT, "utf8");
  assert.match(source, /claim: "UI_OPENED_ONLY"/);
  for (const label of ["preview_selection", "playback", "scalar_inspection", "artifact_downloads", "overview_inspection", "V_gate", "C_gate", "L_gate", "R_gate", "D_gate"]) {
    assert.match(source, new RegExp(`${label}: "UNVERIFIED"`));
  }
  assert.match(source, /reconnaissance does not qualify either task/);
  assert.deepEqual(Object.keys(require(SCRIPT)).sort(), ["allowedRequest", "inspectNativePanels", "safeLogURL", "validateFailureIdentity", "validateRunIdentity", "waitForNativePage"]);
});

test("page readiness uses rendered native identity, not a quiet telemetry worker", async () => {
  const calls = [];
  const visible = name => ({ first() { return this; }, async waitFor(options) {
    assert.deepEqual(options, { state: "visible", timeout: 30000 }); calls.push(name);
  }, async evaluate(fn) { calls.push("decoded-image"); await fn({ decode: async () => {}, naturalWidth: 128, naturalHeight: 128 }); } });
  const page = {
    getByText(text, options) {
      if (String(text) === "/MLflow/i") throw new Error("generic brand locator matched hidden page title");
      return visible([String(text), options]);
    },
    locator(selector) { assert.equal(selector, 'img[src*="data/plugin/images/individualImage"]'); return visible("image"); },
    async waitForLoadState() { throw new Error("telemetry worker never becomes network-idle"); },
  };
  const identity = validateRunIdentity(manifest());
  await waitForNativePage(page, "MLflow", identity);
  assert.deepEqual(calls, [[identity.mlflow.run_id, { exact: true }], ["Finished", { exact: true }]]);
  calls.length = 0;
  await waitForNativePage(page, "TB", identity);
  assert.deepEqual(calls, [["/TensorBoard/i", undefined], "image", "decoded-image"]);
});

test("missing native identity or broken preview remains a failure", async () => {
  const failure = new Error("expected run identity is not visible");
  const page = { getByText() { return { first() { return this; }, async waitFor() { throw failure; } }; } };
  await assert.rejects(waitForNativePage(page, "MLflow", validateRunIdentity(manifest())), error => error === failure);
  const rendered = { first() { return this; }, async waitFor() {}, async evaluate(fn) {
    await fn({ decode: async () => {}, naturalWidth: 0, naturalHeight: 0 });
  } };
  await assert.rejects(waitForNativePage({ getByText: () => rendered, locator: () => rendered }, "TB", validateRunIdentity(manifest())), /decoded/);
});

test("page snapshot retains DOM and image evidence even when called after readiness fails", async () => {
  const written = new Map();
  const sandbox = { require(name) { return name === "node:fs" ? { writeFileSync: (fd, data) => written.set(fd, data) } : require(name); }, module: { exports: {} } };
  vm.runInNewContext(fs.readFileSync(SCRIPT, "utf8") + "\nthis.capture = savePageSnapshot;", sandbox);
  const files = new Map(["MLflow.html", "MLflow.json", "MLflow.png"].map(name => [name, name]));
  const page = { async evaluate() { return { bodyText: "", elements: [] }; }, async content() { return "<body>loading failed</body>"; }, async screenshot() { return Buffer.from("failure pixels"); } };
  await assert.rejects(sandbox.capture(page, "MLflow", files), /expected page is empty/);
  assert.equal(written.get("MLflow.html"), "<body>loading failed</body>");
  assert.equal(JSON.parse(written.get("MLflow.json")).bodyText, "");
  assert.equal(written.get("MLflow.png").toString(), "failure pixels");
});

test("native scalar/artifact tabs use observed controls and preserve failed-panel evidence", async () => {
  const actions = [];
  let failure;
  let trainExpanded = false;
  const page = label => ({
    async bringToFront() { actions.push(`${label}:foreground`); },
    getByRole(role, options) {
      assert(["tab", "button"].includes(role));
      return { async getAttribute(name) { assert.equal(name, "open-button"); return trainExpanded ? "" : null; }, async click() { if (role === "button") trainExpanded = true; actions.push(`${label}:${role}:${options.name}`); } };
    },
    locator(selector) { assert.equal(selector, "tf-scalar-dashboard"); return { async waitFor() {} }; },
    getByLabel() { assert.fail("generic panel probe must not preselect/cache artifacts before native response observation"); },
    getByText(text, options) { assert.equal(options.exact, true); return { first() { return this; }, async click() { actions.push(`${label}:folder:${text}`); }, async waitFor() { if (failure) throw failure; if (text === "train/rl/frames" && !trainExpanded) throw new Error("train category is still collapsed"); actions.push(`${label}:visible:${text}`); } }; },
  });
  const snapshot = async (_, label) => actions.push(`snapshot:${label}`);
  const result = await inspectNativePanels(page("TB"), page("MLflow"), snapshot);
  assert.match(result.scope, /downloaded playback UNVERIFIED/);
  assert(actions.includes("MLflow:folder:evaluation"));
  assert(actions.includes("MLflow:visible:frame_000000000064.mp4"));
  assert(actions.includes("snapshot:TB-scalars") && actions.includes("snapshot:MLflow-artifacts"));
  assert(!actions.some(action => action.includes(":file:")));
  actions.length = 0;
  failure = new Error("native scalar panel failed");
  await assert.rejects(inspectNativePanels(page("TB"), page("MLflow"), snapshot), error => error === failure);
  assert.equal(actions.at(-1), "snapshot:TB-scalars");
  assert(!actions.includes("MLflow:foreground"));
});

test("forced parent death never weakens the pinned browser sandbox", () => {
  const source = fs.readFileSync(SCRIPT, "utf8");
  assert.match(source, /chromiumSandbox: true/);
  assert.match(source, /--no-proxy-server/);
  assert.match(source, /mode === "--guard-forced-exit"/);
  assert.match(source, /fs\.writeSync\(1, "NEXUML_BROWSER_LAUNCHED="/);
  assert.match(source, /process\.kill\(process\.pid, "SIGKILL"\)/);
  assert.doesNotMatch(source, /--no-sandbox|--enable-unsafe-swiftshader/);
});

test("Overview proof runs with selected mode before leaving its native page", () => {
  const source = fs.readFileSync(SCRIPT, "utf8");
  const overview = source.indexOf('await runMLflowOverview(mlflowPage, manifest, overviewDirectory, mlMode)');
  const panels = source.indexOf('await inspectNativePanels(tbPage, mlflowPage');
  assert(overview > 0 && overview < panels);
  assert.match(source, /if \(overviewError\) throw overviewError;/);
  assert.match(source, /savePageSnapshot\(mlflowPage, "MLflow-overview", files\)/);
});

test("failed-run URL identity requires native FAILED and original CLI exit1", () => {
  const input = { status: "FAILED", cli: { status: "FAIL", exit_code: 1 }, experiment_name: "physical-ai-logging-partial-delivery",
    run_id: "a".repeat(32), experiment_id: "1" };
  assert.deepEqual(validateFailureIdentity(input), { mlflow: { status: "Failed", run_id: "a".repeat(32), experiment_id: "1" } });
  for (const replacement of [{ status: "FINISHED" }, { cli: { status: "PASS", exit_code: 0 } }, { run_id: "../other" }, { experiment_id: "1\n" }]) {
    assert.throws(() => validateFailureIdentity({ ...input, ...replacement }), /failed-run identity/);
  }
});

for (const fail of [false, true]) test(`failed-run main branch uses only MLflow and preserves lifecycle/errors (${fail})`, async () => {
  const actions = [], written = new Map(), descriptors = new Map();
  const manifest = { status: "FAILED", cli: { status: "FAIL", exit_code: 1 }, experiment_name: "physical-ai-logging-partial-delivery",
    run_id: "a".repeat(32), experiment_id: "1" };
  const runtime = "/private/browser-runtime", node = `${runtime}/node`;
  const paths = Object.fromEntries(["node", "playwrightModulePath", "chromeBinary", "fontsConf", "fontCache", "homePath"].map(key => [key, `${runtime}/${key}`]));
  paths.libraryPrefix = `${runtime}/clients/usr/lib/x86_64-linux-gnu:${runtime}/clients/lib/x86_64-linux-gnu:/private/native-runtime/usr/lib/x86_64-linux-gnu`;
  const page = { setDefaultTimeout() {}, on() {}, async goto(url) { actions.push(url); return { status: () => 200 }; },
    getByText(text) { return { async waitFor() { actions.push(text); } }; },
    async evaluate() { return { bodyText: "native failure observer mock", elements: [] }; },
    async content() { return "<body>native failure observer mock</body>"; },
    async screenshot() { return Buffer.from("mock PNG"); }, async close() { actions.push("page-close"); } };
  const context = { async route() {}, tracing: { async start() {}, async stop() { actions.push("trace-close"); } },
    async newPage() { actions.push("new-page"); return page; }, async close() { actions.push("context-close"); } };
  const browser = { version: () => "153.0.8010.12", async newContext() { return context; }, async close() { actions.push("browser-close"); } };
  const fakeFS = { lstatSync: () => ({ isDirectory: () => true, uid: 1000 }), realpathSync: p => p,
    openSync(p) { const fd = descriptors.size + 1; descriptors.set(fd, p); return fd; }, closeSync() {},
    mkdirSync(p, options) { assert.equal(options.mode, 0o700); actions.push(p); },
    readFileSync(p) { return JSON.stringify(p.endsWith("setup.json") ? { status: "success", paths } : manifest); },
    writeFileSync(fd, value) { written.set(descriptors.get(fd), value); } };
  const original = new Error("native failed-run observation error");
  const sandbox = { process: { argv: [node, SCRIPT, "/owned", "/manifest.json", "--partial-delivery"], execPath: node,
    env: { UV_PROJECT_ENVIRONMENT: "/private" }, getuid: () => 1000 }, console, Buffer, module: { exports: {} },
    require(name) {
      if (name === "node:fs") return fakeFS;
      if (name === paths.playwrightModulePath) return { chromium: { async launch(options) { assert(options.chromiumSandbox); return browser; } } };
      if (name === "./physical_ai_mlflow_failure.cjs") return { async runMLflowFailure(actualPage, input, output) {
        assert.strictEqual(actualPage, page); assert.equal(input.run_id, manifest.run_id); assert.equal(output, "/owned/mlflow-failure");
        if (fail) throw original;
        return { status: { actual: "Failed" }, run_id: input.run_id };
      } };
      assert(!name.startsWith("./physical_ai_"), "failed-run observer must not start a normal logger journey");
      return require(name);
    } };
  vm.runInNewContext(fs.readFileSync(SCRIPT, "utf8") + "\nthis.run = main;", sandbox);
  if (fail) await assert.rejects(sandbox.run(), error => error === original);
  else await sandbox.run();
  assert.equal(actions.filter(value => value === "new-page").length, 1);
  assert(!actions.some(value => String(value).includes(":6007")));
  assert(actions.includes("Failed") && !actions.includes("Finished"));
  assert(actions.includes("trace-close") && actions.includes("context-close") && actions.includes("browser-close"));
  const probe = JSON.parse(written.get("/owned/probe.json"));
  assert.equal(probe.status, fail ? "FAILED" : "success");
  assert.equal(probe.V_gate, "UNVERIFIED");
  assert.equal(probe.logger_mode, "partial-delivery");
});
