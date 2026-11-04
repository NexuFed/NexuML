// Hermetic helper checks only: no child processes, browser, network or GPU.
const assert = require("node:assert/strict");
const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");
const childProcess = require("node:child_process");
const { EventEmitter } = require("node:events");
const vm = require("node:vm");
const { test } = require("node:test");
const { allocateIpc, readProgress, spawnNative, cleanup, verifyCompletion } = require("./isaaclab_viser_cli_browser.cjs");
const controllerSource = fs.readFileSync(path.join(__dirname, "isaaclab_viser_cli_browser.cjs"), "utf8");

function mockIpcFs(t) {
  const previous = process.env.TMPDIR;
  process.env.TMPDIR = "/tmp/ray";
  t.after(() => {
    if (previous === undefined) delete process.env.TMPDIR;
    else process.env.TMPDIR = previous;
  });
  const calls = [];
  t.mock.method(fs, "realpathSync", (name) => { calls.push(["realpath", name]); return name; });
  t.mock.method(fs, "statfsSync", (name) => { calls.push(["statfs", name]); return { type: 0xEF53 }; });
  t.mock.method(fs, "mkdtempSync", (prefix) => {
    calls.push(["mkdtemp", prefix]);
    return `${prefix}owned`;
  });
  t.mock.method(fs, "writeFileSync", (name, data, options) => {
    calls.push(["write", name, JSON.parse(data), options]);
  });
  t.mock.method(fs, "rmSync", (name, options) => calls.push(["remove", name, options]));
  return calls;
}

test("IPC allocation uses only a fresh mode-specific ext4 directory and exclusive durable receipt", (t) => {
  const calls = mockIpcFs(t);
  const ipc = allocateIpc("/owned/out", "replay-viewer");
  assert.equal(ipc.path, "/tmp/ray/nexuml-spectator-replay-viewer-owned");
  assert.deepEqual(calls, [
    ["realpath", "/tmp/ray"],
    ["statfs", "/tmp/ray"],
    ["mkdtemp", "/tmp/ray/nexuml-spectator-replay-viewer-"],
    ["write", "/owned/out/live-ipc.json", {
      path: ipc.path, type: 0xEF53, storage: "worker-local ext4", mode: "replay-viewer",
      scope: "Transient live IPC only; not checkpoint or evidence storage",
      cleanup: { status: "pending" },
    }, { flag: "wx", mode: 0o600 }],
  ]);
});

test("IPC allocation rejects missing, relative, shared-memory, symlink and non-ext4 roots", async (t) => {
  for (const name of ["missing", "relative", "shared-memory", "symlink", "symlink-alias", "FUSE", "tmpfs"]) {
    await t.test(name, (t) => {
      const calls = mockIpcFs(t);
      if (name === "missing") delete process.env.TMPDIR;
      if (name === "relative") process.env.TMPDIR = "tmp/ray";
      if (name === "shared-memory") process.env.TMPDIR = "/dev/shm";
      if (name === "symlink") t.mock.method(fs, "realpathSync", () => "/disk/ray");
      if (name === "symlink-alias") {
        process.env.TMPDIR = "/tmp/ray-link";
        t.mock.method(fs, "realpathSync", () => "/tmp/ray");
      }
      if (name === "FUSE") t.mock.method(fs, "statfsSync", () => ({ type: 0x65735546 }));
      if (name === "tmpfs") t.mock.method(fs, "statfsSync", () => ({ type: 0x01021994 }));
      assert.throws(() => allocateIpc("/owned/out", "train-viewer"), /Live IPC requires/);
      assert.ok(calls.every(([call]) => ["realpath", "statfs"].includes(call)));
    });
  }
});

test("exclusive receipt conflict removes only fresh IPC, never the existing receipt", (t) => {
  const calls = mockIpcFs(t);
  const error = Object.assign(new Error("existing receipt"), { code: "EEXIST" });
  t.mock.method(fs, "writeFileSync", () => { throw error; });
  assert.throws(() => allocateIpc("/owned/out", "train-viewer"), (actual) => actual === error);
  assert.deepEqual(calls.filter(([call]) => call === "remove"), [
    ["remove", "/tmp/ray/nexuml-spectator-train-viewer-owned", { recursive: true }],
  ]);
});

test("progress reads exactly the owned IPC path with strict JSON and no fallback", (t) => {
  const calls = [];
  const ipc = "/tmp/ray/nexuml-spectator-train-viewer-owned";
  const read = t.mock.method(fs, "readFileSync", (name, encoding) => {
    calls.push([name, encoding]);
    return '{"steps":3}';
  });
  assert.deepEqual(readProgress(ipc), { steps: 3 });
  assert.deepEqual(calls, [[`${ipc}/progress.json`, "utf8"]]);
  const missing = Object.assign(new Error("missing progress"), { code: "ENOENT" });
  read.mock.mockImplementation(() => { throw missing; });
  assert.throws(() => readProgress(ipc), (error) => error === missing);
  assert.equal(read.mock.callCount(), 2);
  read.mock.mockImplementation(() => '{"steps":');
  const parse = JSON.parse;
  let malformed;
  t.mock.method(JSON, "parse", (text) => {
    try { return parse(text); }
    catch (error) { malformed = error; throw error; }
  });
  assert.throws(() => readProgress(ipc), (error) => error === malformed && error instanceof SyntaxError);
  assert.equal(read.mock.callCount(), 3);
});

test("IPC removal follows confirmed child exit even when browser close fails", async (t) => {
  const calls = mockIpcFs(t);
  const ipc = allocateIpc("/owned/out", "train-viewer");
  calls.length = 0;
  const error = new Error("browser close failed");
  t.mock.method(fs, "closeSync", (fd) => calls.push(["close", fd]));
  const browser = { close: async () => { calls.push(["browser"]); throw error; } };
  const child = { exitCode: null, signalCode: null, kill: (signal) => calls.push(["kill", signal]) };
  const waitExit = async () => { calls.push(["exit"]); child.signalCode = "SIGTERM"; };
  await assert.rejects(cleanup(browser, child, waitExit, Promise.resolve(), 9, ipc), (actual) => actual === error);
  assert.deepEqual(calls, [
    ["browser"], ["kill", "SIGTERM"], ["exit"],
    ["remove", ipc.path, { recursive: true }],
    ["write", "/owned/out/live-ipc.json", {
      path: ipc.path, type: 0xEF53, storage: "worker-local ext4", mode: "train-viewer",
      scope: "Transient live IPC only; not checkpoint or evidence storage",
      cleanup: { status: "removed", exit_code: null, signal: "SIGTERM" },
    }, { mode: 0o600 }],
    ["close", 9],
  ]);
});

test("IPC cleanup accepts an already exited child and waits for SIGKILL completion", async (t) => {
  for (const alreadyExited of [true, false]) {
    await t.test(String(alreadyExited), async (t) => {
      const calls = mockIpcFs(t);
      const ipc = allocateIpc("/owned/out", "cancel-viewer");
      calls.length = 0;
      t.mock.method(fs, "closeSync", (fd) => calls.push(["close", fd]));
      const child = { exitCode: alreadyExited ? 143 : null, signalCode: null,
        kill: (signal) => calls.push(["kill", signal]) };
      const completed = { then(resolve) {
        calls.push(["exit"]); child.signalCode = "SIGKILL"; resolve();
      } };
      await cleanup(null, child, async () => { throw new Error("deadline"); }, completed, 10, ipc);
      assert.deepEqual(calls.map(([call]) => call), alreadyExited
        ? ["remove", "write", "close"] : ["kill", "kill", "exit", "remove", "write", "close"]);
      assert.deepEqual(calls.find(([call]) => call === "remove"), ["remove", ipc.path, { recursive: true }]);
      assert.equal(calls.find(([call]) => call === "write")[2].cleanup.status, "removed");
    });
  }
});

test("unconfirmed child retains IPC and closes the descriptor, preserving termination errors", async (t) => {
  for (const failure of ["unconfirmed wait", "kill error", "completion error"]) {
    await t.test(failure, async (t) => {
      const calls = mockIpcFs(t);
      const ipc = allocateIpc("/owned/out", "failure-viewer");
      calls.length = 0;
      t.mock.method(fs, "closeSync", (fd) => calls.push(["close", fd]));
      const error = new Error(failure);
      const child = { exitCode: null, signalCode: null, kill() {
        if (failure === "kill error") throw error;
      } };
      const waitExit = async () => { if (failure === "completion error") throw error; };
      const completed = { then(resolve, reject) { reject(error); } };
      const result = cleanup(null, child, waitExit, completed, 11, ipc);
      if (failure === "unconfirmed wait") await result;
      else await assert.rejects(result, (actual) => actual === error);
      assert.deepEqual(calls.map(([call]) => call), ["write", "close"]);
      assert.deepEqual(calls[0][2].cleanup, { status: "retained", reason: "Owned child exit is unconfirmed" });
      assert.deepEqual(calls[1], ["close", 11]);
    });
  }
});

test("IPC removal failure records incomplete cleanup and still closes the descriptor", async (t) => {
  const calls = mockIpcFs(t);
  const ipc = allocateIpc("/owned/out", "train-viewer");
  calls.length = 0;
  const error = new Error("IPC removal failed");
  t.mock.method(fs, "rmSync", (name) => {
    assert.equal(name, ipc.path);
    throw error;
  });
  t.mock.method(fs, "closeSync", (fd) => calls.push(["close", fd]));
  await assert.rejects(cleanup(null, { exitCode: 0, signalCode: null }, null, null, 12, ipc),
    (actual) => actual === error);
  assert.deepEqual(calls.map(([call]) => call), ["write", "close"]);
  assert.equal(calls[0][2].cleanup.status, "failed");
  assert.deepEqual(calls[1], ["close", 12]);
});

test("IPC failures cannot replace an original browser or child cleanup failure", async (t) => {
  for (const phase of ["browser", "child"]) {
    await t.test(phase, async (t) => {
      const primary = new Error(`original ${phase} failure`);
      const storage = new Error("IPC receipt failed");
      const closed = [];
      t.mock.method(fs, "closeSync", (fd) => closed.push(fd));
      const child = phase === "browser"
        ? { exitCode: 0, signalCode: null }
        : { exitCode: null, signalCode: null, kill() { throw primary; } };
      const browser = phase === "browser" ? { async close() { throw primary; } } : null;
      await assert.rejects(cleanup(browser, child, null, null, 13, {
        cleanup() { throw storage; },
      }), (error) => error === primary && error.cleanupErrors.includes(storage));
      assert.deepEqual(closed, [13]);
    });
  }
});

test("failed allocation receipt removes only its fresh IPC and keeps the write error", (t) => {
  const calls = mockIpcFs(t);
  const primary = new Error("initial receipt failed");
  t.mock.method(fs, "writeFileSync", () => { throw primary; });
  assert.throws(() => allocateIpc("/owned/out", "train-viewer"), (error) => error === primary);
  const created = calls.find(([call]) => call === "mkdtemp");
  const removed = calls.find(([call]) => call === "remove");
  assert.ok(created && removed);
  assert.equal(removed[1], "/tmp/ray/nexuml-spectator-train-viewer-owned");
});

test("original progress assertion survives failing IPC cleanup and descriptor close", async (t) => {
  const primary = new Error("original progress ENOENT");
  const storage = new Error("cleanup receipt failed");
  const descriptor = new Error("descriptor close failed");
  t.mock.method(fs, "closeSync", () => { throw descriptor; });
  await assert.rejects(cleanup(null, { exitCode: 1, signalCode: null }, null, null, 14, {
    cleanup() { throw storage; },
  }, primary), (error) => error === primary);
  assert.deepEqual(primary.cleanupErrors, [storage, descriptor]);
});

test("IPC receipt failure cannot replace an original removal failure", (t) => {
  mockIpcFs(t);
  const ipc = allocateIpc("/owned/out", "train-viewer");
  const removal = new Error("remove failed");
  const storage = new Error("receipt failed");
  t.mock.method(fs, "rmSync", () => { throw removal; });
  t.mock.method(fs, "writeFileSync", () => { throw storage; });
  assert.throws(() => ipc.cleanup({ exitCode: 1, signalCode: null }), (error) => error === removal);
  assert.deepEqual(removal.cleanupErrors, [storage]);
});

test("failed native launch removes only fresh IPC when no process was created", async (t) => {
  for (const asynchronous of [false, true]) {
    await t.test(String(asynchronous), async (t) => {
      const calls = mockIpcFs(t);
      const ipc = allocateIpc("/owned/out", "train-viewer");
      const primary = new Error("native spawn failed");
      const child = Object.assign(new EventEmitter(), { pid: undefined, exitCode: null, signalCode: null });
      t.mock.method(childProcess, "spawn", () => {
        if (!asynchronous) throw primary;
        queueMicrotask(() => child.emit("error", primary));
        return child;
      });
      t.mock.method(fs, "closeSync", (fd) => calls.push(["close", fd]));
      let native, failure;
      try {
        native = spawnNative({ argv: ["/approved/python", "-m", "nexuml.cli.main"], env: {} }, 15, ipc);
        await native.completed;
        assert.equal(native.error, primary);
        failure = native.error;
      } catch (error) { failure = error; }
      assert.equal(failure, primary);
      await assert.rejects(cleanup(null, null, null, null, 15, ipc, failure), (error) => error === primary);
      assert.equal(calls.filter(([call]) => call === "remove").length, 1);
      assert.equal(calls.findLast(([call]) => call === "write")[2].cleanup.reason, "No native process was created");
      assert.deepEqual(calls.at(-1), ["close", 15]);
    });
  }
});

test("native child owns IPC env and an error with a live PID is not proof of exit", async (t) => {
  const calls = mockIpcFs(t);
  const ipc = allocateIpc("/owned/out", "train-viewer");
  const child = Object.assign(new EventEmitter(), { pid: 123, exitCode: null, signalCode: null });
  const launch = { argv: ["/approved/python", "-m", "nexuml.cli.main", "train"], env: { HOME: "native-home" } };
  const before = { ...process.env };
  t.mock.method(childProcess, "spawn", (executable, argv, options) => {
    assert.equal(executable, launch.argv[0]);
    assert.deepEqual(argv, launch.argv.slice(1));
    assert.deepEqual(options.stdio, ["ignore", 16, 16]);
    const expected = { ...before, ...launch.env, NEXUML_SPECTATOR_PROBE_IPC: ipc.path };
    assert.ok(Object.keys(options.env).length === Object.keys(expected).length &&
      Object.entries(expected).every(([key, value]) => options.env[key] === value), "native child environment differs");
    return child;
  });
  const native = spawnNative(launch, 16, ipc);
  assert.ok(Object.keys(process.env).length === Object.keys(before).length &&
    Object.entries(before).every(([key, value]) => process.env[key] === value), "parent environment changed");
  const primary = new Error("live child error");
  let settled = false;
  native.completed.then(() => { settled = true; });
  child.emit("error", primary);
  await Promise.resolve();
  assert.equal(settled, false);
  assert.equal(native.error, primary);
  ipc.cleanup(child);
  assert.ok(!calls.some(([call]) => call === "remove"));
  child.exitCode = 1;
  child.emit("exit", 1, null);
  assert.deepEqual(await native.completed, { code: 1, signal: null });
  ipc.cleanup(child);
  assert.equal(calls.filter(([call]) => call === "remove").length, 1);
});

test("controller launch ownership survives through its failure finally", async (t) => {
  const calls = mockIpcFs(t);
  const primary = new Error("injected browser startup failure");
  const child = Object.assign(new EventEmitter(), {
    pid: 123, exitCode: null, signalCode: null,
    kill(signal) {
      calls.push(["kill", signal]);
      this.signalCode = signal;
      this.emit("exit", null, signal);
    },
  });
  const paths = {
    playwrightModulePath: "mock:playwright", chromeBinary: "/approved/chrome", node: "/approved/node",
    libraryPrefix: "/approved/lib", homePath: "/approved/home", fontsConf: "/approved/fonts", fontCache: "/approved/cache",
  };
  t.mock.method(childProcess, "spawnSync", () => ({ status: 0, stdout: JSON.stringify({
    argv: ["/approved/python", "-m", "nexuml.cli.main", "train"], env: {},
  }) }));
  t.mock.method(childProcess, "spawn", () => child);
  t.mock.method(fs, "mkdirSync", () => {});
  t.mock.method(fs, "openSync", () => 17);
  t.mock.method(fs, "closeSync", (fd) => calls.push(["close", fd]));
  t.mock.method(fs, "existsSync", () => true);
  t.mock.method(fs, "readFileSync", () => JSON.stringify({ status: "success", paths }));
  const localModule = { exports: {} };
  vm.runInNewContext(`${controllerSource}\nmodule.exports.runController = main;`, {
    require(name) {
      return name === "mock:playwright"
        ? { chromium: { async launch() { calls.push(["browser-launch"]); throw primary; } } }
        : require(name);
    },
    module: localModule, __dirname,
    process: { argv: ["node", "fixture", "/owned/out", "train-viewer", "8081"], execPath: paths.node, env: {
      TMPDIR: "/tmp/ray", NEXUML_SPECTATOR_PYTHON: "/approved/python",
      NEXUML_PLAYWRIGHT_MODULE: paths.playwrightModulePath, NEXUML_CHROMIUM_EXECUTABLE: paths.chromeBinary,
      NEXUML_BROWSER_SWIFTSHADER_ICD: "/approved/vk_swiftshader_icd.json", UV_PROJECT_ENVIRONMENT: "/tmp/ray/nexuml-envs/mock",
    } },
    console: { log() {}, error() {} }, setTimeout() { return 0; }, clearTimeout() {},
  });
  await assert.rejects(localModule.exports.runController(), (error) => error === primary);
  assert.ok(calls.findIndex(([call]) => call === "kill") > calls.findIndex(([call]) => call === "browser-launch"));
  assert.ok(calls.findIndex(([call]) => call === "remove") > calls.findIndex(([call]) => call === "kill"));
  assert.equal(calls.findLast(([call]) => call === "write")[2].cleanup.status, "removed");
  assert.deepEqual(calls.at(-1), ["close", 17]);
});

test("browser close rejection still terminates native child and closes the log", async (t) => {
  const calls = [];
  const error = new Error("browser close failed");
  t.mock.method(fs, "closeSync", (fd) => calls.push(["close", fd]));
  const browser = { close: async () => { calls.push(["browser"]); throw error; } };
  const child = { exitCode: null, signalCode: null, kill: (signal) => calls.push(["kill", signal]) };
  await assert.rejects(cleanup(browser, child, async () => calls.push(["wait"]), Promise.resolve(), 7), error);
  assert.deepEqual(calls, [["browser"], ["kill", "SIGTERM"], ["wait"], ["close", 7]]);
});

test("owned child cleanup failure still closes the descriptor", async (t) => {
  const calls = [];
  t.mock.method(fs, "closeSync", (fd) => calls.push(["close", fd]));
  const child = { exitCode: null, signalCode: null, kill: (signal) => calls.push(["kill", signal]) };
  await cleanup(null, child, async () => { throw new Error("wait deadline"); }, Promise.resolve(), 8);
  assert.deepEqual(calls, [["kill", "SIGTERM"], ["kill", "SIGKILL"], ["close", 8]]);
});

test("native completion requires proof artifacts, not just the expected exit code", async (t) => {
  for (const mode of ["train-viewer", "replay-viewer", "failure-viewer", "cancel-viewer"]) {
    await t.test(mode, async (t) => {
      const identity = mode === "cancel-viewer" ? "SIGTERM" : "injected spectator training failure";
      const traceBytes = Buffer.from("trace bytes");
      const trace = { steps: 128, parts: ["trajectory.pt", "trajectory-0001.pt"].map((name) => ({
        path: name, steps: 64, bytes: traceBytes.length,
        sha256: crypto.createHash("sha256").update(traceBytes).digest("hex"),
      })) };
      const files = {
        "native-cleanup.json": JSON.stringify({ status: "PASS", mode, steps: 128 }),
        "logger-finalization.json": JSON.stringify([{ status: mode === "train-viewer" ? "success" : "failed",
          job_scope_active: true, completed: true, error: null }]),
        "trajectory-manifest.json": JSON.stringify(trace),
        "trajectory.pt": traceBytes,
        "trajectory-0001.pt": traceBytes,
        "injection.json": JSON.stringify({ mode, batch_idx: 2, identity }),
        "out.log": `CLI_PROBE_NATIVE_CLOSED_AND_OWNED_THREADS_STOPPED ${mode}\n${identity}`,
      };
      const missing = new Set();
      t.mock.method(fs, "readFileSync", (name) => {
        const key = path.basename(name);
        if (missing.has(key) || !(key in files)) throw new Error(`Missing ${key}`);
        return files[key];
      });
      t.mock.method(fs, "statSync", (name) => {
        const key = path.basename(name);
        if (missing.has(key) || !["trajectory.pt", "trajectory-0001.pt", "training-state.pt", "complete.ckpt"].includes(key)) {
          throw new Error("Missing artifact");
        }
        return { isFile: () => true, size: Buffer.isBuffer(files[key]) ? files[key].length : 12 };
      });
      const result = { code: mode === "failure-viewer" ? 1 : mode === "cancel-viewer" ? 143 : 0 };
      assert.doesNotThrow(() => verifyCompletion("/owned/out", mode, result));
      for (const name of ["native-cleanup.json", "trajectory-manifest.json", "trajectory.pt", "trajectory-0001.pt"]) {
        missing.add(name);
        assert.throws(() => verifyCompletion("/owned/out", mode, result));
        missing.delete(name);
      }
      files["native-cleanup.json"] = JSON.stringify({ status: "FAIL", mode, steps: 128 });
      assert.throws(() => verifyCompletion("/owned/out", mode, result));
      files["native-cleanup.json"] = JSON.stringify({ status: "PASS", mode, steps: 128 });
      files["trajectory.pt"] = Buffer.from(traceBytes).reverse();
      assert.throws(() => verifyCompletion("/owned/out", mode, result));
      files["trajectory.pt"] = traceBytes;
      files["trajectory-manifest.json"] = JSON.stringify({ ...trace, steps: 127 });
      assert.throws(() => verifyCompletion("/owned/out", mode, result));
      files["trajectory-manifest.json"] = JSON.stringify(trace);
      if (mode !== "replay-viewer") {
        const validFinalization = files["logger-finalization.json"];
        files["logger-finalization.json"] = JSON.stringify([{ status: "failed", job_scope_active: true,
          completed: false, error: "logger finalization failed" }]);
        assert.throws(() => verifyCompletion("/owned/out", mode, result));
        files["logger-finalization.json"] = validFinalization;
      }
      if (["failure-viewer", "cancel-viewer"].includes(mode)) {
        missing.add("injection.json");
        assert.throws(() => verifyCompletion("/owned/out", mode, result));
        missing.delete("injection.json");
        files["injection.json"] = JSON.stringify({ mode, batch_idx: 1, identity });
        assert.throws(() => verifyCompletion("/owned/out", mode, result));
        files["injection.json"] = JSON.stringify({ mode, batch_idx: 2, identity: "wrong injection" });
        assert.throws(() => verifyCompletion("/owned/out", mode, result));
        files["injection.json"] = JSON.stringify({ mode, batch_idx: 2, identity });
        if (mode === "failure-viewer") {
          files["out.log"] = `CLI_PROBE_NATIVE_CLOSED_AND_OWNED_THREADS_STOPPED ${mode}`;
          assert.throws(() => verifyCompletion("/owned/out", mode, result));
        }
      }
      files["out.log"] = identity;
      assert.throws(() => verifyCompletion("/owned/out", mode, result));
    });
  }
});
