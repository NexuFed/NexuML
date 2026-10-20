"""Exercise the real packaged-browser invocation without ML or browser dependencies."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("browser_exit", [0, 7])
def test_packaged_browser_wait_drains_service_output(tmp_path, browser_exit):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Packaged smoke requires Node")
    source = (ROOT / "studio/scripts/package-smoke.mjs").read_text()
    helpers = source[source.index("function run(") : source.index("\ntry {\n")]
    start = source.index("  // Exercise the committed complete CPU browser workflow")
    invocation = source[start : source.index("  assert.equal(await readFile", start)]
    cli = tmp_path / "node_modules/@playwright/test/cli.js"
    cli.parent.mkdir(parents=True)
    cli.write_text(
        "const fs=require('node:fs');"
        + (
            "const deadline=Date.now()+2000;const timer=setInterval(()=>{"
            "if(fs.existsSync(process.env.SMOKE_READY)){clearInterval(timer);process.exit(0)}"
            "else if(Date.now()>deadline){clearInterval(timer);process.exit(8)}},10);"
            if browser_exit == 0
            else f"process.exit({browser_exit});"
        )
    )
    driver = f"""
import {{spawn,spawnSync}} from 'node:child_process';
import {{once}} from 'node:events';
import assert from 'node:assert/strict';
import {{join}} from 'node:path';
const root={json.dumps(str(tmp_path))},temporary=root,npm='unused';
const origin='http://127.0.0.1:1',libraryRoot=root;
{helpers}
const service=spawn(process.execPath,['-e',
    "const fs=require('node:fs');fs.writeSync(1,Buffer.alloc(1024*1024,120));"
    +"fs.writeFileSync(process.env.SMOKE_READY,'ready');"]);
let bytes=0;
service.stdout.on('data',chunk=>bytes+=chunk.length);
service.stderr.resume();
const closed=once(service,'close');
try {{
{invocation}
    await closed;
    assert.equal(bytes,1024*1024);
}} finally {{
    service.kill();
    await closed;
}}
"""
    result = subprocess.run(
        [node, "--input-type=module", "-"],
        input=driver,
        capture_output=True,
        text=True,
        timeout=15,
        env={**os.environ, "SMOKE_READY": str(tmp_path / "service-ready")},
    )
    if browser_exit == 0:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0
        assert "7 !== 0" in result.stderr
