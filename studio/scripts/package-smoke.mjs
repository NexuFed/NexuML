import { spawn, spawnSync } from "node:child_process";
import { access, mkdir, mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { delimiter, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import assert from "node:assert/strict";
import { randomBytes } from "node:crypto";
import { freePort } from "../bin/studio.mjs";

const root=resolve(fileURLToPath(new URL("..",import.meta.url)));
const python=process.env.STUDIO_TEST_PYTHON;
if(!python) throw new Error("Set STUDIO_TEST_PYTHON to an existing test installation with NexuML[api] and the base library.");
const temporary=await mkdtemp(join(tmpdir(),"nexuml packaged smoke "));
const npm=process.platform==="win32" ? "npm.cmd" : "npm";
const children=[];
function run(command,args,options={}) {
  if(command===npm) {
    if(!process.env.npm_execpath)throw new Error("Run package smoke through npm run test:package.");
    command=process.execPath;args=[process.env.npm_execpath,...args];
  }
  const result=spawnSync(command,args,{cwd:temporary,encoding:"utf8",...options});
  assert.equal(result.status,0,result.stderr || result.stdout);return result.stdout;
}
async function wait(url,child,headers={}) {
  const deadline=Date.now()+60000;
  while(Date.now()<deadline) {
    assert.equal(child.exitCode,null,"Packaged service exited unexpectedly");
    try {const response=await fetch(url,{headers,signal:AbortSignal.timeout(1000)});if(response.ok)return response;}catch{ /* Bounded startup readiness. */ }
    await new Promise(done=>setTimeout(done,150));
  }
  throw new Error("Packaged startup deadline exceeded");
}
async function stop(child) {
  if(child.exitCode!==null || child.signalCode!==null)return;
  const done=new Promise(resolve=>child.once("exit",resolve));
  if(child.connected)child.send("shutdown");else child.kill("SIGTERM");
  await Promise.race([done,new Promise((_,reject)=>setTimeout(()=>reject(new Error("Launcher cleanup timeout")),20000))]);
}
try {
  run(npm,["run","prepack"],{cwd:root});
  const packed=JSON.parse(run(npm,["pack","--ignore-scripts","--json","--pack-destination",temporary],{cwd:root}))[0];
  assert(packed.files.some(file=>file.path.startsWith(".next/static/") && file.path.endsWith(".woff2")),"Offline fonts missing");
  assert(!packed.files.some(file=>/(^|\/)node_modules\/|(^|\/)\.env|\.next\/cache\/|^tests\/|^src\/|\.next\/dev\//.test(file.path)),"Dev trees/secrets in package");
  await writeFile(join(temporary,"package.json"),'{"private":true}');
  run(npm,["install","--offline","--omit=dev","--ignore-scripts",join(temporary,packed.filename)]);
  const installed=join(temporary,"node_modules/@nexufed/nexuml-studio");
  const work=join(temporary,"working directory");const home=join(temporary,"isolated user home");
  await mkdir(work);await mkdir(home);
  const libraryRoot=join(temporary,"trusted test library");
  const libraryPackage=join(libraryRoot,"studio_smoke_library");await mkdir(libraryPackage,{recursive:true});
  await writeFile(join(libraryPackage,"__init__.py"),"");
  await writeFile(join(libraryPackage,"scale.py"),`from pydantic import Field
from nexuml.core.discovery import layer
from nexuml.core.components import LayerDefinition
from nexuml.core.base_layer import PipelineLayer
@layer("studio-smoke-scale")
class Scale(LayerDefinition):
    scale: float = Field(default=1.0, gt=0, description="Trusted smoke scale")
    def build(self, context):
        return Runtime(scale=self.scale, **context.runtime_kwargs())
class Runtime(PipelineLayer):
    def __init__(self, scale, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
    def forward_tensor(self, x, y=None):
        return x * self.scale
`);
  const env={...process.env,HOME:home,USERPROFILE:home,NEXT_TELEMETRY_DISABLED:"1"};
  run(python,["-c","from pathlib import Path; from nexuml.core.config import ResolvedConfig; from nexuml_library.scenarios.asd.synthetic_linear_ae import synthetic_linear_ae_reconstruction; s=synthetic_linear_ae_reconstruction(feature_shape=(8,),num_samples=32,hidden_dims=[8],latent_dim=2,batch_size=8,max_epochs=1);s.training.accelerator='cpu';s.training.devices=1;ResolvedConfig.from_scenario(s).save(Path('tiny.yaml'))"],{cwd:work,env});
  await writeFile(join(work,"tiny.yaml.studio.json"),JSON.stringify({semantic_revision:"stale-fixture-revision",layout:{ids:{},positions:{data:{x:9999,y:9999}}}}));
  const before=await readFile(join(temporary,"package-lock.json"),"utf8");
  const environmentCheck=["-c","import hashlib,importlib.metadata,json; from pathlib import Path; import sys; records=[(d.metadata['Name'],d.version,hashlib.sha256(d.read_text('RECORD').encode()).hexdigest()) for d in importlib.metadata.distributions() if d.read_text('RECORD')]; cfg=Path(sys.prefix)/'pyvenv.cfg'; print(json.dumps([sorted(records),cfg.read_text() if cfg.exists() else None]))"];
  const beforePython=run(python,environmentCheck,{cwd:work,env});
  const port=await freePort();const origin=`http://127.0.0.1:${port}`;
  const launch=spawn(process.execPath,[join(installed,"bin/studio.mjs"),"--python",python,"--no-open","--port",String(port),work],{env,stdio:["ignore","pipe","pipe","ipc"]});children.push(launch);
  let output="";launch.stdout.on("data",chunk=>output+=chunk);launch.stderr.on("data",chunk=>output+=chunk);
  await wait(origin,launch);
  const runtime=await (await wait(`${origin}/api/runtime`,launch,{Host:`127.0.0.1:${port}`})).json();
  const identity=await (await fetch(`${runtime.api}/api/v1/runtime`,{headers:{Authorization:`Bearer ${runtime.token}`,Origin:origin}})).json();
  assert.equal(identity.working_directory,work);assert.equal(identity.interface_version,1);
  assert(!output.includes(runtime.token),"Token leaked to output");
  const library=await (await fetch(`${runtime.api}/api/v1/registry`,{headers:{Authorization:`Bearer ${runtime.token}`,Origin:origin}})).json();
  assert(library.scenarios.some(s=>s.name==="synthetic-linear-ae-reconstruction"));
  const {chromium}=await import("playwright");
  const browser=await chromium.launch({headless:true});
  try {
    const page=await browser.newPage();
    // Only the two loopback services are reachable: assets/font/UI work offline.
    await page.route("**/*",route=>["127.0.0.1","localhost"].includes(new URL(route.request().url()).hostname) ? route.continue() : route.abort());
    await page.goto(origin);await page.getByLabel("Config path",{exact:true}).fill("tiny.yaml");await page.getByRole("button",{name:"Open YAML",exact:true}).click();await page.getByLabel("Search components").waitFor();
    assert.equal(await page.locator("body").evaluate(element=>getComputedStyle(element).backgroundColor),"rgb(16, 16, 16)");
  }finally{await browser.close();}
  // Exercise the committed complete CPU browser workflow against the installed package.
  run(process.execPath,[join(root,"node_modules/@playwright/test/cli.js"),"test"],{cwd:root,env:{...env,
    PLAYWRIGHT_BROWSERS_PATH:process.env.PLAYWRIGHT_BROWSERS_PATH ?? dirname(dirname(dirname(chromium.executablePath()))),
    STUDIO_TEST_URL:origin,STUDIO_CONFIG_PATH:"tiny.yaml",STUDIO_LIBRARY_ROOT:libraryRoot}});
  assert.equal(await readFile(join(temporary,"package-lock.json"),"utf8"),before,"Startup changed packages");
  assert.equal(run(python,environmentCheck,{cwd:work,env}),beforePython,"Workflow changed the Python installation");
  const savedConfig=(await readdir(work)).find(name=>name.startsWith("browser-saved-") && name.endsWith(".yaml"));
  assert(savedConfig,"Browser did not persist ordinary YAML");
  run(python,["-m","nexuml.cli.main","build",join(work,savedConfig)],{cwd:work,env});
  await stop(launch);
  const stopped=await fetch(`${runtime.api}/api/v1/runtime`,{headers:{Authorization:`Bearer ${runtime.token}`}}).then(()=>false,()=>true);
  assert(stopped,"Owned API survived launcher shutdown");
  await access(join(work,"tiny.yaml"));
  assert(!output.includes("uv sync") && !output.includes("pip install") && !output.includes("next build"));
  if(process.env.STUDIO_TEST_NEXUML) {
    const executablePort=await freePort();const executableOrigin=`http://127.0.0.1:${executablePort}`;
    const selected=spawn(process.execPath,[join(installed,"bin/studio.mjs"),"--nexuml",process.env.STUDIO_TEST_NEXUML,"--no-open","--port",String(executablePort),work],{env,stdio:["ignore","ignore","inherit","ipc"]});children.push(selected);
    await wait(executableOrigin,selected);await stop(selected);
    const pathPort=await freePort();
    const fallback=spawn(process.execPath,[join(installed,"bin/studio.mjs"),"--no-open","--port",String(pathPort),work],{
      env:{...env,PATH:`${dirname(process.env.STUDIO_TEST_NEXUML)}${delimiter}${env.PATH ?? ""}`},stdio:["ignore","ignore","inherit","ipc"]});children.push(fallback);
    await wait(`http://127.0.0.1:${pathPort}`,fallback);await stop(fallback);
  }
  const attachPort=await freePort(),apiPort=await freePort();const attachOrigin=`http://127.0.0.1:${attachPort}`;
  const attachToken=randomBytes(32).toString("base64url");const tokenFile=join(temporary,"private token");
  await writeFile(tokenFile,attachToken,{mode:0o600});
  const independent=spawn(python,["-m","nexuml.cli.main","serve","--directory",work,"--port",String(apiPort),"--origin",attachOrigin],{env:{...env,NEXUML_API_TOKEN:attachToken},stdio:"ignore"});children.push(independent);
  const api=`http://127.0.0.1:${apiPort}`;await wait(`${api}/api/v1/runtime`,independent,{Authorization:`Bearer ${attachToken}`});
  const attached=spawn(process.execPath,[join(installed,"bin/studio.mjs"),"--api",api,"--api-token-file",tokenFile,"--port",String(attachPort),"--no-open",work],{env,stdio:["ignore","ignore","inherit","ipc"]});children.push(attached);
  await wait(attachOrigin,attached);await stop(attached);
  assert.equal(independent.exitCode,null,"Attached API was stopped by the launcher");
  assert.equal((await fetch(`${api}/api/v1/runtime`,{headers:{Authorization:`Bearer ${attachToken}`}})).status,200);
  await stop(independent);
  console.log("Packaged runtime selection, offline fonts, CPU browser workflow, owned and attached shutdown passed.");
}finally{
  for(const child of children)if(child.exitCode===null && child.signalCode===null)await stop(child);
  if(process.env.STUDIO_KEEP_SMOKE)console.log(`Smoke artifacts retained: ${temporary}`);
  else await rm(temporary,{recursive:true,force:true});
}
