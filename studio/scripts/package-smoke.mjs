import { spawn, spawnSync } from "node:child_process";
import { access, chmod, mkdir, mkdtemp, readFile, readdir, realpath, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { delimiter, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import assert from "node:assert/strict";
import { randomBytes } from "node:crypto";
import { createServer, request as httpRequest } from "node:http";
import { createRequire } from "node:module";
import { freePort } from "../bin/studio.mjs";

const root=resolve(fileURLToPath(new URL("..",import.meta.url)));
const python=process.env.STUDIO_TEST_PYTHON;
if(!python) throw new Error("Set STUDIO_TEST_PYTHON to an existing test installation with NexuML[api] and the base library.");
const tarball=process.argv[2] && resolve(process.argv[2]);
if(!tarball)throw new Error("Pass the already built npm tarball: npm run test:package -- /path/to/nexufed-nexuml-studio.tgz");
const temporary=await mkdtemp(join(tmpdir(),"nexuml packaged smoke "));
const npm=process.platform==="win32" ? "npm.cmd" : "npm";
const children=[];
function run(command,args,options={}) {
  if(command===npm) {
    if(!process.env.npm_execpath)throw new Error("Run package smoke through npm run test:package.");
    command=process.execPath;args=[process.env.npm_execpath,...args];
  }
  const result=spawnSync(command,args,{cwd:temporary,encoding:"utf8",...options});
  assert.equal(result.status,0,`${command} ${args.join(" ")} failed\n${result.error ?? ""}${result.stdout ?? ""}${result.stderr ?? ""}`);return result.stdout;
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
  const packed=JSON.parse(run(npm,["pack","--ignore-scripts","--dry-run","--json",tarball]))[0];
  assert.equal(packed.name,"@nexufed/nexuml-studio");
  assert(packed.files.some(file=>file.path==="LICENSE"),"License missing");
  assert(packed.files.some(file=>file.path.startsWith(".next/static/") && file.path.endsWith(".woff2")),"Offline fonts missing");
  assert(packed.files.every(file=>/^(bin\/[^/]+\.mjs|\.next\/(server\/|static\/|[^/]+\.json$|BUILD_ID$)|next\.config\.mjs$|package\.json$|README\.md$|LICENSE$)/.test(file.path)),"Unexpected files in package");
  assert(!packed.files.some(file=>/(^|\/)node_modules\/|(^|\/)\.env|(^|\/)cache\/|\.next\/dev\//.test(file.path)),"Dev trees/secrets in package");
  await writeFile(join(temporary,"package.json"),'{"private":true}');
  // Installation may fetch dependencies; only the installed runtime must work offline.
  run(npm,["install","--omit=dev","--ignore-scripts",tarball],{stdio:"inherit"});
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
  delete env.PYTHONPATH;delete env.PYTHONHOME;
  const browserOpened=join(temporary,"browser-opened.txt");
  if(process.platform==="linux") {
    // Capture the default opener invocation without launching a desktop browser in CI.
    const bin=join(temporary,"browser bin");await mkdir(bin);
    const opener=join(bin,"xdg-open");await writeFile(opener,`#!/usr/bin/env node\nrequire("node:fs").writeFileSync(${JSON.stringify(browserOpened)},process.argv[2]);\n`);await chmod(opener,0o700);
    env.PATH=`${bin}${delimiter}${env.PATH ?? ""}`;
  }
  run(python,["-c","from pathlib import Path; from nexuml.core.config import ResolvedConfig; from nexuml_library.scenarios.asd.synthetic_linear_ae import synthetic_linear_ae_reconstruction; s=synthetic_linear_ae_reconstruction(feature_shape=(8,),num_samples=32,hidden_dims=[8],latent_dim=2,batch_size=8,max_epochs=1);s.training.accelerator='cpu';s.training.devices=1;ResolvedConfig.from_scenario(s).save(Path('tiny.yaml'))"],{cwd:work,env});
  await writeFile(join(work,"tiny.yaml.studio.json"),JSON.stringify({semantic_revision:"stale-fixture-revision",layout:{ids:{},positions:{data:{x:9999,y:9999}}}}));
  const before=await readFile(join(temporary,"package-lock.json"),"utf8");
  const beforeNpm=run(npm,["ls","--all","--json"]);
  const environmentCheck=["-c","import hashlib,importlib.metadata,json; from pathlib import Path; import sys; records=[(d.metadata['Name'],d.version,hashlib.sha256(d.read_text('RECORD').encode()).hexdigest()) for d in importlib.metadata.distributions() if d.read_text('RECORD')]; cfg=Path(sys.prefix)/'pyvenv.cfg'; print(json.dumps([sorted(records),cfg.read_text() if cfg.exists() else None]))"];
  const beforePython=run(python,environmentCheck,{cwd:work,env});
  const port=await freePort();const ownedApiPort=await freePort();const origin=`http://127.0.0.1:${port}`;
  const launch=spawn(process.execPath,[join(installed,"bin/studio.mjs"),"--python",python,...(process.platform==="linux" ? [] : ["--no-open"]),"--port",String(port),"--api-port",String(ownedApiPort),work],{env,stdio:["ignore","pipe","pipe","ipc"]});children.push(launch);
  let output="";launch.stdout.on("data",chunk=>output+=chunk);launch.stderr.on("data",chunk=>output+=chunk);
  await wait(origin,launch);
  const runtime=await (await wait(`${origin}/api/runtime`,launch,{Host:`127.0.0.1:${port}`})).json();
  assert.equal(runtime.api,origin,"Browser transport must use only the frontend origin");
  const identity=await (await fetch(`${runtime.api}/api/v1/runtime`,{headers:{Authorization:`Bearer ${runtime.token}`,Origin:origin}})).json();
  assert.equal(await realpath(identity.working_directory),await realpath(work));assert.equal(identity.interface_version,2);
  assert.equal(resolve(identity.python_executable),resolve(python),"API must report the selected interpreter, not the shell Python");
  assert(!output.includes(runtime.token),"Token leaked to output");
  const library=await (await fetch(`${runtime.api}/api/v1/registry`,{headers:{Authorization:`Bearer ${runtime.token}`,Origin:origin}})).json();
  assert(library.scenarios.some(s=>s.name==="synthetic-linear-ae-reconstruction"));
  const {chromium}=await import("playwright");
  const browser=await chromium.launch({headless:true});
  try {
    const page=await browser.newPage();
    // Assets/font/API/WebSockets all use the same local frontend origin offline.
    await page.route("**/*",route=>["127.0.0.1","localhost"].includes(new URL(route.request().url()).hostname) ? route.continue() : route.abort());
    await page.goto(origin);await page.locator(".welcome").getByRole("button",{name:"Open YAML",exact:true}).click();
    await page.getByLabel("Config path",{exact:true}).fill("tiny.yaml");await page.getByRole("button",{name:"Open file",exact:true}).click();
    await page.getByRole("dialog").waitFor({state:"hidden"});await page.locator(".pipeline-view").waitFor();
    assert.equal(await page.locator("body").evaluate(element=>getComputedStyle(element).backgroundColor),"rgb(16, 16, 16)");
  }finally{await browser.close();}
  if(process.platform==="linux")assert.equal(await readFile(browserOpened,"utf8"),origin,"Default startup did not open the selected Studio URL");
  // Exercise the committed complete CPU browser workflow against the installed package.
  // Keep Playwright's original browser cache; only the launched services isolate HOME.
  run(process.execPath,[join(root,"node_modules/@playwright/test/cli.js"),"test"],{cwd:root,stdio:"inherit",env:{...process.env,
    STUDIO_TEST_URL:origin,STUDIO_CONFIG_PATH:"tiny.yaml",STUDIO_LIBRARY_ROOT:libraryRoot}});
  assert.equal(await readFile(join(temporary,"package-lock.json"),"utf8"),before,"Startup changed packages");
  assert.equal(run(npm,["ls","--all","--json"]),beforeNpm,"Workflow changed installed npm dependencies");
  assert.equal(run(python,environmentCheck,{cwd:work,env}),beforePython,"Workflow changed the Python installation");
  const savedConfig=(await readdir(work)).find(name=>name.startsWith("browser-saved-") && name.endsWith(".yaml"));
  assert(savedConfig,"Browser did not persist ordinary YAML");
  run(python,["-m","nexuml.cli.main","build",join(work,savedConfig)],{cwd:work,env});
  await stop(launch);
  const stopped=await fetch(`http://127.0.0.1:${ownedApiPort}/api/v1/runtime`,{headers:{Authorization:`Bearer ${runtime.token}`}}).then(()=>false,()=>true);
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
  // Exercise the actual Next HTTP/upgrade path, not just mocked Host/Origin headers.
  const hostedOrigin="https://studio.example",proxyToken=randomBytes(32).toString("base64url");
  const proxyFile=join(temporary,"private proxy token");await writeFile(proxyFile,proxyToken,{mode:0o600});
  const hosted=spawn(process.execPath,[join(installed,"bin/studio.mjs"),"--api",api,"--api-token-file",tokenFile,
    "--port",String(attachPort),"--public-origin",hostedOrigin,"--proxy-token-file",proxyFile,"--no-open",work],{env,stdio:["ignore","ignore","inherit","ipc"]});children.push(hosted);
  await wait(attachOrigin,hosted);
  for(const headers of [{},{host:"studio.example",origin:hostedOrigin},{host:"studio.example",origin:hostedOrigin,
    "x-forwarded-host":"studio.example","x-nexuml-studio-proxy":"spoofed"}]) {
    const denied=await fetch(`${attachOrigin}/api/runtime`,{headers});assert.equal(denied.status,401);
    assert(!(await denied.text()).includes(attachToken));
  }
  const session=randomBytes(32).toString("base64url");
  const proxy=createServer((incoming,outgoing)=>{
    if(incoming.headers.cookie!==`session=${session}`){outgoing.writeHead(401).end();return;}
    const upstream=httpRequest(`${attachOrigin}${incoming.url}`,{method:incoming.method,headers:{...incoming.headers,
      host:"studio.example","x-nexuml-studio-proxy":proxyToken}},response=>{outgoing.writeHead(response.statusCode,response.headers);response.pipe(outgoing);});
    upstream.on("error",()=>outgoing.destroy());incoming.pipe(upstream);
  });
  proxy.on("upgrade",(incoming,socket,head)=>{
    if(incoming.headers.cookie!==`session=${session}`){socket.end("HTTP/1.1 401 Unauthorized\r\nConnection: close\r\n\r\n");return;}
    const upstream=httpRequest(`${attachOrigin}${incoming.url}`,{headers:{...incoming.headers,host:"studio.example","x-nexuml-studio-proxy":proxyToken}});
    upstream.on("upgrade",(response,peer,upstreamHead)=>{
      socket.write(`HTTP/1.1 101 Switching Protocols\r\n${Object.entries(response.headers).map(([key,value])=>`${key}: ${value}`).join("\r\n")}\r\n\r\n`);
      if(upstreamHead.length)socket.write(upstreamHead);if(head.length)peer.write(head);
      socket.on("error",()=>peer.destroy());peer.on("error",()=>socket.destroy());socket.pipe(peer).pipe(socket);
    });
    upstream.on("response",response=>{socket.end(`HTTP/1.1 ${response.statusCode} Rejected\r\nConnection: close\r\n\r\n`);response.resume();});
    upstream.on("error",()=>socket.destroy());upstream.end();
  });
  await new Promise(done=>proxy.listen(0,"127.0.0.1",done));const proxyOrigin=`http://127.0.0.1:${proxy.address().port}`;
  try {
    assert.equal((await fetch(`${proxyOrigin}/api/runtime`,{headers:{"x-nexuml-studio-proxy":proxyToken}})).status,401,"Proxy authentication can be bypassed");
    const headers={cookie:`session=${session}`,origin:hostedOrigin,"x-nexuml-studio-proxy":"client-spoof"};
    const bootstrap=await fetch(`${proxyOrigin}/api/runtime`,{headers});assert.equal(bootstrap.status,200);
    assert.equal(bootstrap.headers.get("cache-control"),"no-store");assert.deepEqual(await bootstrap.json(),{api:hostedOrigin,token:attachToken});
    assert.equal((await fetch(`${proxyOrigin}/api/v1/runtime`,{headers})).status,401,"Proxy bypassed backend bearer authentication");
    const authenticated={...headers,authorization:`Bearer ${attachToken}`};
    const identity=await fetch(`${proxyOrigin}/api/v1/runtime`,{headers:authenticated});assert.equal(identity.status,200);
    const operations=await (await fetch(`${proxyOrigin}/api/v1/operations`,{headers:authenticated})).json();assert(operations.operations.length);
    const {WebSocket}=createRequire(join(installed,"package.json"))("next/dist/compiled/ws");
    const path=`/api/v1/operations/${operations.operations[0].id}/events?after=0`;
    const observe=(url,headers,authorized)=>new Promise((done,reject)=>{
      // Next may decline an upgrade without an HTTP error response; no unauthorized open/data is allowed.
      const socket=new WebSocket(url,{headers,handshakeTimeout:authorized ? 10000 : 1000});let settled=false;
      const finish=error=>{if(settled)return;settled=true;clearTimeout(timer);socket.terminate();if(error)reject(error);else done();};
      const timer=setTimeout(()=>finish(new Error("Hosted WebSocket did not settle")),10000);
      socket.on("open",()=>{if(!authorized)finish(new Error("Direct WebSocket bypassed proxy authentication"));else socket.send(JSON.stringify({token:attachToken}));});
      socket.on("unexpected-response",(_,response)=>{response.resume();finish(authorized ? new Error(`Authenticated WebSocket rejected (${response.statusCode})`) : undefined);});
      socket.on("message",data=>{try{assert(authorized,"Unauthorized WebSocket received data");const event=JSON.parse(data.toString());assert.equal(event.kind,"snapshot");assert.equal(event.payload.id,operations.operations[0].id);assert(!data.toString().includes(proxyToken));finish();}catch(error){finish(error);}});
      socket.on("error",error=>finish(authorized ? error : undefined));
      socket.on("close",()=>finish(authorized ? new Error("Authenticated WebSocket closed without observations") : undefined));
    });
    await observe(`${attachOrigin.replace("http:","ws:")}${path}`,{host:"studio.example",origin:hostedOrigin},false);
    await observe(`${proxyOrigin.replace("http:","ws:")}${path}`,headers,true);
  } finally {proxy.closeAllConnections();await new Promise(done=>proxy.close(done));await stop(hosted);}
  await stop(independent);
  console.log("Packaged runtime selection, offline fonts, CPU browser workflow, hosted HTTP/WebSocket security, owned and attached shutdown passed.");
}finally{
  for(const child of children)if(child.exitCode===null && child.signalCode===null)await stop(child);
  if(process.env.STUDIO_KEEP_SMOKE)console.log(`Smoke artifacts retained: ${temporary}`);
  else await rm(temporary,{recursive:true,force:true});
}
