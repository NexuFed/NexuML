import { expect, test } from "vitest";
import { chmod, mkdtemp, writeFile, rm, symlink } from "node:fs/promises";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createServer } from "node:http";
import { argumentsFor, executable, freePort, launch } from "../bin/studio.mjs";

test("explicit executable/interpreter/attach selection, spaced paths and invalid modes",async()=>{
  expect(argumentsFor(["--python","/path with spaces/python","/working directory"]).python).toBe("/path with spaces/python");
  expect(argumentsFor(["--api","http://127.0.0.1:8000","--api-token-file","private.txt"]).api).toBe("http://127.0.0.1:8000");
  expect(()=>argumentsFor(["--python","python","--nexuml","nexuml"])).toThrow("Choose one");
  expect(()=>argumentsFor(["--api-token-file","private.txt"])).toThrow("requires --api");
  const directory=await mkdtemp(join(tmpdir(),"nexuml launcher test "));
  try {
    const file=join(directory,"fake nexuml");await writeFile(file,"fake executable");await chmod(file,0o700);
    expect(await executable(file)).toBe(file);
    await expect(executable("nonexistent-nexuml",{...process.env,PATH:directory})).rejects.toThrow("Executable not found");
    await expect(executable("nonexistent-nexuml",{...process.env,PATH:directory})).rejects.toThrow('uv tool install "nexuml[api,library]"');
    await expect(launch({directory:join(directory,"missing")})).rejects.toThrow(`Select an existing directory: nexuml-studio /path/to/work`);
    await expect(launch({directory:file})).rejects.toThrow(file);
    expect(await freePort()).toBeGreaterThan(0);
  } finally {await rm(directory,{recursive:true,force:true});}
});

test("backend startup exit names the selected executable and recovery command before UI starts",async()=>{
  const directory=await mkdtemp(join(tmpdir(),"nexuml failed backend "));
  const port=await freePort();
  try {
    // Node is deliberately not a Python interpreter; its -m failure is a real child exit.
    await expect(launch({directory,python:process.execPath,port:String(port),"no-open":true})).rejects.toThrow(`NexuML API (${process.execPath}) exited during startup`);
    await expect(fetch(`http://127.0.0.1:${port}`)).rejects.toThrow();
  } finally {await rm(directory,{recursive:true,force:true});}
});

test("launcher hosting is explicit, validates flags/env before services and preserves local defaults",async()=>{
  expect(argumentsFor([],{})).toMatchObject({host:"127.0.0.1","public-origin":undefined});
  const hosted=["--public-origin","https://studio.example","--proxy-token-file","/private/proxy-token"];
  expect(argumentsFor(hosted,{})).toMatchObject({host:"127.0.0.1","public-origin":"https://studio.example"});
  expect(argumentsFor(["--host","192.168.1.20",...hosted],{}).host).toBe("192.168.1.20");
  expect(argumentsFor([],{NEXUML_STUDIO_HOST:"192.168.1.20",NEXUML_STUDIO_PUBLIC_ORIGIN:"https://studio.example",NEXUML_STUDIO_PROXY_TOKEN_FILE:"/private/proxy-token"})).toMatchObject({host:"192.168.1.20","public-origin":"https://studio.example"});
  expect(argumentsFor(["--host","127.0.0.1",...hosted,"--public-origin","https://other.example"],{NEXUML_STUDIO_HOST:"0.0.0.0",NEXUML_STUDIO_PUBLIC_ORIGIN:"https://studio.example"})).toMatchObject({host:"127.0.0.1","public-origin":"https://other.example"});
  expect(argumentsFor(["--host","127.0.0.1","--public-origin",""],{NEXUML_STUDIO_PUBLIC_ORIGIN:"https://studio.example"})).toMatchObject({host:"127.0.0.1","public-origin":""});
  expect(()=>argumentsFor(["--host","0.0.0.0"],{})).toThrow("requires a public HTTPS origin");
  expect(()=>argumentsFor(["--public-origin","http://studio.example"],{})).toThrow();
  expect(()=>argumentsFor(["--public-origin","https://studio.example"],{})).toThrow("--proxy-token-file");
  expect(()=>argumentsFor(["--proxy-token-file","/private/token"],{})).toThrow("--public-origin");
  expect(()=>argumentsFor(["--host","localhost",...hosted],{})).toThrow("IPv4");
  expect(()=>argumentsFor([],{NEXUML_STUDIO_PUBLIC_ORIGIN:"not a URL"})).toThrow();
  await expect(launch({directory:"/missing/working/directory",host:"0.0.0.0","public-origin":""})).rejects.toThrow("requires a public HTTPS origin");
  expect(await freePort(undefined,"0.0.0.0")).toBeGreaterThan(0);
});

test("CLI entrypoint runs when a parent directory is a filesystem alias",async()=>{
  const directory=await mkdtemp(join(tmpdir(),"nexuml launcher alias "));
  try {
    const alias=join(directory,"aliased bin");
    await symlink(fileURLToPath(new URL("../bin",import.meta.url)),alias,process.platform==="win32" ? "junction" : "dir");
    const result=spawnSync(process.execPath,[join(alias,"studio.mjs"),"--help"],{encoding:"utf8"});
    expect(result.status,result.stderr).toBe(0);
    expect(result.stdout).toContain("nexuml-studio [directory]");
  } finally {await rm(directory,{recursive:true,force:true});}
});

test("unsupported selected backend is rejected before UI startup without stopping the attached API",async()=>{
  const directory=await mkdtemp(join(tmpdir(),"nexuml compatibility "));
  const server=createServer((_,response)=>response.end(JSON.stringify({interface_version:2,nexuml_version:"0.3.0",working_directory:directory})));
  try {
    const tokenFile=join(directory,"token");await writeFile(tokenFile,"private-test-token-".repeat(3),{mode:0o600});
    await new Promise<void>(done=>server.listen(0,"127.0.0.1",done));
    const address=server.address();if(!address || typeof address==="string")throw new Error("Expected TCP listener");
    const port=await freePort();
    await expect(launch({directory,api:`http://127.0.0.1:${address.port}`,"api-token-file":tokenFile,
      "public-origin":"https://studio.example","proxy-token-file":tokenFile,port:String(port),"no-open":true})).rejects.toThrow("must be different");
    await expect(launch({directory,api:`http://127.0.0.1:${address.port}`,"api-token-file":tokenFile,port:String(port),"no-open":true})).rejects.toThrow("NexuML >=0.2.1,<0.3.0");
    expect(server.listening).toBe(true);await expect(fetch(`http://127.0.0.1:${port}`)).rejects.toThrow();
  } finally {server.closeAllConnections();await new Promise<void>(done=>server.close(()=>done()));await rm(directory,{recursive:true,force:true});}
});
