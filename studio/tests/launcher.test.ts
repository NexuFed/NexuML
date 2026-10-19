import { expect, test } from "vitest";
import { chmod, mkdtemp, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
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
    expect(await freePort()).toBeGreaterThan(0);
  } finally {await rm(directory,{recursive:true,force:true});}
});

test("launcher hosting is explicit, validates flags/env before services and preserves local defaults",async()=>{
  expect(argumentsFor([],{})).toMatchObject({host:"127.0.0.1","public-origin":undefined});
  expect(argumentsFor(["--public-origin","https://studio.example"],{})).toMatchObject({host:"0.0.0.0","public-origin":"https://studio.example"});
  expect(argumentsFor(["--host","192.168.1.20","--public-origin","https://studio.example"],{}).host).toBe("192.168.1.20");
  expect(argumentsFor([],{NEXUML_STUDIO_HOST:"192.168.1.20",NEXUML_STUDIO_PUBLIC_ORIGIN:"https://studio.example"})).toMatchObject({host:"192.168.1.20","public-origin":"https://studio.example"});
  expect(argumentsFor(["--host","127.0.0.1","--public-origin","https://other.example"],{NEXUML_STUDIO_HOST:"0.0.0.0",NEXUML_STUDIO_PUBLIC_ORIGIN:"https://studio.example"})).toMatchObject({host:"127.0.0.1","public-origin":"https://other.example"});
  expect(argumentsFor(["--host","127.0.0.1","--public-origin",""],{NEXUML_STUDIO_PUBLIC_ORIGIN:"https://studio.example"})).toMatchObject({host:"127.0.0.1","public-origin":""});
  expect(()=>argumentsFor(["--host","0.0.0.0"],{})).toThrow("requires a public HTTPS origin");
  expect(()=>argumentsFor(["--public-origin","http://studio.example"],{})).toThrow();
  expect(()=>argumentsFor(["--host","localhost","--public-origin","https://studio.example"],{})).toThrow("IPv4");
  expect(()=>argumentsFor([],{NEXUML_STUDIO_PUBLIC_ORIGIN:"not a URL"})).toThrow();
  await expect(launch({directory:"/missing/working/directory",host:"0.0.0.0","public-origin":""})).rejects.toThrow("requires a public HTTPS origin");
  expect(await freePort(undefined,"0.0.0.0")).toBeGreaterThan(0);
});
