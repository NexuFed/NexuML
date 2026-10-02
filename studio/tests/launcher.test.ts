import { expect, test } from "vitest";
import { chmod, mkdtemp, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { argumentsFor, executable, freePort } from "../bin/studio.mjs";

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
