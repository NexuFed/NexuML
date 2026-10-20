import {expect,test} from "vitest";
import {matchesSchema,resolveSchema,schemaDefault} from "../model/schema";
import {TerminalBuffer} from "../model/terminal";

test("schema refs, unions, discriminator constants, collections and defaults",()=>{
  const root={$defs:{Ray:{title:"Ray",type:"object",properties:{kind:{const:"ray"},workers:{type:"integer",default:1}},required:["kind"]}}};
  const ray={$ref:"#/$defs/Ray"};
  expect(resolveSchema(ray,root).title).toBe("Ray");
  expect(schemaDefault(ray,root)).toEqual({kind:"ray",workers:1});
  expect(matchesSchema({kind:"local"},ray,root)).toBe(false);
  expect(matchesSchema({kind:"ray"},ray,root)).toBe(true);
  expect(matchesSchema(null,{type:"null"},root)).toBe(true);
  expect(schemaDefault({type:"array",prefixItems:[{type:"integer"},{type:"string"}]})).toEqual([0,""]);
});

test("terminal overwrites CR/backspace, handles CRLF and fragmented ANSI safely",()=>{
  const output=new TerminalBuffer();
  output.write("Download 1%\rDownload 2%\rDownload 100%\r\nReady\n");
  expect(output.text).toBe("Download 100%\nReady\n");
  output.write("long progress\rshort\x1b[");output.write("K\nabc\bX");
  expect(output.text).toBe("Download 100%\nReady\nshort\nabX");
  output.write("\nsecond\x1b[1A\r\x1b[2Kreplaced\x1b[31m!\x1b[0m");
  expect(output.text).toContain("replaced!\nsecond");
  output.write("\x1b]8;;javascript:alert(1)\x07safe\x1b]8;;\x1b\\");
  expect(output.text).not.toContain("javascript");
});

test("terminal cursor and retained lines are bounded",()=>{
  const output=new TerminalBuffer();output.write("line\n".repeat(1100));output.write("\x1b[999999B\x1b[999999C!");
  expect(output.text.split("\n").length).toBeLessThanOrEqual(1001);
  expect(output.text).toContain("raw logs remain available");
  expect(output.text.length).toBeLessThan(20000);
  expect(()=>output.write("\x1b[-999B\x1b[-999Hsafe")).not.toThrow();
});
