import {createElement} from "react";
import {renderToStaticMarkup} from "react-dom/server";
import {expect,test} from "vitest";
import {ExecutionSettings} from "../components/execution-settings";
import type {Catalog,ExecutionBackend} from "../model/types";

const entry:ExecutionBackend={type:"sixth",version:"1",label:"Installation sixth",import_target:"trusted.Sixth",available:false,
  diagnostics:["Missing optional runtime"],capabilities:{status:true},presentation:{target:["endpoint"]},
  schema:{type:"object",properties:{endpoint:{type:"string"},unrendered:{type:"string"}}}};

test("additional runtime schema/diagnostics and missing selections are retained",()=>{
  const value={type:"sixth",version:"1",params:{endpoint:"explicit",unrendered:"keep"}};
  const catalog={execution_backends:[entry],components:[]} as unknown as Catalog;
  const html=renderToStaticMarkup(createElement(ExecutionSettings,{value,catalog,onChange:()=>{}}));
  expect(html).toContain("Installation sixth — dependency unavailable");
  expect(html).toContain("Missing optional runtime");
  expect(html).toContain('data-field="execution.params.endpoint"');
  expect(value.params.unrendered).toBe("keep");
  const missing=renderToStaticMarkup(createElement(ExecutionSettings,{value,catalog:{...catalog,execution_backends:[]},onChange:()=>{}}));
  expect(missing).toContain("not installed");expect(missing).toContain("Configuration retained; launch blocked");
  expect(missing).toContain("explicit");
});
