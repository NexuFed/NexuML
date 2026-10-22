"use client";
import { useId } from "react";
import type { Catalog, Component, RecordValue } from "../model/types";
import { schemaDefault } from "../model/schema";
import { Fields, StructuredField } from "./fields";

export function ExecutionSettings({value,catalog,onChange,errors=[],suggestions}: {
  value:Component;catalog?:Catalog;onChange:(value:Component,validate?:boolean)=>void|Promise<void>;
  errors?:{loc:(string|number)[];message:string}[];suggestions?:Record<string,string[]>;
}) {
  const id=useId();
  const entries=catalog?.execution_backends ?? [];
  const selected=entries.find(entry=>entry.type===value.type && entry.version===value.version);
  const groups=Object.entries(selected?.presentation ?? {}).filter(([name])=>name!=="expert");
  const assigned=new Set<string>();
  const sections=groups.map(([name,paths])=>{const keys=paths.filter(key=>!assigned.has(key));keys.forEach(key=>assigned.add(key));return {name,keys};});
  const remaining=[...new Set([...Object.keys(selected?.schema.properties ?? {}),...Object.keys(value.params)])].filter(key=>!assigned.has(key));
  const fields=(keys:string[])=> <Fields value={Object.fromEntries(Object.entries(value.params).filter(([key])=>keys.includes(key)))}
    schema={{...selected?.schema,properties:Object.fromEntries(Object.entries(selected?.schema.properties ?? {}).filter(([key])=>keys.includes(key))),required:selected?.schema.required?.filter(key=>keys.includes(key))}} root={selected?.schema}
    raw={false} prefix="execution.params." catalog={catalog} suggestions={suggestions} errors={errors} onChange={(params,validate)=>onChange({...value,params:{...value.params,...params}},validate)}/>;
  const error=errors.filter(field=>["execution","execution.type","execution.version"].includes(field.loc.join("."))).map(field=>field.message).join("; ");
  return <section aria-label="Execution selection" data-field="execution" tabIndex={-1} aria-describedby={error ? id : undefined}><label className="field">Run on
    <select aria-label="Execution backend" data-field="execution.type" aria-invalid={!!error} aria-describedby={error ? id : undefined} value={`${value.type}:${value.version}`} onChange={event=>{
      const entry=entries.find(entry=>`${entry.type}:${entry.version}`===event.target.value);
      if(entry)onChange({type:entry.type,version:entry.version,params:schemaDefault(entry.schema) as RecordValue});
    }}>
      {!selected && <option value={`${value.type}:${value.version}`}>{value.type} · v{value.version} (not installed)</option>}
      {entries.map(entry=><option key={`${entry.type}:${entry.version}`} value={`${entry.type}:${entry.version}`}>{entry.label}{entry.available ? "" : " — dependency unavailable"}</option>)}
    </select></label>
    {error && <p id={id} role="alert" className="field-error">{error}</p>}
    {!selected && <p className="field-error">Selected definition is missing. Configuration retained; launch blocked.</p>}
    {selected?.diagnostics.map(message=><p className="field-error" key={message}>{message}</p>)}
    <div key={`${value.type}:${value.version}`} className="fields">{sections.length ? <>{sections.map(({name,keys})=><section key={name}><h3>{name==="target" ? "Destination" : name.replaceAll("_"," ")}</h3>{fields(keys)}</section>)}{remaining.length>0 && <details><summary>Advanced execution settings</summary>{fields(remaining)}</details>}</> : fields(remaining)}
      <details><summary>Definition and raw execution settings</summary><p className="muted">{selected?.import_target}</p><StructuredField label="Raw execution settings" value={value} onChange={next=>onChange(next as Component,true)}/></details>
    </div>
  </section>;
}
