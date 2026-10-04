"use client";
import { useId } from "react";
import type { Catalog, Component, RecordValue } from "../model/types";
import { schemaDefault } from "../model/schema";
import { Fields } from "./fields";

export function ExecutionSettings({value,catalog,onChange,errors=[],suggestions}: {
  value:Component;catalog?:Catalog;onChange:(value:Component)=>void;
  errors?:{loc:(string|number)[];message:string}[];suggestions?:Record<string,string[]>;
}) {
  const id=useId();
  const entries=catalog?.execution_backends ?? [];
  const selected=entries.find(entry=>entry.type===value.type && entry.version===value.version);
  const basic=selected ? [...new Set(Object.entries(selected.presentation ?? {})
    .filter(([name])=>name!=="expert").flatMap(([,paths])=>paths))] : undefined;
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
    <p className="muted">{selected?.import_target ?? "Selected definition is missing. Configuration retained; launch blocked."}</p>
    {selected?.diagnostics.map(message=><p className="field-error" key={message}>{message}</p>)}
    <Fields key={`${value.type}:${value.version}`} value={value.params} schema={selected?.schema} root={selected?.schema}
      prefix="execution.params." catalog={catalog} basic={basic?.length ? basic : undefined}
      suggestions={suggestions} errors={errors} onChange={params=>onChange({...value,params})}/>
  </section>;
}
