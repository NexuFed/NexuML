"use client";
import { useState } from "react";
import { Button } from "./ui/button";
import type { RecordValue, Schema } from "../model/types";

export function StructuredField({ label, value, onChange, error }: {
  label:string; value:unknown; onChange:(value:unknown)=>void|Promise<void>; error?:string;
}) {
  const original = JSON.stringify(value ?? null, null, 2);
  const [text, setText] = useState(original);
  const [failure, setFailure] = useState("");
  const [applying,setApplying] = useState(false);
  return <label className="field structured">{label}
    <textarea aria-label={label} spellCheck={false} value={text} onChange={event=>{setText(event.target.value);setFailure("");}} rows={Math.min(10,Math.max(3,text.split("\n").length))} />
    {text !== original && <Button disabled={applying} onClick={async()=>{setApplying(true);try {await onChange(JSON.parse(text));setFailure("");}catch(error) {setFailure(`${error instanceof Error ? error.message : "Invalid value"}. Your buffer is retained.`);}finally{setApplying(false);}}}>Apply {label}</Button>}
    {(failure || error) && <span className="field-error" role="alert">{failure || error}</span>}
  </label>;
}

export function Fields({ value, schema, root, onChange, prefix = "", errors = [], basic }: {
  value:RecordValue; schema?:Schema; root?:Schema; onChange:(value:RecordValue,validate?:boolean)=>void|Promise<void>;
  prefix?:string; errors?:{loc:(string|number)[];message:string}[]; basic?:string[];
}) {
  const [advanced, setAdvanced] = useState(false);
  const properties = schema?.properties ?? {};
  const keys = [...new Set([...(schema?.required ?? []), ...Object.keys(properties), ...Object.keys(value)])];
  return <div className="fields">
    {basic && <div className="segmented"><Button aria-pressed={!advanced} onClick={()=>setAdvanced(false)}>Basic</Button><Button aria-pressed={advanced} onClick={()=>setAdvanced(true)}>Advanced</Button></div>}
    {keys.filter(key=>!basic || advanced || basic.includes(key)).map(key=>{
      const descriptor = properties[key] ?? {};
      const fieldSchema = descriptor.$ref ? root?.$defs?.[descriptor.$ref.split("/").at(-1)!] ?? descriptor : descriptor;
      const current = value[key] === undefined ? descriptor.default : value[key];
      const error = errors.filter(error=>error.loc.join(".") === `${prefix}${key}`).map(error=>error.message).join("; ");
      const update = (next:unknown,validate=false) => onChange({...value,[key]:next},validate);
      const label = descriptor.title ?? key.replaceAll("_", " ");
      if (fieldSchema.enum && fieldSchema.enum.every(item=>["string","number"].includes(typeof item))) return <label className="field" key={key}>{label}
        <select aria-label={label} value={String(current ?? "")} onChange={event=>update(fieldSchema.enum!.find(item=>String(item)===event.target.value))}>
          <option value="" disabled>Select…</option>{fieldSchema.enum.map(item=><option key={String(item)}>{String(item)}</option>)}
        </select>{error && <span className="field-error">{error}</span>}
      </label>;
      if (fieldSchema.type === "boolean" || typeof current === "boolean") return <label className="field checkbox" key={key}>
        <input type="checkbox" aria-label={label} checked={Boolean(current)} onChange={event=>update(event.target.checked)} />{label}
      </label>;
      if (["number","integer"].includes(fieldSchema.type ?? "") || typeof current === "number") return <label className="field" key={key}>{label}
        <input type="number" aria-label={label} value={current as number ?? ""} step={fieldSchema.type === "integer" ? 1 : "any"} min={fieldSchema.minimum} max={fieldSchema.maximum}
          onChange={event=>update(event.target.value === "" ? null : Number(event.target.value))} aria-invalid={!!error} />
        {error && <span className="field-error">{error}</span>}
      </label>;
      if (fieldSchema.type === "string" || typeof current === "string") return <label className="field" key={key}>{label}
        <input aria-label={label} value={String(current ?? "")} onChange={event=>update(event.target.value)} aria-invalid={!!error} />
        {error && <span className="field-error">{error}</span>}
      </label>;
      return <StructuredField key={`${key}:${JSON.stringify(current)}`} label={key} value={current} onChange={next=>update(next,true)} error={error} />;
    })}
  </div>;
}
