"use client";
import { useState } from "react";
import { Button } from "./ui/button";
import { matchesSchema, resolveSchema, schemaDefault, variantLabel } from "../model/schema";
import type { Catalog, Component, RecordValue, Schema } from "../model/types";

type Errors={loc:(string|number)[];message:string}[];
type Context={root?:Schema;catalog?:Catalog;errors?:Errors;focusPath?:string};

export function StructuredField({ label, value, onChange, error, path }: {
  label:string; value:unknown; onChange:(value:unknown)=>void|Promise<void>; error?:string;path?:string;
}) {
  const original = JSON.stringify(value ?? null, null, 2);
  const [text, setText] = useState(original);
  const [failure, setFailure] = useState("");
  const [applying,setApplying] = useState(false);
  return <label className="field structured">{label}
    <textarea aria-label={label} data-field={path} spellCheck={false} value={text} onChange={event=>{setText(event.target.value);setFailure("");}} rows={Math.min(10,Math.max(3,text.split("\n").length))} />
    {text !== original && <Button disabled={applying} onClick={async()=>{setApplying(true);try {await onChange(JSON.parse(text));setFailure("");}catch(error) {setFailure(`${error instanceof Error ? error.message : "Invalid value"}. Your buffer is retained.`);}finally{setApplying(false);}}}>Apply {label}</Button>}
    {(failure || error) && <span className="field-error" role="alert">{failure || error}</span>}
  </label>;
}

export function ComponentFields({component,catalog,onChange,errors=[],prefix,focusPath,kind}: {
  component:Component;onChange:(value:Component,validate?:boolean)=>void|Promise<void>;prefix:string;kind?:string;
}&Context) {
  const entry=catalog?.components.find(item=>item.name===component.type && item.version===component.version && (!kind || item.kind===kind));
  const entries=catalog?.components.filter(item=>item.kind===(kind ?? entry?.kind)) ?? [];
  return <div className="fields component-fields"><label className="field">Component
    <select aria-label={`Component ${prefix}`} data-field={prefix.replace(/params\.$/,"type")} value={`${component.type}:${component.version}`} onChange={event=>{
      const selected=entries.find(item=>`${item.name}:${item.version}`===event.target.value);
      if(selected)onChange({type:selected.name,version:selected.version,params:schemaDefault(selected.schema) as RecordValue});
    }}><option value={`${component.type}:${component.version}`}>{component.type} · v{component.version}</option>
      {entries.filter(item=>item!==entry).map(item=><option key={`${item.name}:${item.version}`} value={`${item.name}:${item.version}`}>{item.name} · v{item.version}</option>)}
    </select></label>
    <Fields value={component.params} schema={entry?.schema} root={entry?.schema} catalog={catalog} errors={errors} prefix={prefix} focusPath={focusPath} onChange={(params,validate)=>onChange({...component,params},validate)}/>
    <details><summary>Expert: definition & raw component</summary><p className="muted">{entry?.import_target ?? "Unknown installed definition; values are retained."}</p>
      <StructuredField key={JSON.stringify(component)} label="Raw component" value={component} onChange={value=>onChange(value as Component,true)}/>
      {entry && <pre>{JSON.stringify(entry.schema,null,2)}</pre>}
    </details></div>;
}

export function Fields({ value, schema, root=schema, onChange, prefix = "", errors = [], basic, catalog, focusPath }: {
  value:RecordValue; schema?:Schema; onChange:(value:RecordValue,validate?:boolean)=>void|Promise<void>;
  prefix?:string; basic?:string[];
}&Context) {
  const [advanced, setAdvanced] = useState(false);
  const properties = resolveSchema(schema,root)?.properties ?? {};
  const keys = [...new Set([...(schema?.required ?? []), ...(basic ?? []), ...Object.keys(properties), ...Object.keys(value)])];
  return <div className="fields">
    {basic && <div className="segmented"><Button aria-pressed={!advanced} onClick={()=>setAdvanced(false)}>Basic</Button><Button aria-pressed={advanced} onClick={()=>setAdvanced(true)}>Advanced</Button></div>}
    {keys.filter(key=>!basic || advanced || basic.includes(key) || focusPath?.startsWith(`${prefix}${key}`) || errors.some(error=>error.loc.join(".").startsWith(`${prefix}${key}`))).map(key=>
      <ValueField key={key} label={properties[key]?.title ?? key.replaceAll("_"," ")} value={value[key]===undefined ? properties[key]?.default : value[key]}
        schema={properties[key]} root={root} catalog={catalog} errors={errors} focusPath={focusPath} path={`${prefix}${key}`}
        onChange={(next,validate)=>onChange({...value,[key]:next},validate)}/>) }
  </div>;
}

export function ValueField({label,value,schema={},path,onChange,root=schema,catalog,errors=[],focusPath}: {
  label:string;value:unknown;schema?:Schema;path:string;onChange:(value:unknown,validate?:boolean)=>void|Promise<void>;
}&Context) {
  const field=resolveSchema(schema,root);
  const error=errors.filter(error=>error.loc.join(".")===path).map(error=>error.message).join("; ");
  const context={root,catalog,errors,focusPath};
  const control={"aria-label":label,"data-field":path,"aria-invalid":!!error};
  const variants=field.anyOf ?? field.oneOf;
  const expert=<details className="expert-field"><summary>Expert: {label}</summary><StructuredField key={JSON.stringify(value)} label={label} path={`${path}.$raw`} value={value} onChange={next=>onChange(next,true)} error={error}/></details>;
  if(variants){
    const index=Math.max(0,variants.findIndex(item=>matchesSchema(value,item,root)));
    return <div className="field-group"><label className="field">{label}<select {...control} aria-label={`${label} mode`} value={index} onChange={event=>{
      const selected=resolveSchema(variants[Number(event.target.value)],root);
      const initial=schemaDefault(selected,root);
      onChange(selected.type==="object" && value && typeof value==="object" && !Array.isArray(value) ?
        {...initial as RecordValue,...Object.fromEntries(Object.entries(value).filter(([key])=>selected.properties?.[key] && selected.properties[key].const===undefined))} : initial);
    }}>{variants.map((item,index)=><option key={index} value={index}>{variantLabel(item,root)}</option>)}</select></label>
      <ValueField label={label} path={path} value={value} schema={variants[index]} onChange={onChange} {...context}/></div>;
  }
  if(field.type==="null")return <p className="muted">{label}: not set. Runtime defaults apply where supported.</p>;
  const kinds:Record<string,string>={DataSourceDefinition:"data_source",LoaderBackendDefinition:"loader_backend",LayerDefinition:"layer",EvalAlgorithmDefinition:"eval_algorithm"};
  const component=value as Component|undefined;
  const kind=kinds[field.title ?? ""];
  if(kind || (component?.type && component?.version && component?.params)){
    if(!component?.type)return <div className="field"><span>{label}</span><select {...control} value="" onChange={event=>{
      const entry=catalog?.components.find(item=>`${item.name}:${item.version}`===event.target.value && item.kind===kind);
      if(entry)onChange({type:entry.name,version:entry.version,params:schemaDefault(entry.schema)});
    }}><option value="">Choose installed component…</option>{catalog?.components.filter(item=>item.kind===kind).map(item=><option key={`${item.name}:${item.version}`} value={`${item.name}:${item.version}`}>{item.name} · v{item.version}</option>)}</select></div>;
    return <div className="field-group"><h3>{label}</h3><ComponentFields component={component} kind={kind} prefix={`${path}.params.`} onChange={onChange} {...context}/>{error && <span className="field-error">{error}</span>}</div>;
  }
  let input;
  const choices=field.enum ?? (field.const===undefined ? undefined : [field.const]);
  if(choices)input=<select {...control} value={String(value ?? "")} onChange={event=>onChange(choices.find(item=>String(item)===event.target.value))}><option value="" disabled>Select…</option>{value!=null && !choices.includes(value) && <option value={String(value)} disabled>{String(value)} (not in installed choices)</option>}{choices.map(item=><option key={String(item)} value={String(item)}>{String(item)}</option>)}</select>;
  else if(field.type==="boolean" || typeof value==="boolean")input=<input {...control} type="checkbox" checked={Boolean(value)} onChange={event=>onChange(event.target.checked)}/>;
  else if(["number","integer"].includes(field.type ?? "") || typeof value==="number")input=<input {...control} type="number" value={value as number ?? ""} step={field.type==="integer" ? 1 : "any"} min={field.minimum} max={field.maximum} onChange={event=>onChange(event.target.value==="" ? null : Number(event.target.value))}/>;
  else if(field.type==="string" || typeof value==="string")input=<input {...control} value={String(value ?? "")} minLength={field.minLength} maxLength={field.maxLength} onChange={event=>onChange(event.target.value)}/>;
  if(input)return <label className={`field ${field.type==="boolean" || typeof value==="boolean" ? "checkbox" : ""}`}>{label}{input}{field.description && <small className="muted">{field.description}</small>}{error && <span className="field-error" role="alert">{error}</span>}</label>;
  if(field.type==="array" || Array.isArray(value)){
    const items=Array.isArray(value) ? value : [];
    return <div className="field-group" data-field={path} tabIndex={-1}><h3>{label}</h3>{items.map((item,index)=><div className="collection-row" key={index}>
      <ValueField label={`${label} ${index+1}`} value={item} schema={field.prefixItems?.[index] ?? field.items} path={`${path}.${index}`} onChange={(next,validate)=>onChange(items.map((item,i)=>i===index ? next : item),validate)} {...context}/>
      {!field.prefixItems && <Button aria-label={`Remove ${label} ${index+1}`} onClick={()=>onChange(items.filter((_,i)=>i!==index))}>Remove</Button>}
    </div>)}{!field.prefixItems && <Button disabled={field.maxItems!==undefined && items.length>=field.maxItems} onClick={()=>onChange([...items,schemaDefault(field.items ?? {type:"string"},root)])}>Add {label}</Button>}{error && <span className="field-error" role="alert">{error}</span>}{expert}</div>;
  }
  if(field.type==="object" || (value && typeof value==="object")){
    const object=(value ?? {}) as RecordValue;
    if(field.properties)return <div className="field-group" data-field={path} tabIndex={-1}><h3>{label}</h3><Fields value={object} schema={field} prefix={`${path}.`} onChange={onChange} {...context}/>{error && <span className="field-error" role="alert">{error}</span>}{expert}</div>;
    return <MappingField label={label} value={object} schema={field} path={path} onChange={onChange} {...context}/>;
  }
  return <div className="field-group"><span>{label}</span><p className="muted">No typed value schema; use Expert.</p>{expert}</div>;
}

function MappingField({label,value,schema,path,onChange,...context}:{label:string;value:RecordValue;schema:Schema;path:string;onChange:(value:unknown,validate?:boolean)=>void|Promise<void>}&Context){
  const [key,setKey]=useState("");
  const itemSchema=typeof schema.additionalProperties==="object" ? schema.additionalProperties : {};
  return <div className="field-group" data-field={path} tabIndex={-1}><h3>{label}</h3>{Object.entries(value).map(([key,item])=><div className="mapping-row" key={key}>
    <input aria-label={`${label} key ${key}`} defaultValue={key} onBlur={event=>{
      const next=event.target.value.trim();
      if(!next || (next!==key && Object.hasOwn(value,next))){event.target.value=key;return;}
      onChange(Object.fromEntries(Object.entries(value).map(([name,item])=>[name===key ? next : name,item])));
    }}/><ValueField label={`${label} ${key}`} value={item} schema={itemSchema} path={`${path}.${key}`} onChange={(next,validate)=>onChange({...value,[key]:next},validate)} {...context}/>
    <Button aria-label={`Remove ${label} ${key}`} onClick={()=>onChange(Object.fromEntries(Object.entries(value).filter(([name])=>name!==key)))}>Remove</Button>
  </div>)}<div className="toolbar"><input aria-label={`New ${label} key`} value={key} onChange={event=>setKey(event.target.value)}/><Button disabled={!key.trim()||Object.hasOwn(value,key.trim())} onClick={()=>{onChange({...value,[key.trim()]:schemaDefault(Object.keys(itemSchema).length ? itemSchema : {type:"string"},context.root)});setKey("");}}>Add {label} entry</Button></div>
    <details><summary>Expert: {label}</summary><StructuredField key={JSON.stringify(value)} label={label} value={value} path={`${path}.$raw`} onChange={next=>onChange(next,true)}/></details>
  </div>;
}
