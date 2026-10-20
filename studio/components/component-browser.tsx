"use client";
import { Box } from "lucide-react";
import { Button } from "./ui/button";
import { category } from "../model/catalog";
import type {Entry} from "../model/types";

export function ComponentBrowser({entries,query,blocked,add}:{entries:Entry[];query:string;blocked:boolean;add:(entry:Entry)=>void}){
  const filtered=entries.filter(entry=>["layer","data_source","eval_algorithm","loader_backend"].includes(entry.kind) &&
    `${entry.name} ${category(entry).join(" ")} ${entry.import_target}`.toLowerCase().includes(query.toLowerCase()));
  function group(items:Entry[],depth:number):React.ReactNode {
    const groups=Object.groupBy(items,entry=>category(entry)[depth] ?? "Components");
    return Object.entries(groups).sort(([a],[b])=>a.localeCompare(b)).map(([name,items])=>{
      const direct=items!.filter(entry=>category(entry).length<=depth+1);
      const nested=items!.filter(entry=>category(entry).length>depth+1);
      return <details className="component-category" open key={name}><summary>{name} <small>{items!.length}</small></summary>
        {direct.sort((a,b)=>a.name.localeCompare(b.name)||a.version.localeCompare(b.version)).map(entry=><Button key={`${entry.kind}:${entry.name}:${entry.version}`} disabled={blocked} draggable={!blocked}
          onDragStart={event=>{event.dataTransfer.setData("application/x-nexuml-component",JSON.stringify([entry.kind,entry.name,entry.version]));event.dataTransfer.effectAllowed="copy";}}
          onClick={()=>add(entry)}><Box size={15}/><span>{entry.name}<small>v{entry.version}</small></span><span>+</span></Button>)}
        {nested.length>0 && group(nested,depth+1)}
      </details>;
    });
  }
  return <div className="component-list">{group(filtered,0)}{!filtered.length && <p className="muted">No matching installed components.</p>}</div>;
}
