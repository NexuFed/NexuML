"use client";
import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Dialog } from "@base-ui/react/dialog";
import { ApiError, request } from "../model/client";
import type { Catalog, ConnectionInfo, Document, Operation, ResourceSnapshot } from "../model/types";
import { ExecutionSettings } from "./execution-settings";
import { Button } from "./ui/button";

export function RunReview({initial,connection,catalog,sourceChanged,checkpoint,close,accepted}: {
  initial:Document;connection:ConnectionInfo;catalog?:Catalog;sourceChanged:boolean;checkpoint:string;
  close:()=>void;accepted:(operation:Operation,document:Document)=>void;
}) {
  const [config,setConfig]=useState(()=>structuredClone(initial.data));
  const body={data:config,stage_order:initial.stage_order,trainer_checkpoint:checkpoint || null};
  const key=JSON.stringify(body);
  const currentKey=useRef(key);
  useEffect(()=>{currentKey.current=key;},[key]);
  const [settled,setSettled]=useState(key);
  const [review,setReview]=useState<{key:string;document:Document}|null>(null);
  const [failure,setFailure]=useState<{key:string;error:Error}|null>(null);
  const [pending,setPending]=useState(false);
  const confirming=useRef(false);
  const summary=useRef<HTMLDivElement>(null);
  const fields=useRef<HTMLDivElement>(null);
  const backend=catalog?.execution_backends?.find(entry=>entry.type===config.execution.type && entry.version===config.execution.version);
  useEffect(()=>{const timer=setTimeout(()=>setSettled(key),350);return ()=>clearTimeout(timer);},[key]);
  const discovery=useQuery({queryKey:["execution-discovery",key],enabled:settled===key && !!backend?.available,
    queryFn:()=>request<ResourceSnapshot>(connection,"/execution/discover",{data:config,stage_order:initial.stage_order}),staleTime:0});
  const snapshot=discovery.data;
  const reviewed=review?.key===key ? review.document : null;
  const error=failure?.key===key ? failure.error : null;
  const errors=error instanceof ApiError ? error.fields : [];
  const blocked=sourceChanged || !backend?.available || pending || snapshot?.api_supported===false || snapshot?.submission_allowed===false;
  const suggestions=Object.fromEntries(Object.entries(snapshot?.options ?? {}).map(([path,values])=>[`execution.params.${path}`,values]));
  const fail=(error:unknown,captured:string)=>{
    if(currentKey.current!==captured)return;
    setFailure({key:captured,error:error instanceof Error ? error : new Error(String(error))});
    requestAnimationFrame(()=>summary.current?.focus());
  };
  const prepare=async()=>{
    if(confirming.current || blocked)return;
    confirming.current=true;setPending(true);const captured=key;
    try {
      const document=await request<Document>(connection,"/train/prepare",body);
      if(currentKey.current===captured){setReview({key:captured,document});setFailure(null);}
    }catch(error){fail(error,captured);}finally{confirming.current=false;setPending(false);}
  };
  const launch=async()=>{
    if(confirming.current || blocked || !reviewed)return;
    confirming.current=true;setPending(true);const captured=key;
    try {
      const operation=await request<Operation>(connection,"/train",{data:reviewed.data,stage_order:reviewed.stage_order,
        trainer_checkpoint:checkpoint || null,template_revision:reviewed.launch_review?.template_revision ?? null});
      accepted(operation,reviewed);
    }catch(error){setReview(null);fail(error,captured);}finally{confirming.current=false;setPending(false);}
  };
  const focusField=(path:string)=>{
    const controls=[...fields.current?.querySelectorAll<HTMLElement>("[data-field]") ?? []];
    const control=controls.filter(control=>path===control.dataset.field || path.startsWith(`${control.dataset.field}.`))
      .sort((a,b)=>(b.dataset.field?.length ?? 0)-(a.dataset.field?.length ?? 0))[0];
    if(control){for(let parent=control.parentElement;parent;parent=parent.parentElement)if(parent instanceof HTMLDetailsElement)parent.open=true;control.focus();control.scrollIntoView({block:"nearest"});}
  };
  return <Dialog.Root open onOpenChange={open=>{if(!open&&!confirming.current)close();}}><Dialog.Portal>
    <Dialog.Backdrop className="dialog-backdrop"/><Dialog.Popup className="dialog-popup run-dialog" finalFocus={()=>document.querySelector<HTMLButtonElement>("[data-run-trigger]")}>
      <Dialog.Title>Run your scenario</Dialog.Title><Dialog.Description>Choose where NexuML runs. Review is read-only; only confirmation launches work. Cancel discards these edits.</Dialog.Description>
      {error && <div className="execution-error" role="alert" tabIndex={-1} ref={summary}><h3>Review needs attention</h3><p>{error.message}</p>
        {errors.map((field,index)=><Button key={index} onClick={()=>focusField(field.loc.join("."))}>{field.loc.join(".")}: {field.message}</Button>)}</div>}
      {sourceChanged && <p className="field-error" role="alert">Source draft changed. Cancel and reopen Run to review the current draft.</p>}
      <div className="run-grid"><div ref={fields}><fieldset disabled={pending}>
        <ExecutionSettings value={config.execution} catalog={catalog} errors={errors} suggestions={suggestions}
          onChange={async(execution,validate)=>{const next={...config,execution};if(validate){await request<Document>(connection,"/config/validate",{data:next,stage_order:initial.stage_order});if(currentKey.current!==key)throw new Error("Selection changed during validation; apply again");}setConfig(next);}}/></fieldset></div>
        <section className="capacity-panel" aria-label="Sourced execution capacity"><div className="section-title"><h3>Capacity</h3>
          <Button disabled={discovery.isFetching || !backend?.available} onClick={()=>void discovery.refetch()}>Refresh capacity</Button></div>
          <p role="status">{discovery.isFetching ? "Inspecting selected target…" : discovery.isError ? "Inspection failed — capacity unknown" : snapshot ? "Advisory snapshot — not reserved" : "Capacity unknown"}</p>
          {discovery.error && <p className="field-error">{discovery.error.message}</p>}
          {snapshot && <Capacity snapshot={snapshot} stale={discovery.isError}/>}
          <p className="muted">Incomplete capacity alone does not block launch. Native preflight checks permissions, admission and known constraints; the scheduler may queue your workload.</p>
        </section></div>
      <dl className="result-summary"><dt>Scenario / epochs</dt><dd>{config.name} / {String(config.training.max_epochs)}</dd><dt>Resume checkpoint</dt><dd>{checkpoint || "New training"}</dd>
        <dt>Selected target</dt><dd>{snapshot?.target ?? "Selected configuration (see review)"}{snapshot?.namespace ? ` / ${snapshot.namespace}` : ""}</dd>
        <dt>Source revision</dt><dd><code>{initial.semantic_revision}</code></dd>
        <dt>Review status</dt><dd role="status">{pending ? "Validating selected NexuML…" : reviewed ? "Exact selection reviewed; scheduling is not guaranteed" : "Selection changed or not reviewed"}</dd></dl>
      <details><summary>Expert: frozen launch settings and remote handoff</summary><pre>{reviewed?.yaml ?? "Review this selection to capture exact launch settings."}</pre>
        {reviewed?.launch_review && <pre>{JSON.stringify(reviewed.launch_review,null,2)}</pre>}</details>
      <div className="toolbar run-actions"><Button disabled={pending} onClick={close}>Cancel</Button>
        <Button className={reviewed ? undefined : "primary"} disabled={blocked} onClick={()=>void prepare()}>Review selection</Button>
        <Button className={reviewed ? "primary" : undefined} disabled={blocked || !reviewed} onClick={()=>void launch()}>Run on {backend?.label ?? config.execution.type}</Button></div>
    </Dialog.Popup></Dialog.Portal></Dialog.Root>;
}

function Capacity({snapshot,stale}:{snapshot:ResourceSnapshot;stale:boolean}) {
  const [now,setNow]=useState(()=>Date.now());
  useEffect(()=>{const timer=setInterval(()=>setNow(Date.now()),15000);return ()=>clearInterval(timer);},[]);
  const age=Math.max(0,Math.floor((now-Date.parse(snapshot.observed_at))/1000));
  const quantities=(values:Record<string,number>|null)=>values===null ? "Unknown" : Object.entries(values)
    .map(([name,value])=>`${name}: ${value.toLocaleString()} ${snapshot.units[name] ?? "resource units (not physical GPUs)"}`).join(" · ") || "None reported";
  return <><p className="muted">{snapshot.source} · {snapshot.target ?? "selected process"}<br/>
    {age}s old · {stale || age>30 ? "Stale — refresh" : snapshot.complete ? "Complete source scope" : "Partial / unresolved constraints"}</p>
    {snapshot.submission_allowed===false && <p className="field-error" role="alert">Submission denied by the selected target.</p>}
    {snapshot.api_supported===false && <p className="field-error" role="alert">Required native API is unavailable.</p>}
    <details open={!snapshot.complete}><summary>Capacity details</summary><dl className="result-summary"><dt>Visible nodes</dt><dd>{snapshot.visible_nodes ?? "Unknown"}</dd>
      <dt>Matching nodes</dt><dd>{snapshot.matching_nodes ?? "Unknown"}</dd>
      <dt>Allocatable</dt><dd>{quantities(snapshot.allocatable)}</dd><dt>Estimated unallocated</dt><dd>{quantities(snapshot.unallocated)}</dd>
      <dt>Namespace quota remaining</dt><dd>{quantities(snapshot.quota_remaining)}</dd>
      <dt>Submission permission</dt><dd>{snapshot.submission_allowed===null ? "Unknown; preflight checks" : snapshot.submission_allowed ? "Allowed" : "Denied"}</dd></dl></details>
    {snapshot.diagnostics.map(message=><p className="muted" key={message}>{message}</p>)}
    <details><summary>Matching nodes / workload roles</summary><pre>{JSON.stringify({roles:snapshot.roles,nodes:snapshot.nodes,ray_clusters:snapshot.ray_clusters},null,2)}</pre></details></>;
}
