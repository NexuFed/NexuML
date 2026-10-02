"use client";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Download, Square } from "lucide-react";
import { request } from "../model/client";
import type { ConnectionInfo, Document, Operation, RecordValue } from "../model/types";
import { useObservation } from "./use-observation";
import { Button } from "./ui/button";

export function Execution({connection, identity, choose, artifacts=false, report}: {
  connection:ConnectionInfo;identity:string;choose:(id:string)=>void;artifacts?:boolean;report:(error:unknown)=>void;
}) {
  const list=useQuery({queryKey:["operations"],queryFn:()=>request<{operations:Operation[]}>(connection,"/operations"),refetchInterval:2000});
  const operation=useQuery({queryKey:["operation",identity],queryFn:()=>request<Operation>(connection,`/operations/${identity}`),enabled:!!identity,refetchInterval:1000});
  const observation=useObservation(connection,identity);
  const [output,setOutput]=useState("exported-model");
  const [format,setFormat]=useState("train_package");
  const [launch,setLaunch]=useState<{id:string;yaml:string}|null>(null);
  const [acting,setActing]=useState(false);
  const current=operation.data;
  const metrics=observation.events.filter(event=>event.kind==="metrics");
  const keys=[...new Set(metrics.flatMap(event=>Object.keys(event.payload.values as RecordValue ?? {})))];
  const act=async(action:()=>Promise<void>)=>{setActing(true);try {await action();} catch(error){report(error);} finally{setActing(false);}};
  const download=async(index:number,path:string)=>{
    const response=await fetch(`${connection.api}/api/v1/operations/${identity}/artifacts/${index}`,{headers:{Authorization:`Bearer ${connection.token}`}});
    if(!response.ok) throw new Error("Artifact no longer exists or is outside authorized roots.");
    const url=URL.createObjectURL(await response.blob());const link=document.createElement("a");link.href=url;link.download=path.split(/[\\/]/).at(-1)!;link.click();URL.revokeObjectURL(url);
  };
  return <section className="execution-view">
    <div className="view-title"><div><span className="eyebrow">{artifacts?"RETAINED OUTPUTS":"LIVE OBSERVATION"}</span><h1>{artifacts?"Artifacts":"Execution"}</h1></div>
      <label className="field">Observed operation<select aria-label="Observed operation" value={identity} onChange={event=>choose(event.target.value)}>
        <option value="">Select an operation</option>{list.data?.operations.map(item=><option value={item.id} key={item.id}>{item.kind} · {item.id.slice(0,8)} · {item.status}</option>)}
      </select></label></div>
    {!current ? <div className="empty-panel">Start an explicit build or training operation to inspect real results. Unrelated CLI runs are not tracked.</div> : <>
      <div className="status-grid"><div><span>Status</span><strong>{current.status}</strong></div><div><span>Connection</span><strong>{operation.isError ? "Status stale — API unavailable" : observation.state}</strong></div><div><span>Source revision</span><code>{current.semantic_revision.slice(0,12)}</code></div><div><span>Telemetry</span><strong>{current.telemetry}</strong></div></div>
      {observation.gap && <p role="alert">Replay history is incomplete. Inspect the retained logs and terminal result.</p>}
      <div className="toolbar"><Button disabled={acting || current.status!=="running" || !current.cancellation} onClick={()=>act(async()=>{await request(connection,`/operations/${identity}/cancel`,{});await operation.refetch();})}><Square size={14}/>Stop owned local operation</Button>
        {!current.cancellation && <span className="muted">Remote stop unsupported. Driver exit does not stop workers.</span>}
        <Button onClick={()=>act(async()=>setLaunch({id:identity,yaml:(await request<Document>(connection,`/operations/${identity}/config`)).yaml}))}>Inspect launch config</Button></div>
      {launch?.id===identity && <details open><summary>Frozen launch configuration</summary><pre>{launch.yaml}</pre></details>}
      {!artifacts && <><div className="metric-grid">{keys.map(key=>{
        const points=metrics.map(event=>({step:Number(event.payload.step),value:Number((event.payload.values as RecordValue)?.[key])})).filter(point=>Number.isFinite(point.value));
        const min=Math.min(...points.map(point=>point.value)), max=Math.max(...points.map(point=>point.value));
        return <article className="metric" key={key}><span>{key}</span><strong>{points.at(-1)?.value.toPrecision(5)}</strong>
          <svg viewBox="0 0 400 100" role="img" aria-label={`${key}, ${points.length} observed samples, latest ${points.at(-1)?.value}`}>
            <polyline fill="none" stroke="currentColor" strokeWidth="2" points={points.map((point,index)=>`${20+index/Math.max(1,points.length-1)*360},${80-(point.value-min)/Math.max(.000001,max-min)*60}`).join(" ")} />
          </svg><small>Observed callback samples · range {min.toPrecision(3)}–{max.toPrecision(3)}</small></article>;
      })}{!keys.length && <div className="empty-panel">No scalar samples observed yet. No simulated telemetry.</div>}</div>
        <details open><summary>Process logs</summary><pre className="logs" aria-label="Process logs">{observation.events.filter(event=>event.kind==="log").map(event=>String(event.payload.text)).join("") || "Waiting for actual process output…"}</pre></details>
        {observation.truncated && <p className="muted">Showing the latest 2,000 observations. Complete logs remain on disk.</p>}
        <details open><summary>Actual terminal result</summary><pre>{JSON.stringify(current.result ?? {},null,2)}</pre></details></>}
      <h2>Available artifacts</h2><div className="artifact-list">{current.artifacts.map((artifact,index)=><div key={`${artifact.path}:${index}`}><div><span>{artifact.kind}</span><code>{artifact.path}</code></div><Button aria-label={`Download ${artifact.path}`} onClick={()=>act(()=>download(index,artifact.path))}><Download size={16}/></Button></div>)}{!current.artifacts.length && <p className="muted">No artifact has been confirmed. Stopping does not promise a checkpoint.</p>}</div>
      <div className="export-controls"><label className="field">Export format<select value={format} onChange={event=>setFormat(event.target.value)}><option value="train_package">Training package</option><option value="safetensors">SafeTensors</option><option value="onnx">ONNX (installed extra required)</option></select></label>
        <label className="field">Export path<input value={output} onChange={event=>setOutput(event.target.value)}/></label>
        <Button className="primary" disabled={acting || current.status!=="succeeded" || current.kind!=="train" || !current.artifacts.some(artifact=>artifact.kind==="checkpoint")} onClick={()=>act(async()=>{const launched=await request<Operation>(connection,"/export",{source_id:identity,kind:format,output});choose(launched.id);})}>Export trained checkpoint</Button></div>
    </>}
  </section>;
}
