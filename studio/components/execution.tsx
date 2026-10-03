"use client";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Download, Square } from "lucide-react";
import { request } from "../model/client";
import { metricPoints } from "../model/metrics";
import type { ConnectionInfo, Document, Operation, RecordValue } from "../model/types";
import { useObservation } from "./use-observation";
import { Button } from "./ui/button";

export function Execution({connection, identity, choose, artifacts=false, report,editSettings,loadSettings}: {
  connection:ConnectionInfo;identity:string;choose:(id:string)=>void;artifacts?:boolean;report:(error:unknown)=>void;
  editSettings:()=>void;
  loadSettings:(identity:string)=>Promise<void>;
}) {
  const list=useQuery({queryKey:["operations"],queryFn:()=>request<{operations:Operation[]}>(connection,"/operations"),refetchInterval:2000});
  const operation=useQuery({queryKey:["operation",identity],queryFn:()=>request<Operation>(connection,`/operations/${identity}`),enabled:!!identity,refetchInterval:1000});
  const observation=useObservation(connection,identity);
  const [output,setOutput]=useState("exported-model");
  const [format,setFormat]=useState("train_package");
  const [launch,setLaunch]=useState<{id:string;yaml:string}|null>(null);
  const [acting,setActing]=useState(false);
  const current=operation.data;
  const progress=observation.progress;
  const total=typeof progress?.total==="number" && Number.isFinite(progress.total) && progress.total>0 ? progress.total : null;
  const batch=typeof progress?.batch==="number" ? progress.batch : null;
  const failure=current?.result?.error as {message?:string;fields?:{loc:(string|number)[];message:string}[]}|undefined;
  const metrics=observation.metrics;
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
      {!artifacts && <section className="run-progress" aria-label="Native operation progress"><div><strong>{current.status==="succeeded" ? "Completed" : progress?.phase ? String(progress.phase) : current.telemetry.includes("driver") ? "Backend reports driver logs only" : "Preparing runtime / data"}</strong><span>{current.status!=="running" ? current.status : observation.state!=="Connected" ? "Last observed — connection stale" : "Live"}</span></div>
        <progress aria-label="Operation progress" max={total ?? 1} value={current.status==="succeeded" ? total ?? 1 : total && batch!==null ? Math.min(batch,total) : current.status!=="running" ? 0 : undefined}/>
        <p className="muted">{progress?.phase==="train" && typeof progress.epoch==="number" ? `Epoch ${progress.epoch+1}${typeof progress.max_epochs==="number" && progress.max_epochs>0 ? ` / ${progress.max_epochs}` : ""} · ` : ""}{batch!==null ? `${batch}${total ? ` / ${total}` : ""} batches observed` : "No batch total available. Preparation and download updates appear in the process logs."}</p>
      </section>}
      {failure && <section className="execution-error" role="alert"><h2>Operation failed</h2><p>{failure.message ?? "The selected runtime rejected this operation. Inspect its process logs."}</p>
        {failure.fields?.map((field,index)=><p key={index}>{field.loc.join(".")}: {field.message}</p>)}
        <Button onClick={editSettings}>Edit draft settings for a new run</Button><Button disabled={acting} onClick={()=>act(()=>loadSettings(identity))}>Open frozen settings as draft</Button><p className="muted">These errors describe the frozen launch configuration, not later draft edits. Inspect that configuration below; editing the draft does not change this run.</p>
      </section>}
      <div className="toolbar"><Button disabled={acting || current.status!=="running" || !current.cancellation} onClick={()=>act(async()=>{await request(connection,`/operations/${identity}/cancel`,{});await operation.refetch();})}><Square size={14}/>Stop owned local operation</Button>
        {!current.cancellation && <span className="muted">Remote stop unsupported. Driver exit does not stop workers.</span>}
        <Button onClick={()=>act(async()=>setLaunch({id:identity,yaml:(await request<Document>(connection,`/operations/${identity}/config`)).yaml}))}>Inspect launch config</Button></div>
      {launch?.id===identity && <details open><summary>Frozen launch configuration</summary><pre>{launch.yaml}</pre></details>}
      {!artifacts && <><p className="muted">Select losses and metrics in Pipeline → Objectives &amp; metrics (<code>training.loss_keys</code> / <code>training.metric_keys</code>). Total loss is their weighted loss sum. Charts show this run’s reported values, not later draft edits; validation/test metrics appear after those phases complete.</p><div className="metric-grid">{keys.map(key=>{
        const points=metricPoints(metrics,key);
        if(!points.length)return null;
        const min=Math.min(...points.map(point=>point.value)), max=Math.max(...points.map(point=>point.value));
        const firstStep=Math.min(...points.map(point=>point.step)),lastStep=Math.max(...points.map(point=>point.step));
        const x=(step:number)=>55+(step-firstStep)/Math.max(1,lastStep-firstStep)*325;
        const y=(value:number)=>min===max ? 55 : 90-(value-min)/(max-min)*70;
        const latest=points.at(-1)!;
        return <article className="metric" key={key}><span>{key}</span><strong>{points.at(-1)?.value.toPrecision(5)}</strong>
          <svg viewBox="0 0 400 140" role="img" aria-label={`${key}, ${points.length} observed samples, latest ${latest.value} at optimizer step ${latest.step}`}>
            <path className="metric-axis" d="M55 15V100H385"/>
            <text className="metric-tick" x="48" y="24" textAnchor="end">{max.toPrecision(3)}</text>
            {min!==max && <text className="metric-tick" x="48" y="94" textAnchor="end">{min.toPrecision(3)}</text>}
            <polyline fill="none" stroke="currentColor" strokeWidth="2" points={points.map(point=>`${x(point.step)},${y(point.value)}`).join(" ")} />
            <circle cx={x(latest.step)} cy={y(latest.value)} r="3" fill="currentColor"><title>{latest.value} at step {latest.step}</title></circle>
            <text className="metric-tick" x="55" y="117">{firstStep}</text>
            {lastStep!==firstStep && <text className="metric-tick" x="380" y="117" textAnchor="end">{lastStep}</text>}
            <text className="metric-tick" x="220" y="136" textAnchor="middle">Optimizer step</text>
          </svg><small>{points.length} observed samples{points.length===1 ? " · waiting for the next measurement" : ""} · range {min.toPrecision(3)}–{max.toPrecision(3)}</small></article>;
      })}{!keys.length && <div className="empty-panel">No scalar samples observed yet. No simulated telemetry.</div>}</div>
        <details open><summary>Process logs</summary><pre className="logs" aria-label="Process logs">{observation.terminal || "Waiting for actual process output…"}</pre></details>
        {observation.truncated && <p className="muted">Showing the latest 2,000 observations. Complete logs remain on disk.</p>}
        <h2>Actual results</h2><ResultSummary value={Object.fromEntries(Object.entries(current.result ?? {}).filter(([key])=>!["error","artifacts"].includes(key)))}/>
        <details><summary>Expert: actual terminal result JSON</summary><pre>{JSON.stringify(current.result ?? {},null,2)}</pre></details></>}
      <h2>Available artifacts</h2><div className="artifact-list">{current.artifacts.map((artifact,index)=><div key={`${artifact.path}:${index}`}><div><span>{artifact.kind}</span><code>{artifact.path}</code></div><Button aria-label={`Download ${artifact.path}`} onClick={()=>act(()=>download(index,artifact.path))}><Download size={16}/></Button></div>)}{!current.artifacts.length && <p className="muted">No artifact has been confirmed. Stopping does not promise a checkpoint.</p>}</div>
      <div className="export-controls"><label className="field">Export format<select value={format} onChange={event=>setFormat(event.target.value)}><option value="train_package">Training package</option><option value="safetensors">SafeTensors</option><option value="onnx">ONNX (installed extra required)</option></select></label>
        <label className="field">Export path<input value={output} onChange={event=>setOutput(event.target.value)}/></label>
        <Button className="primary" disabled={acting || current.status!=="succeeded" || current.kind!=="train" || !current.artifacts.some(artifact=>artifact.kind==="checkpoint")} onClick={()=>act(async()=>{const launched=await request<Operation>(connection,"/export",{source_id:identity,kind:format,output});choose(launched.id);})}>Export trained checkpoint</Button></div>
    </>}
  </section>;
}

function ResultSummary({value,prefix=""}:{value:unknown;prefix?:string}):React.ReactNode {
  if(value && typeof value==="object"){
    const entries=Object.entries(value);
    if(!entries.length)return <p className="muted">No returned values yet.</p>;
    return <div className="result-group">{entries.map(([key,item])=><ResultSummary key={key} value={item} prefix={`${prefix}${prefix ? " · " : ""}${key.replaceAll("_"," ")}`}/>)}</div>;
  }
  return <dl className="result-summary"><dt>{prefix}</dt><dd>{value==null ? "Not set" : String(value)}</dd></dl>;
}
