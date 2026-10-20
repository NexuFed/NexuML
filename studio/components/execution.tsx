"use client";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Download, Square } from "lucide-react";
import { request } from "../model/client";
import { metricPoints } from "../model/metrics";
import { signature } from "../model/graph";
import type { ConnectionInfo, Document, Operation, RecordValue, Snapshot } from "../model/types";
import { useObservation } from "./use-observation";
import { Button } from "./ui/button";

export function Execution({connection, identity, choose, draft, report,editSettings,loadSettings}: {
  connection:ConnectionInfo;identity:string;choose:(id:string)=>void;draft?:Snapshot|null;report:(error:unknown)=>void;
  editSettings:()=>void;
  loadSettings:(identity:string)=>Promise<void>;
}) {
  const list=useQuery({queryKey:["operations"],queryFn:()=>request<{operations:Operation[]}>(connection,"/operations"),refetchInterval:2000});
  const operation=useQuery({queryKey:["operation",identity],queryFn:()=>request<Operation>(connection,`/operations/${identity}`),enabled:!!identity,refetchInterval:1000});
  const observation=useObservation(connection,identity);
  const [output,setOutput]=useState("exported-model");
  const [format,setFormat]=useState("train_package");
  const frozen=useQuery({queryKey:["operation-config",identity],enabled:!!identity,queryFn:()=>request<Document>(connection,`/operations/${identity}/config`)});
  const [tab,setTab]=useState({id:"",name:""});
  const [acting,setActing]=useState(false);
  const [nativeInspection,setNativeInspection]=useState<{id:string;result:unknown}|null>(null);
  const current=operation.data;
  const activeTab=tab.id===identity ? tab.name : current?.kind==="export" ? "Artifacts" : current?.kind==="train" && current.capabilities?.metrics!==false ? "Metrics" : "Logs";
  const tabs=["Metrics","Logs","Artifacts","Configuration"];
  const draftDiffers=draft && frozen.data?.data && frozen.data?.stage_order ? signature(draft)!==signature({config:frozen.data.data,order:frozen.data.stage_order,ids:{},positions:{}}) : null;
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
    <div className="view-title"><h1>Runs</h1>
      <label className="field">Observed operation<select aria-label="Observed operation" value={identity} onChange={event=>choose(event.target.value)}>
        <option value="">Select an operation</option>{list.data?.operations.map(item=><option value={item.id} key={item.id}>{item.kind} · {item.id.slice(0,8)} · {item.status}</option>)}
      </select></label></div>
    {!current ? <div className="empty-panel">Start an explicit build or training operation to inspect real results. Unrelated CLI runs are not tracked.</div> : <>
      <div className="status-grid"><div><span>Status</span><strong>{current.status}</strong></div><div><span>Connection</span><strong>{operation.isError ? "Status stale — API unavailable" : observation.state}</strong></div><div><span>Source revision</span><code>{current.semantic_revision.slice(0,12)}</code></div><div><span>Telemetry</span><strong>{current.telemetry}</strong></div></div>
      {observation.gap && <p role="alert">Replay history is incomplete. Inspect the retained logs and terminal result.</p>}
      {current.native_reference && <dl className="result-summary"><dt>Native workload</dt><dd>{current.native_reference.kind} · {current.native_reference.context} / {current.native_reference.namespace} / {current.native_reference.name}</dd><dt>Verified UID</dt><dd><code>{current.native_reference.uid}</code></dd></dl>}
      {current.kind==="train" && current.capabilities?.metrics!==false && <section className="run-progress" aria-label="Native operation progress"><div><strong>{current.status==="succeeded" ? "Completed" : progress?.phase ? String(progress.phase) : current.telemetry.includes("driver") ? "Backend reports driver logs only" : "Preparing runtime / data"}</strong><span>{current.status!=="running" ? current.status : observation.state!=="Connected" ? "Last observed — connection stale" : "Live"}</span></div>
        <progress aria-label="Operation progress" max={total ?? 1} value={current.status==="succeeded" ? total ?? 1 : total && batch!==null ? Math.min(batch,total) : current.status!=="running" ? 0 : undefined}/>
        <p className="muted">{progress?.phase==="train" && typeof progress.epoch==="number" ? `Epoch ${progress.epoch+1}${typeof progress.max_epochs==="number" && progress.max_epochs>0 ? ` / ${progress.max_epochs}` : ""} · ` : ""}{batch!==null ? `${batch}${total ? ` / ${total}` : ""} batches observed` : "No batch total available. Preparation and download updates appear in the process logs."}</p>
      </section>}
      {failure && <section className="execution-error" role="alert"><h2>Operation failed</h2><p>{failure.message ?? "The selected runtime rejected this operation. Inspect its process logs."}</p>
        {failure.fields?.map((field,index)=><p key={index}>{field.loc.join(".")}: {field.message}</p>)}
        <Button onClick={editSettings}>Edit draft settings for a new run</Button><Button disabled={acting} onClick={()=>act(()=>loadSettings(identity))}>Open frozen settings as draft</Button><p className="muted">These errors describe the frozen launch configuration, not later draft edits. Inspect that configuration below; editing the draft does not change this run.</p>
      </section>}
      {current.capabilities?.metrics===false && <p className="muted">Native job status and supported log tails only. Scalar metrics and training percentages are unavailable.</p>}
      {current.capabilities?.resume===false && <p className="muted">Trainer checkpoint resume is unsupported by this backend. Use a new Local run for supported checkpoint resume.</p>}
      <div className="toolbar"><Button disabled={acting || !["submitted","pending","running"].includes(current.status) || !current.cancellation || (!!current.capabilities?.native_reference && !current.native_reference)} onClick={()=>act(async()=>{await request(connection,`/operations/${identity}/cancel`,{});await operation.refetch();})}><Square size={14}/>{current.native_reference ? "Stop verified native job" : "Stop owned local operation"}</Button>
        {!current.cancellation && <span className="muted">Remote stop unsupported. Driver exit does not stop workers.</span>}
        {current.native_reference && <Button disabled={acting} onClick={()=>act(async()=>{const result=await request(connection,`/operations/${identity}/inspect`,{});setNativeInspection({id:identity,result});})}>Inspect native reference</Button>}
        <Button onClick={()=>setTab({id:identity,name:"Configuration"})}>Inspect launch config</Button></div>
      {draftDiffers!==null && <p className="muted">Frozen configuration · {draftDiffers ? "Draft differs" : "Draft matches"}</p>}
      <nav className="run-tabs" aria-label="Selected run views">{tabs.map(name=><Button key={name} aria-current={activeTab===name ? "page" : undefined} onClick={()=>setTab({id:identity,name})}>{name}</Button>)}</nav>
      {activeTab==="Configuration" && <section><h2>Frozen launch configuration</h2>{frozen.data?.yaml ? <><pre aria-label="Frozen launch YAML">{frozen.data.yaml}</pre><Button disabled={acting} onClick={()=>act(()=>loadSettings(identity))}>Open frozen settings as draft</Button></> : <p className="field-error">{frozen.error?.message ?? "Frozen configuration unavailable or loading. No draft comparison is claimed."}</p>}<details><summary>Operation details</summary><pre>{JSON.stringify(current,null,2)}</pre></details></section>}
      {nativeInspection?.id===identity && <details open><summary>Explicit native inspection (no resubmission)</summary><pre>{JSON.stringify(nativeInspection.result,null,2)}</pre></details>}
      {activeTab==="Metrics" && <><p className="muted">Total loss is the weighted objective configured in Pipeline → Objectives &amp; metrics. Curves show actual optimizer steps from this frozen run. Other metrics keep their own scales.</p><div className="metric-grid">
        <MetricChart title="Total loss" series={["train/loss","val/loss"].map(key=>({key,points:metricPoints(metrics,key)}))}/>
        {keys.filter(key=>!["train/loss","val/loss"].includes(key)).map(key=><MetricChart key={key} title={key} series={[{key,points:metricPoints(metrics,key)}]}/>)}
        {!keys.length && <div className="empty-panel">{current.capabilities?.metrics===false ? "Scalar telemetry is unsupported by this backend. Open Logs for available observations." : "No scalar samples observed yet. No simulated telemetry."}</div>}</div></>}
      {activeTab==="Logs" && <><h2>Process logs</h2><pre className="logs" aria-label="Process logs">{observation.terminal || "Waiting for actual process output…"}</pre>
        {observation.truncated && <p className="muted">Showing the latest 2,000 observations. Complete logs remain on disk.</p>}
        </>}
      {["Metrics","Logs"].includes(activeTab) && <section><h2>Actual results</h2><ResultSummary value={Object.fromEntries(Object.entries(current.result ?? {}).filter(([key])=>!["error","artifacts"].includes(key)))}/><details><summary>Actual terminal result JSON</summary><pre>{JSON.stringify(current.result ?? {},null,2)}</pre></details></section>}
      {activeTab==="Artifacts" && <><h2>Available artifacts</h2><div className="artifact-list">{current.artifacts.map((artifact,index)=><div key={`${artifact.path}:${index}`}><div><span>{artifact.kind}</span><code>{artifact.path}</code></div>{artifact.path.includes("://") ? <span className="muted">External reference — not a local download</span> : <Button aria-label={`Download ${artifact.path}`} onClick={()=>act(()=>download(index,artifact.path))}><Download size={16}/></Button>}</div>)}{!current.artifacts.length && <p className="muted">No artifact has been confirmed. Stopping does not promise a checkpoint.</p>}</div>
      <div className="export-controls"><label className="field">Export format<select value={format} onChange={event=>setFormat(event.target.value)}><option value="train_package">Training package</option><option value="safetensors">SafeTensors</option><option value="onnx">ONNX (installed extra required)</option></select></label>
        <label className="field">Export path<input value={output} onChange={event=>setOutput(event.target.value)}/></label>
        <Button className="primary" disabled={acting || current.status!=="succeeded" || current.kind!=="train" || current.capabilities?.artifacts===false || !current.artifacts.some(artifact=>artifact.kind==="checkpoint" && !artifact.path.includes("://"))} onClick={()=>act(async()=>{const launched=await request<Operation>(connection,"/export",{source_id:identity,kind:format,output});choose(launched.id);})}>Export trained checkpoint</Button></div></>}
    </>}
  </section>;
}

function MetricChart({title,series}:{title:string;series:{key:string;points:{step:number;value:number}[]}[]}) {
  const available=series.filter(item=>item.points.length);
  const points=available.flatMap(item=>item.points);if(!points.length)return null;
  const min=Math.min(...points.map(point=>point.value)),max=Math.max(...points.map(point=>point.value));
  const first=Math.min(...points.map(point=>point.step)),last=Math.max(...points.map(point=>point.step));
  const x=(step:number)=>55+(step-first)/Math.max(1,last-first)*325;
  const y=(value:number)=>min===max ? 55 : 90-(value-min)/(max-min)*70;
  return <article className="metric"><span>{title}</span><div className="metric-legend">{available.map(({key,points})=><div key={key}><span className={key==="val/loss" ? "validation-label" : "training-label"}>{key}</span><strong>{points.at(-1)!.value.toPrecision(5)}</strong><small>{points.length} observed samples{points.length===1 ? " · single measurement" : ""}</small></div>)}</div>
    <svg viewBox="0 0 400 140" role="img" aria-label={available.map(({key,points})=>`${key}, ${points.length} observed samples, latest ${points.at(-1)!.value} at optimizer step ${points.at(-1)!.step}`).join("; ")}>
      <path className="metric-axis" d="M55 15V100H385"/><text className="metric-tick" x="48" y="24" textAnchor="end">{max.toPrecision(3)}</text>{min!==max && <text className="metric-tick" x="48" y="94" textAnchor="end">{min.toPrecision(3)}</text>}
       {available.map(({key,points})=>{const latest=points.at(-1)!;return <g key={key} className={key==="val/loss" ? "validation-series" : "training-series"}><polyline fill="none" stroke="currentColor" strokeWidth="2" strokeDasharray={key==="val/loss" ? "6 4" : undefined} points={points.map(point=>`${x(point.step)},${y(point.value)}`).join(" ")}/><circle cx={x(latest.step)} cy={y(latest.value)} r="3" fill="currentColor"><title>{key}: {latest.value} at step {latest.step}</title></circle></g>;})}
      <text className="metric-tick" x="55" y="117">{first}</text>{last!==first && <text className="metric-tick" x="380" y="117" textAnchor="end">{last}</text>}<text className="metric-tick" x="220" y="136" textAnchor="middle">Optimizer step</text>
    </svg><small>Range {min.toPrecision(3)}–{max.toPrecision(3)}</small></article>;
}

function ResultSummary({value,prefix=""}:{value:unknown;prefix?:string}):React.ReactNode {
  if(value && typeof value==="object"){
    const entries=Object.entries(value);
    if(!entries.length)return <p className="muted">No returned values yet.</p>;
    return <div className="result-group">{entries.map(([key,item])=><ResultSummary key={key} value={item} prefix={`${prefix}${prefix ? " · " : ""}${key.replaceAll("_"," ")}`}/>)}</div>;
  }
  return <dl className="result-summary"><dt>{prefix}</dt><dd>{value==null ? "Not set" : String(value)}</dd></dl>;
}
