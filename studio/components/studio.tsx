"use client";
import { useEffect, useMemo, useState } from "react";
import type { CSSProperties } from "react";
import { QueryClient, QueryClientProvider, useQuery } from "@tanstack/react-query";
import { useStore } from "zustand";
import { Dialog } from "@base-ui/react/dialog";
import { Box, Check, Code2, FolderOpen, Layers, Library, Play, Redo2, Save, Settings2, Undo2, X } from "lucide-react";
import { Canvas } from "./canvas";
import { ComponentFields, Fields, ValueField } from "./fields";
import { ComponentBrowser } from "./component-browser";
import { Structure } from "./structure";
import { Execution } from "./execution";
import { ExecutionSettings } from "./execution-settings";
import { RunReview } from "./run-review";
import { Button } from "./ui/button";
import { createDraftStore, layoutSignature, yamlText } from "../model/draft";
import { addStage, connect, disconnect, newSnapshot, placeNode, project, signature, transferLayer, validConnection } from "../model/graph";
import { schemaDefault } from "../model/schema";
import { ApiError, request } from "../model/client";
import type { Catalog, ConnectionInfo, Document, Entry, Layout, Operation, RecordValue, Snapshot } from "../model/types";

export function Studio() {
  const [client] = useState(()=>new QueryClient({defaultOptions:{queries:{retry:false,refetchOnWindowFocus:false}}}));
  return <QueryClientProvider client={client}><Workbench /></QueryClientProvider>;
}

function Workbench() {
  const [store] = useState(createDraftStore);
  const state = useStore(store);
  const boot = useQuery({queryKey:["bootstrap"],queryFn:async()=>{
    const response=await fetch("/api/runtime",{cache:"no-store"});const data=await response.json();
    if(!response.ok) throw new Error(data.error);return data as ConnectionInfo;
  }});
  const connection=boot.data;
  const runtime=useQuery({queryKey:["runtime"],enabled:!!connection,queryFn:()=>request<{interface_version:number;nexuml_version:string;python_executable:string;working_directory:string}>(connection!,"/runtime")});
  const catalog=useQuery({queryKey:["catalog"],enabled:!!connection,queryFn:()=>request<Catalog>(connection!,"/registry")});
  const [view,setView]=useState("Pipeline");
  const [panel,setPanel]=useState("Canvas");
  const [leftWidth,setLeftWidth]=useState(280);
  const [rightWidth,setRightWidth]=useState(320);
  const [leftVisible,setLeftVisible]=useState(true);
  const [rightVisible,setRightVisible]=useState(true);
  const [query,setQuery]=useState("");
  const [file,setFile]=useState("scenario.yaml");
  const [scenario,setScenario]=useState("");
  const [libraryPath,setLibraryPath]=useState("");
  const [showLibraries,setShowLibraries]=useState(false);
  const [showYaml,setShowYaml]=useState(false);
  const [yamlBuffer,setYamlBuffer]=useState<string|null>(null);
  const [message,setMessage]=useState("");
  const [errors,setErrors]=useState<ApiError["fields"]>([]);
  const [focusPath,setFocusPath]=useState("");
  const [focusRequest,setFocusRequest]=useState(0);
  const [pending,setPending]=useState(false);
  const [operationId,setOperationId]=useState("");
  const [build,setBuild]=useState<{id:string;source:string}|null>(null);
  const [review,setReview]=useState<{document:Document;source:string}|null>(null);
  const [checkpoint,setCheckpoint]=useState("");
  const [layoutRevision,setLayoutRevision]=useState<string|null>(null);
  const [destinationStage,setDestinationStage]=useState("");
  const [stageName,setStageName]=useState("");
  const [creation,setCreation]=useState<{position?:{x:number;y:number}}|null>(null);
  const [insertion,setInsertion]=useState<{entry:Entry;position:{x:number;y:number}}|null>(null);
  const [insertionSlot,setInsertionSlot]=useState("append");
  const [transferStage,setTransferStage]=useState("");
  const [transferSlot,setTransferSlot]=useState("append");
  const draft=state.draft;
  const selected=draft?.selected ?? "data";
  const graph=useMemo(()=>draft ? project(draft,catalog.data) : null,[draft,catalog.data]);
  const selectedNode=graph?.nodes.find(node=>node.id===selected);
  const currentSignature=draft ? signature(draft) : "";
  const dirty=!!draft && (currentSignature!==state.saved || layoutSignature(draft)!==state.savedLayout);
  const buildResult=useQuery({queryKey:["operation",build?.id],enabled:!!connection && !!build,
    queryFn:()=>request<Operation>(connection!,`/operations/${build!.id}`),refetchInterval:query=>query.state.data?.status==="running" ? 1000 : false});
  const buildCurrent=build?.source===currentSignature;
  const buildFailure=buildResult.data?.result?.error as {message?:string;fields?:ApiError["fields"]}|undefined;
  const schema=catalog.data?.schema;
  const modelSchema=(name:string)=>schema?.$defs?.[name];
  const fieldContext={catalog:catalog.data,focusPath};
  useEffect(()=>{
    if(!focusPath)return;
    const controls=[...document.querySelectorAll<HTMLElement>("[data-field]")];
    const matches=controls.filter(control=>control.dataset.field===focusPath || focusPath.startsWith(`${control.dataset.field}.`))
      .sort((a,b)=>(b.dataset.field?.length ?? 0)-(a.dataset.field?.length ?? 0) || Number(b.matches("input,textarea"))-Number(a.matches("input,textarea")));
    const control=matches[0];
    if(!control)return;
    for(let parent=control.parentElement;parent;parent=parent.parentElement)if(parent instanceof HTMLDetailsElement)parent.open=true;
    control.scrollIntoView({block:"nearest"});control.focus();
  },[focusPath,view,selected,focusRequest]);
  const navigate=(location:(string|number)[])=>{
    if(location[0]==="pipeline" && location[1]==="stages" && draft){setView("Pipeline");select(draft.ids[String(location[2])]?.[Number(location[3])] ?? `stage:${location[2]}`);}
    else if(location[0]==="data"){setView("Pipeline");select("data");}
    else if(location[0]==="evaluation" && location[1]==="algorithms" && Number.isInteger(location[2])){setView("Pipeline");select(`evaluation:${location[2]}`);}
    else setView("Training");
    setRightVisible(true);setFocusPath(location.join("."));setFocusRequest(value=>value+1);
  };
  const report=(error:unknown)=>{setMessage(error instanceof Error ? error.message : String(error));setErrors(error instanceof ApiError ? error.fields : []);};
  const act=async(action:()=>Promise<void>)=>{if(pending)return;setPending(true);setMessage("");setErrors([]);try{await action();}catch(error){report(error);}finally{setPending(false);}};
  const replace=(document:Document,layout?:Layout)=>{state.load(document,layout);setFile(document.path ?? "scenario.yaml");setYamlBuffer(null);setLayoutRevision(null);setFocusPath("");setDestinationStage("");setInsertionSlot("append");};
  const change=(next:Snapshot)=>{state.change(next);setErrors([]);};
  const edit=async(path:(string|number)[],value:unknown,validate=false)=>{
    const original=store.getState().draft;
    if(!original)return;const next=structuredClone(original);let target:unknown=next.config;
    for(const key of path.slice(0,-1))target=(target as RecordValue)[key];
    (target as RecordValue)[path.at(-1)!]=value;
    if(validate){
      try{
        await request<Document>(connection!,"/config/validate",{data:next.config,stage_order:next.order});
        if(signature(store.getState().draft!)!==signature(original))throw new Error("Draft changed during validation; apply again");
        next.positions=store.getState().draft!.positions;
        next.names=store.getState().draft!.names;
        next.sizes=store.getState().draft!.sizes;
        next.selected=store.getState().draft!.selected;
      }catch(error){report(error);throw error;}
    }
    change(next);
  };
  const discard=()=>!(dirty || state.blocked) || window.confirm("Replace this unsaved draft and unapplied YAML buffer?");
  const payload=()=>({data:draft!.config,stage_order:draft!.order});
  const defaults=(entry:Entry)=>({type:entry.name,version:entry.version,params:schemaDefault(entry.schema) as RecordValue});
  const slotIndex=(slot:string,count:number)=>slot==="append" ? count : slot.startsWith("after:") ? Number(slot.slice(6))+1 : Number(slot);
  const addComponent=(entry:Entry,droppedStage?:string,position?:{x:number;y:number},slot?:number)=>{
    if(!draft)return;const next=structuredClone(draft);
    if(entry.kind==="layer"){
      if(position && !droppedStage){setDestinationStage("");setInsertionSlot("append");setInsertion({entry,position});return;}
      const stage=droppedStage ?? (next.order.includes(destinationStage) ? destinationStage : next.order[0]);
      if(!stage){setMessage("Add a named ordered stage first.");return;}
      const index=slot ?? (droppedStage ? next.ids[stage].length : Math.min(slotIndex(insertionSlot,next.ids[stage].length),next.ids[stage].length));
      const id=crypto.randomUUID();next.config.pipeline.stages[stage].splice(index,0,{component:defaults(entry),keys_in:[],keys_out:[]});next.ids[stage].splice(index,0,id);next.selected=id;
      // A new ordered insertion does not rearrange the other freely placed layers.
      for(const node of graph?.nodes ?? [])next.positions[node.id]=node.position;
      if(position){
        const parent=graph?.nodes.find(node=>node.id===`stage:${stage}`)?.position ?? {x:0,y:0};
        const point={x:position.x-parent.x,y:position.y-parent.y};
        next.positions[id]={x:Math.max(20,point.x),y:Math.max(130,point.y)};
      }
      else next.positions[id]={x:20,y:Math.max(130,graph?.nodes.find(node=>node.id===`stage:${stage}`)?.data.minHeight ?? 130)};
    }else if(entry.kind==="data_source"){next.config.data.source=defaults(entry);next.selected="data";if(position)next.positions.data=position;}
    else if(entry.kind==="eval_algorithm"){
      next.config.evaluation.algorithms.push({algorithm:defaults(entry)});
      next.selected=`evaluation:${next.config.evaluation.algorithms.length-1}`;
      if(position)next.positions[`evaluation:${next.config.evaluation.algorithms.length-1}`]=position;
    }else if(entry.kind==="loader_backend") {
      (next.config.data.loader as RecordValue).backend=defaults(entry);
    }else{setMessage(`Edit ${entry.kind} in its structured config fields.`);return;}
    change(next);
  };
  const removeNode=(id:string)=>{
    if(!draft)return;const node=graph?.nodes.find(node=>node.id===id);if(!node)return;const next=structuredClone(draft);
    if(node.data.kind==="layer"){next.config.pipeline.stages[node.data.stage!].splice(node.data.index!,1);next.ids[node.data.stage!].splice(node.data.index!,1);}
    else if(node.data.kind==="evaluation"){
      next.config.evaluation.algorithms.splice(node.data.index!,1);
      for(let index=node.data.index!;index<draft.config.evaluation.algorithms.length;index++){
        const current=`evaluation:${index}`,following=`evaluation:${index+1}`;
        if(draft.positions[following])next.positions[current]=draft.positions[following];else delete next.positions[current];
        if(draft.names?.[following])next.names![current]=draft.names[following];else delete next.names?.[current];
      }
    }
    else if(node.data.kind==="stage"){
      if(!window.confirm("Remove this stage and its layers?"))return;
      delete next.config.pipeline.stages[node.data.stage!];delete next.ids[node.data.stage!];next.order=next.order.filter(stage=>stage!==node.data.stage);
    }else return;
    if(node.data.kind!=="evaluation"){delete next.positions[id];delete next.names?.[id];delete next.sizes?.[id];}next.selected="data";change(next);
  };
  const select=(id:string)=>{state.select(id);setTransferStage("");setTransferSlot("append");if(window.innerWidth<1280)setPanel("Properties");};
  const createStage=(position?:{x:number;y:number})=>{setStageName("");setCreation({position});};
  const selectedStage=selectedNode?.data.stage;
  const targetStage=draft?.order.includes(transferStage) ? transferStage : selectedStage ?? "";
  const stageSkipped=(stage:string)=>(draft?.config.data.skip_pipeline_stages as string[]|undefined)?.includes(stage);
  const showComponents=(stage:string)=>{setDestinationStage(stage);setInsertionSlot("append");setLeftVisible(true);setPanel("Components");requestAnimationFrame(()=>document.querySelector<HTMLInputElement>('[aria-label="Search components"]')?.focus());};
  const save=()=>act(async()=>{
    const document=await request<Document>(connection!,"/config/save",{...payload(),path:file,
      base_revision:file===state.path ? state.baseRevision : null});
    state.markSaved(document,draft!);
    const layout=await request<{base_revision:string}>(connection!,"/config/layout/save",{path:file,
      base_revision:file===state.path ? layoutRevision : null,semantic_revision:document.semantic_revision,
      layout:{ids:draft!.ids,positions:draft!.positions,names:draft!.names,sizes:draft!.sizes}});
    state.markLayoutSaved(draft!);setLayoutRevision(layout.base_revision);setMessage("Configuration and separate layout saved.");
  });

  if(!connection || runtime.isError || runtime.data?.interface_version!==2) return <main className="startup"><div className="brand"><Layers/>NexuML <span>STUDIO</span></div><h1>{boot.isError || runtime.isError || (runtime.data && runtime.data.interface_version!==2) ? "Connection unavailable" : "Connecting to your runtime"}</h1><p>{boot.error?.message ?? runtime.error?.message ?? (runtime.data && runtime.data.interface_version!==2 ? "Studio requires interface 2. Use matching Python and Studio versions." : "Verifying the selected local NexuML installation…")}</p><p>Start with <code>nexuml-studio --python /path/to/python .</code> or choose an existing <code>nexuml</code> executable. Nothing is installed automatically.</p><Button onClick={()=>{void boot.refetch();void runtime.refetch();}}>Retry connection</Button></main>;
  return <main className="workbench">
    <header className="topbar"><div className="brand"><Layers size={22}/>NexuML <span>STUDIO</span></div>
      <div className="breadcrumb"><FolderOpen size={14}/><span title={runtime.data.working_directory}>{runtime.data.working_directory.split(/[\\/]/).at(-1)}</span><span>/</span><strong>{draft?.config.name ?? "Select a scenario"}</strong></div>
      <span className="save-state">{state.blocked?"Unapplied YAML":dirty?"Unsaved draft":"Saved"}</span>
      <div className="top-actions"><Button disabled={!draft || pending || state.blocked} onClick={save}><Save size={15}/>Save</Button>
        <Button disabled={!draft || pending || state.blocked} onClick={()=>act(async()=>{const checked=await request<Document>(connection,"/config/validate",payload());setMessage(`Fields valid · ${checked.semantic_revision.slice(0,12)}. No code compiled.`);})}><Check size={15}/>Check fields</Button>
         <Button data-run-trigger className="primary" disabled={!draft || pending || state.blocked || graph?.unconnected} title={graph?.unconnected ? "Connect required inputs before launching" : undefined} onClick={()=>act(async()=>{const checked=await request<Document>(connection,checkpoint ? "/train/prepare" : "/config/validate",{...payload(),...(checkpoint ? {trainer_checkpoint:checkpoint} : {})});setReview({document:checked,source:currentSignature});})}><Play size={15}/>Run…</Button></div>
    </header>
    <nav className="nav-tabs" aria-label="Workbench views"><div>{["Pipeline","Training","Execution","Artifacts"].map(tab=><Button key={tab} aria-current={view===tab?"page":undefined} onClick={()=>setView(tab)}>{tab}</Button>)}</div><div>
      <Button aria-pressed={showLibraries} onClick={()=>setShowLibraries(!showLibraries)}><Library size={15}/>Libraries</Button>
      <Button disabled={!draft} aria-pressed={showYaml} onClick={()=>setShowYaml(!showYaml)}><Code2 size={15}/>YAML</Button>
      <details className="panel-settings"><summary>Panels</summary><div>
        <Button aria-pressed={leftVisible} onClick={()=>setLeftVisible(!leftVisible)}>Component panel</Button>
        <label className="field">Component panel width<input type="range" min="220" max="380" step="20" value={leftWidth} onChange={event=>setLeftWidth(Number(event.target.value))}/></label>
        <Button aria-pressed={rightVisible} onClick={()=>setRightVisible(!rightVisible)}>Property panel</Button>
        <label className="field">Property panel width<input type="range" min="260" max="440" step="20" value={rightWidth} onChange={event=>setRightWidth(Number(event.target.value))}/></label>
      </div></details></div></nav>
    {showLibraries && <section className="library-panel"><h2>Existing library sources</h2><p className="muted">Adding a root imports trusted local Python on fresh discovery. No dependencies are installed.</p>
      {catalog.data?.libraries.packages.map(name=><span className="tag" key={name}>{name}</span>)}
      {catalog.data?.libraries.roots.map(root=><div className="library-row" key={root}><code>{root}</code><Button disabled={pending} onClick={()=>act(async()=>{await request(connection,"/libraries",{path:root},"DELETE");await catalog.refetch();})}>Remove root</Button></div>)}
      <div className="toolbar"><input aria-label="Local library root" value={libraryPath} placeholder="/path/to/trusted/library" onChange={event=>setLibraryPath(event.target.value)}/><Button disabled={pending || !libraryPath} onClick={()=>act(async()=>{await request(connection,"/libraries",{path:libraryPath});await catalog.refetch();setLibraryPath("");})}>Add root</Button><Button onClick={()=>void catalog.refetch()}>Refresh discovery</Button></div>
    </section>}
    <section className="source-bar"><label>Scenario<select aria-label="Discovered scenario" value={scenario} onChange={event=>setScenario(event.target.value)}><option value="">Choose installed recipe…</option>{catalog.data?.scenarios.map(item=><option key={item.name}>{item.name}</option>)}</select></label><Button disabled={pending || !scenario} onClick={()=>{if(discard())void act(async()=>replace(await request<Document>(connection,`/scenarios/${encodeURIComponent(scenario)}/resolve`,{})));}}>Resolve recipe</Button>
      <label>Config path<input aria-label="Config path" value={file} onChange={event=>setFile(event.target.value)}/></label><Button disabled={pending || !file} onClick={()=>{if(discard())void act(async()=>{
        const document=await request<Document>(connection,"/config/load",{path:file});
        const sidecar=await request<{semantic_revision?:string;layout?:Layout;base_revision:string|null}>(connection,"/config/layout/load",{path:file});
        replace(document,sidecar.semantic_revision===document.semantic_revision ? sidecar.layout : undefined);setLayoutRevision(sidecar.base_revision);
        if(sidecar.layout && sidecar.semantic_revision!==document.semantic_revision)setMessage("External semantic revision changed. Stale layout ignored.");
      });}}>Open YAML</Button></section>
    {showYaml && draft && <section className="yaml-panel"><div className="section-title"><h2>Ordinary NexuML YAML</h2><Button aria-label="Close YAML" onClick={()=>setShowYaml(false)}><X size={16}/></Button></div><p className="muted">Changes apply only after field validation. Invalid/unapplied text blocks graph edits; source comments and formatting are not preserved.</p>
      <textarea aria-label="Scenario YAML" disabled={pending} spellCheck={false} value={yamlBuffer ?? yamlText(draft)} onChange={event=>{setYamlBuffer(event.target.value);state.block(true);}} />
      <div className="toolbar"><Button className="primary" disabled={pending || yamlBuffer===null} onClick={()=>act(async()=>{
        const document=await request<Document>(connection,"/config/validate",{yaml:yamlBuffer});state.block(false);
        state.change(newSnapshot(document.data,document.stage_order,draft));setYamlBuffer(null);setMessage("YAML applied to the single configuration draft.");
      })}>Apply YAML</Button><Button disabled={yamlBuffer===null} onClick={()=>{if(window.confirm("Discard the unapplied YAML buffer?")){setYamlBuffer(null);state.block(false);}}}>Discard buffer</Button></div>
    </section>}
    {view==="Pipeline" && draft && graph ? <>
      <div className="panel-tabs"><Button aria-pressed={panel==="Components"} onClick={()=>setPanel("Components")}>Components</Button><Button aria-pressed={panel==="Canvas"} onClick={()=>setPanel("Canvas")}>Canvas</Button><Button aria-pressed={panel==="Properties"} onClick={()=>setPanel("Properties")}>Properties</Button></div>
      <div className={`editor-grid panel-${panel.toLowerCase()} ${leftVisible?"":"hide-components"} ${rightVisible?"":"hide-properties"}`}
        style={{"--left-panel":`${leftVisible?leftWidth:0}px`,"--right-panel":`${rightVisible?rightWidth:0}px`} as CSSProperties}>
        <aside className="components-panel"><div className="section-title"><h2>Components</h2><Box size={16}/></div><input aria-label="Search components" placeholder="Search installed components…" value={query} onChange={event=>setQuery(event.target.value)}/>
          <label className="field">Insert into stage<select aria-label="Insert into stage" value={draft.order.includes(destinationStage) ? destinationStage : draft.order[0] || ""} onChange={event=>{setDestinationStage(event.target.value);setInsertionSlot("append");}}>{draft.order.map(stage=><option key={stage}>{stage}</option>)}</select></label>
          <label className="field">Layer insertion slot<select aria-label="Layer insertion slot" value={insertionSlot} onChange={event=>setInsertionSlot(event.target.value)}><option value="append">Append (last)</option>{(draft.ids[destinationStage] ?? draft.ids[draft.order[0]] ?? []).map((id,index)=><optgroup key={id} label={`Layer ${index+1}`}><option value={index}>Before layer {index+1}</option><option value={`after:${index}`}>After layer {index+1}</option></optgroup>)}</select></label>
          {stageSkipped(destinationStage || draft.order[0]) && <p className="field-error">Skipped destination: inserted layers will not execute.</p>}
          <ComponentBrowser entries={catalog.data?.components ?? []} query={query} blocked={state.blocked || pending} add={addComponent}/>
          <Structure snapshot={draft} selected={selected} select={select} change={change} blocked={state.blocked || pending} report={report} create={()=>createStage()}/>
        </aside>
        <Canvas snapshot={draft} selected={selected} select={select} change={change} blocked={state.blocked || pending} catalog={catalog.data} insert={addComponent} report={report} remove={removeNode} create={createStage} addLayer={showComponents}/>
        <aside className="properties-panel"><div className="section-title"><h2>Properties</h2><Settings2 size={16}/></div><fieldset disabled={state.blocked || pending}><legend>{selectedNode?.data.title ?? "Select a node"}</legend>
          {selectedNode && <label className="field">Display name<input aria-label="Node display name" placeholder={selectedNode.data.title} value={draft.names?.[selected] ?? ""} onChange={event=>change({...draft,names:{...draft.names,[selected]:event.target.value}})}/><small className="muted">Visual only; output key names below control routing.</small></label>}
          {selectedNode?.data.kind==="data" && <Fields value={draft.config.data} schema={modelSchema("DataSpec")} root={schema} prefix="data." errors={errors} basic={["source","input_shapes","feature_key","datasets","targets","loader"]} {...fieldContext} onChange={(value,validate)=>edit(["data"],value,validate)}/>}
          {selectedNode?.data.kind==="layer" && (()=>{
            const {stage,index}=selectedNode.data;const layer=draft.config.pipeline.stages[stage!][index!];
            return <><ComponentFields kind="layer" component={layer.component} errors={errors} {...fieldContext} prefix={`pipeline.stages.${stage}.${index}.component.params.`} onChange={(value,validate)=>edit(["pipeline","stages",stage!,index!,"component"],value,validate)}/>
              <Fields value={Object.fromEntries(Object.entries(layer).filter(([key])=>key!=="component"))} schema={{properties:Object.fromEntries(Object.entries(modelSchema("LayerSpec")?.properties ?? {}).filter(([key])=>key!=="component"))}} root={schema} prefix={`pipeline.stages.${stage}.${index}.`} errors={errors} {...fieldContext} basic={["keys_in","keys_out","label_key","label_in_x","meta_in","meta_out"]} onChange={(value,validate)=>edit(["pipeline","stages",stage!,index!],{...layer,...value},validate)}/>
              <details><summary>Move / transfer layer</summary><label className="field">Destination stage<select aria-label="Transfer destination stage" value={targetStage} onChange={event=>{setTransferStage(event.target.value);setTransferSlot("append");}}>{draft.order.map(name=><option key={name}>{name}</option>)}</select></label>
                <label className="field">Execution insertion slot<select aria-label="Transfer insertion slot" value={transferSlot} onChange={event=>setTransferSlot(event.target.value)}><option value="append">Append (last)</option>{draft.ids[targetStage]?.map((id,slot)=><optgroup key={id} label={`Layer ${slot+1}`}><option value={slot}>Before layer {slot+1}</option><option value={`after:${slot}`}>After layer {slot+1}</option></optgroup>)}</select></label>
                <p>Move to {targetStage}, {transferSlot==="append" ? "append" : transferSlot.startsWith("after:") ? `after layer ${Number(transferSlot.slice(6))+1}` : `before layer ${Number(transferSlot)+1}`}. {stageSkipped(targetStage) && <strong>Skipped: this layer will not execute.</strong>}</p>
                <Button onClick={()=>change(transferLayer(draft,selected,targetStage,slotIndex(transferSlot,draft.ids[targetStage].length)))}>Move layer</Button></details>
              <Button className="danger" onClick={()=>removeNode(selected)}>Remove layer</Button></>;
          })()}
          {selectedNode?.data.kind==="objective" && <Fields value={{loss_keys:draft.config.training.loss_keys,metric_keys:draft.config.training.metric_keys}} schema={{properties:Object.fromEntries(Object.entries(modelSchema("TrainingSpec")?.properties ?? {}).filter(([key])=>["loss_keys","metric_keys"].includes(key)))}} root={schema} prefix="training." errors={errors} {...fieldContext} onChange={(value,validate)=>edit(["training"],{...draft.config.training,...value},validate)} />}
          {selectedNode?.data.kind==="evaluation" && (()=>{
            const index=selectedNode.data.index!;const item=draft.config.evaluation.algorithms[index];
            return <><ComponentFields kind="eval_algorithm" component={item.algorithm} errors={errors} {...fieldContext} prefix={`evaluation.algorithms.${index}.algorithm.params.`} onChange={(value,validate)=>edit(["evaluation","algorithms",index,"algorithm"],value,validate)}/><Fields value={Object.fromEntries(Object.entries(item).filter(([key])=>key!=="algorithm"))} schema={{properties:Object.fromEntries(Object.entries(modelSchema("EvalAlgorithmSpec")?.properties ?? {}).filter(([key])=>key!=="algorithm"))}} prefix={`evaluation.algorithms.${index}.`} root={schema} errors={errors} {...fieldContext} onChange={(value,validate)=>edit(["evaluation","algorithms",index],{...item,...value},validate)}/><Button className="danger" onClick={()=>removeNode(selected)}>Remove evaluation</Button></>;
          })()}
          {selectedNode?.data.kind==="stage" && <><p className="muted">Drag the header to move with layers. Structure / Execution order controls change execution, never canvas placement.</p>{stageSkipped(selectedStage!) && <p className="field-error">Skipped stage: its layers will not execute.</p>}<Button onClick={()=>showComponents(selectedStage!)}>Add layer</Button>
            <details><summary>Container size</summary>{(["width","height"] as const).map(axis=><label className="field" key={axis}>{axis}<input type="number" aria-label={`Stage ${axis}`} min={axis==="width" ? selectedNode.data.minWidth : selectedNode.data.minHeight} value={Number(selectedNode.style?.[axis])} onChange={event=>change({...draft,sizes:{...draft.sizes,[selected]:{width:Number(selectedNode.style?.width),height:Number(selectedNode.style?.height),[axis]:Math.max(axis==="width" ? selectedNode.data.minWidth! : selectedNode.data.minHeight!,Number(event.target.value))}}})}/></label>)}</details><Button className="danger" onClick={()=>removeNode(selected)}>Remove stage</Button></>}
          {selectedNode && <><details><summary>Key connections (drag alternative)</summary>{selectedNode.data.inputs.map(input=><div className="field" key={input.id}>{input.add ? `Add ${input.domain} input` : `${input.alias ?? input.field}: ${input.key || "Unconnected"}`}<select aria-label={`Connect ${input.key || (input.add ? `new ${input.domain} input` : input.field)}`} value="" onChange={event=>{
            const [source,handle]=JSON.parse(event.target.value);change(connect(draft,{source,sourceHandle:handle,target:selected,targetHandle:input.id},catalog.data));
          }}><option value="">Choose preceding producer…</option>{graph.nodes.flatMap(node=>node.data.outputs.filter(output=>validConnection(draft,{source:node.id,sourceHandle:output.id,target:selected,targetHandle:input.id},catalog.data)).map(output=><option key={`${node.id}:${output.id}`} value={JSON.stringify([node.id,output.id])}>{node.data.title} → {output.key} ({output.domain})</option>))}</select>
            {graph.edges.some(edge=>edge.target===selected && edge.targetHandle===input.id) && <Button disabled={!!input.defaultKey} title={input.defaultKey ? "Inherited runtime default: change the key or disable the evaluator" : "Remove this key route"} aria-label={`Remove connection ${input.key}`} onClick={()=>change(disconnect(draft,[{target:selected,targetHandle:input.id}],catalog.data))}>Remove connection</Button>}
            {input.defaultKey && <small className="muted">Runtime default: {input.defaultKey}; clearing the field restores it.</small>}
          </div>)}</details>
            <details><summary>Visual position</summary>{["x","y"].map(axis=><label className="field" key={axis}>{axis}<input type="number" aria-label={`Node position ${axis}`} value={selectedNode.position[axis as "x"|"y"]} onChange={event=>change(placeNode(draft,selected,{...selectedNode.position,[axis]:Number(event.target.value)}))}/></label>)}</details></>}
        </fieldset></aside>
      </div>
    </> : view==="Training" && draft ? <section className="training-view"><div className="view-title"><div><span className="eyebrow">EXISTING SCENARIO SETTINGS</span><h1>Training configuration</h1></div><p className="muted">Changes edit your draft, not a running operation.</p></div><fieldset disabled={state.blocked || pending}>
      <div className="training-grid"><article><h2>Training</h2><ValueField label="Scenario name" path="name" schema={schema?.properties?.name} value={draft.config.name} onChange={value=>edit(["name"],value)}/><Fields value={draft.config.training} schema={modelSchema("TrainingSpec")} root={schema} basic={["max_epochs","batch_size","lr","accelerator","devices","precision","optimizer","scheduler"]} prefix="training." errors={errors} {...fieldContext} onChange={(value,validate)=>edit(["training"],value,validate)}/></article>
         <article><h2>Data & loader</h2><Fields value={draft.config.data.loader as RecordValue} schema={modelSchema("LoaderSpec")} root={schema} prefix="data.loader." errors={errors} {...fieldContext} onChange={(value,validate)=>edit(["data","loader"],value,validate)}/><h2>Execution</h2><ExecutionSettings value={draft.config.execution} catalog={catalog.data} errors={errors} onChange={value=>void edit(["execution"],value)}/><details><summary>Evaluation settings</summary><ValueField label="Evaluation" path="evaluation" schema={schema?.properties?.evaluation} root={schema} value={draft.config.evaluation} errors={errors} {...fieldContext} onChange={(value,validate)=>edit(["evaluation"],value,validate)}/></details></article>
        <article><h2>Checkpoint & logging</h2>{["checkpoint","logging","callbacks","exports"].map(key=><ValueField key={key} label={key.replaceAll("_"," ")} path={key} schema={schema?.properties?.[key]} root={schema} value={draft.config[key]} errors={errors} {...fieldContext} onChange={(value,validate)=>edit([key],value,validate)}/>)}<label className="field">Resume Trainer checkpoint<input aria-label="Resume Trainer checkpoint" placeholder="Optional authorized local .ckpt path" value={checkpoint} onChange={event=>setCheckpoint(event.target.value)}/></label><p className="muted">Resume follows the checkpoint scenario semantics. Local only. Review the checkpoint-derived source at launch. Checkpoints are trusted Python inputs.</p></article></div>
      </fieldset></section> : view==="Execution" || view==="Artifacts" ? <Execution connection={connection} identity={operationId} choose={setOperationId} artifacts={view==="Artifacts"} report={report} editSettings={()=>setView("Training")} loadSettings={identity=>act(async()=>{if(!discard())return;replace(await request<Document>(connection,`/operations/${identity}/config`));setView("Training");})}/> : <section className="welcome"><span className="eyebrow">YOUR RUNTIME. YOUR SCENARIO.</span><h1>Build on what you already have.</h1><p>Choose an installed scenario recipe or open existing NexuML YAML.<br/>Studio edits your configuration. NexuML runs it.</p><p className="muted">Discovery is loading trusted installed definitions. Explicit resolution/build can execute Python.</p>{catalog.isError && <p className="field-error">{catalog.error.message}</p>}</section>}
    <section className="diagnostics" aria-label="Problems and checks"><div className="diagnostic-heading"><strong>Problems & checks</strong><div><Button disabled={!state.past.length || state.blocked || pending} onClick={state.undo} aria-label="Undo"><Undo2 size={15}/></Button><Button disabled={!state.future.length || state.blocked || pending} onClick={state.redo} aria-label="Redo"><Redo2 size={15}/></Button><Button disabled={!draft || pending || state.blocked} onClick={()=>act(async()=>{
      const operation=await request<Operation>(connection,"/build",payload());setBuild({id:operation.id,source:currentSignature});setOperationId(operation.id);setMessage("Executable build started; constructors and dummy forwards run in the selected interpreter.");
    })}>Build check (executes code)</Button></div></div>
      <div role="status">{pending ? "Waiting for selected NexuML…" : message || "Field checks do not compile. Build is explicit."}</div>
      {build && <div className="build-state">Build: {buildResult.data?.status ?? "connecting"}{!buildCurrent && " · stale (draft changed)"}{buildResult.data?.status==="succeeded" && buildCurrent && <code> · Final key shapes: {Object.entries(buildResult.data.result?.shapes as RecordValue ?? {}).map(([key,value])=>`${key} [${Array.isArray(value) ? value.join(" × ") : value}]`).join(" · ")}</code>}{buildResult.data?.status==="failed" && <p className="field-error" role="alert">{buildFailure?.message ?? "Build failed. Inspect its process logs and frozen launch configuration."}</p>}
        {buildCurrent && buildFailure?.fields?.map((field,index)=><Button key={index} onClick={()=>navigate(field.loc)}>{field.loc.join(".")}: {field.message}</Button>)}
        {buildResult.data?.result && <details><summary>Expert: build result JSON</summary><pre>{JSON.stringify(buildResult.data.result,null,2)}</pre></details>}
      </div>}
      <div className="problems-list">{errors.map((error,index)=><Button key={index} onClick={()=>navigate(error.loc)}>{error.loc.join(".")}: {error.message}</Button>)}{graph?.problems.map(problem=><p key={problem}>{problem}</p>)}{catalog.data?.errors.map((error,index)=><p key={index}>{error.module}: {error.message}</p>)}</div>
    </section>
     <footer className="statusbar"><span className="connection-dot"/>Local runtime · NexuML {runtime.data.nexuml_version}<code title={runtime.data.python_executable}>{runtime.data.python_executable}</code><span>Interface 2 · {draft ? `${draft.order.length} ordered stages` : "No scenario selected"}</span></footer>
    <Dialog.Root open={!!creation} onOpenChange={open=>{if(!open)setCreation(null);}}><Dialog.Portal><Dialog.Backdrop className="dialog-backdrop"/><Dialog.Popup className="dialog-popup"><Dialog.Title>Add ordered stage</Dialog.Title><Dialog.Description>Insert {selectedStage ? `after ${selectedStage}` : "last"}. Canvas position never determines execution order.</Dialog.Description>
      <label className="field">Stage name<input aria-label="New stage name" value={stageName} onChange={event=>setStageName(event.target.value)}/></label>{draft?.order.includes(stageName.trim()) && <p className="field-error">Stage name already exists.</p>}
      <div className="toolbar"><Button onClick={()=>setCreation(null)}>Cancel</Button><Button className="primary" disabled={!draft || !stageName.trim() || draft.order.includes(stageName.trim()) || state.blocked || pending} onClick={()=>{change(addStage(draft!,stageName,selectedStage,creation?.position));setCreation(null);}}>Create stage</Button></div>
    </Dialog.Popup></Dialog.Portal></Dialog.Root>
    <Dialog.Root open={!!insertion} onOpenChange={open=>{if(!open)setInsertion(null);}}><Dialog.Portal><Dialog.Backdrop className="dialog-backdrop"/><Dialog.Popup className="dialog-popup"><Dialog.Title>Choose layer destination</Dialog.Title><Dialog.Description>This drop is outside every stage. Choose its ordered membership explicitly.</Dialog.Description>
      <label className="field">Stage<select aria-label="Dropped layer destination" value={destinationStage} onChange={event=>setDestinationStage(event.target.value)}><option value="">Choose a stage…</option>{draft?.order.map(stage=><option key={stage}>{stage}</option>)}</select></label><p>Append to {destinationStage || "chosen stage"}. {stageSkipped(destinationStage) && "Skipped: layer will not execute."}</p>
      <div className="toolbar"><Button onClick={()=>setInsertion(null)}>Cancel</Button><Button disabled={!destinationStage || state.blocked || pending} onClick={()=>{addComponent(insertion!.entry,destinationStage,undefined,draft!.ids[destinationStage].length);setInsertion(null);}}>Insert layer</Button></div>
    </Dialog.Popup></Dialog.Portal></Dialog.Root>
     {review && <RunReview initial={review.document} connection={connection} catalog={catalog.data}
       sourceChanged={review.source!==currentSignature} checkpoint={checkpoint} close={()=>setReview(null)}
       accepted={(operation,document)=>{const current=store.getState().draft;if(current){const next=structuredClone(current);next.config.execution=document.data.execution;change(next);}setOperationId(operation.id);setView("Execution");setReview(null);}}/>}
  </main>;
}
