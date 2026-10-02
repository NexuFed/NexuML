"use client";
import { useMemo, useState } from "react";
import type { CSSProperties } from "react";
import { QueryClient, QueryClientProvider, useQuery } from "@tanstack/react-query";
import { useStore } from "zustand";
import { Dialog } from "@base-ui/react/dialog";
import { ArrowDown, ArrowUp, Box, Check, Code2, FolderOpen, Layers, Library, Play, Redo2, Save, Settings2, Undo2, X } from "lucide-react";
import { Canvas } from "./canvas";
import { Fields, StructuredField } from "./fields";
import { Execution } from "./execution";
import { Button } from "./ui/button";
import { createDraftStore, layoutSignature, yamlText } from "../model/draft";
import { connect, moveLayer, newSnapshot, project, signature, validConnection } from "../model/graph";
import { ApiError, request } from "../model/client";
import type { Catalog, Component, ConnectionInfo, Document, Entry, Operation, RecordValue, Snapshot } from "../model/types";

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
  const [selected,setSelected]=useState("data");
  const [query,setQuery]=useState("");
  const [file,setFile]=useState("scenario.yaml");
  const [scenario,setScenario]=useState("");
  const [libraryPath,setLibraryPath]=useState("");
  const [showLibraries,setShowLibraries]=useState(false);
  const [showYaml,setShowYaml]=useState(false);
  const [yamlBuffer,setYamlBuffer]=useState<string|null>(null);
  const [message,setMessage]=useState("");
  const [errors,setErrors]=useState<ApiError["fields"]>([]);
  const [pending,setPending]=useState(false);
  const [operationId,setOperationId]=useState("");
  const [build,setBuild]=useState<{id:string;source:string}|null>(null);
  const [review,setReview]=useState<{document:Document;source:string}|null>(null);
  const [checkpoint,setCheckpoint]=useState("");
  const [layoutRevision,setLayoutRevision]=useState<string|null>(null);
  const [destinationStage,setDestinationStage]=useState("");
  const [stageName,setStageName]=useState("");
  const draft=state.draft;
  const graph=useMemo(()=>draft ? project(draft) : null,[draft]);
  const selectedNode=graph?.nodes.find(node=>node.id===selected);
  const currentSignature=draft ? signature(draft) : "";
  const dirty=!!draft && (currentSignature!==state.saved || layoutSignature(draft)!==state.savedLayout);
  const buildResult=useQuery({queryKey:["operation",build?.id],enabled:!!connection && !!build,
    queryFn:()=>request<Operation>(connection!,`/operations/${build!.id}`),refetchInterval:query=>query.state.data?.status==="running" ? 1000 : false});
  const buildCurrent=build?.source===currentSignature;
  const schema=catalog.data?.schema;
  const modelSchema=(name:string)=>schema?.$defs?.[name];
  const report=(error:unknown)=>{setMessage(error instanceof Error ? error.message : String(error));setErrors(error instanceof ApiError ? error.fields : []);};
  const act=async(action:()=>Promise<void>)=>{if(pending)return;setPending(true);setMessage("");setErrors([]);try{await action();}catch(error){report(error);}finally{setPending(false);}};
  const replace=(document:Document,layout?:Pick<Snapshot,"positions"|"ids">)=>{state.load(document,layout);setFile(document.path ?? "scenario.yaml");setSelected("data");setYamlBuffer(null);setLayoutRevision(null);};
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
      }catch(error){report(error);throw error;}
    }
    change(next);
  };
  const discard=()=>!(dirty || state.blocked) || window.confirm("Replace this unsaved draft and unapplied YAML buffer?");
  const payload=()=>({data:draft!.config,stage_order:draft!.order});
  const defaults=(entry:Entry):Component=>({type:entry.name,version:entry.version,
    params:Object.fromEntries(Object.entries(entry.schema.properties ?? {}).filter(([,field])=>field.default!==undefined).map(([key,field])=>[key,structuredClone(field.default)]))});
  const addComponent=(entry:Entry)=>{
    if(!draft)return;const next=structuredClone(draft);
    if(entry.kind==="layer"){
      const stage=next.order.includes(destinationStage) ? destinationStage : next.order[0];
      if(!stage){setMessage("Add a named ordered stage first.");return;}
      const id=crypto.randomUUID();next.config.pipeline.stages[stage].push({component:defaults(entry),keys_in:[],keys_out:[]});next.ids[stage].push(id);setSelected(id);
    }else if(entry.kind==="data_source")next.config.data.source=defaults(entry);
    else if(entry.kind==="eval_algorithm"){
      next.config.evaluation.algorithms.push({algorithm:defaults(entry)});
      setSelected(`evaluation:${next.config.evaluation.algorithms.length-1}`);
    }else if(entry.kind==="loader_backend") {
      (next.config.data.loader as RecordValue).backend=defaults(entry);
    }else{setMessage(`Edit ${entry.kind} in its structured config fields.`);return;}
    change(next);
  };
  const select=(id:string)=>{setSelected(id);if(window.innerWidth<1280)setPanel("Properties");};
  const save=()=>act(async()=>{
    const document=await request<Document>(connection!,"/config/save",{...payload(),path:file,
      base_revision:file===state.path ? state.baseRevision : null});
    state.markSaved(document,draft!);
    const layout=await request<{base_revision:string}>(connection!,"/config/layout/save",{path:file,
      base_revision:file===state.path ? layoutRevision : null,semantic_revision:document.semantic_revision,
      layout:{ids:draft!.ids,positions:draft!.positions}});
    state.markLayoutSaved(draft!);setLayoutRevision(layout.base_revision);setMessage("Configuration and separate layout saved.");
  });

  if(!connection || runtime.isError || runtime.data?.interface_version!==1) return <main className="startup"><div className="brand"><Layers/>NexuML <span>STUDIO</span></div><h1>{boot.isError || runtime.isError ? "Connection unavailable" : "Connecting to your runtime"}</h1><p>{boot.error?.message ?? runtime.error?.message ?? "Verifying the selected local NexuML installation…"}</p><p>Start with <code>nexuml-studio --python /path/to/python .</code> or choose an existing <code>nexuml</code> executable. Nothing is installed automatically.</p><Button onClick={()=>{void boot.refetch();void runtime.refetch();}}>Retry connection</Button></main>;
  return <main className="workbench">
    <header className="topbar"><div className="brand"><Layers size={22}/>NexuML <span>STUDIO</span></div>
      <div className="breadcrumb"><FolderOpen size={14}/><span title={runtime.data.working_directory}>{runtime.data.working_directory.split(/[\\/]/).at(-1)}</span><span>/</span><strong>{draft?.config.name ?? "Select a scenario"}</strong></div>
      <span className="save-state">{state.blocked?"Unapplied YAML":dirty?"Unsaved draft":"Saved"}</span>
      <div className="top-actions"><Button disabled={!draft || pending || state.blocked} onClick={save}><Save size={15}/>Save</Button>
        <Button disabled={!draft || pending || state.blocked} onClick={()=>act(async()=>{const checked=await request<Document>(connection,"/config/validate",payload());setMessage(`Fields valid · ${checked.semantic_revision.slice(0,12)}. No code compiled.`);})}><Check size={15}/>Check fields</Button>
        <Button className="primary" disabled={!draft || pending || state.blocked} onClick={()=>act(async()=>{const checked=await request<Document>(connection,checkpoint ? "/train/prepare" : "/config/validate",{...payload(),...(checkpoint ? {trainer_checkpoint:checkpoint} : {})});setReview({document:checked,source:currentSignature});})}><Play size={15}/>Run…</Button></div>
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
        const sidecar=await request<{semantic_revision?:string;layout?:Pick<Snapshot,"positions"|"ids">;base_revision:string|null}>(connection,"/config/layout/load",{path:file});
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
          <label className="field">Insert into stage<select aria-label="Insert into stage" value={destinationStage || draft.order[0] || ""} onChange={event=>setDestinationStage(event.target.value)}>{draft.order.map(stage=><option key={stage}>{stage}</option>)}</select></label>
          <div className="component-list">{catalog.data?.components.filter(entry=>["layer","data_source","eval_algorithm","loader_backend"].includes(entry.kind) && `${entry.kind} ${entry.name}`.toLowerCase().includes(query.toLowerCase())).map(entry=><Button key={`${entry.kind}:${entry.name}:${entry.version}`} disabled={state.blocked || pending} onClick={()=>addComponent(entry)}><Box size={15}/><span>{entry.name}<small>{entry.kind} · v{entry.version}</small></span><span>+</span></Button>)}</div>
          <h2>Execution order</h2><div className="outline">
            <Button aria-pressed={selected==="data"} onClick={()=>select("data")}>Data configuration</Button>
            {draft.order.map((stage,stageIndex)=><div key={stage} className="outline-stage"><div className="outline-row"><Button onClick={()=>select(`stage:${stage}`)}>{stageIndex+1}. {stage}</Button>
              <Button aria-label={`Move stage ${stage} up`} disabled={state.blocked || pending || stageIndex===0} onClick={()=>{const next=structuredClone(draft);[next.order[stageIndex-1],next.order[stageIndex]]=[stage,next.order[stageIndex-1]];change(next);}}><ArrowUp size={12}/></Button>
              <Button aria-label={`Move stage ${stage} down`} disabled={state.blocked || pending || stageIndex===draft.order.length-1} onClick={()=>{const next=structuredClone(draft);[next.order[stageIndex+1],next.order[stageIndex]]=[stage,next.order[stageIndex+1]];change(next);}}><ArrowDown size={12}/></Button></div>
              {draft.config.pipeline.stages[stage].map((layer,index)=><div className="outline-row" key={draft.ids[stage][index]}><Button aria-pressed={selected===draft.ids[stage][index]} onClick={()=>select(draft.ids[stage][index])}>{index+1}. {layer.component.type}</Button><Button aria-label={`Move ${layer.component.type} up`} disabled={state.blocked || pending || index===0} onClick={()=>change(moveLayer(draft,stage,index,-1))}><ArrowUp size={12}/></Button><Button aria-label={`Move ${layer.component.type} down`} disabled={state.blocked || pending || index===draft.ids[stage].length-1} onClick={()=>change(moveLayer(draft,stage,index,1))}><ArrowDown size={12}/></Button></div>)}
            </div>)}
            <Button onClick={()=>select("objectives")}>Objectives & metrics</Button>{draft.config.evaluation.algorithms.map((item,index)=><Button key={index} onClick={()=>select(`evaluation:${index}`)}>{item.algorithm.type}</Button>)}
          </div><div className="toolbar"><input aria-label="New stage name" placeholder="New stage name" value={stageName} onChange={event=>setStageName(event.target.value)}/><Button disabled={state.blocked || pending || !stageName || draft.order.includes(stageName)} onClick={()=>{const next=structuredClone(draft);next.order.push(stageName);next.config.pipeline.stages={...next.config.pipeline.stages,[stageName]:[]};next.ids={...next.ids,[stageName]:[]};change(next);setStageName("");}}>Add stage</Button></div>
        </aside>
        <Canvas snapshot={draft} selected={selected} select={select} change={change} blocked={state.blocked || pending}/>
        <aside className="properties-panel"><div className="section-title"><h2>Properties</h2><Settings2 size={16}/></div><fieldset disabled={state.blocked || pending}><legend>{selectedNode?.data.title ?? "Select a node"}</legend>
          {selectedNode?.data.kind==="data" && <><Fields value={draft.config.data} schema={modelSchema("DataSpec")} root={schema} prefix="data." errors={errors} basic={["source","input_shapes","feature_key","datasets","targets","loader"]} onChange={(value,validate)=>edit(["data"],value,validate)}/>
            {draft.config.data.source && <ComponentFields component={draft.config.data.source} catalog={catalog.data} errors={errors} prefix="data.source.params." onChange={(value,validate)=>edit(["data","source"],value,validate)}/>}</>}
          {selectedNode?.data.kind==="layer" && (()=>{
            const {stage,index}=selectedNode.data;const layer=draft.config.pipeline.stages[stage!][index!];
            return <><ComponentFields component={layer.component} catalog={catalog.data} errors={errors} prefix={`pipeline.stages.${stage}.${index}.component.params.`} onChange={(value,validate)=>edit(["pipeline","stages",stage!,index!,"component"],value,validate)}/>
              <Fields value={Object.fromEntries(Object.entries(layer).filter(([key])=>key!=="component"))} schema={{properties:Object.fromEntries(Object.entries(modelSchema("LayerSpec")?.properties ?? {}).filter(([key])=>key!=="component"))}} root={schema} prefix={`pipeline.stages.${stage}.${index}.`} errors={errors} basic={["keys_in","keys_out","label_key","label_in_x","meta_in","meta_out"]} onChange={(value,validate)=>edit(["pipeline","stages",stage!,index!],{...layer,...value},validate)}/>
              <Button className="danger" onClick={()=>{const next=structuredClone(draft);next.config.pipeline.stages[stage!].splice(index!,1);next.ids[stage!].splice(index!,1);change(next);setSelected("data");}}>Remove layer</Button></>;
          })()}
          {selectedNode?.data.kind==="objective" && <Fields value={{loss_keys:draft.config.training.loss_keys,metric_keys:draft.config.training.metric_keys}} onChange={(value,validate)=>edit(["training"],{...draft.config.training,...value},validate)} />}
          {selectedNode?.data.kind==="evaluation" && (()=>{
            const index=selectedNode.data.index!;const item=draft.config.evaluation.algorithms[index];
            return <><ComponentFields component={item.algorithm} catalog={catalog.data} errors={errors} prefix={`evaluation.algorithms.${index}.algorithm.params.`} onChange={(value,validate)=>edit(["evaluation","algorithms",index,"algorithm"],value,validate)}/><Fields value={Object.fromEntries(Object.entries(item).filter(([key])=>key!=="algorithm"))} schema={{properties:Object.fromEntries(Object.entries(modelSchema("EvalAlgorithmSpec")?.properties ?? {}).filter(([key])=>key!=="algorithm"))}} prefix={`evaluation.algorithms.${index}.`} root={schema} errors={errors} onChange={(value,validate)=>edit(["evaluation","algorithms",index],{...item,...value},validate)}/><Button className="danger" onClick={()=>{const next=structuredClone(draft);next.config.evaluation.algorithms.splice(index,1);change(next);setSelected("data");}}>Remove evaluation</Button></>;
          })()}
          {selectedNode?.data.kind==="stage" && <><p className="muted">Order is changed in the outline, never by placement.</p><Button className="danger" onClick={()=>{if(window.confirm("Remove this stage and its layers?")){const next=structuredClone(draft);delete next.config.pipeline.stages[selectedNode.data.stage!];delete next.ids[selectedNode.data.stage!];next.order=next.order.filter(stage=>stage!==selectedNode.data.stage);change(next);setSelected("data");}}}>Remove stage</Button></>}
          {selectedNode && <><details><summary>Key connections (drag alternative)</summary>{selectedNode.data.inputs.map(input=><label className="field" key={input.id}>{input.alias ?? input.field}: {input.key}<select aria-label={`Connect ${input.key}`} value="" onChange={event=>{
            const [source,handle]=JSON.parse(event.target.value);change(connect(draft,{source,sourceHandle:handle,target:selected,targetHandle:input.id}));
          }}><option value="">Choose preceding producer…</option>{graph.nodes.flatMap(node=>node.data.outputs.filter(output=>validConnection(draft,{source:node.id,sourceHandle:output.id,target:selected,targetHandle:input.id})).map(output=><option key={`${node.id}:${output.id}`} value={JSON.stringify([node.id,output.id])}>{node.data.title} → {output.key} ({output.domain})</option>))}</select></label>)}</details>
            <details><summary>Visual position</summary>{["x","y"].map(axis=><label className="field" key={axis}>{axis}<input type="number" aria-label={`Node position ${axis}`} value={selectedNode.position[axis as "x"|"y"]} onChange={event=>change({...draft,positions:{...draft.positions,[selected]:{...selectedNode.position,[axis]:Number(event.target.value)}}})}/></label>)}</details></>}
        </fieldset></aside>
      </div>
    </> : view==="Training" && draft ? <section className="training-view"><div className="view-title"><div><span className="eyebrow">EXISTING SCENARIO SETTINGS</span><h1>Training configuration</h1></div><p className="muted">Changes edit your draft, not a running operation.</p></div><fieldset disabled={state.blocked || pending}>
      <div className="training-grid"><article><h2>Training</h2><Fields value={draft.config.training} schema={modelSchema("TrainingSpec")} root={schema} basic={["max_epochs","batch_size","lr","accelerator","devices","precision","optimizer","scheduler"]} prefix="training." errors={errors} onChange={(value,validate)=>edit(["training"],value,validate)}/></article>
        <article><h2>Data & loader</h2><Fields value={draft.config.data.loader as RecordValue} schema={modelSchema("LoaderSpec")} root={schema} prefix="data.loader." errors={errors} onChange={(value,validate)=>edit(["data","loader"],value,validate)}/><h2>Execution</h2><StructuredField key={JSON.stringify(draft.config.execution)} label="execution" value={draft.config.execution} onChange={value=>edit(["execution"],value,true)}/><p className="muted">Local: available. Ray: {catalog.data?.backends.ray.available ? "installed; existing cluster only" : "not installed in this runtime"}. No silent backend substitution.</p></article>
        <article><h2>Checkpoint & logging</h2>{["checkpoint","logging","callbacks","exports"].map(key=><StructuredField key={`${key}:${JSON.stringify(draft.config[key])}`} label={key} value={draft.config[key]} onChange={value=>edit([key],value,true)}/>)}<label className="field">Resume Trainer checkpoint<input aria-label="Resume Trainer checkpoint" placeholder="Optional authorized local .ckpt path" value={checkpoint} onChange={event=>setCheckpoint(event.target.value)}/></label><p className="muted">Resume follows the checkpoint scenario semantics. Local only. Review the checkpoint-derived source at launch. Checkpoints are trusted Python inputs.</p></article></div>
      </fieldset></section> : view==="Execution" || view==="Artifacts" ? <Execution connection={connection} identity={operationId} choose={setOperationId} artifacts={view==="Artifacts"} report={report}/> : <section className="welcome"><span className="eyebrow">YOUR RUNTIME. YOUR SCENARIO.</span><h1>Build on what you already have.</h1><p>Choose an installed scenario recipe or open existing NexuML YAML.<br/>Studio edits your configuration. NexuML runs it.</p><p className="muted">Discovery is loading trusted installed definitions. Explicit resolution/build can execute Python.</p>{catalog.isError && <p className="field-error">{catalog.error.message}</p>}</section>}
    <section className="diagnostics" aria-label="Problems and checks"><div className="diagnostic-heading"><strong>Problems & checks</strong><div><Button disabled={!state.past.length || state.blocked || pending} onClick={state.undo} aria-label="Undo"><Undo2 size={15}/></Button><Button disabled={!state.future.length || state.blocked || pending} onClick={state.redo} aria-label="Redo"><Redo2 size={15}/></Button><Button disabled={!draft || pending || state.blocked} onClick={()=>act(async()=>{
      const operation=await request<Operation>(connection,"/build",payload());setBuild({id:operation.id,source:currentSignature});setOperationId(operation.id);setMessage("Executable build started; constructors and dummy forwards run in the selected interpreter.");
    })}>Build check (executes code)</Button></div></div>
      <div role="status">{pending ? "Waiting for selected NexuML…" : message || "Field checks do not compile. Build is explicit."}</div>
      {build && <p className="build-state">Build: {buildResult.data?.status ?? "connecting"}{!buildCurrent && " · stale (draft changed)"}{buildResult.data?.status==="succeeded" && buildCurrent && <code> · Final key shapes: {JSON.stringify(buildResult.data.result?.shapes)}</code>}{buildResult.data?.status==="failed" && <code>{JSON.stringify(buildResult.data.result)}</code>}</p>}
      <div className="problems-list">{errors.map((error,index)=><Button key={index} onClick={()=>{const location=error.loc;if(location[0]==="pipeline" && location[1]==="stages" && draft){select(draft.ids[String(location[2])]?.[Number(location[3])] ?? "data");setView("Pipeline");}else setView("Training");}}>{error.loc.join(".")}: {error.message}</Button>)}{graph?.problems.map(problem=><p key={problem}>{problem}</p>)}{catalog.data?.errors.map((error,index)=><p key={index}>{error.module}: {error.message}</p>)}</div>
    </section>
    <footer className="statusbar"><span className="connection-dot"/>Local runtime · NexuML {runtime.data.nexuml_version}<code title={runtime.data.python_executable}>{runtime.data.python_executable}</code><span>Interface 1 · {draft ? `${draft.order.length} ordered stages` : "No scenario selected"}</span></footer>
    <Dialog.Root open={!!review} onOpenChange={open=>{if(!open&&!pending)setReview(null);}}><Dialog.Portal><Dialog.Backdrop className="dialog-backdrop"/><Dialog.Popup className="dialog-popup"><Dialog.Title>Launch existing NexuML training</Dialog.Title><Dialog.Description>Trusted component constructors and configured callbacks execute local Python. This freezes a source configuration; later draft edits do not change the run.</Dialog.Description>
      <pre>{JSON.stringify({source_revision:review?.document.semantic_revision,execution:review?.document.data.execution,training:review?.document.data.training,checkpoint_resume:checkpoint || null,exports:review?.document.data.exports},null,2)}</pre>
      <div className="toolbar"><Button disabled={pending} onClick={()=>setReview(null)}>Cancel</Button><Button className="primary" disabled={pending || !review || review.source!==currentSignature} onClick={()=>act(async()=>{
        const launched=await request<Operation>(connection,"/train",{data:review!.document.data,stage_order:review!.document.stage_order,trainer_checkpoint:checkpoint || null});setOperationId(launched.id);setView("Execution");setReview(null);
      })}>Confirm & start training</Button></div></Dialog.Popup></Dialog.Portal></Dialog.Root>
  </main>;
}

function ComponentFields({component,catalog,onChange,errors,prefix}: {
  component:Component;catalog?:Catalog;onChange:(value:Component,validate?:boolean)=>void|Promise<void>;errors:ApiError["fields"];prefix:string;
}) {
  const entry=catalog?.components.find(item=>item.name===component.type && item.version===component.version);
  return <><div className="component-identity"><strong>{component.type}</strong><span>v{component.version}</span></div>
    <p className="muted">{entry?.import_target ?? "Unknown definition: fields are retained, resolve this library before validation."}</p>
    <Fields value={component.params} schema={entry?.schema} root={entry?.schema} errors={errors} prefix={prefix} onChange={(params,validate)=>onChange({...component,params},validate)}/>
    {entry && <details><summary>Definition schema & descriptions</summary><pre>{JSON.stringify(entry.schema,null,2)}</pre></details>}</>;
}
