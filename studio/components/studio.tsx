"use client";
import { useEffect, useMemo, useState } from "react";
import type { CSSProperties } from "react";
import { QueryClient, QueryClientProvider, useQuery } from "@tanstack/react-query";
import { useStore } from "zustand";
import { Dialog } from "@base-ui/react/dialog";
import { Check, Code2, ChevronDown, Layers, Library, Play, Redo2, Save, Undo2, X } from "lucide-react";
import { Canvas } from "./canvas";
import { ComponentFields, Fields, ValueField } from "./fields";
import { ComponentBrowser } from "./component-browser";
import { Structure } from "./structure";
import { Execution } from "./execution";
import { ExecutionSettings } from "./execution-settings";
import { RunReview } from "./run-review";
import { Button } from "./ui/button";
import { createDraftStore, layoutSignature, yamlText } from "../model/draft";
import { addStage, connect, disconnect, newSnapshot, placeNode, project, signature, STAGE_INSET, transferLayer, validConnection } from "../model/graph";
import { schemaDefault } from "../model/schema";
import { ApiError, request } from "../model/client";
import type { Catalog, ConnectionInfo, Document, Entry, Layout, Operation, RecordValue, Snapshot } from "../model/types";

const trainingSections=["Basics","Optimizer and schedule","Data loading","Execution","Checkpoints and logging"];
const basicTraining=["max_epochs","batch_size","lr","accelerator","devices","precision"];

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
  const [leftTab,setLeftTab]=useState("Structure");
  const [inspectorTab,setInspectorTab]=useState("Settings");
  const [trainingSection,setTrainingSection]=useState("Basics");
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
  const [fileAction,setFileAction]=useState<"recipe"|"open"|"save"|null>(null);
  const [problemsOpen,setProblemsOpen]=useState(false);
  const [dismissedBuild,setDismissedBuild]=useState("");
  const [yamlBuffer,setYamlBuffer]=useState<string|null>(null);
  const [message,setMessage]=useState("");
  const [errors,setErrors]=useState<ApiError["fields"]>([]);
  const [checkFailed,setCheckFailed]=useState(false);
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
  const [insertion,setInsertion]=useState<{entry:Entry;position?:{x:number;y:number}}|null>(null);
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
    const controls=[...document.querySelectorAll<HTMLElement>("[data-field]")].filter(control=>!control.closest("[hidden]"));
    const matches=controls.filter(control=>control.dataset.field===focusPath || focusPath.startsWith(`${control.dataset.field}.`))
      .sort((a,b)=>(b.dataset.field?.length ?? 0)-(a.dataset.field?.length ?? 0) || Number(b.matches("input,textarea"))-Number(a.matches("input,textarea")));
    const control=matches[0];
    if(!control)return;
    for(let parent=control.parentElement;parent;parent=parent.parentElement)if(parent instanceof HTMLDetailsElement)parent.open=true;
    control.scrollIntoView({block:"nearest"});control.focus();
  },[focusPath,view,selected,focusRequest,inspectorTab,trainingSection]);
  const navigate=(location:(string|number)[])=>{
    setShowYaml(false);setInspectorTab(location.some(part=>["keys_in","keys_out","label_key","label_in_x","meta_in","meta_out"].includes(String(part))) ? "Routing" : "Settings");
    if(location[0]==="pipeline" && location[1]==="stages" && draft){setView("Pipeline");select(draft.ids[String(location[2])]?.[Number(location[3])] ?? `stage:${location[2]}`);}
    else if(location[0]==="data"){setView("Pipeline");select("data");}
    else if(location[0]==="evaluation" && location[1]==="algorithms" && Number.isInteger(location[2])){setView("Pipeline");select(`evaluation:${location[2]}`);}
    else if(location[0]==="training" && ["loss_keys","metric_keys"].includes(String(location[1]))){setView("Pipeline");select("objectives");}
    else {setView("Training");setTrainingSection(location[0]==="execution" ? "Execution" : ["checkpoint","logging","callbacks","exports"].includes(String(location[0])) ? "Checkpoints and logging" : ["optimizer","scheduler"].includes(String(location[1])) ? "Optimizer and schedule" : "Basics");}
    if(location[0]==="data" && location[1]==="loader"){setView("Training");setTrainingSection("Data loading");}
    setRightVisible(true);setFocusPath(location.join("."));setFocusRequest(value=>value+1);
  };
  const report=(error:unknown)=>{setMessage(error instanceof Error ? error.message : String(error));setErrors(error instanceof ApiError ? error.fields : []);setCheckFailed(true);setProblemsOpen(true);};
  const act=async(action:()=>Promise<void>)=>{if(pending)return;setPending(true);setMessage("");setErrors([]);setCheckFailed(false);try{await action();}catch(error){report(error);}finally{setPending(false);}};
  const replace=(document:Document,layout?:Layout)=>{state.load(document,layout);setFile(document.path ?? "scenario.yaml");setYamlBuffer(null);setShowYaml(false);setLayoutRevision(null);setFocusPath("");setDestinationStage("");setInsertionSlot("append");setLeftTab("Structure");setInspectorTab("Settings");};
  const change=(next:Snapshot)=>{state.change(next);setErrors([]);setCheckFailed(false);};
  const edit=async(path:(string|number)[],value:unknown,validate=false)=>{
    const original=store.getState().draft;
    if(!original)return;const next=structuredClone(original);
    if(!path.length)next.config={...next.config,...value as RecordValue};
    else {let target:unknown=next.config;for(const key of path.slice(0,-1))target=(target as RecordValue)[key];(target as RecordValue)[path.at(-1)!]=value;}
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
      const stage=droppedStage ?? (next.order.includes(destinationStage) ? destinationStage : selectedNode?.data.stage);
      if((position && !droppedStage) || !stage){setDestinationStage("");setInsertionSlot("append");setInsertion({entry,position});return;}
      const index=slot ?? (droppedStage ? next.ids[stage].length : Math.min(slotIndex(insertionSlot,next.ids[stage].length),next.ids[stage].length));
      const id=crypto.randomUUID();next.config.pipeline.stages[stage].splice(index,0,{component:defaults(entry),keys_in:[],keys_out:[]});next.ids[stage].splice(index,0,id);next.selected=id;
      // A new ordered insertion does not rearrange the other freely placed layers.
      for(const node of graph?.nodes ?? [])next.positions[node.id]=node.position;
      if(position){
        const parent=graph?.nodes.find(node=>node.id===`stage:${stage}`)?.position ?? {x:0,y:0};
        const point={x:position.x-parent.x,y:position.y-parent.y};
        next.positions[id]={x:Math.max(20,point.x),y:Math.max(STAGE_INSET,point.y)};
      }
      else next.positions[id]={x:20,y:Math.max(STAGE_INSET,graph?.nodes.find(node=>node.id===`stage:${stage}`)?.data.minHeight ?? STAGE_INSET)};
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
  const showComponents=(stage:string)=>{setDestinationStage(stage);setInsertionSlot("append");setLeftTab("Components");setLeftVisible(true);setPanel("Components");requestAnimationFrame(()=>document.querySelector<HTMLInputElement>('[aria-label="Search components"]')?.focus());};
  const insertStage=draft?.order.includes(destinationStage) ? destinationStage : selectedNode?.data.stage ?? "";
  const save=(path=file)=>act(async()=>{
    const document=await request<Document>(connection!,"/config/save",{...payload(),path,
      base_revision:path===state.path ? state.baseRevision : null});
    state.markSaved(document,draft!);
    store.setState({savedLayout:""});
    if(path!==state.path)setLayoutRevision(null);
    const layout=await request<{base_revision:string}>(connection!,"/config/layout/save",{path,
      base_revision:path===state.path ? layoutRevision : null,semantic_revision:document.semantic_revision,
      layout:{ids:draft!.ids,positions:draft!.positions,names:draft!.names,sizes:draft!.sizes}});
    state.markLayoutSaved(draft!);setLayoutRevision(layout.base_revision);setFileAction(null);setMessage("Configuration and separate layout saved.");
  });
  const openFile=()=>{if(discard())void act(async()=>{
    const document=await request<Document>(connection!,"/config/load",{path:file});
    const sidecar=await request<{semantic_revision?:string;layout?:Layout;base_revision:string|null}>(connection!,"/config/layout/load",{path:file});
    replace(document,sidecar.semantic_revision===document.semantic_revision ? sidecar.layout : undefined);setLayoutRevision(sidecar.base_revision);setFileAction(null);
    if(sidecar.layout && sidecar.semantic_revision!==document.semantic_revision)setMessage("External semantic revision changed. Stale layout ignored.");
  });};
  const saveDraft=()=>{if(!state.saved){setFileAction("save");return;}setFile(state.path);void save(state.path);};
  const buildStatus=!build ? "Not checked" : !buildCurrent ? "Needs rechecking" : buildResult.data?.status==="succeeded" ? "Passed" : buildResult.data?.status==="failed" ? "Failed" : "Checking";
  const problemCount=Math.max(errors.length,checkFailed ? 1 : 0)+(graph?.problems.length ?? 0)+(buildCurrent && buildResult.data?.status==="failed" ? Math.max(1,buildFailure?.fields?.length ?? 0) : 0);
  const expandedProblems=problemsOpen || (buildResult.data?.status==="failed" && dismissedBuild!==build?.id);
  const trainingFields=(keys:string[],basic?:string[])=>draft && <Fields value={Object.fromEntries(Object.entries(draft.config.training).filter(([key])=>keys.includes(key)))}
    schema={{...modelSchema("TrainingSpec"),properties:Object.fromEntries(Object.entries(modelSchema("TrainingSpec")?.properties ?? {}).filter(([key])=>keys.includes(key))),required:modelSchema("TrainingSpec")?.required?.filter(key=>keys.includes(key))}}
    root={schema} basic={basic} prefix="training." errors={errors} rawLabel={keys.includes("optimizer") ? "Raw optimizer settings" : "Raw training settings"} {...fieldContext} onChange={(value,validate)=>edit(["training"],{...draft.config.training,...value},validate)}/>;

  if(!connection || runtime.isError || runtime.data?.interface_version!==2) return <main className="startup"><div className="brand"><Layers/>NexuML <span>STUDIO</span></div><h1>{boot.isError || runtime.isError || (runtime.data && runtime.data.interface_version!==2) ? "Connection unavailable" : "Connecting to your runtime"}</h1><p>{boot.error?.message ?? runtime.error?.message ?? (runtime.data && runtime.data.interface_version!==2 ? "Studio requires interface 2. Use matching Python and Studio versions." : "Verifying the selected local NexuML installation…")}</p><p>Start with <code>nexuml-studio --python /path/to/python .</code> or choose an existing <code>nexuml</code> executable. Nothing is installed automatically.</p><Button onClick={()=>{void boot.refetch();void runtime.refetch();}}>Retry connection</Button></main>;
  return <main className="workbench">
    <header className="topbar"><div className="brand"><Layers size={22}/>NexuML <span>STUDIO</span></div>
      <details className="scenario-menu"><summary aria-label="Scenario menu" title={runtime.data.working_directory}>{draft?.config.name ?? "Scenario"}<ChevronDown size={14}/></summary><div>
        <Button onClick={event=>{event.currentTarget.closest("details")!.open=false;setFileAction("recipe");}}>Choose a recipe</Button>
        <Button onClick={event=>{event.currentTarget.closest("details")!.open=false;setFileAction("open");}}>Open YAML</Button>
         <Button disabled={!draft || pending || state.blocked} onClick={event=>{event.currentTarget.closest("details")!.open=false;saveDraft();}}>Save</Button>
        <Button disabled={!draft || pending || state.blocked} onClick={event=>{event.currentTarget.closest("details")!.open=false;setFileAction("save");}}>Save as…</Button>
      </div></details>
      <span className="save-state">{!draft ? "No scenario loaded" : state.blocked?"Unapplied YAML":!state.saved?"Unsaved recipe":dirty?"Unsaved changes":"Saved"}</span>
      <div className="top-actions"><Button disabled={!state.past.length || state.blocked || pending} onClick={state.undo} aria-label="Undo"><Undo2 size={15}/></Button><Button disabled={!state.future.length || state.blocked || pending} onClick={state.redo} aria-label="Redo"><Redo2 size={15}/></Button>
        <Button disabled={!draft || pending || state.blocked} onClick={saveDraft}><Save size={15}/>Save</Button>
        <details className="check-menu"><summary><Check size={15}/>Check<ChevronDown size={14}/></summary><div>
          <Button disabled={!draft || pending || state.blocked} onClick={event=>{event.currentTarget.closest("details")!.open=false;void act(async()=>{const checked=await request<Document>(connection,"/config/validate",payload());setMessage(`Fields valid · ${checked.semantic_revision.slice(0,12)}. No code compiled.`);});}}>Check fields</Button><p className="muted">Validates configuration; does not execute model code.</p>
          <Button disabled={!draft || pending || state.blocked} onClick={event=>{event.currentTarget.closest("details")!.open=false;void act(async()=>{const operation=await request<Operation>(connection,"/build",payload());setBuild({id:operation.id,source:currentSignature});setOperationId(operation.id);setMessage("Executable build started; constructors and dummy forwards run in the selected interpreter.");});}}>Build model</Button><p className="muted">Executes constructors and dummy forwards in trusted Python.</p>
        </div></details>
         <Button data-run-trigger className="primary" disabled={!draft || pending || state.blocked || graph?.unconnected} title={graph?.unconnected ? "Connect required inputs before launching" : undefined} onClick={()=>act(async()=>{const checked=await request<Document>(connection,checkpoint ? "/train/prepare" : "/config/validate",{...payload(),...(checkpoint ? {trainer_checkpoint:checkpoint} : {})});setReview({document:checked,source:currentSignature});})}><Play size={15}/>Run…</Button></div>
    </header>
    <nav className="nav-tabs" aria-label="Workbench views"><div>{["Pipeline","Training","Runs"].map(tab=><Button key={tab} aria-current={view===tab?"page":undefined} onClick={()=>{setView(tab);setShowYaml(false);}}>{tab}</Button>)}</div><div>
      <Button aria-pressed={showLibraries} onClick={()=>setShowLibraries(!showLibraries)}><Library size={15}/>Libraries</Button>
      <Button disabled={!draft} aria-pressed={showYaml && view!=="Runs"} onClick={()=>{if(view==="Runs"){setView("Pipeline");setShowYaml(true);}else setShowYaml(!showYaml);}}><Code2 size={15}/>YAML</Button>
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
    {draft && <section hidden={!showYaml || view==="Runs"} className="yaml-panel"><div className="section-title"><h2>Ordinary NexuML YAML</h2><Button aria-label="Close YAML" onClick={()=>setShowYaml(false)}><X size={16}/></Button></div><p className="muted">Changes apply only after field validation. Invalid/unapplied text blocks graph edits; source comments and formatting are not preserved.</p>
      <textarea aria-label="Scenario YAML" disabled={pending} spellCheck={false} value={yamlBuffer ?? yamlText(draft)} onChange={event=>{setYamlBuffer(event.target.value);state.block(true);}} />
      <div className="toolbar"><Button className="primary" disabled={pending || yamlBuffer===null} onClick={()=>act(async()=>{
        const document=await request<Document>(connection,"/config/validate",{yaml:yamlBuffer});state.block(false);
        state.change(newSnapshot(document.data,document.stage_order,draft));setYamlBuffer(null);setMessage("YAML applied to the single configuration draft.");
      })}>Apply YAML</Button><Button disabled={yamlBuffer===null} onClick={()=>{if(window.confirm("Discard the unapplied YAML buffer?")){setYamlBuffer(null);state.block(false);}}}>Discard buffer</Button></div>
    </section>}
    {state.blocked && (!showYaml || view==="Runs") && <div className="buffer-notice">Unapplied YAML retained. Editing, saving and launch are blocked.<Button onClick={()=>{setView("Pipeline");setShowYaml(true);}}>Return to YAML</Button></div>}
    {draft && graph && <section className="pipeline-view" hidden={view!=="Pipeline" || showYaml}>
      <div className="panel-tabs"><Button aria-pressed={panel==="Components"} onClick={()=>setPanel("Components")}>Structure / Components</Button><Button aria-pressed={panel==="Canvas"} onClick={()=>setPanel("Canvas")}>Canvas</Button><Button aria-pressed={panel==="Properties"} onClick={()=>setPanel("Properties")}>Properties</Button></div>
      <div className={`editor-grid panel-${panel.toLowerCase()} ${leftVisible?"":"hide-components"} ${rightVisible?"":"hide-properties"}`}
        style={{"--left-panel":`${leftVisible?leftWidth:0}px`,"--right-panel":`${rightVisible?rightWidth:0}px`} as CSSProperties}>
        <aside className="components-panel"><div className="section-title"><div className="segmented" aria-label="Navigation panel">{["Structure","Components"].map(tab=><Button key={tab} aria-pressed={leftTab===tab} onClick={()=>setLeftTab(tab)}>{tab}</Button>)}</div><Button aria-label="Collapse navigation panel" onClick={()=>{setLeftVisible(false);document.querySelector<HTMLElement>(".panel-settings summary")?.focus();}}><X size={14}/></Button></div>
          <div hidden={leftTab!=="Components"}><input aria-label="Search components" placeholder="Search installed components…" value={query} onChange={event=>setQuery(event.target.value)}/>
          <label className="field">Insert into stage<select aria-label="Insert into stage" value={insertStage} onChange={event=>{setDestinationStage(event.target.value);setInsertionSlot("append");}}><option value="">Choose a destination…</option>{draft.order.map(stage=><option key={stage}>{stage}</option>)}</select></label>
          <label className="field">Layer insertion slot<select aria-label="Layer insertion slot" value={insertionSlot} onChange={event=>setInsertionSlot(event.target.value)}><option value="append">Append (last)</option>{(draft.ids[insertStage] ?? []).map((id,index)=><optgroup key={id} label={`Layer ${index+1}`}><option value={index}>Before layer {index+1}</option><option value={`after:${index}`}>After layer {index+1}</option></optgroup>)}</select></label>
          {stageSkipped(insertStage) && <p className="field-error">Skipped destination: inserted layers will not execute.</p>}
          <ComponentBrowser entries={catalog.data?.components ?? []} query={query} blocked={state.blocked || pending} add={addComponent}/>
          </div><div hidden={leftTab!=="Structure"}><Structure snapshot={draft} selected={selected} select={select} change={change} blocked={state.blocked || pending} report={report} create={()=>createStage()}/></div>
        </aside>
        {leftVisible && <PanelSeparator label="Navigation panel width" value={leftWidth} min={220} max={380} change={setLeftWidth}/>}
        <Canvas snapshot={draft} selected={selected} select={select} change={change} blocked={state.blocked || pending} catalog={catalog.data} insert={addComponent} report={report} remove={removeNode} create={createStage} addLayer={showComponents}/>
        {rightVisible && <PanelSeparator label="Property panel width" value={rightWidth} min={260} max={440} change={setRightWidth} right/>}
        <aside className="properties-panel"><div className="section-title"><h2>Properties</h2><Button aria-label="Collapse property panel" onClick={()=>{setRightVisible(false);document.querySelector<HTMLElement>(".panel-settings summary")?.focus();}}><X size={14}/></Button></div><fieldset disabled={state.blocked || pending}><legend>{selectedNode?.data.title ?? "Select a node"}</legend>
          <div className="segmented inspector-tabs">{["Settings","Routing"].map(tab=><Button key={tab} aria-pressed={inspectorTab===tab} onClick={()=>setInspectorTab(tab)}>{tab}</Button>)}</div>
          {selectedNode && <details><summary>Edit display name</summary><label className="field">Display name<input aria-label="Node display name" placeholder={selectedNode.data.title} value={draft.names?.[selected] ?? ""} onChange={event=>change({...draft,names:{...draft.names,[selected]:event.target.value}})}/><small className="muted">Visual only; routing keys are unchanged.</small></label></details>}
          <div hidden={inspectorTab!=="Settings"}>
          {selectedNode?.data.kind==="data" && <Fields value={draft.config.data} schema={modelSchema("DataSpec")} root={schema} prefix="data." errors={errors} basic={["source","input_shapes","feature_key","datasets","targets"]} {...fieldContext} onChange={(value,validate)=>edit(["data"],value,validate)}/>}
          {selectedNode?.data.kind==="layer" && (()=>{
            const {stage,index}=selectedNode.data;const layer=draft.config.pipeline.stages[stage!][index!];
             return <><ComponentFields key={selected} kind="layer" component={layer.component} errors={errors} {...fieldContext} prefix={`pipeline.stages.${stage}.${index}.component.params.`} onChange={(value,validate)=>edit(["pipeline","stages",stage!,index!,"component"],value,validate)}/>
              <details><summary>Advanced layer settings</summary><Fields value={Object.fromEntries(Object.entries(layer).filter(([key])=>!["component","keys_in","keys_out","label_key","label_in_x","meta_in","meta_out"].includes(key)))} schema={{properties:Object.fromEntries(Object.entries(modelSchema("LayerSpec")?.properties ?? {}).filter(([key])=>!["component","keys_in","keys_out","label_key","label_in_x","meta_in","meta_out"].includes(key)))}} root={schema} prefix={`pipeline.stages.${stage}.${index}.`} errors={errors} {...fieldContext} onChange={(value,validate)=>edit(["pipeline","stages",stage!,index!],{...layer,...value},validate)}/></details>
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
          </div><div hidden={inspectorTab!=="Routing"}>
           {selectedNode?.data.kind==="layer" && (()=>{const {stage,index}=selectedNode.data;const layer=draft.config.pipeline.stages[stage!][index!];return <Fields key={selected} value={Object.fromEntries(Object.entries(layer).filter(([key])=>["keys_in","keys_out","label_key","label_in_x","meta_in","meta_out"].includes(key)))} schema={{properties:Object.fromEntries(Object.entries(modelSchema("LayerSpec")?.properties ?? {}).filter(([key])=>["keys_in","keys_out","label_key","label_in_x","meta_in","meta_out"].includes(key)))}} root={schema} prefix={`pipeline.stages.${stage}.${index}.`} errors={errors} rawLabel="Raw routing settings" {...fieldContext} onChange={(value,validate)=>edit(["pipeline","stages",stage!,index!],{...layer,...value},validate)}/>;})()}
          {selectedNode && <details open><summary>Key connections (drag alternative)</summary>{selectedNode.data.inputs.map(input=><div className="field" key={input.id}>{input.add ? `Add ${input.domain} input` : `${input.alias ?? input.field}: ${input.key || "Unconnected"}`}<select aria-label={`Connect ${input.key || (input.add ? `new ${input.domain} input` : input.field)}`} value="" onChange={event=>{
            const [source,handle]=JSON.parse(event.target.value);change(connect(draft,{source,sourceHandle:handle,target:selected,targetHandle:input.id},catalog.data));
          }}><option value="">Choose preceding producer…</option>{graph.nodes.flatMap(node=>node.data.outputs.filter(output=>validConnection(draft,{source:node.id,sourceHandle:output.id,target:selected,targetHandle:input.id},catalog.data)).map(output=><option key={`${node.id}:${output.id}`} value={JSON.stringify([node.id,output.id])}>{node.data.title} → {output.key} ({output.domain})</option>))}</select>
            {graph.edges.some(edge=>edge.target===selected && edge.targetHandle===input.id) && <Button disabled={!!input.defaultKey} title={input.defaultKey ? "Inherited runtime default: change the key or disable the evaluator" : "Remove this key route"} aria-label={`Remove connection ${input.key}`} onClick={()=>change(disconnect(draft,[{target:selected,targetHandle:input.id}],catalog.data))}>Remove connection</Button>}
            {input.defaultKey && <small className="muted">Runtime default: {input.defaultKey}; clearing the field restores it.</small>}
          </div>)}</details>}
          </div>{selectedNode && <details><summary>Advanced layout</summary><details><summary>Visual position</summary>{["x","y"].map(axis=><label className="field" key={axis}>{axis}<input type="number" aria-label={`Node position ${axis}`} value={selectedNode.position[axis as "x"|"y"]} onChange={event=>change(placeNode(draft,selected,{...selectedNode.position,[axis]:Number(event.target.value)}))}/></label>)}</details></details>}
        </fieldset></aside>
      </div>
    </section>}
    {draft && <section className="training-view" hidden={view!=="Training" || showYaml}><div className="view-title"><h1>Training configuration</h1><p className="muted">Changes edit your draft, not a running operation.</p></div>
      <div className="training-layout"><nav className="training-sections" aria-label="Training sections">{trainingSections.map(section=><Button key={section} aria-current={trainingSection===section ? "page" : undefined} onClick={()=>setTrainingSection(section)}>{section}</Button>)}</nav>
      <label className="field training-section-picker">Section<select aria-label="Training section" value={trainingSection} onChange={event=>setTrainingSection(event.target.value)}>{trainingSections.map(section=><option key={section}>{section}</option>)}</select></label>
      <fieldset disabled={state.blocked || pending} className="training-content">{trainingSections.map(section=><article key={section} hidden={trainingSection!==section}><h2>{section}</h2>
        {section==="Basics" && <><ValueField label="Scenario name" path="name" schema={schema?.properties?.name} value={draft.config.name} onChange={value=>edit(["name"],value)}/>
          {(draft.config.data.loader as RecordValue)?.batch_size!=null && <div className="buffer-notice">Loader batch size overrides training batch size: {String((draft.config.data.loader as RecordValue).batch_size)}<Button onClick={()=>setTrainingSection("Data loading")}>Edit loader override</Button></div>}
          {trainingFields([...new Set([...Object.keys(modelSchema("TrainingSpec")?.properties ?? {}),...Object.keys(draft.config.training)])].filter(key=>!["optimizer","scheduler","loss_keys","metric_keys"].includes(key)),basicTraining)}
          <Button onClick={()=>{setView("Pipeline");setInspectorTab("Settings");select("objectives");}}>Edit objectives and metrics</Button>
          <details><summary>Evaluation settings</summary><ValueField label="Evaluation" path="evaluation" schema={schema?.properties?.evaluation} root={schema} value={draft.config.evaluation} errors={errors} {...fieldContext} onChange={(value,validate)=>edit(["evaluation"],value,validate)}/></details></>}
        {section==="Optimizer and schedule" && trainingFields(["optimizer","scheduler"])}
        {section==="Data loading" && <Fields value={draft.config.data.loader as RecordValue} schema={modelSchema("LoaderSpec")} root={schema} prefix="data.loader." errors={errors} rawLabel="Raw loader settings" {...fieldContext} onChange={(value,validate)=>edit(["data","loader"],value,validate)}/>}
        {section==="Execution" && <><p className="muted">Backend placement resources are separate from trainer accelerator/devices in Basics.</p><ExecutionSettings value={draft.config.execution} catalog={catalog.data} errors={errors} onChange={(value,validate)=>edit(["execution"],value,validate)}/></>}
        {section==="Checkpoints and logging" && <><Fields value={Object.fromEntries(["checkpoint","logging","callbacks","exports"].map(key=>[key,draft.config[key]]))} schema={{properties:Object.fromEntries(Object.entries(schema?.properties ?? {}).filter(([key])=>["checkpoint","logging","callbacks","exports"].includes(key)))}} root={schema} errors={errors} rawLabel="Raw checkpoint and logging settings" {...fieldContext} onChange={(value,validate)=>edit([],value,validate)}/>
          <label className="field">Resume Trainer checkpoint<input aria-label="Resume Trainer checkpoint" placeholder="Optional authorized local .ckpt path" value={checkpoint} onChange={event=>setCheckpoint(event.target.value)}/></label><p className="muted">Resume follows checkpoint scenario semantics. Local only. Review the checkpoint-derived source at launch. Checkpoints are trusted Python inputs.</p></>}
      </article>)}</fieldset></div></section>}
    <div className="runs-view" hidden={view!=="Runs"}><Execution connection={connection} draft={draft} identity={operationId} choose={setOperationId} report={report} editSettings={()=>setView("Training")} loadSettings={identity=>act(async()=>{if(!discard())return;replace(await request<Document>(connection,`/operations/${identity}/config`));setView("Training");})}/></div>
    {!draft && view!=="Runs" && <section className="welcome"><h1>Start with a scenario.</h1><p>Choose an installed recipe or open NexuML YAML.<br/>Your configuration stays local. NexuML runs it.</p><div className="toolbar"><Button className="primary" onClick={()=>setFileAction("recipe")}>Choose a recipe</Button><Button onClick={()=>setFileAction("open")}>Open YAML</Button></div><p className="muted">Discovery imports trusted definitions. Recipe resolution and build can execute Python.</p>{catalog.isError && <p className="field-error">{catalog.error.message}</p>}</section>}
    <section className="diagnostics" aria-label="Problems and checks"><div className="diagnostic-heading"><Button aria-expanded={expandedProblems} onClick={()=>{setProblemsOpen(!expandedProblems);setDismissedBuild(build?.id ?? "");}}>Problems ({problemCount})<ChevronDown size={14}/></Button><span>Build: {buildStatus}</span><span className="muted">Local runtime · NexuML {runtime.data.nexuml_version}</span></div>
      <div role="status">{pending ? "Waiting for selected NexuML…" : message}</div>
      <div hidden={!expandedProblems} className="diagnostic-details">
      {build && <div className="build-state">Build: {buildResult.data?.status ?? "connecting"}{!buildCurrent && " · stale (draft changed)"}{buildResult.data?.status==="succeeded" && buildCurrent && <code> · Final key shapes: {Object.entries(buildResult.data.result?.shapes as RecordValue ?? {}).map(([key,value])=>`${key} [${Array.isArray(value) ? value.join(" × ") : value}]`).join(" · ")}</code>}{buildResult.data?.status==="failed" && <p className="field-error" role="alert">{buildFailure?.message ?? "Build failed. Inspect its process logs and frozen launch configuration."}</p>}
        {buildCurrent && buildFailure?.fields?.map((field,index)=><Button key={index} onClick={()=>navigate(field.loc)}>{field.loc.join(".")}: {field.message}</Button>)}
        {buildResult.data?.result && <details><summary>Expert: build result JSON</summary><pre>{JSON.stringify(buildResult.data.result,null,2)}</pre></details>}
      </div>}
      <div className="problems-list" aria-label="Draft problems">{errors.map((error,index)=><Button key={index} onClick={()=>navigate(error.loc)}>{error.loc.join(".")}: {error.message}</Button>)}{graph?.problems.map(problem=><p key={problem}>{problem}</p>)}</div>
       <details><summary>Runtime and discovery details ({catalog.data?.errors.length ?? 0})</summary><code>{runtime.data.python_executable}</code><p>Interface 2 · {draft ? `${draft.order.length} ordered stages` : "No scenario selected"}</p>{catalog.data?.errors.map((error,index)=><p key={index}>{error.module}: {error.message}</p>)}</details>
      </div>
    </section>
    <Dialog.Root open={!!fileAction} onOpenChange={open=>{if(!open&&!pending)setFileAction(null);}}><Dialog.Portal><Dialog.Backdrop className="dialog-backdrop"/><Dialog.Popup className="dialog-popup" finalFocus={()=>document.querySelector<HTMLElement>(".scenario-menu summary")}>
      <Dialog.Title>{fileAction==="recipe" ? "Choose a recipe" : fileAction==="save" ? "Save configuration as" : "Open YAML"}</Dialog.Title><Dialog.Description>{fileAction==="recipe" ? "Resolving an installed recipe executes trusted Python." : "Paths are relative to your selected working directory, or authorized absolute paths."}</Dialog.Description>
      {fileAction==="recipe" ? <label className="field">Installed scenario<select aria-label="Discovered scenario" value={scenario} onChange={event=>setScenario(event.target.value)}><option value="">Choose installed recipe…</option>{catalog.data?.scenarios.map(item=><option key={item.name}>{item.name}</option>)}</select></label> : <label className="field">Config path<input aria-label="Config path" value={file} onChange={event=>setFile(event.target.value)}/></label>}
      {message && <p className="field-error" role="alert">{message}</p>}
      <div className="toolbar"><Button disabled={pending} onClick={()=>setFileAction(null)}>Cancel</Button><Button className="primary" disabled={pending || (fileAction==="recipe" ? !scenario : !file)} onClick={()=>{
        if(fileAction==="save")void save();else if(fileAction==="open")openFile();else if(discard())void act(async()=>{replace(await request<Document>(connection,`/scenarios/${encodeURIComponent(scenario)}/resolve`,{}));setFileAction(null);});
      }}>{fileAction==="recipe" ? "Resolve recipe" : fileAction==="save" ? "Save file" : "Open file"}</Button></div>
    </Dialog.Popup></Dialog.Portal></Dialog.Root>
    <Dialog.Root open={!!creation} onOpenChange={open=>{if(!open)setCreation(null);}}><Dialog.Portal><Dialog.Backdrop className="dialog-backdrop"/><Dialog.Popup className="dialog-popup"><Dialog.Title>Add ordered stage</Dialog.Title><Dialog.Description>Insert {selectedStage ? `after ${selectedStage}` : "last"}. Canvas position never determines execution order.</Dialog.Description>
      <label className="field">Stage name<input aria-label="New stage name" value={stageName} onChange={event=>setStageName(event.target.value)}/></label>{draft?.order.includes(stageName.trim()) && <p className="field-error">Stage name already exists.</p>}
      <div className="toolbar"><Button onClick={()=>setCreation(null)}>Cancel</Button><Button className="primary" disabled={!draft || !stageName.trim() || draft.order.includes(stageName.trim()) || state.blocked || pending} onClick={()=>{change(addStage(draft!,stageName,selectedStage,creation?.position));setCreation(null);}}>Create stage</Button></div>
    </Dialog.Popup></Dialog.Portal></Dialog.Root>
    <Dialog.Root open={!!insertion} onOpenChange={open=>{if(!open)setInsertion(null);}}><Dialog.Portal><Dialog.Backdrop className="dialog-backdrop"/><Dialog.Popup className="dialog-popup"><Dialog.Title>Choose layer destination</Dialog.Title><Dialog.Description>Choose ordered membership explicitly before inserting this layer.</Dialog.Description>
      <label className="field">Stage<select aria-label="Dropped layer destination" value={destinationStage} onChange={event=>setDestinationStage(event.target.value)}><option value="">Choose a stage…</option>{draft?.order.map(stage=><option key={stage}>{stage}</option>)}</select></label><p>Append to {destinationStage || "chosen stage"}. {stageSkipped(destinationStage) && "Skipped: layer will not execute."}</p>
      <div className="toolbar"><Button onClick={()=>setInsertion(null)}>Cancel</Button><Button disabled={!destinationStage || state.blocked || pending} onClick={()=>{addComponent(insertion!.entry,destinationStage,undefined,draft!.ids[destinationStage].length);setInsertion(null);}}>Insert layer</Button></div>
    </Dialog.Popup></Dialog.Portal></Dialog.Root>
     {review && <RunReview initial={review.document} connection={connection} catalog={catalog.data}
       sourceChanged={review.source!==currentSignature} checkpoint={checkpoint} close={()=>setReview(null)}
        accepted={(operation,document)=>{const current=store.getState().draft;if(current){const next=structuredClone(current);next.config.execution=document.data.execution;change(next);}setOperationId(operation.id);setView("Runs");setReview(null);}}/>}
  </main>;
}

function PanelSeparator({label,value,min,max,change,right=false}:{label:string;value:number;min:number;max:number;change:(width:number)=>void;right?:boolean}) {
  return <div className={`panel-separator ${right ? "right" : "left"}`} role="separator" aria-label={label} aria-orientation="vertical" aria-valuenow={value} aria-valuemin={min} aria-valuemax={max} tabIndex={0}
    onKeyDown={event=>{if(["ArrowLeft","ArrowRight","Home","End"].includes(event.key)){event.preventDefault();change(event.key==="Home" ? min : event.key==="End" ? max : Math.max(min,Math.min(max,value+(event.key==="ArrowRight" ? 20 : -20)*(right ? -1 : 1))));}}}
    onPointerDown={event=>{event.currentTarget.setPointerCapture(event.pointerId);event.currentTarget.dataset.start=String(event.clientX);event.currentTarget.dataset.width=String(value);}}
    onPointerMove={event=>{if(event.currentTarget.hasPointerCapture(event.pointerId))change(Math.max(min,Math.min(max,Number(event.currentTarget.dataset.width)+(event.clientX-Number(event.currentTarget.dataset.start))*(right ? -1 : 1))));}}/>;
}
