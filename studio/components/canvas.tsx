"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { Background, Controls, Handle, MiniMap, NodeResizeControl, Position, ReactFlow, ReactFlowProvider, useReactFlow, useUpdateNodeInternals } from "@xyflow/react";
import type { NodeProps, Connection } from "@xyflow/react";
import { ArrowDownRight, Box, Database, Flag, Layers, ScanLine } from "lucide-react";
import type { CardNode, Catalog, Entry, Snapshot } from "../model/types";
import { connect, disconnect, placeNode, project, transferLayer, validConnection } from "../model/graph";
import { Button } from "./ui/button";
import { StageOrderStrip } from "./structure";

function Card({id, data, selected}:NodeProps<CardNode>) {
  const updateInternals=useUpdateNodeInternals();
  useEffect(()=>updateInternals(id),[id,data.inputs,data.outputs,updateInternals]);
  const Icon = data.kind === "data" ? Database : data.kind === "stage" ? Layers : data.kind === "objective" ? Flag : data.kind === "evaluation" ? ScanLine : Box;
  if(data.kind==="stage")return <div className={`node-card stage ${selected ? "selected" : ""} ${data.dropTarget ? "drop-target" : ""}`}>
    {selected && !data.blocked && <NodeResizeControl position="bottom-right" minWidth={data.minWidth} minHeight={data.minHeight}
      onResizeStart={data.resizeStart as ()=>void} onResizeEnd={(_,size)=>(data.resizeEnd as (size:{width:number;height:number})=>void)(size)} style={{background:"transparent",border:0}}><span title={`Resize stage ${data.title}`} aria-label={`Resize stage ${data.title}`}><ArrowDownRight size={20}/></span></NodeResizeControl>}
    <div className="stage-header"><div className="node-heading"><Icon size={16}/><span>Stage {data.executionPosition}</span><span>⋮⋮</span></div><h3 title={data.title}>{data.title}</h3><p>{data.summary}{data.skipped && " · Skipped: will not execute"}</p></div>
    {data.empty && <div className="empty-stage"><p>Drop a layer here</p><Button className="nodrag nopan" disabled={!!data.blocked} onClick={data.addLayer as ()=>void}>Add layer</Button></div>}
  </div>;
  return <div className={`node-card ${data.kind} ${selected ? "selected" : ""}`}>
    <div className="node-heading"><Icon size={16} /><span>{data.kind}</span>{data.executionPosition && <span className="execution-badge">{data.executionPosition}</span>}</div>
    <h3>{data.title}</h3><p>{data.summary}</p>
    <div className="node-ports">{data.inputs.map(input=><div className="port-row input-port" key={input.id}>
      <Handle type="target" position={Position.Left} id={input.id} aria-label={`${data.title} input ${input.key}`} />
      <span>{input.add ? `+ ${input.field==="loss_keys" ? "Loss" : input.field==="metric_keys" ? "Metric" : input.field==="label_key" ? "Label" : "Feature"} input` : `${input.alias ? `${input.alias} ← ` : ""}${input.key || (input.required ? "Unconnected" : `${input.field.split(".").at(-1)?.replaceAll("_"," ")} · optional`)}`}</span><small>{input.domain}{input.defaultKey===input.key && " · default"}</small>
    </div>)}{data.outputs.map(output=><div className="port-row output-port" key={output.id}>
      <span>{output.key}</span><small>{output.domain}</small>
      <Handle type="source" position={Position.Right} id={output.id} aria-label={`${data.title} output ${output.key}`} />
    </div>)}</div>
  </div>;
}
const nodeTypes = { card:Card };

function Editor({snapshot, selected, select, change, blocked,catalog,insert,report,remove,create,addLayer}: {
  snapshot:Snapshot; selected:string; select:(id:string)=>void; change:(next:Snapshot)=>void; blocked:boolean;
  catalog?:Catalog;insert:(entry:Entry,stage?:string,position?:{x:number;y:number})=>void;report:(error:unknown)=>void;remove:(id:string)=>void;
  create:(position?:{x:number;y:number})=>void;addLayer:(stage:string)=>void;
}) {
  const [edgeId,setEdgeId]=useState("");
  const [measured,setMeasured] = useState<Record<string,{width:number;height:number}>>({});
  const [preview,setPreview]=useState<Snapshot|null>(null);
  const gesture=useRef<Snapshot|null>(null);
  const cancelled=useRef(false);
  const [dropStage,setDropStage]=useState("");
  const base=useMemo(()=>project(snapshot,catalog,measured),[snapshot,catalog,measured]);
  const graph=useMemo(()=>project(preview ?? snapshot,catalog,measured),[preview,snapshot,catalog,measured]);
  const selectedEdge=graph.edges.find(edge=>edge.id===edgeId);
  const removable=!graph.nodes.find(node=>node.id===selectedEdge?.target)?.data.inputs.find(input=>input.id===selectedEdge?.targetHandle)?.defaultKey;
  const flow = useReactFlow();
  const begin=(id:string)=>{cancelled.current=false;gesture.current={...structuredClone(snapshot),selected:id};};
  const finish=()=>{const next=gesture.current;gesture.current=null;setPreview(null);setDropStage("");if(next && !cancelled.current)change(next);};
  const cancel=()=>{cancelled.current=true;gesture.current=null;setPreview(null);setDropStage("");};
  const nodes=graph.nodes.map(node=>({...node,selected:node.id===selected,draggable:!blocked,measured:measured[node.id],
    position:preview?.positions[node.id] ?? node.position,
    style:preview && !preview.selected?.startsWith("stage:") && node.data.kind==="stage" ? base.nodes.find(item=>item.id===node.id)?.style : node.style,
    data:{...node.data,blocked,dropTarget:node.data.stage===dropStage,resizeStart:()=>begin(node.id),resizeEnd:(size:{width:number;height:number})=>{
      if(gesture.current)gesture.current={...gesture.current,sizes:{...gesture.current.sizes,[node.id]:{width:size.width,height:size.height}}};finish();
    },addLayer:()=>addLayer(node.data.stage!)}}));
  const movingStage=preview && snapshot.order.find(stage=>snapshot.ids[stage].includes(preview.selected ?? ""));
  // Stage hit testing uses committed bounds, not a frame growing around a dragged child.
  const destination=(node:CardNode)=>{
    const parent=base.nodes.find(item=>item.id===node.parentId);const size=measured[node.id] ?? {width:250,height:170};
    const point={x:(parent?.position.x ?? 0)+node.position.x+size.width/2,y:(parent?.position.y ?? 0)+node.position.y+size.height/2};
    return [...base.nodes].reverse().filter(item=>item.data.kind==="stage").find(item=>point.x>=item.position.x && point.y>=item.position.y+110 && point.x<=item.position.x+Number(item.style?.width) && point.y<=item.position.y+Number(item.style?.height));
  };
  const act=(action:()=>Snapshot)=>{if(blocked)return;try{change(action());}catch(error){report(error);}};
  const removeEdge=()=>{if(selectedEdge)act(()=>disconnect(snapshot,[selectedEdge],catalog));};
  return <div className="canvas" onKeyDown={event=>{
    if(event.key==="Escape"){cancel();return;}
    if(blocked || !["Delete","Backspace"].includes(event.key) || (event.target as HTMLElement).closest("input,select,textarea,button"))return;
    event.preventDefault();event.stopPropagation();if(selectedEdge)removeEdge();else remove(selected);
  }} onDragOver={event=>{if(!blocked && (event.dataTransfer.types.includes("application/x-nexuml-component") || event.dataTransfer.types.includes("application/x-nexuml-stage"))){event.preventDefault();event.dataTransfer.dropEffect="copy";const point=flow.screenToFlowPosition({x:event.clientX,y:event.clientY});const stage=flow.getIntersectingNodes({...point,width:1,height:1}).find(node=>node.data.kind==="stage");setDropStage(stage?.data.stage as string ?? "");}}}
    onDragLeave={()=>setDropStage("")}
    onDrop={event=>{
      setDropStage("");if(blocked)return;
      if(event.dataTransfer.getData("application/x-nexuml-stage")){event.preventDefault();create(flow.screenToFlowPosition({x:event.clientX,y:event.clientY}));return;}
      const value=event.dataTransfer.getData("application/x-nexuml-component");if(!value)return;event.preventDefault();
      try{
        const [kind,name,version]=JSON.parse(value);const entry=catalog?.components.find(entry=>entry.kind===kind && entry.name===name && entry.version===version);if(!entry)return;
        const position=flow.screenToFlowPosition({x:event.clientX,y:event.clientY});
        const stage=flow.getIntersectingNodes({...position,width:1,height:1}).find(node=>node.data.kind==="stage");
        insert(entry,stage?.data.stage as string|undefined,position);
      }catch(error){report(error);}
    }}>
    <div className="canvas-toolbar"><span>ORDERED PIPELINE</span><div>
      {selectedEdge && <Button className="danger" disabled={blocked || !removable} title={removable ? "Remove this key route (Delete/Backspace)" : "Inherited runtime default: change the key or disable the evaluator"} onClick={removeEdge}>Remove connection</Button>}
      <Button onClick={()=>flow.fitView({padding:.15})}>Fit view</Button>
      <Button disabled={blocked} onClick={()=>{change({...snapshot,positions:{},sizes:{}});setTimeout(()=>flow.fitView({padding:.15}),30);}}>Arrange</Button>
    </div></div>
    <StageOrderStrip {...{snapshot,selected,blocked,select,change,report}}/>
    <ReactFlow nodes={nodes} edges={graph.edges.map(edge=>({...edge,selected:edge.id===edgeId,reconnectable:!blocked}))}
      nodeTypes={nodeTypes} defaultViewport={{x:30,y:80,zoom:.85}} minZoom={.15} maxZoom={1.5} colorMode="dark"
      nodesDraggable={!blocked} nodesConnectable={!blocked} deleteKeyCode={null}
       selectionOnDrag={false} multiSelectionKeyCode={null} elevateNodesOnSelect={false}
      isValidConnection={value=>validConnection(snapshot,value as Connection,catalog)}
      onConnect={value=>act(()=>connect(snapshot,value,catalog))}
      onReconnect={(edge,value)=>act(()=>connect(edge.target===value.target && edge.targetHandle===value.targetHandle ? snapshot : disconnect(snapshot,[edge],catalog),value,catalog))}
      onEdgeClick={(_,edge)=>setEdgeId(edge.id)}
      onEdgesChange={changes=>{for(const update of changes)if(update.type==="select" && update.selected)setEdgeId(update.id);}}
      onPaneClick={()=>setEdgeId("")}
      onNodeClick={(_,node)=>{setEdgeId("");select(node.id);}}
      onNodesChange={changes=>{let next=gesture.current;let changed=false;
        for(const update of changes)if(update.type==="select" && update.selected){setEdgeId("");select(update.id);}
        for(const update of changes) if(update.type==="dimensions" && update.dimensions){
          const dimensions=update.dimensions;setMeasured(previous=>previous[update.id]?.width===dimensions.width && previous[update.id]?.height===dimensions.height ? previous : {...previous,[update.id]:dimensions});
          if(next && update.setAttributes){next={...next,sizes:{...next.sizes,[update.id]:dimensions}};changed=true;}
        }
        for(const update of changes) if(update.type==="position" && update.position){
          if(next){next={...next,positions:{...next.positions,[update.id]:update.position}};changed=true;}
          else if(!cancelled.current && !update.dragging)change(placeNode(snapshot,update.id,update.position));
        }
        if(changed && next){gesture.current=next;setPreview(next);}}}
      onNodeDragStart={(_,node)=>{begin(node.id);gesture.current!.selected=node.id;}}
      onNodeDrag={(_,node)=>{if(node.data.kind==="layer")setDropStage(destination(node as CardNode)?.data.stage ?? "");}}
      onNodeDragStop={(_,node)=>{
        if(cancelled.current || !gesture.current){cancel();return;}
        if(node.data.kind==="layer"){
          const target=destination(node as CardNode);if(!target){cancel();report(new Error("Layer drop outside stages: original placement restored."));return;}
          const parent=base.nodes.find(item=>item.id===node.parentId)!;
          const point={x:parent.position.x+node.position.x-target.position.x,y:parent.position.y+node.position.y-target.position.y};
          if(target.data.stage!==node.data.stage)gesture.current=transferLayer(snapshot,node.id,target.data.stage!,snapshot.ids[target.data.stage!].length,point);
          else gesture.current=placeNode(gesture.current,node.id,point);
        }else gesture.current=placeNode(gesture.current,node.id,node.position);
        finish();}}>
      <Background color="#434750" gap={24} size={1} />
      <Controls showInteractive={false} />
      <MiniMap pannable zoomable nodeColor="#0a3d74" maskColor="#101010b0" />
    </ReactFlow>
    <div className="canvas-caption" aria-live="polite">{dropStage && dropStage===movingStage ? `Reposition inside ${dropStage} · execution order unchanged` : dropStage ? `Move / insert into ${dropStage}, append (last)${(snapshot.config.data.skip_pipeline_stages as string[]|undefined)?.includes(dropStage) ? " · Skipped: will not execute" : ""}` : "Drag headers to move stages; resize selected stages. Execution order is explicit above."}</div>
  </div>;
}

export function Canvas(props:React.ComponentProps<typeof Editor>) {
  return <ReactFlowProvider><Editor {...props} /></ReactFlowProvider>;
}
