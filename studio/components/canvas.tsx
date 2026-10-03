"use client";
import { useEffect, useMemo, useState } from "react";
import { Background, Controls, Handle, MiniMap, Position, ReactFlow, ReactFlowProvider, useReactFlow, useUpdateNodeInternals } from "@xyflow/react";
import type { NodeProps, Connection } from "@xyflow/react";
import { Box, Database, Flag, Layers, ScanLine } from "lucide-react";
import type { CardNode, Catalog, Entry, Snapshot } from "../model/types";
import { connect, disconnect, project, validConnection } from "../model/graph";
import { Button } from "./ui/button";

function Card({id, data, selected}:NodeProps<CardNode>) {
  const updateInternals=useUpdateNodeInternals();
  useEffect(()=>updateInternals(id),[id,data.inputs,data.outputs,updateInternals]);
  const Icon = data.kind === "data" ? Database : data.kind === "stage" ? Layers : data.kind === "objective" ? Flag : data.kind === "evaluation" ? ScanLine : Box;
  return <div className={`node-card ${data.kind} ${selected ? "selected" : ""}`}>
    <div className="node-heading"><Icon size={16} /><span>{data.kind}</span></div>
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

function Editor({snapshot, selected, select, change, blocked,catalog,insert,report,remove}: {
  snapshot:Snapshot; selected:string; select:(id:string)=>void; change:(next:Snapshot)=>void; blocked:boolean;
  catalog?:Catalog;insert:(entry:Entry,stage?:string,position?:{x:number;y:number})=>void;report:(error:unknown)=>void;remove:(id:string)=>void;
}) {
  const graph = useMemo(()=>project(snapshot,catalog),[snapshot,catalog]);
  const [edgeId,setEdgeId]=useState("");
  const selectedEdge=graph.edges.find(edge=>edge.id===edgeId);
  const removable=!graph.nodes.find(node=>node.id===selectedEdge?.target)?.data.inputs.find(input=>input.id===selectedEdge?.targetHandle)?.defaultKey;
  const [measured,setMeasured] = useState<Record<string,{width:number;height:number}>>({});
  const nodes=useMemo(()=>graph.nodes.map(node=>({...node,measured:measured[node.id],selected:node.id===selected})),[graph,measured,selected]);
  const flow = useReactFlow();
  const act=(action:()=>Snapshot)=>{if(blocked)return;try{change(action());}catch(error){report(error);}};
  const removeEdge=()=>{if(selectedEdge)act(()=>disconnect(snapshot,[selectedEdge],catalog));};
  return <div className="canvas" onKeyDown={event=>{
    if(blocked || !["Delete","Backspace"].includes(event.key) || (event.target as HTMLElement).closest("input,select,textarea,button"))return;
    event.preventDefault();event.stopPropagation();if(selectedEdge)removeEdge();else remove(selected);
  }} onDragOver={event=>{if(!blocked && event.dataTransfer.types.includes("application/x-nexuml-component")){event.preventDefault();event.dataTransfer.dropEffect="copy";}}}
    onDrop={event=>{
      if(blocked)return;const value=event.dataTransfer.getData("application/x-nexuml-component");if(!value)return;event.preventDefault();
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
      <Button disabled={blocked} onClick={()=>{change({...snapshot,positions:{}});setTimeout(()=>flow.fitView({padding:.15}),30);}}>Arrange</Button>
    </div></div>
    <ReactFlow nodes={nodes} edges={graph.edges.map(edge=>({...edge,selected:edge.id===edgeId,reconnectable:!blocked}))}
      nodeTypes={nodeTypes} defaultViewport={{x:30,y:80,zoom:.85}} minZoom={.15} maxZoom={1.5} colorMode="dark"
      nodesDraggable={!blocked} nodesConnectable={!blocked} deleteKeyCode={null}
      isValidConnection={value=>validConnection(snapshot,value as Connection,catalog)}
      onConnect={value=>act(()=>connect(snapshot,value,catalog))}
      onReconnect={(edge,value)=>act(()=>connect(edge.target===value.target && edge.targetHandle===value.targetHandle ? snapshot : disconnect(snapshot,[edge],catalog),value,catalog))}
      onEdgeClick={(_,edge)=>setEdgeId(edge.id)}
      onEdgesChange={changes=>{for(const update of changes)if(update.type==="select" && update.selected)setEdgeId(update.id);}}
      onPaneClick={()=>setEdgeId("")}
      onNodeClick={(_,node)=>{setEdgeId("");select(node.id);}}
      onNodesChange={changes=>{const positions = {...snapshot.positions};let changed=false;
        for(const update of changes)if(update.type==="select" && update.selected){setEdgeId("");select(update.id);}
        for(const update of changes) if(update.type==="dimensions" && update.dimensions){
          const dimensions=update.dimensions;setMeasured(previous=>({...previous,[update.id]:dimensions}));
        }
        for(const update of changes) if(update.type==="position" && update.position){positions[update.id]=update.position;changed=true;}
        if(changed) change({...snapshot,positions});}}
      onNodeDragStop={(_,node)=>change({...snapshot,positions:{...snapshot.positions,[node.id]:node.position}})}>
      <Background color="#434750" gap={24} size={1} />
      <Controls showInteractive={false} />
      <MiniMap pannable zoomable nodeColor="#0a3d74" maskColor="#101010b0" />
    </ReactFlow>
    <div className="canvas-caption">Drop components into stages. Select a connection to remove it. Placement is visual; execution follows the outline.</div>
  </div>;
}

export function Canvas(props:React.ComponentProps<typeof Editor>) {
  return <ReactFlowProvider><Editor {...props} /></ReactFlowProvider>;
}
