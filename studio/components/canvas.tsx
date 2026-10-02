"use client";
import { useEffect, useMemo, useState } from "react";
import { Background, Controls, Handle, MiniMap, Position, ReactFlow, ReactFlowProvider, useReactFlow, useUpdateNodeInternals } from "@xyflow/react";
import type { NodeProps, Connection } from "@xyflow/react";
import { Box, Database, Flag, Layers, ScanLine } from "lucide-react";
import type { CardNode, Snapshot } from "../model/types";
import { connect, project, validConnection } from "../model/graph";
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
      <span>{input.alias ? `${input.alias} ← ` : ""}{input.key}</span><small>{input.domain}</small>
    </div>)}{data.outputs.map(output=><div className="port-row output-port" key={output.id}>
      <span>{output.key}</span><small>{output.domain}</small>
      <Handle type="source" position={Position.Right} id={output.id} aria-label={`${data.title} output ${output.key}`} />
    </div>)}</div>
  </div>;
}
const nodeTypes = { card:Card };

function Editor({snapshot, selected, select, change, blocked}: {
  snapshot:Snapshot; selected:string; select:(id:string)=>void; change:(next:Snapshot)=>void; blocked:boolean;
}) {
  const graph = useMemo(()=>project(snapshot),[snapshot]);
  const [measured,setMeasured] = useState<Record<string,{width:number;height:number}>>({});
  const nodes=useMemo(()=>graph.nodes.map(node=>({...node,measured:measured[node.id],selected:node.id===selected})),[graph,measured,selected]);
  const flow = useReactFlow();
  return <div className="canvas">
    <div className="canvas-toolbar"><span>ORDERED PIPELINE</span><div>
      <Button onClick={()=>flow.fitView({padding:.15})}>Fit view</Button>
      <Button disabled={blocked} onClick={()=>{change({...snapshot,positions:{}});setTimeout(()=>flow.fitView({padding:.15}),30);}}>Arrange</Button>
    </div></div>
    <ReactFlow nodes={nodes} edges={graph.edges}
      nodeTypes={nodeTypes} defaultViewport={{x:30,y:80,zoom:.85}} minZoom={.15} maxZoom={1.5} colorMode="dark"
      nodesDraggable={!blocked} nodesConnectable={!blocked} deleteKeyCode={null}
      isValidConnection={value=>validConnection(snapshot,value as Connection)}
      onConnect={value=>change(connect(snapshot,value))}
      onNodeClick={(_,node)=>select(node.id)}
      onNodesChange={changes=>{const positions = {...snapshot.positions};let changed=false;
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
    <div className="canvas-caption">Placement is visual. Execution follows the ordered outline.</div>
  </div>;
}

export function Canvas(props:React.ComponentProps<typeof Editor>) {
  return <ReactFlowProvider><Editor {...props} /></ReactFlowProvider>;
}
