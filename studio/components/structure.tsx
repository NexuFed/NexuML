"use client";
import { useState } from "react";
import { ArrowDown, ArrowUp, GripVertical, Layers } from "lucide-react";
import { reorderStage, transferLayer } from "../model/graph";
import type { Snapshot } from "../model/types";
import { Button } from "./ui/button";

const mime="application/x-nexuml-order";
function OrderHandle({kind,id,blocked}:{kind:"stage"|"layer";id:string;blocked:boolean}) {
  return <Button className="order-handle" aria-label={`Drag ${kind} ${id} to reorder`} title="Drag to an insertion marker; move buttons are the non-drag alternative" disabled={blocked} draggable={!blocked}
    onDragStart={event=>{event.dataTransfer.setData(mime,JSON.stringify({kind,id}));event.dataTransfer.effectAllowed="move";}}><GripVertical size={14}/></Button>;
}

function OrderSlot({snapshot,kind,stage,slot,blocked,change,report}:{snapshot:Snapshot;kind:"stage"|"layer";stage?:string;slot:number;blocked:boolean;change:(next:Snapshot)=>void;report:(error:unknown)=>void}) {
  const [over,setOver]=useState(false);
  const skipped=stage && (snapshot.config.data.skip_pipeline_stages as string[]|undefined)?.includes(stage);
  const label=kind==="stage" ? slot===snapshot.order.length ? "Move stage last" : `Move stage before ${snapshot.order[slot]}` : `Move layer to ${stage}, ${slot===snapshot.ids[stage!].length ? "append" : `before layer ${slot+1}`}${skipped ? " · Skipped: will not execute" : ""}`;
  return <div className={`order-slot ${over ? "over" : ""}`} data-order-kind={kind} data-order-stage={stage} data-order-slot={slot} aria-label={label}
    onDragOver={event=>{if(!blocked && event.dataTransfer.types.includes(mime)){event.preventDefault();event.stopPropagation();event.dataTransfer.dropEffect="move";setOver(true);}}}
    onDragLeave={()=>setOver(false)} onDrop={event=>{
      setOver(false);if(blocked)return;const value=event.dataTransfer.getData(mime);if(!value)return;event.preventDefault();event.stopPropagation();
      try{const item=JSON.parse(value);if(item.kind!==kind){report(new Error("Use a matching stage or layer insertion marker."));return;}
        change(kind==="stage" ? reorderStage(snapshot,item.id,slot) : transferLayer(snapshot,item.id,stage!,slot));
      }catch(error){report(error);}
    }}><span>{label}</span></div>;
}

type Props={snapshot:Snapshot;selected:string;blocked:boolean;select:(id:string)=>void;change:(next:Snapshot)=>void;report:(error:unknown)=>void};
export function StageOrderStrip({snapshot,selected,blocked,select,change,report}:Props) {
  return <div className="stage-order-strip" aria-label="Stage execution order"><strong>Execution order</strong><small>Drag handles to reorder</small><div>
    {snapshot.order.map((stage,index)=><div className="stage-order-item" key={stage}>
      <OrderSlot {...{snapshot,blocked,change,report}} kind="stage" slot={index}/>
      <OrderHandle kind="stage" id={stage} blocked={blocked}/><Button aria-pressed={selected===`stage:${stage}`} onClick={()=>select(`stage:${stage}`)}>{index+1}. {stage}</Button>
    </div>)}<OrderSlot {...{snapshot,blocked,change,report}} kind="stage" slot={snapshot.order.length}/>
  </div></div>;
}

export function Structure({snapshot,selected,blocked,select,change,report,create}:{create:()=>void}&Props) {
  return <><h2>Structure</h2><Button draggable={!blocked} disabled={blocked} aria-label="Stage item" title="Drag onto the canvas or click to add a stage"
    onDragStart={event=>{event.dataTransfer.setData("application/x-nexuml-stage","new");event.dataTransfer.effectAllowed="copy";}} onClick={create}><Layers size={16}/>Stage <span>+</span></Button>
    <div className="outline" aria-label="Ordered Structure"><Button aria-pressed={selected==="data"} onClick={()=>select("data")}>Data configuration</Button>
      {snapshot.order.map((stage,stageIndex)=><div className="outline-stage" key={stage}>
        <OrderSlot {...{snapshot,blocked,change,report}} kind="stage" slot={stageIndex}/>
        <div className="outline-row"><Button aria-pressed={selected===`stage:${stage}`} onClick={()=>select(`stage:${stage}`)}>{stageIndex+1}. {stage}</Button><OrderHandle kind="stage" id={stage} blocked={blocked}/>
          <Button aria-label={`Move stage ${stage} up`} disabled={blocked || stageIndex===0} onClick={()=>change(reorderStage(snapshot,stage,stageIndex-1))}><ArrowUp size={12}/></Button>
          <Button aria-label={`Move stage ${stage} down`} disabled={blocked || stageIndex===snapshot.order.length-1} onClick={()=>change(reorderStage(snapshot,stage,stageIndex+2))}><ArrowDown size={12}/></Button></div>
        {snapshot.config.pipeline.stages[stage].map((layer,index)=><div key={snapshot.ids[stage][index]}>
          <OrderSlot {...{snapshot,blocked,change,report,stage}} kind="layer" slot={index}/>
          <div className="outline-row"><Button aria-pressed={selected===snapshot.ids[stage][index]} onClick={()=>select(snapshot.ids[stage][index])}>{index+1}. {layer.component.type}</Button><OrderHandle kind="layer" id={snapshot.ids[stage][index]} blocked={blocked}/>
            <Button aria-label={`Move ${layer.component.type} up`} disabled={blocked || index===0} onClick={()=>change(transferLayer(snapshot,snapshot.ids[stage][index],stage,index-1))}><ArrowUp size={12}/></Button>
            <Button aria-label={`Move ${layer.component.type} down`} disabled={blocked || index===snapshot.ids[stage].length-1} onClick={()=>change(transferLayer(snapshot,snapshot.ids[stage][index],stage,index+2))}><ArrowDown size={12}/></Button></div>
        </div>)}<OrderSlot {...{snapshot,blocked,change,report,stage}} kind="layer" slot={snapshot.ids[stage].length}/>
      </div>)}<OrderSlot {...{snapshot,blocked,change,report}} kind="stage" slot={snapshot.order.length}/>
      <Button onClick={()=>select("objectives")}>Objectives & metrics</Button>{snapshot.config.evaluation.algorithms.map((item,index)=><Button key={index} onClick={()=>select(`evaluation:${index}`)}>{item.algorithm.type}</Button>)}
    </div><Button disabled={blocked} onClick={create}>Add stage</Button></>;
}
