import type { Connection, Edge } from "@xyflow/react";
import type { CardNode, Catalog, Component, Config, Graph, Layer, Port, RecordValue, Snapshot } from "./types";

export const STAGE_INSET=90;

const port = (key: string, domain = "x", field = "keys_in", alias?: string): Port =>
  ({ id: `${field}:${alias ?? key}`, key, domain, field, alias });

// Omitted parameter defaults live in the projection until a route is edited.
const collectionKeys=(value:unknown,inputs:Port[],field:string):string[]=>Array.isArray(value) ? value : inputs.filter(input=>input.field===field).map(input=>input.key);

export function newSnapshot(config: Config, order: string[], previous?: Snapshot): Snapshot {
  const pool = new Map<string, string[]>();
  for (const stage of previous?.order ?? []) {
    previous!.config.pipeline.stages[stage].forEach((layer, index) => {
      const key = JSON.stringify(layer);
      pool.set(key, [...(pool.get(key) ?? []), previous!.ids[stage][index]]);
    });
  }
  const ids = Object.fromEntries(order.map(stage => [stage, config.pipeline.stages[stage].map(layer =>
    pool.get(JSON.stringify(layer))?.shift() ?? crypto.randomUUID())]));
  return { config, order, ids, positions: previous?.positions ?? {}, sizes:previous?.sizes ?? {}, names:previous?.names ?? {},selected:previous?.selected ?? "data" };
}

export function signature(snapshot: Snapshot): string {
  return JSON.stringify([snapshot.order, snapshot.config],(_,value)=>value && typeof value==="object" && !Array.isArray(value) ? Object.fromEntries(Object.entries(value).sort(([a],[b])=>a.localeCompare(b))) : value);
}

export function project(snapshot: Snapshot, catalog?:Catalog, measured:Record<string,{width:number;height:number}>={}): Graph {
  const { config, order, ids, positions } = snapshot;
  const graph: Graph = { nodes: [], edges: [], problems: [] };
  const producers = new Map<string, { node: string; handle: string; rank: number }>();
  const definition=(component:Component,kind:string)=>catalog?.components.find(item=>item.kind===kind && item.name===component.type && item.version===component.version)?.io;
  const add = (node: CardNode) => {
    node.position = positions[node.id] ?? node.position;
    graph.nodes.push(node);
    for (const input of node.data.inputs) {
      if(!input.key){if(input.required){graph.unconnected=true;graph.problems.push(`${node.data.title}: unconnected required input ${input.alias ?? input.field}.`);}continue;}
      const from = producers.get(`${input.domain}:${input.key}`);
      if (from) graph.edges.push({ id: `${node.id}:${input.id}`, source: from.node, sourceHandle: from.handle,
        target: node.id, targetHandle: input.id, label: input.alias ? `${input.alias} ← ${input.key}` : input.key,
        className: input.domain === "meta" ? "metadata-edge" : undefined });
      else graph.problems.push(`${node.data.title}: ${input.domain}:${input.key} has no preceding declared producer. Use routing fields for runtime-only keys.`);
    }
    for (const output of node.data.outputs) producers.set(`${output.domain}:${output.key}`,
      { node: node.id, handle: output.id, rank: node.data.rank });
  };
  const features = Object.keys(config.data.input_shapes ?? {});
  if (!features.length) features.push(config.data.feature_key ?? "features");
  const labels = (config.data.targets ?? []).map(target => target.key);
  const sources = [config.data.source, ...(config.data.datasets ?? []).map(dataset => dataset.source)].filter(Boolean);
  for(const source of sources){
    const declared=definition(source!,"data_source")?.labels;
    if(declared==null)graph.problems.push(`${source!.type}: native source labels are unknown. Explicit scenario labels remain available; other labels resolve at runtime.`);
    else for(const key of declared)if(!labels.includes(key))labels.push(key);
  }
  add({ id: "data", type: "card", position: {x: 30, y: 150}, data: { kind: "data", title: "Data",
    summary: sources.map(source => source!.type).join(" · ") || "Configured data source",
    inputs: [], outputs: [...features.map(key => port(key, "x", "output")), ...labels.map(key => port(key, "y", "label"))], rank: -1 }});
  let rank = 0;
  order.forEach((stage, column) => {
    const layers = config.pipeline.stages[stage];
    const stageNode:CardNode={ id: `stage:${stage}`, type: "card", position: positions[`stage:${stage}`] ?? {x: 350 + column * 330, y: 40},
      data: {kind: "stage", title: stage, inputs: [], outputs: [], stage, rank: -1,
        summary: `Stage ${column + 1} · ${layers.length} layers`,executionPosition:String(column+1),empty:!layers.length,
        skipped:(config.data.skip_pipeline_stages as string[] | undefined)?.includes(stage)},dragHandle:".stage-header",zIndex:-1 };
    graph.nodes.push(stageNode);
    layers.forEach((layer, index) => {
      const inputs = Array.isArray(layer.keys_in) ? layer.keys_in.map(key => ({...port(key),required:true})) :
        Object.entries(layer.keys_in).map(([alias, key]) => ({...port(key, "x", "keys_in", alias),required:true}));
      const labels = layer.label_key == null ? [] : Array.isArray(layer.label_key) ? layer.label_key : [layer.label_key];
      inputs.push(...labels.map(key => ({...port(key, layer.label_in_x ? "x" : "y", "label_key"),required:false})));
      inputs.push(...Object.entries(layer.meta_in ?? {}).map(([alias, key]) => ({...port(key, "meta", "meta_in", alias),required:true})));
      inputs.push({...port("", "x", "keys_in", "$new"),add:true,required:false});
      if(!labels.length)inputs.push({...port("", layer.label_in_x ? "x" : "y", "label_key", "$new"),add:true,required:false});
      const outputs = [...layer.keys_out.map(key => port(key, "x", "output")),
        ...Object.entries(layer.meta_out ?? {}).map(([alias, key]) => port(key, "meta", "meta_out", alias))];
      if ((config.data.skip_pipeline_stages as string[] | undefined)?.includes(stage)) {
        graph.problems.push(`${stage}: skipped by data.skip_pipeline_stages; retained visually, not executable.`);
        graph.nodes.push({ id: ids[stage][index], type: "card", parentId:`stage:${stage}`, position: {x: 20, y: STAGE_INSET + index * 250},
          data: {kind: "layer", title: layer.component.type, summary: "Skipped stage", inputs: [], outputs: [], stage, index, executionPosition:`${column+1}.${index+1}`,rank: rank++} });
        return;
      }
      add({ id: ids[stage][index], type: "card", parentId:`stage:${stage}`, position: {x: 20, y: STAGE_INSET + index * 250},
        data: {kind: "layer", title: layer.component.type, summary: `v${layer.component.version} · ${Object.entries(layer.component.params).slice(0,2).map(([key,value]) => `${key}: ${JSON.stringify(value)}`).join(" · ")}`,
          inputs, outputs, stage, index, executionPosition:`${column+1}.${index+1}`,rank: rank++} });
    });
    const children=graph.nodes.filter(node=>node.parentId===stageNode.id);
    let width=290,height=layers.length ? 120 : 190;
    for(const child of children){
      const point=positions[child.id] ?? child.position;
      child.position={x:Math.max(20,point.x),y:Math.max(STAGE_INSET,point.y)};
      const size=measured[child.id] ?? {width:250,height:120+(child.data.inputs.length+child.data.outputs.length)*34};
      width=Math.max(width,child.position.x+size.width+20);height=Math.max(height,child.position.y+size.height+20);
    }
    stageNode.data.minWidth=width;stageNode.data.minHeight=height;
    stageNode.style={width:Math.max(width,snapshot.sizes?.[stageNode.id]?.width ?? 290),height:Math.max(height,snapshot.sizes?.[stageNode.id]?.height ?? (120+layers.length*250))};
  });
  const right = 350 + order.length * 330;
  add({ id: "objectives", type: "card", position: {x: right, y: 80}, data: {
    kind: "objective", title: "Objectives & metrics", summary: "training.loss_keys · metric_keys",
    inputs: [...Object.keys(config.training.loss_keys ?? {}).map(key => port(key, "x", "loss_keys")),
      ...(config.training.metric_keys ?? []).map(key => port(key, "x", "metric_keys")),
      {...port("", "x", "loss_keys", "$new"),add:true}, {...port("", "x", "metric_keys", "$new"),add:true}], outputs: [], rank: rank++ } });
  (config.evaluation.algorithms ?? []).forEach((evaluation, index) => {
    const inputs = [];
    const io=definition(evaluation.algorithm,"eval_algorithm");
    if(!io?.inputs_known)graph.problems.push(`${evaluation.algorithm.type}: native evaluator context inputs are unknown. Edit explicit scenario routes manually; no context connections are inferred.`);
    for(const binding of io?.inputs ?? []){
      const context=binding.source==="context";
      const field=context ? binding.field : `algorithm.params.${binding.field}`;
      const configured=context ? evaluation[binding.field] : evaluation.algorithm.params[binding.field];
      const value=context ? configured || binding.default : configured===undefined ? binding.default : configured;
      const keys=Array.isArray(value) && value.length ? value : [typeof value==="string" ? value : ""];
      for(const key of keys)inputs.push({...port(key,binding.domain,field),collection:binding.collection,
        defaultKey:context && typeof binding.default==="string" ? binding.default : undefined,required:binding.required});
    }
    for (const axis of evaluation.axis_keys ?? []) {
      const value = typeof axis === "string" ? { key: axis, source: "y" } : axis;
      if (value.source === "metadata") graph.problems.push(`${evaluation.algorithm.type}: metadata axis ${value.key} resolves from dataset columns at runtime.`);
      else inputs.push(port(value.key, value.source, "axis_keys"));
    }
    add({ id: `evaluation:${index}`, type: "card", position: {x: right, y: 420 + index * 260}, data: {
      kind: "evaluation", title: evaluation.name as string || evaluation.algorithm.type, summary: evaluation.enabled===false ? "Disabled evaluation" : "Post-training evaluation", inputs:evaluation.enabled===false ? [] : inputs, outputs: [], index, rank: rank++ } });
  });
  for(const node of graph.nodes)node.data.title=snapshot.names?.[node.id] || node.data.title;
  return graph;
}

export function validConnection(graph: Graph, connection: Connection): boolean {
  const source = graph.nodes.find(node => node.id === connection.source);
  const target = graph.nodes.find(node => node.id === connection.target);
  const output = source?.data.outputs.find(port => port.id === connection.sourceHandle);
  const input = target?.data.inputs.find(port => port.id === connection.targetHandle);
  if (!source || !target || !output?.key || !input || source.data.rank >= target.data.rank || output.domain !== input.domain) return false;
  // A repeated key routes to the most recent preceding writer, never an arbitrary older node.
  const preceding = graph.nodes.filter(node => node.data.rank < target.data.rank &&
    node.data.outputs.some(port => port.key === output.key && port.domain === output.domain));
  return preceding.at(-1)?.id === source.id;
}

export function connect(snapshot: Snapshot, connection: Connection, catalog?:Catalog): Snapshot {
  const graph = project(snapshot,catalog);
  if (!validConnection(graph, connection)) throw new Error("Choose the applicable preceding producer with a matching key domain.");
  const source = graph.nodes.find(node => node.id === connection.source)!;
  const target = graph.nodes.find(node => node.id === connection.target)!;
  const output = source.data.outputs.find(port => port.id === connection.sourceHandle)!;
  const input = target.data.inputs.find(port => port.id === connection.targetHandle)!;
  const next = structuredClone(snapshot);
  if (target.data.kind === "layer") {
    const layer: Layer = next.config.pipeline.stages[target.data.stage!][target.data.index!];
    if (input.field === "keys_in") {
      if (Array.isArray(layer.keys_in)) {
        if(input.add)layer.keys_in.push(output.key);
        else layer.keys_in[layer.keys_in.indexOf(input.key)] = output.key;
      }else layer.keys_in[input.add ? output.key : input.alias!] = output.key;
    } else if (input.field === "label_key") {
      if (Array.isArray(layer.label_key)) {if(input.add)layer.label_key.push(output.key);else layer.label_key[layer.label_key.indexOf(input.key)] = output.key;}
      else layer.label_key = output.key;
    } else layer.meta_in![input.alias!] = output.key;
  } else if (target.data.kind === "objective") {
    if (input.field === "loss_keys") {
      const weight = input.add ? 1 : next.config.training.loss_keys[input.key];
      delete next.config.training.loss_keys[input.key];
      next.config.training.loss_keys[output.key] = weight;
    } else next.config.training.metric_keys = input.add ? [...next.config.training.metric_keys,output.key] : next.config.training.metric_keys.map(key => key === input.key ? output.key : key);
  } else if (target.data.kind === "evaluation") {
    const evaluation = next.config.evaluation.algorithms[target.data.index!];
    if (input.field === "axis_keys") evaluation.axis_keys = evaluation.axis_keys?.map(axis =>
      typeof axis === "string" ? (axis === input.key ? output.key : axis) : axis.key === input.key && axis.source === output.domain ? {...axis, key:output.key} : axis);
    else{
      const parts=input.field.split(".");let value=evaluation as RecordValue;
      for(const part of parts.slice(0,-1))value=value[part] as RecordValue;
      const key=parts.at(-1)!;
      if(input.collection){
        const keys=collectionKeys(value[key],target.data.inputs,input.field);
        value[key]=keys.includes(input.key) ? keys.map(item=>item===input.key ? output.key : item) : [...keys,output.key];
      }else value[key]=output.key;
    }
  }
  return next;
}

export function disconnect(snapshot:Snapshot,edges:Pick<Edge,"target"|"targetHandle">[],catalog?:Catalog):Snapshot {
  const graph=project(snapshot,catalog);const next=structuredClone(snapshot);
  for(const edge of edges){
    const node=graph.nodes.find(node=>node.id===edge.target);
    const input=node?.data.inputs.find(input=>input.id===edge.targetHandle);
    if(!node || !input)continue;
    if(input.defaultKey)throw new Error("This evaluator inherits a runtime default. Change its key or explicitly disable the evaluator instead of hiding its connection.");
    if(node.data.kind==="layer"){
      const layer=next.config.pipeline.stages[node.data.stage!][node.data.index!];
      if(input.field==="keys_in"){
        if(Array.isArray(layer.keys_in))layer.keys_in=layer.keys_in.map(key=>key===input.key ? "" : key);
        else layer.keys_in[input.alias!]="";
      }else if(input.field==="label_key")layer.label_key=Array.isArray(layer.label_key) ? layer.label_key.filter(key=>key!==input.key) : null;
      else layer.meta_in![input.alias!]="";
    }else if(node.data.kind==="objective"){
      if(input.field==="loss_keys")delete next.config.training.loss_keys[input.key];
      else next.config.training.metric_keys=next.config.training.metric_keys.filter(key=>key!==input.key);
    }else if(node.data.kind==="evaluation"){
      const evaluation=next.config.evaluation.algorithms[node.data.index!];
      if(input.field==="axis_keys")evaluation.axis_keys=evaluation.axis_keys?.filter(axis=>!(typeof axis==="string" ? axis===input.key : axis.key===input.key && axis.source===input.domain));
      else{
        const parts=input.field.split(".");let value=evaluation as RecordValue;
        for(const part of parts.slice(0,-1))value=value[part] as RecordValue;
        const key=parts.at(-1)!;value[key]=input.collection ? collectionKeys(value[key],node.data.inputs,input.field).filter(item=>item!==input.key) : "";
      }
    }
  }
  return next;
}

export function moveLayer(snapshot: Snapshot, stage: string, index: number, delta: number): Snapshot {
  const next = structuredClone(snapshot);
  const target = index + delta;
  if (target < 0 || target >= next.ids[stage].length) return next;
  for (const values of [next.ids[stage], next.config.pipeline.stages[stage]]) {
    [values[index], values[target]] = [values[target], values[index]];
  }
  return next;
}

// Explicit list slots, never spatial sorting. Keep identical layers' IDs intact.
export function transferLayer(snapshot:Snapshot,id:string,stage:string,slot:number,position?:{x:number;y:number}):Snapshot {
  const source=snapshot.order.find(name=>snapshot.ids[name].includes(id));
  if(!source || !snapshot.order.includes(stage) || !Number.isInteger(slot) || slot<0 || slot>snapshot.ids[stage].length)throw new Error("Choose an existing stage and insertion slot.");
  const index=snapshot.ids[source].indexOf(id);const target=slot-(source===stage && index<slot ? 1 : 0);
  if(source===stage && target===index && !position)return snapshot;
  const next=structuredClone(snapshot);
  for(const node of project(snapshot).nodes)next.positions[node.id] ??= node.position;
  const [layer]=next.config.pipeline.stages[source].splice(index,1);next.ids[source].splice(index,1);
  next.config.pipeline.stages[stage].splice(target,0,layer);next.ids[stage].splice(target,0,id);
  if(position)next.positions[id]={x:Math.max(20,position.x),y:Math.max(STAGE_INSET,position.y)};
  else if(source!==stage){
    const graph=project(snapshot);const parent=graph.nodes.find(node=>node.id===`stage:${stage}`)!;
    next.positions[id]={x:20,y:Math.max(STAGE_INSET,parent.data.minHeight!-20)};
  }
  next.selected=id;return next;
}

export function reorderStage(snapshot:Snapshot,stage:string,slot:number):Snapshot {
  const index=snapshot.order.indexOf(stage);
  if(index<0 || !Number.isInteger(slot) || slot<0 || slot>snapshot.order.length)throw new Error("Choose an existing stage and insertion slot.");
  if(slot-(index<slot ? 1 : 0)===index)return snapshot;
  const next=structuredClone(snapshot);for(const node of project(snapshot).nodes)next.positions[node.id] ??= node.position;
  next.order.splice(index,1);next.order.splice(slot-(index<slot ? 1 : 0),0,stage);return next;
}

export function addStage(snapshot:Snapshot,name:string,after?:string,position?:{x:number;y:number}):Snapshot {
  name=name.trim();if(!name || snapshot.order.includes(name))throw new Error("Use a unique, non-empty stage name.");
  const next=structuredClone(snapshot);const index=after ? next.order.indexOf(after) : -1;
  const nodes=project(snapshot).nodes;
  for(const node of nodes)next.positions[node.id] ??= node.position;
  next.order.splice(index<0 ? next.order.length : index+1,0,name);
  next.config.pipeline.stages={...next.config.pipeline.stages,[name]:[]};next.ids={...next.ids,[name]:[]};
  next.positions[`stage:${name}`]=position ?? {x:Math.max(...nodes.map(node=>node.position.x+Number(node.style?.width ?? 250)))+40,y:40};
  next.selected=`stage:${name}`;return next;
}

export function placeNode(snapshot:Snapshot,id:string,position:{x:number;y:number}):Snapshot {
  const next=structuredClone(snapshot);const layer=snapshot.order.some(stage=>snapshot.ids[stage].includes(id));
  next.positions[id]=layer ? {x:Math.max(20,position.x),y:Math.max(STAGE_INSET,position.y)} : position;return next;
}
