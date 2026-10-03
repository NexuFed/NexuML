import type { Connection, Edge } from "@xyflow/react";
import type { CardNode, Catalog, Component, Config, Graph, Layer, Port, RecordValue, Schema, Snapshot } from "./types";
import { resolveSchema } from "./schema";

const port = (key: string, domain = "x", field = "keys_in", alias?: string): Port =>
  ({ id: `${field}:${alias ?? key}`, key, domain, field, alias });

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
  return { config, order, ids, positions: previous?.positions ?? {}, names:previous?.names ?? {} };
}

export function signature(snapshot: Snapshot): string {
  return JSON.stringify([snapshot.order, snapshot.config]);
}

export function project(snapshot: Snapshot, catalog?:Catalog): Graph {
  const { config, order, ids, positions } = snapshot;
  const graph: Graph = { nodes: [], edges: [], problems: [] };
  const producers = new Map<string, { node: string; handle: string; rank: number }>();
  const definition=(component:Component,kind:string)=>catalog?.components.find(item=>item.kind===kind && item.name===component.type && item.version===component.version)?.schema;
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
  const declared=sources.flatMap(source=>definition(source!,"data_source")?.["x-nexuml-outputs"] ?? []);
  for(const output of declared)if(output.domain==="y" && !labels.includes(output.key))labels.push(output.key);
  add({ id: "data", type: "card", position: {x: 30, y: 150}, data: { kind: "data", title: "Data",
    summary: sources.map(source => source!.type).join(" · ") || "Configured data source",
    inputs: [], outputs: [...features.map(key => port(key, "x", "output")), ...labels.map(key => port(key, "y", "label"))], rank: -1 }});
  let rank = 0;
  order.forEach((stage, column) => {
    const layers = config.pipeline.stages[stage];
    graph.nodes.push({ id: `stage:${stage}`, type: "card", position: {x: 350 + column * 330, y: 40},
      data: {kind: "stage", title: stage, inputs: [], outputs: [], stage, rank: -1,
        summary: `Stage ${column + 1} · ${layers.length} layers`}, draggable: false,
      style:{width:290,height:Math.max(130,160 + layers.length * 290)},zIndex:-1 });
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
        graph.nodes.push({ id: ids[stage][index], type: "card", parentId:`stage:${stage}`, position: {x: 20, y: 130 + index * 290},
          data: {kind: "layer", title: layer.component.type, summary: "Skipped stage", inputs: [], outputs: [], stage, index, rank: rank++} });
        return;
      }
      add({ id: ids[stage][index], type: "card", parentId:`stage:${stage}`, position: {x: 20, y: 130 + index * 290},
        data: {kind: "layer", title: layer.component.type, summary: `v${layer.component.version} · ${Object.entries(layer.component.params).slice(0,2).map(([key,value]) => `${key}: ${JSON.stringify(value)}`).join(" · ")}`,
          inputs, outputs, stage, index, rank: rank++} });
    });
  });
  const right = 350 + order.length * 330;
  add({ id: "objectives", type: "card", position: {x: right, y: 80}, data: {
    kind: "objective", title: "Objectives & metrics", summary: "training.loss_keys · metric_keys",
    inputs: [...Object.keys(config.training.loss_keys ?? {}).map(key => port(key, "x", "loss_keys")),
      ...(config.training.metric_keys ?? []).map(key => port(key, "x", "metric_keys")),
      {...port("", "x", "loss_keys", "$new"),add:true}, {...port("", "x", "metric_keys", "$new"),add:true}], outputs: [], rank: rank++ } });
  (config.evaluation.algorithms ?? []).forEach((evaluation, index) => {
    const inputs = [];
    const schema=definition(evaluation.algorithm,"eval_algorithm");
    const routing:NonNullable<Schema["x-nexuml-routing"]>=schema?.["x-nexuml-routing"] ?? {feature_key:{domain:"x"},label_key:{domain:"y"}};
    for(const [field,contract] of Object.entries(routing)){
      const configured=field.split(".").reduce<unknown>((value,key)=>(value as RecordValue|undefined)?.[key],evaluation);
      const property=field.startsWith("algorithm.params.") ? schema?.properties?.[field.slice("algorithm.params.".length)] : undefined;
      const resolved=resolveSchema(property,schema);
      const collection=resolved.type==="array" || (resolved.anyOf ?? resolved.oneOf)?.some(item=>resolveSchema(item,schema).type==="array");
      const value=configured ?? property?.default ?? contract.default;
      const keys=Array.isArray(value) && value.length ? value : [typeof value==="string" && value ? value : contract.default ?? ""];
      for(const key of keys)inputs.push({...port(key,contract.domain,field),collection,defaultKey:contract.default,required:contract.required});
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

export function validConnection(snapshot: Snapshot, connection: Connection, catalog?:Catalog): boolean {
  const graph = project(snapshot,catalog);
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
  if (!validConnection(snapshot, connection,catalog)) throw new Error("Choose the applicable preceding producer with a matching key domain.");
  const graph = project(snapshot,catalog);
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
      value[key]=input.collection ? input.key ? ((value[key] ?? []) as string[]).map(item=>item===input.key ? output.key : item) : [...(value[key] as string[] ?? []),output.key] : output.key;
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
        const key=parts.at(-1)!;value[key]=Array.isArray(value[key]) ? (value[key] as string[]).filter(item=>item!==input.key) : "";
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
