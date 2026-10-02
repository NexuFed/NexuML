import type { Connection } from "@xyflow/react";
import type { CardNode, Config, Graph, Layer, Port, Snapshot } from "./types";

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
  return { config, order, ids, positions: previous?.positions ?? {} };
}

export function signature(snapshot: Snapshot): string {
  return JSON.stringify([snapshot.order, snapshot.config]);
}

export function project(snapshot: Snapshot): Graph {
  const { config, order, ids, positions } = snapshot;
  const graph: Graph = { nodes: [], edges: [], problems: [] };
  const producers = new Map<string, { node: string; handle: string; rank: number }>();
  const add = (node: CardNode) => {
    node.position = positions[node.id] ?? node.position;
    graph.nodes.push(node);
    for (const input of node.data.inputs) {
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
      const inputs = Array.isArray(layer.keys_in) ? layer.keys_in.map(key => port(key)) :
        Object.entries(layer.keys_in).map(([alias, key]) => port(key, "x", "keys_in", alias));
      const labels = layer.label_key == null ? [] : Array.isArray(layer.label_key) ? layer.label_key : [layer.label_key];
      inputs.push(...labels.map(key => port(key, layer.label_in_x ? "x" : "y", "label_key")));
      inputs.push(...Object.entries(layer.meta_in ?? {}).map(([alias, key]) => port(key, "meta", "meta_in", alias)));
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
      ...(config.training.metric_keys ?? []).map(key => port(key, "x", "metric_keys"))], outputs: [], rank: rank++ } });
  (config.evaluation.algorithms ?? []).forEach((evaluation, index) => {
    const inputs = [];
    if (evaluation.feature_key) inputs.push(port(evaluation.feature_key, "x", "feature_key"));
    if (evaluation.label_key) inputs.push(port(evaluation.label_key, "y", "label_key"));
    for (const axis of evaluation.axis_keys ?? []) {
      const value = typeof axis === "string" ? { key: axis, source: "y" } : axis;
      if (value.source === "metadata") graph.problems.push(`${evaluation.algorithm.type}: metadata axis ${value.key} resolves from dataset columns at runtime.`);
      else inputs.push(port(value.key, value.source, "axis_keys"));
    }
    add({ id: `evaluation:${index}`, type: "card", position: {x: right, y: 420 + index * 260}, data: {
      kind: "evaluation", title: evaluation.algorithm.type, summary: "Post-training evaluation", inputs, outputs: [], index, rank: rank++ } });
  });
  return graph;
}

export function validConnection(snapshot: Snapshot, connection: Connection): boolean {
  const graph = project(snapshot);
  const source = graph.nodes.find(node => node.id === connection.source);
  const target = graph.nodes.find(node => node.id === connection.target);
  const output = source?.data.outputs.find(port => port.id === connection.sourceHandle);
  const input = target?.data.inputs.find(port => port.id === connection.targetHandle);
  if (!source || !target || !output || !input || source.data.rank >= target.data.rank || output.domain !== input.domain) return false;
  // A repeated key routes to the most recent preceding writer, never an arbitrary older node.
  const preceding = graph.nodes.filter(node => node.data.rank < target.data.rank &&
    node.data.outputs.some(port => port.key === output.key && port.domain === output.domain));
  return preceding.at(-1)?.id === source.id;
}

export function connect(snapshot: Snapshot, connection: Connection): Snapshot {
  if (!validConnection(snapshot, connection)) throw new Error("Choose the applicable preceding producer with a matching key domain.");
  const graph = project(snapshot);
  const source = graph.nodes.find(node => node.id === connection.source)!;
  const target = graph.nodes.find(node => node.id === connection.target)!;
  const output = source.data.outputs.find(port => port.id === connection.sourceHandle)!;
  const input = target.data.inputs.find(port => port.id === connection.targetHandle)!;
  const next = structuredClone(snapshot);
  if (target.data.kind === "layer") {
    const layer: Layer = next.config.pipeline.stages[target.data.stage!][target.data.index!];
    if (input.field === "keys_in") {
      if (Array.isArray(layer.keys_in)) layer.keys_in[layer.keys_in.indexOf(input.key)] = output.key;
      else layer.keys_in[input.alias!] = output.key;
    } else if (input.field === "label_key") {
      if (Array.isArray(layer.label_key)) layer.label_key[layer.label_key.indexOf(input.key)] = output.key;
      else layer.label_key = output.key;
    } else layer.meta_in![input.alias!] = output.key;
  } else if (target.data.kind === "objective") {
    if (input.field === "loss_keys") {
      const weight = next.config.training.loss_keys[input.key];
      delete next.config.training.loss_keys[input.key];
      next.config.training.loss_keys[output.key] = weight;
    } else next.config.training.metric_keys = next.config.training.metric_keys.map(key => key === input.key ? output.key : key);
  } else if (target.data.kind === "evaluation") {
    const evaluation = next.config.evaluation.algorithms[target.data.index!];
    if (input.field === "axis_keys") evaluation.axis_keys = evaluation.axis_keys?.map(axis =>
      typeof axis === "string" ? (axis === input.key ? output.key : axis) : axis.key === input.key && axis.source === output.domain ? {...axis, key:output.key} : axis);
    else evaluation[input.field] = output.key;
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
