import { describe, expect, test } from "vitest";
import { parseDocument } from "yaml";
import { addStage, connect, disconnect, moveLayer, newSnapshot, placeNode, project, reorderStage, signature, transferLayer, validConnection } from "../model/graph";
import {category} from "../model/catalog";
import { createDraftStore, layoutSignature, yamlText } from "../model/draft";
import type { Catalog, Config, Document, Layer } from "../model/types";

const layer=(name:string,keys:string[],output:string[]):Layer=>({component:{type:name,version:"1",params:{factory:"custom.Module",kwargs:{untouched:[1,2]}}},keys_in:keys,keys_out:output});
function fixture():Config {
  return {name:"ordered",data:{source:null,datasets:[],feature_key:"features",input_shapes:{features:[8]},targets:[{key:"labels"}]},
    pipeline:{stages:{"10":[{...layer("first",["features"],["features"]),meta_out:{width:"latent_width"}}],
      "2":[{...layer("second",[],["loss"]),keys_in:{input:"features"},label_key:"labels",meta_in:{size:"latent_width"}}]}},
    training:{loss_keys:{loss:1},metric_keys:[]},evaluation:{algorithms:[{algorithm:{type:"evaluation",version:"1",params:{}},feature_key:"features",label_key:"labels",axis_keys:[{key:"domain",source:"metadata"}]}]},
    logging:{custom:"untouched"},execution:{type:"local",version:"1",params:{}}};
}

describe("ordered config projection",()=>{
  test("semantic signatures ignore mapping order but preserve stage order and serialized types",()=>{
    const original=newSnapshot(fixture(),["10","2"]);
    const reordered=structuredClone(original);reordered.config=Object.fromEntries(Object.entries(original.config).reverse()) as Config;
    reordered.config.pipeline.stages=Object.fromEntries(Object.entries(reordered.config.pipeline.stages).reverse());
    expect(signature(reordered)).toBe(signature(original));
    reordered.order.reverse();expect(signature(reordered)).not.toBe(signature(original));
    reordered.order.reverse();reordered.config.training.lr="1e-3";
    const numeric=structuredClone(reordered);numeric.config.training.lr=.001;
    expect(signature(reordered)).not.toBe(signature(numeric));
  });
  test("explicit integer stage order, preceding overwritten producers and aliases",()=>{
    const snapshot=newSnapshot(fixture(),["10","2"]);const graph=project(snapshot);
    expect(graph.nodes.filter(node=>node.data.kind==="layer").map(node=>node.data.title)).toEqual(["first","second"]);
    const second=snapshot.ids["2"][0];
    expect(graph.edges.find(edge=>edge.target===second && edge.targetHandle==="keys_in:input")?.source).toBe(snapshot.ids["10"][0]);
    expect(graph.edges.find(edge=>edge.target===second && edge.targetHandle==="label_key:labels")?.source).toBe("data");
    expect(graph.edges.find(edge=>edge.target===second && edge.targetHandle==="meta_in:size")?.source).toBe(snapshot.ids["10"][0]);
    expect(graph.edges.find(edge=>edge.target==="evaluation:0" && edge.targetHandle==="feature_key:features")?.source).toBe(snapshot.ids["10"][0]);
    expect(graph.edges.find(edge=>edge.target==="objectives")?.source).toBe(second);
    expect(graph.problems.join()).toContain("resolves from dataset columns");
    const prior=signature(snapshot);snapshot.positions[snapshot.ids["10"][0]]={x:100,y:200};
    expect(signature(snapshot)).toBe(prior);
    expect(project(snapshot).nodes.find(node=>node.id===snapshot.ids["10"][0])?.position).toEqual({x:100,y:200});
  });
  test("connections reject forward references and old overwritten producers",()=>{
    const snapshot=newSnapshot(fixture(),["10","2"]);
    const target=snapshot.ids["2"][0];
    const correct={source:snapshot.ids["10"][0],sourceHandle:"output:features",target,targetHandle:"keys_in:input"};
    expect(validConnection(snapshot,correct)).toBe(true);
    expect(validConnection(snapshot,{...correct,source:"data"})).toBe(false);
    expect(validConnection(snapshot,{source:target,sourceHandle:"output:loss",target:snapshot.ids["10"][0],targetHandle:"keys_in:features"})).toBe(false);
    expect(connect(snapshot,correct).config.pipeline.stages["2"][0].keys_in).toEqual({input:"features"});
  });
  test("reordering keeps placement identity and configuration remains ordinary YAML",()=>{
    const config=fixture();config.pipeline.stages["10"].push(layer("third",["features"],["different"]));
    const snapshot=newSnapshot(config,["10","2"]);const id=snapshot.ids["10"][0];
    snapshot.positions[id]={x:4,y:8};const moved=moveLayer(snapshot,"10",0,1);
    expect(moved.ids["10"][1]).toBe(id);expect(moved.positions[id]).toEqual({x:4,y:8});
    const text=yamlText(moved);expect(text.indexOf('"10":')).toBeLessThan(text.indexOf('"2":'));
    expect(text).not.toContain(id);expect(parseDocument(text).toJS().pipeline.stages["10"][1].component.params.kwargs.untouched).toEqual([1,2]);
  });
  test("undo/redo and blocked invalid YAML preserve data; separate stores cannot share drafts",()=>{
    const store=createDraftStore();const other=createDraftStore();const config=fixture();
    const document:Document={data:config,stage_order:["10","2"],yaml:"",semantic_revision:"test",path:"scenario.yaml",base_revision:"filehash"};
    store.getState().load(document);const original=store.getState().draft!;
    const updated=structuredClone(original);updated.config.name="edited";store.getState().change(updated);
    store.getState().undo();expect(store.getState().draft?.config.name).toBe("ordered");
    store.getState().redo();expect(store.getState().draft?.config.name).toBe("edited");
    store.getState().block(true);store.getState().change(original);store.getState().undo();
    expect(store.getState().draft?.config.name).toBe("edited");expect(other.getState().draft).toBe(null);
    expect(store.getState().draft?.config.logging).toEqual({custom:"untouched"});
    const equivalent=newSnapshot(structuredClone(updated.config),updated.order,updated);
    expect(equivalent.ids).toEqual(updated.ids);
  });
  test("placement edits do not stale builds, but remain unsaved; delayed saves cannot mark later edits saved",()=>{
    const store=createDraftStore();const document:Document={data:fixture(),stage_order:["10","2"],yaml:"",semantic_revision:"test",path:"scenario.yaml",base_revision:"filehash"};
    store.getState().load(document);const source=store.getState().draft!;
    const placed=structuredClone(source);placed.positions[source.ids["10"][0]]={x:9,y:10};store.getState().change(placed);
    expect(signature(placed)).toBe(store.getState().saved);
    expect(layoutSignature(placed)).not.toBe(store.getState().savedLayout);
    expect(store.getState().past).toHaveLength(1);
    const edited=structuredClone(placed);edited.config.name="later edit";store.getState().change(edited);
    store.getState().markSaved(document,source);store.getState().markLayoutSaved(source);
    expect(signature(store.getState().draft!)).not.toBe(store.getState().saved);
    expect(layoutSignature(store.getState().draft!)).not.toBe(store.getState().savedLayout);
  });
  test("invalid placement mappings fall back without changing executable configuration",()=>{
    const store=createDraftStore();const document:Document={data:fixture(),stage_order:["10","2"],yaml:"",semantic_revision:"test"};
    store.getState().load(document,{ids:{"10":["duplicate"],"2":["duplicate"]},positions:{}});
    expect(store.getState().draft!.ids["10"][0]).not.toBe("duplicate");
    const source=store.getState().draft!;
    store.getState().load(document,{ids:source.ids,positions:{[source.ids["10"][0]]:{x:NaN,y:0}}});
    expect(store.getState().draft!.positions).toEqual({});
    expect(store.getState().draft!.config).toEqual(document.data);
  });
  test("empty input handles connect, removals preserve aliases and undo restores routes",()=>{
    const snapshot=newSnapshot(fixture(),["10","2"]);const target=snapshot.ids["2"][0];
    const edge=project(snapshot).edges.find(edge=>edge.target===target && edge.targetHandle==="keys_in:input")!;
    const disconnected=disconnect(snapshot,[edge]);
    expect(disconnected.config.pipeline.stages["2"][0].keys_in).toEqual({input:""});
    expect(project(disconnected).unconnected).toBe(true);
    const listEdge=project(snapshot).edges.find(edge=>edge.target===snapshot.ids["10"][0] && edge.targetHandle==="keys_in:features")!;
    const missing=disconnect(snapshot,[listEdge]);
    expect(missing.config.pipeline.stages["10"][0].keys_in).toEqual([""]);
    expect(project(missing).unconnected).toBe(true);
    const connected=connect(disconnected,{source:edge.source,sourceHandle:edge.sourceHandle!,target,targetHandle:"keys_in:input"});
    expect(connected.config.pipeline.stages["2"][0].keys_in).toEqual({input:"features"});
    connected.config.pipeline.stages["2"][0].keys_in=[];
    expect(connect(connected,{source:edge.source,sourceHandle:edge.sourceHandle!,target,targetHandle:"keys_in:$new"}).config.pipeline.stages["2"][0].keys_in).toEqual(["features"]);
    const store=createDraftStore();store.getState().load({data:snapshot.config,stage_order:snapshot.order,yaml:"",semantic_revision:"source"},{ids:snapshot.ids,positions:{}});
    store.getState().change(disconnected);store.getState().undo();expect(store.getState().draft!.config).toEqual(snapshot.config);
  });
  test("installed metadata projects dataset labels, evaluator parameter routes and honest defaults",()=>{
    const config=fixture();config.data.source={type:"Images",version:"1",params:{}};config.data.targets=[];
    config.evaluation.algorithms[0]={algorithm:{type:"Report",version:"1",params:{score_key:"loss"}},label_key:"class_labels"};
    const catalog:Catalog={scenarios:[],errors:[],libraries:{packages:[],roots:[]},schema:{},backends:{},components:[{kind:"data_source",name:"Images",version:"1",import_target:"local.data.images.Images",schema:{"x-nexuml-outputs":[{domain:"y",key:"class_labels"}]}},
      {kind:"data_source",name:"Report",version:"1",import_target:"local.data.Report",schema:{}},
      {kind:"eval_algorithm",name:"Report",version:"1",import_target:"local.evaluation.report.Report",schema:{"x-nexuml-routing":{label_key:{domain:"y",default:"y_true"},"algorithm.params.score_key":{domain:"x",required:true}}}}]};
    const snapshot=newSnapshot(config,["10","2"]);const graph=project(snapshot,catalog);
    expect(graph.edges.find(edge=>edge.target==="evaluation:0" && edge.targetHandle==="label_key:class_labels")?.source).toBe("data");
    const score=graph.edges.find(edge=>edge.target==="evaluation:0" && edge.targetHandle==="algorithm.params.score_key:loss")!;
    expect(score.source).toBe(snapshot.ids["2"][0]);
    const changed=disconnect(snapshot,[score],catalog);expect(changed.config.evaluation.algorithms[0].algorithm.params.score_key).toBe("");expect(project(changed,catalog).unconnected).toBe(true);
    expect(()=>disconnect(snapshot,[{target:"evaluation:0",targetHandle:"label_key:class_labels"}],catalog)).toThrow("inherits a runtime default");
  });
  test("names are sidecar-only, restore and undo; installed category metadata wins",()=>{
    const document:Document={data:fixture(),stage_order:["10","2"],yaml:"",semantic_revision:"source"};
    const store=createDraftStore();store.getState().load(document);const original=store.getState().draft!;
    const named={...original,names:{data:"Input images",[original.ids["10"][0]]:"Encoder"}};store.getState().change(named);
    expect(signature(named)).toBe(signature(original));expect(yamlText(named)).not.toContain("Input images");expect(layoutSignature(named)).not.toBe(layoutSignature(original));
    store.getState().undo();expect(store.getState().draft?.names?.data).toBeUndefined();
    store.getState().load(document,{ids:named.ids,positions:named.positions,names:named.names});expect(project(store.getState().draft!).nodes[0].data.title).toBe("Input images");
    expect(category({kind:"layer",name:"ResNet",version:"1",import_target:"library.layers.model.resnet.ResNet",schema:{"x-nexuml-category":["Models","Vision"]}})).toEqual(["Pipeline layers","Models","Vision"]);
  });
  test("empty label, objective and evaluator collection ports keep their configured types",()=>{
    const config=fixture();config.pipeline.stages["10"][0].label_key=[];config.training.loss_keys={};config.training.metric_keys=[];
    config.evaluation.algorithms=[{algorithm:{type:"Report",version:"1",params:{label_keys:null}}}];
    const catalog:Catalog={scenarios:[],errors:[],libraries:{packages:[],roots:[]},schema:{},backends:{},components:[{kind:"eval_algorithm",name:"Report",version:"1",import_target:"local.Report",schema:{properties:{label_keys:{anyOf:[{type:"array",items:{type:"string"}},{type:"null"}]}},"x-nexuml-routing":{"algorithm.params.label_keys":{domain:"y"}}}}]};
    let snapshot=newSnapshot(config,["10","2"]);
    snapshot=connect(snapshot,{source:"data",sourceHandle:"label:labels",target:snapshot.ids["10"][0],targetHandle:"label_key:$new"},catalog);
    expect(snapshot.config.pipeline.stages["10"][0].label_key).toEqual(["labels"]);
    snapshot=connect(snapshot,{source:"data",sourceHandle:"label:labels",target:"evaluation:0",targetHandle:"algorithm.params.label_keys:"},catalog);
    expect(snapshot.config.evaluation.algorithms[0].algorithm.params.label_keys).toEqual(["labels"]);
    snapshot=connect(snapshot,{source:snapshot.ids["2"][0],sourceHandle:"output:loss",target:"objectives",targetHandle:"loss_keys:$new"},catalog);
    expect(snapshot.config.training.loss_keys).toEqual({loss:1});
    snapshot=connect(snapshot,{source:snapshot.ids["2"][0],sourceHandle:"output:loss",target:"objectives",targetHandle:"metric_keys:$new"},catalog);
    expect(snapshot.config.training.metric_keys).toEqual(["loss"]);
  });
  test("stage positions/sizes carry children, contain measured bounds and remain layout-only",()=>{
    const snapshot=newSnapshot(fixture(),["10","2"]);const id=snapshot.ids["10"][0];
    const moved=placeNode(snapshot,"stage:10",{x:1000,y:400});moved.sizes={"stage:10":{width:800,height:900}};
    const graph=project(moved);const stage=graph.nodes.find(node=>node.id==="stage:10")!;
    expect(stage.position).toEqual({x:1000,y:400});expect(stage.dragHandle).toBe(".stage-header");
    expect(graph.nodes.find(node=>node.id===id)?.parentId).toBe("stage:10");
    expect(graph.nodes.find(node=>node.id===id)?.data.executionPosition).toBe("1.1");
    expect(stage.style).toEqual({width:800,height:900});expect(signature(moved)).toBe(signature(snapshot));
    const placed=placeNode(moved,id,{x:700,y:850});const contained=project(placed,undefined,{[id]:{width:250,height:350}}).nodes.find(node=>node.id===stage.id)!;
    expect(contained.style).toEqual({width:970,height:1220});
    expect(placeNode(placed,id,{x:-50,y:10}).positions[id]).toEqual({x:20,y:90});
  });
  test("explicit stage slots retain integer-name order, creation defaults and YAML fidelity",()=>{
    const snapshot=newSnapshot(fixture(),["10","2"]);const position=project(snapshot).nodes.find(node=>node.id==="stage:10")!.position;
    const moved=reorderStage(snapshot,"10",2);expect(moved.order).toEqual(["2","10"]);
    expect(project(moved).nodes.find(node=>node.id==="stage:10")!.position).toEqual(position);
    expect(yamlText(moved).indexOf('"2":')).toBeLessThan(yamlText(moved).indexOf('"10":'));
    const added=addStage(snapshot,"Empty","10",{x:999,y:888});expect(added.order).toEqual(["10","Empty","2"]);
    expect(added.selected).toBe("stage:Empty");expect(project(added).nodes.find(node=>node.id==="stage:Empty")?.data.empty).toBe(true);
    expect(addStage(snapshot,"Last").order).toEqual(["10","2","Last"]);
    expect(()=>addStage(snapshot," 10 ")).toThrow("unique");expect(()=>addStage(snapshot," ")).toThrow("non-empty");
    expect(()=>reorderStage(snapshot,"10",-1)).toThrow("slot");
  });
  test("transfers retain duplicate-layer identity/routing/selection and one transaction undo/redo",()=>{
    const config=fixture();config.pipeline.stages["10"].push(structuredClone(config.pipeline.stages["10"][0]));
    config.data.skip_pipeline_stages=["2"];
    const store=createDraftStore();const document:Document={data:config,stage_order:["10","2"],yaml:"",semantic_revision:"source"};store.getState().load(document);
    const source=store.getState().draft!;const [first,second]=source.ids["10"];store.getState().select(second);
    const moved=transferLayer(store.getState().draft!,second,"2",0,{x:40,y:200});store.getState().change(moved);
    expect(moved.ids["10"]).toEqual([first]);expect(moved.ids["2"][0]).toBe(second);expect(moved.selected).toBe(second);
    expect(moved.config.pipeline.stages["2"][0]).toEqual(config.pipeline.stages["10"][1]);expect(moved.config.logging).toEqual(config.logging);
    expect(moved.positions[second]).toEqual({x:40,y:200});expect(moved.config.data.skip_pipeline_stages).toEqual(["2"]);
    expect(project(moved).nodes.find(node=>node.id===second)?.parentId).toBe("stage:2");expect(project(moved).nodes.find(node=>node.id===second)?.data.summary).toBe("Skipped stage");
    expect(store.getState().past).toHaveLength(1);store.getState().undo();expect(store.getState().draft?.ids["10"]).toEqual([first,second]);expect(store.getState().draft?.selected).toBe(second);
    store.getState().redo();expect(store.getState().draft).toEqual(moved);expect(signature(moved)).not.toBe(signature(source));
    expect(()=>transferLayer(moved,second,"missing",0)).toThrow("existing stage");expect(()=>transferLayer(moved,second,"2",20)).toThrow("slot");
    const noChange=transferLayer(moved,second,"2",1);store.getState().change(noChange);expect(store.getState().past).toHaveLength(1);
  });
  test("sequence edits reproject overwritten/alias/metadata dependencies without repairing keys",()=>{
    const snapshot=newSnapshot(fixture(),["10","2"]);const target=snapshot.ids["2"][0];
    const moved=transferLayer(snapshot,target,"10",0);const graph=project(moved);
    expect(graph.edges.find(edge=>edge.target===target && edge.targetHandle==="keys_in:input")?.source).toBe("data");
    expect(graph.edges.find(edge=>edge.target===target && edge.targetHandle==="meta_in:size")).toBeUndefined();
    expect(graph.edges.find(edge=>edge.target===target && edge.targetHandle==="label_key:labels")?.source).toBe("data");
    expect(graph.problems.join()).toContain("meta:latent_width has no preceding");expect(moved.config.pipeline.stages["10"][0]).toEqual(snapshot.config.pipeline.stages["2"][0]);
    expect(graph.problems.join()).toContain("resolves from dataset columns");expect(moved.config.training).toEqual(snapshot.config.training);
  });
  test("sizes restore separately, older/invalid sizes fall back, layout undo retains build signature",()=>{
    const store=createDraftStore();const document:Document={data:fixture(),stage_order:["10","2"],yaml:"",semantic_revision:"source"};store.getState().load(document);
    const source=store.getState().draft!;const moved=placeNode(source,"stage:10",{x:900,y:100});moved.sizes={"stage:10":{width:800,height:900}};
    store.getState().change(moved);expect(store.getState().past).toHaveLength(1);store.getState().undo();expect(store.getState().draft?.sizes).toEqual({});store.getState().redo();expect(store.getState().draft).toEqual(moved);
    expect(signature(moved)).toBe(signature(source));expect(yamlText(moved)).not.toContain("900");
    store.getState().load(document,moved);expect(store.getState().draft?.sizes).toEqual(moved.sizes);
    store.getState().load(document,{ids:moved.ids,positions:moved.positions});expect(store.getState().draft?.sizes).toEqual({});
    store.getState().load(document,{ids:moved.ids,positions:moved.positions,sizes:{"stage:10":{width:NaN,height:-1}}});expect(store.getState().draft?.sizes).toEqual({});
    store.getState().block(true);store.getState().change(moved);expect(store.getState().past).toHaveLength(0);
  });
});
