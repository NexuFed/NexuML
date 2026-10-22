import { describe, expect, test } from "vitest";
import { parseDocument } from "yaml";
import { addStage, connect, disconnect, moveLayer, newSnapshot, placeNode, project, reorderStage, signature, transferLayer, validConnection } from "../model/graph";
import {category} from "../model/catalog";
import { createDraftStore, layoutSignature, yamlText } from "../model/draft";
import type { Catalog, ComponentIO, Config, Document, Entry, InputBinding, Layer, Schema } from "../model/types";

const layer=(name:string,keys:string[],output:string[]):Layer=>({component:{type:name,version:"1",params:{factory:"custom.Module",kwargs:{untouched:[1,2]}}},keys_in:keys,keys_out:output});
const binding=(source:InputBinding["source"],field:string,domain:InputBinding["domain"],options:Partial<InputBinding>={}):InputBinding=>({source,field,domain,collection:false,required:false,default:null,...options});
const entry=(kind:string,name:string,io:ComponentIO={labels:null,inputs_known:false,inputs:[]},schema:Schema={}):Entry=>({kind,name,version:"1",import_target:`local.${name}`,schema,io});
const catalogOf=(components:Entry[]):Catalog=>({scenarios:[],errors:[],libraries:{packages:[],roots:[]},schema:{},backends:{},components});
const contextCatalog=catalogOf([entry("eval_algorithm","evaluation",{labels:null,inputs_known:true,inputs:[binding("context","feature_key","x"),binding("context","label_key","y")]})]);
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
    const snapshot=newSnapshot(fixture(),["10","2"]);const graph=project(snapshot,contextCatalog);
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
    const snapshot=newSnapshot(fixture(),["10","2"]);const graph=project(snapshot);
    const target=snapshot.ids["2"][0];
    const correct={source:snapshot.ids["10"][0],sourceHandle:"output:features",target,targetHandle:"keys_in:input"};
    expect(validConnection(graph,correct)).toBe(true);
    expect(validConnection(graph,{...correct,source:"data"})).toBe(false);
    expect(validConnection(graph,{source:target,sourceHandle:"output:loss",target:snapshot.ids["10"][0],targetHandle:"keys_in:features"})).toBe(false);
    expect(connect(snapshot,correct).config.pipeline.stages["2"][0].keys_in).toEqual({input:"features"});
  });
  test("connection checks reject missing handles, empty outputs, self references and mismatched domains",()=>{
    const config=fixture();config.pipeline.stages["10"][0].keys_out.push("");
    const snapshot=newSnapshot(config,["10","2"]);const graph=project(snapshot);
    const connection={source:snapshot.ids["10"][0],sourceHandle:"output:features",target:snapshot.ids["2"][0],targetHandle:"keys_in:input"};
    for(const change of [
      {source:"missing"},{target:"missing"},{sourceHandle:null},{targetHandle:null},
      {sourceHandle:"output:missing"},{targetHandle:"keys_in:missing"},{sourceHandle:"output:"},
      {target:connection.source,targetHandle:"keys_in:features"},
      {source:"data",sourceHandle:"label:labels"},{sourceHandle:"meta_out:width"},
      {targetHandle:"label_key:labels"},{targetHandle:"meta_in:size"},
    ])expect(validConnection(graph,{...connection,...change})).toBe(false);
    expect(validConnection(graph,{...connection,sourceHandle:"meta_out:width",targetHandle:"meta_in:size"})).toBe(true);
    expect(validConnection(graph,{...connection,source:"data",sourceHandle:"label:labels",targetHandle:"label_key:labels"})).toBe(true);
    expect(()=>connect(snapshot,{...connection,sourceHandle:"meta_out:width"})).toThrow("matching key domain");
  });
  test("only the last writer before each target is eligible, even with a later overwrite",()=>{
    const config=fixture();config.pipeline.stages["2"].push(layer("later",["features"],["features"]));
    const snapshot=newSnapshot(config,["10","2"]);const graph=project(snapshot,contextCatalog);
    const first=snapshot.ids["10"][0], [second,later]=snapshot.ids["2"];
    const connection={source:first,sourceHandle:"output:features",target:second,targetHandle:"keys_in:input"};
    expect(validConnection(graph,connection)).toBe(true);
    expect(validConnection(graph,{...connection,source:later})).toBe(false);
    expect(validConnection(graph,{...connection,source:"data"})).toBe(false);
    const evaluation={...connection,target:"evaluation:0",targetHandle:"feature_key:features"};
    expect(validConnection(graph,evaluation)).toBe(false);
    expect(validConnection(graph,{...evaluation,source:later})).toBe(true);
    expect(graph.edges.find(edge=>edge.target===second && edge.targetHandle===connection.targetHandle)?.source).toBe(first);
    expect(graph.edges.find(edge=>edge.target===evaluation.target && edge.targetHandle===evaluation.targetHandle)?.source).toBe(later);
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
  test("connections use post-disconnect handles rather than stale list-input handles",()=>{
    const snapshot=newSnapshot(fixture(),["10","2"]);const graph=project(snapshot);
    const edge=graph.edges.find(edge=>edge.target===snapshot.ids["10"][0])!;
    const connection={source:edge.source,sourceHandle:edge.sourceHandle!,target:edge.target,targetHandle:edge.targetHandle!};
    expect(validConnection(graph,connection)).toBe(true);
    const disconnected=disconnect(snapshot,[edge]);const current=project(disconnected);
    expect(validConnection(current,connection)).toBe(false);
    expect(()=>connect(disconnected,connection)).toThrow("preceding producer");
    const replacement={...connection,targetHandle:"keys_in:"};
    expect(validConnection(current,replacement)).toBe(true);
    expect(connect(disconnected,replacement).config.pipeline.stages["10"][0].keys_in).toEqual(["features"]);
  });
  test("native I/O projects dataset labels, evaluator parameter routes and honest defaults",()=>{
    const config=fixture();config.data.source={type:"Images",version:"1",params:{}};config.data.targets=[];
    config.evaluation.algorithms[0]={algorithm:{type:"Report",version:"1",params:{score_key:"loss"}},label_key:"class_labels"};
    const catalog=catalogOf([entry("data_source","Images",{labels:["class_labels"],inputs_known:false,inputs:[]}),
      entry("data_source","Report"),entry("eval_algorithm","Report",{labels:null,inputs_known:true,inputs:[
        binding("context","label_key","y",{default:"y_true"}),binding("parameter","score_key","x",{required:true})]})]);
    const snapshot=newSnapshot(config,["10","2"]);const graph=project(snapshot,catalog);
    expect(graph.edges.find(edge=>edge.target==="evaluation:0" && edge.targetHandle==="label_key:class_labels")?.source).toBe("data");
    const score=graph.edges.find(edge=>edge.target==="evaluation:0" && edge.targetHandle==="algorithm.params.score_key:loss")!;
    expect(score.source).toBe(snapshot.ids["2"][0]);
    const changed=disconnect(snapshot,[score],catalog);expect(changed.config.evaluation.algorithms[0].algorithm.params.score_key).toBe("");expect(project(changed,catalog).unconnected).toBe(true);
    expect(()=>disconnect(snapshot,[{target:"evaluation:0",targetHandle:"label_key:class_labels"}],catalog)).toThrow("inherits a runtime default");
  });
  test("evaluator connection checks use native defaults and routing domains, not schema defaults",()=>{
    const config=fixture();config.evaluation.algorithms=[{algorithm:{type:"Report",version:"1",params:{}}}];
    const catalog=catalogOf([entry("eval_algorithm","Report",{labels:null,inputs_known:true,inputs:[
      binding("parameter","score_key","x",{default:"loss"}),binding("context","label_key","y",{default:"labels"})]},
    {properties:{score_key:{type:"string",default:"not_the_native_default"}}})]);
    const snapshot=newSnapshot(config,["10","2"]);const graph=project(snapshot,catalog);
    const score={source:snapshot.ids["2"][0],sourceHandle:"output:loss",target:"evaluation:0",targetHandle:"algorithm.params.score_key:loss"};
    const label={source:"data",sourceHandle:"label:labels",target:"evaluation:0",targetHandle:"label_key:labels"};
    expect(validConnection(graph,score)).toBe(true);expect(validConnection(graph,label)).toBe(true);
    expect(validConnection(graph,{...score,source:"data",sourceHandle:"label:labels"})).toBe(false);
    expect(validConnection(graph,{...score,targetHandle:"algorithm.params.score_key:"})).toBe(false);
    expect(connect(snapshot,score,catalog).config.evaluation.algorithms[0].algorithm.params.score_key).toBe("loss");
    expect(connect(snapshot,label,catalog).config.evaluation.algorithms[0].label_key).toBe("labels");
    expect(snapshot.config.evaluation.algorithms[0].algorithm.params).toEqual({});
    expect(()=>disconnect(snapshot,[label],catalog)).toThrow("inherits a runtime default");
    const removed=disconnect(snapshot,[score],catalog);
    expect(removed.config.evaluation.algorithms[0].algorithm.params.score_key).toBe("");
    expect(project(removed,catalog).nodes.find(node=>node.id==="evaluation:0")?.data.inputs[0].key).toBe("");
    expect(connect(removed,{...score,targetHandle:"algorithm.params.score_key:"},catalog).config.evaluation.algorithms[0].algorithm.params.score_key).toBe("loss");
  });
  test("names are sidecar-only, restore and undo; groups follow module provenance",()=>{
    const document:Document={data:fixture(),stage_order:["10","2"],yaml:"",semantic_revision:"source"};
    const store=createDraftStore();store.getState().load(document);const original=store.getState().draft!;
    const named={...original,names:{data:"Input images",[original.ids["10"][0]]:"Encoder"}};store.getState().change(named);
    expect(signature(named)).toBe(signature(original));expect(yamlText(named)).not.toContain("Input images");expect(layoutSignature(named)).not.toBe(layoutSignature(original));
    store.getState().undo();expect(store.getState().draft?.names?.data).toBeUndefined();
    store.getState().load(document,{ids:named.ids,positions:named.positions,names:named.names});expect(project(store.getState().draft!).nodes[0].data.title).toBe("Input images");
    expect(category({...entry("layer","ResNet"),import_target:"library.layers.model.resnet.ResNet"})).toEqual(["Pipeline layers","Models"]);
  });
  test("empty label, objective and evaluator collection ports keep their configured types",()=>{
    const config=fixture();config.pipeline.stages["10"][0].label_key=[];config.training.loss_keys={};config.training.metric_keys=[];
    config.evaluation.algorithms=[{algorithm:{type:"Report",version:"1",params:{label_keys:null}}}];
    const catalog=catalogOf([entry("eval_algorithm","Report",{labels:null,inputs_known:true,inputs:[binding("parameter","label_keys","y",{collection:true})]},
      {properties:{label_keys:{anyOf:[{type:"array",items:{type:"string"}},{type:"null"}]}}})]);
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
  test("known source labels supplement configured labels; unknown is not known-empty",()=>{
    const config=fixture();config.evaluation.algorithms=[];
    config.data.source={type:"Images",version:"1",params:{}};
    config.data.datasets=[{source:{type:"NoLabels",version:"1",params:{}}},{source:{type:"Dynamic",version:"1",params:{}}}];
    const catalog=catalogOf([entry("data_source","Images",{labels:["labels","class_labels"],inputs_known:false,inputs:[]}),
      entry("data_source","NoLabels",{labels:[],inputs_known:false,inputs:[]}),entry("data_source","Dynamic")]);
    const snapshot=newSnapshot(config,["10","2"]);const before=signature(snapshot);const graph=project(snapshot,catalog);
    expect(graph.nodes[0].data.outputs.filter(port=>port.domain==="y").map(port=>port.key)).toEqual(["labels","class_labels"]);
    expect(graph.problems.filter(problem=>problem.includes("native source labels"))).toEqual([expect.stringContaining("Dynamic:")]);
    expect(signature(snapshot)).toBe(before);
    const missing=project(snapshot,catalogOf([]));
    expect(missing.problems.filter(problem=>problem.includes("native source labels"))).toHaveLength(3);
    expect(missing.nodes[0].data.outputs.filter(port=>port.domain==="y").map(port=>port.key)).toEqual(["labels"]);
  });
  test.each([undefined,null,""])("context fallback matches Python for %s and keeps the inherited removal guard",configured=>{
    const config=fixture();config.evaluation.algorithms=[{algorithm:{type:"Report",version:"1",params:{}},feature_key:configured,label_key:configured}];
    const catalog=catalogOf([entry("eval_algorithm","Report",{labels:null,inputs_known:true,inputs:[
      binding("context","feature_key","x",{default:"features"}),binding("context","label_key","y",{required:true})]})]);
    const snapshot=newSnapshot(config,["10","2"]);const graph=project(snapshot,catalog);
    const inputs=graph.nodes.find(node=>node.id==="evaluation:0")!.data.inputs;
    expect(inputs.map(input=>[input.field,input.key,input.defaultKey])).toEqual([["feature_key","features","features"],["label_key","",undefined]]);
    expect(graph.unconnected).toBe(true);expect(graph.problems.join()).toContain("unconnected required input label_key");
    expect(()=>disconnect(snapshot,[{target:"evaluation:0",targetHandle:"feature_key:features"}],catalog)).toThrow("inherits a runtime default");
    const changed=connect(snapshot,{source:snapshot.ids["2"][0],sourceHandle:"output:loss",target:"evaluation:0",targetHandle:"feature_key:features"},catalog);
    expect(changed.config.evaluation.algorithms[0].feature_key).toBe("loss");
    expect(project(changed,catalog).nodes.find(node=>node.id==="evaluation:0")!.data.inputs[0].key).toBe("loss");
    expect(()=>disconnect(changed,[{target:"evaluation:0",targetHandle:"feature_key:loss"}],catalog)).toThrow("inherits a runtime default");
  });
  test.each([undefined,null,""])("parameter defaults apply only when absent, not configured %s",configured=>{
    const config=fixture();config.evaluation.algorithms=[{algorithm:{type:"Report",version:"1",params:{score_key:configured}}}];
    const catalog=catalogOf([entry("eval_algorithm","Report",{labels:null,inputs_known:true,inputs:[binding("parameter","score_key","x",{default:"loss"})]})]);
    const snapshot=newSnapshot(config,["10","2"]);const graph=project(snapshot,catalog);
    const input=graph.nodes.find(node=>node.id==="evaluation:0")!.data.inputs[0];
    expect(input.key).toBe(configured===undefined ? "loss" : "");expect(input.defaultKey).toBeUndefined();
    const changed=connect(snapshot,{source:snapshot.ids["2"][0],sourceHandle:"output:loss",target:"evaluation:0",targetHandle:input.id},catalog);
    expect(changed.config.evaluation.algorithms[0].algorithm.params.score_key).toBe("loss");
    expect(disconnect(changed,[{target:"evaluation:0",targetHandle:"algorithm.params.score_key:loss"}],catalog).config.evaluation.algorithms[0].algorithm.params.score_key).toBe("");
  });
  test.each([{configured:undefined},{configured:null},{configured:[]},{configured:[""]}])("nullable collections stay arrays through connect/remove/reconnect from $configured",({configured})=>{
    const config=fixture();config.evaluation.algorithms=[{algorithm:{type:"Report",version:"1",params:{label_keys:configured}}}];
    const catalog=catalogOf([entry("eval_algorithm","Report",{labels:null,inputs_known:true,inputs:[binding("parameter","label_keys","y",{collection:true,required:true})]})]);
    let snapshot=newSnapshot(config,["10","2"]);const empty={target:"evaluation:0",targetHandle:"algorithm.params.label_keys:"};
    expect(project(snapshot,catalog).unconnected).toBe(true);
    const connection={...empty,source:"data",sourceHandle:"label:labels"};
    snapshot=connect(snapshot,connection,catalog);expect(snapshot.config.evaluation.algorithms[0].algorithm.params.label_keys).toEqual(["labels"]);
    snapshot=disconnect(snapshot,[{...empty,targetHandle:"algorithm.params.label_keys:labels"}],catalog);
    expect(snapshot.config.evaluation.algorithms[0].algorithm.params.label_keys).toEqual([]);expect(project(snapshot,catalog).unconnected).toBe(true);
    expect(connect(snapshot,connection,catalog).config.evaluation.algorithms[0].algorithm.params.label_keys).toEqual(["labels"]);
  });
  test("omitted collection defaults retain every projected key on reconnect and removal",()=>{
    const config=fixture();config.evaluation.algorithms=[{algorithm:{type:"Report",version:"1",params:{}}}];
    const catalog=catalogOf([entry("eval_algorithm","Report",{labels:null,inputs_known:true,inputs:[binding("parameter","scores","x",{collection:true,default:["loss","features"]})]})]);
    const snapshot=newSnapshot(config,["10","2"]);const score={target:"evaluation:0",targetHandle:"algorithm.params.scores:loss"};
    const changed=connect(snapshot,{...score,source:snapshot.ids["10"][0],sourceHandle:"output:features"},catalog);
    expect(changed.config.evaluation.algorithms[0].algorithm.params.scores).toEqual(["features","features"]);
    expect(disconnect(snapshot,[score],catalog).config.evaluation.algorithms[0].algorithm.params.scores).toEqual(["features"]);
    expect(disconnect(snapshot,[score,{...score,targetHandle:"algorithm.params.scores:features"}],catalog).config.evaluation.algorithms[0].algorithm.params.scores).toEqual([]);
    for(const value of [null,[]]){
      const cleared=structuredClone(snapshot);cleared.config.evaluation.algorithms[0].algorithm.params.scores=value;
      expect(project(cleared,catalog).nodes.find(node=>node.id==="evaluation:0")!.data.inputs.map(input=>input.key)).toEqual([""]);
      expect(connect(cleared,{source:snapshot.ids["2"][0],sourceHandle:"output:loss",target:"evaluation:0",targetHandle:"algorithm.params.scores:"},catalog).config.evaluation.algorithms[0].algorithm.params.scores).toEqual(["loss"]);
    }
    expect(snapshot.config.evaluation.algorithms[0].algorithm.params).toEqual({});
    catalog.components[0].io.inputs[0].default=["","loss","features"];
    expect(disconnect(snapshot,[score],catalog).config.evaluation.algorithms[0].algorithm.params.scores).toEqual(["","features"]);
  });
  test("unknown context keeps declared parameter and axis bindings without guessing context edges",()=>{
    const snapshot=newSnapshot(fixture(),["10","2"]);snapshot.config.evaluation.algorithms[0].algorithm.params={reconstructed_key:"loss",unannotated_key:"features"};
    const catalog=catalogOf([entry("eval_algorithm","evaluation",{labels:null,inputs_known:false,inputs:[binding("parameter","reconstructed_key","x")]},
      {properties:{unannotated_key:{type:"string",default:"features"}}})]);
    const before=signature(snapshot);const graph=project(snapshot,catalog);const evaluation=graph.nodes.find(node=>node.id==="evaluation:0")!;
    expect(evaluation.data.inputs.map(input=>input.field)).toEqual(["algorithm.params.reconstructed_key"]);
    expect(graph.edges.filter(edge=>edge.target===evaluation.id).map(edge=>edge.targetHandle)).toEqual(["algorithm.params.reconstructed_key:loss"]);
    expect(graph.problems.join()).toContain("native evaluator context inputs are unknown");
    expect(graph.problems.join()).toContain("resolves from dataset columns");expect(signature(snapshot)).toBe(before);
    const absent=project(snapshot);expect(absent.nodes.find(node=>node.id===evaluation.id)!.data.inputs).toEqual([]);
    expect(absent.problems.join()).toContain("native evaluator context inputs are unknown");
    const knownEmpty=project(snapshot,catalogOf([entry("eval_algorithm","evaluation",{labels:null,inputs_known:true,inputs:[]})]));
    expect(knownEmpty.problems.join()).not.toContain("native evaluator context inputs are unknown");
    snapshot.config.evaluation.algorithms[0].axis_keys=["labels",{key:"features",source:"x"}];
    expect(project(snapshot,catalog).nodes.find(node=>node.id===evaluation.id)!.data.inputs.map(input=>[input.field,input.domain,input.key])).toEqual([
      ["algorithm.params.reconstructed_key","x","loss"],["axis_keys","y","labels"],["axis_keys","x","features"]]);
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
