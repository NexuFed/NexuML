import { describe, expect, test } from "vitest";
import { parseDocument } from "yaml";
import { connect, moveLayer, newSnapshot, project, signature, validConnection } from "../model/graph";
import { createDraftStore, layoutSignature, yamlText } from "../model/draft";
import type { Config, Document, Layer } from "../model/types";

const layer=(name:string,keys:string[],output:string[]):Layer=>({component:{type:name,version:"1",params:{factory:"custom.Module",kwargs:{untouched:[1,2]}}},keys_in:keys,keys_out:output});
function fixture():Config {
  return {name:"ordered",data:{source:null,datasets:[],feature_key:"features",input_shapes:{features:[8]},targets:[{key:"labels"}]},
    pipeline:{stages:{"10":[{...layer("first",["features"],["features"]),meta_out:{width:"latent_width"}}],
      "2":[{...layer("second",[],["loss"]),keys_in:{input:"features"},label_key:"labels",meta_in:{size:"latent_width"}}]}},
    training:{loss_keys:{loss:1},metric_keys:[]},evaluation:{algorithms:[{algorithm:{type:"evaluation",version:"1",params:{}},feature_key:"features",label_key:"labels",axis_keys:[{key:"domain",source:"metadata"}]}]},
    logging:{custom:"untouched"},execution:{kind:"local"}};
}

describe("ordered config projection",()=>{
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
    expect(store.getState().past).toHaveLength(0);
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
});
