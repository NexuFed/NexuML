import {expect,test} from "vitest";
import {createDraftStore,layoutSignature} from "../model/draft";
import {signature} from "../model/graph";
import type {Document} from "../model/types";

test("save conflict revisions advance without marking newer config or layout saved",()=>{
  const document:Document={data:{name:"example",data:{input_shapes:{features:[8]},feature_key:"features",source:null,datasets:[],targets:[]},pipeline:{stages:{}},training:{loss_keys:{},metric_keys:[]},evaluation:{algorithms:[]},execution:{type:"local",version:"1",params:{}}},stage_order:[],yaml:"",path:"example.yaml",base_revision:"first",semantic_revision:"semantic-first"};
  const store=createDraftStore();store.getState().load(document);
  const source=store.getState().draft!;
  expect(store.getState().baseRevision).toBe("first");
  const placed={...source,positions:{data:{x:100,y:200}}};store.getState().change(placed);
  expect(signature(placed)).toBe(store.getState().saved);
  expect(layoutSignature(placed)).not.toBe(store.getState().savedLayout);
  const edited={...placed,config:{...placed.config,name:"newer edit"}};store.getState().change(edited);
  store.getState().markSaved({...document,base_revision:"second",semantic_revision:"semantic-second"},source);
  store.getState().markLayoutSaved(source);
  expect(store.getState().baseRevision).toBe("second");
  expect(store.getState().draft).toBe(edited);
  expect(signature(edited)).not.toBe(store.getState().saved);
  expect(layoutSignature(edited)).not.toBe(store.getState().savedLayout);
});
