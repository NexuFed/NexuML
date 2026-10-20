import { describe, expect, test } from "vitest";
import { category, validateCatalog } from "../model/catalog";
import type { Catalog, Entry, InputBinding } from "../model/types";

const parameter:InputBinding={source:"parameter",field:"reconstructed_key",domain:"x",collection:false,required:true,default:null};
const entry:Entry={kind:"eval_algorithm",name:"ExternalReport",version:"1",import_target:"contributed.reports.ExternalReport",schema:{},
  io:{labels:null,inputs_known:false,inputs:[parameter]}};
function fixture():Catalog {
  return {components:[structuredClone(entry)],scenarios:[{name:"example"}],errors:[{module:"optional",message:"unavailable"}],
    libraries:{roots:["/local/library"],packages:["contributed"]},schema:{$defs:{}},backends:{local:{available:true}},
    execution_backends:[{type:"local",version:"1",label:"Local",schema:{},import_target:"local.Local",available:true,
      diagnostics:[],capabilities:{training:true},presentation:{basic:["workers"]}}]};
}

describe("native I/O catalog boundary",()=>{
  test("accepts unknown, known-empty, scalar, nullable and collection bindings without resolving draft values",()=>{
    const catalog=fixture();catalog.components.push(
      {...entry,name:"Known",io:{labels:["class_labels"],inputs_known:true,inputs:[
        {source:"context",field:"feature_key",domain:"x",collection:false,required:false,default:"latent"},
        {source:"context",field:"label_key",domain:"y",collection:false,required:true,default:null},
        {...parameter,required:false,default:"loss"},
        {...parameter,field:"labels",domain:"y",collection:true,required:false,default:["truth","class_labels"]},
        {...parameter,field:"optional_labels",domain:"y",collection:true,required:false,default:null},
        {...parameter,field:"empty_labels",domain:"y",collection:true,required:false,default:[]},
      ]}},
      {...entry,name:"Empty",io:{labels:[],inputs_known:true,inputs:[]}},
      {...entry,name:"Unknown",io:{labels:null,inputs_known:false,inputs:[]}},
    );
    expect(validateCatalog(catalog)).toBe(catalog);
    expect(validateCatalog(catalog).components[0].io).toEqual({labels:null,inputs_known:false,inputs:[parameter]});
    delete catalog.execution_backends;expect(validateCatalog(catalog)).toBe(catalog);
  });

  test("rejects missing native I/O even when a matching catalog has ordinary schemas",()=>{
    const missing:Record<string,unknown>={...entry};delete missing.io;
    expect(()=>validateCatalog({...fixture(),components:[missing]})).toThrow("native I/O catalog at components[0].io");
    expect(()=>validateCatalog({...fixture(),components:[missing]})).toThrow("select matching NexuML/Studio packages");
  });

  const invalidIO:[string,unknown][]=[
    ["null",null],["array",[]],["string","unknown"],["empty object",{}],
    ["missing labels",{inputs_known:true,inputs:[]}],
    ["non-array labels",{labels:"labels",inputs_known:true,inputs:[]}],
    ["non-string label",{labels:[1],inputs_known:true,inputs:[]}],
    ["missing completeness",{labels:null,inputs:[]}],
    ["non-boolean completeness",{labels:null,inputs_known:"false",inputs:[]}],
    ["missing inputs",{labels:null,inputs_known:false}],
    ["null inputs",{labels:null,inputs_known:false,inputs:null}],
    ["mapping inputs",{labels:null,inputs_known:false,inputs:{}}],
  ];
  test.each(invalidIO)("rejects %s I/O descriptors",(_name,io)=>{
    expect(()=>validateCatalog({...fixture(),components:[{...entry,io}]})).toThrow("native I/O catalog at components[0].io");
  });

  const invalidBindings:[string,unknown][]=[
    ["null",null],["array",[]],["string","feature_key"],["empty object",{}],
    ["unknown source",{...parameter,source:"schema"}],
    ["missing source",{...parameter,source:undefined}],
    ["unknown domain",{...parameter,domain:"metadata"}],
    ["missing domain",{...parameter,domain:undefined}],
    ["missing collection",{...parameter,collection:undefined}],
    ["non-boolean collection",{...parameter,collection:1}],
    ["missing required",{...parameter,required:undefined}],
    ["non-boolean required",{...parameter,required:"false"}],
    ["missing default",{...parameter,default:undefined}],
    ["numeric scalar default",{...parameter,default:5}],
    ["array scalar default",{...parameter,default:["loss"]}],
    ["scalar collection default",{...parameter,collection:true,default:"loss"}],
    ["non-string collection default",{...parameter,collection:true,default:[null]}],
    ["non-local field",{...parameter,field:"algorithm.params.reconstructed_key"}],
    ["path field",{...parameter,field:"../reconstructed_key"}],
    ["indexed field",{...parameter,field:"keys[0]"}],
    ["empty field",{...parameter,field:""}],
    ["prototype field",{...parameter,field:"__proto__"}],
    ["prototype collection field",{...parameter,field:"__proto__",collection:true,default:[]}],
    ["non-string field",{...parameter,field:1}],
    ["unknown context slot",{...parameter,source:"context"}],
    ["wrong feature domain",{...parameter,source:"context",field:"feature_key",domain:"y"}],
    ["wrong label domain",{...parameter,source:"context",field:"label_key",domain:"x"}],
    ["context collection",{...parameter,source:"context",field:"feature_key",collection:true}],
  ];
  test.each(invalidBindings)("rejects %s bindings",(_name,binding)=>{
    expect(()=>validateCatalog({...fixture(),components:[{...entry,io:{...entry.io,inputs:[binding]}}]})).toThrow("native I/O catalog at components[0].io.inputs[0]");
  });

  test("rejects duplicate bindings but keeps context and parameter fields distinct",()=>{
    const catalog=fixture();catalog.components[0].io.inputs=[parameter,{...parameter}];
    expect(()=>validateCatalog(catalog)).toThrow("duplicate binding");
    catalog.components[0].io.inputs=[{...parameter,field:"feature_key"},{...parameter,source:"context",field:"feature_key"}];
    expect(validateCatalog(catalog)).toBe(catalog);
  });

  test.each([
    null,[],{},
    {...fixture(),components:null},
    {...fixture(),components:[null]},
    {...fixture(),components:[{...entry,schema:null}]},
    {...fixture(),components:[{...entry,name:1}]},
    {...fixture(),scenarios:null},
    {...fixture(),scenarios:[{}]},
    {...fixture(),errors:[null]},
    {...fixture(),errors:[{module:"source",message:null}]},
    {...fixture(),libraries:null},
    {...fixture(),libraries:{roots:[],packages:[1]}},
    {...fixture(),schema:[]},
    {...fixture(),backends:null},
    {...fixture(),backends:{local:{available:"yes"}}},
    {...fixture(),execution_backends:[{}]},
  ].map(value=>({value})))("rejects malformed catalog structure without a consumer render crash (%#)",({value})=>{
    expect(()=>validateCatalog(value)).toThrow("native I/O catalog");
    expect(()=>validateCatalog(value)).toThrow("select matching NexuML/Studio packages");
  });
});

describe("component module grouping",()=>{
  test.each([
    ["layer","nexuml.library.layers.model.resnet.ResNet",["Pipeline layers","Models"]],
    ["data_source","nexuml.library.data.image.mnist.MNIST",["Data sources","Image"]],
    ["eval_algorithm","nexuml.library.evaluation.report.Report",["Evaluations"]],
    ["layer","contributed.layers.audio.encoder.Encoder",["Pipeline layers","Audio"]],
    ["eval_algorithm","outside_pkg.reports.Report",["Evaluations","Outside Pkg"]],
    ["custom_kind","outside_pkg.Custom",["Custom Kind","Outside Pkg"]],
  ])("groups %s from %s",(kind,import_target,expected)=>{
    expect(category({...entry,kind,import_target})).toEqual(expected);
  });
});
