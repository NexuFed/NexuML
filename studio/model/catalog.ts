import type {Catalog,Entry} from "./types";

const kinds:Record<string,string>={layer:"Pipeline layers",data_source:"Data sources",eval_algorithm:"Evaluations",loader_backend:"Data loaders"};
const title=(word:string)=>word.replaceAll("_"," ").replace(/\b\w/g,char=>char.toUpperCase());

export function category(entry:Entry):string[]{
  const parts=entry.import_target.split(".");
  const index=parts.findIndex(part=>["layers","data","evaluation","loaders"].includes(part));
  const path=index>=0 ? parts.slice(index+1,-2).map(part=>part==="model" ? "Models" : title(part)) : [title(parts[0] ?? "Other")];
  return [kinds[entry.kind] ?? title(entry.kind),...path];
}

const record=(value:unknown):value is Record<string,unknown>=>value!==null && typeof value==="object" && !Array.isArray(value);
const strings=(value:unknown):value is string[]=>Array.isArray(value) && value.every(item=>typeof item==="string");
function invalid(path:string):never {
  throw new Error(`Invalid native I/O catalog at ${path}; select matching NexuML/Studio packages and restart Studio.`);
}

export function validateCatalog(value:unknown):Catalog {
  if(!record(value))invalid("catalog");
  if(!Array.isArray(value.components))invalid("components");
  for(const [index,entry] of value.components.entries()){
    const path=`components[${index}]`;
    if(!record(entry) || ![entry.kind,entry.name,entry.version,entry.import_target].every(item=>typeof item==="string") || !record(entry.schema))invalid(path);
    const io=entry.io;
    if(!record(io))invalid(`${path}.io`);
    if(io.labels!==null && !strings(io.labels))invalid(`${path}.io.labels`);
    if(typeof io.inputs_known!=="boolean")invalid(`${path}.io.inputs_known`);
    if(!Array.isArray(io.inputs))invalid(`${path}.io.inputs`);
    const fields=new Set<string>();
    for(const [index,binding] of io.inputs.entries()){
      const input=`${path}.io.inputs[${index}]`;
      if(!record(binding))invalid(input);
      if(binding.source!=="context" && binding.source!=="parameter")invalid(`${input}.source`);
      if(typeof binding.field!=="string" || binding.field==="__proto__" || !/^[_\p{ID_Start}][\p{ID_Continue}]*$/u.test(binding.field))invalid(`${input}.field (expected a local field name)`);
      if(binding.domain!=="x" && binding.domain!=="y")invalid(`${input}.domain`);
      if(typeof binding.collection!=="boolean")invalid(`${input}.collection`);
      if(typeof binding.required!=="boolean")invalid(`${input}.required`);
      if(binding.source==="context" && (binding.collection || !((binding.field==="feature_key" && binding.domain==="x") || (binding.field==="label_key" && binding.domain==="y"))))invalid(`${input} (expected scalar feature_key/x or label_key/y context)`);
      if(binding.default!==null && !(binding.collection ? strings(binding.default) : typeof binding.default==="string"))invalid(`${input}.default`);
      const field=`${binding.source}:${binding.field}`;
      if(fields.has(field))invalid(`${input}.field (duplicate binding)`);
      fields.add(field);
    }
  }
  if(!Array.isArray(value.scenarios) || !value.scenarios.every(item=>record(item) && typeof item.name==="string"))invalid("scenarios");
  if(!Array.isArray(value.errors) || !value.errors.every(item=>record(item) && typeof item.module==="string" && typeof item.message==="string"))invalid("errors");
  if(!record(value.libraries) || !strings(value.libraries.roots) || !strings(value.libraries.packages))invalid("libraries");
  if(!record(value.schema))invalid("schema");
  if(!record(value.backends) || !Object.values(value.backends).every(item=>record(item) && typeof item.available==="boolean"))invalid("backends");
  if(value.execution_backends!==undefined && (!Array.isArray(value.execution_backends) || !value.execution_backends.every(item=>record(item) &&
    [item.type,item.version,item.label,item.import_target].every(value=>typeof value==="string") && record(item.schema) && typeof item.available==="boolean" && strings(item.diagnostics) &&
    record(item.capabilities) && Object.values(item.capabilities).every(value=>typeof value==="boolean") && record(item.presentation) && Object.values(item.presentation).every(strings))))invalid("execution_backends");
  return value as Catalog;
}
