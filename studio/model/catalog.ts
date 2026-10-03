import type {Entry} from "./types";

const kinds:Record<string,string>={layer:"Pipeline layers",data_source:"Data sources",eval_algorithm:"Evaluations",loader_backend:"Data loaders"};
const title=(word:string)=>word.replaceAll("_"," ").replace(/\b\w/g,char=>char.toUpperCase());

export function category(entry:Entry):string[]{
  const parts=entry.import_target.split(".");
  const index=parts.findIndex(part=>["layers","data","evaluation","loaders"].includes(part));
  const path=entry.schema["x-nexuml-category"] ?? (index>=0 ? parts.slice(index+1,-2).map(part=>part==="model" ? "Models" : title(part)) : [title(parts[0] ?? "Other")]);
  return [kinds[entry.kind] ?? title(entry.kind),...path];
}
