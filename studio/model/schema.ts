import type { Schema } from "./types";

export function resolveSchema(schema:Schema={}, root:Schema=schema):Schema {
  if(schema.$ref){
    const referenced=root.$defs?.[schema.$ref.split("/").at(-1)!];
    if(referenced)return {...referenced,...Object.fromEntries(Object.entries(schema).filter(([key])=>key!=="$ref"))};
  }
  if(schema.allOf)return Object.assign({},...schema.allOf.map(item=>resolveSchema(item,root)),schema);
  return schema;
}

export function matchesSchema(value:unknown,schema:Schema,root:Schema):boolean {
  const field=resolveSchema(schema,root);
  if(field.const!==undefined)return value===field.const;
  if(field.enum)return field.enum.includes(value);
  if(field.type==="null")return value==null;
  if(value==null)return false;
  if(field.type==="array")return Array.isArray(value);
  if(field.type==="object")return typeof value==="object" && !Array.isArray(value) &&
    Object.entries(field.properties ?? {}).every(([key,property])=>property.const===undefined || (value as Record<string,unknown>)[key]===property.const);
  if(field.type==="integer")return typeof value==="number" && Number.isInteger(value);
  if(field.type==="number")return typeof value==="number";
  return typeof value===field.type;
}

export function schemaDefault(schema:Schema={},root:Schema=schema):unknown {
  const field=resolveSchema(schema,root);
  if(field.default!==undefined)return structuredClone(field.default);
  if(field.const!==undefined)return field.const;
  if(field.enum)return field.enum[0];
  const variants=field.anyOf ?? field.oneOf;
  if(variants)return schemaDefault(variants[0],root);
  if(field.type==="object")return Object.fromEntries(Object.entries(field.properties ?? {})
    .filter(([key,item])=>field.required?.includes(key)||item.default!==undefined||item.const!==undefined)
    .map(([key,item])=>[key,schemaDefault(item,root)]));
  if(field.type==="array")return field.prefixItems?.map(item=>schemaDefault(item,root)) ?? [];
  if(field.type==="boolean")return false;
  if(field.type==="number" || field.type==="integer")return field.minimum ?? (field.exclusiveMinimum!==undefined ? field.exclusiveMinimum+1 : 0);
  if(field.type==="string")return "";
  return null;
}

export function variantLabel(schema:Schema,root:Schema):string {
  const field=resolveSchema(schema,root);
  return field.title ?? (field.const!==undefined ? String(field.const) : field.type==="null" ? "Not set / default" : field.enum ? field.enum.map(String).join(" / ") : field.type ?? "Value");
}
