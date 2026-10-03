import type {Event,RecordValue} from "./types";

export function metricPoints(events:Event[],key:string):{step:number;value:number}[] {
  const points:{step:number;value:number}[]=[];
  for(const event of events){
    const value=(event.payload.values as RecordValue|undefined)?.[key];
    const step=event.payload.step;
    if(typeof value!=="number" || !Number.isFinite(value) || typeof step!=="number" || !Number.isFinite(step))continue;
    const previous=points.at(-1);
    if(previous?.step!==step || previous.value!==value)points.push({step,value});
  }
  return points;
}
