import {expect,test} from "vitest";
import {metricPoints} from "../model/metrics";
import type {Event} from "../model/types";

test("metric points retain actual steps and scalars without stale duplicate samples or invented zeroes",()=>{
  const sample=(step:number,values:Record<string,unknown>):Event=>({kind:"metrics",payload:{step,values}});
  expect(metricPoints([
    sample(1,{"train/loss":2}),sample(1,{"train/loss":2}),
    sample(2,{"val/loss":3}),sample(3,{"train/loss":null}),
    sample(4,{"train/loss":"1"}),sample(5,{"train/loss":Infinity}),
    sample(10,{"train/loss":0.5}),sample(10,{"train/loss":0.4}),
  ],"train/loss")).toEqual([{step:1,value:2},{step:10,value:0.5},{step:10,value:0.4}]);
});
