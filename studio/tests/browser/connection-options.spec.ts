import {expect,test} from "@playwright/test";
import {parse,stringify} from "yaml";
import type {Config} from "../../model/types";
import {applyYaml,inspector,loadStudio,openFile} from "./helpers";

test("routing options retain only the applicable preceding writer for each key",async({page})=>{
  const failures:string[]=[];page.on("pageerror",error=>failures.push(error.message));
  await loadStudio(page);await openFile(page);
  await page.getByRole("button",{name:"YAML",exact:true}).click();
  const config=parse(await page.getByLabel("Scenario YAML").inputValue()) as Config;
  const component=config.pipeline.stages.Encoder[0].component;
  config.data.input_shapes={features:[8],other:[8]};config.data.skip_pipeline_stages=[];
  config.pipeline.stages={Routes:[
    {component,keys_in:["features"],keys_out:["features","first_only"],meta_out:{width:"latent_width"}},
    {component,keys_in:["features"],keys_out:["features","second_only"]},
    {component,keys_in:{input:"features"},keys_out:["final"]},
  ]};
  config.training.loss_keys={final:1};config.training.metric_keys=[];config.evaluation.algorithms=[];
  await page.getByLabel("Scenario YAML").fill(stringify(config));await applyYaml(page);
  await page.getByRole("button",{name:"Close YAML",exact:true}).click();
  const outline=page.locator(".outline");
  const placeholder="Choose preceding producer…";
  await outline.getByRole("button",{name:`1. ${component.type}`,exact:true}).click();await inspector(page,"Routing");
  const first=await page.locator(".react-flow__node.selected").getAttribute("data-id");
  await expect(page.getByLabel("Connect features",{exact:true}).locator("option")).toHaveText([
    placeholder,"Data → features (x)","Data → other (x)",
  ]);
  await outline.getByRole("button",{name:`2. ${component.type}`,exact:true}).click();
  const second=await page.locator(".react-flow__node.selected").getAttribute("data-id");
  await expect(page.getByLabel("Connect features",{exact:true}).locator("option")).toHaveText([
    placeholder,"Data → other (x)",`${component.type} → features (x)`,`${component.type} → first_only (x)`,
  ]);
  await outline.getByRole("button",{name:`3. ${component.type}`,exact:true}).click();
  const expected=[placeholder,"Data → other (x)",`${component.type} → first_only (x)`,`${component.type} → features (x)`,`${component.type} → second_only (x)`];
  const route=page.getByLabel("Connect features",{exact:true});
  await expect(route.locator("option")).toHaveText(expected);
  expect(await route.locator("option").evaluateAll(options=>options.slice(1).map(option=>JSON.parse((option as HTMLOptionElement).value)))).toEqual([
    ["data","output:other"],[first,"output:first_only"],[second,"output:features"],[second,"output:second_only"],
  ]);
  await route.selectOption({label:`${component.type} → first_only (x)`});
  await expect(page.locator('input[data-field="pipeline.stages.Routes.2.keys_in.input"]')).toHaveValue("first_only");
  await expect(page.getByLabel("Connect first_only",{exact:true}).locator("option")).toHaveText(expected);
  await page.getByRole("button",{name:"Undo",exact:true}).click();
  await expect(page.locator('input[data-field="pipeline.stages.Routes.2.keys_in.input"]')).toHaveValue("features");
  await expect(route.locator("option")).toHaveText(expected);
  expect(failures).toEqual([]);
});
