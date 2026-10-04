import {expect,test} from "@playwright/test";
import {openFile,trainingSection} from "./helpers";

test("runtime schema staging, stale target responses, accessible review and native controls",async({page},testInfo)=>{
  const errors:string[]=[];page.on("pageerror",error=>errors.push(error.message));
  const snapshot=(target:string)=>({backend:"sixth-fixture",source:"fixture native inventory",target,namespace:"test",observed_at:new Date(Date.now()-120000).toISOString(),
    complete:false,reachable:true,api_supported:true,submission_allowed:true,visible_nodes:8,matching_nodes:3,
    allocatable:{cpu:8,"shared.example/gpu":4},unallocated:{"shared.example/gpu":0},quota_remaining:null,
    units:{cpu:"cores","shared.example/gpu":"shared resource units"},nodes:[],roles:[],targets:[],options:{},diagnostics:["Pod reads restricted; remaining capacity unknown."]});
  await page.route("**/api/v1/registry",async route=>{
    const response=await route.fetch();const data=await response.json();
    data.execution_backends.push({type:"sixth-fixture",version:"1",label:"Sixth fixture",available:true,diagnostics:[],import_target:"fixture.Remote",
      capabilities:{native_reference:true,metrics:false,cancellation:true},presentation:{target:["target"],resources:["replicas"]},
      schema:{type:"object",properties:{target:{type:"string",default:"initial"},replicas:{type:"integer",default:1,minimum:1}}}});
    data.execution_backends.push({...data.execution_backends.at(-1),type:"generic-fixture",label:"Generic fixture",presentation:{}});
    await route.fulfill({json:data});
  });
  let old:()=>void=()=>{};let oldRequested=false;
  const held=new Promise<void>(resolve=>{old=resolve;});
  await page.route("**/api/v1/execution/discover",async route=>{
    const body=route.request().postDataJSON();const execution=body.data.execution;
    if(execution.type!=="sixth-fixture"){await route.fulfill({json:{...snapshot("local"),backend:"local"}});return;}
    const target=execution.params.target;
    if(target==="older"){oldRequested=true;await held;}
    await route.fulfill({json:{...snapshot(target),submission_allowed:target!=="denied",api_supported:target!=="unsupported"}});
  });
  let prepares=0,launches=0,valid=true,phase="pending";
  let frozen:Record<string,unknown>|null=null;
  await page.route("**/api/v1/train/prepare",async route=>{
    prepares++;const body=route.request().postDataJSON();
    if(!valid){await route.fulfill({status:422,json:{error:{code:"validation",message:"Replicas must be positive.",fields:[{loc:["execution","params","replicas"],message:"Must be positive."}]}}});return;}
    await route.fulfill({json:{data:body.data,stage_order:body.stage_order,yaml:"frozen-fixture",semantic_revision:"reviewed",launch_review:{template_revision:"fixture-revision"}}});
  });
  const native={backend:"sixth-fixture",context:"selected",namespace:"test",api_version:"fixture/v1",kind:"Fixture",name:"owned",uid:"verified-uid"};
  const operation=()=>({id:"f".repeat(32),kind:"train",status:phase,sequence:0,semantic_revision:"reviewed",stage_order:[],created_at:new Date().toISOString(),
    capabilities:{native_reference:true,metrics:false,resume:false,cancellation:true},cancellation:true,native_reference:native,telemetry:"native status only",
    result:{status:phase},artifacts:[{kind:"external checkpoint",path:"s3://owned/checkpoint"}]});
  await page.route("**/api/v1/train",async route=>{launches++;frozen=route.request().postDataJSON();await route.fulfill({status:202,json:operation()});});
  await page.route("**/api/v1/operations",route=>route.fulfill({json:{operations:launches ? [operation()] : []}}));
  await page.route("**/api/v1/operations/*",route=>route.fulfill({json:operation()}));
  await page.route("**/api/v1/operations/*/cancel",route=>{phase="cancelled";return route.fulfill({json:operation()});});
  await page.goto("/");await openFile(page);
  const run=page.getByRole("button",{name:"Run…",exact:true});await expect(run).toBeEnabled({timeout:30000});
  await run.press("Enter");const dialog=page.getByRole("dialog");await expect(dialog).toBeVisible();
  await dialog.locator("summary").filter({hasText:"Definition and raw execution settings"}).click();const initialExecution=await dialog.getByLabel("Raw execution settings",{exact:true}).inputValue();await dialog.getByLabel("Raw execution settings",{exact:true}).fill("null");await dialog.getByRole("button",{name:"Apply Raw execution settings",exact:true}).click();await expect(dialog.locator(".field-error")).toContainText(["buffer is retained"]);await expect(dialog.getByLabel("Execution backend",{exact:true})).toHaveValue("local:1");await dialog.getByLabel("Raw execution settings",{exact:true}).fill(initialExecution);await dialog.locator("summary").filter({hasText:"Definition and raw execution settings"}).click();
  await dialog.getByLabel("Execution backend",{exact:true}).selectOption("generic-fixture:1");await expect(dialog.locator('input[data-field="execution.params.target"]')).toHaveValue("initial");await expect(dialog.locator('input[data-field="execution.params.replicas"]')).toHaveValue("1");
  await dialog.getByLabel("Execution backend",{exact:true}).selectOption("sixth-fixture:1");
  await expect(dialog.locator('[data-field="execution.params.replicas"]')).toHaveValue("1");
  await dialog.locator('[data-field="execution.params.target"]').fill("discarded");
  await dialog.getByRole("button",{name:"Cancel",exact:true}).press("Enter");await expect(run).toBeFocused();
  await run.click();await expect(dialog.getByLabel("Execution backend",{exact:true})).toHaveValue("local:1");
  await dialog.getByLabel("Execution backend",{exact:true}).selectOption("sixth-fixture:1");
  const target=dialog.locator('input[data-field="execution.params.target"]');await target.fill("older");
  await expect.poll(()=>oldRequested).toBe(true);await target.fill("newer");
  await expect(dialog.getByLabel("Sourced execution capacity")).toContainText("newer");old();
  await expect(dialog.getByLabel("Sourced execution capacity")).not.toContainText("older");
  await expect(dialog.getByLabel("Sourced execution capacity")).toContainText("shared.example/gpu: 0");
  await expect(dialog.getByLabel("Sourced execution capacity")).toContainText("Unknown");
  await expect(dialog.getByLabel("Sourced execution capacity")).toContainText("Stale — refresh");
  await target.fill("denied");await expect(dialog.getByText("Submission denied by the selected target.")).toBeVisible();await expect(dialog.getByRole("button",{name:"Review selection",exact:true})).toBeDisabled();
  await target.fill("unsupported");await expect(dialog.getByText("Required native API is unavailable.")).toBeVisible();await expect(dialog.getByRole("button",{name:"Review selection",exact:true})).toBeDisabled();await target.fill("newer");await expect(dialog.getByRole("button",{name:"Review selection",exact:true})).toBeEnabled();
  valid=false;await dialog.getByRole("button",{name:"Review selection",exact:true}).click();
  await expect(dialog.locator(".execution-error")).toBeFocused();
  await dialog.locator(".execution-error").getByRole("button").click();
  const replicas=dialog.locator('input[data-field="execution.params.replicas"]');await expect(replicas).toBeFocused();
  await expect(replicas).toHaveAttribute("aria-describedby");valid=true;
  await replicas.fill("2");await dialog.getByRole("button",{name:"Review selection",exact:true}).click();
  const confirm=dialog.getByRole("button",{name:"Run on Sixth fixture",exact:true});await expect(confirm).toBeEnabled();
  await replicas.fill("3");await expect(confirm).toBeDisabled();
  await dialog.getByRole("button",{name:"Review selection",exact:true}).click();await expect(confirm).toBeEnabled();
  await page.screenshot({path:testInfo.outputPath("run-review-desktop.png"),fullPage:true});
  await page.emulateMedia({reducedMotion:"reduce"});
  await page.setViewportSize({width:390,height:844});
  await confirm.scrollIntoViewIfNeeded();expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
  await page.screenshot({path:testInfo.outputPath("run-review-mobile.png"),fullPage:true});
  await confirm.evaluate(element=>{(element as HTMLButtonElement).click();(element as HTMLButtonElement).click();});await expect(page.getByLabel("Observed operation")).toHaveValue("f".repeat(32));
  expect(launches).toBe(1);expect(prepares).toBe(3);
  expect((frozen as unknown as {data:{execution:{params:{replicas:number}}}}).data.execution.params.replicas).toBe(3);
  await expect(page.locator(".status-grid")).toContainText("pending");await expect(page.getByRole("progressbar")).toHaveCount(0);
  phase="running";await expect(page.locator(".status-grid")).toContainText("running");
  await expect(page.getByText("Trainer checkpoint resume is unsupported by this backend.",{exact:false})).toBeVisible();
  await expect(page.getByRole("button",{name:"Logs",exact:true})).toHaveAttribute("aria-current","page");
  await page.screenshot({path:testInfo.outputPath("runs-unsupported-mobile.png"),fullPage:true});
  await page.getByRole("navigation",{name:"Selected run views"}).getByRole("button",{name:"Artifacts",exact:true}).click();
  await expect(page.getByRole("button",{name:"Export trained checkpoint",exact:true})).toBeDisabled();
  await expect(page.getByRole("button",{name:"Download s3://owned/checkpoint",exact:true})).toHaveCount(0);
  await page.getByRole("button",{name:"Stop verified native job",exact:true}).click();await expect(page.locator(".status-grid")).toContainText("cancelled");
  await page.getByRole("button",{name:"Training",exact:true}).click();await trainingSection(page,"Execution");await expect(page.getByLabel("Execution backend",{exact:true})).toHaveValue("sixth-fixture:1");
  await page.getByRole("button",{name:"Undo",exact:true}).click();await expect(page.getByLabel("Execution backend",{exact:true})).toHaveValue("local:1");
  expect(errors).toEqual([]);
});
