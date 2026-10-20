import {expect,type Page,type Request} from "@playwright/test";

export async function apiAction(page:Page,path:string,action:()=>Promise<unknown>,{method="POST",status=200,match=()=>true}:{method?:string;status?:number;match?:(request:Request)=>boolean}={}) {
  // Selected-interpreter calls can outlast locator assertions; wait for this response's body, not network idle.
  const [response]=await Promise.all([page.waitForResponse(async response=>{
    if(new URL(response.url()).pathname!==`/api/v1${path}` || response.request().method()!==method || !match(response.request()))return false;
    await response.finished();return true;
  },{timeout:30000}),action()]);
  expect(response.status(),`${method} /api/v1${path}`).toBe(status);
  return response;
}

async function workbenchIdle(page:Page) {
  await expect(page.locator('.diagnostics > [role="status"]')).not.toHaveText("Waiting for selected NexuML…");
}

export async function loadStudio(page:Page,navigate:()=>Promise<unknown>=()=>page.goto("/")) {
  await apiAction(page,"/operations",()=>apiAction(page,"/registry",navigate,{method:"GET"}),{method:"GET"});
  await expect(page.locator(".workbench")).toBeVisible();
}

export async function fileAction(page:Page,action:"Open YAML"|"Save as…") {
  if(await page.locator(".welcome").isVisible() && action==="Open YAML")await page.locator(".welcome").getByRole("button",{name:action,exact:true}).click();
  else {await page.getByLabel("Scenario menu").click();await page.locator(".scenario-menu").getByRole("button",{name:action,exact:true}).click();}
}

export async function openFile(page:Page,path=process.env.STUDIO_CONFIG_PATH ?? "tiny.yaml") {
  await fileAction(page,"Open YAML");
  await page.getByLabel("Config path",{exact:true}).fill(path);
  await apiAction(page,"/config/layout/load",()=>apiAction(page,"/config/load",()=>page.getByRole("button",{name:"Open file",exact:true}).click()));
  await workbenchIdle(page);
  await expect(page.getByRole("dialog")).toBeHidden();
  await expect(page.locator(".pipeline-view")).toBeVisible();
}

export async function save(page:Page,action:()=>Promise<unknown>=()=>page.getByRole("button",{name:"Save file",exact:true}).click()) {
  await apiAction(page,"/config/layout/save",()=>apiAction(page,"/config/save",action));
  await workbenchIdle(page);
  await expect(page.getByRole("dialog")).toBeHidden();
}

export async function applyYaml(page:Page,status=200) {
  await apiAction(page,"/config/validate",()=>page.getByRole("button",{name:"Apply YAML",exact:true}).click(),{status});
  await workbenchIdle(page);
}

export async function check(page:Page,name:"Check fields"|"Build model"="Check fields",status=name==="Build model" ? 202 : 200) {
  await page.locator(".check-menu summary").click();
  await apiAction(page,name==="Build model" ? "/build" : "/config/validate",()=>page.getByRole("button",{name,exact:true}).click(),{status});
  await workbenchIdle(page);
}

export async function openRun(page:Page,action:()=>Promise<unknown>=()=>page.getByRole("button",{name:"Run…",exact:true}).click()) {
  await apiAction(page,"/execution/discover",()=>apiAction(page,"/config/validate",action));
  await workbenchIdle(page);
  await expect(page.getByRole("dialog",{name:"Run your scenario",exact:true})).toBeVisible();
  await expect(page.getByLabel("Sourced execution capacity").getByRole("status")).toHaveText("Advisory snapshot — not reserved");
}

export async function reviewRun(page:Page,status=200) {
  const dialog=page.getByRole("dialog",{name:"Run your scenario",exact:true});
  await apiAction(page,"/train/prepare",()=>dialog.getByRole("button",{name:"Review selection",exact:true}).click(),{status});
  await expect(dialog.locator('.result-summary [role="status"]')).not.toHaveText("Validating selected NexuML…");
}

export async function observe(page:Page,identity:string,status=200) {
  await apiAction(page,`/operations/${identity}/config`,()=>apiAction(page,`/operations/${identity}`,()=>page.getByLabel("Observed operation").selectOption(identity),{method:"GET"}),{method:"GET",status});
}

export async function navigation(page:Page,tab:"Structure"|"Components") {
  await page.locator(".components-panel .section-title").getByRole("button",{name:tab,exact:true}).click();
}

export async function inspector(page:Page,tab:"Settings"|"Routing") {
  await page.locator(".inspector-tabs").getByRole("button",{name:tab,exact:true}).click();
}

export async function trainingSection(page:Page,section:string) {
  if(await page.getByLabel("Training section",{exact:true}).isVisible())await page.getByLabel("Training section",{exact:true}).selectOption(section);
  else await page.getByRole("navigation",{name:"Training sections"}).getByRole("button",{name:section,exact:true}).click();
}
