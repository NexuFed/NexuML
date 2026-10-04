import {expect,type Page} from "@playwright/test";

export async function fileAction(page:Page,action:"Open YAML"|"Save as…") {
  if(await page.locator(".welcome").isVisible() && action==="Open YAML")await page.locator(".welcome").getByRole("button",{name:action,exact:true}).click();
  else {await page.getByLabel("Scenario menu").click();await page.locator(".scenario-menu").getByRole("button",{name:action,exact:true}).click();}
}

export async function openFile(page:Page,path=process.env.STUDIO_CONFIG_PATH ?? "tiny.yaml") {
  await fileAction(page,"Open YAML");
  await page.getByLabel("Config path",{exact:true}).fill(path);
  await page.getByRole("button",{name:"Open file",exact:true}).click();
  await expect(page.getByRole("dialog")).toBeHidden({timeout:30000});
  await expect(page.locator(".pipeline-view")).toBeVisible();
}

export async function check(page:Page,name="Check fields") {
  await page.locator(".check-menu summary").click();
  await page.getByRole("button",{name,exact:true}).click();
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
