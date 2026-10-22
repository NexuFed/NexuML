import {expect,test} from "@playwright/test";
import {applyYaml,loadStudio,navigation,openFile} from "./helpers";

for(const identity of [{interface_version:2},{interface_version:4},{interface_version:undefined}]){
  test(`browser rejects catalog interface ${identity.interface_version ?? "missing"} before discovery`,async({page})=>{
    const catalogs:string[]=[];
    page.on("request",request=>{if(new URL(request.url()).pathname==="/api/v1/registry")catalogs.push(request.url());});
    await page.route("**/api/v1/runtime",async route=>{
      const response=await route.fetch();await route.fulfill({response,json:{...await response.json(),...identity}});
    });
    await page.goto("/");
    await expect(page.locator(".startup")).toContainText("Studio requires");
    await expect(page.locator(".workbench")).toHaveCount(0);
    expect(catalogs).toEqual([]);
  });
}

test("missing native catalog contract fails clearly without exposing editable views",async({page})=>{
  const errors:string[]=[];page.on("pageerror",error=>errors.push(error.message));
  await page.route("**/api/v1/registry",async route=>{
    const response=await route.fetch(),catalog=await response.json();delete catalog.components[0].io;
    await route.fulfill({response,json:catalog});
  });
  await page.goto("/");
  await expect(page.locator(".startup")).toContainText(/native.*catalog|catalog.*native/i);
  await expect(page.locator(".workbench")).toHaveCount(0);expect(errors).toEqual([]);
});

test("native fixed choices and source label ports need no runtime build",async({page})=>{
  const builds:string[]=[];page.on("request",request=>{if(/\/api\/v1\/(build|train)$/.test(new URL(request.url()).pathname))builds.push(request.url());});
  await loadStudio(page);await openFile(page);await navigation(page,"Components");
  await page.getByLabel("Search components").fill("latent_visualizer");
  await page.locator(".component-list").getByRole("button",{name:/latent_visualizer/}).click();
  await page.locator(".properties-panel .component-fields>.fields>details>summary").getByText("Advanced",{exact:true}).click();
  await expect(page.locator(".properties-panel").getByLabel("Method",{exact:true})).toHaveValue("tsne");
  await page.locator(".properties-panel").getByLabel("Method",{exact:true}).selectOption("umap");
  await page.getByRole("button",{name:"YAML",exact:true}).click();
  const valid=await page.getByLabel("Scenario YAML").inputValue();expect(valid).toContain("method: umap");
  const invalid=valid.replace("method: umap","method: unsupported");
  await page.getByLabel("Scenario YAML").fill(invalid);await applyYaml(page,422);
  await expect(page.getByLabel("Scenario YAML")).toHaveValue(invalid);
  await expect(page.locator(".problems-list")).toContainText("method");
  await page.getByLabel("Scenario YAML").fill(valid);await applyYaml(page);
  await page.getByRole("button",{name:"Close YAML",exact:true}).click();
  await page.getByLabel("Search components").fill("CIFAR10Dataset");
  await page.locator(".component-list").getByRole("button",{name:/CIFAR10Dataset/}).click();
  await expect(page.locator('.react-flow__node[data-id="data"]')).toContainText("class_labels");
  expect(builds).toEqual([]);
});
