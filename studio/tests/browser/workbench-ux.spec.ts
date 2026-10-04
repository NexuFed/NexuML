import {expect,test} from "@playwright/test";
import {check,fileAction,navigation,openFile,trainingSection} from "./helpers";

test("source menu, first recipe save, retained YAML and explicit checks",async({page})=>{
  await page.goto("/");
  await page.locator(".welcome").getByRole("button",{name:"Choose a recipe",exact:true}).click();
  await page.getByLabel("Discovered scenario").selectOption("synthetic-linear-ae-reconstruction");
  await page.getByRole("button",{name:"Resolve recipe",exact:true}).click();
  await expect(page.getByRole("dialog")).toBeHidden({timeout:30000});
  await expect(page.locator(".save-state")).toHaveText("Unsaved recipe");
  await page.locator(".top-actions").getByRole("button",{name:"Save",exact:true}).click();
  await expect(page.getByRole("dialog")).toBeVisible();
  const saved=`ux-recipe-${Date.now()}.yaml`;
  await page.getByLabel("Config path",{exact:true}).fill(saved);
  await page.getByRole("button",{name:"Save file",exact:true}).click();
  await expect(page.locator(".save-state")).toHaveText("Saved",{timeout:30000});
  await openFile(page);
  await page.getByRole("button",{name:"1. LinearEncoder",exact:true}).first().click();
  await page.locator(".properties-panel summary").filter({hasText:"Edit display name"}).click();
  await page.getByLabel("Node display name").fill("Sidecar label");
  await fileAction(page,"Save as…");
  const path=`ux-file-${Date.now()}.yaml`;
  await page.getByLabel("Config path",{exact:true}).fill(path);
  await page.getByRole("button",{name:"Save file",exact:true}).click();
  await expect(page.getByRole("dialog")).toBeHidden();
  await openFile(page,path);
  await expect(page.locator(".react-flow__node")).toContainText(["Sidecar label"]);
  await page.getByRole("button",{name:"YAML",exact:true}).click();
  const yaml=await page.getByLabel("Scenario YAML").inputValue();
  await page.getByLabel("Scenario YAML").fill("invalid: [yaml");
  await page.getByRole("button",{name:"Apply YAML",exact:true}).click();
  await expect(page.getByRole("status").first()).not.toContainText("Waiting");
  await expect(page.getByRole("button",{name:/Problems \(/})).toHaveAttribute("aria-expanded","true");
  await page.getByRole("button",{name:"Close YAML",exact:true}).click();
  await expect(page.locator(".buffer-notice")).toBeVisible();
  await expect(page.locator(".top-actions").getByRole("button",{name:"Save",exact:true})).toBeDisabled();
  await page.getByRole("button",{name:"Runs",exact:true}).click();
  await expect(page.locator(".execution-view")).toBeVisible();
  await page.getByRole("button",{name:"Return to YAML",exact:true}).click();
  await expect(page.getByLabel("Scenario YAML")).toHaveValue("invalid: [yaml");
  await page.getByLabel("Scenario YAML").fill(yaml);
  await page.getByRole("button",{name:"Apply YAML",exact:true}).click();
  await expect(page.getByRole("status").first()).toContainText("YAML applied");
  await page.getByRole("button",{name:"Close YAML",exact:true}).click();
  await check(page);
  await expect(page.getByRole("status").first()).toContainText("Fields valid");
  await expect(page.locator(".diagnostic-heading")).toContainText("Not checked");
  await check(page,"Build model");
  await expect(page.locator(".diagnostic-heading")).toContainText("Passed",{timeout:30000});
  await page.getByRole("button",{name:/Problems \(/}).click();
  await expect(page.locator(".build-state")).toContainText("reconstructed");
  await page.getByRole("button",{name:"1. LinearEncoder",exact:true}).first().click();
  await page.getByLabel("Output Dim",{exact:true}).fill("4");
  await expect(page.locator(".diagnostic-heading")).toContainText("Needs rechecking");
  await fileAction(page,"Open YAML");
  page.once("dialog",dialog=>void dialog.dismiss());
  await page.getByRole("button",{name:"Open file",exact:true}).click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await page.getByRole("button",{name:"Cancel",exact:true}).click();
  await expect(page.getByLabel("Output Dim",{exact:true})).toHaveValue("4");
  await page.route("**/api/v1/config/save",route=>route.fulfill({status:409,json:{error:{code:"conflict",message:"File changed externally. Reload or save to a new path."}}}));
  await page.locator(".top-actions").getByRole("button",{name:"Save",exact:true}).click();
  await expect(page.getByRole("status").first()).toContainText("File changed externally");
  await expect(page.locator(".save-state")).toHaveText("Unsaved changes");
  await page.unroute("**/api/v1/config/save");
  await page.route("**/api/v1/config/layout/save",route=>route.fulfill({status:409,json:{error:{code:"conflict",message:"Layout changed externally. Reload or save to a new path."}}}));
  await page.locator(".top-actions").getByRole("button",{name:"Save",exact:true}).click();
  await expect(page.getByRole("status").first()).toContainText("Layout changed externally");await expect(page.locator(".save-state")).toHaveText("Unsaved changes");
  await page.unroute("**/api/v1/config/layout/save");
  await page.locator(".top-actions").getByRole("button",{name:"Save",exact:true}).click();await expect(page.locator(".save-state")).toHaveText("Saved",{timeout:30000});
});

test("documented panel/keyboard paths and responsive visual acceptance",async({page},testInfo)=>{
  const submissions:string[]=[];page.on("request",request=>{if(request.method()==="POST" && /\/api\/v1\/(train|build|export)$/.test(new URL(request.url()).pathname))submissions.push(request.url());});
  await page.goto("/");await openFile(page);
  await expect(page.locator('.react-flow__node[data-id="data"]')).toHaveCSS("width","250px");
  await expect.poll(()=>page.locator('.react-flow__node[data-id="data"]').evaluate(node=>node.getBoundingClientRect().width)).toBeGreaterThan(240);
  await navigation(page,"Components");await page.getByLabel("Search components").fill("LinearEncoder");await navigation(page,"Structure");await navigation(page,"Components");await expect(page.getByLabel("Search components")).toHaveValue("LinearEncoder");
  await expect(page.getByLabel("Insert into stage",{exact:true})).toHaveValue("");
  await page.locator(".component-list button").filter({hasText:"LinearEncoder"}).first().press("Enter");await expect(page.getByRole("dialog")).toContainText("Choose layer destination");
  await expect(page.getByRole("button",{name:"Insert layer",exact:true})).toBeDisabled();await page.getByRole("button",{name:"Cancel",exact:true}).click();
  await navigation(page,"Structure");await page.locator(".outline").getByRole("button",{name:"1. LinearEncoder",exact:true}).nth(1).press("Enter");await page.getByRole("button",{name:"Focus selected",exact:true}).press("Enter");
  await expect.poll(()=>page.locator('.react-flow__node.selected').evaluate(node=>node.getBoundingClientRect().width)).toBeGreaterThan(240);
  const contrast=await page.evaluate(()=>{
    const luminance=(color:string)=>{const channels=color.match(/[\d.]+/g)!.slice(0,3).map(value=>{const channel=Number(value)/255;return channel<=.04045 ? channel/12.92 : ((channel+.055)/1.055)**2.4;});return .2126*channels[0]+.7152*channels[1]+.0722*channels[2];};
    const ratio=(a:string,b:string)=>{const values=[luminance(a),luminance(b)].sort((x,y)=>y-x);return (values[0]+.05)/(values[1]+.05);};
    const action=getComputedStyle(document.querySelector('[data-run-trigger]')!),input=getComputedStyle(document.querySelector('.properties-panel input[data-field$=".output_dim"]')!),panel=getComputedStyle(document.querySelector('.properties-panel')!);
    return {action:ratio(action.color,action.backgroundColor),text:ratio(input.color,input.backgroundColor),border:ratio(input.borderTopColor,input.backgroundColor),borderOnPanel:ratio(input.borderTopColor,panel.backgroundColor),secondary:ratio(getComputedStyle(document.querySelector('.properties-panel .muted')!).color,panel.backgroundColor)};
  });
  expect(contrast.action).toBeGreaterThanOrEqual(4.5);expect(contrast.text).toBeGreaterThanOrEqual(4.5);expect(contrast.secondary).toBeGreaterThanOrEqual(4.5);expect(contrast.border).toBeGreaterThanOrEqual(3);expect(contrast.borderOnPanel).toBeGreaterThanOrEqual(3);
  await page.locator(".properties-panel summary").filter({hasText:"Advanced layout"}).click();await page.locator(".properties-panel summary").filter({hasText:"Visual position"}).click();await page.getByLabel("Node position x",{exact:true}).fill("42");await page.getByRole("button",{name:"Arrange",exact:true}).press("Enter");await expect(page.getByLabel("Node position x",{exact:true})).toHaveValue("20");await page.getByRole("button",{name:"Undo",exact:true}).click();await expect(page.getByLabel("Node position x",{exact:true})).toHaveValue("42");await page.getByRole("button",{name:"Focus selected",exact:true}).click();
  for(const width of [1440,1280]){
    await page.setViewportSize({width,height:width===1280 ? 800 : 1000});
    await expect.poll(()=>page.locator(".canvas").evaluate(element=>element.getBoundingClientRect().height)).toBeGreaterThan(480);
    const separator=page.getByRole("separator",{name:"Navigation panel width"});await separator.focus();await separator.press("End");await expect(separator).toHaveAttribute("aria-valuenow","380");await separator.press("Home");await expect(separator).toHaveAttribute("aria-valuenow","220");
    expect(await page.locator(".components-panel").evaluate(panel=>{const bounds=panel.getBoundingClientRect();return [...panel.querySelectorAll(".section-title button,.outline button")].every(button=>{const rect=button.getBoundingClientRect();return rect.left>=bounds.left && rect.right<=bounds.right && button.scrollWidth<=button.clientWidth;});})).toBe(true);
    await page.getByRole("button",{name:"Collapse navigation panel",exact:true}).press("Enter");await expect(page.locator(".panel-settings summary")).toBeFocused();await page.locator(".panel-settings summary").press("Enter");await page.getByRole("button",{name:"Component panel",exact:true}).press("Enter");await page.locator(".panel-settings summary").press("Enter");
    await page.getByRole("separator",{name:"Property panel width"}).press("ArrowLeft");
    await page.screenshot({path:testInfo.outputPath(`pipeline-${width}.png`)});
  }
  const navigationWidth=page.getByRole("separator",{name:"Navigation panel width"}),propertyWidth=page.getByRole("separator",{name:"Property panel width"});await navigationWidth.press("Home");await propertyWidth.press("Home");for(let index=0;index<3;index++){await navigationWidth.press("ArrowRight");await propertyWidth.press("ArrowLeft");}
  for(const [width,height] of [[1440,1000],[1024,900],[390,844]]){
    await page.setViewportSize({width,height});await page.getByRole("button",{name:"Pipeline",exact:true}).click();
    if(width<1280)await page.getByRole("button",{name:"Canvas",exact:true}).click();
    await page.screenshot({path:testInfo.outputPath(`pipeline-normal-${width}.png`)});
    await page.getByRole("button",{name:"Training",exact:true}).click();await trainingSection(page,"Basics");await page.screenshot({path:testInfo.outputPath(`training-${width}.png`)});
    await page.getByRole("button",{name:"Runs",exact:true}).click();await page.screenshot({path:testInfo.outputPath(`runs-empty-${width}.png`)});
    const retained=await page.getByLabel("Observed operation").locator("option").evaluateAll(options=>options.map(option=>({value:(option as HTMLOptionElement).value,text:option.textContent})).filter(option=>option.text?.startsWith("train") && option.text.includes("succeeded")));
    if(retained.length){await page.getByLabel("Observed operation").selectOption(retained.at(-1)!.value);await expect(page.locator(".status-grid")).toContainText("succeeded");await expect(page.locator(".metric").first()).toBeVisible();await page.screenshot({path:testInfo.outputPath(`runs-${width}.png`)});if(width<1280){await page.locator(".metric").first().scrollIntoViewIfNeeded();await page.screenshot({path:testInfo.outputPath(`runs-metrics-${width}.png`)});await page.locator(".runs-view").evaluate(panel=>{panel.scrollTop=0;});}await page.getByLabel("Observed operation").selectOption("");}
    expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
    if(width<1280){const sizes=await page.locator('.topbar button:visible,.nav-tabs button:visible,.topbar summary:visible,.run-tabs button:visible').evaluateAll(elements=>elements.map(element=>({label:element.textContent,height:element.getBoundingClientRect().height,width:element.getBoundingClientRect().width})));expect(sizes.filter(item=>item.height<44 || item.width<44)).toEqual([]);const boxes=await page.locator(".top-actions>*").evaluateAll(elements=>elements.map(element=>({left:element.getBoundingClientRect().left,right:element.getBoundingClientRect().right})));expect(boxes.slice(1).every((box,index)=>box.left>=boxes[index].right)).toBe(true);}
  }
  await page.setViewportSize({width:1440,height:1000});await page.getByRole("button",{name:"Pipeline",exact:true}).click();await page.getByLabel("Output Dim",{exact:true}).fill(".5");await check(page);await expect(page.locator(".problems-list")).toContainText("output_dim");await page.screenshot({path:testInfo.outputPath("pipeline-error-1440.png")});
  await page.emulateMedia({reducedMotion:"reduce"});expect(await page.evaluate(()=>matchMedia("(prefers-reduced-motion: reduce)").matches)).toBe(true);
  expect(submissions).toEqual([]);
});
