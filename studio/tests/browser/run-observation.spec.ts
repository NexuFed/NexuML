import {expect,test} from "@playwright/test";
import {openFile} from "./helpers";

test("frozen semantic comparison, shared loss coordinates and retained observations",async({page})=>{
  let launches=0;page.on("request",request=>{if(request.method()==="POST" && /\/api\/v1\/(train|build|export)$/.test(new URL(request.url()).pathname))launches++;});
  const id="e".repeat(32);let unavailable=false;
  await page.goto("/");
  const document=await page.evaluate(async()=>{const connection=await(await fetch("/api/runtime")).json();return (await fetch(`${connection.api}/api/v1/config/load`,{method:"POST",headers:{Authorization:`Bearer ${connection.token}`,"Content-Type":"application/json"},body:JSON.stringify({path:"tiny.yaml"})})).json();});
  const operation={id,kind:"train",status:"succeeded",sequence:0,semantic_revision:document.semantic_revision,telemetry:"fixture callbacks",cancellation:false,capabilities:{metrics:true,resume:true},result:{},artifacts:[]};
  await page.route("**/api/v1/operations",route=>route.fulfill({json:{operations:[operation]}}));
  await page.route(`**/api/v1/operations/${id}`,route=>route.fulfill({json:operation}));
  await page.route(`**/api/v1/operations/${id}/config`,route=>route.fulfill(unavailable ? {status:404,json:{error:{message:"Frozen source unavailable"}}} : {json:{...document,data:Object.fromEntries(Object.entries(document.data).reverse())}}));
  let subscriptions=0;
  await page.routeWebSocket(`**/api/v1/operations/${id}/events*`,socket=>{
    subscriptions++;socket.onMessage(()=>{
      socket.send(JSON.stringify({kind:"snapshot",payload:operation}));
      for(const [sequence,step,values] of [[1,2,{"train/loss":10,"accuracy":.5,"custom/scalar":100}],[2,7,{"train/loss":6}],[3,12,{"val/loss":2}]] as const)socket.send(JSON.stringify({kind:"metrics",sequence,payload:{step,values}}));
      socket.send(JSON.stringify({kind:"log",sequence:4,payload:{text:"Retained process output\n"}}));
      socket.send(JSON.stringify({kind:"replay_gap",sequence:5,payload:{}}));
      socket.close({code:1008,reason:"Fixture disconnected"});
    });
  });
  await page.reload();await page.getByRole("button",{name:"Runs",exact:true}).click();await page.getByLabel("Observed operation").selectOption(id);
  await expect(page.locator(".status-grid")).toContainText("reload to reconnect");await expect(page.getByText("Replay history is incomplete.",{exact:false})).toBeVisible();
  await expect(page.getByText("Draft matches",{exact:false})).toHaveCount(0);
  const total=page.locator(".metric").filter({has:page.locator("span",{hasText:/^Total loss$/})});
  await expect(total).toContainText("train/loss");await expect(total).toContainText("val/loss");await expect(total).toContainText("single measurement");
  expect(await total.locator(".training-series polyline").getAttribute("points")).toBe("55,20 217.5,55");
  expect(await total.locator(".validation-series circle").getAttribute("cx")).toBe("380");expect(await total.locator(".validation-series circle").getAttribute("cy")).toBe("90");
  await expect(total.locator(".validation-series polyline")).toHaveAttribute("stroke-dasharray","6 4");await expect(total).toContainText("Optimizer step");
  await expect(page.locator(".metric")).toHaveCount(3);await expect(page.locator(".metric").filter({hasText:"accuracy"})).toContainText("0.50000");await expect(page.locator(".metric").filter({hasText:"custom/scalar"})).toContainText("100.00");
  await page.getByRole("button",{name:"Logs",exact:true}).click();await expect(page.getByLabel("Process logs")).toContainText("Retained process output");await page.getByRole("button",{name:"Artifacts",exact:true}).click();await page.getByRole("button",{name:"Metrics",exact:true}).click();expect(subscriptions).toBe(1);
  await page.getByRole("button",{name:"Pipeline",exact:true}).click();await openFile(page);await page.getByRole("button",{name:"Runs",exact:true}).click();await expect(page.getByText("Draft matches",{exact:false})).toBeVisible();
  await page.getByRole("button",{name:"Pipeline",exact:true}).click();await page.getByRole("button",{name:"Move stage Encoder down",exact:true}).click();await page.getByRole("button",{name:"Runs",exact:true}).click();await expect(page.getByText("Draft differs",{exact:false})).toBeVisible();
  await page.getByRole("button",{name:"Configuration",exact:true}).click();await expect(page.getByLabel("Frozen launch YAML")).toContainText("Encoder:");
  page.once("dialog",dialog=>void dialog.dismiss());await page.getByRole("button",{name:"Open frozen settings as draft",exact:true}).click();await expect(page.locator(".execution-view")).toBeVisible();
  page.once("dialog",dialog=>void dialog.accept());await page.getByRole("button",{name:"Open frozen settings as draft",exact:true}).click();await expect(page.locator(".training-view")).toBeVisible();await page.getByRole("button",{name:"Runs",exact:true}).click();await expect(page.getByText("Draft matches",{exact:false})).toBeVisible();
  unavailable=true;await page.reload();await page.getByRole("button",{name:"Runs",exact:true}).click();await page.getByLabel("Observed operation").selectOption(id);await page.getByRole("button",{name:"Configuration",exact:true}).click();await expect(page.getByText("Frozen source unavailable",{exact:false})).toBeVisible();await expect(page.getByText(/Draft (matches|differs)/)).toHaveCount(0);
  expect(launches).toBe(0);
});
