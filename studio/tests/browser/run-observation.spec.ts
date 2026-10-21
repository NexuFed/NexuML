import {expect,test} from "@playwright/test";
import {apiAction,loadStudio,observe,openFile} from "./helpers";

test("frozen semantic comparison, shared loss coordinates and retained observations",async({page})=>{
  let launches=0;page.on("request",request=>{if(request.method()==="POST" && /\/api\/v1\/(train|build|export)$/.test(new URL(request.url()).pathname))launches++;});
  const id="e".repeat(32);let unavailable=false;
  await loadStudio(page);
  const loaded=await apiAction(page,"/config/load",()=>page.evaluate(async()=>{const connection=await(await fetch("/api/runtime")).json();await fetch(`${connection.api}/api/v1/config/load`,{method:"POST",headers:{Authorization:`Bearer ${connection.token}`,"Content-Type":"application/json"},body:JSON.stringify({path:"tiny.yaml"})});}));
  const document=await loaded.json();
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
  await loadStudio(page,()=>page.reload());await page.getByRole("button",{name:"Runs",exact:true}).click();await observe(page,id);
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
  page.once("dialog",dialog=>void dialog.accept());await apiAction(page,`/operations/${id}/config`,()=>page.getByRole("button",{name:"Open frozen settings as draft",exact:true}).click(),{method:"GET"});await expect(page.locator(".training-view")).toBeVisible();await page.getByRole("button",{name:"Runs",exact:true}).click();await expect(page.getByText("Draft matches",{exact:false})).toBeVisible();
  unavailable=true;await loadStudio(page,()=>page.reload());await page.getByRole("button",{name:"Runs",exact:true}).click();await observe(page,id,404);await page.getByRole("button",{name:"Configuration",exact:true}).click();await expect(page.getByText("Frozen source unavailable",{exact:false})).toBeVisible();await expect(page.getByText(/Draft (matches|differs)/)).toHaveCount(0);
  expect(launches).toBe(0);
});

test("bounded observations retain replay cursors, independent metrics/logs and terminal completion",async({page})=>{
  const id="f".repeat(32);let complete=false;let sendMore=()=>{};
  const cursors:number[]=[];
  await loadStudio(page);
  const loaded=await apiAction(page,"/config/load",()=>page.evaluate(async()=>{const connection=await(await fetch("/api/runtime")).json();await fetch(`${connection.api}/api/v1/config/load`,{method:"POST",headers:{Authorization:`Bearer ${connection.token}`,"Content-Type":"application/json"},body:JSON.stringify({path:"tiny.yaml"})});}));
  const document=await loaded.json();
  const operation={id,kind:"train",status:"running",sequence:0,semantic_revision:document.semantic_revision,telemetry:"fixture callbacks",cancellation:false,capabilities:{metrics:true,resume:true},result:{},artifacts:[]};
  await page.route("**/api/v1/operations",route=>route.fulfill({json:{operations:[operation]}}));
  await page.route(`**/api/v1/operations/${id}`,route=>route.fulfill({json:{...operation,status:complete ? "succeeded" : "running"}}));
  await page.route(`**/api/v1/operations/${id}/config`,route=>route.fulfill({json:document}));
  await page.routeWebSocket(`**/api/v1/operations/${id}/events*`,socket=>{
    cursors.push(Number(new URL(socket.url()).searchParams.get("after")));
    socket.onMessage(()=>{
      socket.send(JSON.stringify({kind:"snapshot",payload:operation}));
      if(cursors.length===1){
        for(let sequence=1;sequence<=1999;sequence++)socket.send(JSON.stringify({kind:"metrics",sequence,payload:{step:sequence,values:sequence===1 ? {"old-only":1} : {"train/loss":sequence}}}));
        sendMore=()=>{
          for(const sequence of [2000,2001])socket.send(JSON.stringify({kind:"metrics",sequence,payload:{step:sequence,values:{"train/loss":sequence}}}));
          socket.send(JSON.stringify({kind:"metrics",sequence:2001,payload:{step:2001,values:{"duplicate-only":999}}}));
          socket.send(JSON.stringify({kind:"log",sequence:2002,payload:{text:Array.from({length:1100},(_,index)=>`line ${index}\n`).join("")}}));
          socket.send(JSON.stringify({kind:"progress",sequence:2003,payload:{phase:"train",batch:8,total:10}}));
          socket.send(JSON.stringify({kind:"replay_gap",sequence:2004,payload:{}}));
          socket.close({code:1012,reason:"Reconnect fixture"});
        };
      }else{
        socket.send(JSON.stringify({kind:"log",sequence:2002,payload:{text:"DUPLICATE LOG"}}));
        socket.send(JSON.stringify({kind:"log",sequence:2005,payload:{text:"after reconnect\n"}}));
        complete=true;
        socket.send(JSON.stringify({kind:"terminal",sequence:2006,payload:{}}));
        socket.close({code:1000,reason:"Complete"});
      }
    });
  });
  await loadStudio(page,()=>page.reload());await page.getByRole("button",{name:"Runs",exact:true}).click();await observe(page,id);
  await expect(page.locator(".metric").filter({hasText:"old-only"})).toBeVisible();
  await page.getByRole("button",{name:"Logs",exact:true}).click();
  await expect(page.getByText("Showing the latest 2,000 observations.",{exact:false})).toHaveCount(0);
  sendMore();
  await expect(page.locator(".status-grid")).toContainText("Complete");
  await expect(page.locator(".status-grid")).toContainText("succeeded");
  expect(cursors).toEqual([0,2004]);
  await expect(page.getByText("Replay history is incomplete.",{exact:false})).toBeVisible();
  await expect(page.getByText("Showing the latest 2,000 observations.",{exact:false})).toBeVisible();
  await expect(page.getByLabel("Native operation progress")).toContainText("8 / 10 batches observed");
  const logs=await page.getByLabel("Process logs").innerText();
  expect(logs).toContain("Earlier terminal lines omitted");expect(logs).toContain("after reconnect");
  expect(logs).not.toContain("DUPLICATE LOG");expect(logs.split("\n").length).toBeLessThanOrEqual(1001);
  await page.getByRole("button",{name:"Metrics",exact:true}).click();
  await expect(page.locator(".metric")).toHaveCount(1);
  await expect(page.locator(".metric")).toContainText("2000 observed samples");
  expect((await page.locator(".training-series polyline").getAttribute("points"))?.split(" ")).toHaveLength(2000);
});
