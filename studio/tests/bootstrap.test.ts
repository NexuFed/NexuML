import { afterEach, expect, test, vi } from "vitest";
import { GET } from "../app/api/runtime/route";

afterEach(()=>vi.unstubAllEnvs());
test("bootstrap is runtime-only, exact host/origin/fetch-metadata protected and uncached",async()=>{
  vi.stubEnv("NEXUML_STUDIO_ORIGIN","http://127.0.0.1:3000");vi.stubEnv("NEXUML_STUDIO_API","http://127.0.0.1:8000");vi.stubEnv("NEXUML_STUDIO_TOKEN","private-runtime-value");
  const allowed={host:"127.0.0.1:3000",origin:"http://127.0.0.1:3000","sec-fetch-site":"same-origin"};
  const response=GET(new Request("http://127.0.0.1:3000/api/runtime",{headers:allowed}));
  expect(response.status).toBe(200);expect(response.headers.get("cache-control")).toBe("no-store");
  expect(await response.json()).toEqual({api:"http://127.0.0.1:3000",token:"private-runtime-value"});
  for(const rejected of [{...allowed,host:"evil.test"},{...allowed,origin:"http://evil.test"},{...allowed,"sec-fetch-site":"cross-site"}]) {
    const denied=GET(new Request("http://127.0.0.1:3000/api/runtime",{headers:rejected}));expect(denied.status).toBe(403);expect(await denied.text()).not.toContain("private-runtime-value");
  }
  vi.stubEnv("NEXUML_STUDIO_TOKEN","");expect(GET(new Request("http://127.0.0.1:3000/api/runtime",{headers:allowed})).status).toBe(503);
});

test("HTTPS bootstrap requires proxy authentication for public and local access",async()=>{
  vi.stubEnv("NEXUML_STUDIO_ORIGIN","http://127.0.0.1:3000");vi.stubEnv("NEXUML_STUDIO_API","http://127.0.0.1:8000");vi.stubEnv("NEXUML_STUDIO_TOKEN","private-runtime-value");
  vi.stubEnv("NEXUML_STUDIO_PUBLIC_ORIGIN","https://studio.example");
  const proxyToken="private-proxy-token-".repeat(3);vi.stubEnv("NEXUML_STUDIO_PROXY_TOKEN",proxyToken);
  for(const origin of ["https://studio.example","http://127.0.0.1:3000"]) {
    const headers={host:new URL(origin).host,origin,"sec-fetch-site":"same-origin"};
    expect(GET(new Request(`${origin}/api/runtime`,{headers})).status).toBe(401);
    const response=GET(new Request(`${origin}/api/runtime`,{headers:{...headers,"x-nexuml-studio-proxy":proxyToken}}));
    expect(response.status).toBe(200);expect(response.headers.get("cache-control")).toBe("no-store");
    expect(await response.json()).toEqual({api:origin,token:"private-runtime-value"});
  }
});
