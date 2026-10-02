import { afterEach, expect, test, vi } from "vitest";
import { GET } from "../app/api/runtime/route";

afterEach(()=>vi.unstubAllEnvs());
test("bootstrap is runtime-only, exact host/origin/fetch-metadata protected and uncached",async()=>{
  vi.stubEnv("NEXUML_STUDIO_ORIGIN","http://127.0.0.1:3000");vi.stubEnv("NEXUML_STUDIO_API","http://127.0.0.1:8000");vi.stubEnv("NEXUML_STUDIO_TOKEN","private-runtime-value");
  const allowed={host:"127.0.0.1:3000",origin:"http://127.0.0.1:3000","sec-fetch-site":"same-origin"};
  const response=GET(new Request("http://127.0.0.1:3000/api/runtime",{headers:allowed}));
  expect(response.status).toBe(200);expect(response.headers.get("cache-control")).toBe("no-store");
  expect((await response.json()).token).toBe("private-runtime-value");
  for(const rejected of [{...allowed,host:"evil.test"},{...allowed,origin:"http://evil.test"},{...allowed,"sec-fetch-site":"cross-site"}]) {
    const denied=GET(new Request("http://127.0.0.1:3000/api/runtime",{headers:rejected}));expect(denied.status).toBe(403);expect(await denied.text()).not.toContain("private-runtime-value");
  }
  vi.stubEnv("NEXUML_STUDIO_TOKEN","");expect(GET(new Request("http://127.0.0.1:3000/api/runtime",{headers:allowed})).status).toBe(503);
});
