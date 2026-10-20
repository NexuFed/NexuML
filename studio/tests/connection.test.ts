import { afterEach, expect, test, vi } from "vitest";
import { NextRequest } from "next/server";
import { connectionFor, listenHost, publicOrigin } from "../bin/connection.mjs";
import { proxy, config } from "../proxy";
import { useObservation as observe } from "../components/use-observation";

vi.mock("react", () => ({
  useEffect: (effect: () => void) => effect(),
  useState: (initial: unknown) => [initial, vi.fn()],
}));
vi.mock("@tanstack/react-query", () => ({ useQueryClient: () => ({}) }));

const env = {
  NEXUML_STUDIO_ORIGIN: "http://127.0.0.1:3000",
  NEXUML_STUDIO_API: "http://127.0.0.1:8000",
  NEXUML_STUDIO_TOKEN: "private-runtime-value",
};
const proxyToken = "private-proxy-token-".repeat(3);
const hostedEnv = { ...env, NEXUML_STUDIO_PUBLIC_ORIGIN: "https://studio.example", NEXUML_STUDIO_PROXY_TOKEN: proxyToken };
const allowed = { host: "127.0.0.1:3000", origin: env.NEXUML_STUDIO_ORIGIN, "sec-fetch-site": "same-origin" };
const request = (headers: Record<string, string> = allowed) => new Request(`${env.NEXUML_STUDIO_ORIGIN}/api/runtime`, { headers });

afterEach(() => { vi.unstubAllEnvs(); vi.unstubAllGlobals(); });

test("exact origins and opt-in IPv4 hosting", () => {
  expect(publicOrigin(undefined)).toBeUndefined();
  expect(publicOrigin("")).toBeUndefined();
  expect(publicOrigin("https://studio.example:8443")).toBe("https://studio.example:8443");
  for (const value of ["http://studio.example", "https://studio.example/", "https://studio.example/path",
    "https://user:secret@studio.example", "https://studio.example?x=1", "https://studio.example#fragment",
    "https://*.example", " https://studio.example", "not a URL"]) {
    expect(() => publicOrigin(value)).toThrow();
  }
  expect(listenHost(undefined, undefined)).toBe("127.0.0.1");
  expect(listenHost(undefined, "https://studio.example")).toBe("127.0.0.1");
  expect(listenHost("192.168.1.20", "https://studio.example")).toBe("192.168.1.20");
  expect(listenHost("127.0.0.1", "https://studio.example")).toBe("127.0.0.1");
  for (const value of ["0.0.0.0", "192.168.1.20", "localhost", "::1", "127.1", "999.0.0.1", ""]) {
    expect(() => listenHost(value, undefined)).toThrow();
  }
});

test("local and public connections use actual Host, not the request URL or forwarded metadata", () => {
  expect(connectionFor(request(), env)).toEqual({ api: env.NEXUML_STUDIO_API,
    localOrigin: env.NEXUML_STUDIO_ORIGIN, browserOrigin: env.NEXUML_STUDIO_ORIGIN, token: env.NEXUML_STUDIO_TOKEN });
  const hosted = { ...hostedEnv, NEXUML_STUDIO_PUBLIC_ORIGIN: "https://studio.example:8443" };
  expect(connectionFor(request({...allowed,"x-nexuml-studio-proxy":proxyToken}), hosted)).toMatchObject({ browserOrigin: env.NEXUML_STUDIO_ORIGIN });
  expect(connectionFor(request({ host: "studio.example:8443", origin: hosted.NEXUML_STUDIO_PUBLIC_ORIGIN,
    "x-forwarded-host": "evil.example", "x-forwarded-proto": "http", "sec-fetch-site": "same-origin", "x-nexuml-studio-proxy":proxyToken }), hosted))
    .toMatchObject({ browserOrigin: hosted.NEXUML_STUDIO_PUBLIC_ORIGIN });
  expect(connectionFor(request({ host: allowed.host }), env)).not.toBeInstanceOf(Response);
  expect(connectionFor(request({ ...allowed, "sec-fetch-site": "none" }), env)).not.toBeInstanceOf(Response);
  for (const api of ["http://localhost:8000", "http://[::1]:8000"]) {
    expect(connectionFor(request(), { ...env, NEXUML_STUDIO_API: api })).toMatchObject({ api });
  }
});

test("foreign Host, Origin, Fetch Metadata and forwarded-header spoofs fail closed without secrets", async () => {
  const hosted = hostedEnv;
  for (const headers of [{ ...allowed, host: "evil.example" }, { ...allowed, host: "" },
    { ...allowed, origin: "https://evil.example" }, { ...allowed, origin: "" },
    { ...allowed, origin: hosted.NEXUML_STUDIO_PUBLIC_ORIGIN },
    { ...allowed, host: "studio.example" },
    { ...allowed, "sec-fetch-site": "cross-site" }, { ...allowed, "sec-fetch-site": "same-site" },
    { ...allowed, host: "evil.example", "x-forwarded-host": "studio.example", "x-forwarded-proto": "https" }]) {
    const response = connectionFor(request({...headers,"x-nexuml-studio-proxy":proxyToken}), hosted);
    expect(response).toBeInstanceOf(Response);
    if (!(response instanceof Response)) throw new Error("Expected rejection.");
    expect(response.status).toBe(403);
    expect(response.headers.get("cache-control")).toBe("no-store");
    expect(await response.text()).not.toContain(env.NEXUML_STUDIO_TOKEN);
  }
});

test("hosted connections require the proxy credential even with spoofed local/public browser headers", async () => {
  for (const host of [allowed.host,"studio.example"]) for (const token of ["", "incorrect", env.NEXUML_STUDIO_TOKEN]) {
    const response=connectionFor(request({host,origin:host===allowed.host ? allowed.origin : hostedEnv.NEXUML_STUDIO_PUBLIC_ORIGIN,
      "x-nexuml-studio-proxy":token,"sec-fetch-site":"same-origin"}),hostedEnv);
    expect(response).toBeInstanceOf(Response);
    if(!(response instanceof Response))throw new Error("Expected rejection");
    expect(response.status).toBe(401);expect(await response.text()).not.toContain(env.NEXUML_STUDIO_TOKEN);
  }
  expect(connectionFor(request(),{...hostedEnv,NEXUML_STUDIO_PROXY_TOKEN:""})).toBeInstanceOf(Response);
});

test("missing, malformed, non-loopback and identical runtime origins are uncached configuration errors", async () => {
  for (const patch of [{ NEXUML_STUDIO_TOKEN: "" }, { NEXUML_STUDIO_ORIGIN: undefined },
    { NEXUML_STUDIO_API: undefined }, { NEXUML_STUDIO_ORIGIN: "not a URL" },
    { NEXUML_STUDIO_ORIGIN: "https://127.0.0.1:3000" }, { NEXUML_STUDIO_ORIGIN: "http://evil.example" },
    { NEXUML_STUDIO_API: "http://127.0.0.1:8000/" }, { NEXUML_STUDIO_API: "http://127.0.0.1:8000/path" },
    { NEXUML_STUDIO_API: "https://127.0.0.1:8000" }, { NEXUML_STUDIO_API: "http://evil.example" },
    { NEXUML_STUDIO_API: "http://user:secret@127.0.0.1:8000" },
    { NEXUML_STUDIO_API: env.NEXUML_STUDIO_ORIGIN }, { NEXUML_STUDIO_PUBLIC_ORIGIN: "http://studio.example" }]) {
    const response = connectionFor(request(), { ...env, ...patch });
    expect(response).toBeInstanceOf(Response);
    if (!(response instanceof Response)) throw new Error("Expected rejection.");
    expect(response.status).toBe(503);
    expect(response.headers.get("cache-control")).toBe("no-store");
    expect(await response.text()).not.toContain(env.NEXUML_STUDIO_TOKEN);
  }
});

test("dynamic proxy fixes upstream origin, preserves auth/query/upgrade and strips cookies and proxy metadata", () => {
  for (const [key, value] of Object.entries(env)) vi.stubEnv(key, value);
  vi.stubEnv("NEXUML_STUDIO_PUBLIC_ORIGIN", "https://studio.example");
  vi.stubEnv("NEXUML_STUDIO_PROXY_TOKEN", proxyToken);
  const headers = { host: "studio.example", origin: "https://studio.example", authorization: "Bearer browser-token",
    "x-nexuml-interface": "2", cookie: "proxy-session=private", forwarded: "host=evil.example;proto=https",
    "x-forwarded-host": "evil.example", "x-forwarded-proto": "https", "x-forwarded-for": "203.0.113.1",
    "x-forwarded-port": "443", "x-real-ip": "203.0.113.1", via: "proxy",
    upgrade: "websocket", connection: "Upgrade", "sec-websocket-key": "browser-key", "x-nexuml-studio-proxy":proxyToken };
  const incoming = new NextRequest("http://127.0.0.1:3000/api/v1/operations/one/events?after=7", { headers });
  const response = proxy(incoming);
  expect(config.matcher).toBe("/api/v1/:path*");
  expect(response.headers.get("x-middleware-rewrite")).toBe("http://127.0.0.1:8000/api/v1/operations/one/events?after=7");
  for (const [key, value] of Object.entries({ authorization: headers.authorization, "x-nexuml-interface": "2",
    host: "127.0.0.1:8000", origin: env.NEXUML_STUDIO_ORIGIN, upgrade: "websocket", connection: "Upgrade",
    "sec-websocket-key": "browser-key" })) expect(response.headers.get(`x-middleware-request-${key}`)).toBe(value);
  for (const key of ["cookie", "forwarded", "x-forwarded-host", "x-forwarded-proto", "x-forwarded-for",
    "x-forwarded-port", "x-real-ip", "via", "x-nexuml-studio-proxy"]) expect(response.headers.has(`x-middleware-request-${key}`)).toBe(false);
  expect([...response.headers.values()].join()).not.toContain(env.NEXUML_STUDIO_TOKEN);
  expect(incoming.headers.get("cookie")).toBe(headers.cookie);
  vi.stubEnv("NEXUML_STUDIO_API", "http://127.0.0.1:9001");
  const unauthenticated = proxy(new NextRequest(incoming.url, { headers: { host: headers.host, origin: headers.origin,"x-nexuml-studio-proxy":proxyToken } }));
  expect(unauthenticated.headers.get("x-middleware-rewrite")).toContain("127.0.0.1:9001");
  expect(unauthenticated.headers.has("x-middleware-request-authorization")).toBe(false);
  expect(proxy(new NextRequest(incoming.url, { headers: { host: "evil.example","x-nexuml-studio-proxy":proxyToken } })).status).toBe(403);
  vi.stubEnv("NEXUML_STUDIO_TOKEN", "");
  expect(proxy(incoming).status).toBe(503);
});

test("dev HMR allowlist contains only loopback and the explicit public hostname", async () => {
  vi.stubEnv("NEXUML_STUDIO_PUBLIC_ORIGIN", "");
  vi.resetModules();
  expect((await import("../next.config.mjs")).default.allowedDevOrigins).toEqual(["127.0.0.1"]);
  vi.stubEnv("NEXUML_STUDIO_PUBLIC_ORIGIN", "https://studio.example:8443");
  vi.resetModules();
  expect((await import("../next.config.mjs")).default.allowedDevOrigins).toEqual(["127.0.0.1", "studio.example"]);
  vi.stubEnv("NEXUML_STUDIO_PUBLIC_ORIGIN", "https://*.example");
  vi.resetModules();
  await expect(import("../next.config.mjs")).rejects.toThrow();
});

test("observation uses WS locally and WSS on HTTPS with unchanged first-frame authentication", () => {
  for (const [api, expected] of [[env.NEXUML_STUDIO_ORIGIN, "ws:"], ["https://studio.example", "wss:"]]) {
    const send = vi.fn();
    const Socket = vi.fn(function (url: URL) { return { url, send, close: vi.fn() }; });
    vi.stubGlobal("WebSocket", Socket);
    observe({ api, token: env.NEXUML_STUDIO_TOKEN }, "operation-one");
    const url = Socket.mock.calls[0][0] as unknown as URL;
    expect(url.protocol).toBe(expected);
    expect(url.pathname).toBe("/api/v1/operations/operation-one/events");
    expect(url.search).toBe("?after=0");
    expect(url.toString()).not.toContain(env.NEXUML_STUDIO_TOKEN);
    const socket = Socket.mock.results[0].value as { onopen: () => void };
    socket.onopen();
    expect(send).toHaveBeenCalledWith(JSON.stringify({ token: env.NEXUML_STUDIO_TOKEN }));
  }
});
