import { isIP } from "node:net";

function exactOrigin(value, protocol) {
  const url = new URL(value);
  if (url.protocol !== protocol || value !== url.origin || url.hostname.includes("*")) {
    throw new Error("Use an exact origin without credentials, path, query or fragment.");
  }
  return url;
}

/** @param {string | undefined} value */
export function publicOrigin(value) {
  return value ? exactOrigin(value, "https:").origin : undefined;
}

/** @param {string | undefined} value @param {string | undefined} configuredPublicOrigin */
export function listenHost(value, configuredPublicOrigin) {
  const host = value ?? (configuredPublicOrigin ? "0.0.0.0" : "127.0.0.1");
  if (isIP(host) !== 4) throw new Error("Studio host must be an IPv4 address.");
  if (host !== "127.0.0.1" && !configuredPublicOrigin) {
    throw new Error("A non-loopback Studio host requires a public HTTPS origin.");
  }
  return host;
}

function loopbackOrigin(value) {
  const url = exactOrigin(value, "http:");
  if (!["127.0.0.1", "localhost", "[::1]"].includes(url.hostname)) {
    throw new Error("Use a loopback HTTP origin.");
  }
  return url.origin;
}

function rejected(status, error) {
  return Response.json({ error }, { status, headers: { "Cache-Control": "no-store" } });
}

/** @param {Request} request @param {Record<string, string | undefined>} env */
export function connectionFor(request, env = process.env) {
  let localOrigin, api, hostedOrigin;
  const token = env.NEXUML_STUDIO_TOKEN;
  try {
    localOrigin = loopbackOrigin(env.NEXUML_STUDIO_ORIGIN);
    api = loopbackOrigin(env.NEXUML_STUDIO_API);
    hostedOrigin = publicOrigin(env.NEXUML_STUDIO_PUBLIC_ORIGIN);
    if (!token || api === localOrigin) throw new Error("Missing runtime settings.");
  } catch {
    return rejected(503, "Start Studio with its launcher to select NexuML.");
  }
  const headers = request.headers;
  const host = headers.get("host");
  const browserOrigin = host === new URL(localOrigin).host ? localOrigin :
    hostedOrigin && host === new URL(hostedOrigin).host ? hostedOrigin : undefined;
  const origin = headers.get("origin");
  if (!browserOrigin || (origin !== null && origin !== browserOrigin) ||
      ![null, "same-origin", "none"].includes(headers.get("sec-fetch-site"))) {
    return rejected(403, "Origin rejected.");
  }
  return { api, localOrigin, browserOrigin, token };
}
