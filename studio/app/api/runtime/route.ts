export const dynamic = "force-dynamic";

export function GET(request: Request) {
  const origin = process.env.NEXUML_STUDIO_ORIGIN;
  const token = process.env.NEXUML_STUDIO_TOKEN;
  const api = process.env.NEXUML_STUDIO_API;
  if (!origin || !token || !api) {
    return Response.json({ error: "Start Studio with its launcher to select NexuML." }, { status: 503 });
  }
  const headers = request.headers;
  if (headers.get("host") !== new URL(origin).host ||
      (headers.get("origin") && headers.get("origin") !== origin) ||
      ![null, "same-origin", "none"].includes(headers.get("sec-fetch-site"))) {
    return Response.json({ error: "Origin rejected." }, { status: 403 });
  }
  return Response.json({ api, token }, { headers: { "Cache-Control": "no-store" } });
}
