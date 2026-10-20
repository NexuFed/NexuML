import { connectionFor } from "../../../bin/connection.mjs";

export const dynamic = "force-dynamic";

export function GET(request: Request) {
  const connection = connectionFor(request);
  if (connection instanceof Response) return connection;
  return Response.json({ api: connection.browserOrigin, token: connection.token },
    { headers: { "Cache-Control": "no-store" } });
}
