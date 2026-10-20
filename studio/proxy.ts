import { NextResponse, type NextRequest } from "next/server";
import { connectionFor } from "./bin/connection.mjs";

export function proxy(request: NextRequest) {
  const connection = connectionFor(request);
  if (connection instanceof Response) return connection;
  const destination = new URL(connection.api);
  destination.pathname = request.nextUrl.pathname;
  destination.search = request.nextUrl.search;
  const headers = new Headers(request.headers);
  for (const key of [...headers.keys()]) {
    if (["cookie", "forwarded", "x-real-ip", "via", "x-netbird-user", "x-netbird-groups", "x-nexuml-studio-proxy"].includes(key) || key.startsWith("x-forwarded-")) {
      headers.delete(key);
    }
  }
  headers.set("host", destination.host);
  headers.set("origin", connection.localOrigin);
  return NextResponse.rewrite(destination, { request: { headers } });
}

export const config = { matcher: "/api/v1/:path*" };
