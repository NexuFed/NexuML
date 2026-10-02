import type { ConnectionInfo } from "./types";

export class ApiError extends Error {
  constructor(public code: string, message: string, public fields: {loc:(string|number)[];message:string}[] = []) {
    super(message);
  }
}

export async function request<T>(connection: ConnectionInfo, path: string, body?: unknown, method?: string): Promise<T> {
  const response = await fetch(`${connection.api}/api/v1${path}`, {
    method: method ?? (body === undefined ? "GET" : "POST"), cache: "no-store",
    headers: { Authorization: `Bearer ${connection.token}`, "X-NexuML-Interface": "1", "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const result = await response.json();
  if (!response.ok) throw new ApiError(result.error?.code ?? "connection", result.error?.message ?? "API request failed.", result.error?.fields);
  return result;
}
