/** Transport-level types shared by the real client (api.ts) and the mock layer. */

export type Query = Record<string, string | number | undefined | null>;

export interface RequestSpec {
  method: "GET" | "POST";
  /** Path below /api, e.g. "/alerts/17". */
  path: string;
  query?: Query;
  body?: unknown;
}

export class ApiError extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}
