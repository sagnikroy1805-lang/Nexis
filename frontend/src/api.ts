/**
 * Typed client for docs/api_contract.md.
 *
 * Data source selection, decided once per page load:
 *   - VITE_MOCK=1 (npm run dev:mock)       -> fixtures from src/mocks/, "mock".
 *   - GET /api/health answers with JSON    -> the real backend, "live".
 *   - otherwise                            -> fixtures, "mock-fallback", and the
 *                                             app shows "Demo data — backend not reachable".
 *
 * The mock module is imported lazily so a live build never executes it.
 */
import type {
  AccountId,
  AccountTransactions,
  AccountTransactionsParams,
  Alert,
  AlertDetail,
  AlertList,
  AlertListParams,
  DispositionRequest,
  DriftReport,
  Health,
  InvestigatorResult,
  ModelLadder,
  Network,
  NetworkParams,
  RingList,
  Summary,
} from "./types";
import { ApiError, type Query, type RequestSpec } from "./http";

export type DataSource = "live" | "mock" | "mock-fallback";

export interface DataSourceState {
  source: DataSource | "checking";
  health: Health | null;
  /** Why the backend was not used, when source is mock-fallback. */
  reason: string | null;
}

export { ApiError };
export type { RequestSpec };

const HEALTH_TIMEOUT_MS = 3000;
const REQUEST_TIMEOUT_MS = 20000;

// ---------------------------------------------------------------------------
// Data-source detection
// ---------------------------------------------------------------------------

let state: DataSourceState = { source: "checking", health: null, reason: null };
let detection: Promise<DataSource> | null = null;
const listeners = new Set<() => void>();

function setState(next: DataSourceState): void {
  state = next;
  listeners.forEach((l) => l());
}

export function subscribeDataSource(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function getDataSourceState(): DataSourceState {
  return state;
}

export function mockForced(): boolean {
  return import.meta.env.VITE_MOCK === "1";
}

async function detect(): Promise<DataSource> {
  if (mockForced()) {
    const { mockRequest } = await import("./mocks");
    const health = (await mockRequest({ method: "GET", path: "/health" })) as Health;
    setState({ source: "mock", health, reason: null });
    return "mock";
  }
  try {
    const health = await fetchJson<Health>({ method: "GET", path: "/health" }, HEALTH_TIMEOUT_MS);
    if (!health || typeof health.status !== "string") {
      throw new Error("unexpected /api/health response");
    }
    setState({ source: "live", health, reason: null });
    return "live";
  } catch (err) {
    const { mockRequest } = await import("./mocks");
    const health = (await mockRequest({ method: "GET", path: "/health" })) as Health;
    setState({ source: "mock-fallback", health, reason: errorMessage(err) });
    return "mock-fallback";
  }
}

export function dataSource(): Promise<DataSource> {
  if (!detection) detection = detect();
  return detection;
}

/** Re-probe the backend (banner "Retry" button). Resolves to the new source. */
export async function recheckBackend(): Promise<DataSource> {
  detection = null;
  setState({ source: "checking", health: null, reason: null });
  return dataSource();
}

// ---------------------------------------------------------------------------
// Transport
// ---------------------------------------------------------------------------

function buildUrl(path: string, query?: Query): string {
  const params = new URLSearchParams();
  if (query) {
    for (const [k, v] of Object.entries(query)) {
      if (v !== undefined && v !== null && v !== "") params.set(k, String(v));
    }
  }
  const qs = params.toString();
  return `/api${path}${qs ? `?${qs}` : ""}`;
}

async function fetchJson<T>(spec: RequestSpec, timeoutMs = REQUEST_TIMEOUT_MS): Promise<T> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(buildUrl(spec.path, spec.query), {
      method: spec.method,
      headers: {
        Accept: "application/json",
        ...(spec.body !== undefined ? { "Content-Type": "application/json" } : {}),
      },
      body: spec.body !== undefined ? JSON.stringify(spec.body) : undefined,
      signal: controller.signal,
    });
    const text = await res.text();
    let payload: unknown = null;
    if (text) {
      try {
        payload = JSON.parse(text);
      } catch {
        // A dev-server SPA fallback answers /api/* with index.html; treat as unreachable.
        throw new ApiError(res.status || 502, `Non-JSON response from ${spec.path}`);
      }
    }
    if (!res.ok) {
      throw new ApiError(res.status, detailOf(payload) ?? `${res.status} ${res.statusText}`);
    }
    return payload as T;
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new ApiError(504, `Request to ${spec.path} timed out`);
    }
    throw err;
  } finally {
    window.clearTimeout(timer);
  }
}

function detailOf(payload: unknown): string | null {
  if (payload && typeof payload === "object" && "detail" in payload) {
    const d = (payload as { detail: unknown }).detail;
    return typeof d === "string" ? d : JSON.stringify(d);
  }
  return null;
}

export function errorMessage(err: unknown): string {
  if (err instanceof Error) return err.message;
  return String(err);
}

async function request<T>(spec: RequestSpec): Promise<T> {
  const source = await dataSource();
  if (source === "live") return fetchJson<T>(spec);
  const { mockRequest } = await import("./mocks");
  return (await mockRequest(spec)) as T;
}

// ---------------------------------------------------------------------------
// Endpoints (one function per contract route)
// ---------------------------------------------------------------------------

const enc = encodeURIComponent;

export const api = {
  health: () => request<Health>({ method: "GET", path: "/health" }),

  summary: () => request<Summary>({ method: "GET", path: "/summary" }),

  alerts: (p: AlertListParams = {}) =>
    request<AlertList>({
      method: "GET",
      path: "/alerts",
      query: { status: p.status, limit: p.limit, offset: p.offset, sort: p.sort },
    }),

  alert: (alertId: number) => request<AlertDetail>({ method: "GET", path: `/alerts/${alertId}` }),

  disposition: (alertId: number, body: DispositionRequest) =>
    request<Alert>({ method: "POST", path: `/alerts/${alertId}/disposition`, body }),

  investigate: (alertId: number) =>
    request<InvestigatorResult>({ method: "POST", path: `/alerts/${alertId}/investigate` }),

  network: (account: AccountId, p: NetworkParams = {}) =>
    request<Network>({
      method: "GET",
      path: `/accounts/${enc(account)}/network`,
      query: { until: p.until, hops: p.hops, limit: p.limit },
    }),

  accountTransactions: (account: AccountId, p: AccountTransactionsParams = {}) =>
    request<AccountTransactions>({
      method: "GET",
      path: `/accounts/${enc(account)}/transactions`,
      query: { until: p.until, limit: p.limit },
    }),

  rings: (limit?: number) => request<RingList>({ method: "GET", path: "/rings", query: { limit } }),

  drift: () => request<DriftReport>({ method: "GET", path: "/drift" }),

  models: () => request<ModelLadder>({ method: "GET", path: "/models" }),
};
