/**
 * In-browser mock of the NEXIS API (docs/api_contract.md).
 *
 * `mockRequest` routes a RequestSpec to a handler, validates query parameters
 * the way the contract describes, and returns deep copies so the UI cannot
 * mutate the fixture store. Dispositions persist for the lifetime of the page.
 */
import { ApiError, type RequestSpec } from "../http";
import type {
  Alert,
  AlertDetail,
  AlertList,
  AlertStatus,
  Disposition,
  Health,
  Summary,
} from "../types";
import { accountTransactions, evidenceFor, investigate, network } from "./evidence";
import { driftReport, modelLadder } from "./fixtures";
import { DATASET, MODEL_VERSION, REPLAY_END, REPLAY_START, THRESHOLD, buildWorld, type World } from "./world";

let world: World | null = null;
const statusOverrides = new Map<number, AlertStatus>();
const notes = new Map<number, string>();

function getWorld(): World {
  if (!world) world = buildWorld();
  return world;
}

function alertsNow(): Alert[] {
  return getWorld().alerts.map((a) => ({ ...a, status: statusOverrides.get(a.alert_id) ?? a.status }));
}

function findAlert(id: number): Alert {
  const a = alertsNow().find((x) => x.alert_id === id);
  if (!a) throw new ApiError(404, `Alert ${id} not found`);
  return a;
}

function intParam(v: unknown, def: number, min: number, max: number, name: string): number {
  if (v === undefined || v === null || v === "") return def;
  const n = Number(v);
  if (!Number.isInteger(n) || n < min || n > max) {
    throw new ApiError(422, `${name} must be an integer in [${min}, ${max}]`);
  }
  return n;
}

const STATUS_FILTERS = ["open", "escalated", "dismissed", "needs_info", "all"];
const DISPOSITIONS: Disposition[] = ["escalated", "dismissed", "needs_info"];

function delay(): Promise<void> {
  return new Promise((r) => setTimeout(r, 120 + Math.random() * 220));
}

function clone<T>(x: T): T {
  return JSON.parse(JSON.stringify(x)) as T;
}

function handle(spec: RequestSpec): unknown {
  const { method, path } = spec;
  const q = spec.query ?? {};
  let m: RegExpMatchArray | null;

  if (method === "GET" && path === "/health") {
    const h: Health = { status: "ok", model_version: MODEL_VERSION, db: "mock" };
    return h;
  }

  if (method === "GET" && path === "/summary") {
    const alerts = alertsNow();
    const s: Summary = {
      model_version: MODEL_VERSION,
      dataset: DATASET,
      replay_period: { start: REPLAY_START, end: REPLAY_END },
      transactions_scored: 416866,
      alerts_total: alerts.length,
      alerts_open: alerts.filter((a) => a.status === "open").length,
      alert_budget: 40,
      threshold: THRESHOLD,
      metrics: { pr_auc: 0.61, pr_auc_std: 0.01, prevalence: 0.00147, recall_at_budget: 0.52, precision_at_budget: 0.64 },
    };
    return s;
  }

  if (method === "GET" && path === "/alerts") {
    const status = (q.status ?? "open") as string;
    if (!STATUS_FILTERS.includes(status)) throw new ApiError(422, `Unknown status '${status}'`);
    const sort = (q.sort ?? "risk_score") as string;
    if (sort !== "risk_score" && sort !== "occurred_at") throw new ApiError(422, `Unknown sort '${sort}'`);
    const limit = intParam(q.limit, 50, 1, 500, "limit");
    const offset = intParam(q.offset, 0, 0, Number.MAX_SAFE_INTEGER, "offset");
    let items = alertsNow().filter((a) => status === "all" || a.status === status);
    items =
      sort === "risk_score"
        ? items.sort((a, b) => b.risk_score - a.risk_score || a.rank - b.rank)
        : items.sort((a, b) => b.occurred_at.localeCompare(a.occurred_at));
    const list: AlertList = { total: items.length, items: items.slice(offset, offset + limit) };
    return list;
  }

  if ((m = path.match(/^\/alerts\/(\d+)$/)) && method === "GET") {
    const alert = findAlert(Number(m[1]));
    const detail: AlertDetail = { alert, evidence: evidenceFor(getWorld(), alert) };
    return detail;
  }

  if ((m = path.match(/^\/alerts\/(\d+)\/disposition$/)) && method === "POST") {
    const id = Number(m[1]);
    findAlert(id);
    const body = (spec.body ?? {}) as { disposition?: string; note?: string };
    if (!DISPOSITIONS.includes(body.disposition as Disposition)) {
      throw new ApiError(422, "disposition must be one of escalated, dismissed, needs_info");
    }
    statusOverrides.set(id, body.disposition as Disposition);
    notes.set(id, typeof body.note === "string" ? body.note : "");
    return findAlert(id);
  }

  if ((m = path.match(/^\/alerts\/(\d+)\/investigate$/)) && method === "POST") {
    const alert = findAlert(Number(m[1]));
    return investigate(getWorld(), alert, evidenceFor(getWorld(), alert));
  }

  if ((m = path.match(/^\/accounts\/([^/]+)\/network$/)) && method === "GET") {
    const account = decodeURIComponent(m[1]);
    const w = getWorld();
    if (!w.byAccount.has(account)) throw new ApiError(404, `Account ${account} not found`);
    const hops = intParam(q.hops, 2, 1, 2, "hops");
    const limit = intParam(q.limit, 300, 1, 5000, "limit");
    return network(w, account, (q.until as string | undefined) || undefined, hops, limit);
  }

  if ((m = path.match(/^\/accounts\/([^/]+)\/transactions$/)) && method === "GET") {
    const account = decodeURIComponent(m[1]);
    const w = getWorld();
    if (!w.byAccount.has(account)) throw new ApiError(404, `Account ${account} not found`);
    const limit = intParam(q.limit, 100, 1, 5000, "limit");
    return { items: accountTransactions(w, account, (q.until as string | undefined) || undefined, limit) };
  }

  if (method === "GET" && path === "/rings") {
    const limit = intParam(q.limit, 50, 1, 500, "limit");
    return { items: getWorld().rings.slice(0, limit) };
  }

  if (method === "GET" && path === "/drift") return driftReport();
  if (method === "GET" && path === "/models") return modelLadder();

  throw new ApiError(404, `No mock route for ${method} ${path}`);
}

export async function mockRequest(spec: RequestSpec): Promise<unknown> {
  await delay();
  try {
    return clone(handle(spec));
  } catch (err) {
    if (err instanceof ApiError) throw err;
    throw new ApiError(500, err instanceof Error ? err.message : String(err));
  }
}

/** Exposed for the scratch sanity check; not used by the app. */
export const __mockInternals = { getWorld, handle };
