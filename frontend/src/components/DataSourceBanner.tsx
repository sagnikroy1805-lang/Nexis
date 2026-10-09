import { useState, useSyncExternalStore } from "react";
import { getDataSourceState, recheckBackend, subscribeDataSource } from "../api";

export function useDataSourceState() {
  return useSyncExternalStore(subscribeDataSource, getDataSourceState);
}

/** Tiny pill shown whenever the screen is not showing backend data. */
export function DemoPill() {
  const s = useDataSourceState();
  const [busy, setBusy] = useState(false);

  if (s.source === "mock") {
    return (
      <span
        className="rounded-full bg-ink px-2.5 py-0.5 text-[10px] font-medium tracking-[0.14em] text-gold uppercase"
        title="Mock mode (VITE_MOCK=1): fixture records only."
        role="status"
      >
        Demo data
      </span>
    );
  }
  if (s.source === "mock-fallback") {
    const retry = async () => {
      setBusy(true);
      const next = await recheckBackend();
      setBusy(false);
      // Views fetched fixtures; reload so every view refetches from the backend.
      if (next === "live") window.location.reload();
    };
    return (
      <span className="inline-flex items-center gap-1.5" role="alert">
        <span
          className="rounded-full bg-ink px-2.5 py-0.5 text-[10px] font-medium tracking-[0.14em] text-gold uppercase"
          title={`Backend not reachable${s.reason ? ` (${s.reason})` : ""}. Showing fixture records.`}
        >
          Demo data — backend not reachable
        </span>
        <button type="button" className="pill pill-sm" onClick={retry} disabled={busy}>
          {busy ? "…" : "Retry"}
        </button>
      </span>
    );
  }
  return null;
}
