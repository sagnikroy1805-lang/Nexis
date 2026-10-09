import { useCallback, useEffect, useRef, useState, type DependencyList, type RefObject } from "react";
import { errorMessage } from "../api";

export interface ApiState<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
  reload: () => void;
  setData: (next: T) => void;
}

/**
 * Run `fn` whenever `deps` change; ignore responses from superseded calls.
 *
 * `keepPrevious` keeps the last payload on screen while the next one loads
 * (pagination, filters). Leave it false where stale data would be misleading,
 * e.g. switching from one alert's evidence to another's.
 */
export function useApi<T>(fn: () => Promise<T>, deps: DependencyList, keepPrevious = false): ApiState<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [nonce, setNonce] = useState(0);
  const fnRef = useRef(fn);
  fnRef.current = fn;

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    if (!keepPrevious) setData(null);
    fnRef.current().then(
      (d) => {
        if (cancelled) return;
        setData(d);
        setLoading(false);
      },
      (e: unknown) => {
        if (cancelled) return;
        setError(errorMessage(e));
        setLoading(false);
      },
    );
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, nonce]);

  const reload = useCallback(() => setNonce((n) => n + 1), []);
  return { data, error, loading, reload, setData };
}

/** Width of an element, tracked with ResizeObserver (charts render at real pixels). */
export function useElementWidth<E extends HTMLElement>(): [RefObject<E>, number] {
  const ref = useRef<E>(null);
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    setWidth(el.clientWidth);
    const ro = new ResizeObserver((entries) => {
      for (const e of entries) setWidth(Math.floor(e.contentRect.width));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, width];
}
