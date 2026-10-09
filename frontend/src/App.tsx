import { AnimatePresence, MotionConfig, motion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { Link, NavLink, Route, Routes, useLocation } from "react-router-dom";
import { DemoPill, useDataSourceState } from "./components/DataSourceBanner";
import { Orbits } from "./components/Orbits";
import { PageTitle } from "./components/ui";
import AlertDetail from "./views/AlertDetail";
import AlertQueue from "./views/AlertQueue";
import Dashboard from "./views/Dashboard";
import Drift from "./views/Drift";
import GraphExplorer from "./views/GraphExplorer";
import Models from "./views/Models";
import Rings from "./views/Rings";

const NAV = [
  { to: "/", label: "Dashboard", match: (p: string) => p === "/" },
  { to: "/alerts", label: "Alerts", match: (p: string) => p.startsWith("/alerts") },
  { to: "/graph", label: "Graph", match: (p: string) => p.startsWith("/graph") },
  { to: "/rings", label: "Rings", match: (p: string) => p.startsWith("/rings") },
  { to: "/drift", label: "Drift", match: (p: string) => p.startsWith("/drift") },
  { to: "/models", label: "Models", match: (p: string) => p.startsWith("/models") },
];

/**
 * Which navigations count as a page change (and so animate). Moving within
 * the graph (recentre) or the ring list keeps the page mounted, so its data
 * stays on screen while the next request loads.
 */
function pageKey(pathname: string): string {
  const parts = pathname.split("/").filter(Boolean);
  if (parts.length === 0) return "dashboard";
  if (parts[0] === "alerts") return parts[1] ? `alert:${parts[1]}` : "alerts";
  return parts[0];
}

function NavLinks({ pathname, vertical = false }: { pathname: string; vertical?: boolean }) {
  return (
    <ul className={vertical ? "flex flex-col gap-3" : "flex items-center gap-7 lg:gap-9"}>
      {NAV.map((n) => {
        const active = n.match(pathname);
        return (
          <li key={n.to}>
            <NavLink
              to={n.to}
              aria-current={active ? "page" : undefined}
              className={`relative text-[13px] tracking-wide transition-opacity hover:opacity-70 ${
                active ? "font-medium after:absolute after:-bottom-1.5 after:left-0 after:h-px after:w-full after:bg-ink" : ""
              }`}
            >
              {n.label}
            </NavLink>
          </li>
        );
      })}
    </ul>
  );
}

function Menu({ pathname }: { pathname: string }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const s = useDataSourceState();
  useEffect(() => setOpen(false), [pathname]);
  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDoc);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const source = s.source === "live" ? "Live backend" : s.source === "checking" ? "Connecting" : "Demo data";
  return (
    <div className="relative" ref={ref}>
      <button
        type="button"
        className="flex h-8 w-8 flex-col items-end justify-center gap-[7px]"
        aria-label="Menu"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
      >
        <span className={`block h-px bg-ink transition-all duration-300 ${open ? "w-7 translate-y-1 rotate-45" : "w-7"}`} />
        <span className={`block h-px bg-ink transition-all duration-300 ${open ? "w-7 -translate-y-1 -rotate-45" : "w-5"}`} />
      </button>
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            transition={{ duration: 0.2 }}
            className="glass absolute right-0 z-30 mt-3 w-64 text-[13px]"
          >
            <div className="lg:hidden">
              <NavLinks pathname={pathname} vertical />
              <div className="my-4 h-px bg-ink/15" />
            </div>
            <div className="label">Source</div>
            <div className="mt-1">{source}</div>
            {s.health && (
              <>
                <div className="label mt-3">Model</div>
                <div className="mono mt-1 break-all">{s.health.model_version}</div>
              </>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function NotFound() {
  return (
    <div>
      <PageTitle title="Not found" />
      <Link to="/" className="pill mt-6">
        Dashboard
      </Link>
    </div>
  );
}

export default function App() {
  const location = useLocation();
  const key = pageKey(location.pathname);

  return (
    <MotionConfig reducedMotion="user">
      {/* Mounted once, outside the animated route tree: never restarts on navigation. */}
      <Orbits />
      {/* overflow-x: clip is a safety net; layouts are built not to overflow. */}
      <div className="relative z-10 min-h-screen overflow-x-clip">
        <header className="relative mx-auto flex max-w-[1400px] items-start justify-between px-6 pt-6 md:px-10">
          <Link to="/" className="leading-none" aria-label="NEXIS dashboard">
            <span className="block font-script text-[40px] leading-[0.8]">Nexis</span>
            <span className="label mt-1 block pl-1 text-[9px] tracking-[0.32em]">Research</span>
          </Link>
          <nav aria-label="Main" className="absolute top-9 left-1/2 hidden -translate-x-1/2 lg:block">
            <NavLinks pathname={location.pathname} />
          </nav>
          <div className="flex items-center gap-4 pt-1">
            <DemoPill />
            <Menu pathname={location.pathname} />
          </div>
        </header>

        {/*
          Route transitions. <Routes location> pins each page to the location it
          was rendered for, so the exiting page keeps its own params while it
          fades out ("wait": exit completes before the next page enters).
        */}
        <AnimatePresence mode="wait" initial={false} onExitComplete={() => window.scrollTo(0, 0)}>
          <motion.main
            key={key}
            initial={{ opacity: 0, y: 18, scale: 0.995 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -10, scale: 0.995 }}
            transition={{ duration: 0.38, ease: [0.22, 1, 0.36, 1] }}
            className="mx-auto max-w-[1400px] px-6 pt-8 pb-20 md:px-10"
          >
            <Routes location={location}>
              <Route index element={<Dashboard />} />
              <Route path="alerts" element={<AlertQueue />} />
              <Route path="alerts/:id" element={<AlertDetail />} />
              <Route path="graph" element={<GraphExplorer />} />
              <Route path="graph/:account" element={<GraphExplorer />} />
              <Route path="rings" element={<Rings />} />
              <Route path="rings/:ringId" element={<Rings />} />
              <Route path="drift" element={<Drift />} />
              <Route path="models" element={<Models />} />
              <Route path="*" element={<NotFound />} />
            </Routes>
          </motion.main>
        </AnimatePresence>
      </div>
    </MotionConfig>
  );
}
