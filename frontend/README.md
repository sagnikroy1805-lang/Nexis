# NEXIS analyst dashboard

React front end for the NEXIS research prototype. It consumes the JSON API in
[`../docs/api_contract.md`](../docs/api_contract.md) and ranks transactions by
risk score for analyst review.

A risk score is model output on recorded data, not a finding (PROJECT_RULES.md rule 5).
The UI never states that a person did anything. It offers no action on an
account; the only write is an analyst *disposition* on an alert.

## Run

Requires Node 20+ (tested on Node 24) and npm.

```bash
cd frontend
npm install

# Demo data: in-browser fixtures from src/mocks/, no backend needed
npm run dev:mock          # http://localhost:5173

# Real backend: proxies /api to http://localhost:8000
npm run dev               # set NEXIS_API_URL to proxy elsewhere

npm run build             # tsc -b (type check) + vite build -> dist/
npm run preview           # serve dist/ (also proxies /api)
```

Data source, decided once per page load:

| Situation | Behaviour |
|---|---|
| `VITE_MOCK=1` (`npm run dev:mock`, via `.env.mock`) | Fixtures. A small **Demo data** pill shows in the header. |
| `GET /api/health` answers with JSON | Live backend. |
| `/api/health` fails, times out (3 s) or returns non-JSON | Fixtures, with a **Demo data — backend not reachable** pill and a Retry button. |

## Views

| Route | Shows |
|---|---|
| `/` **Dashboard** | Hero wordmark plus the headline figures from `/api/summary`: transactions scored, open/total alerts, threshold, and PR-AUC ± std with prevalence beside it (never accuracy). Below them, the five highest-scored open alerts and a link to Alerts. |
| `/alerts` **Alerts** | The `/api/alerts` table on a glass panel: rank, risk bar, time, sender → receiver, amount, format, top two reasons (truncated, full text on hover) and status. Status, sort and page size are filters; the table is paginated, and clicking a row opens the alert. |
| `/alerts/:id` **Alert detail** | Risk score vs threshold, feature contributions as diverging bars (gold = up, black = down), rule hits, the evidence graph (important edges highlighted, with a link into the explorer at the alert time), the transaction, sender/receiver context, and the candidate ring. The `source_record_ids` are listed. The **Investigator** button calls POST `/investigate` and shows the summary, each claim with its cited ids, an unverified flag (re-checked client-side against the packet), the llm/template mode and any warnings. **Disposition** offers Escalate / Dismiss / Needs info with a note, and calls POST `/disposition`. The `language_note` stays visible. |
| `/graph/:account?until=…` **Graph** | Cytoscape view of `/api/accounts/:id/network` with controls for hops (1/2), edge limit and cut-off. Node size and gold fill encode risk; the centre account has a black ring. Edge colour runs from white (low risk) to black (high risk) and width scales with log amount. Click a node or edge for details; double-click a node, or press "Recentre", to recentre while keeping `until`. Shows "Only transactions before … are shown" and a **Truncated** pill. The side list comes from `/api/accounts/:id/transactions`. |
| `/rings`, `/rings/:id` **Rings** | Candidate rings from `/api/rings`. Selecting one shows its members and a graph and table of its payments, resolved from each member's transaction list. |
| `/drift` **Drift** | PSI line chart per window (feature PSI gold, score PSI black) with reference lines at warn 0.10 and drift 0.25, plus a status glyph per window. Also an events table with the action taken, and a windows table with the top shifted features. |
| `/models` **Models** | The modelling ladder grouped by rung. Shows PR-AUC, ROC-AUC, recall at FPR 1e-3 and precision at budget as mean ± std, PR-AUC ÷ prevalence, and n_seeds. Rows with fewer than 5 seeds are flagged, and a missing std shows as `n/a*`. Footnote: "A difference smaller than the std is not a result." |

## Design

- **Colours:** vermilion `#DB3A1E` page, gold `#FFA800`, ink `#0B0B0B`, and thin white orbit lines at 25% opacity. `#DB3A1E` keeps black text at about 4.6:1 contrast (`#D9361C` would be 4.49:1).
- **Panels:** the dashboard sits directly on the vermilion. Every other page puts its text on frosted-glass panels (`.glass`: white at 70%, 12px backdrop blur, rounded-3xl, white/40 border, soft shadow), so the animated background shows through, blurred. On the panels, charts use gold and black with light-grey gridlines, and graph edges run from light grey (low risk) to black (high risk).
- **Fonts (Google Fonts):** *Bodoni Moda* (weights 500–600, optical size pinned to 18 so the hairlines stay visible at hero sizes) for display and figures, *Pinyon Script* for the logo, *Inter* for UI text. Numbers use tabular figures, and the wordmark and figures are sized with `clamp()`.
- **Motion:** golden circles drift along the orbits (SVG `animateMotion`). That background is mounted once and never restarts on navigation. Route changes use framer-motion `AnimatePresence` (`mode="wait"`, fade with a slight slide and scale, about 380 ms). `<Routes location>` pins each exiting page to its own URL. Moving within the graph or rings pages keeps the page mounted. Table rows stagger in, and bars and lines grow on load. Everything is static under `prefers-reduced-motion`.
- **No horizontal page scroll:** KPI grids auto-fit and wrap, wide tables scroll inside their own panel, and the app root has `overflow-x: clip` as a safety net.
- **Stack:** Vite 6, React 18, TypeScript 5.9, Tailwind CSS 4 (`@tailwindcss/vite`), framer-motion 12, react-router 7 and cytoscape 3. Cytoscape is code-split and only loads when a graph renders.

## Layout

```
src/
  api.ts            typed client, one function per contract route; mock/live switch
  http.ts           RequestSpec + ApiError shared by client and mocks
  types.ts          TypeScript shapes for every contract payload
  format.ts         number/time formatting, labels, colour encodings
  App.tsx           header (logo, nav, menu), orbit background, routes
  components/       ui primitives, CytoGraph (lazy cytoscape), charts, Orbits
  views/            AlertQueue, AlertDetail, GraphExplorer, Rings, Drift, Models
  mocks/            deterministic fixture world + route handlers (contract-shaped)
```

The mock world (`src/mocks/world.ts`) is seeded and deterministic: 46 accounts,
about 115 transactions, 3 candidate rings and 40 alerts. Every mock endpoint derives
from it, so evidence, graph, rings and queue agree. Like the backend, it only uses
records strictly before the cut-off time.
