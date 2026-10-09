import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// The FastAPI backend serves the contract in docs/api_contract.md under /api.
// In dev we proxy to it so the browser sees a single origin.
const BACKEND = process.env.NEXIS_API_URL ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: BACKEND,
        changeOrigin: true,
      },
    },
  },
  preview: {
    port: 4173,
    proxy: {
      "/api": {
        target: BACKEND,
        changeOrigin: true,
      },
    },
  },
  build: {
    // cytoscape alone is ~400 kB minified; it is split into its own chunk.
    chunkSizeWarningLimit: 700,
    rollupOptions: {
      output: {
        manualChunks: {
          cytoscape: ["cytoscape"],
        },
      },
    },
  },
});
