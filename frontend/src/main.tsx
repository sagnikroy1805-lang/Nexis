import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { dataSource } from "./api";
import App from "./App";
import "./styles.css";

// Start probing /api/health immediately; every request waits on this decision.
void dataSource();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </StrictMode>,
);
