import { useEffect, useState } from "react";
import { ApiError, getHealth } from "../lib/api";
import { Icon, Spinner } from "./Icon";

type HealthState = {
  kind: "checking" | "online" | "unavailable" | "offline";
  message: string;
};
export function HealthBadge() {
  const [state, setState] = useState<HealthState>({
    kind: "checking",
    message: "Checking API",
  });
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    let active = true;
    let pending = false;
    const controller = new AbortController();
    async function check() {
      if (pending) return;
      pending = true;
      try {
        const health = await getHealth(controller.signal);
        if (active)
          setState({
            kind: "online",
            message: `API online · ${health.model.replaceAll("_", " ")}`,
          });
      } catch (error) {
        if (active)
          setState({
            kind:
              error instanceof ApiError && error.status === 503
                ? "unavailable"
                : "offline",
            message:
              error instanceof ApiError
                ? error.message
                : "Unable to check API health.",
          });
      } finally {
        pending = false;
      }
    }
    void check();
    const interval = setInterval(() => void check(), 30_000);
    return () => {
      active = false;
      controller.abort();
      clearInterval(interval);
    };
  }, [attempt]);
  const online = state.kind === "online";
  const label = {
    checking: "Checking API",
    online: "API online",
    unavailable: "Model unavailable",
    offline: "API unreachable",
  }[state.kind];
  return (
    <div className="flex items-center gap-2">
      <span
        role="status"
        title={state.message}
        className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-medium ${online ? "border-emerald-200 bg-emerald-50 text-emerald-800" : state.kind === "checking" ? "border-line bg-white text-muted" : "border-red-200 bg-red-50 text-red-800"}`}
      >
        {state.kind === "checking" ? (
          <Spinner className="h-3 w-3" />
        ) : (
          <span
            className={`h-1.5 w-1.5 rounded-full ${online ? "bg-emerald-600" : "bg-red-600"}`}
          />
        )}
        {label}
      </span>
      {state.kind !== "online" && state.kind !== "checking" && (
        <button
          type="button"
          onClick={() => {
            setState({ kind: "checking", message: "Checking API" });
            setAttempt((n) => n + 1);
          }}
          className="rounded-full p-2 text-muted transition hover:bg-white hover:text-ink"
          aria-label="Recheck API health"
          title="Recheck API health"
        >
          <Icon name="refresh" className="h-4 w-4" />
        </button>
      )}
    </div>
  );
}
