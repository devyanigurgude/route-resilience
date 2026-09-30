import { createFileRoute, Link, useLocation } from "@tanstack/react-router";
import {
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  LoaderCircle,
  Route as RouteIcon,
} from "lucide-react";
import { useEffect, useState } from "react";

import { Navbar } from "@/components/Navbar";
import { postRoute, readAnalysisId, readFailedEdge, RouteResponse } from "@/lib/backend";

export const Route = createFileRoute("/routing")({
  head: () => ({ meta: [{ title: "Route Query — ROUTE RESILIENCE" }] }),
  component: RoutePage,
});

function RoutePage() {
  const { searchStr } = useLocation();
  const queryAnalysisId = new URLSearchParams(searchStr).get("analysis");
  const [analysisId, setAnalysisId] = useState(queryAnalysisId ?? readAnalysisId());
  const [result, setResult] = useState<RouteResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [origin, setOrigin] = useState("0");
  const [destination, setDestination] = useState("1");

  useEffect(() => {
    if (queryAnalysisId) setAnalysisId(queryAnalysisId);
  }, [queryAnalysisId]);

  const runRoute = async () => {
    if (!analysisId) {
      setError("Run an image analysis in Explore before querying a route.");
      return;
    }
    const start = Number(origin);
    const end = Number(destination);
    if (!Number.isInteger(start) || !Number.isInteger(end)) {
      setError("Origin and destination must be integer node IDs.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      setResult(await postRoute(analysisId, start, end, readFailedEdge()));
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Route request failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="relative min-h-screen bg-background text-foreground">
      <div
        aria-hidden
        className="explore-bg-grid pointer-events-none absolute inset-0 opacity-[0.04]"
      />
      <Navbar />
      <main className="relative z-10 px-6 py-6 sm:px-10 lg:px-14">
        <div className="mb-6 border-b border-border/40 pb-6">
          <p className="mb-1.5 font-display text-[0.58rem] tracking-[0.28em] text-primary/70 uppercase">
            ROUTING QUERY
          </p>
          <h1 className="font-display text-3xl font-light sm:text-4xl">
            Disrupted <span className="font-medium text-primary">Route</span>
          </h1>
          <p className="mt-3 text-sm text-muted-foreground">
            Analysis: {analysisId ?? "No analysis selected"}
          </p>
        </div>
        <section className="mb-6 border border-border/60 bg-card/70 p-4">
          <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
            <div>
              <p className="font-display text-[0.6rem] tracking-widest text-primary uppercase">
                ORIGIN → DESTINATION
              </p>
              <div className="mt-2 flex items-center gap-2">
                <input
                  aria-label="Origin node"
                  value={origin}
                  onChange={(event) => setOrigin(event.target.value)}
                  className="w-24 border border-border/60 bg-background/60 px-3 py-2 text-sm"
                />
                <span className="text-muted-foreground">→</span>
                <input
                  aria-label="Destination node"
                  value={destination}
                  onChange={(event) => setDestination(event.target.value)}
                  className="w-24 border border-border/60 bg-background/60 px-3 py-2 text-sm"
                />
              </div>
            </div>
            <button
              type="button"
              onClick={runRoute}
              disabled={loading}
              className="flex items-center justify-center gap-2 bg-primary px-6 py-3 font-display text-[0.68rem] font-bold tracking-[0.2em] text-primary-foreground uppercase disabled:opacity-50"
            >
              {loading ? (
                <>
                  <LoaderCircle size={14} className="animate-spin" /> ROUTING...
                </>
              ) : (
                <>
                  <RouteIcon size={14} /> QUERY ROUTE
                </>
              )}
            </button>
          </div>
        </section>
        {error && (
          <div className="mb-6 flex gap-3 border border-amber-500/40 bg-amber-500/5 p-4 text-amber-100">
            <AlertTriangle size={18} className="text-amber-400" />
            <p className="text-sm">{error}</p>
          </div>
        )}
        {result && (
          <section className="grid gap-4 lg:grid-cols-3">
            <div className="border border-border/60 bg-card/70 p-5 lg:col-span-2">
              <div className="flex items-center gap-2 text-primary">
                <CheckCircle2 size={17} />
                <p className="font-display text-[0.62rem] tracking-widest uppercase">
                  REAL ROUTE RESPONSE
                </p>
              </div>
              <p className="mt-5 text-sm text-muted-foreground">State: {result.current_state}</p>
              <p className="mt-2 text-sm text-muted-foreground">
                Path:{" "}
                {result.route.length
                  ? result.route.join(" → ")
                  : "No route available after disruption"}
              </p>
              <p className="mt-2 text-sm text-muted-foreground">
                Failed edge: {readFailedEdge().join(" ↔ ")}
              </p>
            </div>
            <div className="border border-border/60 bg-card/70 p-5">
              <p className="font-display text-[0.6rem] tracking-widest text-primary uppercase">
                ROUTE METRICS
              </p>
              <p className="mt-4 font-display text-2xl font-bold text-primary">
                {result.route_exists ? `${result.route_length?.toFixed(2)} px` : "UNREACHABLE"}
              </p>
              <p className="mt-2 text-sm text-muted-foreground">
                Baseline: {result.baseline_route_length?.toFixed(2) ?? "N/A"} px
              </p>
              <p className="mt-2 text-sm text-muted-foreground">
                Detour ratio: {result.detour_ratio?.toFixed(3) ?? "N/A"}
              </p>
            </div>
          </section>
        )}
        <div className="mt-6">
          <Link
            to="/recovery"
            search={{ analysis: analysisId ?? undefined }}
            className="inline-flex items-center gap-2 border border-primary/70 bg-primary/10 px-4 py-3 font-display text-[0.6rem] tracking-[0.16em] text-primary uppercase"
          >
            RECOVERY <ArrowRight size={14} />
          </Link>
        </div>
      </main>
    </div>
  );
}
