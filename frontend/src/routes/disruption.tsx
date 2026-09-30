import { createFileRoute, Link, useLocation } from "@tanstack/react-router";
import {
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  LoaderCircle,
  Network,
  Play,
  ShieldAlert,
} from "lucide-react";
import { useEffect, useState } from "react";

import { Navbar } from "@/components/Navbar";
import {
  DisruptionResponse,
  postDisruption,
  readAnalysisId,
  readFailedEdge,
  saveFailedEdge,
} from "@/lib/backend";

export const Route = createFileRoute("/disruption")({
  head: () => ({ meta: [{ title: "Disruption Simulation — ROUTE RESILIENCE" }] }),
  component: DisruptionPage,
});

function Metric({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="border border-border/60 bg-card/70 p-4">
      <p className="font-display text-[0.58rem] tracking-[0.18em] text-muted-foreground uppercase">
        {label}
      </p>
      <p className="mt-2 font-display text-2xl font-bold text-primary">{value}</p>
      {sub && (
        <p className="mt-1 font-display text-[0.58rem] tracking-[0.14em] text-muted-foreground uppercase">
          {sub}
        </p>
      )}
    </div>
  );
}

function DisruptionPage() {
  const { searchStr } = useLocation();
  const queryAnalysisId = new URLSearchParams(searchStr).get("analysis");
  const [analysisId, setAnalysisId] = useState(queryAnalysisId ?? readAnalysisId());
  const [edgeStart, setEdgeStart] = useState(String(readFailedEdge()[0]));
  const [edgeEnd, setEdgeEnd] = useState(String(readFailedEdge()[1]));
  const [result, setResult] = useState<DisruptionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (queryAnalysisId) setAnalysisId(queryAnalysisId);
  }, [queryAnalysisId]);

  const handleSimulate = async () => {
    if (!analysisId) {
      setError("Run an image analysis in Explore before simulating a disruption.");
      return;
    }
    const edge: [number, number] = [Number(edgeStart), Number(edgeEnd)];
    if (!Number.isInteger(edge[0]) || !Number.isInteger(edge[1])) {
      setError("Both failed edge endpoints must be integer node IDs.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      setResult(await postDisruption(analysisId, edge));
      saveFailedEdge(edge);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Disruption request failed.");
    } finally {
      setLoading(false);
    }
  };

  const scenario = result?.scenario;
  const metrics = result?.impact_metrics;

  return (
    <div className="relative min-h-screen bg-background text-foreground">
      <div
        aria-hidden
        className="explore-bg-grid pointer-events-none absolute inset-0 opacity-[0.04]"
      />
      <Navbar />
      <main className="relative z-10 px-6 py-6 sm:px-10 lg:px-14">
        <div className="mb-6 border-b border-border/40 pb-6">
          <p className="mb-1.5 font-display text-[0.58rem] tracking-[0.28em] text-rose-400/80 uppercase">
            FAILURE EXPERIMENT
          </p>
          <h1 className="font-display text-3xl font-light sm:text-4xl">
            Road Disruption <span className="font-medium text-rose-400">Simulation</span>
          </h1>
          <p className="mt-3 max-w-2xl text-sm text-muted-foreground">
            Analysis: {analysisId ?? "No analysis selected"}
          </p>
        </div>
        <section
          className="mb-6 border border-border/60 bg-card/70 p-4"
          aria-label="Failure controls"
        >
          <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
            <div>
              <p className="font-display text-[0.6rem] tracking-widest text-primary uppercase">
                FAILED EDGE
              </p>
              <div className="mt-2 flex items-center gap-2">
                <input
                  aria-label="Failed edge start node"
                  value={edgeStart}
                  onChange={(event) => setEdgeStart(event.target.value)}
                  className="w-24 border border-border/60 bg-background/60 px-3 py-2 text-sm text-foreground"
                />
                <span className="text-muted-foreground">↔</span>
                <input
                  aria-label="Failed edge end node"
                  value={edgeEnd}
                  onChange={(event) => setEdgeEnd(event.target.value)}
                  className="w-24 border border-border/60 bg-background/60 px-3 py-2 text-sm text-foreground"
                />
              </div>
            </div>
            <button
              type="button"
              onClick={handleSimulate}
              disabled={loading}
              className="flex items-center justify-center gap-2 bg-rose-500 px-6 py-3 font-display text-[0.68rem] font-bold tracking-[0.2em] text-white uppercase disabled:opacity-50"
            >
              {loading ? (
                <>
                  <LoaderCircle size={14} className="animate-spin" /> CALLING API...
                </>
              ) : (
                <>
                  <Play size={14} fill="currentColor" /> SIMULATE FAILURE
                </>
              )}
            </button>
          </div>
        </section>
        {error && (
          <div className="mb-6 flex items-start gap-3 border border-amber-500/40 bg-amber-500/5 p-4 text-amber-100">
            <AlertTriangle size={18} className="mt-0.5 text-amber-400" />
            <div>
              <p className="font-display text-[0.62rem] tracking-[0.16em] uppercase">API ERROR</p>
              <p className="mt-2 text-sm">{error}</p>
            </div>
          </div>
        )}
        {!result && !error && (
          <div className="mb-6 border border-dashed border-border/60 p-10 text-center text-muted-foreground">
            <Network className="mx-auto mb-3 text-primary" />
            <p className="font-display text-[0.65rem] tracking-[0.16em] uppercase">
              Run the disruption API to load real impact metrics
            </p>
          </div>
        )}
        {result && metrics && scenario && (
          <>
            <section className="mb-6 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
              <Metric
                label="Efficiency drop"
                value={`${(metrics.efficiency_drop_frac * 100).toFixed(2)}%`}
                sub={`Before ${metrics.before_efficiency.toFixed(6)}`}
              />
              <Metric
                label="Failed edge"
                value={`${scenario.failed_edges[0]?.[0]} ↔ ${scenario.failed_edges[0]?.[1]}`}
                sub="Explicit API scenario"
              />
              <Metric
                label="Components"
                value={String(result.network_state.disrupted.n_components)}
                sub={`+${metrics.newly_disconnected_components} new`}
              />
              <Metric
                label="Disrupted efficiency"
                value={metrics.after_efficiency.toFixed(6)}
                sub="Real backend metric"
              />
              <Metric
                label="OD service"
                value={
                  result.affected_od_service
                    ? `${(result.affected_od_service.disrupted_service_performance * 100).toFixed(1)}%`
                    : "N/A"
                }
                sub="Disrupted performance"
              />
            </section>
            <section className="grid gap-6 lg:grid-cols-2">
              <div className="border border-border/60 bg-card/70 p-5">
                <div className="flex items-center gap-2 text-rose-400">
                  <ShieldAlert size={17} />
                  <p className="font-display text-[0.62rem] tracking-widest uppercase">
                    REAL FAILURE RESULT
                  </p>
                </div>
                <p className="mt-4 text-sm text-muted-foreground">
                  Failed nodes:{" "}
                  {scenario.failed_nodes.length ? scenario.failed_nodes.join(", ") : "none"}
                </p>
                <p className="mt-2 text-sm text-muted-foreground">
                  Baseline: {result.network_state.baseline.n_edges} edges /{" "}
                  {result.network_state.baseline.n_components} components
                </p>
                <p className="mt-2 text-sm text-muted-foreground">
                  Disrupted: {result.network_state.disrupted.n_edges} edges /{" "}
                  {result.network_state.disrupted.n_components} components
                </p>
              </div>
              <div className="border border-border/60 bg-card/70 p-5">
                <div className="flex items-center gap-2 text-primary">
                  <CheckCircle2 size={17} />
                  <p className="font-display text-[0.62rem] tracking-widest uppercase">
                    NEXT API STEPS
                  </p>
                </div>
                <div className="mt-5 flex flex-wrap gap-3">
                  <Link
                    to="/routing"
                    search={{ analysis: analysisId ?? undefined }}
                    className="inline-flex items-center gap-2 border border-primary/70 bg-primary/10 px-4 py-3 font-display text-[0.6rem] tracking-[0.16em] text-primary uppercase"
                  >
                    ROUTE <ArrowRight size={14} />
                  </Link>
                  <Link
                    to="/recovery"
                    search={{ analysis: analysisId ?? undefined }}
                    className="inline-flex items-center gap-2 border border-primary/70 bg-primary/10 px-4 py-3 font-display text-[0.6rem] tracking-[0.16em] text-primary uppercase"
                  >
                    RECOVERY <ArrowRight size={14} />
                  </Link>
                </div>
              </div>
            </section>
          </>
        )}
      </main>
    </div>
  );
}
