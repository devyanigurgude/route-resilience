import { createFileRoute, useLocation } from "@tanstack/react-router";
import { AlertTriangle, CheckCircle2, LoaderCircle, RotateCcw } from "lucide-react";
import { useEffect, useState } from "react";

import { Navbar } from "@/components/Navbar";
import { postRecovery, readAnalysisId, readFailedEdge, RecoveryResponse } from "@/lib/backend";

export const Route = createFileRoute("/recovery")({
  head: () => ({ meta: [{ title: "Recovery Analysis — ROUTE RESILIENCE" }] }),
  component: RecoveryPage,
});

function RecoveryPage() {
  const { searchStr } = useLocation();
  const queryAnalysisId = new URLSearchParams(searchStr).get("analysis");
  const [analysisId, setAnalysisId] = useState(queryAnalysisId ?? readAnalysisId());
  const [result, setResult] = useState<RecoveryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (queryAnalysisId) setAnalysisId(queryAnalysisId);
  }, [queryAnalysisId]);

  const runRecovery = async () => {
    if (!analysisId) {
      setError("Run an image analysis in Explore before recovery.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      setResult(await postRecovery(analysisId, readFailedEdge()));
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Recovery request failed.");
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
            RECOVERY ANALYSIS
          </p>
          <h1 className="font-display text-3xl font-light sm:text-4xl">
            Network <span className="font-medium text-primary">Recovery</span>
          </h1>
          <p className="mt-3 text-sm text-muted-foreground">
            Analysis: {analysisId ?? "No analysis selected"} · Failed edge:{" "}
            {readFailedEdge().join(" ↔ ")}
          </p>
        </div>
        <section className="mb-6 flex flex-col gap-4 border border-border/60 bg-card/70 p-5 md:flex-row md:items-center md:justify-between">
          <div>
            <p className="font-display text-[0.62rem] tracking-widest text-primary uppercase">
              REAL RECOVERY ENGINE
            </p>
            <p className="mt-2 text-sm text-muted-foreground">
              Restore the persisted failed edge with explicit failure ordering.
            </p>
          </div>
          <button
            type="button"
            onClick={runRecovery}
            disabled={loading}
            className="flex items-center justify-center gap-2 bg-primary px-6 py-3 font-display text-[0.68rem] font-bold tracking-[0.2em] text-primary-foreground uppercase disabled:opacity-50"
          >
            {loading ? (
              <>
                <LoaderCircle size={14} className="animate-spin" /> RECOVERING...
              </>
            ) : (
              <>
                <RotateCcw size={14} /> RUN RECOVERY
              </>
            )}
          </button>
        </section>
        {error && (
          <div className="mb-6 flex gap-3 border border-amber-500/40 bg-amber-500/5 p-4 text-amber-100">
            <AlertTriangle size={18} className="text-amber-400" />
            <p className="text-sm">{error}</p>
          </div>
        )}
        {result && (
          <>
            <section className="mb-6 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
              <Metric label="Recovery index" value={result.resilience_metrics.RI.toFixed(3)} />
              <Metric
                label="Baseline performance"
                value={result.resilience_metrics.P_baseline.toFixed(3)}
              />
              <Metric
                label="Disrupted performance"
                value={result.resilience_metrics.P_D.toFixed(3)}
              />
              <Metric
                label="Recovered performance"
                value={result.resilience_metrics.P_R.toFixed(3)}
              />
              <Metric
                label="Recovery gain"
                value={result.resilience_metrics.RecoveryGain.toFixed(3)}
              />
            </section>
            <section className="grid gap-6 lg:grid-cols-2">
              <div className="border border-border/60 bg-card/70 p-5">
                <div className="flex items-center gap-2 text-primary">
                  <CheckCircle2 size={17} />
                  <p className="font-display text-[0.62rem] tracking-widest uppercase">
                    REAL RECOVERY RESULT
                  </p>
                </div>
                <p className="mt-4 text-sm text-muted-foreground">
                  Restored edges:{" "}
                  {result.recovery_summary.recovered_edges
                    .map((edge) => edge.join(" ↔ "))
                    .join(", ") || "none"}
                </p>
                <p className="mt-2 text-sm text-muted-foreground">
                  Assets restored: {result.recovery_summary.assets_restored}
                </p>
                <p className="mt-2 text-sm text-muted-foreground">
                  Recovered efficiency: {result.final_recovered_state.efficiency.toFixed(6)}
                </p>
              </div>
              <div className="border border-border/60 bg-card/70 p-5">
                <p className="font-display text-[0.62rem] tracking-widest text-primary uppercase">
                  RESTORATION SEQUENCE
                </p>
                {result.restoration_sequence.map((step) => (
                  <div
                    key={step.step}
                    className="mt-4 border border-border/50 bg-background/40 p-3 text-sm text-muted-foreground"
                  >
                    Step {step.step}: restored {step.restored_asset_type}{" "}
                    {Array.isArray(step.restored_asset)
                      ? step.restored_asset.join(" ↔ ")
                      : step.restored_asset}{" "}
                    · efficiency {step.efficiency.toFixed(6)}
                  </div>
                ))}
              </div>
            </section>
          </>
        )}
      </main>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="border border-border/60 bg-card/70 p-4">
      <p className="font-display text-[0.58rem] tracking-[0.18em] text-muted-foreground uppercase">
        {label}
      </p>
      <p className="mt-2 font-display text-2xl font-bold text-primary">{value}</p>
    </div>
  );
}
