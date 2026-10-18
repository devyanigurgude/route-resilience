import { createFileRoute, Link, useLocation } from "@tanstack/react-router";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  HeartPulse,
  HelpCircle,
  LoaderCircle,
  Network,
  RotateCcw,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import { useEffect, useState } from "react";

import {
  getAnalysis,
  getGraph,
  getScenario,
  postRecovery,
  readAnalysisId,
  readRecoveryResult,
  saveRecoveryResult,
  RecoveryResponse,
  SAVED_ANALYSIS_KEY,
  type GraphData,
  type NodeId,
  type StoredScenario,
} from "@/lib/backend";
import { NetworkDiagram } from "@/components/NetworkDiagram";

export const Route = createFileRoute("/recovery")({
  head: () => ({ meta: [{ title: "Route Resilience" }] }),
  component: RecoveryPage,
});

function RecoveryPage() {
  const { searchStr } = useLocation();
  const queryAnalysisId = new URLSearchParams(searchStr).get("analysis");
  const [analysisId, setAnalysisId] = useState(queryAnalysisId ?? readAnalysisId());
  const [result, setResult] = useState<RecoveryResponse | null>(null);
  const [graph, setGraph] = useState<GraphData | null>(null);
  const [imageUrl, setImageUrl] = useState<string | undefined>();
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [activeScenario, setActiveScenario] = useState<StoredScenario | null>(null);

  useEffect(() => {
    if (queryAnalysisId) {
      setAnalysisId(queryAnalysisId);
      window.localStorage.setItem(SAVED_ANALYSIS_KEY, queryAnalysisId);
    }
  }, [queryAnalysisId]);

  useEffect(() => {
    if (!analysisId) {
      setActiveScenario(null);
      return;
    }
    const controller = new AbortController();

    // Restore cached recovery result if available
    const cached = readRecoveryResult(analysisId);
    if (cached) setResult(cached);

    getScenario(analysisId, controller.signal)
      .then((scenario) => {
        if (!controller.signal.aborted) setActiveScenario(scenario);
      })
      .catch((requestError: unknown) => {
        if (!controller.signal.aborted) {
          setActiveScenario(null);
          setError(
            requestError instanceof Error
              ? requestError.message
              : "Unable to load disruption scenario.",
          );
        }
      });
    return () => controller.abort();
  }, [analysisId]);

  useEffect(() => {
    if (!analysisId) return;
    const controller = new AbortController();
    getAnalysis<{
      analysis_id: string;
      input_image_ref?: string;
      artifacts: { original?: string; healed_graph_json?: string };
    }>(analysisId, controller.signal)
      .then((analysis) => {
        if (!analysis.artifacts.healed_graph_json) throw new Error("Analysis graph unavailable.");
        setImageUrl(analysis.artifacts.original ?? analysis.input_image_ref);
        return getGraph<GraphData>(analysis.artifacts.healed_graph_json, controller.signal);
      })
      .then((payload) => {
        if (!controller.signal.aborted) setGraph(payload);
      })
      .catch(() => {
        if (!controller.signal.aborted) setGraph(null);
      });
    return () => controller.abort();
  }, [analysisId]);

  const runRecovery = async () => {
    if (!analysisId) {
      setError("Please run an image analysis first.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      if (!activeScenario) {
        setError("Please simulate a disruption before computing recovery.");
        return;
      }
      const response = await postRecovery(
        analysisId,
        activeScenario.failedEdges,
        activeScenario.failedNodes,
      );
      saveRecoveryResult(analysisId, response);
      setResult(response);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Recovery request failed.");
    } finally {
      setLoading(false);
    }
  };

  const failedDescription =
    activeScenario?.failedEdges.length
      ? activeScenario.failedEdges.map((edge) => `#${edge[0]} ↔ #${edge[1]}`).join(", ")
      : activeScenario?.failedNodes.length
        ? `Node #${activeScenario.failedNodes.join(", ")}`
        : "None";

  return (
    <main className="workflow-page mx-auto min-h-[calc(100vh-104px)] w-full max-w-6xl px-5 py-8 text-foreground sm:px-8">
      {/* Header */}
      <div className="mb-6 border-b border-border pb-5 flex flex-col sm:flex-row sm:items-end sm:justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center rounded-md bg-teal-500/10 px-2 py-0.5 text-xs font-semibold text-teal-700 dark:text-teal-300 border border-teal-500/20">
              05 · Recover
            </span>
            <span className="text-xs text-muted-foreground">Step 5 of 5</span>
          </div>
          <h1 className="mt-2 text-2xl sm:text-3xl font-bold tracking-tight text-foreground">
            Network Recovery & Resilience Optimization
          </h1>
          <p className="mt-1.5 text-sm text-muted-foreground max-w-2xl">
            Simulate repair sequencing and asset restoration to calculate the Resilience Index (RI)
            and evaluate post-disruption network recovery.
          </p>
        </div>

        {analysisId && (
          <span className="text-xs font-mono text-muted-foreground bg-muted/60 px-3 py-1.5 rounded-lg border border-border self-start sm:self-end">
            Analysis #{analysisId.slice(0, 8)}
          </span>
        )}
      </div>

      {/* Control Card */}
      <section className="rounded-xl border border-border bg-card p-5 sm:p-6 shadow-xs mb-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-5">
          <div>
            <h2 className="text-xs font-bold text-foreground uppercase tracking-wider flex items-center gap-1.5">
              <HeartPulse size={15} className="text-teal-600 dark:text-teal-400" />
              <span>Restoration Policy</span>
            </h2>
            <p className="mt-1 text-xs text-muted-foreground">
              Disrupted Asset to Restore:{" "}
              <strong className="text-foreground">{failedDescription}</strong>
            </p>
            <p className="text-[0.68rem] text-muted-foreground mt-0.5">
              Sequential repair ordering prioritizes critical infrastructure bridges to maximize
              connectivity recovery.
            </p>
          </div>

          <button
            type="button"
            onClick={runRecovery}
            disabled={loading || !analysisId || !activeScenario}
            className="inline-flex items-center justify-center gap-2 rounded-xl bg-teal-700 hover:bg-teal-800 dark:bg-teal-600 dark:hover:bg-teal-700 px-6 py-3 text-xs font-semibold text-white shadow-xs transition disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? (
              <>
                <LoaderCircle size={15} className="animate-spin" />
                <span>Optimizing Recovery...</span>
              </>
            ) : (
              <>
                <RotateCcw size={15} />
                <span>Execute Recovery Plan</span>
              </>
            )}
          </button>
        </div>
      </section>

      {/* Error Banner */}
      {error && (
        <div className="mb-6 flex items-start gap-3 rounded-xl border border-amber-500/30 bg-amber-500/10 p-4 text-xs sm:text-sm text-amber-800 dark:text-amber-300">
          <AlertTriangle size={18} className="text-amber-600 dark:text-amber-400 shrink-0 mt-0.5" />
          <p>{error}</p>
        </div>
      )}

      {/* Empty State when no disruption exists */}
      {!activeScenario && !result && (
        <div className="rounded-xl border border-dashed border-border bg-card p-12 text-center max-w-md mx-auto">
          <div className="w-12 h-12 rounded-2xl bg-amber-500/10 text-amber-600 dark:text-amber-400 flex items-center justify-center mx-auto mb-3">
            <AlertTriangle size={24} />
          </div>
          <h3 className="text-sm font-bold text-foreground">No Active Disruption</h3>
          <p className="mt-1 text-xs text-muted-foreground">
            Simulate a road or node failure in the Disrupt step before running recovery
            optimization.
          </p>
          <Link
            to="/disruption"
            search={{ analysis: analysisId ?? undefined }}
            className="mt-4 inline-flex items-center gap-1.5 rounded-xl bg-teal-700 hover:bg-teal-800 dark:bg-teal-600 dark:hover:bg-teal-700 px-4 py-2 text-xs font-semibold text-white shadow-xs transition"
          >
            <span>Go to Disrupt</span>
            <ArrowRight size={14} />
          </Link>
        </div>
      )}

      {/* Recovery Results */}
      {result && (
        <div className="space-y-6">
          {/* Plain-Language Resilience Summary Banner */}
          <div className="rounded-xl border border-teal-500/30 bg-teal-500/10 p-5 sm:p-6 text-foreground">
            <div className="flex items-start gap-3.5">
              <div className="rounded-xl bg-teal-500/20 p-2 text-teal-700 dark:text-teal-300 shrink-0">
                <ShieldCheck className="h-5 w-5" />
              </div>
              <div className="flex-1 min-w-0">
                <h3 className="text-sm font-bold text-foreground">
                  Resilience Recovery Index:{" "}
                  {(result.resilience_metrics.RI * 100).toFixed(1)}% Restored
                </h3>
                <p className="mt-1 text-xs sm:text-sm leading-relaxed text-muted-foreground">
                  Restoring the failed asset returned network efficiency from{" "}
                  {(result.resilience_metrics.P_D * 100).toFixed(1)}% back to{" "}
                  {(result.resilience_metrics.P_R * 100).toFixed(1)}%, achieving a Recovery Gain of{" "}
                  +{(result.resilience_metrics.RecoveryGain * 100).toFixed(1)}%.
                </p>
              </div>
            </div>
          </div>

          {/* Metric Cards Grid */}
          <section className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
            <div className="rounded-xl border border-border bg-card p-3.5 shadow-xs">
              <span className="block text-[0.62rem] font-bold text-muted-foreground uppercase tracking-wide">
                Resilience Index (RI)
              </span>
              <p className="mt-1 text-lg font-bold text-teal-700 dark:text-teal-300">
                {result.resilience_metrics.RI.toFixed(3)}
              </p>
              <span className="text-[0.62rem] text-muted-foreground">Global retention ratio</span>
            </div>

            <div className="rounded-xl border border-border bg-card p-3.5 shadow-xs">
              <span className="block text-[0.62rem] font-bold text-muted-foreground uppercase tracking-wide">
                Disrupted Efficiency
              </span>
              <p className="mt-1 text-lg font-bold text-destructive">
                {(result.resilience_metrics.P_D * 100).toFixed(1)}%
              </p>
              <span className="text-[0.62rem] text-muted-foreground">During active failure</span>
            </div>

            <div className="rounded-xl border border-border bg-card p-3.5 shadow-xs">
              <span className="block text-[0.62rem] font-bold text-muted-foreground uppercase tracking-wide">
                Recovered Efficiency
              </span>
              <p className="mt-1 text-lg font-bold text-teal-700 dark:text-teal-300">
                {(result.resilience_metrics.P_R * 100).toFixed(1)}%
              </p>
              <span className="text-[0.62rem] text-muted-foreground">Post-restoration state</span>
            </div>

            <div className="rounded-xl border border-border bg-card p-3.5 shadow-xs">
              <span className="block text-[0.62rem] font-bold text-muted-foreground uppercase tracking-wide">
                Recovery Gain
              </span>
              <p className="mt-1 text-lg font-bold text-teal-600 dark:text-teal-400">
                +{(result.resilience_metrics.RecoveryGain * 100).toFixed(1)}%
              </p>
              <span className="text-[0.62rem] text-muted-foreground">Total efficiency recaptured</span>
            </div>
          </section>

          {/* Restoration Sequence & State Breakdown */}
          <section className="grid gap-4 md:grid-cols-2">
            <div className="rounded-xl border border-border bg-card p-5 shadow-xs">
              <h3 className="text-xs font-bold text-foreground uppercase tracking-wider mb-3 flex items-center gap-1.5">
                <CheckCircle2 size={15} className="text-teal-600 dark:text-teal-400" />
                <span>Restored Infrastructure</span>
              </h3>
              <div className="space-y-2 text-xs text-muted-foreground">
                <div className="flex justify-between py-1.5 border-b border-border/50">
                  <span>Restored Roads:</span>
                  <span className="font-mono font-semibold text-foreground">
                    {result.recovery_summary.recovered_edges.length
                      ? result.recovery_summary.recovered_edges
                          .map((e) => `#${e[0]} ↔ #${e[1]}`)
                          .join(", ")
                      : "None"}
                  </span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-border/50">
                  <span>Assets Restored:</span>
                  <span className="font-semibold text-foreground">
                    {result.recovery_summary.assets_restored} asset
                  </span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-border/50">
                  <span>Final Components:</span>
                  <span className="font-semibold text-foreground">
                    {result.final_recovered_state.n_components} connected zone(s)
                  </span>
                </div>
                <div className="flex justify-between py-1.5">
                  <span>Active Network:</span>
                  <span className="font-semibold text-foreground">
                    {result.final_recovered_state.n_nodes} nodes ·{" "}
                    {result.final_recovered_state.n_edges} roads
                  </span>
                </div>
              </div>
            </div>

            <div className="rounded-xl border border-border bg-card p-5 shadow-xs">
              <h3 className="text-xs font-bold text-foreground uppercase tracking-wider mb-3 flex items-center gap-1.5">
                <RotateCcw size={14} className="text-teal-600 dark:text-teal-400" />
                <span>Sequential Repair Order</span>
              </h3>
              {result.restoration_sequence.length === 0 ? (
                <p className="text-xs text-muted-foreground">
                  No sequential steps required for this scenario.
                </p>
              ) : (
                <div className="space-y-2 text-xs">
                  {result.restoration_sequence.map((step) => (
                    <div
                      key={step.step}
                      className="flex items-center justify-between p-2.5 rounded-xl bg-muted/40 border border-border"
                    >
                      <span className="font-semibold text-foreground">
                        Step {step.step}: Restored {step.restored_asset_type}{" "}
                        <span className="font-mono text-teal-600 dark:text-teal-400">
                          {Array.isArray(step.restored_asset)
                            ? `#${step.restored_asset[0]} ↔ #${step.restored_asset[1]}`
                            : `#${step.restored_asset}`}
                        </span>
                      </span>
                      <span className="font-mono text-[0.68rem] text-muted-foreground">
                        {(step.efficiency * 100).toFixed(1)}% eff
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </section>

          {/* Diagram showing restored edges */}
          {graph && (
            <section className="rounded-xl border border-border bg-card p-5 sm:p-6 shadow-xs">
              <div className="flex items-center justify-between gap-3 mb-4">
                <div>
                  <h3 className="text-sm font-bold text-foreground">Recovered Network Topology</h3>
                  <p className="text-xs text-muted-foreground">
                    Restored road segments are re-energized in vibrant emerald green.
                  </p>
                </div>
              </div>

              <NetworkDiagram
                graph={graph}
                imageUrl={imageUrl}
                restoredEdges={result.recovery_summary.recovered_edges as [NodeId, NodeId][]}
              />
            </section>
          )}

          {/* Navigation Action */}
          <div className="flex items-center justify-between pt-2">
            <Link
              to="/routing"
              search={{ analysis: analysisId ?? undefined }}
              className="text-xs font-semibold text-muted-foreground hover:text-foreground transition"
            >
              ← Back to Rerouting
            </Link>

            <Link
              to="/network"
              search={{ analysis: analysisId ?? undefined }}
              className="inline-flex items-center gap-2 rounded-xl bg-teal-700 hover:bg-teal-800 dark:bg-teal-600 dark:hover:bg-teal-700 px-6 py-3 text-sm font-semibold text-white shadow-xs transition"
            >
              <span>Return to Network Overview</span>
              <ArrowRight size={16} />
            </Link>
          </div>
        </div>
      )}
    </main>
  );
}
