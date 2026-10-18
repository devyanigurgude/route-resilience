import { createFileRoute, Link, useLocation, useNavigate } from "@tanstack/react-router";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  HelpCircle,
  Layers,
  LoaderCircle,
  Network,
  Play,
  RotateCcw,
  Route as RouteIcon,
  ShieldAlert,
  Sliders,
  Sparkles,
  Zap,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import {
  clearPendingFailure,
  clearPendingNodeFailure,
  DisruptionResponse,
  getAnalysis,
  getCriticality,
  getGraph,
  getScenario,
  postDisruption,
  postNodeDisruption,
  readAnalysisId,
  readPendingFailure,
  readPendingNodeFailure,
  saveScenario,
  SAVED_ANALYSIS_KEY,
  type FailedEdge,
  type GraphData,
  type NodeId,
} from "@/lib/backend";
import { NetworkDiagram } from "@/components/NetworkDiagram";

export const Route = createFileRoute("/disruption")({
  head: () => ({
    meta: [
      { title: "Route Resilience" },
      {
        name: "description",
        content: "Simulate targeted road or node failures and evaluate impact on network connectivity.",
      },
    ],
  }),
  component: DisruptionPage,
});

function DisruptionPage() {
  const navigate = useNavigate();
  const { searchStr } = useLocation();
  const queryAnalysisId = new URLSearchParams(searchStr).get("analysis");
  const queryEdge = new URLSearchParams(searchStr).get("edge");
  const queryNode = new URLSearchParams(searchStr).get("node");

  const [analysisId, setAnalysisId] = useState(queryAnalysisId ?? readAnalysisId());
  const [targetType, setTargetType] = useState<"edge" | "node">("edge");

  // Edge failure selection
  const [edgeStart, setEdgeStart] = useState("");
  const [edgeEnd, setEdgeEnd] = useState("");

  // Node failure selection
  const [selectedNodeId, setSelectedNodeId] = useState("");

  const [result, setResult] = useState<DisruptionResponse | null>(null);
  const [graph, setGraph] = useState<GraphData | null>(null);
  const [imageUrl, setImageUrl] = useState<string | undefined>();
  const [criticalityReport, setCriticalityReport] = useState<any | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (queryAnalysisId) {
      setAnalysisId(queryAnalysisId);
      window.localStorage.setItem(SAVED_ANALYSIS_KEY, queryAnalysisId);
    }
  }, [queryAnalysisId]);

  // Load existing scenario or pending selections
  useEffect(() => {
    if (!analysisId) return;

    const controller = new AbortController();
    const loadState = async () => {
      try {
        let edgeFromQuery: [string, string] | null = null;
        if (queryEdge) {
          const parsed = JSON.parse(queryEdge);
          if (Array.isArray(parsed) && parsed.length === 2) {
            edgeFromQuery = [String(parsed[0]), String(parsed[1])];
          }
        }

        const scenario = await getScenario(analysisId, controller.signal);
        if (controller.signal.aborted) return;

        if (scenario?.disruption) {
          setResult(scenario.disruption);
        }

        if (queryNode) {
          setTargetType("node");
          setSelectedNodeId(queryNode);
        } else if (edgeFromQuery) {
          setTargetType("edge");
          setEdgeStart(edgeFromQuery[0]);
          setEdgeEnd(edgeFromQuery[1]);
        } else {
          // Check pending storage
          const pendingNode = readPendingNodeFailure(analysisId);
          const pendingEdge = readPendingFailure(analysisId);

          if (pendingNode !== null) {
            setTargetType("node");
            setSelectedNodeId(String(pendingNode));
          } else if (pendingEdge) {
            setTargetType("edge");
            setEdgeStart(String(pendingEdge[0]));
            setEdgeEnd(String(pendingEdge[1]));
          } else if (scenario?.failedEdges?.length) {
            setTargetType("edge");
            setEdgeStart(String(scenario.failedEdges[0][0]));
            setEdgeEnd(String(scenario.failedEdges[0][1]));
          } else if (scenario?.failedNodes?.length) {
            setTargetType("node");
            setSelectedNodeId(String(scenario.failedNodes[0]));
          }
        }
      } catch (err) {
        if (!controller.signal.aborted) {
          setError("Failed to load scenario details.");
        }
      }
    };

    void loadState();
    return () => controller.abort();
  }, [analysisId, queryEdge, queryNode]);

  // Load Graph & Criticality for intuitive selection
  useEffect(() => {
    if (!analysisId) return;

    const controller = new AbortController();
    getAnalysis<{
      analysis_id: string;
      input_image_ref?: string;
      artifacts: { original?: string; healed_graph_json?: string; criticality?: string };
    }>(analysisId, controller.signal)
      .then((analysis) => {
        if (!analysis.artifacts.healed_graph_json) throw new Error("Graph data unavailable.");
        setImageUrl(analysis.artifacts.original ?? analysis.input_image_ref);
        return getGraph<GraphData>(analysis.artifacts.healed_graph_json, controller.signal);
      })
      .then((g) => {
        if (!controller.signal.aborted) setGraph(g);
      })
      .catch(() => {
        if (!controller.signal.aborted) setGraph(null);
      });

    getCriticality<{ criticality: any }>(analysisId, controller.signal)
      .then((crit) => {
        if (!controller.signal.aborted) setCriticalityReport(crit.criticality);
      })
      .catch(() => {
        if (!controller.signal.aborted) setCriticalityReport(null);
      });

    return () => controller.abort();
  }, [analysisId]);

  // Candidate Critical Roads & Junctions list for quick selection
  const criticalRoadOptions = useMemo(() => {
    if (!criticalityReport?.edge_criticality || !graph) return [];
    return criticalityReport.edge_criticality.slice(0, 8).map((item: any, idx: number) => {
      const u = item.edge_id[0];
      const v = item.edge_id[1];
      return {
        u: String(u),
        v: String(v),
        label: `Road #${u} ↔ #${v} (Criticality Rank #${idx + 1}${
          item.is_bridge ? ", Bridge" : ""
        })`,
      };
    });
  }, [criticalityReport, graph]);

  const junctionNodeOptions = useMemo(() => {
    if (!graph) return [];
    const degMap = new Map<string, number>();
    graph.edges.forEach((e) => {
      degMap.set(String(e.source), (degMap.get(String(e.source)) || 0) + 1);
      degMap.set(String(e.target), (degMap.get(String(e.target)) || 0) + 1);
    });

    const junctions: Array<{ id: string; degree: number; label: string }> = [];
    degMap.forEach((deg, id) => {
      if (deg >= 2) {
        junctions.push({
          id,
          degree: deg,
          label: `Node #${id} (${deg >= 3 ? "Major Junction" : "Way-node"}, ${deg} connections)`,
        });
      }
    });
    return junctions.sort((a, b) => b.degree - a.degree).slice(0, 10);
  }, [graph]);

  // Simulate Disruption
  const handleSimulate = async () => {
    if (!analysisId) {
      setError("Please run an image analysis first.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      let response: DisruptionResponse;
      if (targetType === "node") {
        if (!selectedNodeId) {
          throw new Error("Please select a junction node to simulate failure.");
        }
        response = await postNodeDisruption(analysisId, selectedNodeId);
        clearPendingNodeFailure(analysisId);
      } else {
        if (!edgeStart || !edgeEnd) {
          throw new Error("Please select a road segment with both endpoint nodes.");
        }
        const edge: [string, string] = [edgeStart, edgeEnd];
        response = await postDisruption(analysisId, edge);
        clearPendingFailure(analysisId);
      }

      saveScenario(analysisId, response);
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Disruption request failed.");
    } finally {
      setLoading(false);
    }
  };

  const handleDiagramEdgeClick = (edge: FailedEdge) => {
    setTargetType("edge");
    setEdgeStart(String(edge[0]));
    setEdgeEnd(String(edge[1]));
  };

  const scenario = result?.scenario;
  const metrics = result?.impact_metrics;
  const failedEdges = scenario?.failed_edges ?? [];
  const failedNodes = scenario?.failed_nodes ?? [];

  // Plain language impact calculation
  const efficiencyDropPct = metrics ? (metrics.efficiency_drop_frac * 100).toFixed(1) : "0";
  const newDisconnections = metrics?.newly_disconnected_components ?? 0;
  const odService = result?.affected_od_service;

  return (
    <main className="workflow-page mx-auto min-h-[calc(100vh-104px)] w-full max-w-6xl px-5 py-8 text-foreground sm:px-8">
      {/* Page Header */}
      <div className="mb-6 border-b border-border pb-5 flex flex-col sm:flex-row sm:items-end sm:justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center rounded-md bg-destructive/10 px-2 py-0.5 text-xs font-semibold text-destructive border border-destructive/20">
              03 · Disrupt
            </span>
            <span className="text-xs text-muted-foreground">Step 3 of 5</span>
          </div>
          <h1 className="mt-2 text-2xl sm:text-3xl font-bold tracking-tight text-foreground">
            Simulate Road & Junction Failures
          </h1>
          <p className="mt-1.5 text-sm text-muted-foreground max-w-2xl">
            Evaluate how the loss of specific lifelines or critical intersections degrades overall
            connectivity and fragments travel corridors.
          </p>
        </div>

        {analysisId && (
          <span className="text-xs font-mono text-muted-foreground bg-muted/40 px-3 py-1.5 rounded-lg border border-border self-start sm:self-end">
            Analysis #{analysisId.slice(0, 8)}
          </span>
        )}
      </div>

      {/* Control Card: Target Selection (Requirement 4: Obvious Selection) */}
      <section className="rounded-xl border border-border bg-card p-5 sm:p-6 shadow-xs mb-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-border pb-4 mb-5">
          <div>
            <h2 className="text-sm font-bold text-foreground uppercase tracking-wide">
              Select Failure Target
            </h2>
            <p className="text-xs text-muted-foreground">
              Choose whether to sever a single road segment or disable an entire intersection.
            </p>
          </div>

          {/* Toggle between Road and Node */}
          <div className="flex items-center p-1 bg-muted/40 rounded-xl border border-border text-xs font-semibold self-start sm:self-auto">
            <button
              type="button"
              onClick={() => setTargetType("edge")}
              className={`px-3 py-1.5 rounded-lg transition-colors ${
                targetType === "edge"
                  ? "bg-card text-destructive shadow-2xs border border-border/80 font-bold"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              Road Segment
            </button>
            <button
              type="button"
              onClick={() => setTargetType("node")}
              className={`px-3 py-1.5 rounded-lg transition-colors ${
                targetType === "node"
                  ? "bg-card text-destructive shadow-2xs border border-border/80 font-bold"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              Junction / Node
            </button>
          </div>
        </div>

        {/* Inputs depending on Target Type */}
        {targetType === "edge" ? (
          <div className="space-y-4">
            {/* Quick dropdown for top critical roads */}
            {criticalRoadOptions.length > 0 && (
              <div>
                <label className="block text-xs font-semibold text-foreground mb-1">
                  Quick Select: Most Critical Lifeline Roads
                </label>
                <select
                  value={`${edgeStart}-${edgeEnd}`}
                  onChange={(e) => {
                    const found = criticalRoadOptions.find(
                      (o: any) => `${o.u}-${o.v}` === e.target.value,
                    );
                    if (found) {
                      setEdgeStart(found.u);
                      setEdgeEnd(found.v);
                    }
                  }}
                  className="w-full rounded-lg border border-border bg-card px-3 py-2 text-xs text-foreground shadow-2xs focus:border-primary focus:ring-1 focus:ring-primary outline-none"
                >
                  <option value="">-- Choose a high-criticality road to disrupt --</option>
                  {criticalRoadOptions.map((opt: any) => (
                    <option key={`${opt.u}-${opt.v}`} value={`${opt.u}-${opt.v}`}>
                      {opt.label}
                    </option>
                  ))}
                </select>
              </div>
            )}

            {/* Custom endpoints */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
              <div>
                <label className="block text-xs font-medium text-muted-foreground mb-1">
                  Endpoint Node A:
                </label>
                <input
                  type="text"
                  value={edgeStart}
                  onChange={(e) => setEdgeStart(e.target.value)}
                  placeholder="e.g. 12"
                  className="w-full rounded-lg border border-border bg-card px-3.5 py-2 text-sm text-foreground shadow-2xs focus:border-primary focus:ring-1 focus:ring-primary outline-none font-mono"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-muted-foreground mb-1">
                  Endpoint Node B:
                </label>
                <input
                  type="text"
                  value={edgeEnd}
                  onChange={(e) => setEdgeEnd(e.target.value)}
                  placeholder="e.g. 15"
                  className="w-full rounded-lg border border-border bg-card px-3.5 py-2 text-sm text-foreground shadow-2xs focus:border-primary focus:ring-1 focus:ring-primary outline-none font-mono"
                />
              </div>
            </div>
            <p className="text-xs text-muted-foreground">
              Tip: You can also click directly on any road segment on the map below.
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            {/* Quick dropdown for junction nodes */}
            {junctionNodeOptions.length > 0 && (
              <div>
                <label className="block text-xs font-semibold text-foreground mb-1">
                  Quick Select: Major Junctions & Intersections
                </label>
                <select
                  value={selectedNodeId}
                  onChange={(e) => setSelectedNodeId(e.target.value)}
                  className="w-full rounded-lg border border-border bg-card px-3 py-2 text-xs text-foreground shadow-2xs focus:border-primary focus:ring-1 focus:ring-primary outline-none"
                >
                  <option value="">-- Choose an intersection node to disable --</option>
                  {junctionNodeOptions.map((opt) => (
                    <option key={opt.id} value={opt.id}>
                      {opt.label}
                    </option>
                  ))}
                </select>
              </div>
            )}

            <div>
              <label className="block text-xs font-medium text-muted-foreground mb-1">
                Target Node ID:
              </label>
              <input
                type="text"
                value={selectedNodeId}
                onChange={(e) => setSelectedNodeId(e.target.value)}
                placeholder="e.g. 8"
                className="w-full max-w-sm rounded-lg border border-border bg-card px-3.5 py-2 text-sm text-foreground shadow-2xs focus:border-primary focus:ring-1 focus:ring-primary outline-none font-mono"
              />
            </div>
          </div>
        )}

        {/* Trigger Button */}
        <div className="mt-5 pt-4 border-t border-border flex items-center justify-between">
          <span className="text-xs text-muted-foreground">
            {targetType === "edge"
              ? edgeStart && edgeEnd
                ? `Ready to sever Road #${edgeStart} ↔ #${edgeEnd}`
                : "Select an edge above to simulate"
              : selectedNodeId
                ? `Ready to disable Junction Node #${selectedNodeId}`
                : "Select a node above to simulate"}
          </span>

          <button
            type="button"
            onClick={handleSimulate}
            disabled={
              loading ||
              !analysisId ||
              (targetType === "edge" ? !edgeStart || !edgeEnd : !selectedNodeId)
            }
            className="inline-flex items-center gap-2 rounded-lg bg-destructive px-5 py-2.5 text-xs font-semibold text-destructive-foreground shadow-xs hover:bg-destructive/90 transition disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? (
              <>
                <LoaderCircle size={15} className="animate-spin" />
                <span>Simulating...</span>
              </>
            ) : (
              <>
                <Play size={15} />
                <span>Simulate Failure</span>
              </>
            )}
          </button>
        </div>
      </section>

      {/* Error Banner */}
      {error && (
        <div className="mb-6 flex items-start gap-3 rounded-xl border border-destructive/40 bg-destructive/10 p-4 text-xs sm:text-sm text-destructive">
          <AlertTriangle size={18} className="text-destructive shrink-0 mt-0.5" />
          <p>{error}</p>
        </div>
      )}

      {/* Simulation Results (Requirement 4: Before/After & Plain Language Impact) */}
      {result && metrics && scenario && (
        <div className="space-y-6">
          {/* Plain-Language Impact Banner */}
          <div className="rounded-xl border border-destructive/30 bg-destructive/10 p-5 sm:p-6">
            <div className="flex items-start gap-3.5">
              <div className="rounded-xl bg-destructive/20 p-2 text-destructive shrink-0 border border-destructive/30">
                <ShieldAlert className="h-5 w-5" />
              </div>
              <div className="flex-1 min-w-0">
                <h3 className="text-sm font-bold text-foreground">
                  Failure Impact:{" "}
                  {failedEdges.length
                    ? `Road #${failedEdges[0].join(" ↔ #")} Severed`
                    : `Junction #${failedNodes.join(", ")} Disabled`}
                </h3>

                <p className="mt-1 text-xs sm:text-sm text-muted-foreground leading-relaxed">
                  {newDisconnections > 0
                    ? `This failure critically partitions the network into ${result.network_state.disrupted.n_components} isolated zones (+${newDisconnections} newly severed components). Global network travel efficiency dropped by ${efficiencyDropPct}%.`
                    : `Network remains connected in a single zone, but travel efficiency decreased by ${efficiencyDropPct}% due to forced detours around the disruption.`}
                </p>

                {odService && (
                  <p className="mt-2 text-xs text-muted-foreground">
                    <strong className="text-foreground">Route impact:</strong> {odService.severed_pairs_count} evaluated route
                    pairs are completely severed (unreachable), while {odService.detoured_pairs_count}{" "}
                    pairs require detours.
                  </p>
                )}
              </div>
            </div>
          </div>

          {/* Before vs After Connectivity Comparison Table */}
          <section className="rounded-xl border border-border bg-card p-5 sm:p-6 shadow-xs">
            <h3 className="text-xs font-bold text-foreground uppercase tracking-wider mb-4 flex items-center gap-2">
              <Activity size={15} className="text-teal-600 dark:text-teal-400" />
              <span>Before & After Connectivity Comparison</span>
            </h3>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
              {/* Components */}
              <div className="bg-muted/40 rounded-lg p-3 border border-border">
                <span className="block text-[0.62rem] font-semibold text-muted-foreground uppercase">
                  Connected Zones
                </span>
                <div className="mt-1 flex items-baseline gap-2">
                  <span className="text-base font-bold text-foreground font-mono">
                    {result.network_state.baseline.n_components}
                  </span>
                  <span className="text-muted-foreground">→</span>
                  <span className="text-base font-bold text-destructive font-mono">
                    {result.network_state.disrupted.n_components}
                  </span>
                </div>
                <span className="text-[0.62rem] text-muted-foreground">
                  {newDisconnections > 0 ? `+${newDisconnections} severed pockets` : "Intact zone"}
                </span>
              </div>

              {/* Network Efficiency */}
              <div className="bg-muted/40 rounded-lg p-3 border border-border">
                <span className="block text-[0.62rem] font-semibold text-muted-foreground uppercase">
                  Travel Efficiency
                </span>
                <div className="mt-1 flex items-baseline gap-2">
                  <span className="text-base font-bold text-foreground font-mono">
                    {(metrics.before_efficiency * 100).toFixed(1)}%
                  </span>
                  <span className="text-muted-foreground">→</span>
                  <span className="text-base font-bold text-destructive font-mono">
                    {(metrics.after_efficiency * 100).toFixed(1)}%
                  </span>
                </div>
                <span className="text-[0.62rem] text-destructive font-semibold">
                  -{efficiencyDropPct}% drop
                </span>
              </div>

              {/* Traversable Roads */}
              <div className="bg-muted/40 rounded-lg p-3 border border-border">
                <span className="block text-[0.62rem] font-semibold text-muted-foreground uppercase">
                  Active Roads
                </span>
                <div className="mt-1 flex items-baseline gap-2">
                  <span className="text-base font-bold text-foreground font-mono">
                    {result.network_state.baseline.n_edges}
                  </span>
                  <span className="text-muted-foreground">→</span>
                  <span className="text-base font-bold text-foreground font-mono">
                    {result.network_state.disrupted.n_edges}
                  </span>
                </div>
                <span className="text-[0.62rem] text-muted-foreground">
                  -{result.network_state.baseline.n_edges - result.network_state.disrupted.n_edges}{" "}
                  removed
                </span>
              </div>

              {/* OD Route Service */}
              <div className="bg-muted/40 rounded-lg p-3 border border-border">
                <span className="block text-[0.62rem] font-semibold text-muted-foreground uppercase">
                  Service Delivery
                </span>
                <div className="mt-1">
                  <span className="text-base font-bold text-foreground font-mono">
                    {odService
                      ? `${(odService.disrupted_service_performance * 100).toFixed(0)}%`
                      : "N/A"}
                  </span>
                </div>
                <span className="text-[0.62rem] text-muted-foreground">
                  {odService ? `${odService.severed_pairs_count} severed pairs` : "Baseline OK"}
                </span>
              </div>
            </div>
          </section>

          {/* Disrupted Road Network Visualization */}
          {graph && (
            <section className="rounded-xl border border-border bg-card p-5 sm:p-6 shadow-xs">
              <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
                <div>
                  <h3 className="text-sm font-bold text-foreground">Disrupted Topology Map</h3>
                  <p className="text-xs text-muted-foreground">
                    Failed segments are highlighted in bold crimson red.
                  </p>
                </div>
                <div className="flex items-center gap-2 text-xs">
                  <span className="inline-flex items-center gap-1.5 text-destructive font-semibold bg-destructive/10 border border-destructive/20 px-2.5 py-1 rounded-lg">
                    <span className="w-2 h-2 rounded-full bg-destructive" />
                    Failed:{" "}
                    {failedEdges.length
                      ? failedEdges.map((e) => `#${e[0]} ↔ #${e[1]}`).join(", ")
                      : `Node #${failedNodes.join(", ")}`}
                  </span>
                </div>
              </div>

              <NetworkDiagram
                graph={graph}
                imageUrl={imageUrl}
                failedEdges={failedEdges as [NodeId, NodeId][]}
                onEdgeClick={handleDiagramEdgeClick}
              />
            </section>
          )}

          {/* Continue Action */}
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4 pt-2">
            <Link
              to="/network"
              search={{ analysis: analysisId ?? undefined }}
              className="text-xs font-semibold text-muted-foreground hover:text-foreground transition"
            >
              ← Back to Network View
            </Link>

            <Link
              to="/routing"
              search={{ analysis: analysisId ?? undefined }}
              className="inline-flex items-center gap-2 rounded-lg bg-primary px-6 py-2.5 text-sm font-semibold text-primary-foreground shadow-xs hover:bg-primary/90 transition"
            >
              <span>Continue to Reroute Around Failure</span>
              <ArrowRight size={16} />
            </Link>
          </div>
        </div>
      )}
    </main>
  );
}
