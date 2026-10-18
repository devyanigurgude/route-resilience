import { createFileRoute, Link, useLocation, useNavigate } from "@tanstack/react-router";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  Clock,
  CornerDownRight,
  HelpCircle,
  LoaderCircle,
  MapPin,
  Navigation,
  Network,
  RotateCcw,
  Route as RouteIcon,
  ShieldAlert,
  Sparkles,
  XCircle,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import {
  getAnalysis,
  getGraph,
  getScenario,
  postRoute,
  readAnalysisId,
  readRouteResult,
  saveRouteResult,
  RouteResponse,
  SAVED_ANALYSIS_KEY,
  type FailedEdge,
  type GraphData,
  type NodeId,
} from "@/lib/backend";
import { NetworkDiagram } from "@/components/NetworkDiagram";

export const Route = createFileRoute("/routing")({
  head: () => ({ meta: [{ title: "Route Resilience" }] }),
  component: RoutePage,
});

function RoutePage() {
  const navigate = useNavigate();
  const { searchStr } = useLocation();
  const queryAnalysisId = new URLSearchParams(searchStr).get("analysis");
  const queryOrigin = new URLSearchParams(searchStr).get("origin");
  const queryDestination = new URLSearchParams(searchStr).get("destination");

  const [analysisId, setAnalysisId] = useState(queryAnalysisId ?? readAnalysisId());
  const [result, setResult] = useState<RouteResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [graphLoading, setGraphLoading] = useState(false);

  const [origin, setOrigin] = useState<NodeId | null>(null);
  const [destination, setDestination] = useState<NodeId | null>(null);
  const [nodeIds, setNodeIds] = useState<NodeId[]>([]);
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [imageUrl, setImageUrl] = useState<string | undefined>();
  const [blockedEdges, setBlockedEdges] = useState<FailedEdge[]>([]);

  useEffect(() => {
    if (queryAnalysisId) {
      setAnalysisId(queryAnalysisId);
      window.localStorage.setItem(SAVED_ANALYSIS_KEY, queryAnalysisId);
    }
  }, [queryAnalysisId]);

  useEffect(() => {
    if (!analysisId) {
      setGraphLoading(false);
      setError("Please select or run an image analysis first.");
      return;
    }
    const controller = new AbortController();
    setGraphLoading(true);
    setNodeIds([]);
    setGraphData(null);
    setImageUrl(undefined);
    setError(null);

    // Restore cached route result if available
    const cached = readRouteResult(analysisId);
    if (cached) setResult(cached);

    getAnalysis<{
      analysis_id: string;
      input_image_ref?: string;
      artifacts: { original?: string; healed_graph_json?: string };
    }>(analysisId, controller.signal)
      .then((analysis) => {
        if (!analysis.artifacts.healed_graph_json) {
          throw new Error("Road network data unavailable for this analysis.");
        }
        setImageUrl(analysis.artifacts.original ?? analysis.input_image_ref);
        return getGraph<GraphData>(analysis.artifacts.healed_graph_json, controller.signal);
      })
      .then(async (graph) => {
        if (controller.signal.aborted) return;
        setGraphData(graph);
        const ids = graph.nodes.map((node) => node.id);
        setNodeIds(ids);

        // Load disruption scenario to identify blocked edges
        const scenario = await getScenario(analysisId, controller.signal);
        if (controller.signal.aborted) return;
        const failed = scenario?.failedEdges ?? [];
        setBlockedEdges(failed);

        // Set initial origin & destination
        if (queryOrigin) {
          setOrigin(queryOrigin);
        } else if (failed.length > 0) {
          setOrigin(failed[0][0]);
        }

        if (queryDestination) {
          setDestination(queryDestination);
        } else if (failed.length > 0) {
          setDestination(failed[0][1]);
        }
      })
      .catch((requestError: unknown) => {
        if (!controller.signal.aborted) {
          setError(
            requestError instanceof Error
              ? requestError.message
              : "Road network data unavailable.",
          );
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setGraphLoading(false);
      });

    return () => controller.abort();
  }, [analysisId, queryOrigin, queryDestination]);

  const runRoute = async () => {
    if (!analysisId) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await postRoute(
        analysisId,
        origin,
        destination,
        blockedEdges,
        blockedEdges.length > 0 ? "disrupted" : "baseline",
      );
      saveRouteResult(analysisId, response);
      setOrigin(response.origin);
      setDestination(response.destination);
      setResult(response);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Route calculation failed.");
    } finally {
      setLoading(false);
    }
  };

  const toggleBlockedEdge = (edge: FailedEdge) => {
    setBlockedEdges((current) => {
      const alreadyBlocked = current.some(
        ([s, t]) =>
          (String(s) === String(edge[0]) && String(t) === String(edge[1])) ||
          (String(s) === String(edge[1]) && String(t) === String(edge[0])),
      );
      return alreadyBlocked
        ? current.filter(
            ([s, t]) =>
              !(
                (String(s) === String(edge[0]) && String(t) === String(edge[1])) ||
                (String(s) === String(edge[1]) && String(t) === String(edge[0]))
              ),
          )
        : [...current, edge];
    });
    setResult(null);
  };

  // Node Click handler on map
  const handleDiagramNodeClick = (nodeId: NodeId) => {
    if (!origin) {
      setOrigin(nodeId);
    } else if (!destination && nodeId !== origin) {
      setDestination(nodeId);
    } else {
      setOrigin(nodeId);
      setDestination(null);
    }
    setResult(null);
  };

  // Affected nodes calculation (Requirement 5)
  const { bypassedNodes, detourNodes } = useMemo(() => {
    if (!result || !result.original_route || !result.alternative_route) {
      return { bypassedNodes: [], detourNodes: [] };
    }
    const origSet = new Set(result.original_route.map(String));
    const altSet = new Set(result.alternative_route.map(String));

    const bypassed = result.original_route.filter((n) => !altSet.has(String(n)));
    const detour = result.alternative_route.filter((n) => !origSet.has(String(n)));
    return { bypassedNodes: bypassed, detourNodes: detour };
  }, [result]);

  const hasAlternative =
    result?.route_status === "alternative_available" &&
    result.alternative_route &&
    result.alternative_route.length > 0;
  const isOriginalIntact = result?.route_status === "original_route_available";
  const isUnreachable =
    result?.route_exists === false || result?.route_status === "no_alternative_route";

  const detourExtraPx = result?.extra_distance ? result.extra_distance.toFixed(1) : null;
  const detourPct =
    result?.detour_ratio && result.detour_ratio > 1
      ? ((result.detour_ratio - 1) * 100).toFixed(1)
      : null;

  return (
    <main className="workflow-page mx-auto min-h-[calc(100vh-104px)] w-full max-w-6xl px-5 py-8 text-foreground sm:px-8">
      {/* Header */}
      <div className="mb-6 border-b border-border pb-5 flex flex-col sm:flex-row sm:items-end sm:justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center rounded-md bg-teal-500/10 px-2 py-0.5 text-xs font-semibold text-teal-700 dark:text-teal-300 border border-teal-500/20">
              04 · Reroute
            </span>
            <span className="text-xs text-muted-foreground">Step 4 of 5</span>
          </div>
          <h1 className="mt-2 text-2xl sm:text-3xl font-bold tracking-tight text-foreground">
            Reroute Around Network Disruptions
          </h1>
          <p className="mt-1.5 text-sm text-muted-foreground max-w-2xl">
            Compute detour corridors around severed or degraded lifelines, identify affected nodes,
            and evaluate extra travel cost.
          </p>
        </div>

        {analysisId && (
          <span className="text-xs font-mono text-muted-foreground bg-muted/60 px-3 py-1.5 rounded-lg border border-border self-start sm:self-end">
            Analysis #{analysisId.slice(0, 8)}
          </span>
        )}
      </div>

      {/* Origin & Destination Controls */}
      <section className="rounded-xl border border-border bg-card p-5 sm:p-6 shadow-xs mb-6">
        <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-5">
          <div className="flex-1">
            <h2 className="text-xs font-bold text-foreground uppercase tracking-wider mb-2 flex items-center gap-1.5">
              <Navigation size={14} className="text-teal-600 dark:text-teal-400" />
              <span>Route Endpoints (Origin → Destination)</span>
            </h2>

            <div className="flex flex-wrap items-center gap-3">
              <div className="min-w-44 flex-1">
                <label className="block text-xs font-medium text-muted-foreground mb-1">
                  Origin Node:
                </label>
                <select
                  value={origin === null ? "" : String(origin)}
                  onChange={(e) => {
                    setOrigin(e.target.value ? e.target.value : null);
                    setResult(null);
                  }}
                  disabled={graphLoading || !nodeIds.length}
                  className="w-full rounded-xl border border-border bg-background px-3 py-2 text-xs font-medium text-foreground shadow-xs focus:border-teal-500 focus:ring-1 focus:ring-teal-500/20 outline-none font-mono"
                >
                  <option value="">Auto-select origin node</option>
                  {nodeIds.map((id) => (
                    <option key={String(id)} value={String(id)}>
                      Node #{String(id)}
                    </option>
                  ))}
                </select>
              </div>

              <div className="self-end pb-2 text-muted-foreground font-bold hidden sm:block">→</div>

              <div className="min-w-44 flex-1">
                <label className="block text-xs font-medium text-muted-foreground mb-1">
                  Destination Node:
                </label>
                <select
                  value={destination === null ? "" : String(destination)}
                  onChange={(e) => {
                    setDestination(e.target.value ? e.target.value : null);
                    setResult(null);
                  }}
                  disabled={graphLoading || !nodeIds.length}
                  className="w-full rounded-xl border border-border bg-background px-3 py-2 text-xs font-medium text-foreground shadow-xs focus:border-teal-500 focus:ring-1 focus:ring-teal-500/20 outline-none font-mono"
                >
                  <option value="">Auto-select destination node</option>
                  {nodeIds.map((id) => (
                    <option key={String(id)} value={String(id)}>
                      Node #{String(id)}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <p className="mt-2.5 text-xs text-muted-foreground">
              Leave unset to automatically identify connected route pairs, or click any node on the
              map below.
            </p>
          </div>

          {/* Action Button */}
          <button
            type="button"
            onClick={runRoute}
            disabled={loading || graphLoading || !analysisId}
            className="inline-flex items-center justify-center gap-2 rounded-xl bg-teal-700 hover:bg-teal-800 dark:bg-teal-600 dark:hover:bg-teal-700 px-6 py-3 text-xs font-semibold text-white shadow-xs transition disabled:opacity-50"
          >
            {loading ? (
              <>
                <LoaderCircle size={15} className="animate-spin" />
                <span>Computing Detour...</span>
              </>
            ) : (
              <>
                <RouteIcon size={15} />
                <span>{blockedEdges.length > 0 ? "Find Alternative Detour" : "Find Path"}</span>
              </>
            )}
          </button>
        </div>

        {/* Blocked Edges Bar */}
        {blockedEdges.length > 0 && (
          <div className="mt-4 pt-3.5 border-t border-border flex flex-wrap items-center gap-2 text-xs">
            <span className="font-semibold text-foreground">Blocked Roads:</span>
            {blockedEdges.map((edge) => (
              <span
                key={`${edge[0]}-${edge[1]}`}
                onClick={() => toggleBlockedEdge(edge)}
                className="inline-flex items-center gap-1.5 rounded-lg border border-destructive/30 bg-destructive/10 px-2.5 py-1 text-destructive font-mono text-[0.68rem] cursor-pointer hover:bg-destructive/20"
                title="Click to toggle/unblock"
              >
                Road #{String(edge[0])} ↔ #{String(edge[1])} ×
              </span>
            ))}
            <button
              type="button"
              onClick={() => {
                setBlockedEdges([]);
                setResult(null);
              }}
              className="text-xs text-muted-foreground hover:text-foreground underline ml-1"
            >
              Clear all blockages
            </button>
          </div>
        )}
      </section>

      {/* Error Banner */}
      {error && (
        <div className="mb-6 flex items-start gap-3 rounded-xl border border-amber-500/30 bg-amber-500/10 p-4 text-xs sm:text-sm text-amber-800 dark:text-amber-300">
          <AlertTriangle size={18} className="text-amber-600 dark:text-amber-400 shrink-0 mt-0.5" />
          <p>{error}</p>
        </div>
      )}

      {/* REROUTE RESULTS DISPLAY (Requirement 5) */}
      {result && (
        <div className="space-y-6">
          {/* Reachability Status Badge & Plain Language Summary */}
          <div
            className={`rounded-xl border p-5 sm:p-6 ${
              isUnreachable
                ? "border-destructive/30 bg-destructive/10 text-destructive-foreground"
                : hasAlternative
                  ? "border-teal-500/30 bg-teal-500/10 text-foreground"
                  : "border-blue-500/30 bg-blue-500/10 text-foreground"
            }`}
          >
            <div className="flex items-start gap-3.5">
              <div
                className={`rounded-xl p-2 shrink-0 ${
                  isUnreachable
                    ? "bg-destructive/20 text-destructive"
                    : hasAlternative
                      ? "bg-teal-500/20 text-teal-700 dark:text-teal-300"
                      : "bg-blue-500/20 text-blue-700 dark:text-blue-300"
                }`}
              >
                {isUnreachable ? (
                  <XCircle className="h-5 w-5" />
                ) : (
                  <CheckCircle2 className="h-5 w-5" />
                )}
              </div>

              <div className="flex-1 min-w-0">
                <div className="flex flex-wrap items-center gap-2 mb-1">
                  <h3 className="text-sm font-bold">
                    {isUnreachable
                      ? "Destination Unreachable (Route Severed)"
                      : hasAlternative
                        ? "Destination Reachable via Alternative Detour"
                        : "Original Route Traversable (No Detour Needed)"}
                  </h3>
                  <span
                    className={`text-[0.62rem] font-bold uppercase tracking-wider px-2 py-0.5 rounded-md ${
                      isUnreachable
                        ? "bg-destructive/20 text-destructive"
                        : hasAlternative
                          ? "bg-teal-500/20 text-teal-800 dark:text-teal-200"
                          : "bg-blue-500/20 text-blue-800 dark:text-blue-200"
                    }`}
                  >
                    {result.route_status.replace(/_/g, " ")}
                  </span>
                </div>

                <p className="text-xs sm:text-sm leading-relaxed opacity-90">
                  {isUnreachable
                    ? `Due to network partitions, no alternative path currently exists between Node #${result.origin} and Node #${result.destination}.`
                    : hasAlternative
                      ? `Vehicles are diverted around blocked segments. The detour adds ${detourExtraPx} px (+${detourPct}% distance) compared to the original direct path.`
                      : `The original path between Node #${result.origin} and Node #${result.destination} is fully clear and functional.`}
                </p>
              </div>
            </div>
          </div>

          {/* Metric Comparison Cards (Original vs Alternative & Extra Cost) */}
          <section className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
            {/* Original Distance */}
            <div className="rounded-xl border border-border bg-card p-3.5 shadow-xs">
              <span className="block text-[0.62rem] font-bold text-muted-foreground uppercase tracking-wide">
                Original Route
              </span>
              <div className="mt-1 flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-blue-600 dark:bg-blue-500" />
                <span className="text-base font-bold text-foreground">
                  {result.original_route_length
                    ? `${result.original_route_length.toFixed(1)} px`
                    : "—"}
                </span>
              </div>
              <span className="text-[0.62rem] text-muted-foreground">
                {result.original_route?.length ?? 0} nodes
              </span>
            </div>

            {/* Alternative Distance */}
            <div className="rounded-xl border border-border bg-card p-3.5 shadow-xs">
              <span className="block text-[0.62rem] font-bold text-muted-foreground uppercase tracking-wide">
                Alternative Route
              </span>
              <div className="mt-1 flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-teal-600 dark:bg-teal-400" />
                <span className="text-base font-bold text-teal-700 dark:text-teal-300">
                  {result.alternative_route_length
                    ? `${result.alternative_route_length.toFixed(1)} px`
                    : isUnreachable
                      ? "Severed"
                      : "Same"}
                </span>
              </div>
              <span className="text-[0.62rem] text-muted-foreground">
                {result.alternative_route?.length ?? 0} nodes
              </span>
            </div>

            {/* Extra Distance / Cost */}
            <div className="rounded-xl border border-border bg-card p-3.5 shadow-xs">
              <span className="block text-[0.62rem] font-bold text-muted-foreground uppercase tracking-wide">
                Extra Distance
              </span>
              <div className="mt-1 text-base font-bold text-foreground">
                {detourExtraPx ? `+${detourExtraPx} px` : "0 px"}
              </div>
              <span
                className={`text-[0.62rem] font-semibold ${
                  detourPct ? "text-amber-600 dark:text-amber-400" : "text-teal-600 dark:text-teal-400"
                }`}
              >
                {detourPct ? `+${detourPct}% detour penalty` : "No detour penalty"}
              </span>
            </div>

            {/* Destination Reachability */}
            <div className="rounded-xl border border-border bg-card p-3.5 shadow-xs">
              <span className="block text-[0.62rem] font-bold text-muted-foreground uppercase tracking-wide">
                Reachability
              </span>
              <div className="mt-1 text-base font-bold text-foreground">
                {isUnreachable ? "Severed" : "100% Reachable"}
              </div>
              <span
                className={`text-[0.62rem] font-semibold ${
                  isUnreachable ? "text-destructive" : "text-teal-600 dark:text-teal-400"
                }`}
              >
                {isUnreachable ? "Destination unreachable" : "Traversable corridor"}
              </span>
            </div>
          </section>

          {/* Node Traversal & Affected Nodes Details (Requirement 5) */}
          <section className="rounded-xl border border-border bg-card p-5 sm:p-6 shadow-xs">
            <h3 className="text-xs font-bold text-foreground uppercase tracking-wider mb-4 flex items-center gap-2">
              <Network size={15} className="text-teal-600 dark:text-teal-400" />
              <span>Route Node Breakdown & Bypassed Junctions</span>
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
              {/* Original Route Node Sequence */}
              <div className="rounded-xl bg-muted/40 p-4 border border-border">
                <span className="font-bold text-foreground flex items-center gap-2 mb-2">
                  <span className="w-2 h-2 rounded-full bg-blue-600 dark:bg-blue-500" />
                  Original Route Nodes:
                </span>
                <div className="flex flex-wrap items-center gap-1.5 font-mono">
                  {result.original_route && result.original_route.length > 0 ? (
                    result.original_route.map((node, i) => (
                      <span key={i} className="flex items-center gap-1">
                        <span className="bg-card border border-border px-2 py-0.5 rounded-md text-foreground">
                          #{String(node)}
                        </span>
                        {i < result.original_route.length - 1 && (
                          <span className="text-muted-foreground">→</span>
                        )}
                      </span>
                    ))
                  ) : (
                    <span className="text-muted-foreground">None available</span>
                  )}
                </div>
              </div>

              {/* Alternative Detour Node Sequence */}
              <div className="rounded-xl bg-teal-500/10 p-4 border border-teal-500/20">
                <span className="font-bold text-teal-800 dark:text-teal-200 flex items-center gap-2 mb-2">
                  <span className="w-2 h-2 rounded-full bg-teal-600 dark:bg-teal-400" />
                  Alternative Detour Nodes:
                </span>
                <div className="flex flex-wrap items-center gap-1.5 font-mono">
                  {result.alternative_route && result.alternative_route.length > 0 ? (
                    result.alternative_route.map((node, i) => (
                      <span key={i} className="flex items-center gap-1">
                        <span className="bg-card border border-teal-500/30 px-2 py-0.5 rounded-md text-teal-800 dark:text-teal-200 font-bold">
                          #{String(node)}
                        </span>
                        {i < result.alternative_route.length - 1 && (
                          <span className="text-teal-500/60">→</span>
                        )}
                      </span>
                    ))
                  ) : (
                    <span className="text-muted-foreground">No alternative path</span>
                  )}
                </div>
              </div>
            </div>

            {/* Affected & Bypassed Summary */}
            {(bypassedNodes.length > 0 || detourNodes.length > 0) && (
              <div className="mt-4 pt-3.5 border-t border-border grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                {bypassedNodes.length > 0 && (
                  <div className="flex items-center gap-2">
                    <span className="text-muted-foreground font-medium">Bypassed Blocked Nodes:</span>
                    <span className="font-mono text-destructive font-semibold">
                      {bypassedNodes.map((n) => `#${String(n)}`).join(", ")}
                    </span>
                  </div>
                )}
                {detourNodes.length > 0 && (
                  <div className="flex items-center gap-2">
                    <span className="text-muted-foreground font-medium">New Detour Nodes:</span>
                    <span className="font-mono text-teal-700 dark:text-teal-300 font-semibold">
                      {detourNodes.map((n) => `#${String(n)}`).join(", ")}
                    </span>
                  </div>
                )}
              </div>
            )}
          </section>

          {/* Interactive Network Diagram */}
          {graphData && (
            <section className="rounded-xl border border-border bg-card p-5 sm:p-6 shadow-xs">
              <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
                <div>
                  <h3 className="text-sm font-bold text-foreground">Route Alignment Map</h3>
                  <p className="text-xs text-muted-foreground">
                    Original route in blue; alternative detour in green; blocked road in red.
                  </p>
                </div>
              </div>

              <NetworkDiagram
                graph={graphData}
                imageUrl={imageUrl}
                originalRoute={result.original_route}
                alternativeRoute={result.alternative_route}
                failedEdges={blockedEdges}
                onEdgeClick={toggleBlockedEdge}
                onNodeClick={handleDiagramNodeClick}
              />
            </section>
          )}

          {/* Navigation to Recovery */}
          <div className="flex items-center justify-between pt-2">
            <Link
              to="/disruption"
              search={{ analysis: analysisId ?? undefined }}
              className="text-xs font-semibold text-muted-foreground hover:text-foreground transition"
            >
              ← Back to Disruption
            </Link>

            <Link
              to="/recovery"
              search={{ analysis: analysisId ?? undefined }}
              className="inline-flex items-center gap-2 rounded-xl bg-teal-700 hover:bg-teal-800 dark:bg-teal-600 dark:hover:bg-teal-700 px-6 py-3 text-sm font-semibold text-white shadow-xs transition"
            >
              <span>Continue to Recovery Optimization</span>
              <ArrowRight size={16} />
            </Link>
          </div>
        </div>
      )}
    </main>
  );
}
