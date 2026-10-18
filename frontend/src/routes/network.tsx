import { createFileRoute, Link, useLocation, useNavigate } from "@tanstack/react-router";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  Check,
  Compass,
  Copy,
  ExternalLink,
  Eye,
  EyeOff,
  Flame,
  GitBranch,
  Layers,
  CheckCircle2,
  HelpCircle,
  Loader2,
  Maximize2,
  Network,
  RotateCcw,
  Route as RouteIcon,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  X,
  XCircle,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import React, { useEffect, useMemo, useRef, useState } from "react";

import {
  getAnalysis,
  getCriticality,
  getGraph,
  getScenario,
  readAnalysisId,
  savePendingFailure,
  savePendingNodeFailure,
  SAVED_ANALYSIS_KEY,
  type CandidateOcclusion,
  type ContinuityCheck,
  type FailedEdge,
  type GraphData,
  type NodeId,
  type RecoveryEvidence,
  type StoredScenario,
} from "@/lib/backend";

type LoadState = "loading" | "success" | "error" | "missing";

type GraphSummary = {
  n_nodes: number;
  n_edges: number;
  n_components: number;
  total_length_px: number;
  skeleton_connectivity_ratio?: number;
};

type AnalysisResponse = {
  analysis_id: string;
  status: string;
  input_image_ref?: string;
  artifacts: {
    original?: string;
    binary_mask?: string;
    skeleton?: string;
    repaired_network?: string;
    critical_roads?: string;
    visualization?: string;
    graph_json?: string;
    healed_graph_json?: string;
    summary_json?: string;
    criticality?: string;
  };
  summary: {
    threshold: number;
    before_healing: GraphSummary;
    after_healing: GraphSummary;
    delta_after_minus_before: Record<string, number>;
    healed_edge_count: number;
    occlusion_recovered_edge_count?: number;
    coordinate_system: string;
    graph_status?: string;
    graph_quality?: {
      component_sizes?: number[];
      isolated_node_count?: number;
      endpoint_count?: number;
      junction_count?: number;
      routable_component_count?: number;
    };
  };
};

type CriticalityEdge = {
  edge_id: [string | number, string | number] | string[];
  length_px?: number;
  confidence?: number | null;
  healed?: boolean;
  occlusion_recovered?: boolean;
  width_px?: number | null;
  edge_betweenness?: number;
  detour_cost_px?: number | null;
  detour_disconnected?: boolean;
  is_bridge?: boolean;
  standalone_efficiency_drop?: number;
};

type CriticalityReport = {
  edge_criticality?: CriticalityEdge[];
  top_vulnerable_edges_by_betweenness?: Array<[string | number, string | number] | string[]>;
  articulation_points?: Array<string | number>;
};

type CriticalityResponse = {
  analysis_id: string;
  status: string;
  criticality: CriticalityReport;
};

type LayerState = {
  satellite: boolean;
  roads: boolean;
  criticality: boolean;
  occlusions: boolean;
  junctions: boolean;
};

export const Route = createFileRoute("/network")({
  head: () => ({
    meta: [
      { title: "Route Resilience" },
      {
        name: "description",
        content: "Interactive satellite road topology, network healing, and critical lifelines.",
      },
    ],
  }),
  component: GeospatialAnalysisPage,
});

function GeospatialAnalysisPage() {
  const navigate = useNavigate();
  const { searchStr } = useLocation();
  const queryParamId = new URLSearchParams(searchStr).get("analysis") ?? undefined;

  const [activeAnalysisId, setActiveAnalysisId] = useState<string | undefined>(
    queryParamId || readAnalysisId() || undefined,
  );

  const [loadState, setLoadState] = useState<LoadState>(activeAnalysisId ? "loading" : "missing");
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [criticality, setCriticality] = useState<CriticalityResponse | null>(null);
  const [currentScenario, setCurrentScenario] = useState<StoredScenario | null>(null);

  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [isGraphLoading, setIsGraphLoading] = useState(false);
  const [imageSize, setImageSize] = useState<{ width: number; height: number } | null>(null);

  // Layer toggles
  const [layers, setLayers] = useState<LayerState>({
    satellite: true,
    roads: true,
    criticality: true,
    occlusions: true,
    junctions: true,
  });

  // Selected road edge
  const [selectedEdge, setSelectedEdge] = useState<{
    source: NodeId;
    target: NodeId;
    length_px?: number;
    confidence?: number;
    healed?: boolean;
    inferred?: boolean;
    occlusion_recovered?: boolean;
    occlusion_suspected?: boolean;
    recovery_type?: string;
    gap_px?: number;
    heading_alignment?: number;
    dot_a?: number;
    dot_b?: number;
    mutual_heading?: number;
    lateral_offset_px?: number;
    sinuosity?: number;
    probability_evidence?: number;
    supported_fraction?: number;
    flank_contrast?: number | null;
    cloud_coverage?: number | null;
    acceptance_reason?: string;
    checks?: ContinuityCheck[];
    evidence?: RecoveryEvidence;
    betweenness?: number;
    detour_cost_px?: number | null;
    is_bridge?: boolean;
    rank?: number;
  } | null>(null);

  // Selected candidate occlusion
  const [selectedCandidate, setSelectedCandidate] = useState<CandidateOcclusion | null>(null);

  // Selected node (Requirement 3)
  const [selectedNode, setSelectedNode] = useState<{
    id: NodeId;
    pos?: [number, number];
    degree: number;
    neighbors: Array<{ id: NodeId; length_px?: number }>;
    isArticulationPoint?: boolean;
    isJunction?: boolean;
  } | null>(null);

  // Hovered node
  const [hoveredNodeId, setHoveredNodeId] = useState<NodeId | null>(null);

  // Node label visibility mode (Requirement 3: Number every node, control to show/hide)
  const [nodeLabelMode, setNodeLabelMode] = useState<"all" | "hubs" | "focus">("all");

  // Pan & Zoom
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const dragStartRef = useRef({ x: 0, y: 0 });
  const [copiedId, setCopiedId] = useState(false);

  // Sync queryParam or localStorage
  useEffect(() => {
    if (queryParamId) {
      setActiveAnalysisId(queryParamId);
      if (typeof window !== "undefined") {
        window.localStorage.setItem(SAVED_ANALYSIS_KEY, queryParamId);
      }
    } else {
      const stored = readAnalysisId();
      if (stored) {
        setActiveAnalysisId(stored);
      } else {
        setLoadState("missing");
      }
    }
  }, [queryParamId]);

  useEffect(() => {
    if (!activeAnalysisId) return;
    setAnalysis(null);
    setGraphData(null);
    setCriticality(null);
    setSelectedEdge(null);
    setSelectedNode(null);
    setImageSize(null);
  }, [activeAnalysisId]);

  // Load Analysis Metadata
  useEffect(() => {
    if (!activeAnalysisId) {
      setAnalysis(null);
      setLoadState("missing");
      return;
    }

    const controller = new AbortController();
    setLoadState("loading");
    setErrorMessage(null);

    getAnalysis<AnalysisResponse>(activeAnalysisId, controller.signal)
      .then((payload) => {
        if (controller.signal.aborted) return;
        setAnalysis(payload);
        setLoadState("success");
      })
      .catch((err: unknown) => {
        if (err instanceof DOMException && err.name === "AbortError") return;
        setErrorMessage(err instanceof Error ? err.message : "Unable to load analysis.");
        setLoadState("error");
      });

    return () => controller.abort();
  }, [activeAnalysisId]);

  // Load Scenario (Disruptions)
  useEffect(() => {
    if (!activeAnalysisId) {
      setCurrentScenario(null);
      return;
    }
    const controller = new AbortController();
    getScenario(activeAnalysisId, controller.signal)
      .then((scenario) => {
        if (!controller.signal.aborted) setCurrentScenario(scenario);
      })
      .catch(() => {
        if (!controller.signal.aborted) setCurrentScenario(null);
      });
    return () => controller.abort();
  }, [activeAnalysisId]);

  // Load Criticality Report
  useEffect(() => {
    if (!activeAnalysisId) return;
    const controller = new AbortController();
    getCriticality<CriticalityResponse>(activeAnalysisId, controller.signal)
      .then((payload) => {
        if (!controller.signal.aborted) setCriticality(payload);
      })
      .catch(() => {
        if (!controller.signal.aborted) setCriticality(null);
      });
    return () => controller.abort();
  }, [activeAnalysisId]);

  // Load Vector Graph Data
  useEffect(() => {
    if (!analysis?.artifacts?.healed_graph_json) return;
    const controller = new AbortController();
    setIsGraphLoading(true);

    getGraph<GraphData>(analysis.artifacts.healed_graph_json, controller.signal)
      .then((data) => {
        if (!controller.signal.aborted && Array.isArray(data.nodes) && Array.isArray(data.edges)) {
          setGraphData(data);
        }
      })
      .catch(() => {
        if (!controller.signal.aborted) setGraphData(null);
      })
      .finally(() => {
        if (!controller.signal.aborted) setIsGraphLoading(false);
      });

    return () => controller.abort();
  }, [analysis]);

  // Node Degree Map & Adjacency
  const { nodeAdjacency, importantNodeIds } = useMemo(() => {
    const adj = new Map<string, Array<{ id: NodeId; length_px?: number }>>();
    if (!graphData) return { nodeAdjacency: adj, importantNodeIds: new Set<string>() };

    graphData.nodes.forEach((n) => adj.set(String(n.id), []));
    graphData.edges.forEach((e) => {
      const s = String(e.source);
      const t = String(e.target);
      const len = e.attributes?.length_px;
      adj.get(s)?.push({ id: e.target, length_px: len });
      adj.get(t)?.push({ id: e.source, length_px: len });
    });

    const important = new Set<string>();
    adj.forEach((neighbors, idStr) => {
      if (neighbors.length >= 3) {
        important.add(idStr); // Junction
      }
    });

    // Articulation points are also important
    const artPoints = criticality?.criticality.articulation_points ?? [];
    artPoints.forEach((p) => important.add(String(p)));

    return { nodeAdjacency: adj, importantNodeIds: important };
  }, [graphData, criticality]);

  // Collision-aware label placements for readable numbering (Requirement 3)
  const nodeLabelPlacements = useMemo(() => {
    if (!graphData?.nodes) return new Map<string, { dx: number; dy: number }>();
    const placements = new Map<string, { dx: number; dy: number }>();
    const occupied: Array<{ x: number; y: number; w: number; h: number }> = [];

    graphData.nodes.forEach((node) => {
      const pos = node.attributes?.pos;
      if (!Array.isArray(pos) || pos.length < 2) return;
      const col = pos[1];
      const row = pos[0];
      const idStr = String(node.id);
      const textW = idStr.length * 6.5 + 8;
      const h = 13;

      const candidates = [
        { dx: 4, dy: -8, x: col + 2, y: row - 14 },
        { dx: 4, dy: 6, x: col + 2, y: row },
        { dx: -(textW + 4), dy: -8, x: col - textW - 4, y: row - 14 },
        { dx: -(textW + 4), dy: 6, x: col - textW - 4, y: row },
      ];

      let chosen = candidates[0];
      for (const cand of candidates) {
        const collides = occupied.some(
          (box) =>
            Math.abs(cand.x - box.x) < (textW + box.w) / 2 + 1 &&
            Math.abs(cand.y - box.y) < (h + box.h) / 2 + 1,
        );
        if (!collides) {
          chosen = cand;
          break;
        }
      }
      occupied.push({ x: chosen.x, y: chosen.y, w: textW, h });
      placements.set(idStr, { dx: chosen.dx, dy: chosen.dy });
    });

    return placements;
  }, [graphData]);

  // Zoom & Pan handlers
  const handleZoomIn = () => setZoom((z) => Math.min(z + 0.35, 4.5));
  const handleZoomOut = () => setZoom((z) => Math.max(z - 0.35, 0.65));
  const handleResetView = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0) return;
    setIsDragging(true);
    dragStartRef.current = { x: e.clientX - pan.x, y: e.clientY - pan.y };
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;
    setPan({
      x: e.clientX - dragStartRef.current.x,
      y: e.clientY - dragStartRef.current.y,
    });
  };

  const handleMouseUp = () => setIsDragging(false);

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY < 0 ? 0.15 : -0.15;
    setZoom((z) => Math.max(0.65, Math.min(4.5, z + delta)));
  };

  // Helper for Edge Criticality
  const getEdgeCriticalityInfo = (u: NodeId, v: NodeId) => {
    const list = criticality?.criticality.edge_criticality || [];
    const topBetweenness = criticality?.criticality.top_vulnerable_edges_by_betweenness || [];

    const found = list.find((item) => {
      const id = item.edge_id;
      if (!Array.isArray(id) || id.length < 2) return false;
      return (
        (String(id[0]) === String(u) && String(id[1]) === String(v)) ||
        (String(id[0]) === String(v) && String(id[1]) === String(u))
      );
    });

    const rankIndex = topBetweenness.findIndex((pair) => {
      if (!Array.isArray(pair) || pair.length < 2) return false;
      return (
        (String(pair[0]) === String(u) && String(pair[1]) === String(v)) ||
        (String(pair[0]) === String(v) && String(pair[1]) === String(u))
      );
    });

    return {
      critEdge: found,
      rank: rankIndex >= 0 ? rankIndex + 1 : undefined,
    };
  };

  // Select Edge
  const handleSelectEdge = (u: NodeId, v: NodeId, edgeAttr?: any) => {
    setSelectedNode(null);
    setSelectedCandidate(null);
    const { critEdge, rank } = getEdgeCriticalityInfo(u, v);
    const ev = edgeAttr?.evidence;
    setSelectedEdge({
      source: u,
      target: v,
      length_px: critEdge?.length_px ?? edgeAttr?.length_px,
      confidence: critEdge?.confidence ?? edgeAttr?.confidence ?? undefined,
      healed: Boolean(critEdge?.healed ?? edgeAttr?.healed ?? false),
      inferred: Boolean(edgeAttr?.inferred ?? false),
      occlusion_recovered: Boolean(
        critEdge?.occlusion_recovered ?? edgeAttr?.occlusion_recovered ?? false,
      ),
      occlusion_suspected: Boolean(edgeAttr?.occlusion_suspected ?? false),
      recovery_type: edgeAttr?.recovery_type,
      gap_px: ev?.gap_px ?? edgeAttr?.gap_px,
      heading_alignment: ev?.heading_alignment ?? edgeAttr?.heading_alignment,
      dot_a: ev?.dot_a ?? edgeAttr?.dot_a,
      dot_b: ev?.dot_b ?? edgeAttr?.dot_b,
      mutual_heading: ev?.mutual_heading ?? edgeAttr?.mutual_heading,
      lateral_offset_px: ev?.lateral_offset_px ?? edgeAttr?.lateral_offset_px,
      sinuosity: ev?.sinuosity ?? edgeAttr?.sinuosity,
      probability_evidence: ev?.probability_evidence ?? edgeAttr?.probability_evidence,
      supported_fraction: ev?.supported_fraction ?? edgeAttr?.supported_fraction,
      flank_contrast: ev?.flank_contrast ?? edgeAttr?.flank_contrast,
      cloud_coverage: ev?.cloud_coverage ?? edgeAttr?.cloud_coverage,
      acceptance_reason: ev?.acceptance_reason ?? edgeAttr?.acceptance_reason,
      checks: ev?.checks ?? edgeAttr?.checks ?? [],
      evidence: ev,
      betweenness: critEdge?.edge_betweenness,
      detour_cost_px: critEdge?.detour_cost_px,
      is_bridge: critEdge?.is_bridge,
      rank,
    });
  };

  // Select Candidate Occlusion
  const handleSelectCandidate = (cand: CandidateOcclusion) => {
    setSelectedNode(null);
    setSelectedEdge(null);
    setSelectedCandidate(cand);
  };

  // Select Node (Requirement 3)
  const handleSelectNode = (nodeId: NodeId, pos?: [number, number]) => {
    setSelectedEdge(null);
    setSelectedCandidate(null);
    const neighbors = nodeAdjacency.get(String(nodeId)) || [];
    const isArticulation = Boolean(
      criticality?.criticality.articulation_points?.some((p) => String(p) === String(nodeId)),
    );
    setSelectedNode({
      id: nodeId,
      pos,
      degree: neighbors.length,
      neighbors,
      isArticulationPoint: isArticulation,
      isJunction: neighbors.length >= 3,
    });
  };

  // Actions for selected items
  const handleSimulateEdgeFailure = (u: NodeId, v: NodeId) => {
    if (!activeAnalysisId) return;
    savePendingFailure(activeAnalysisId, [u, v]);
    void navigate({
      to: "/disruption",
      search: { analysis: activeAnalysisId, edge: JSON.stringify([u, v]) },
    });
  };

  const handleSimulateNodeFailure = (nodeId: NodeId) => {
    if (!activeAnalysisId) return;
    savePendingNodeFailure(activeAnalysisId, nodeId);
    void navigate({
      to: "/disruption",
      search: { analysis: activeAnalysisId, node: String(nodeId) },
    });
  };

  const handleRouteFromNode = (nodeId: NodeId) => {
    if (!activeAnalysisId) return;
    void navigate({
      to: "/routing",
      search: { analysis: activeAnalysisId, origin: String(nodeId) },
    });
  };

  const handleRouteToNode = (nodeId: NodeId) => {
    if (!activeAnalysisId) return;
    void navigate({
      to: "/routing",
      search: { analysis: activeAnalysisId, destination: String(nodeId) },
    });
  };

  const artifacts = analysis?.artifacts;
  const graphExtent = graphData?.edges
    .flatMap((edge) => edge.attributes.pts ?? [])
    .reduce(
      (bounds, point) => ({
        width: Math.max(bounds.width, point[1] + 1),
        height: Math.max(bounds.height, point[0] + 1),
      }),
      { width: 1, height: 1 },
    );
  const canvasWidth = imageSize?.width ?? graphExtent?.width ?? 1;
  const canvasHeight = imageSize?.height ?? graphExtent?.height ?? 1;

  const nodeCount = graphData?.nodes.length ?? analysis?.summary?.after_healing?.n_nodes ?? 0;
  const edgeCount = graphData?.edges.length ?? analysis?.summary?.after_healing?.n_edges ?? 0;
  const compCount = analysis?.summary?.after_healing?.n_components ?? 1;
  const healedCount = analysis?.summary?.healed_edge_count ?? 0;
  const occlusionCount = analysis?.summary?.occlusion_recovered_edge_count ?? 0;

  return (
    <div className="flex-1 flex flex-col min-h-[calc(100vh-104px)] bg-background text-foreground">
      {/* Missing Analysis State */}
      {loadState === "missing" && (
        <main className="flex-1 flex items-center justify-center p-8">
          <div className="max-w-md w-full bg-card text-card-foreground rounded-2xl border border-border shadow-xs p-8 text-center">
            <div className="w-12 h-12 rounded-2xl bg-primary/10 text-primary flex items-center justify-center mx-auto mb-3">
              <Compass size={24} />
            </div>
            <h2 className="text-lg font-bold text-foreground">No Active Analysis</h2>
            <p className="mt-1.5 text-xs text-muted-foreground">
              Select or upload a satellite image in the Analyze stage to inspect its road network.
            </p>
            <Link
              to="/explore"
              className="mt-5 inline-flex items-center gap-2 rounded-xl bg-primary text-primary-foreground px-5 py-2.5 text-xs font-semibold hover:bg-primary/90 transition shadow-xs"
            >
              <Sparkles size={14} />
              <span>Go to Analyze</span>
            </Link>
          </div>
        </main>
      )}

      {/* Loading State */}
      {loadState === "loading" && (
        <main className="flex-1 flex items-center justify-center p-8">
          <div className="flex flex-col items-center gap-3 text-center">
            <Loader2 className="w-8 h-8 animate-spin text-primary" />
            <p className="text-sm font-semibold text-foreground">Loading Network Workspace...</p>
            <p className="text-xs text-muted-foreground">Loading satellite imagery and graph topology</p>
          </div>
        </main>
      )}

      {/* Error State */}
      {loadState === "error" && (
        <main className="flex-1 flex items-center justify-center p-8">
          <div className="max-w-md w-full bg-card text-card-foreground rounded-2xl border border-destructive/50 p-8 text-center">
            <div className="w-12 h-12 rounded-2xl bg-destructive/10 text-destructive flex items-center justify-center mx-auto mb-3">
              <AlertTriangle size={24} />
            </div>
            <h2 className="text-base font-bold text-foreground">Unable to Load Network</h2>
            <p className="mt-1.5 text-xs text-muted-foreground">{errorMessage}</p>
            <Link
              to="/explore"
              className="mt-5 inline-flex items-center gap-2 rounded-xl bg-primary text-primary-foreground px-4 py-2 text-xs font-semibold hover:bg-primary/90 transition shadow-xs"
            >
              Return to Analyze
            </Link>
          </div>
        </main>
      )}

      {/* Workspace (Success State) */}
      {loadState === "success" && analysis && (
        <div className="flex-1 flex flex-col lg:flex-row gap-5 p-4 sm:p-6 max-w-[1720px] mx-auto w-full">
          {/* Main Interactive Canvas */}
          <section
            className="flex-1 flex flex-col min-h-[560px] lg:min-h-[640px] bg-card rounded-2xl border border-border shadow-xs overflow-hidden relative"
            aria-label="Interactive road network canvas"
          >
            {/* Canvas Header Bar */}
            <div className="flex flex-wrap items-center justify-between gap-2 px-4 py-2.5 border-b border-border bg-card/95 backdrop-blur-xs z-20">
              <div className="flex items-center gap-2.5">
                <span className="w-2.5 h-2.5 rounded-full bg-teal-600 dark:bg-teal-400" />
                <span className="text-xs font-bold text-foreground uppercase tracking-wide">
                  Road Network View
                </span>
                <span className="text-[0.68rem] text-muted-foreground font-mono hidden sm:inline">
                  {imageSize ? `${imageSize.width}×${imageSize.height} px` : "Pixel-Space"}
                </span>
              </div>

              {/* Status Tags & Label Mode Control (Requirement 3) */}
              <div className="flex items-center gap-2 text-xs">
                <div className="flex items-center gap-1 rounded-lg border border-border bg-muted/40 p-0.5 text-[0.62rem] font-medium text-foreground">
                  <span className="text-muted-foreground px-1 hidden sm:inline">Labels:</span>
                  <button
                    type="button"
                    onClick={() => setNodeLabelMode("all")}
                    className={`px-1.5 py-0.5 rounded transition ${
                      nodeLabelMode === "all"
                        ? "bg-primary text-primary-foreground font-semibold"
                        : "text-muted-foreground hover:text-foreground"
                    }`}
                    title="Number every node with its stable ID"
                  >
                    All ({graphData?.nodes?.length ?? 0})
                  </button>
                  <button
                    type="button"
                    onClick={() => setNodeLabelMode("hubs")}
                    className={`px-1.5 py-0.5 rounded transition ${
                      nodeLabelMode === "hubs"
                        ? "bg-primary text-primary-foreground font-semibold"
                        : "text-muted-foreground hover:text-foreground"
                    }`}
                    title="Show numbers for junctions and articulation points"
                  >
                    Hubs ({importantNodeIds.size})
                  </button>
                  <button
                    type="button"
                    onClick={() => setNodeLabelMode("focus")}
                    className={`px-1.5 py-0.5 rounded transition ${
                      nodeLabelMode === "focus"
                        ? "bg-primary text-primary-foreground font-semibold"
                        : "text-muted-foreground hover:text-foreground"
                    }`}
                    title="Only show labels on hover or click"
                  >
                    Hover
                  </button>
                </div>

                {healedCount > 0 && (
                  <span className="hidden sm:inline-flex items-center gap-1.5 bg-teal-500/10 text-teal-700 dark:text-teal-300 border border-teal-500/20 rounded-md px-2 py-0.5 text-[0.68rem] font-medium">
                    <span className="w-1.5 h-1.5 rounded-full bg-teal-500" />
                    {healedCount} Healed
                  </span>
                )}
                {selectedNode && (
                  <span className="inline-flex items-center gap-1 bg-teal-500/15 text-teal-800 dark:text-teal-200 border border-teal-500/30 rounded-md px-2 py-0.5 text-[0.68rem] font-semibold">
                    Node #{String(selectedNode.id)} Selected
                  </span>
                )}
                {selectedEdge && (
                  <span className="inline-flex items-center gap-1 bg-primary/10 text-primary border border-primary/20 rounded-md px-2 py-0.5 text-[0.68rem] font-semibold">
                    Road #{String(selectedEdge.source)} ↔ #{String(selectedEdge.target)}
                  </span>
                )}
              </div>
            </div>

            {/* Interactive Viewport */}
            <div
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
              onWheel={handleWheel}
              className={`flex-1 relative overflow-hidden bg-slate-950 select-none flex items-center justify-center ${
                isDragging ? "cursor-grabbing" : "cursor-grab"
              }`}
            >
              {/* Subtle background grid */}
              <div
                aria-hidden
                className="absolute inset-0 opacity-10 pointer-events-none"
                style={{
                  backgroundImage:
                    "linear-gradient(to right, #38bdf8 1px, transparent 1px), linear-gradient(to bottom, #38bdf8 1px, transparent 1px)",
                  backgroundSize: "32px 32px",
                }}
              />

              {/* Transformed Stage */}
              <div
                className="relative transition-transform duration-75 ease-out shadow-2xl"
                style={{
                  transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                  transformOrigin: "center center",
                  width: "min(100%, 720px)",
                  aspectRatio: `${canvasWidth} / ${canvasHeight}`,
                }}
              >
                {/* 1. Base Satellite Layer */}
                {layers.satellite && artifacts?.original && (
                  <img
                    src={artifacts.original}
                    alt="Satellite base"
                    onLoad={(event) =>
                      setImageSize({
                        width: event.currentTarget.naturalWidth,
                        height: event.currentTarget.naturalHeight,
                      })
                    }
                    className="absolute inset-0 h-full w-full object-fill rounded-sm pointer-events-none select-none"
                    draggable={false}
                  />
                )}

                {/* 2. Vector SVG Overlay */}
                {graphData && (
                  <svg
                    viewBox={`0 0 ${canvasWidth} ${canvasHeight}`}
                    className="absolute inset-0 w-full h-full overflow-visible z-10"
                  >
                    {/* Glow filters */}
                    <defs>
                      <filter id="glow-cyan" x="-20%" y="-20%" width="140%" height="140%">
                        <feDropShadow dx="0" dy="0" stdDeviation="2" floodColor="#00E5FF" />
                      </filter>
                      <filter id="glow-coral" x="-20%" y="-20%" width="140%" height="140%">
                        <feDropShadow dx="0" dy="0" stdDeviation="2" floodColor="#FF5A36" />
                      </filter>
                      <filter id="glow-teal" x="-20%" y="-20%" width="140%" height="140%">
                        <feDropShadow dx="0" dy="0" stdDeviation="2.5" floodColor="#14B8A6" />
                      </filter>
                      <filter id="glow-purple" x="-20%" y="-20%" width="140%" height="140%">
                        <feDropShadow dx="0" dy="0" stdDeviation="2.5" floodColor="#C084FC" />
                      </filter>
                    </defs>

                    {/* EDGES / ROADS */}
                    {layers.roads &&
                      graphData.edges.map((edge, idx) => {
                        const isFailed = Boolean(
                          currentScenario?.failedEdges.some(
                            ([u, v]) =>
                              (String(u) === String(edge.source) &&
                                String(v) === String(edge.target)) ||
                              (String(u) === String(edge.target) &&
                                String(v) === String(edge.source)),
                          ),
                        );
                        const isOcclusionRecovered = Boolean(edge.attributes?.occlusion_recovered);
                        const isInferred = Boolean(edge.attributes?.inferred);
                        const isHealed = Boolean(edge.attributes?.healed);
                        const { critEdge, rank } = getEdgeCriticalityInfo(edge.source, edge.target);
                        const isCritical = Boolean(rank !== undefined);

                        const isSelected =
                          selectedEdge &&
                          ((String(selectedEdge.source) === String(edge.source) &&
                            String(selectedEdge.target) === String(edge.target)) ||
                            (String(selectedEdge.source) === String(edge.target) &&
                              String(selectedEdge.target) === String(edge.source)));

                        const isConnectedToSelectedNode =
                          selectedNode &&
                          (String(selectedNode.id) === String(edge.source) ||
                            String(selectedNode.id) === String(edge.target));

                        const pts = edge.attributes?.pts;
                        if (!Array.isArray(pts) || pts.length < 2) return null;
                        const pathD = pts
                           .map((p, i) => `${i === 0 ? "M" : "L"} ${p[1]} ${p[0]}`)
                          .join(" ");

                        // Color mapping
                        let strokeColor = "#0066FF"; // Detected road: Bright electric blue
                        let strokeWidth = 1.8;
                        let filterStr: string | undefined = undefined;

                        if (isFailed) {
                          strokeColor = "#DC2626"; // Disrupted/Failed: Crimson red
                          strokeWidth = 3.2;
                        } else if (isSelected) {
                          strokeColor = isInferred || isOcclusionRecovered ? "#14B8A6" : "#00E5FF";
                          strokeWidth = 3.6;
                          filterStr = isInferred || isOcclusionRecovered ? "url(#glow-teal)" : "url(#glow-cyan)";
                        } else if (isConnectedToSelectedNode) {
                          strokeColor = "#00E5FF"; // Selected: Glowing Cyan
                          strokeWidth = 3.4;
                          filterStr = "url(#glow-cyan)";
                        } else if (layers.criticality && isCritical) {
                          strokeColor = "#FF5A36"; // Critical lifeline: Coral red
                          strokeWidth = 2.4;
                          filterStr = "url(#glow-coral)";
                        } else if (isOcclusionRecovered) {
                          strokeColor = "#10B981"; // Inferred/Occlusion-recovered: Emerald Green
                          strokeWidth = isInferred ? 3.0 : 2.2;
                        } else if (isHealed) {
                          strokeColor = "#F59E0B"; // Healed gap: Warm amber
                          strokeWidth = 2.0;
                        }

                        return (
                          <g key={`edge-${edge.source}-${edge.target}-${idx}`}>
                            {/* Hit-box */}
                            <path
                              d={pathD}
                              fill="none"
                              stroke="transparent"
                              strokeWidth={12}
                              className="cursor-pointer"
                              onClick={(e) => {
                                e.stopPropagation();
                                handleSelectEdge(edge.source, edge.target, edge.attributes);
                              }}
                            >
                              <title>{`Road #${edge.source} ↔ #${edge.target}`}</title>
                            </path>

                            {/* Rendered path */}
                            <path
                              d={pathD}
                              fill="none"
                              stroke={strokeColor}
                              strokeWidth={strokeWidth}
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeDasharray={isInferred ? "8 4" : undefined}
                              filter={filterStr}
                              className="pointer-events-none transition-all duration-150"
                            />
                          </g>
                        );
                      })}

                    {/* SUSPECTED OCCLUSIONS (Dashed Purple Lines) */}
                    {layers.occlusions &&
                      (graphData.candidate_occlusions ?? []).map((cand: any, idx: number) => {
                        if (cand.recovered) return null;
                        const isCandSelected =
                          selectedCandidate &&
                          ((String(selectedCandidate.source) === String(cand.source) &&
                            String(selectedCandidate.target) === String(cand.target)) ||
                            (String(selectedCandidate.source) === String(cand.target) &&
                              String(selectedCandidate.target) === String(cand.source)));

                        return (
                          <g key={`cand-occ-${cand.source}-${cand.target}-${idx}`}>
                            {/* Interactive Hit Target */}
                            <line
                              x1={cand.pos_source[1]}
                              y1={cand.pos_source[0]}
                              x2={cand.pos_target[1]}
                              y2={cand.pos_target[0]}
                              stroke="transparent"
                              strokeWidth={14}
                              className="cursor-pointer"
                              onClick={(e) => {
                                e.stopPropagation();
                                handleSelectCandidate(cand);
                              }}
                            >
                              <title>{`Suspected Occlusion Gap (${cand.gap_px}px) - Click to view Recovery Evidence`}</title>
                            </line>

                            {/* Selection Halo */}
                            {isCandSelected && (
                              <line
                                x1={cand.pos_source[1]}
                                y1={cand.pos_source[0]}
                                x2={cand.pos_target[1]}
                                y2={cand.pos_target[0]}
                                stroke="#D8B4FE"
                                strokeWidth={7}
                                strokeLinecap="round"
                                className="pointer-events-none opacity-70"
                              />
                            )}

                            {/* Rendered dashed line */}
                            <line
                              x1={cand.pos_source[1]}
                              y1={cand.pos_source[0]}
                              x2={cand.pos_target[1]}
                              y2={cand.pos_target[0]}
                              stroke={isCandSelected ? "#A855F7" : "#C084FC"}
                              strokeWidth={isCandSelected ? 3.2 : 2.0}
                              strokeDasharray={isCandSelected ? "6 3" : "4 3"}
                              strokeLinecap="round"
                              filter={isCandSelected ? "url(#glow-purple)" : undefined}
                              className="pointer-events-none opacity-90 transition-all duration-150"
                            />
                          </g>
                        );
                      })}

                    {/* NODES (Requirement 3: Number every node, collision-aware placement, control to show/hide) */}
                    {graphData.nodes.map((node) => {
                      const pos = node.attributes?.pos;
                      if (!Array.isArray(pos) || pos.length < 2) return null;
                      const [row, col] = pos;
                      const nodeIdStr = String(node.id);
                      const isImportant = importantNodeIds.has(nodeIdStr);
                      const isSelected = selectedNode && String(selectedNode.id) === nodeIdStr;
                      const isHovered = hoveredNodeId === node.id;
                      const isDisruptedNode = Boolean(
                        currentScenario?.failedNodes.some((n) => String(n) === nodeIdStr),
                      );

                      // Check if this node is an endpoint of the selected inferred edge or candidate occlusion
                      const isEndpointOfSelectedEdge =
                        selectedEdge &&
                        (String(selectedEdge.source) === nodeIdStr ||
                          String(selectedEdge.target) === nodeIdStr);
                      const isEndpointOfSelectedCandidate =
                        selectedCandidate &&
                        (String(selectedCandidate.source) === nodeIdStr ||
                          String(selectedCandidate.target) === nodeIdStr);
                      const isSelectedInferredEndpoint =
                        (Boolean(
                          selectedEdge &&
                            (selectedEdge.inferred ||
                              selectedEdge.occlusion_recovered ||
                              selectedEdge.healed),
                        ) &&
                          Boolean(isEndpointOfSelectedEdge)) ||
                        Boolean(isEndpointOfSelectedCandidate);

                      // Radius & appearance
                      const radius = isSelected
                        ? 4.5
                        : isSelectedInferredEndpoint
                          ? 4.0
                          : isImportant
                            ? 3.0
                            : 2.0;
                      const fill = isDisruptedNode
                        ? "#DC2626"
                        : isSelected
                          ? "#14B8A6"
                          : isSelectedInferredEndpoint
                            ? isEndpointOfSelectedCandidate
                              ? "#A855F7"
                              : "#14B8A6"
                            : isImportant
                              ? "#FFFFFF"
                              : isHovered
                                ? "#94A3B8"
                                : "#64748B";
                      const stroke = isSelected
                        ? "#0F766E"
                        : isSelectedInferredEndpoint
                          ? isEndpointOfSelectedCandidate
                            ? "#7E22CE"
                            : "#0F766E"
                          : isImportant
                            ? "#0D9488"
                            : "#090D16";
                      const strokeWidth = isSelected || isSelectedInferredEndpoint ? 2 : 1;

                      const shouldShowLabel =
                        isHovered ||
                        isSelected ||
                        isSelectedInferredEndpoint ||
                        (layers.junctions && nodeLabelMode === "all") ||
                        (layers.junctions && nodeLabelMode === "hubs" && isImportant);

                      const offset = nodeLabelPlacements.get(nodeIdStr) ?? { dx: 4, dy: -8 };
                      const textW = nodeIdStr.length * 6.5 + 8;

                      return (
                        <g
                          key={`node-${node.id}`}
                          className="cursor-pointer"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleSelectNode(node.id, pos);
                          }}
                          onMouseEnter={() => setHoveredNodeId(node.id)}
                          onMouseLeave={() => setHoveredNodeId(null)}
                        >
                          {/* Invisible wider hit target */}
                          <circle cx={col} cy={row} r={10} fill="transparent" />

                          {/* Accent highlight ring for inferred / candidate endpoints */}
                          {isSelectedInferredEndpoint && (
                            <circle
                              cx={col}
                              cy={row}
                              r={7.5}
                              fill="none"
                              stroke={isEndpointOfSelectedCandidate ? "#C084FC" : "#14B8A6"}
                              strokeWidth={2.4}
                              className="animate-pulse"
                            />
                          )}

                          {/* Node Dot */}
                          <circle
                            cx={col}
                            cy={row}
                            r={radius}
                            fill={fill}
                            stroke={stroke}
                            strokeWidth={strokeWidth}
                            className="transition-all duration-150"
                          />

                          {/* Numbered stable graph ID label */}
                          {shouldShowLabel && (
                            <g
                              transform={`translate(${col + offset.dx}, ${row + offset.dy})`}
                              className="pointer-events-none select-none"
                            >
                              <rect
                                x={-2}
                                y={-9}
                                width={textW}
                                height={13}
                                rx={3}
                                fill={
                                  isSelected
                                    ? "#0f766e"
                                    : isDisruptedNode
                                      ? "#DC2626"
                                      : isImportant
                                        ? "rgba(15, 23, 42, 0.92)"
                                        : "rgba(15, 23, 42, 0.82)"
                                }
                                stroke={
                                  isSelected
                                    ? "#2dd4bf"
                                    : isDisruptedNode
                                      ? "#f87171"
                                      : isImportant
                                        ? "rgba(20, 184, 166, 0.6)"
                                        : "rgba(100, 116, 139, 0.4)"
                                }
                                strokeWidth={isSelected ? 1.2 : 0.7}
                              />
                              <text
                                x={2}
                                y={1}
                                fill="#FFFFFF"
                                fontSize="8.5"
                                fontFamily="ui-monospace, monospace"
                                fontWeight="600"
                              >
                                #{nodeIdStr}
                              </text>
                            </g>
                          )}
                        </g>
                      );
                    })}
                  </svg>
                )}
              </div>

              {/* Visual Legend (Requirement 2 & 4: Scientific theme, accessible contrast) */}
              <div className="absolute top-4 right-4 z-20 bg-card/95 backdrop-blur-md rounded-xl border border-border shadow-xs p-3 text-[0.68rem] text-card-foreground pointer-events-auto">
                <div className="font-bold text-foreground text-[0.7rem] mb-2 uppercase tracking-wider flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-teal-600 dark:bg-teal-400" />
                  <span>Network Legend</span>
                </div>
                <div className="flex flex-col gap-1.5">
                  <div className="flex items-center gap-2">
                    <span className="w-3.5 h-1 rounded-full bg-[#3B82F6]" />
                    <span className="text-muted-foreground font-medium">Detected Road</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="w-3.5 h-1 rounded-full bg-[#0D9488]" />
                    <span className="text-muted-foreground font-medium">Inferred Road (Occlusion)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="w-3.5 h-1 rounded-full bg-[#D97706]" />
                    <span className="text-muted-foreground font-medium">Healed Gap</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="w-3.5 h-1 rounded-full bg-[#EA580C]" />
                    <span className="text-muted-foreground font-medium">Critical Lifeline</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="w-3.5 border-t-2 border-dashed border-[#A855F7]" />
                    <span className="text-muted-foreground font-medium">Suspected Occlusion</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full border border-teal-700 bg-white" />
                    <span className="text-muted-foreground font-medium">Graph Node (#ID)</span>
                  </div>
                </div>
              </div>

              {/* Floating Canvas Zoom Controls */}
              <div className="absolute bottom-4 left-4 z-20 flex items-center gap-1 bg-card/95 backdrop-blur-md border border-border rounded-xl p-1.5 shadow-xs text-foreground">
                <button
                  type="button"
                  onClick={handleZoomIn}
                  className="w-7 h-7 rounded-lg flex items-center justify-center text-muted-foreground hover:text-foreground hover:bg-muted transition"
                  title="Zoom In"
                >
                  <ZoomIn size={15} />
                </button>
                <button
                  type="button"
                  onClick={handleZoomOut}
                  className="w-7 h-7 rounded-lg flex items-center justify-center text-muted-foreground hover:text-foreground hover:bg-muted transition"
                  title="Zoom Out"
                >
                  <ZoomOut size={15} />
                </button>
                <div className="h-4 w-px bg-border mx-0.5" />
                <button
                  type="button"
                  onClick={handleResetView}
                  className="w-7 h-7 rounded-lg flex items-center justify-center text-muted-foreground hover:text-foreground hover:bg-muted transition"
                  title="Reset View"
                >
                  <RotateCcw size={14} />
                </button>
                <span className="px-1.5 font-mono text-[0.65rem] font-semibold text-muted-foreground">
                  {Math.round(zoom * 100)}%
                </span>
              </div>
            </div>

            {/* Bottom Help Text */}
            <div className="px-4 py-2 bg-muted/40 border-t border-border text-[0.68rem] text-muted-foreground flex flex-wrap items-center justify-between gap-2">
              <span className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-teal-600 dark:bg-teal-400" />
                Click any node or road to inspect connectivity and simulate disruptions.
              </span>
              <span className="text-muted-foreground/80 hidden sm:inline">Pan: Drag · Zoom: Scroll</span>
            </div>
          </section>

          {/* Right Inspector & Controls Panel */}
          <aside className="w-full lg:w-88 flex flex-col gap-4">
            {/* 1. Network Vital Metrics */}
            <div className="bg-card rounded-xl border border-border shadow-xs p-5">
              <div className="flex items-center justify-between pb-3 border-b border-border mb-3.5">
                <div>
                  <span className="text-[0.6rem] font-bold text-teal-600 dark:text-teal-400 uppercase tracking-wider">
                    Topology Overview
                  </span>
                  <h2 className="text-sm font-bold text-foreground">Network Metrics</h2>
                </div>
                <div className="w-7 h-7 rounded-lg bg-teal-500/10 text-teal-600 dark:text-teal-400 flex items-center justify-center border border-teal-500/20">
                  <Activity size={15} />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2.5 text-xs">
                <div className="bg-muted/40 rounded-lg p-2.5 border border-border">
                  <span className="block text-[0.6rem] font-semibold text-muted-foreground uppercase">
                    Nodes
                  </span>
                  <span className="text-base font-bold text-foreground font-mono">{nodeCount}</span>
                </div>

                <div className="bg-muted/40 rounded-lg p-2.5 border border-border">
                  <span className="block text-[0.6rem] font-semibold text-muted-foreground uppercase">
                    Roads
                  </span>
                  <span className="text-base font-bold text-foreground font-mono">{edgeCount}</span>
                </div>

                <div className="bg-muted/40 rounded-lg p-2.5 border border-border">
                  <span className="block text-[0.6rem] font-semibold text-muted-foreground uppercase">
                    Components
                  </span>
                  <span className="text-base font-bold text-foreground font-mono">{compCount}</span>
                </div>

                <div className="bg-muted/40 rounded-lg p-2.5 border border-border">
                  <span className="block text-[0.6rem] font-semibold text-muted-foreground uppercase">
                    Healed Gaps
                  </span>
                  <span className="text-base font-bold text-teal-600 dark:text-teal-400 font-mono">
                    {healedCount}
                  </span>
                </div>
              </div>

              {/* Status pill */}
              <div className="mt-3.5 pt-3 border-t border-border flex items-center justify-between text-xs">
                <span className="text-muted-foreground">Connectivity:</span>
                <span
                  className={`inline-flex items-center gap-1 font-semibold px-2 py-0.5 rounded-md text-[0.68rem] ${
                    compCount === 1
                      ? "bg-teal-500/10 text-teal-700 dark:text-teal-300 border border-teal-500/25"
                      : "bg-amber-500/10 text-amber-700 dark:text-amber-300 border border-amber-500/25"
                  }`}
                >
                  <span
                    className={`w-1.5 h-1.5 rounded-full ${compCount === 1 ? "bg-teal-500" : "bg-amber-500"}`}
                  />
                  {compCount === 1 ? "Connected" : `Fragmented (${compCount} Zones)`}
                </span>
              </div>
            </div>

            {/* 2. Layer Visibility Toggles */}
            <div className="bg-card rounded-xl border border-border shadow-xs p-5">
              <h3 className="text-xs font-bold text-foreground uppercase tracking-wider mb-3 flex items-center gap-1.5">
                <Layers size={14} className="text-teal-600 dark:text-teal-400" />
                <span>Layer Display</span>
              </h3>
              <div className="space-y-2 text-xs">
                <label className="flex items-center justify-between p-2 rounded-lg hover:bg-muted/50 cursor-pointer transition">
                  <span className="text-foreground font-medium">Satellite Imagery</span>
                  <input
                    type="checkbox"
                    checked={layers.satellite}
                    onChange={(e) => setLayers((l) => ({ ...l, satellite: e.target.checked }))}
                    className="rounded text-teal-600 focus:ring-0"
                  />
                </label>
                <label className="flex items-center justify-between p-2 rounded-lg hover:bg-muted/50 cursor-pointer transition">
                  <span className="text-foreground font-medium">Road Centerlines</span>
                  <input
                    type="checkbox"
                    checked={layers.roads}
                    onChange={(e) => setLayers((l) => ({ ...l, roads: e.target.checked }))}
                    className="rounded text-teal-600 focus:ring-0"
                  />
                </label>
                <label className="flex items-center justify-between p-2 rounded-lg hover:bg-muted/50 cursor-pointer transition">
                  <span className="text-foreground font-medium">Critical Lifelines</span>
                  <input
                    type="checkbox"
                    checked={layers.criticality}
                    onChange={(e) => setLayers((l) => ({ ...l, criticality: e.target.checked }))}
                    className="rounded text-teal-600 focus:ring-0"
                  />
                </label>
                <label className="flex items-center justify-between p-2 rounded-lg hover:bg-muted/50 cursor-pointer transition">
                  <span className="text-foreground font-medium">Occlusions & Inferred</span>
                  <input
                    type="checkbox"
                    checked={layers.occlusions}
                    onChange={(e) => setLayers((l) => ({ ...l, occlusions: e.target.checked }))}
                    className="rounded text-teal-600 focus:ring-0"
                  />
                </label>
                <label className="flex items-center justify-between p-2 rounded-lg hover:bg-muted/50 cursor-pointer transition">
                  <span className="text-foreground font-medium">Graph Node Labels (#ID)</span>
                  <input
                    type="checkbox"
                    checked={layers.junctions}
                    onChange={(e) => setLayers((l) => ({ ...l, junctions: e.target.checked }))}
                    className="rounded text-teal-600 focus:ring-0"
                  />
                </label>
              </div>
            </div>

            {/* 3. Detail Inspector (Selected Node or Selected Road) - Requirement 3 */}
            {selectedNode && (
              <div className="bg-card rounded-xl border border-teal-500/60 dark:border-teal-400/60 shadow-xs p-5 animate-in fade-in duration-150">
                <div className="flex items-center justify-between border-b border-border pb-2.5 mb-3">
                  <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-teal-500" />
                    <h3 className="font-bold text-sm text-foreground">
                      Node #{String(selectedNode.id)}
                    </h3>
                  </div>
                  <button
                    type="button"
                    onClick={() => setSelectedNode(null)}
                    className="p-1 rounded-md text-muted-foreground hover:text-foreground hover:bg-muted"
                  >
                    <X size={15} />
                  </button>
                </div>

                <div className="space-y-2 text-xs">
                  <div className="flex justify-between py-1 border-b border-border/50">
                    <span className="text-muted-foreground">Node Type:</span>
                    <span className="font-semibold text-foreground">
                      {selectedNode.isJunction
                        ? "Junction / Intersection"
                        : selectedNode.isArticulationPoint
                          ? "Bridge Junction"
                          : "Road Terminal"}
                    </span>
                  </div>

                  <div className="flex justify-between py-1 border-b border-border/50">
                    <span className="text-muted-foreground">Connections (Degree):</span>
                    <span className="font-bold text-teal-600 dark:text-teal-400 font-mono">
                      {selectedNode.degree} connected {selectedNode.degree === 1 ? "road" : "roads"}
                    </span>
                  </div>

                  {selectedNode.pos && (
                    <div className="flex justify-between py-1 border-b border-border/50">
                      <span className="text-muted-foreground">Pixel Position:</span>
                      <span className="font-mono text-muted-foreground">
                        ({selectedNode.pos[0].toFixed(0)}, {selectedNode.pos[1].toFixed(0)})
                      </span>
                    </div>
                  )}

                  {/* Connected Neighbors */}
                  <div className="pt-2">
                    <span className="block text-[0.62rem] font-bold text-muted-foreground uppercase mb-1.5">
                      Connected Roads:
                    </span>
                    <div className="space-y-1 max-h-28 overflow-y-auto">
                      {selectedNode.neighbors.map((neighbor) => (
                        <div
                          key={String(neighbor.id)}
                          onClick={() => handleSelectEdge(selectedNode.id, neighbor.id)}
                          className="flex items-center justify-between p-1.5 rounded-lg bg-muted/40 hover:bg-muted border border-border/70 cursor-pointer transition text-[0.68rem]"
                        >
                          <span className="font-medium text-foreground">
                            ↔ Node #{String(neighbor.id)}
                          </span>
                          <span className="font-mono text-muted-foreground">
                            {neighbor.length_px ? `${neighbor.length_px.toFixed(1)} px` : "—"}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Actions for Node */}
                  <div className="pt-3 flex flex-col gap-2">
                    <button
                      type="button"
                      onClick={() => handleSimulateNodeFailure(selectedNode.id)}
                      className="w-full inline-flex items-center justify-center gap-1.5 rounded-lg bg-destructive text-destructive-foreground py-2 text-xs font-semibold hover:bg-destructive/90 transition shadow-xs"
                    >
                      <ShieldAlert size={14} />
                      <span>Simulate Node Failure</span>
                    </button>
                    <div className="grid grid-cols-2 gap-2">
                      <button
                        type="button"
                        onClick={() => handleRouteFromNode(selectedNode.id)}
                        className="rounded-lg border border-border bg-card py-1.5 text-xs font-semibold text-foreground hover:bg-muted/60 transition shadow-2xs"
                      >
                        Route From
                      </button>
                      <button
                        type="button"
                        onClick={() => handleRouteToNode(selectedNode.id)}
                        className="rounded-lg border border-border bg-card py-1.5 text-xs font-semibold text-foreground hover:bg-muted/60 transition shadow-2xs"
                      >
                        Route To
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* 3A. RECOVERY EVIDENCE PANEL (for Inferred Roads or Candidate Occlusions) */}
            {(selectedCandidate || (selectedEdge && (selectedEdge.inferred || selectedEdge.occlusion_recovered || selectedEdge.healed || selectedEdge.evidence))) && (() => {
              const isInferredEdge = Boolean(
                selectedEdge &&
                  (selectedEdge.inferred ||
                    selectedEdge.occlusion_recovered ||
                    selectedEdge.healed ||
                    selectedEdge.evidence),
              );

              const activeEvidence = selectedCandidate
                ? {
                    source: selectedCandidate.source,
                    target: selectedCandidate.target,
                    isRecovered: selectedCandidate.recovered,
                    status: selectedCandidate.status,
                    recoveryType: selectedCandidate.occlusion_kind ?? "occlusion_candidate",
                    gapPx: selectedCandidate.gap_px,
                    headingAlignment: selectedCandidate.heading_alignment,
                    mutualHeading: selectedCandidate.mutual_heading,
                    lateralOffsetPx: selectedCandidate.lateral_offset_px,
                    cloudCoverage: selectedCandidate.cloud_coverage,
                    sinuosity: selectedCandidate.evidence?.sinuosity,
                    probabilityEvidence: selectedCandidate.evidence?.probability_evidence,
                    supportedFraction: selectedCandidate.evidence?.supported_fraction,
                    flankContrast: selectedCandidate.evidence?.flank_contrast,
                    acceptanceReason: selectedCandidate.acceptance_reason ?? selectedCandidate.evidence?.acceptance_reason,
                    rejectionReason: selectedCandidate.rejection_reason,
                    checks: selectedCandidate.checks ?? selectedCandidate.evidence?.checks ?? [],
                    isCandidateOnly: true,
                  }
                : isInferredEdge && selectedEdge
                  ? {
                      source: selectedEdge.source,
                      target: selectedEdge.target,
                      isRecovered: true,
                      status: "recovered",
                      recoveryType: selectedEdge.recovery_type ?? (selectedEdge.occlusion_recovered ? "occlusion_continuity" : "evidence_gated_direct"),
                      gapPx: selectedEdge.gap_px ?? selectedEdge.length_px,
                      headingAlignment: selectedEdge.heading_alignment,
                      dotA: selectedEdge.dot_a,
                      dotB: selectedEdge.dot_b,
                      mutualHeading: selectedEdge.mutual_heading,
                      lateralOffsetPx: selectedEdge.lateral_offset_px,
                      cloudCoverage: selectedEdge.cloud_coverage,
                      sinuosity: selectedEdge.sinuosity,
                      probabilityEvidence: selectedEdge.probability_evidence ?? selectedEdge.confidence,
                      supportedFraction: selectedEdge.supported_fraction,
                      flankContrast: selectedEdge.flank_contrast,
                      acceptanceReason: selectedEdge.acceptance_reason ?? (selectedEdge.recovery_type === "cloud_continuity" ? "Accepted via Cloud Occlusion Continuity: collinear tangents and cloud coverage bridge the obscured span." : selectedEdge.occlusion_recovered ? "Accepted via Occlusion-Aware Collinear Continuity: strong collinear endpoint alignment justifies bridging across low-confidence occlusion span." : "Accepted via Direct Evidence-Gated Gap: straight segment with strong road probability support."),
                      rejectionReason: undefined,
                      checks: selectedEdge.checks ?? [],
                      isCandidateOnly: false,
                    }
                  : null;

              if (!activeEvidence) return null;

              return (
                <div className="bg-card rounded-xl border border-teal-500/50 shadow-xs p-5 animate-in fade-in duration-150 space-y-3.5">
                  {/* Panel Header */}
                  <div className="flex items-start justify-between border-b border-border pb-2.5">
                    <div>
                      <div className="flex items-center gap-1.5">
                        <Sparkles size={14} className="text-teal-600 dark:text-teal-400" />
                        <span className="text-[0.62rem] font-bold uppercase tracking-wider text-teal-700 dark:text-teal-300">
                          Recovery Evidence
                        </span>
                      </div>
                      <h3 className="font-bold text-sm text-foreground mt-0.5">
                        Road #{String(activeEvidence.source)} ↔ #{String(activeEvidence.target)}
                      </h3>
                    </div>
                    <button
                      type="button"
                      onClick={() => {
                        setSelectedEdge(null);
                        setSelectedCandidate(null);
                      }}
                      className="p-1 rounded-md text-muted-foreground hover:text-foreground hover:bg-muted"
                      title="Close panel"
                    >
                      <X size={15} />
                    </button>
                  </div>

                  {/* Status Badge */}
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-muted-foreground">Status:</span>
                    <span
                      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[0.68rem] font-bold uppercase tracking-wide border ${
                        activeEvidence.isRecovered
                          ? "bg-teal-500/10 text-teal-700 dark:text-teal-300 border-teal-500/30"
                          : "bg-destructive/10 text-destructive border-destructive/20"
                      }`}
                    >
                      {activeEvidence.isRecovered ? (
                        <>
                          <CheckCircle2 size={12} />
                          <span>Accepted & Inferred</span>
                        </>
                      ) : (
                        <>
                          <XCircle size={12} />
                          <span>Rejected Candidate</span>
                        </>
                      )}
                    </span>
                  </div>

                  {/* Reason Callout Box */}
                  {activeEvidence.isRecovered && activeEvidence.acceptanceReason && (
                    <div className="rounded-lg bg-teal-500/10 border border-teal-500/20 p-2.5 text-xs text-foreground">
                      <span className="block font-bold text-[0.65rem] uppercase text-teal-800 dark:text-teal-200 mb-0.5">
                        Why Accepted
                      </span>
                      <p className="text-[0.72rem] leading-relaxed text-muted-foreground">
                        {activeEvidence.acceptanceReason}
                      </p>
                    </div>
                  )}

                  {!activeEvidence.isRecovered && activeEvidence.rejectionReason && (
                    <div className="rounded-lg bg-destructive/10 border border-destructive/20 p-2.5 text-xs text-foreground">
                      <span className="block font-bold text-[0.65rem] uppercase text-destructive mb-0.5">
                        Rejection Reason
                      </span>
                      <p className="text-[0.72rem] leading-relaxed text-muted-foreground">
                        {activeEvidence.rejectionReason}
                      </p>
                    </div>
                  )}

                  {/* Evidence Values Grid */}
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div className="bg-muted/40 p-2 rounded-lg border border-border/70">
                      <span className="text-[0.6rem] font-bold uppercase text-muted-foreground block">
                        Gap Distance
                      </span>
                      <span className="text-xs font-mono font-bold text-foreground">
                        {activeEvidence.gapPx !== undefined ? `${activeEvidence.gapPx.toFixed(1)} px` : "—"}
                      </span>
                    </div>

                    <div className="bg-muted/40 p-2 rounded-lg border border-border/70">
                      <span className="text-[0.6rem] font-bold uppercase text-muted-foreground block">
                        Heading Align
                      </span>
                      <span className="text-xs font-mono font-bold text-foreground">
                        {activeEvidence.headingAlignment !== undefined ? activeEvidence.headingAlignment.toFixed(3) : "—"}
                      </span>
                    </div>

                    <div className="bg-muted/40 p-2 rounded-lg border border-border/70">
                      <span className="text-[0.6rem] font-bold uppercase text-muted-foreground block">
                        Mutual Tangent
                      </span>
                      <span className="text-xs font-mono font-bold text-foreground">
                        {activeEvidence.mutualHeading !== undefined ? activeEvidence.mutualHeading.toFixed(3) : "—"}
                      </span>
                    </div>

                    <div className="bg-muted/40 p-2 rounded-lg border border-border/70">
                      <span className="text-[0.6rem] font-bold uppercase text-muted-foreground block">
                        Lateral Offset
                      </span>
                      <span className="text-xs font-mono font-bold text-foreground">
                        {activeEvidence.lateralOffsetPx !== undefined ? `${activeEvidence.lateralOffsetPx.toFixed(1)} px` : "—"}
                      </span>
                    </div>
                  </div>

                  {/* Probability / Cloud Evidence */}
                  {(activeEvidence.cloudCoverage !== undefined || activeEvidence.probabilityEvidence !== undefined) && (
                    <div className="space-y-1 text-xs pt-1 border-t border-border/50">
                      {activeEvidence.cloudCoverage !== undefined && (
                        <div className="flex justify-between py-0.5">
                          <span className="text-muted-foreground">Cloud Mask Coverage:</span>
                          <span className="font-mono font-semibold text-foreground">
                            {(activeEvidence.cloudCoverage * 100).toFixed(1)}%
                          </span>
                        </div>
                      )}
                      {activeEvidence.probabilityEvidence !== undefined && (
                        <div className="flex justify-between py-0.5">
                          <span className="text-muted-foreground">Probability Evidence:</span>
                          <span className="font-mono font-semibold text-foreground">
                            {activeEvidence.probabilityEvidence.toFixed(3)}
                          </span>
                        </div>
                      )}
                      {activeEvidence.supportedFraction !== undefined && (
                        <div className="flex justify-between py-0.5">
                          <span className="text-muted-foreground">Supported Fraction:</span>
                          <span className="font-mono font-semibold text-foreground">
                            {(activeEvidence.supportedFraction * 100).toFixed(1)}%
                          </span>
                        </div>
                      )}
                      {activeEvidence.flankContrast !== undefined && activeEvidence.flankContrast !== null && (
                        <div className="flex justify-between py-0.5">
                          <span className="text-muted-foreground">Flank Contrast:</span>
                          <span className="font-mono font-semibold text-foreground">
                            {activeEvidence.flankContrast > 0 ? `+${activeEvidence.flankContrast.toFixed(3)}` : activeEvidence.flankContrast.toFixed(3)}
                          </span>
                        </div>
                      )}
                      {activeEvidence.sinuosity !== undefined && (
                        <div className="flex justify-between py-0.5">
                          <span className="text-muted-foreground">Path Sinuosity:</span>
                          <span className="font-mono font-semibold text-foreground">
                            {activeEvidence.sinuosity.toFixed(2)}
                          </span>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Continuity Checks Actually Used by the Algorithm */}
                  {activeEvidence.checks && activeEvidence.checks.length > 0 && (
                    <div className="pt-2 border-t border-border/50">
                      <span className="block text-[0.62rem] font-bold text-muted-foreground uppercase mb-1.5 flex items-center justify-between">
                        <span>Continuity Checks ({activeEvidence.checks.length})</span>
                        <span className="text-[0.6rem] text-teal-600 dark:text-teal-400 font-normal">Algorithm Evaluation</span>
                      </span>
                      <div className="space-y-1.5 max-h-36 overflow-y-auto pr-1">
                        {activeEvidence.checks.map((chk, cIdx) => (
                          <div
                            key={`chk-${chk.name}-${cIdx}`}
                            className="flex items-start justify-between gap-2 p-1.5 rounded-lg bg-muted/30 border border-border/60 text-[0.68rem]"
                          >
                            <div className="flex items-start gap-1.5 min-w-0">
                              {chk.passed ? (
                                <CheckCircle2 size={13} className="text-teal-600 dark:text-teal-400 shrink-0 mt-0.5" />
                              ) : (
                                <XCircle size={13} className="text-destructive shrink-0 mt-0.5" />
                              )}
                              <div>
                                <span className="font-medium text-foreground block leading-tight">{chk.name}</span>
                                <span className="font-mono text-[0.62rem] text-muted-foreground block leading-tight mt-0.5">{chk.detail}</span>
                              </div>
                            </div>
                            <span
                              className={`shrink-0 font-bold uppercase text-[0.58rem] px-1 py-0.2 rounded ${
                                chk.passed ? "text-teal-700 dark:text-teal-300 bg-teal-500/10" : "text-destructive bg-destructive/10"
                              }`}
                            >
                              {chk.passed ? "Pass" : "Fail"}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Actions */}
                  {!activeEvidence.isCandidateOnly && (
                    <div className="pt-2 border-t border-border/50">
                      <button
                        type="button"
                        onClick={() =>
                          handleSimulateEdgeFailure(activeEvidence.source, activeEvidence.target)
                        }
                        className="w-full inline-flex items-center justify-center gap-1.5 rounded-lg bg-destructive text-destructive-foreground py-2 text-xs font-semibold hover:bg-destructive/90 transition shadow-xs"
                      >
                        <ShieldAlert size={14} />
                        <span>Simulate Disruption on this Inferred Road</span>
                      </button>
                    </div>
                  )}
                </div>
              );
            })()}

            {/* 3B. STANDARD ROAD INSPECTOR (for normal detected roads) */}
            {selectedEdge && !selectedEdge.inferred && !selectedEdge.occlusion_recovered && !selectedEdge.healed && !selectedEdge.evidence && (
              <div className="bg-card rounded-xl border border-primary/60 shadow-xs p-5 animate-in fade-in duration-150">
                <div className="flex items-center justify-between border-b border-border pb-2.5 mb-3">
                  <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-primary" />
                    <h3 className="font-bold text-sm text-foreground">
                      Road #{String(selectedEdge.source)} ↔ #{String(selectedEdge.target)}
                    </h3>
                  </div>
                  <button
                    type="button"
                    onClick={() => setSelectedEdge(null)}
                    className="p-1 rounded-md text-muted-foreground hover:text-foreground hover:bg-muted"
                  >
                    <X size={15} />
                  </button>
                </div>

                <div className="space-y-2 text-xs">
                  <div className="flex justify-between py-1 border-b border-border/50">
                    <span className="text-muted-foreground">Classification:</span>
                    <span className="font-semibold text-foreground">
                      Detected Road
                    </span>
                  </div>

                  <div className="flex justify-between py-1 border-b border-border/50">
                    <span className="text-muted-foreground">Segment Length:</span>
                    <span className="font-mono font-semibold text-foreground">
                      {selectedEdge.length_px !== undefined
                        ? `${selectedEdge.length_px.toFixed(1)} px`
                        : "—"}
                    </span>
                  </div>

                  <div className="flex justify-between py-1 border-b border-border/50">
                    <span className="text-muted-foreground">Criticality:</span>
                    <span
                      className={`font-semibold ${selectedEdge.rank ? "text-amber-600 dark:text-amber-400" : "text-foreground"}`}
                    >
                      {selectedEdge.rank
                        ? `Critical (Rank #${selectedEdge.rank})`
                        : selectedEdge.is_bridge
                          ? "Bridge Segment"
                          : "Normal"}
                    </span>
                  </div>

                  {selectedEdge.betweenness !== undefined && (
                    <div className="flex justify-between py-1 border-b border-border/50">
                      <span className="text-muted-foreground">Betweenness Centrality:</span>
                      <span className="font-mono text-muted-foreground">
                        {selectedEdge.betweenness.toFixed(4)}
                      </span>
                    </div>
                  )}

                  {/* Actions for Road */}
                  <div className="pt-3">
                    <button
                      type="button"
                      onClick={() =>
                        handleSimulateEdgeFailure(selectedEdge.source, selectedEdge.target)
                      }
                      className="w-full inline-flex items-center justify-center gap-1.5 rounded-lg bg-destructive text-destructive-foreground py-2 text-xs font-semibold hover:bg-destructive/90 transition shadow-xs"
                    >
                      <ShieldAlert size={14} />
                      <span>Simulate Road Failure</span>
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* 3C. INFERRED CONNECTIONS & CANDIDATES BROWSER */}
            {graphData && (
              <div className="bg-card rounded-xl border border-border p-4 shadow-xs">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[0.62rem] font-bold text-muted-foreground uppercase tracking-wider">
                    Inferred Roads & Occlusions
                  </span>
                  <span className="text-[0.62rem] font-mono text-teal-700 dark:text-teal-300 bg-teal-500/10 px-2 py-0.5 rounded border border-teal-500/20">
                    {graphData.edges.filter((e) => e.attributes.inferred || e.attributes.occlusion_recovered || e.attributes.healed).length} Inferred
                  </span>
                </div>

                <div className="space-y-1.5 max-h-40 overflow-y-auto pr-1">
                  {/* Recovered Inferred Edges */}
                  {graphData.edges
                    .filter((e) => e.attributes.inferred || e.attributes.occlusion_recovered || e.attributes.healed)
                    .map((edge) => {
                      const isEdgeSel =
                        selectedEdge &&
                        ((String(selectedEdge.source) === String(edge.source) &&
                          String(selectedEdge.target) === String(edge.target)) ||
                          (String(selectedEdge.source) === String(edge.target) &&
                            String(selectedEdge.target) === String(edge.source)));
                      return (
                        <div
                          key={`inferred-list-${edge.source}-${edge.target}`}
                          onClick={() => handleSelectEdge(edge.source, edge.target, edge.attributes)}
                          className={`flex items-center justify-between p-2 rounded-lg border text-xs cursor-pointer transition ${
                            isEdgeSel
                              ? "border-teal-500 bg-teal-500/10 text-foreground"
                              : "border-border/70 bg-muted/40 hover:bg-muted text-muted-foreground hover:text-foreground"
                          }`}
                        >
                          <div className="flex items-center gap-1.5">
                            <span className="w-2 h-2 rounded-full bg-teal-500 shrink-0" />
                            <span className="font-mono font-semibold">
                              #{String(edge.source)} ↔ #{String(edge.target)}
                            </span>
                          </div>
                          <span className="text-[0.62rem] font-mono text-muted-foreground">
                            {edge.attributes.length_px ? `${edge.attributes.length_px.toFixed(1)} px` : "—"}
                          </span>
                        </div>
                      );
                    })}

                  {/* Rejected Candidate Occlusions */}
                  {(graphData.candidate_occlusions ?? [])
                    .filter((c) => !c.recovered)
                    .map((cand, cIdx) => {
                      const isCandSel =
                        selectedCandidate &&
                        ((String(selectedCandidate.source) === String(cand.source) &&
                          String(selectedCandidate.target) === String(cand.target)) ||
                          (String(selectedCandidate.source) === String(cand.target) &&
                            String(selectedCandidate.target) === String(cand.source)));
                      return (
                        <div
                          key={`cand-list-${cand.source}-${cand.target}-${cIdx}`}
                          onClick={() => handleSelectCandidate(cand)}
                          className={`flex items-center justify-between p-2 rounded-lg border text-xs cursor-pointer transition ${
                            isCandSel
                              ? "border-purple-500 bg-purple-500/10 text-foreground"
                              : "border-border/70 bg-muted/40 hover:bg-muted text-muted-foreground hover:text-foreground"
                          }`}
                        >
                          <div className="flex items-center gap-1.5">
                            <span className="w-2 h-2 rounded-full bg-purple-500 shrink-0" />
                            <span className="font-mono font-semibold">
                              #{String(cand.source)} ↔ #{String(cand.target)}
                            </span>
                            <span className="text-[0.58rem] uppercase font-bold text-destructive px-1 py-0.2 bg-destructive/10 rounded">
                              Rejected
                            </span>
                          </div>
                          <span className="text-[0.62rem] font-mono text-muted-foreground">
                            {cand.gap_px ? `${cand.gap_px.toFixed(1)} px` : "—"}
                          </span>
                        </div>
                      );
                    })}

                  {graphData.edges.filter((e) => e.attributes.inferred || e.attributes.occlusion_recovered || e.attributes.healed).length === 0 &&
                    (!graphData.candidate_occlusions || graphData.candidate_occlusions.filter((c) => !c.recovered).length === 0) && (
                      <p className="text-[0.68rem] text-muted-foreground p-1 text-center">
                        No inferred roads or candidate gaps detected for this network.
                      </p>
                    )}
                </div>
              </div>
            )}

            {/* 4. Workflow Navigation Card */}
            <div className="bg-muted/30 rounded-xl border border-border p-4">
              <span className="text-[0.62rem] font-bold text-muted-foreground uppercase tracking-wider block mb-2">
                Next Steps
              </span>
              <div className="flex flex-col gap-2">
                <Link
                  to="/disruption"
                  search={{ analysis: activeAnalysisId }}
                  className="flex items-center justify-between p-2.5 rounded-lg bg-card border border-border text-xs font-semibold text-foreground hover:border-destructive/60 hover:text-destructive transition shadow-2xs"
                >
                  <span className="flex items-center gap-2">
                    <ShieldAlert size={15} className="text-destructive" />
                    <span>03 · Disrupt Network</span>
                  </span>
                  <ArrowRight size={14} />
                </Link>
                <Link
                  to="/routing"
                  search={{ analysis: activeAnalysisId }}
                  className="flex items-center justify-between p-2.5 rounded-lg bg-card border border-border text-xs font-semibold text-foreground hover:border-teal-500/60 hover:text-teal-600 dark:hover:text-teal-400 transition shadow-2xs"
                >
                  <span className="flex items-center gap-2">
                    <RouteIcon size={15} className="text-teal-600 dark:text-teal-400" />
                    <span>04 · Reroute Around Failure</span>
                  </span>
                  <ArrowRight size={14} />
                </Link>
              </div>
            </div>
          </aside>
        </div>
      )}
    </div>
  );
}
