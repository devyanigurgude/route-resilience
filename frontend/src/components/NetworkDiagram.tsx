import type { FailedEdge, GraphData, NodeId } from "@/lib/backend";
import { Tag } from "lucide-react";
import { useMemo, useState } from "react";

type NetworkDiagramProps = {
  graph: GraphData;
  imageUrl?: string | undefined;
  route?: NodeId[] | undefined;
  originalRoute?: NodeId[] | undefined;
  alternativeRoute?: NodeId[] | undefined;
  failedEdges?: FailedEdge[] | undefined;
  restoredEdges?: FailedEdge[] | undefined;
  criticalEdges?: FailedEdge[] | undefined;
  onEdgeClick?: ((edge: FailedEdge) => void) | undefined;
  onNodeClick?: ((nodeId: NodeId) => void) | undefined;
  selectedNodeId?: NodeId | null | undefined;
};

function sameEdge(a: NodeId, b: NodeId, edge: FailedEdge) {
  return (
    (String(a) === String(edge[0]) && String(b) === String(edge[1])) ||
    (String(a) === String(edge[1]) && String(b) === String(edge[0]))
  );
}

export function NetworkDiagram({
  graph,
  imageUrl,
  route = [],
  originalRoute = [],
  alternativeRoute = [],
  failedEdges = [],
  restoredEdges = [],
  criticalEdges = [],
  onEdgeClick,
  onNodeClick,
  selectedNodeId,
}: NetworkDiagramProps) {
  const [imageSize, setImageSize] = useState<{ width: number; height: number } | null>(null);
  const [hoveredNode, setHoveredNode] = useState<NodeId | null>(null);
  const [labelMode, setLabelMode] = useState<"all" | "hubs" | "focus">("all");

  const extent = graph.edges
    .flatMap((edge) => edge.attributes.pts ?? [])
    .reduce(
      (bounds, point) => ({
        width: Math.max(bounds.width, point[1] + 1),
        height: Math.max(bounds.height, point[0] + 1),
      }),
      { width: 1, height: 1 },
    );
  const width = imageSize?.width ?? extent.width;
  const height = imageSize?.height ?? extent.height;

  // Compute node degrees, neighbors, and hubs
  const { nodeDegree, nodeNeighbors, importantNodes } = useMemo(() => {
    const deg = new Map<string, number>();
    const neighbors = new Map<string, string[]>();

    graph.edges.forEach((e) => {
      const u = String(e.source);
      const v = String(e.target);
      deg.set(u, (deg.get(u) || 0) + 1);
      deg.set(v, (deg.get(v) || 0) + 1);

      if (!neighbors.has(u)) neighbors.set(u, []);
      if (!neighbors.has(v)) neighbors.set(v, []);
      neighbors.get(u)!.push(v);
      neighbors.get(v)!.push(u);
    });

    const important = new Set<string>();
    deg.forEach((d, id) => {
      if (d >= 3) important.add(id);
    });

    // Route endpoints are also important
    if (originalRoute.length > 0) {
      important.add(String(originalRoute[0]));
      important.add(String(originalRoute[originalRoute.length - 1]));
    }
    if (alternativeRoute.length > 0) {
      important.add(String(alternativeRoute[0]));
      important.add(String(alternativeRoute[alternativeRoute.length - 1]));
    }
    return { nodeDegree: deg, nodeNeighbors: neighbors, importantNodes: important };
  }, [graph, originalRoute, alternativeRoute]);

  // Collision-aware label placements for readable numbering
  const labelPlacements = useMemo(() => {
    const placements = new Map<string, { dx: number; dy: number }>();
    const occupied: Array<{ x: number; y: number; w: number; h: number }> = [];

    graph.nodes.forEach((node) => {
      const pos = node.attributes?.pos;
      if (!pos) return;
      const col = pos[1];
      const row = pos[0];
      const idStr = String(node.id);
      const textW = idStr.length * 6.5 + 8;
      const h = 12;

      // Quadrants: top-right, bottom-right, top-left, bottom-left
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
  }, [graph.nodes]);

  const hasRouteData = originalRoute.length > 0 || alternativeRoute.length > 0 || route.length > 0;

  // Active inspected node
  const activeNodeId = hoveredNode ?? selectedNodeId ?? null;
  const activeNodeDegree = activeNodeId !== null ? nodeDegree.get(String(activeNodeId)) ?? 0 : null;
  const activeNodeNeighborCount =
    activeNodeId !== null ? (nodeNeighbors.get(String(activeNodeId))?.length ?? 0) : null;

  return (
    <div
      className="relative overflow-hidden rounded-xl border border-border bg-[#0a0d12] select-none shadow-xs"
      style={{ aspectRatio: `${width} / ${height}` }}
    >
      {/* 1. Base Satellite Layer (Unaltered natural colors) */}
      {imageUrl && (
        <img
          src={imageUrl}
          alt="Satellite base"
          onLoad={(event) =>
            setImageSize({
              width: event.currentTarget.naturalWidth,
              height: event.currentTarget.naturalHeight,
            })
          }
          className="absolute h-full w-full object-fill pointer-events-none select-none satellite-imagery"
          style={{ filter: "none" }}
          draggable={false}
        />
      )}

      {/* 2. Vector SVG Overlay */}
      <svg
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
        className="relative z-10 block h-full w-full"
      >
        <defs>
          <filter id="diagram-glow-green" x="-20%" y="-20%" width="140%" height="140%">
            <feDropShadow dx="0" dy="0" stdDeviation="2" floodColor="#10B981" />
          </filter>
          <filter id="diagram-glow-blue" x="-20%" y="-20%" width="140%" height="140%">
            <feDropShadow dx="0" dy="0" stdDeviation="2" floodColor="#0ea5e9" />
          </filter>
        </defs>

        {/* ROAD EDGES */}
        {graph.edges.map((edge) => {
          const points = edge.attributes.pts;
          if (!points || points.length < 2) return null;
          const path = points
            .map(([row, col], index) => `${index === 0 ? "M" : "L"} ${col} ${row}`)
            .join(" ");

          const isFailed = failedEdges.some((item) => sameEdge(edge.source, edge.target, item));
          const isRestored = restoredEdges.some((item) => sameEdge(edge.source, edge.target, item));

          const isOriginalRoute = originalRoute.some((node, index) => {
            const nextNode = originalRoute[index + 1];
            return (
              index < originalRoute.length - 1 &&
              nextNode !== undefined &&
              sameEdge(edge.source, edge.target, [node, nextNode])
            );
          });

          const isAlternativeRoute = alternativeRoute.some((node, index) => {
            const nextNode = alternativeRoute[index + 1];
            return (
              index < alternativeRoute.length - 1 &&
              nextNode !== undefined &&
              sameEdge(edge.source, edge.target, [node, nextNode])
            );
          });

          const isRoute = route.some((node, index) => {
            const nextNode = route[index + 1];
            return (
              index < route.length - 1 &&
              nextNode !== undefined &&
              sameEdge(edge.source, edge.target, [node, nextNode])
            );
          });

          const isCritical = criticalEdges.some((item) => sameEdge(edge.source, edge.target, item));
          const isOcclusionRecovered = Boolean(edge.attributes.occlusion_recovered);
          const isInferred = Boolean(edge.attributes.inferred);
          const isHealed = Boolean(edge.attributes.healed);

          // Route hierarchy
          let color = "#3b82f6"; // default detected road: vibrant scientific blue
          let strokeWidth = 1.6;
          let filter: string | undefined = undefined;

          if (isFailed) {
            color = "#DC2626"; // Crimson
            strokeWidth = 3.2;
          } else if (isAlternativeRoute) {
            color = "#10B981"; // Emerald Green for alternative detour
            strokeWidth = 3.4;
            filter = "url(#diagram-glow-green)";
          } else if (isOriginalRoute) {
            color = "#2563EB"; // Blue for original route
            strokeWidth = 3.0;
            filter = "url(#diagram-glow-blue)";
          } else if (isRoute) {
            color = "#14b8a6";
            strokeWidth = 2.8;
          } else if (isRestored) {
            color = "#10B981";
            strokeWidth = 2.2;
          } else if (isCritical) {
            color = "#EA580C"; // Burnt Orange
            strokeWidth = 2.2;
          } else if (isOcclusionRecovered) {
            color = "#0D9488"; // Inferred road across occlusion: teal
            strokeWidth = 2.2;
          } else if (isHealed) {
            color = "#D97706"; // Healed gap: amber
            strokeWidth = 1.8;
          }

          return (
            <g key={`${String(edge.source)}-${String(edge.target)}`}>
              {/* Visible Path */}
              <path
                d={path}
                fill="none"
                stroke={color}
                strokeWidth={strokeWidth}
                strokeDasharray={
                  isFailed && isOriginalRoute
                    ? "4 3"
                    : isInferred && !isAlternativeRoute && !isOriginalRoute
                      ? "8 4"
                      : undefined
                }
                strokeLinecap="round"
                strokeLinejoin="round"
                filter={filter}
                className="pointer-events-none transition-all duration-150"
              />

              {/* Edge Hit Target */}
              {onEdgeClick && (
                <path
                  d={path}
                  fill="none"
                  stroke="transparent"
                  strokeWidth={12}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="cursor-pointer"
                  onClick={() => onEdgeClick([edge.source, edge.target])}
                >
                  <title>{`Road #${edge.source} ↔ #${edge.target}`}</title>
                </path>
              )}
            </g>
          );
        })}

        {/* CANDIDATE OCCLUSIONS */}
        {(graph.candidate_occlusions ?? []).map((cand, idx) => {
          if (cand.recovered) return null;
          return (
            <line
              key={`cand-occ-${idx}`}
              x1={cand.pos_source[1]}
              y1={cand.pos_source[0]}
              x2={cand.pos_target[1]}
              y2={cand.pos_target[0]}
              stroke="#A855F7"
              strokeWidth={1.8}
              strokeDasharray="4 3"
              strokeLinecap="round"
              className="pointer-events-none opacity-85"
            >
              <title>{`Suspected occlusion gap: ${cand.gap_px}px`}</title>
            </line>
          );
        })}

        {/* NODES (Numbered with stable graph IDs, collision-aware placement) */}
        {graph.nodes.map((node) => {
          const position = node.attributes.pos;
          if (!position) return null;
          const nodeIdStr = String(node.id);
          const isImportant = importantNodes.has(nodeIdStr);
          const isSelected = selectedNodeId && String(selectedNodeId) === nodeIdStr;
          const isHovered = hoveredNode === node.id;
          const col = position[1];
          const row = position[0];

          // Determine label visibility based on mode
          const shouldShowLabel =
            isHovered ||
            isSelected ||
            labelMode === "all" ||
            (labelMode === "hubs" && isImportant);

          const offset = labelPlacements.get(nodeIdStr) ?? { dx: 4, dy: -8 };
          const textW = nodeIdStr.length * 6.5 + 8;

          return (
            <g
              key={nodeIdStr}
              className={onNodeClick ? "cursor-pointer" : undefined}
              onClick={() => onNodeClick?.(node.id)}
              onMouseEnter={() => setHoveredNode(node.id)}
              onMouseLeave={() => setHoveredNode(null)}
            >
              {/* Hitbox */}
              {onNodeClick && <circle cx={col} cy={row} r={9} fill="transparent" />}

              {/* Dot */}
              <circle
                cx={col}
                cy={row}
                r={isSelected ? 4.0 : isImportant ? 2.8 : 1.8}
                fill={
                  isSelected
                    ? "#14b8a6"
                    : isImportant
                      ? "#ffffff"
                      : isHovered
                        ? "#94a3b8"
                        : "#64748b"
                }
                stroke={isSelected ? "#0f766e" : isImportant ? "#0d9488" : "#090d16"}
                strokeWidth={isSelected ? 2.0 : isImportant ? 1.0 : 0.8}
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
                        : isImportant
                          ? "rgba(15, 23, 42, 0.92)"
                          : "rgba(15, 23, 42, 0.82)"
                    }
                    stroke={
                      isSelected
                        ? "#2dd4bf"
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

      {/* Control: Label Mode Toggle */}
      <div className="absolute top-2 left-2 z-20 flex items-center gap-1 rounded-lg border border-border/80 bg-card/95 px-1.5 py-1 text-[0.62rem] font-medium text-foreground backdrop-blur-md shadow-xs">
        <Tag size={12} className="text-muted-foreground mr-0.5" />
        <span className="text-muted-foreground mr-1 hidden sm:inline">Labels:</span>
        <button
          type="button"
          onClick={() => setLabelMode("all")}
          className={`px-1.5 py-0.5 rounded transition ${
            labelMode === "all"
              ? "bg-primary text-primary-foreground font-semibold"
              : "text-muted-foreground hover:text-foreground"
          }`}
          title="Number every node with its stable ID"
        >
          All ({graph.nodes.length})
        </button>
        <button
          type="button"
          onClick={() => setLabelMode("hubs")}
          className={`px-1.5 py-0.5 rounded transition ${
            labelMode === "hubs"
              ? "bg-primary text-primary-foreground font-semibold"
              : "text-muted-foreground hover:text-foreground"
          }`}
          title="Show numbers for junctions and route terminals"
        >
          Hubs ({importantNodes.size})
        </button>
        <button
          type="button"
          onClick={() => setLabelMode("focus")}
          className={`px-1.5 py-0.5 rounded transition ${
            labelMode === "focus"
              ? "bg-primary text-primary-foreground font-semibold"
              : "text-muted-foreground hover:text-foreground"
          }`}
          title="Only show label on hover or click"
        >
          Hover
        </button>
      </div>

      {/* Floating Node Metric Badge (Requirement 3: clicking/hovering reveals ID and connections) */}
      {activeNodeId !== null && (
        <div className="absolute bottom-2 left-2 z-20 flex items-center gap-2 rounded-lg border border-border/80 bg-card/95 px-2.5 py-1.5 text-[0.68rem] text-foreground backdrop-blur-md shadow-xs">
          <span className="font-mono font-bold text-teal-600 dark:text-teal-400">
            Node #{String(activeNodeId)}
          </span>
          <span className="text-muted-foreground">·</span>
          <span>
            {activeNodeDegree} {activeNodeDegree === 1 ? "connection" : "connections"}
          </span>
          {importantNodes.has(String(activeNodeId)) && (
            <>
              <span className="text-muted-foreground">·</span>
              <span className="rounded bg-teal-500/10 px-1 py-0.2 text-[0.6rem] font-semibold text-teal-700 dark:text-teal-300 border border-teal-500/20">
                Major Hub
              </span>
            </>
          )}
        </div>
      )}

      {/* Visual Legend */}
      <div className="absolute bottom-2 right-2 z-20 flex flex-wrap items-center gap-2.5 rounded-lg border border-border/80 bg-card/95 px-2.5 py-1.5 text-[0.65rem] font-medium text-foreground backdrop-blur-md shadow-xs">
        {hasRouteData ? (
          <>
            <span className="flex items-center gap-1.5">
              <span className="h-2.5 w-2.5 rounded-full bg-[#2563EB]" /> Original Route
            </span>
            <span className="flex items-center gap-1.5">
              <span className="h-2.5 w-2.5 rounded-full bg-[#10B981]" /> Detour Route
            </span>
            {failedEdges.length > 0 && (
              <span className="flex items-center gap-1.5">
                <span className="h-2.5 w-2.5 rounded-full bg-[#DC2626]" /> Blocked Road
              </span>
            )}
          </>
        ) : (
          <>
            <span className="flex items-center gap-1">
              <span className="h-2 w-2 rounded-full bg-[#3B82F6]" /> Detected
            </span>
            <span className="flex items-center gap-1">
              <span className="h-2 w-2 rounded-full bg-[#0D9488]" /> Inferred
            </span>
            <span className="flex items-center gap-1">
              <span className="h-2 w-2 rounded-full bg-[#D97706]" /> Healed
            </span>
            <span className="flex items-center gap-1">
              <span className="h-2 w-2 rounded-full bg-[#A855F7]" /> Occlusion
            </span>
            {criticalEdges.length > 0 && (
              <span className="flex items-center gap-1">
                <span className="h-2 w-2 rounded-full bg-[#EA580C]" /> Critical
              </span>
            )}
          </>
        )}
      </div>
    </div>
  );
}
