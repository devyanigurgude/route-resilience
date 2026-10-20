export const API_BASE =
  (import.meta as { env?: Record<string, string> }).env?.VITE_API_BASE ||
  "http://127.0.0.1:8000";
export const SAVED_ANALYSIS_KEY = "route-resilience.analysis-id";
const SCENARIO_KEY_PREFIX = "route-resilience.scenario.";
const PENDING_FAILURE_KEY_PREFIX = "route-resilience.pending-failure.";
const ROUTE_RESULT_KEY_PREFIX = "route-resilience.route-result.";
const RECOVERY_RESULT_KEY_PREFIX = "route-resilience.recovery-result.";
const REQUEST_TIMEOUT_MS = 15_000;

export type NodeId = string | number;
export type FailedEdge = [NodeId, NodeId];

export type ApiErrorPayload = { detail?: string | Record<string, unknown> };

export type DisruptionResponse = {
  analysis_id: string;
  status: string;
  scenario: {
    strategy: string;
    target_type: string;
    failed_nodes: Array<string | number>;
    failed_edges: FailedEdge[];
  };
  impact_metrics: {
    efficiency_drop_frac: number;
    newly_disconnected_components: number;
    before_efficiency: number;
    after_efficiency: number;
  };
  network_state: {
    baseline: {
      n_nodes: number;
      n_edges: number;
      n_components: number;
      network_efficiency: number;
    };
    disrupted: {
      n_nodes: number;
      n_edges: number;
      n_components: number;
      network_efficiency: number;
    };
  };
  affected_od_service?: {
    total_od_pairs_evaluated: number;
    disrupted_service_performance: number;
    performance_drop: number;
    severed_pairs_count: number;
    detoured_pairs_count: number;
    intact_pairs_count: number;
  } | null;
};

export type RouteResponse = {
  analysis_id: string;
  status: string;
  origin: string | number;
  destination: string | number;
  current_state: string;
  route_status:
    | "original_route_available"
    | "alternative_available"
    | "no_alternative_route"
    | "no_original_route";
  route_reason?: string | null;
  warnings?: string[];
  route_exists: boolean;
  route: Array<string | number>;
  original_route: NodeId[];
  alternative_route: NodeId[];
  blocked_edges: FailedEdge[];
  blocked_nodes?: NodeId[];
  route_length: number | null;
  original_route_length: number | null;
  alternative_route_length: number | null;
  extra_distance: number | null;
  baseline_route_length: number | null;
  disrupted_route_length: number | null;
  detour_ratio: number | null;
  baseline_vs_disrupted: { baseline: number | null; disrupted: number | null };
};

export type RecoveryResponse = {
  analysis_id: string;
  status: string;
  scenario: {
    failed_edges: Array<[string | number, string | number]>;
    budget: number;
    ordering: string;
  };
  disrupted_state: {
    n_nodes: number;
    n_edges: number;
    n_components: number;
    efficiency: number;
    efficiency_drop_frac: number;
  };
  restoration_sequence: Array<{
    step: number;
    restored_asset_type: string;
    restored_asset: string | number | [string | number, string | number];
    efficiency: number;
  }>;
  recovery_summary: {
    recovered_nodes: Array<string | number>;
    recovered_edges: Array<[string | number, string | number]>;
    assets_restored: number;
  };
  final_recovered_state: {
    n_nodes: number;
    n_edges: number;
    n_components: number;
    efficiency: number;
  };
  resilience_metrics: {
    P_baseline: number;
    P_D: number;
    P_R: number;
    RecoveryGain: number;
    RI: number;
  };
};

export type StoredScenario = {
  analysisId: string;
  failedEdges: FailedEdge[];
  failedNodes: NodeId[];
  disruption: DisruptionResponse;
};

export type ContinuityCheck = {
  name: string;
  passed: boolean;
  detail: string;
};

export type RecoveryEvidence = {
  gap_px?: number;
  heading_alignment?: number;
  dot_a?: number;
  dot_b?: number;
  mutual_heading?: number;
  lateral_offset_px?: number;
  probability_evidence?: number;
  supported_fraction?: number;
  flank_contrast?: number | null;
  sinuosity?: number | null;
  cloud_coverage?: number | null;
  acceptance_reason?: string;
  rejection_reason?: string;
  checks?: ContinuityCheck[];
  [key: string]: unknown;
};

export type CandidateOcclusion = {
  source: NodeId;
  target: NodeId;
  pos_source: [number, number];
  pos_target: [number, number];
  gap_px: number;
  heading_alignment: number;
  mutual_heading: number;
  lateral_offset_px: number;
  status: "candidate" | "recovered" | "rejected";
  recovered: boolean;
  rejection_reason?: string;
  cloud_coverage?: number;
  occlusion_kind?: string;
  /** True when the endpoint evidence identifies this gap as a suspected obstruction. */
  suspected_occlusion?: boolean;
  checks?: ContinuityCheck[];
  evidence?: RecoveryEvidence;
  acceptance_reason?: string;
};

export type GraphData = {
  coordinate_system?: string;
  candidate_occlusions?: CandidateOcclusion[];
  nodes: Array<{ id: NodeId; attributes: { pos?: [number, number]; [key: string]: unknown } }>;
  edges: Array<{
    source: NodeId;
    target: NodeId;
    attributes: {
      pts?: Array<[number, number]>;
      confidence?: number;
      healed?: boolean;
      inferred?: boolean;
      occlusion_suspected?: boolean;
      occlusion_recovered?: boolean;
      recovery_type?: string;
      length_px?: number;
      width_px?: number;
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
      [key: string]: unknown;
    };
  }>;
};

function scenarioKey(analysisId: string) {
  return `${SCENARIO_KEY_PREFIX}${analysisId}`;
}

function pendingFailureKey(analysisId: string) {
  return `${PENDING_FAILURE_KEY_PREFIX}${analysisId}`;
}

function routeResultKey(analysisId: string) {
  return `${ROUTE_RESULT_KEY_PREFIX}${analysisId}`;
}

function recoveryResultKey(analysisId: string) {
  return `${RECOVERY_RESULT_KEY_PREFIX}${analysisId}`;
}

async function request<T>(
  path: string,
  init: RequestInit = {},
  url = `${API_BASE}${path}`,
): Promise<T> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  const signal = init.signal
    ? AbortSignal.any([init.signal, controller.signal])
    : controller.signal;
  let response: Response;
  try {
    response = await fetch(url, { ...init, signal });
  } catch (error) {
    if (controller.signal.aborted) throw new Error("Request timed out. Please retry.");
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new Error("The FastAPI backend is unavailable.");
  } finally {
    window.clearTimeout(timeout);
  }

  const payload = (await response.json().catch(() => ({}))) as T | ApiErrorPayload;
  if (!response.ok) {
    const detail =
      typeof payload === "object" && payload !== null && "detail" in payload
        ? (payload as ApiErrorPayload).detail
        : undefined;
    throw new Error(
      typeof detail === "string" ? detail : `Backend request failed (${response.status}).`,
    );
  }
  return payload as T;
}

export type AnalysisItem = {
  analysis_id: string;
  status: string;
  created_at?: string;
  input_image_name?: string;
  input_image_ref?: string;
  artifacts?: {
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
    [key: string]: string | undefined;
  };
  summary?: {
    threshold?: number;
    image_dimensions?: { height: number; width: number };
    before_healing?: {
      n_nodes: number;
      n_edges: number;
      n_components: number;
      total_length_px?: number;
    };
    after_healing?: {
      n_nodes: number;
      n_edges: number;
      n_components: number;
      total_length_px?: number;
    };
    graph_quality?: {
      component_sizes?: number[];
      isolated_node_count?: number;
      endpoint_count?: number;
      junction_count?: number;
      routable_component_count?: number;
    };
    mask_quality?: {
      road_fraction?: number;
      road_pixel_count?: number;
    };
    graph_status?: string;
    warnings?: string[];
    healed_edge_count?: number;
    occlusion_recovered_edge_count?: number;
  };
};

const ANALYSES_HISTORY_KEY = "route-resilience.analyses-history";

export function saveAnalysisToHistory(item: AnalysisItem) {
  if (typeof window === "undefined" || !item.analysis_id) return;
  try {
    const raw = window.localStorage.getItem(ANALYSES_HISTORY_KEY);
    const existing: AnalysisItem[] = raw ? JSON.parse(raw) : [];
    const filtered = existing.filter((a) => a.analysis_id !== item.analysis_id);
    const updated = [item, ...filtered].slice(0, 30);
    window.localStorage.setItem(ANALYSES_HISTORY_KEY, JSON.stringify(updated));
  } catch {
    // Ignore localStorage errors
  }
}

export function getLocalAnalysisHistory(): AnalysisItem[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(ANALYSES_HISTORY_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

export async function getAnalyses(signal?: AbortSignal): Promise<AnalysisItem[]> {
  try {
    const res = await request<{ analyses: AnalysisItem[] }>("/api/analyses", signal ? { signal } : {});
    const serverList = res.analyses || [];
    // Merge server list with local history to ensure any recent uploads are represented
    const localList = getLocalAnalysisHistory();
    const map = new Map<string, AnalysisItem>();
    serverList.forEach((item) => map.set(item.analysis_id, item));
    localList.forEach((item) => {
      if (!map.has(item.analysis_id)) {
        map.set(item.analysis_id, item);
      }
    });
    return Array.from(map.values()).sort((a, b) => {
      const timeA = a.created_at ? new Date(a.created_at).getTime() : 0;
      const timeB = b.created_at ? new Date(b.created_at).getTime() : 0;
      return timeB - timeA;
    });
  } catch {
    return getLocalAnalysisHistory();
  }
}

export function postDisruption(analysisId: string, failedEdge: FailedEdge) {
  return request<DisruptionResponse>(`/api/analysis/${encodeURIComponent(analysisId)}/disrupt`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ strategy: "explicit", target_type: "edge", failed_edges: [failedEdge] }),
  });
}

export function postNodeDisruption(analysisId: string, failedNode: NodeId) {
  return request<DisruptionResponse>(`/api/analysis/${encodeURIComponent(analysisId)}/disrupt`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ strategy: "explicit", target_type: "node", failed_nodes: [failedNode] }),
  });
}

export function postRoute(
  analysisId: string,
  origin?: NodeId | null,
  destination?: NodeId | null,
  failedEdges?: FailedEdge[],
  currentState: "baseline" | "disrupted" = "disrupted",
) {
  const body: Record<string, unknown> = {
    current_state: currentState,
  };
  if (origin !== undefined && origin !== null) body["origin"] = origin;
  if (destination !== undefined && destination !== null) body["destination"] = destination;
  if (failedEdges !== undefined) body["failed_edges"] = failedEdges;
  return request<RouteResponse>(`/api/analysis/${encodeURIComponent(analysisId)}/route`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function postRecovery(
  analysisId: string,
  failedEdges?: FailedEdge[],
  failedNodes?: NodeId[],
) {
  const body: Record<string, unknown> = {
    budget: 1,
    ordering: "explicit_failure_order",
    weight: "length_px",
    od_seed: 42,
    od_max_pairs: 100,
  };
  if (failedEdges !== undefined) body["failed_edges"] = failedEdges;
  if (failedNodes !== undefined) body["failed_nodes"] = failedNodes;
  return request<RecoveryResponse>(`/api/analysis/${encodeURIComponent(analysisId)}/recovery`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function readAnalysisId(): string | null {
  return typeof window === "undefined" ? null : window.localStorage.getItem(SAVED_ANALYSIS_KEY);
}

export function getAnalysis<T>(analysisId: string, signal?: AbortSignal) {
  return request<T>(
    `/api/analysis/${encodeURIComponent(analysisId)}`,
    signal ? { signal } : {},
  );
}

export async function deleteAnalysis(
  analysisId: string,
): Promise<{ status: string; analysis_id: string; message: string }> {
  const result = await request<{ status: string; analysis_id: string; message: string }>(
    `/api/analysis/${encodeURIComponent(analysisId)}`,
    {
      method: "DELETE",
    },
  );
  removeAnalysisFromHistory(analysisId);
  return result;
}

export function removeAnalysisFromHistory(analysisId: string) {
  if (typeof window === "undefined" || !analysisId) return;
  try {
    const raw = window.localStorage.getItem(ANALYSES_HISTORY_KEY);
    if (!raw) return;
    const existing: AnalysisItem[] = JSON.parse(raw);
    const filtered = existing.filter((a) => a.analysis_id !== analysisId);
    window.localStorage.setItem(ANALYSES_HISTORY_KEY, JSON.stringify(filtered));
  } catch {
    // Ignore localStorage errors
  }
}

export function getCriticality<T>(analysisId: string, signal?: AbortSignal) {
  return request<T>(
    `/api/analysis/${encodeURIComponent(analysisId)}/criticality`,
    signal ? { signal } : {},
  );
}

export async function getScenario(analysisId: string, signal?: AbortSignal) {
  const response = await request<{
    analysis_id: string;
    disruption: DisruptionResponse | null;
  }>(`/api/analysis/${encodeURIComponent(analysisId)}/scenario`, signal ? { signal } : {});
  if (response.analysis_id !== analysisId) {
    throw new Error("Disruption scenario does not match this analysis.");
  }
  if (!response.disruption) {
    clearScenario(analysisId);
    return null;
  }
  const scenario: StoredScenario = {
    analysisId,
    failedEdges: response.disruption.scenario.failed_edges,
    failedNodes: response.disruption.scenario.failed_nodes,
    disruption: response.disruption,
  };
  cacheScenario(scenario);
  return scenario;
}

export function getGraph<T>(url: string, signal?: AbortSignal) {
  return request<T>("", signal ? { signal } : {}, url);
}

export function saveScenario(analysisId: string, disruption: DisruptionResponse) {
  if (!disruption.scenario.failed_edges.length && !disruption.scenario.failed_nodes.length) return;
  window.localStorage.removeItem(routeResultKey(analysisId));
  window.localStorage.removeItem(recoveryResultKey(analysisId));
  cacheScenario({
    analysisId,
    failedEdges: disruption.scenario.failed_edges,
    failedNodes: disruption.scenario.failed_nodes,
    disruption,
  });
}

function cacheScenario(scenario: StoredScenario) {
  window.localStorage.setItem(scenarioKey(scenario.analysisId), JSON.stringify(scenario));
}

export function clearScenario(analysisId: string) {
  window.localStorage.removeItem(scenarioKey(analysisId));
}

export function savePendingFailure(analysisId: string, edge: FailedEdge) {
  window.localStorage.setItem(pendingFailureKey(analysisId), JSON.stringify(edge));
}

export function readPendingFailure(analysisId: string): FailedEdge | null {
  try {
    const value = JSON.parse(window.localStorage.getItem(pendingFailureKey(analysisId)) ?? "null");
    return Array.isArray(value) && value.length === 2 ? (value as FailedEdge) : null;
  } catch {
    return null;
  }
}

export function clearPendingFailure(analysisId: string) {
  window.localStorage.removeItem(pendingFailureKey(analysisId));
}

const PENDING_NODE_FAILURE_KEY_PREFIX = "route-resilience.pending-node-failure.";
function pendingNodeFailureKey(analysisId: string) {
  return `${PENDING_NODE_FAILURE_KEY_PREFIX}${analysisId}`;
}

export function savePendingNodeFailure(analysisId: string, node: NodeId) {
  window.localStorage.setItem(pendingNodeFailureKey(analysisId), JSON.stringify(node));
}

export function readPendingNodeFailure(analysisId: string): NodeId | null {
  try {
    const raw = window.localStorage.getItem(pendingNodeFailureKey(analysisId));
    return raw !== null ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

export function clearPendingNodeFailure(analysisId: string) {
  window.localStorage.removeItem(pendingNodeFailureKey(analysisId));
}

export function saveRouteResult(analysisId: string, result: RouteResponse) {
  window.localStorage.setItem(routeResultKey(analysisId), JSON.stringify(result));
}

export function readRouteResult(analysisId: string): RouteResponse | null {
  try {
    const result = JSON.parse(
      window.localStorage.getItem(routeResultKey(analysisId)) ?? "null",
    ) as RouteResponse | null;
    return result?.analysis_id === analysisId ? result : null;
  } catch {
    return null;
  }
}

export function saveRecoveryResult(analysisId: string, result: RecoveryResponse) {
  window.localStorage.setItem(recoveryResultKey(analysisId), JSON.stringify(result));
}

export function readRecoveryResult(analysisId: string): RecoveryResponse | null {
  try {
    const result = JSON.parse(
      window.localStorage.getItem(recoveryResultKey(analysisId)) ?? "null",
    ) as RecoveryResponse | null;
    return result?.analysis_id === analysisId ? result : null;
  } catch {
    return null;
  }
}
