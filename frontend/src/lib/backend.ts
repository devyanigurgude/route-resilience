export const API_BASE = "http://127.0.0.1:8000";
export const SAVED_ANALYSIS_KEY = "route-resilience.analysis-id";
export const SAVED_FAILURE_KEY = "route-resilience.failed-edge";

export type ApiErrorPayload = { detail?: string | Record<string, unknown> };

export type DisruptionResponse = {
  analysis_id: string;
  status: string;
  scenario: {
    strategy: string;
    target_type: string;
    failed_nodes: Array<string | number>;
    failed_edges: Array<[string | number, string | number]>;
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
  route_exists: boolean;
  route: Array<string | number>;
  route_length: number | null;
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

async function request<T>(path: string, init: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, init);
  } catch {
    throw new Error("The FastAPI backend is unavailable. Start it on 127.0.0.1:8000.");
  }

  const payload = (await response.json().catch(() => ({}))) as T | ApiErrorPayload;
  if (!response.ok) {
    const detail = "detail" in payload ? payload.detail : undefined;
    throw new Error(
      typeof detail === "string" ? detail : `Backend request failed (${response.status}).`,
    );
  }
  return payload as T;
}

export function postDisruption(analysisId: string, failedEdge: [number, number]) {
  return request<DisruptionResponse>(`/api/analysis/${encodeURIComponent(analysisId)}/disrupt`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ strategy: "explicit", target_type: "edge", failed_edges: [failedEdge] }),
  });
}

export function postRoute(
  analysisId: string,
  origin: number,
  destination: number,
  failedEdge: [number, number],
) {
  return request<RouteResponse>(`/api/analysis/${encodeURIComponent(analysisId)}/route`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      origin,
      destination,
      current_state: "disrupted",
      failed_edges: [failedEdge],
      failed_nodes: [],
    }),
  });
}

export function postRecovery(analysisId: string, failedEdge: [number, number]) {
  return request<RecoveryResponse>(`/api/analysis/${encodeURIComponent(analysisId)}/recovery`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      failed_edges: [failedEdge],
      failed_nodes: [],
      budget: 1,
      ordering: "explicit_failure_order",
      weight: "length_px",
      od_seed: 42,
      od_max_pairs: 100,
    }),
  });
}

export function readAnalysisId(): string | null {
  return typeof window === "undefined" ? null : window.localStorage.getItem(SAVED_ANALYSIS_KEY);
}

export function readFailedEdge(): [number, number] {
  if (typeof window !== "undefined") {
    const saved = window.localStorage.getItem(SAVED_FAILURE_KEY);
    if (saved) {
      try {
        const edge = JSON.parse(saved) as unknown;
        if (
          Array.isArray(edge) &&
          edge.length === 2 &&
          edge.every((value) => Number.isInteger(value))
        )
          return [edge[0] as number, edge[1] as number];
      } catch {
        // Fall through to the default request values.
      }
    }
  }
  return [0, 1];
}

export function saveFailedEdge(edge: [number, number]) {
  window.localStorage.setItem(SAVED_FAILURE_KEY, JSON.stringify(edge));
}
