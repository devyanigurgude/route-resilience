"""Fixed-OD pixel-space service and the Phase III resilience proxy."""

from typing import Dict, Iterable, List

from .routing import evaluate_od_pairs


def service_score(baseline_cost_px, state_route: Dict) -> float:
    """Return 0 when unreachable, else min(1, baseline_cost / state_cost)."""
    if not state_route["reachable"]:
        return 0.0
    state_cost = state_route["cost_px"]
    if baseline_cost_px is None or state_cost is None or state_cost <= 0:
        return 0.0
    return float(min(1.0, baseline_cost_px / state_cost))


def evaluate_state_service(G, baseline_od_pairs: Iterable[Dict], weight: str = "length_px") -> Dict:
    """Evaluate a graph against an unchanged set of baseline-connected OD pairs."""
    pairs = list(baseline_od_pairs)
    routes = evaluate_od_pairs(G, pairs, weight=weight)
    records: List[Dict] = []
    for baseline, state in zip(pairs, routes):
        records.append({
            "origin": baseline["origin"],
            "destination": baseline["destination"],
            "baseline": baseline,
            "state": state,
            "service_score": service_score(baseline["cost_px"], state),
        })
    performance = sum(record["service_score"] for record in records) / len(records) if records else 0.0
    return {"performance": float(performance), "pair_records": records}


def evaluate_resilience(baseline_graph, disrupted_graph, recovered_graph, baseline_od_pairs, weight: str = "length_px") -> Dict:
    """Compute P_D, P_R, RecoveryGain, and RI from the fixed baseline OD set."""
    baseline = evaluate_state_service(baseline_graph, baseline_od_pairs, weight=weight)
    disrupted = evaluate_state_service(disrupted_graph, baseline_od_pairs, weight=weight)
    recovered = evaluate_state_service(recovered_graph, baseline_od_pairs, weight=weight)
    p_disrupted, p_recovered = disrupted["performance"], recovered["performance"]
    combined = []
    for base, disrupted_record, recovered_record in zip(
        baseline["pair_records"], disrupted["pair_records"], recovered["pair_records"]
    ):
        combined.append({
            "origin": base["origin"], "destination": base["destination"],
            "baseline": base["state"], "disrupted": disrupted_record["state"],
            "recovered": recovered_record["state"],
            "baseline_service": base["service_score"],
            "disrupted_service": disrupted_record["service_score"],
            "recovered_service": recovered_record["service_score"],
        })
    return {
        "P_baseline": baseline["performance"],
        "P_D": p_disrupted,
        "P_R": p_recovered,
        "RecoveryGain": float(p_recovered - p_disrupted),
        "RI": float((p_disrupted + p_recovered) / 2.0),
        "od_routes": combined,
    }
