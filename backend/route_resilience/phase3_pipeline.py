"""CPU-only Phase III orchestration for pixel-space NetworkX road graphs."""

import json
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import networkx as nx
import numpy as np

from .criticality import criticality_report
from .disruption import create_failure_scenario, simulate_removal
from .evaluate_topology import graph_summary
from .recovery import reinstate_baseline_assets
from .resilience import evaluate_resilience
from .routing import select_baseline_od_pairs


def run_phase3_pipeline(
    G_extracted: nx.Graph,
    G_healed: nx.Graph,
    output_dir,
    scenario_specs: Iterable[Dict],
    od_pair_limit: Optional[int] = 200,
    od_seed: int = 42,
    weight: str = "length_px",
    top_n: int = 10,
    default_recovery_budget: Optional[int] = None,
) -> Dict:
    """Run criticality, failure, reinstatement recovery, and resilience artifacts.

    ``G_healed`` is always the fixed Phase III baseline. This runner never
    calls segmentation inference and is fully testable with synthetic graphs.
    """
    output_dir = Path(output_dir)
    directories = {name: output_dir / name for name in ("graph", "criticality", "scenarios", "recovery", "resilience")}
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    _write_json(directories["graph"] / "graph_extracted.json", graph_to_jsonable(G_extracted))
    _write_json(directories["graph"] / "graph_healed.json", graph_to_jsonable(G_healed))
    _write_json(directories["graph"] / "graph_summary.json", {
        "G_extracted": graph_summary(G_extracted), "G_healed": graph_summary(G_healed),
        "coordinate_system": "pixel-space",
    })

    report = criticality_report(G_healed, weight=weight, top_n=top_n)
    _write_json(directories["criticality"] / "criticality.json", report)
    od_pairs = select_baseline_od_pairs(G_healed, max_pairs=od_pair_limit, seed=od_seed, weight=weight)
    _write_json(directories["scenarios"] / "od_pairs.json", {
        "baseline_graph_state": "G_healed", "weight": weight, "selection_seed": od_seed,
        "od_pair_count": len(od_pairs), "pairs": od_pairs,
    })

    outcomes = []
    for spec in scenario_specs:
        scenario = _resolve_scenario(G_healed, spec, weight)
        scenario_id = scenario["scenario_id"]
        _write_json(directories["scenarios"] / f"{scenario_id}.json", scenario)
        disruption = simulate_removal(
            G_healed, edges_to_remove=scenario["failed_edges"], nodes_to_remove=scenario["failed_nodes"], weight=weight
        )
        G_disrupted = disruption.pop("surviving_graph")
        _write_json(directories["scenarios"] / f"{scenario_id}_disrupted_graph.json", graph_to_jsonable(G_disrupted))
        _write_json(directories["scenarios"] / f"{scenario_id}_disruption_metrics.json", disruption)

        budget = spec.get("recovery_budget", default_recovery_budget)
        recovery = reinstate_baseline_assets(
        G_healed,
        G_disrupted,
        scenario["failed_nodes"],
        scenario["failed_edges"],
        budget=budget,
        ordering=spec.get("recovery_ordering", "explicit_failure_order"),
        weight=weight,
        )
        G_recovered = recovery.pop("recovered_graph")
        _write_json(directories["recovery"] / f"{scenario_id}_recovery_policy.json", recovery)
        _write_json(directories["recovery"] / f"{scenario_id}_recovered_graph.json", graph_to_jsonable(G_recovered))
        _write_json(directories["recovery"] / f"{scenario_id}_recovery_metrics.json", {
            "graph_summary": graph_summary(G_recovered), **recovery,
        })

        resilience = evaluate_resilience(G_healed, G_disrupted, G_recovered, od_pairs, weight=weight)
        routes = resilience.pop("od_routes")
        _write_json(directories["resilience"] / f"{scenario_id}_od_routes.json", {"pairs": routes})
        _write_json(directories["resilience"] / f"{scenario_id}_resilience.json", resilience)
        outcomes.append({"scenario_id": scenario_id, **resilience})

    _write_json(directories["resilience"] / "resilience_summary.json", {
        "baseline_graph_state": "G_healed", "weight": weight,
        "formula": "RI = (P_D + P_R) / 2", "scenarios": outcomes,
    })
    return {"criticality": report, "od_pairs": od_pairs, "scenarios": outcomes, "output_dir": output_dir}


def _resolve_scenario(G: nx.Graph, spec: Dict, weight: str) -> Dict:
    if "failed_nodes" in spec or "failed_edges" in spec:
        return {
            "scenario_id": spec["scenario_id"], "baseline_graph_state": "G_healed",
            "failure_type": spec.get("failure_type", "mixed"),
            "selection_method": spec.get("selection_method", "explicit"),
            "selection_metric": spec.get("selection_metric"), "random_seed": spec.get("random_seed"),
            "requested_failure_count": len(spec.get("failed_nodes", [])) + len(spec.get("failed_edges", [])),
            "failed_nodes": list(spec.get("failed_nodes", [])),
            "failed_edges": [tuple(edge) for edge in spec.get("failed_edges", [])], "weight": weight,
        }
    return create_failure_scenario(
        G, scenario_id=spec["scenario_id"], failure_type=spec["failure_type"],
        selection_method=spec["selection_method"], count=spec.get("count", 1),
        seed=spec.get("seed"), weight=weight,
    )


def graph_to_jsonable(graph: nx.Graph) -> Dict[str, Any]:
    return {
        "coordinate_system": "pixel-space",
        "nodes": [{"id": _json_safe(n), "attributes": _json_safe(data)} for n, data in graph.nodes(data=True)],
        "edges": [
            {"source": _json_safe(u), "target": _json_safe(v), "attributes": _json_safe(data)}
            for u, v, data in graph.edges(data=True)
        ],
    }


def graph_from_jsonable(payload: Dict[str, Any]) -> nx.Graph:
    """Load the simple node-link JSON emitted by this module or real graph artifacts."""
    G = nx.Graph()
    for node in payload["nodes"]:
        G.add_node(node["id"], **node.get("attributes", {}))
    for edge in payload["edges"]:
        G.add_edge(edge["source"], edge["target"], **edge.get("attributes", {}))
    return G


def load_graph_json(path) -> nx.Graph:
    with Path(path).open(encoding="utf-8") as handle:
        graph = graph_from_jsonable(json.load(handle))
    confidence_count = sum(
        1 for _, _, data in graph.edges(data=True) if data.get("confidence") is not None
    )
    if graph.number_of_edges() and confidence_count == 0:
        warnings.warn(
            f"Loaded graph {path} has edges but no confidence attributes; "
            "confidence-aware analysis will use no segmentation evidence.",
            stacklevel=2,
        )
    return graph


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


def _write_json(path: Path, payload: Dict):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, allow_nan=False)
