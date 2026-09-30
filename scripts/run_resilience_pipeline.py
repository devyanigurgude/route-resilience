"""Run CPU-only Phase III on exported pixel-space graph JSON artifacts."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from route_resilience.phase3_pipeline import load_graph_json, run_phase3_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run Phase III on pixel-space NetworkX graph JSON; no inference required.")
    parser.add_argument("--graph-healed", required=True, help="healed_graph.json / graph_healed.json")
    parser.add_argument("--graph-extracted", help="graph.json / graph_extracted.json; defaults to healed graph")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scenario-id", default="scenario_001")
    parser.add_argument("--failure-type", choices=["node", "edge"], default="edge")
    parser.add_argument("--selection-method", default="top_edge_betweenness")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--od-pair-limit", type=int, default=200)
    parser.add_argument("--recovery-budget", type=int)
    parser.add_argument(
    "--recovery-ordering",
    choices=["explicit_failure_order", "criticality_greedy"],
    default="explicit_failure_order",
)
    args = parser.parse_args()

    G_healed = load_graph_json(args.graph_healed)
    G_extracted = load_graph_json(args.graph_extracted) if args.graph_extracted else G_healed.copy()
    result = run_phase3_pipeline(
        G_extracted, G_healed, args.output_dir,
        scenario_specs=[{
            "scenario_id": args.scenario_id, "failure_type": args.failure_type,
            "selection_method": args.selection_method, "count": args.count,
            "seed": args.seed,
            "recovery_budget": args.recovery_budget,
            "recovery_ordering": args.recovery_ordering,
        }],
        od_pair_limit=args.od_pair_limit, od_seed=args.seed,
    )
    print(f"Saved Phase III artifacts to {result['output_dir']}")


if __name__ == "__main__":
    main()
