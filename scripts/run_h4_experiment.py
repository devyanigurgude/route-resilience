import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from route_resilience.phase3_pipeline import load_graph_json
from route_resilience.criticality import (
    add_confidence_weight,
    criticality_ranking_stability,
    top_n_edges_by_betweenness,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GRAPH_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "test_batch"
    / "4526"
    / "phase3"
    / "graph"
    / "graph_healed.json"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "h4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    print("=" * 60)
    print("ROUTE RESILIENCE - H4 CONFIDENCE-AWARE CRITICALITY")
    print("=" * 60)

    G = load_graph_json(GRAPH_PATH)

    print(f"\nNodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")

    # --------------------------------------------------
    # 1. CONFIDENCE-BLIND BASELINE
    # --------------------------------------------------

    blind_top = top_n_edges_by_betweenness(
        G,
        n=10,
        weight=None,
    )

    blind_stability = criticality_ranking_stability(
        G,
        n_bootstrap=50,
        top_n=10,
        weight=None,
        confidence_aware=False,
        seed=42,
    )

    # --------------------------------------------------
    # 2. ADD CONFIDENCE WEIGHT
    # --------------------------------------------------

    G_conf = G.copy()

    add_confidence_weight(G_conf)

    # --------------------------------------------------
    # 3. CONFIDENCE-AWARE RANKING
    # --------------------------------------------------

    confidence_top = top_n_edges_by_betweenness(
        G_conf,
        n=10,
        weight="inv_confidence",
    )

    confidence_stability = criticality_ranking_stability(
        G_conf,
        n_bootstrap=50,
        top_n=10,
        weight="inv_confidence",
        confidence_aware=True,
        seed=42,
    )

    # --------------------------------------------------
    # 4. COMPARE TOP-10 RANKINGS
    # --------------------------------------------------

    blind_edges = {edge for edge, _ in blind_top}
    confidence_edges = {edge for edge, _ in confidence_top}

    intersection = blind_edges & confidence_edges
    union = blind_edges | confidence_edges

    ranking_jaccard = (
        len(intersection) / len(union)
        if union
        else 1.0
    )

    result = {
        "graph": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
        },
        "confidence_blind": {
            "weight": None,
            "top_10_edges": [
                {
                    "edge": list(edge),
                    "betweenness": float(score),
                }
                for edge, score in blind_top
            ],
            "stability": blind_stability,
        },
        "confidence_aware": {
            "weight": "inv_confidence",
            "top_10_edges": [
                {
                    "edge": list(edge),
                    "betweenness": float(score),
                }
                for edge, score in confidence_top
            ],
            "stability": confidence_stability,
        },
        "ranking_comparison": {
            "top_10_intersection": len(intersection),
            "top_10_union": len(union),
            "jaccard_overlap": ranking_jaccard,
        },
    }

    output_path = OUTPUT_DIR / "h4_result.json"

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    # --------------------------------------------------
    # 5. PRINT SUMMARY
    # --------------------------------------------------

    print("\n" + "-" * 60)
    print("CONFIDENCE-BLIND")
    print("-" * 60)

    print(
        f"Stability: "
        f"{blind_stability['mean_top_n_overlap']:.4f}"
    )

    print("\n" + "-" * 60)
    print("CONFIDENCE-AWARE")
    print("-" * 60)

    print(
        f"Stability: "
        f"{confidence_stability['mean_top_n_overlap']:.4f}"
    )

    print("\n" + "-" * 60)
    print("RANKING COMPARISON")
    print("-" * 60)

    print(f"Top-10 intersection : {len(intersection)}")
    print(f"Top-10 union        : {len(union)}")
    print(f"Jaccard overlap     : {ranking_jaccard:.4f}")

    print("\nSaved to:")
    print(output_path)


if __name__ == "__main__":
    main()
