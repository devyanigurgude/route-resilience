from pathlib import Path
import json
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from route_resilience.phase3_pipeline import load_graph_json
from route_resilience.criticality import (
    add_confidence_weight,
    criticality_ranking_stability,
)
    

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = PROJECT_ROOT / "outputs" / "test_batch"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "h4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Evaluate all usable graphs by default.
MAX_IMAGES = None

N_BOOTSTRAP = 50
TOP_N = 10
SEED = 42


def run_condition(G, ranking_weight, confidence_aware):
    return criticality_ranking_stability(
        G,
        n_bootstrap=N_BOOTSTRAP,
        top_n=TOP_N,
        weight=ranking_weight,
        confidence_aware=confidence_aware,
        seed=SEED,
    )


def main():

    print("=" * 70)
    print("ROUTE RESILIENCE - FULL H4 2x2 EXPERIMENT")
    print("=" * 70)

    # ---------------------------------------------------------
    # Find graph artifacts
    # ---------------------------------------------------------

    graph_paths = sorted(
        TEST_DIR.glob("*/phase3/graph/graph_healed.json")
    )

    print(f"\nAvailable graph files: {len(graph_paths)}")

    # Only use graphs with at least 2 edges.
    usable = []

    for path in graph_paths:
        try:
            G = load_graph_json(path)

            if G.number_of_nodes() >= 2 and G.number_of_edges() >= 2:
                usable.append(path)

        except Exception:
            pass

    print(f"Usable graphs: {len(usable)}")

    if MAX_IMAGES is not None:
        usable = usable[:MAX_IMAGES]

    print(f"Graphs selected for H4: {len(usable)}")

    results = []

    # ---------------------------------------------------------
    # Process graphs
    # ---------------------------------------------------------

    for index, graph_path in enumerate(usable, start=1):

        filename = graph_path.parents[2].name

        print(
            f"\n[{index}/{len(usable)}] {filename}"
        )

        try:

            G = load_graph_json(graph_path)

            # Add confidence-derived edge weight.
            G_conf = G.copy()
            add_confidence_weight(G_conf)

            # -------------------------------------------------
            # CONDITION 1
            # Blind ranking + uniform perturbation
            # -------------------------------------------------

            blind_uniform = run_condition(
                G,
                ranking_weight=None,
                confidence_aware=False,
            )

            # -------------------------------------------------
            # CONDITION 2
            # Blind ranking + confidence perturbation
            # -------------------------------------------------

            blind_confidence = run_condition(
                G,
                ranking_weight=None,
                confidence_aware=True,
            )

            # -------------------------------------------------
            # CONDITION 3
            # Confidence-aware ranking + uniform perturbation
            # -------------------------------------------------

            aware_uniform = run_condition(
                G_conf,
                ranking_weight="inv_confidence",
                confidence_aware=False,
            )

            # -------------------------------------------------
            # CONDITION 4
            # Confidence-aware ranking + confidence perturbation
            # -------------------------------------------------

            aware_confidence = run_condition(
                G_conf,
                ranking_weight="inv_confidence",
                confidence_aware=True,
            )

            row = {
                "filename": filename,
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),

                "blind_uniform": blind_uniform[
                    "mean_top_n_overlap"
                ],

                "blind_confidence": blind_confidence[
                    "mean_top_n_overlap"
                ],

                "aware_uniform": aware_uniform[
                    "mean_top_n_overlap"
                ],

                "aware_confidence": aware_confidence[
                    "mean_top_n_overlap"
                ],
            }

            results.append(row)

            print(
                f"  Blind + Uniform     : "
                f"{row['blind_uniform']:.4f}"
            )

            print(
                f"  Blind + Confidence  : "
                f"{row['blind_confidence']:.4f}"
            )

            print(
                f"  Aware + Uniform     : "
                f"{row['aware_uniform']:.4f}"
            )

            print(
                f"  Aware + Confidence  : "
                f"{row['aware_confidence']:.4f}"
            )

        except Exception as e:

            print(f"  FAILED: {filename} -> {e}")

    # ---------------------------------------------------------
    # Save per-graph results
    # ---------------------------------------------------------

    results_df = pd.DataFrame(results)

    csv_path = OUTPUT_DIR / "h4_results.csv"

    results_df.to_csv(
        csv_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Aggregate results
    # ---------------------------------------------------------

    summary = {
        "graphs_evaluated": len(results_df),
        "n_bootstrap": N_BOOTSTRAP,
        "top_n": TOP_N,
        "seed": SEED,

        "mean_stability": {
            "blind_uniform": float(
                results_df["blind_uniform"].mean()
            ),

            "blind_confidence": float(
                results_df["blind_confidence"].mean()
            ),

            "aware_uniform": float(
                results_df["aware_uniform"].mean()
            ),

            "aware_confidence": float(
                results_df["aware_confidence"].mean()
            ),
        },
    }

    # ---------------------------------------------------------
    # Calculate isolated effects
    # ---------------------------------------------------------

    summary["effects"] = {

        # Effect of confidence-aware perturbation
        # while keeping ranking blind.
        "perturbation_effect_blind": (
            summary["mean_stability"]["blind_confidence"]
            - summary["mean_stability"]["blind_uniform"]
        ),

        # Effect of confidence-aware ranking
        # under uniform perturbation.
        "ranking_effect_uniform": (
            summary["mean_stability"]["aware_uniform"]
            - summary["mean_stability"]["blind_uniform"]
        ),

        # Effect of confidence-aware ranking
        # under confidence-based perturbation.
        "ranking_effect_confidence": (
            summary["mean_stability"]["aware_confidence"]
            - summary["mean_stability"]["blind_confidence"]
        ),

        # Full proposed pipeline vs blind baseline.
        "full_effect": (
            summary["mean_stability"]["aware_confidence"]
            - summary["mean_stability"]["blind_uniform"]
        ),
    }

    json_path = OUTPUT_DIR / "h4_summary.json"

    with open(json_path, "w") as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    # ---------------------------------------------------------
    # Final report
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("H4 FINAL SUMMARY")
    print("=" * 70)

    means = summary["mean_stability"]

    print(
        f"\nBlind + Uniform perturbation     : "
        f"{means['blind_uniform']:.4f}"
    )

    print(
        f"Blind + Confidence perturbation : "
        f"{means['blind_confidence']:.4f}"
    )

    print(
        f"Aware + Uniform perturbation    : "
        f"{means['aware_uniform']:.4f}"
    )

    print(
        f"Aware + Confidence perturbation: "
        f"{means['aware_confidence']:.4f}"
    )

    print("\n" + "-" * 70)
    print("ISOLATED EFFECTS")
    print("-" * 70)

    effects = summary["effects"]

    print(
        f"\nPerturbation effect (blind) : "
        f"{effects['perturbation_effect_blind']:+.4f}"
    )

    print(
        f"Ranking effect (uniform)   : "
        f"{effects['ranking_effect_uniform']:+.4f}"
    )

    print(
        f"Ranking effect (confidence): "
        f"{effects['ranking_effect_confidence']:+.4f}"
    )

    print(
        f"Full effect                : "
        f"{effects['full_effect']:+.4f}"
    )

    print("\nSaved:")
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()