"""Evaluate occlusion-aware recovery vs baseline on synthetic occlusion benchmarks.

Runs both modes on controlled road scenarios and reports:
- Connected component count (lower = better)
- Recovered connections (true positive healed edges)
- False positives (edges incorrectly added to non-roads)
- Precision and recall on known ground truth

Usage:
    python scripts/evaluate_occlusion_recovery.py
"""

import json
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np

# Allow the documented ``python scripts/evaluate_occlusion_recovery.py``
# invocation from a source checkout.  Without this, Python only adds the
# scripts directory to sys.path and cannot import the project package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from route_resilience.graph_extraction import confidence_gated_gap_healing


OUTPUT_DIR = Path("outputs/occlusion_aware_recovery")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Benchmark construction helpers
# ---------------------------------------------------------------------------

def _add_horizontal_segment(G, prefix, y, x_start, n_nodes, step=1.0):
    for i in range(n_nodes):
        x = x_start + i * step
        G.add_node(f"{prefix}{i}", pos=(y, x))
        if i > 0:
            prev_x = x - step
            G.add_edge(
                f"{prefix}{i-1}", f"{prefix}{i}",
                length_px=step,
                pts=np.array([(y, prev_x), (y, x)]),
            )
    return f"{prefix}0", f"{prefix}{n_nodes - 1}"


def _build_occlusion_scenario(gap_px, trunk_len, prob_gap, image_size=(100, 200)):
    """Build a straight-road pair with a synthetic low-confidence gap."""
    H, W = image_size
    G = nx.Graph()
    y = H // 2
    left_start, left_end = _add_horizontal_segment(G, "L", y, 10, trunk_len + 1)
    right_x0 = 10 + trunk_len + gap_px
    right_start, right_end = _add_horizontal_segment(G, "R", y, right_x0, trunk_len + 1)

    prob_map = np.full((H, W), 0.02, dtype=np.float32)
    # Left trunk: high confidence
    prob_map[y - 2:y + 3, 10:10 + trunk_len + 2] = 0.90
    # Gap: low confidence simulating tree canopy / shadow
    gap_start = 10 + trunk_len
    gap_end = gap_start + gap_px
    prob_map[y - 2:y + 3, gap_start:gap_end + 2] = prob_gap
    # Right trunk: high confidence
    prob_map[y - 2:y + 3, int(right_x0):int(right_x0) + trunk_len + 2] = 0.90

    ground_truth_edge = (left_end, right_start)
    return G, prob_map, ground_truth_edge


def run_scenario(scenario_name, G, prob_map, gt_edge, baseline_kwargs, occlusion_kwargs, is_true_connection=True):
    """Run baseline vs occlusion recovery on one scenario, return metrics dict."""
    a, b = gt_edge

    def measure(G_in, kwargs):
        G_copy = G_in.copy()
        t0 = time.perf_counter()
        G_out = confidence_gated_gap_healing(G_copy, prob_map, **kwargs)
        elapsed = time.perf_counter() - t0
        n_components = nx.number_connected_components(G_out)
        largest_component = max((len(c) for c in nx.connected_components(G_out)), default=0)
        healed = G_out.has_edge(a, b)
        occlusion_recovered = healed and bool(G_out[a][b].get("occlusion_recovered"))
        # Count all healed edges
        total_healed = sum(1 for u, v, d in G_out.edges(data=True) if d.get("healed"))
        return {
            "n_components": n_components,
            "largest_component_ratio": round(largest_component / max(1, G_out.number_of_nodes()), 4),
            "ground_truth_healed": healed,
            "occlusion_recovered": occlusion_recovered,
            "total_healed_edges": total_healed,
            "elapsed_ms": round(elapsed * 1000, 2),
        }

    baseline = measure(G, baseline_kwargs)
    improved = measure(G, occlusion_kwargs)

    return {
        "scenario": scenario_name,
        "is_true_connection": is_true_connection,
        "baseline": baseline,
        "improved": improved,
        "improvement": {
            "components_reduced": baseline["n_components"] - improved["n_components"],
            "gt_edge_newly_recovered": (not baseline["ground_truth_healed"]) and improved["ground_truth_healed"],
        }
    }


# ---------------------------------------------------------------------------
# Benchmark suite
# ---------------------------------------------------------------------------

BASELINE_KWARGS = dict(
    max_gap_px=15.0,
    min_gap_confidence=0.15,
    enable_occlusion_recovery=False,
)

OCCLUSION_KWARGS = dict(
    max_gap_px=15.0,
    min_gap_confidence=0.15,
    enable_occlusion_recovery=True,
    max_occlusion_gap_px=30.0,
    min_occlusion_confidence=0.05,
    occlusion_heading_threshold=0.75,
    max_lateral_offset_px=4.0,
)


scenarios = [
    # (name, gap_px, trunk_len, prob_in_gap)
    ("small_gap_high_prob",       10, 15, 0.5),   # Standard healing handles this
    ("medium_gap_medium_prob",    15, 15, 0.12),  # Borderline for standard healing
    ("canopy_gap_20px_p08",       20, 15, 0.08),  # Typical tree canopy occlusion
    ("canopy_gap_25px_p06",       25, 15, 0.06),  # Dense canopy
    ("shadow_gap_18px_p10",       18, 15, 0.10),  # Shadow occlusion
    ("severe_gap_28px_p04",       28, 15, 0.04),  # Extreme occlusion
    ("short_trunk_no_link",        8,  4, 0.05),  # Short trunks → should be rejected
]

scenarios = [
    ("small_gap_high_prob", 10, 15, 0.50, True),
    ("low_confidence_gap_15px", 15, 15, 0.12, True),
    ("canopy_like_gap_20px", 20, 15, 0.08, True),
    ("canopy_like_gap_25px", 25, 15, 0.06, True),
    ("shadow_like_gap_18px", 18, 15, 0.10, True),
    ("severe_gap_28px", 28, 15, 0.04, True),
    # Negative control: a noise-sized segment must not become a road link.
    ("short_trunk_no_link", 8, 4, 0.05, False),
]

results = []
print(f"{'Scenario':<35} {'Baseline CC':>11} {'Improved CC':>11} {'GT Recovered':>14} {'Occ Rec':>10}")
print("-" * 85)

for name, gap, trunk, prob_gap, is_true_connection in scenarios:
    G, prob_map, gt_edge = _build_occlusion_scenario(gap, trunk, prob_gap)
    r = run_scenario(
        name, G, prob_map, gt_edge, BASELINE_KWARGS, OCCLUSION_KWARGS,
        is_true_connection=is_true_connection,
    )
    results.append(r)

    b_cc = r["baseline"]["n_components"]
    i_cc = r["improved"]["n_components"]
    gt_rec = r["improved"]["ground_truth_healed"]
    occ_rec = r["improved"]["occlusion_recovered"]

    print(f"{name:<35} {b_cc:>11} {i_cc:>11} {str(gt_rec):>14} {str(occ_rec):>10}")

print()

def aggregate(mode):
    true_cases = [r for r in results if r["is_true_connection"]]
    recovered = [r for r in results if r[mode]["ground_truth_healed"]]
    true_positive = sum(r["is_true_connection"] for r in recovered)
    false_positive = len(recovered) - true_positive
    false_negative = len(true_cases) - true_positive
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 1.0
    recall = true_positive / len(true_cases) if true_cases else 1.0
    return {
        "precision": round(precision, 4), "recall": round(recall, 4),
        "recovered_connections": len(recovered), "true_positive": true_positive,
        "false_positive": false_positive, "false_negative": false_negative,
        "component_count": sum(r[mode]["n_components"] for r in results),
        "mean_largest_component_ratio": round(
            sum(r[mode]["largest_component_ratio"] for r in results) / len(results), 4
        ),
    }

summary = {"baseline": aggregate("baseline"), "improved": aggregate("improved")}
print("Baseline:", summary["baseline"])
print("Improved:", summary["improved"])

# Save results
out_file = OUTPUT_DIR / "metrics.json"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump({"scenarios": results, "summary": summary}, f, indent=2)
print(f"\nMetrics saved to {out_file}")
