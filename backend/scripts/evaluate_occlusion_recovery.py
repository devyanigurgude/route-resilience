"""Evaluate occlusion-aware recovery vs baseline on synthetic occlusion benchmarks.

Runs both modes on controlled road scenarios and reports:
- Connected component count (lower = better)
- Recovered connections (true positive healed edges)
- False positives (edges incorrectly added to non-roads)
- Observed vs Inferred/Recovered road pixels and length
- Path sinuosity and recovery type
- Multi-orientation (0, 15, 30, 45 deg) and verified occlusion stress cases
- Precision and recall on known ground truth

Usage:
    python backend/scripts/evaluate_occlusion_recovery.py
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from route_resilience.graph_extraction import confidence_gated_gap_healing


# ---------------------------------------------------------------------------
# Benchmark construction helpers
# ---------------------------------------------------------------------------

def _add_oriented_segment(
    G: nx.Graph,
    prefix: str,
    start_pos: Tuple[float, float],
    angle_deg: float,
    n_nodes: int,
    step: float = 1.0,
) -> Tuple[str, str]:
    """Add a straight road segment of n_nodes at a specified orientation."""
    rad = np.radians(angle_deg)
    dr, dc = float(np.sin(rad)), float(np.cos(rad))
    y0, x0 = start_pos
    for i in range(n_nodes):
        pos = (y0 + i * step * dr, x0 + i * step * dc)
        G.add_node(f"{prefix}{i}", pos=pos)
        if i > 0:
            prev = (y0 + (i - 1) * step * dr, x0 + (i - 1) * step * dc)
            G.add_edge(
                f"{prefix}{i-1}", f"{prefix}{i}",
                length_px=step,
                pts=np.array([prev, pos]),
            )
    return f"{prefix}0", f"{prefix}{n_nodes - 1}"


def _build_occlusion_scenario(
    gap_px: float,
    trunk_len: int,
    prob_gap: float,
    angle_deg: float = 0.0,
    with_occlusion_mask: bool = False,
    misaligned_turn_deg: float = 0.0,
    image_size: Tuple[int, int] = (160, 220),
) -> Tuple[nx.Graph, np.ndarray, Tuple[str, str], Optional[np.ndarray]]:
    """Build a straight-road pair with a synthetic gap at arbitrary orientation and occlusion."""
    H, W = image_size
    G = nx.Graph()

    rad = np.radians(angle_deg)
    dr, dc = float(np.sin(rad)), float(np.cos(rad))

    # Center scenario in image
    total_len = 2 * trunk_len + gap_px
    y0 = max(20.0, (H - total_len * dr) / 2.0)
    x0 = max(20.0, (W - total_len * dc) / 2.0)

    left_start, left_end = _add_oriented_segment(G, "L", (y0, x0), angle_deg, trunk_len + 1)

    r_start_y = y0 + trunk_len * dr + gap_px * dr
    r_start_x = x0 + trunk_len * dc + gap_px * dc
    right_angle = angle_deg + misaligned_turn_deg
    right_start, right_end = _add_oriented_segment(
        G, "R", (r_start_y, r_start_x), right_angle, trunk_len + 1
    )

    prob_map = np.full((H, W), 0.02, dtype=np.float32)

    # Left trunk high confidence
    for i in range(trunk_len + 1):
        r, c = int(round(y0 + i * dr)), int(round(x0 + i * dc))
        prob_map[max(0, r - 2):min(H, r + 3), max(0, c - 2):min(W, c + 3)] = 0.90

    # Right trunk high confidence
    r_rad = np.radians(right_angle)
    r_dr, r_dc = float(np.sin(r_rad)), float(np.cos(r_rad))
    for i in range(trunk_len + 1):
        rr, rc = int(round(r_start_y + i * r_dr)), int(round(r_start_x + i * r_dc))
        prob_map[max(0, rr - 2):min(H, rr + 3), max(0, rc - 2):min(W, rc + 3)] = 0.90

    # Gap corridor road evidence (if any)
    if prob_gap > 0.02:
        for t in np.linspace(0, 1, max(4, int(gap_px * 2))):
            gr = int(round(y0 + trunk_len * dr + t * gap_px * dr))
            gc = int(round(x0 + trunk_len * dc + t * gap_px * dc))
            prob_map[max(0, gr - 2):min(H, gr + 3), max(0, gc - 2):min(W, gc + 3)] = prob_gap

    # Occlusion mask covering gap
    occ_mask = None
    if with_occlusion_mask:
        occ_mask = np.zeros((H, W), dtype=bool)
        for t in np.linspace(0, 1, max(4, int(gap_px * 2))):
            gr = int(round(y0 + trunk_len * dr + t * gap_px * dr))
            gc = int(round(x0 + trunk_len * dc + t * gap_px * dc))
            occ_mask[max(0, gr - 6):min(H, gr + 7), max(0, gc - 6):min(W, gc + 7)] = True

    ground_truth_edge = (left_end, right_start)
    return G, prob_map, ground_truth_edge, occ_mask


def run_scenario(
    scenario_name: str,
    G: nx.Graph,
    prob_map: np.ndarray,
    gt_edge: Tuple[str, str],
    baseline_kwargs: Dict[str, Any],
    occlusion_kwargs: Dict[str, Any],
    occ_mask: Optional[np.ndarray] = None,
    is_true_connection: bool = True,
    angle_deg: float = 0.0,
    gap_px: float = 0.0,
) -> Dict[str, Any]:
    """Run baseline vs occlusion recovery on one scenario, distinguishing observed from inferred road."""
    a, b = gt_edge

    def measure(G_in: nx.Graph, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        G_copy = G_in.copy()
        run_kwargs = dict(kwargs)
        if occ_mask is not None and run_kwargs.get("enable_occlusion_recovery"):
            run_kwargs["occlusion_mask"] = occ_mask

        t0 = time.perf_counter()
        G_out = confidence_gated_gap_healing(G_copy, prob_map, **run_kwargs)
        elapsed = time.perf_counter() - t0

        n_components = nx.number_connected_components(G_out)
        largest_component = max((len(c) for c in nx.connected_components(G_out)), default=0)
        healed = G_out.has_edge(a, b)
        edge_data = G_out[a][b] if healed else {}
        occlusion_recovered = healed and bool(edge_data.get("occlusion_recovered"))
        recovery_type = edge_data.get("recovery_type") if healed else None
        sinuosity = round(float(edge_data.get("sinuosity", 1.0)), 4) if healed else None

        # Distinguish directly observed vs inferred road elements
        observed_edges = sum(1 for _, _, d in G_out.edges(data=True) if not d.get("healed", False))
        inferred_edges = sum(1 for _, _, d in G_out.edges(data=True) if d.get("healed", False))
        observed_road_length_px = round(
            sum(float(d.get("length_px", 0.0)) for _, _, d in G_out.edges(data=True) if not d.get("healed", False)), 2
        )
        inferred_road_length_px = round(
            sum(float(d.get("length_px", 0.0)) for _, _, d in G_out.edges(data=True) if d.get("healed", False)), 2
        )

        return {
            "n_components": n_components,
            "largest_component_ratio": round(largest_component / max(1, G_out.number_of_nodes()), 4),
            "ground_truth_healed": healed,
            "occlusion_recovered": occlusion_recovered,
            "recovery_type": recovery_type,
            "sinuosity": sinuosity,
            "total_healed_edges": inferred_edges,
            "observed_edges": observed_edges,
            "inferred_edges": inferred_edges,
            "observed_road_length_px": observed_road_length_px,
            "inferred_road_length_px": inferred_road_length_px,
            "elapsed_ms": round(elapsed * 1000, 2),
        }

    baseline = measure(G, baseline_kwargs)
    improved = measure(G, occlusion_kwargs)

    return {
        "scenario": scenario_name,
        "angle_deg": angle_deg,
        "gap_px": gap_px,
        "is_true_connection": is_true_connection,
        "baseline": baseline,
        "improved": improved,
        "improvement": {
            "components_reduced": baseline["n_components"] - improved["n_components"],
            "gt_edge_newly_recovered": (not baseline["ground_truth_healed"]) and improved["ground_truth_healed"],
        },
    }


# ---------------------------------------------------------------------------
# Benchmark Suite Definition
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


def get_default_scenarios() -> List[Dict[str, Any]]:
    return [
        # Original horizontal scenarios (angle=0)
        {"name": "small_gap_high_prob", "gap": 10.0, "trunk": 15, "prob": 0.50, "angle": 0.0, "with_mask": False, "turn": 0.0, "is_true": True},
        {"name": "low_confidence_gap_15px", "gap": 15.0, "trunk": 15, "prob": 0.12, "angle": 0.0, "with_mask": False, "turn": 0.0, "is_true": True},
        {"name": "canopy_like_gap_20px", "gap": 20.0, "trunk": 15, "prob": 0.08, "angle": 0.0, "with_mask": False, "turn": 0.0, "is_true": True},
        {"name": "canopy_like_gap_25px", "gap": 25.0, "trunk": 15, "prob": 0.06, "angle": 0.0, "with_mask": False, "turn": 0.0, "is_true": True},
        {"name": "shadow_like_gap_18px", "gap": 18.0, "trunk": 15, "prob": 0.10, "angle": 0.0, "with_mask": False, "turn": 0.0, "is_true": True},
        {"name": "severe_gap_28px", "gap": 28.0, "trunk": 15, "prob": 0.04, "angle": 0.0, "with_mask": False, "turn": 0.0, "is_true": True},
        {"name": "short_trunk_no_link", "gap": 8.0, "trunk": 4, "prob": 0.05, "angle": 0.0, "with_mask": False, "turn": 0.0, "is_true": False},

        # Extended multi-orientation and verified occlusion cases (Priority 1 Regression Matrix)
        {
            "name": "diagonal_30deg_gap22px_zero_ev_with_mask",
            "gap": 22.0, "trunk": 15, "prob": 0.00, "angle": 30.0, "with_mask": True, "turn": 0.0, "is_true": True,
        },
        {
            "name": "diagonal_30deg_gap22px_zero_ev_no_mask",
            "gap": 22.0, "trunk": 15, "prob": 0.00, "angle": 30.0, "with_mask": False, "turn": 0.0, "is_true": False,
        },
        {
            "name": "diagonal_45deg_gap20px_canopy_mask",
            "gap": 20.0, "trunk": 15, "prob": 0.01, "angle": 45.0, "with_mask": True, "turn": 0.0, "is_true": True,
        },
        {
            "name": "diagonal_15deg_gap25px_shadow_mask",
            "gap": 25.0, "trunk": 15, "prob": 0.01, "angle": 15.0, "with_mask": True, "turn": 0.0, "is_true": True,
        },
        {
            "name": "diagonal_30deg_gap35px_exceeds_limit",
            "gap": 35.0, "trunk": 15, "prob": 0.00, "angle": 30.0, "with_mask": True, "turn": 0.0, "is_true": False,
        },
        {
            "name": "misaligned_turn_45deg_zero_ev",
            "gap": 22.0, "trunk": 15, "prob": 0.00, "angle": 30.0, "with_mask": True, "turn": 45.0, "is_true": False,
        },
    ]


def run_benchmark_suite(
    output_dir: Optional[Path] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    if output_dir is None:
        output_dir = WORKSPACE_ROOT / "outputs" / "occlusion_recovery_benchmark"
    output_dir = Path(output_dir)

    out_file = output_dir / "extended_metrics.json"
    if out_file.exists() and not overwrite:
        raise FileExistsError(
            f"Benchmark output {out_file} already exists. "
            "Pass --overwrite or choose a different --output-dir to avoid data loss."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = get_default_scenarios()
    results = []

    print("=" * 115)
    print(f"{'Scenario':<42} {'Angle':>5} {'Gap':>5} | {'Base CC':>7} {'Imp CC':>7} {'GT Healed':>10} {'Occ Rec':>8} {'Sinuosity':>10}")
    print("-" * 115)

    for sc in scenarios:
        G, prob_map, gt_edge, occ_mask = _build_occlusion_scenario(
            gap_px=sc["gap"],
            trunk_len=sc["trunk"],
            prob_gap=sc["prob"],
            angle_deg=sc["angle"],
            with_occlusion_mask=sc["with_mask"],
            misaligned_turn_deg=sc["turn"],
        )
        r = run_scenario(
            scenario_name=sc["name"],
            G=G,
            prob_map=prob_map,
            gt_edge=gt_edge,
            baseline_kwargs=BASELINE_KWARGS,
            occlusion_kwargs=OCCLUSION_KWARGS,
            occ_mask=occ_mask,
            is_true_connection=sc["is_true"],
            angle_deg=sc["angle"],
            gap_px=sc["gap"],
        )
        results.append(r)

        b_cc = r["baseline"]["n_components"]
        i_cc = r["improved"]["n_components"]
        gt_rec = r["improved"]["ground_truth_healed"]
        occ_rec = r["improved"]["occlusion_recovered"]
        sin_val = r["improved"]["sinuosity"]
        sin_str = f"{sin_val:.4f}" if sin_val is not None else "N/A"

        print(f"{sc['name']:<42} {sc['angle']:>5.0f} {sc['gap']:>5.1f} | {b_cc:>7} {i_cc:>7} {str(gt_rec):>10} {str(occ_rec):>8} {sin_str:>10}")

    print("=" * 115)

    def aggregate(mode: str) -> Dict[str, Any]:
        true_cases = [r for r in results if r["is_true_connection"]]
        recovered = [r for r in results if r[mode]["ground_truth_healed"]]
        true_positive = sum(r["is_true_connection"] for r in recovered)
        false_positive = len(recovered) - true_positive
        false_negative = len(true_cases) - true_positive
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 1.0
        recall = true_positive / len(true_cases) if true_cases else 1.0
        total_observed_len = sum(r[mode]["observed_road_length_px"] for r in results)
        total_inferred_len = sum(r[mode]["inferred_road_length_px"] for r in results)

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "recovered_connections": len(recovered),
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "component_count": sum(r[mode]["n_components"] for r in results),
            "mean_largest_component_ratio": round(
                sum(r[mode]["largest_component_ratio"] for r in results) / len(results), 4
            ),
            "total_observed_road_length_px": round(total_observed_len, 2),
            "total_inferred_road_length_px": round(total_inferred_len, 2),
        }

    summary = {"baseline": aggregate("baseline"), "improved": aggregate("improved")}
    print("\nSummary Metrics:")
    print("  Baseline:", summary["baseline"])
    print("  Improved:", summary["improved"])

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"scenarios": results, "summary": summary}, f, indent=2)

    print(f"\nExtended metrics saved safely to: {out_file}")
    return {"scenarios": results, "summary": summary}


def main():
    parser = argparse.ArgumentParser(description="Evaluate occlusion-aware recovery vs baseline on synthetic benchmarks.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (defaults safely to outputs/occlusion_recovery_benchmark without overwriting historical runs)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing outputs in output-dir")
    args = parser.parse_args()

    run_benchmark_suite(output_dir=args.output_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
