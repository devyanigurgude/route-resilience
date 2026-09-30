from pathlib import Path
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from route_resilience.config import Config
from route_resilience.real_graph_pipeline import (
    load_model_for_inference,
    load_rgb_image,
    predict_probability_map,
    graph_from_probability_map,
    save_artifacts,
)
from route_resilience.phase3_pipeline import run_phase3_pipeline
from route_resilience.train import get_device


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLIT_FILE = PROJECT_ROOT / "RouteResilience" / "data_split.csv"
IMAGE_DIR = PROJECT_ROOT / "RouteResilience" / "images_png"
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best_unet_smp_resnet34_focaldice.pth"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "test_batch"


def main():

    print("=" * 60)
    print("ROUTE RESILIENCE - TEST BATCH + PHASE III")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load test split
    # ---------------------------------------------------------

    df = pd.read_csv(SPLIT_FILE)
    test_df = df[df["split"] == "test"].copy()

    print(f"Total test images: {len(test_df)}")

    test_df = test_df

    print(f"Processing: {len(test_df)} images")

    # ---------------------------------------------------------
    # 2. Model configuration
    # ---------------------------------------------------------

    cfg = Config()

    cfg.model.arch = "unet_smp"
    cfg.model.encoder_name = "resnet34"
    cfg.model.encoder_weights = "imagenet"
    cfg.train.loss = "focal_dice"

    device = get_device("cpu")

    print("\nLoading trained model...")
    print(f"Device: {device}")

    model, checkpoint = load_model_for_inference(
        CHECKPOINT,
        cfg.model,
        cfg.train.loss,
        device,
    )

    print("\nModel loaded successfully!")
    print(f"Checkpoint epoch: {checkpoint.get('epoch')}")
    print(f"Best validation IoU: {checkpoint.get('best_val_iou')}")

    # ---------------------------------------------------------
    # 3. Process images
    # ---------------------------------------------------------

    results = []

    for i, row in enumerate(test_df.itertuples(index=False), start=1):

        filename = row.filename
        image_path = IMAGE_DIR / filename

        print("\n" + "-" * 60)
        print(f"[{i}/{len(test_df)}] Processing {filename}")

        if not image_path.exists():
            print(f"ERROR: Image not found: {image_path}")
            continue

        try:

            # -------------------------------------------------
            # Inference
            # -------------------------------------------------

            rgb_image = load_rgb_image(image_path)

            probability_map = predict_probability_map(
                model,
                rgb_image,
                device,
            )

            # -------------------------------------------------
            # Graph extraction + healing
            # -------------------------------------------------

            image_output = OUTPUT_DIR / Path(filename).stem

            result = graph_from_probability_map(
                probability_map,
                0.5,
                cfg.graph,
            )

            save_artifacts(
                result,
                rgb_image,
                image_output,
            )

            summary = result["summary"]

            print("\nGraph generated:")
            print(f"Nodes: {summary['after_healing']['n_nodes']}")
            print(f"Edges: {summary['after_healing']['n_edges']}")
            print(f"Components: {summary['after_healing']['n_components']}")
            print(f"Healed edges: {summary['healed_edge_count']}")

            # -------------------------------------------------
            # Phase III
            # -------------------------------------------------

            phase3_output = image_output / "phase3"

            scenario_specs = [
                {
                    "scenario_id": "targeted_edge_betweenness_top_01",
                    "failure_type": "edge",
                    "selection_method": "top_edge_betweenness",
                    "count": 1,
                    "seed": 42,
                    "recovery_budget": 1,
                }
            ]

            phase3_result = run_phase3_pipeline(
                G_extracted=result["G_extracted"],
                G_healed=result["G_healed"],
                output_dir=phase3_output,
                scenario_specs=scenario_specs,
                od_pair_limit=100,
                od_seed=42,
                weight="length_px",
                top_n=10,
                default_recovery_budget=1,
            )

            # -------------------------------------------------
            # Extract resilience result
            # -------------------------------------------------

            resilience = phase3_result["scenarios"][0]

            print("\nPhase III:")
            print(f"P_baseline: {resilience['P_baseline']:.3f}")
            print(f"P_D:        {resilience['P_D']:.3f}")
            print(f"P_R:        {resilience['P_R']:.3f}")
            print(f"RecoveryGain: {resilience['RecoveryGain']:.3f}")
            print(f"RI:         {resilience['RI']:.3f}")

            # -------------------------------------------------
            # Store final result
            # -------------------------------------------------

            results.append({
                "filename": filename,
                "road_percentage": row.road_percentage,

                "nodes": summary["after_healing"]["n_nodes"],
                "edges": summary["after_healing"]["n_edges"],
                "components": summary["after_healing"]["n_components"],
                "healed_edges": summary["healed_edge_count"],
                "connectivity_ratio":
                    summary["after_healing"]["skeleton_connectivity_ratio"],

                "P_baseline": resilience["P_baseline"],
                "P_D": resilience["P_D"],
                "P_R": resilience["P_R"],
                "RecoveryGain": resilience["RecoveryGain"],
                "RI": resilience["RI"],
            })

        except Exception as e:

            print(f"\nFAILED: {filename}")
            print(f"Error: {e}")

    # ---------------------------------------------------------
    # 4. Save final results
    # ---------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)

    results_file = OUTPUT_DIR / "batch_resilience_results.csv"

    results_df.to_csv(results_file, index=False)

    print("\n" + "=" * 60)
    print("BATCH COMPLETE")
    print("=" * 60)

    print(results_df)

    print(f"\nResults saved to:")
    print(results_file)

    print(f"\nSuccessful: {len(results)}")
    print(f"Failed: {len(test_df) - len(results)}")

    if len(results_df) > 0:
        print(f"\nMean RI: {results_df['RI'].mean():.4f}")


if __name__ == "__main__":
    main()