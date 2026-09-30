import time
from pathlib import Path
from PIL import Image
import numpy as np
import networkx as nx

from route_resilience.config import Config
from route_resilience.real_graph_pipeline import (
    load_model_for_inference,
    predict_probability_map,
    graph_from_probability_map,
    save_artifacts
)
from route_resilience.criticality import criticality_report
from route_resilience.phase3_pipeline import run_phase3_pipeline
from route_resilience.train import get_device

def audit_real_image():
    project_root = Path(".").resolve()
    img_path = project_root / "RouteResilience" / "images_png" / "1.png"
    if not img_path.exists():
        print(f"Sample image not found at {img_path}")
        return

    checkpoint_path = project_root / "checkpoints" / "best_unet_smp_resnet34_focaldice.pth"
    print(f"[1] Loading sample image: {img_path}")
    raw_img = Image.open(img_path).convert("RGB")
    raw_rgb_np = np.asarray(raw_img, dtype=np.uint8)
    print(f"    Image size: {raw_img.size}, array shape: {raw_rgb_np.shape}, mode: {raw_img.mode}")

    cfg = Config()
    cfg.model.arch = "unet_smp"
    cfg.model.encoder_name = "resnet34"
    cfg.model.encoder_weights = "imagenet"
    cfg.train.loss = "focal_dice"
    device = get_device("cpu")

    print("[2] Loading model weights from checkpoint...")
    t0 = time.time()
    model, checkpoint = load_model_for_inference(
        str(checkpoint_path),
        cfg.model,
        cfg.train.loss,
        device
    )
    print(f"    Model loaded in {time.time() - t0:.2f}s")

    print("[3] Running UNet road detection inference...")
    t0 = time.time()
    prob_map = predict_probability_map(model, raw_rgb_np, device)
    print(f"    Inference completed in {time.time() - t0:.2f}s")
    print(f"    Prob map shape: {prob_map.shape}, min: {prob_map.min():.4f}, max: {prob_map.max():.4f}, mean: {prob_map.mean():.4f}")

    threshold = 0.50
    binary_mask = prob_map >= threshold
    road_pixel_count = int(binary_mask.sum())
    total_pixels = binary_mask.size
    print(f"    Road pixels at threshold {threshold}: {road_pixel_count} / {total_pixels} ({road_pixel_count/total_pixels*100:.2f}%)")

    print("[4] Running Skeletonization, Graph Extraction & Topological Healing...")
    t0 = time.time()
    pipeline_res = graph_from_probability_map(prob_map, threshold=threshold, graph_cfg=cfg.graph)
    print(f"    Graph extraction finished in {time.time() - t0:.2f}s")
    
    G_raw = pipeline_res["graph_before"]
    G_healed = pipeline_res["graph_after"]
    summary = pipeline_res["summary"]
    print(f"    G_extracted nodes: {G_raw.number_of_nodes()}, edges: {G_raw.number_of_edges()}, components: {summary['before_healing']['n_components']}")
    print(f"    G_healed nodes: {G_healed.number_of_nodes()}, edges: {G_healed.number_of_edges()}, components: {summary['after_healing']['n_components']}")
    print(f"    Healed edges bridged: {summary['healed_edge_count']}")

    print("[5] Running Criticality Assessment...")
    t0 = time.time()
    crit = criticality_report(G_healed, weight="length_px", top_n=5)
    print(f"    Criticality finished in {time.time() - t0:.2f}s")
    print(f"    Top bottleneck edges: {len(crit.get('edge_criticality', []))}")
    print(f"    Top bottleneck nodes: {len(crit.get('node_criticality', []))}")
    print(f"    Articulation points count: {len(crit.get('topology', {}).get('articulation_points', []))}")
    print(f"    Bridge edges count: {len(crit.get('topology', {}).get('bridges', []))}")

    print("[6] Running Disruption & Recovery (Phase 3 Pipeline)...")
    out_dir = project_root / "outputs" / "audit_test_run"
    scenarios = [{"scenario_id": "test_random_1", "type": "random_failure", "count": 1, "seed": 42}]
    p3_res = run_phase3_pipeline(
        G_extracted=G_raw,
        G_healed=G_healed,
        output_dir=out_dir,
        scenario_specs=scenarios,
        od_pair_limit=50
    )
    print(f"    Phase 3 run succeeded! Scenarios evaluated: {len(p3_res.get('outcomes', []))}")
    outcome = p3_res['outcomes'][0]
    print(f"    Outcome scenario_id: {outcome.get('scenario_id')}, resilience_index: {outcome.get('resilience_index')}")
    print("ALL_PIPELINE_STAGES_AUDITED_SUCCESSFULLY")

if __name__ == "__main__":
    audit_real_image()
