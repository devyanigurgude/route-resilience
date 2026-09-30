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
from route_resilience.train import get_device

def run_real_satellite_image_pipeline():
    project_root = Path(".").resolve()
    img_path = project_root / "RouteResilience" / "images_png" / "1.png"
    assert img_path.exists(), f"Image not found at {img_path}"

    print(f"--- [STAGE 1] Loading Real Satellite Image ---")
    pil_image = Image.open(img_path)
    print(f"Input image loaded: {img_path.name}, size: {pil_image.size}, mode: {pil_image.mode}")

    cfg = Config()
    cfg.model.arch = "unet_smp"
    cfg.model.encoder_name = "resnet34"
    cfg.model.encoder_weights = "imagenet"
    cfg.train.loss = "focal_dice"
    device = get_device("cpu")
    checkpoint_path = project_root / "checkpoints" / "best_unet_smp_resnet34_focaldice.pth"

    print(f"--- [STAGE 2] Loading U-Net Model Weights ---")
    t0 = time.time()
    model, checkpoint = load_model_for_inference(
        str(checkpoint_path),
        cfg.model,
        cfg.train.loss,
        device
    )
    print(f"Model loaded successfully in {time.time() - t0:.2f}s")

    print(f"--- [STAGE 3] Running U-Net Road Detection (PIL input directly) ---")
    t0 = time.time()
    # Testing direct PIL Image input to predict_probability_map via preprocess_rgb_image
    prob_map = predict_probability_map(model, pil_image, device)
    print(f"Probability map generated in {time.time() - t0:.2f}s")
    print(f"Prob map shape: {prob_map.shape}, min: {prob_map.min():.4f}, max: {prob_map.max():.4f}, mean: {prob_map.mean():.4f}")

    print(f"--- [STAGE 4] Generating Binary Mask ---")
    threshold = 0.50
    binary_mask = prob_map >= threshold
    road_pixels = int(binary_mask.sum())
    print(f"Threshold: {threshold}, road pixels: {road_pixels} ({road_pixels / binary_mask.size * 100:.2f}%)")

    print(f"--- [STAGE 5 & 6] Skeletonization & Graph Extraction ---")
    t0 = time.time()
    graph_res = graph_from_probability_map(prob_map, threshold=threshold, graph_cfg=cfg.graph)
    print(f"Graph extraction & healing executed in {time.time() - t0:.2f}s")

    G_extracted = graph_res["graph_before"]
    print(f"Extracted Graph (G_extracted): {G_extracted.number_of_nodes()} nodes, {G_extracted.number_of_edges()} edges, {graph_res['summary']['before_healing']['n_components']} components")

    print(f"--- [STAGE 7] Topological Healing ---")
    G_healed = graph_res["graph_after"]
    summary = graph_res["summary"]
    print(f"Healed Graph (G_healed): {G_healed.number_of_nodes()} nodes, {G_healed.number_of_edges()} edges, {summary['after_healing']['n_components']} components")
    print(f"Healed gap edges bridged: {summary['healed_edge_count']}")

    print(f"--- [STAGE 8] Criticality Assessment ---")
    t0 = time.time()
    crit = criticality_report(G_healed, weight="length_px", top_n=5)
    print(f"Criticality assessment completed in {time.time() - t0:.2f}s")
    edge_crits = crit.get("edge_criticality", [])
    node_crits = crit.get("node_criticality", [])
    print(f"Critical edges identified: {len(edge_crits)}")
    for i, edge_info in enumerate(edge_crits[:3], 1):
        print(f"  Top {i} Critical Edge: {edge_info.get('edge_id')} | Length: {edge_info.get('length_px', 0):.1f}px | Betweenness: {edge_info.get('edge_betweenness', 0):.4f}")
    print(f"Critical nodes identified: {len(node_crits)}")
    for i, node_info in enumerate(node_crits[:3], 1):
        print(f"  Top {i} Critical Node: {node_info.get('node_id')} | Betweenness: {node_info.get('betweenness', 0):.4f}")

    print("\nSUCCESS: FULL_CORE_PIPELINE_VERIFIED_ON_REAL_IMAGE")

if __name__ == "__main__":
    run_real_satellite_image_pipeline()
