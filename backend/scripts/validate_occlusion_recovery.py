"""Validation script: Compares Old vs New Occlusion-Aware Gap Recovery on Satellite Tiles."""

import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from route_resilience.config import GraphConfig, ModelConfig
from route_resilience.data import preprocess_rgb_image
from route_resilience.graph_extraction import (
    attach_width,
    clean_road_mask,
    confidence_gated_gap_healing,
    mask_to_skeleton,
    skeleton_to_graph,
)
from route_resilience.real_graph_pipeline import (
    canopy_occlusion_mask,
    cloud_occlusion_mask,
    load_model_for_inference,
    photometric_occlusion_mask,
    predict_probability_map,
    shadow_occlusion_mask,
)


def run_old_gap_healing(graph_before, prob_map, graph_cfg, rgb_image):
    """Simulates pre-change behavior: only cloud mask passed as occlusion_mask, no zero-evidence exception."""
    c_mask = cloud_occlusion_mask(rgb_image) if rgb_image is not None else None
    return confidence_gated_gap_healing(
        graph_before.copy(),
        prob_map,
        max_gap_px=graph_cfg.max_gap_px,
        min_gap_confidence=graph_cfg.min_gap_confidence,
        tangent_dot_threshold=graph_cfg.tangent_dot_threshold,
        continuity_penalty_weight=graph_cfg.continuity_penalty_weight,
        min_component_len=getattr(graph_cfg, "min_component_len", 0.0),
        min_total_component_len=getattr(graph_cfg, "min_total_component_len", 0.0),
        max_sinuosity=getattr(graph_cfg, "max_sinuosity", 1.5),
        min_flank_contrast=getattr(graph_cfg, "min_flank_contrast", None),
        filter_tile_borders=getattr(graph_cfg, "filter_tile_borders", False),
        enable_occlusion_recovery=getattr(graph_cfg, "enable_occlusion_recovery", False),
        max_occlusion_gap_px=getattr(graph_cfg, "max_occlusion_gap_px", 30.0),
        min_occlusion_confidence=getattr(graph_cfg, "min_occlusion_confidence", 0.05),
        occlusion_heading_threshold=getattr(graph_cfg, "occlusion_heading_threshold", 0.75),
        max_lateral_offset_px=getattr(graph_cfg, "max_lateral_offset_px", 4.0),
        occlusion_mask=None,  # Old behavior did not use occlusion_mask for short gaps
        cloud_mask=c_mask,
        enable_cloud_occlusion_recovery=getattr(graph_cfg, "enable_cloud_occlusion_recovery", False),
        max_cloud_occlusion_gap_px=getattr(graph_cfg, "max_cloud_occlusion_gap_px", 240.0),
        cloud_heading_threshold=getattr(graph_cfg, "cloud_heading_threshold", 0.78),
        cloud_min_component_len=getattr(graph_cfg, "cloud_min_component_len", 20.0),
        cloud_min_coverage=getattr(graph_cfg, "cloud_min_coverage", 0.55),
        cloud_max_sinuosity=getattr(graph_cfg, "cloud_max_sinuosity", 1.35),
    )


def run_new_gap_healing(graph_before, prob_map, graph_cfg, rgb_image):
    """New behavior: photometric mask (cloud + shadow + canopy) passed with verified occlusion continuity."""
    c_mask = cloud_occlusion_mask(rgb_image) if rgb_image is not None else None
    photo_mask = photometric_occlusion_mask(rgb_image) if rgb_image is not None else None
    return confidence_gated_gap_healing(
        graph_before.copy(),
        prob_map,
        max_gap_px=graph_cfg.max_gap_px,
        min_gap_confidence=graph_cfg.min_gap_confidence,
        tangent_dot_threshold=graph_cfg.tangent_dot_threshold,
        continuity_penalty_weight=graph_cfg.continuity_penalty_weight,
        min_component_len=getattr(graph_cfg, "min_component_len", 0.0),
        min_total_component_len=getattr(graph_cfg, "min_total_component_len", 0.0),
        max_sinuosity=getattr(graph_cfg, "max_sinuosity", 1.5),
        min_flank_contrast=getattr(graph_cfg, "min_flank_contrast", None),
        filter_tile_borders=getattr(graph_cfg, "filter_tile_borders", False),
        enable_occlusion_recovery=getattr(graph_cfg, "enable_occlusion_recovery", False),
        max_occlusion_gap_px=getattr(graph_cfg, "max_occlusion_gap_px", 30.0),
        min_occlusion_confidence=getattr(graph_cfg, "min_occlusion_confidence", 0.05),
        occlusion_heading_threshold=getattr(graph_cfg, "occlusion_heading_threshold", 0.75),
        max_lateral_offset_px=getattr(graph_cfg, "max_lateral_offset_px", 4.0),
        occlusion_mask=photo_mask,
        cloud_mask=c_mask,
        enable_cloud_occlusion_recovery=getattr(graph_cfg, "enable_cloud_occlusion_recovery", False),
        max_cloud_occlusion_gap_px=getattr(graph_cfg, "max_cloud_occlusion_gap_px", 240.0),
        cloud_heading_threshold=getattr(graph_cfg, "cloud_heading_threshold", 0.78),
        cloud_min_component_len=getattr(graph_cfg, "cloud_min_component_len", 20.0),
        cloud_min_coverage=getattr(graph_cfg, "cloud_min_coverage", 0.55),
        cloud_max_sinuosity=getattr(graph_cfg, "cloud_max_sinuosity", 1.35),
    )


def overlay_graph(ax, G, img_shape, title):
    """Draws graph edges and healed edges over dark background."""
    ax.imshow(np.zeros((*img_shape, 3), dtype=np.uint8))
    for u, v, data in G.edges(data=True):
        pts = data.get("pts")
        if pts is not None and len(pts) >= 2:
            pts = np.asarray(pts)
            if data.get("occlusion_recovered", False):
                ax.plot(pts[:, 1], pts[:, 0], color="magenta", linewidth=2.5, label="Occlusion Healed")
            elif data.get("healed", False):
                ax.plot(pts[:, 1], pts[:, 0], color="cyan", linewidth=1.8, label="Standard Healed")
            else:
                ax.plot(pts[:, 1], pts[:, 0], color="white", linewidth=1.0, alpha=0.8)
        else:
            pu, pv = G.nodes[u]["pos"], G.nodes[v]["pos"]
            color = "magenta" if data.get("occlusion_recovered") else ("cyan" if data.get("healed") else "white")
            ax.plot([pu[1], pv[1]], [pu[0], pv[0]], color=color, linewidth=1.5)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def main():
    root = Path(__file__).resolve().parent.parent.parent
    img_dir = root / "RouteResilience" / "images_png"
    mask_dir = root / "RouteResilience" / "masks_png"
    output_dir = root / "backend" / "outputs" / "occlusion_validation_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and config
    ckpt_path = root / "backend" / "checkpoints" / "best_unet_smp_resnet34_focaldice.pth"
    thresh_path = root / "backend" / "checkpoints" / "threshold_unet_smp_resnet34_focaldice.json"

    with open(thresh_path) as f:
        threshold = float(json.load(f)["threshold"])

    import torch
    from route_resilience.config import Config
    from route_resilience.train import _torch_load_checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_checkpoint = _torch_load_checkpoint(ckpt_path, device)
    saved_model_dict = raw_checkpoint.get("config", {}).get("model", {})
    saved_train_dict = raw_checkpoint.get("config", {}).get("train", {})
    cfg = Config()
    cfg.model.arch = saved_model_dict.get("arch", "unet_smp")
    cfg.model.encoder_name = saved_model_dict.get("encoder_name", "resnet34")
    cfg.model.encoder_weights = "imagenet"
    loss_name = saved_train_dict.get("loss", "focal_dice")
    model, _ = load_model_for_inference(ckpt_path, cfg.model, loss_name, device)
    graph_cfg = cfg.graph

    import pandas as pd
    split_df = pd.read_csv(root / "RouteResilience" / "data_split_grouped.csv")
    test_files = split_df[split_df["split"] == "test"]["filename"].tolist()

    selected_tiles = []
    print(f"Scanning test tiles for road candidates with occlusion (trees/shadows/clouds)...", flush=True)

    # Screen up to 60 test tiles to find 6 with interesting road-occlusion interaction
    for fname in test_files:
        p_img = img_dir / fname
        p_mask = mask_dir / fname
        if not p_img.exists() or not p_mask.exists():
            continue
        gt_mask = np.array(Image.open(p_mask).convert("L")) > 0
        road_px = np.count_nonzero(gt_mask)
        if road_px < 500:  # Need substantial roads
            continue

        rgb = np.array(Image.open(p_img).convert("RGB"))
        p_occ = photometric_occlusion_mask(rgb)
        occ_px = np.count_nonzero(p_occ)
        if occ_px > 1000:  # Has significant canopy, shadows, or clouds
            selected_tiles.append((fname, p_img, p_mask))
            if len(selected_tiles) >= 6:
                break

    print(f"Selected {len(selected_tiles)} test tiles: {[t[0] for t in selected_tiles]}", flush=True)
    results = []

    for fname, p_img, p_mask in selected_tiles:
        rgb = np.array(Image.open(p_img).convert("RGB"))
        gt_mask = np.array(Image.open(p_mask).convert("L")) > 0

        # Predict road probability
        prob_map = predict_probability_map(model, rgb, device)

        # Build initial graph from probability map
        binary_mask = clean_road_mask(prob_map >= threshold, min_component_pixels=int(graph_cfg.min_edge_length_px))
        skeleton = mask_to_skeleton(binary_mask)
        graph_before = skeleton_to_graph(skeleton, prob_map=prob_map, min_edge_length_px=graph_cfg.min_edge_length_px)
        graph_before = attach_width(graph_before, binary_mask)

        # Run Old vs New
        g_old = run_old_gap_healing(graph_before, prob_map, graph_cfg, rgb)
        g_new = run_new_gap_healing(graph_before, prob_map, graph_cfg, rgb)

        old_healed_edges = [e for e in g_old.edges(data=True) if e[2].get("healed", False)]
        new_healed_edges = [e for e in g_new.edges(data=True) if e[2].get("healed", False)]
        new_occ_edges = [e for e in g_new.edges(data=True) if e[2].get("occlusion_recovered", False)]

        # Check alignment of newly recovered edges with ground truth mask
        newly_added = []
        old_edge_keys = {frozenset((u, v)) for u, v, _ in g_old.edges(data=True)}
        for u, v, data in g_new.edges(data=True):
            if frozenset((u, v)) not in old_edge_keys and data.get("healed", False):
                newly_added.append((u, v, data))

        alignments = []
        from scipy.ndimage import distance_transform_edt
        dist_to_gt = distance_transform_edt(~gt_mask)

        for u, v, d in newly_added:
            pts = d.get("pts")
            if pts is not None and len(pts) >= 2:
                pts = np.asarray(pts)
                r = np.clip(np.rint(pts[:, 0]).astype(int), 0, gt_mask.shape[0] - 1)
                c = np.clip(np.rint(pts[:, 1]).astype(int), 0, gt_mask.shape[1] - 1)
                mean_dist = float(np.mean(dist_to_gt[r, c]))
                alignments.append(mean_dist)

        avg_dist = float(np.mean(alignments)) if alignments else 0.0
        plausible = (avg_dist <= 6.0) if alignments else True

        tile_result = {
            "filename": fname,
            "components_before": nx.number_connected_components(graph_before),
            "components_old": nx.number_connected_components(g_old),
            "components_new": nx.number_connected_components(g_new),
            "old_healed_count": len(old_healed_edges),
            "new_healed_count": len(new_healed_edges),
            "new_occlusion_healed": len(new_occ_edges),
            "net_connections_added": len(newly_added),
            "mean_gt_deviation_px": round(avg_dist, 2),
            "plausible": plausible,
        }
        results.append(tile_result)

        # Plot 6-panel side-by-side figure
        fig, axes = plt.subplots(1, 6, figsize=(24, 4.2))
        axes[0].imshow(rgb)
        axes[0].set_title(f"Input: {fname}", fontsize=10)
        axes[0].axis("off")

        axes[1].imshow(gt_mask, cmap="gray")
        axes[1].set_title("Ground Truth Mask", fontsize=10)
        axes[1].axis("off")

        axes[2].imshow(prob_map, cmap="inferno")
        axes[2].set_title(f"U-Net Prob (th={threshold:.2f})", fontsize=10)
        axes[2].axis("off")

        # Color-coded occlusion mask: red=shadow, green=canopy, blue=cloud
        c_m = cloud_occlusion_mask(rgb)
        s_m = shadow_occlusion_mask(rgb)
        k_m = canopy_occlusion_mask(rgb)
        occ_rgb = np.zeros_like(rgb)
        occ_rgb[s_m] = [180, 50, 50]    # Red for shadow
        occ_rgb[k_m] = [50, 180, 50]    # Green for canopy
        occ_rgb[c_m] = [80, 150, 240]   # Blue for cloud
        axes[3].imshow(occ_rgb)
        axes[3].set_title("Occlusion (R:shad, G:can, B:cld)", fontsize=9)
        axes[3].axis("off")

        overlay_graph(axes[4], g_old, rgb.shape[:2], f"Old Graph ({len(old_healed_edges)} healed)")
        overlay_graph(axes[5], g_new, rgb.shape[:2], f"New Graph ({len(new_healed_edges)} healed, {len(new_occ_edges)} occ)")

        plt.tight_layout()
        save_path = output_dir / f"{Path(fname).stem}_validation.png"
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
        print(f"Saved: {save_path.name} | Old: {len(old_healed_edges)} healed -> New: {len(new_healed_edges)} healed (+{len(newly_added)} net new)", flush=True)

    # Print summary
    print("\n" + "=" * 80, flush=True)
    print("VALIDATION SUMMARY: OLD VS NEW OCCLUSION RECOVERY", flush=True)
    print("=" * 80, flush=True)
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False), flush=True)

    summary_json_path = output_dir / "validation_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved summary to {summary_json_path}", flush=True)


if __name__ == "__main__":
    main()
