"""CPU-only integration test for probability map to healed graph artifacts."""

import json
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from route_resilience.config import GraphConfig
from route_resilience.real_graph_pipeline import (
    graph_from_probability_map,
    graph_to_jsonable,
    predict_probability_map,
    save_artifacts,
)


class _ZeroLogitModel(torch.nn.Module):
    def forward(self, image):
        return torch.zeros((image.shape[0], 1, image.shape[2], image.shape[3]), device=image.device)


def test_probability_inference_preserves_native_dimensions_and_stitches_tiles():
    model = _ZeroLogitModel()
    image_shapes = [(73, 141), (256, 256), (320, 512), (511, 733)]

    for height, width in image_shapes:
        image = np.zeros((height, width, 3), dtype=np.uint8)
        probability = predict_probability_map(model, image, "cpu")
        assert probability.shape == (height, width)
        assert np.isfinite(probability).all()
        assert np.allclose(probability, 0.5)


def _run_script_module():
    script_path = Path(__file__).parents[1] / "scripts" / "run_graph_pipeline.py"
    if not script_path.exists():
        script_path = Path(__file__).parents[2] / "backend" / "scripts" / "run_graph_pipeline.py"
    spec = importlib.util.spec_from_file_location("run_graph_pipeline", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_threshold_artifact_must_match_validation_checkpoint_config(tmp_path):
    module = _run_script_module()
    checkpoint = {"config": {"model": {"arch": "unet_smp"}}}
    artifact = tmp_path / "threshold.json"
    artifact.write_text(json.dumps({
        "threshold": 0.35,
        "selection_split": "validation",
        "config": checkpoint["config"],
    }), encoding="utf-8")

    assert module.resolve_threshold(str(artifact), checkpoint) == 0.35
    assert module.resolve_threshold("0.5", checkpoint) == 0.5

    artifact.write_text(json.dumps({"threshold": 0.35, "selection_split": "test"}), encoding="utf-8")
    with pytest.raises(ValueError, match="selection_split"):
        module.resolve_threshold(str(artifact), checkpoint)


def test_checkpointed_graph_configuration_must_match():
    module = _run_script_module()
    graph_cfg = GraphConfig()
    module.validate_graph_config({"config": {"graph": {
        "max_gap_px": graph_cfg.max_gap_px,
        "min_gap_confidence": graph_cfg.min_gap_confidence,
        "min_edge_length_px": graph_cfg.min_edge_length_px,
    }}}, graph_cfg)

    with pytest.raises(ValueError, match="Graph configuration"):
        module.validate_graph_config({"config": {"graph": {}}}, graph_cfg)


def test_legacy_checkpoint_accepts_only_compatibility_defaults():
    """Legacy checkpoints accept the current production continuity defaults only."""
    module = _run_script_module()
    legacy = {"config": {"graph": {
        "max_gap_px": 15.0,
        "min_gap_confidence": 0.15,
        "min_edge_length_px": 3.0,
    }}}
    module.validate_graph_config(legacy, GraphConfig())

    with pytest.raises(ValueError, match="Graph configuration"):
        module.validate_graph_config(legacy, GraphConfig(enable_occlusion_recovery=False))


def test_probability_map_to_healed_graph_and_artifacts(tmp_path):
    probability = np.full((120, 120), 0.05, dtype=np.float32)
    probability[58:63, 8:112] = 0.92
    probability[55:66, 60:68] = 0.24  # below segmentation threshold, above healing evidence threshold

    result = graph_from_probability_map(probability, threshold=0.5, graph_cfg=GraphConfig(max_gap_px=20))
    assert result["binary_mask"].dtype == bool
    assert result["probability_map"].shape == (120, 120)
    assert result["summary"]["healed_edge_count"] >= 1
    assert "occlusion_recovered_edge_count" in result["summary"]
    assert result["summary"]["after_healing"]["n_components"] < result["summary"]["before_healing"]["n_components"]

    rgb = np.full((120, 120, 3), 128, dtype=np.uint8)
    save_artifacts(result, rgb, tmp_path)
    expected = {
        "probability.npy", "binary_mask.png", "skeleton.png", "summary.json",
        "graph.json", "healed_graph.json", "repaired_network.png", "visualization.png",
    }
    assert expected == {path.name for path in tmp_path.iterdir()}
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    graph = json.loads((tmp_path / "healed_graph.json").read_text(encoding="utf-8"))
    assert summary["coordinate_system"].startswith("pixel-space")
    assert "occlusion_recovered_edge_count" in summary
    assert graph["coordinate_system"] == "pixel-space"
    assert any(edge["attributes"]["healed"] for edge in graph["edges"])


def test_pipeline_summary_counts_occlusion_recovered_edges():
    # Construct two segments with a curved occluded gap
    probability = np.full((100, 100), 0.02, dtype=np.float32)
    # Segment 1
    probability[48:53, 10:35] = 0.95
    # Segment 2
    probability[48:53, 55:80] = 0.95
    # Occlusion straight gap at cols 35:55 on row 50 has 0.02 (below threshold & confidence),
    # but a curved arc bows up to row 56 between cols 34 and 56 with 0.25 (below threshold 0.5,
    # so not in binary mask, but >= min_gap_confidence 0.15).
    for col in range(34, 57):
        row = int(round(50 + 6 * np.sin((col - 34) / 22.0 * np.pi)))
        probability[row - 1:row + 2, col] = 0.25

    result = graph_from_probability_map(
        probability, threshold=0.5, graph_cfg=GraphConfig(max_gap_px=24, min_gap_confidence=0.15)
    )

    assert result["summary"]["healed_edge_count"] >= 1
    assert result["summary"]["occlusion_recovered_edge_count"] >= 1
    assert any(
        edge["attributes"].get("occlusion_recovered") is True
        for edge in graph_to_jsonable(result["graph_after"])["edges"]
    )


def test_graph_extraction_detects_intersections_and_preserves_curved_roads():
    intersection = np.full((128, 160), 0.02, dtype=np.float32)
    intersection[61:67, 8:152] = 0.92
    intersection[8:120, 77:83] = 0.92
    intersection_result = graph_from_probability_map(
        intersection, threshold=0.5, graph_cfg=GraphConfig()
    )
    assert intersection_result["summary"]["graph_quality"]["junction_count"] >= 1
    assert intersection_result["summary"]["after_healing"]["n_components"] == 1

    curved = np.full((128, 160), 0.02, dtype=np.float32)
    road = np.zeros(curved.shape, dtype=bool)
    for col in range(12, 148):
        row = int(round(64 + 28 * np.sin((col - 12) / 135 * np.pi)))
        road[max(0, row - 2):row + 3, col] = True
    curved[road] = 0.92
    curve_result = graph_from_probability_map(
        curved, threshold=0.5, graph_cfg=GraphConfig()
    )
    assert curve_result["summary"]["after_healing"]["n_components"] == 1
    assert curve_result["summary"]["graph_quality"]["endpoint_count"] == 2


def test_extraction_diagnostics_keep_separated_roads_unconnected():
    probability = np.full((128, 160), 0.02, dtype=np.float32)
    probability[18:23, 8:48] = 0.92
    probability[98:103, 112:152] = 0.92

    result = graph_from_probability_map(
        probability, threshold=0.5, graph_cfg=GraphConfig()
    )

    assert result["summary"]["graph_status"] == "fragmented"
    assert result["summary"]["after_healing"]["n_components"] == 2
    assert result["summary"]["graph_quality"]["routable_component_count"] == 2
    assert any("No unsupported links were added" in warning for warning in result["summary"]["warnings"])
    assert result["summary"]["image_dimensions"] == {"height": 128, "width": 160}


def test_mask_cleanup_removes_small_specks_without_changing_connected_roads():
    probability = np.full((96, 128), 0.02, dtype=np.float32)
    probability[47:50, 8:120] = 0.92
    probability[12, 20] = 0.99
    probability[18, 28:30] = 0.99

    result = graph_from_probability_map(
        probability, threshold=0.5, graph_cfg=GraphConfig()
    )

    assert result["summary"]["mask_quality"]["filtered_noise_pixel_count"] == 3
    assert result["summary"]["graph_quality"]["component_sizes"] == [2]
    assert result["summary"]["graph_status"] == "connected"


def test_empty_prediction_reports_non_routable_diagnostics():
    probability = np.full((81, 137), 0.05, dtype=np.float32)

    result = graph_from_probability_map(
        probability, threshold=0.5, graph_cfg=GraphConfig()
    )

    assert result["summary"]["graph_status"] == "insufficient"
    assert result["summary"]["graph_quality"]["routable_component_count"] == 0
    assert len(result["summary"]["warnings"]) >= 2
