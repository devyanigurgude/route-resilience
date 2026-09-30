"""CPU-only integration test for probability map to healed graph artifacts."""

import json
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from route_resilience.config import GraphConfig
from route_resilience.real_graph_pipeline import graph_from_probability_map, save_artifacts


def _run_script_module():
    script_path = Path(__file__).parents[1] / "scripts" / "run_graph_pipeline.py"
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


def test_probability_map_to_healed_graph_and_artifacts(tmp_path):
    probability = np.full((120, 120), 0.05, dtype=np.float32)
    probability[58:63, 8:112] = 0.92
    probability[55:66, 60:68] = 0.24  # below segmentation threshold, above healing evidence threshold

    result = graph_from_probability_map(probability, threshold=0.5, graph_cfg=GraphConfig(max_gap_px=20))
    assert result["binary_mask"].dtype == bool
    assert result["probability_map"].shape == (120, 120)
    assert result["summary"]["healed_edge_count"] >= 1
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
    assert graph["coordinate_system"] == "pixel-space"
    assert any(edge["attributes"]["healed"] for edge in graph["edges"])
