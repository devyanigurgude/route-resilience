"""Unit and regression tests for test-set topology and path-length evaluation."""

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from PIL import Image

from route_resilience.evaluate_topology import simplified_apls
from route_resilience.phase3_pipeline import graph_to_jsonable
from scripts.evaluate_test_topology import (
    extract_gt_graph,
    evaluate_single_image,
)


def test_extract_gt_graph_from_valid_mask(tmp_path):
    # Create a simple straight road mask (vertical line)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 50] = 255
    mask_path = tmp_path / "test_mask.png"
    Image.fromarray(mask).save(mask_path)

    G = extract_gt_graph(mask_path, min_edge_length_px=2.0)
    assert G is not None
    assert G.number_of_nodes() >= 2
    assert G.number_of_edges() >= 1


def test_extract_gt_graph_from_empty_mask(tmp_path):
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask_path = tmp_path / "empty_mask.png"
    Image.fromarray(mask).save(mask_path)

    G = extract_gt_graph(mask_path)
    assert G is None


def test_extract_gt_graph_nonexistent_file(tmp_path):
    G = extract_gt_graph(tmp_path / "nonexistent.png")
    assert G is None


def test_simplified_apls_identical_graphs():
    # Graph with 3 connected nodes in a line
    G = nx.Graph()
    G.add_node(0, pos=(10.0, 10.0))
    G.add_node(1, pos=(10.0, 50.0))
    G.add_node(2, pos=(10.0, 90.0))
    G.add_edge(0, 1, length_px=40.0)
    G.add_edge(1, 2, length_px=40.0)

    result = simplified_apls(G_pred=G, G_gt=G, snap_radius=5.0, n_pairs=50, seed=42)
    assert result["score"] == pytest.approx(1.0)
    assert result["miss_rate"] == 0.0
    assert result["n_pairs_evaluated"] > 0


def test_simplified_apls_disconnected_pred_graph():
    # GT is connected 0-1-2
    G_gt = nx.Graph()
    G_gt.add_node(0, pos=(10.0, 10.0))
    G_gt.add_node(1, pos=(10.0, 50.0))
    G_gt.add_node(2, pos=(10.0, 90.0))
    G_gt.add_edge(0, 1, length_px=40.0)
    G_gt.add_edge(1, 2, length_px=40.0)

    # Pred is disconnected (missing edge 1-2)
    G_pred = nx.Graph()
    G_pred.add_node(0, pos=(10.0, 10.0))
    G_pred.add_node(1, pos=(10.0, 50.0))
    G_pred.add_node(2, pos=(10.0, 90.0))
    G_pred.add_edge(0, 1, length_px=40.0)

    result = simplified_apls(G_pred=G_pred, G_gt=G_gt, snap_radius=5.0, n_pairs=50, seed=42)
    assert result["score"] < 1.0
    assert result["miss_rate"] > 0.0

