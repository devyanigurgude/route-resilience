"""
Raster mask -> confidence-aware NetworkX road graph.

This is the module where fabricated topology most easily enters the
pipeline (Phase 0, Part 1's central risk), so every function here is
deliberately conservative:
  - gap healing is confidence-gated (uses the probability map as
    evidence), never pure nearest-neighbour geometry
  - healed edges are tagged `healed=True` so they can be filtered out,
    down-weighted, or audited separately from directly-observed edges
  - width is estimated from a distance transform so criticality
    analysis has something better than unweighted topology to use

Grade separation (flyovers/underpasses) is NOT handled in this module
-- it needs a DEM or OSM bridge/tag layer, which is out of scope for a
raster-only reconstruction. `resolve_grade_separation` is stubbed
below with a clear TODO rather than silently doing nothing.
"""

from typing import Optional

import networkx as nx
import numpy as np
import sknw
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize


def mask_to_skeleton(binary_mask: np.ndarray) -> np.ndarray:
    """binary_mask: 2D bool/uint8 array, True/1 = predicted road pixel."""
    return skeletonize(binary_mask.astype(bool))


def _sample_along_path(image: np.ndarray, pts: np.ndarray) -> float:
    """Mean value of `image` at the integer pixel coordinates in `pts` (row, col)."""
    rows = np.clip(pts[:, 0].astype(int), 0, image.shape[0] - 1)
    cols = np.clip(pts[:, 1].astype(int), 0, image.shape[1] - 1)
    return float(image[rows, cols].mean())


def skeleton_to_graph(
    skeleton: np.ndarray,
    prob_map: Optional[np.ndarray] = None,
    min_edge_length_px: float = 3.0,
) -> nx.Graph:
    """
    Converts a skeleton into a NetworkX graph using sknw, then attaches:
      - node 'pos': (row, col) pixel coordinate
      - edge 'length_px': pixel path length
      - edge 'confidence': mean segmentation probability along the edge
        path (requires prob_map; falls back to 1.0 if not provided --
        meaning "not confidence-aware", make sure that's intentional)
      - edge 'width_px': estimated road width from a distance transform
        of the binary mask implied by the skeleton's source

    Short spurs (degree-1 dead ends shorter than `min_edge_length_px`)
    are pruned -- these are almost always skeletonization artifacts,
    not real short roads.
    """
    if skeleton.sum() == 0:
        return nx.Graph()

    raw = sknw.build_sknw(skeleton, multi=False)

    G = nx.Graph()
    for n, data in raw.nodes(data=True):
        o = np.asarray(data["o"])
        G.add_node(n, pos=(float(o[0]), float(o[1])))

    for u, v, data in raw.edges(data=True):
        pts = np.asarray(data["pts"])
        length_px = float(data.get("weight", len(pts)))
        confidence = _sample_along_path(prob_map, pts) if prob_map is not None else 1.0
        G.add_edge(
            u, v,
            length_px=length_px,
            confidence=confidence,
            healed=False,
            pts=pts,
        )

    G = _prune_short_spurs(G, min_edge_length_px)
    return G


def _prune_short_spurs(G: nx.Graph, min_edge_length_px: float) -> nx.Graph:
    changed = True
    while changed:
        changed = False
        for n in list(G.nodes()):
            if G.degree(n) == 1:
                (neighbor,) = G.neighbors(n)
                length = G.edges[n, neighbor]["length_px"]
                if length < min_edge_length_px:
                    G.remove_node(n)
                    changed = True
    return G


def attach_width(G: nx.Graph, binary_mask: np.ndarray) -> nx.Graph:
    """
    Estimates road width per edge from a Euclidean distance transform
    of the binary mask (distance to nearest background pixel, doubled,
    sampled along the skeleton path -- a standard, cheap width proxy).
    This is what makes weighted (capacity-aware) criticality possible
    instead of treating every edge as equally important (Phase 0, §D).
    """
    if binary_mask.sum() == 0:
        return G
    dist = distance_transform_edt(binary_mask)
    for u, v, data in G.edges(data=True):
        pts = data["pts"]
        rows = np.clip(pts[:, 0].astype(int), 0, dist.shape[0] - 1)
        cols = np.clip(pts[:, 1].astype(int), 0, dist.shape[1] - 1)
        widths = dist[rows, cols] * 2.0
        data["width_px"] = float(np.median(widths)) if len(widths) else 1.0
    return G


def confidence_gated_gap_healing(
    G: nx.Graph,
    prob_map: np.ndarray,
    max_gap_px: float = 15.0,
    min_gap_confidence: float = 0.15,
) -> nx.Graph:
    """
    Connects nearby degree-1 endpoint nodes IF AND ONLY IF the mean
    predicted probability along the straight line between them clears
    `min_gap_confidence`. This is deliberately not the same threshold
    used for the segmentation decision (typically ~0.4-0.5) -- it's a
    much lower evidence bar, because we only need "some support that a
    road plausibly continues here", not "a confident detection". Pure
    nearest-neighbour healing (e.g. MST-style, connect regardless of
    evidence) is explicitly NOT implemented here -- see Phase 0, §C.

    Healed edges get `healed=True` and `confidence` = the sampled
    evidence, so they can be filtered/down-weighted downstream and are
    never indistinguishable from directly observed edges.
    """
    endpoints = [n for n in G.nodes() if G.degree(n) == 1]
    positions = {n: np.array(G.nodes[n]["pos"]) for n in endpoints}

    candidates = []
    for i, a in enumerate(endpoints):
        for b in endpoints[i + 1:]:
            if G.has_edge(a, b):
                continue
            dist = np.linalg.norm(positions[a] - positions[b])
            if dist <= max_gap_px:
                candidates.append((dist, a, b))

    candidates.sort(key=lambda c: c[0])  # heal shortest gaps first

    for dist, a, b in candidates:
        if a not in G or b not in G:
            continue
        if G.degree(a) != 1 or G.degree(b) != 1:
            continue  # may have been healed already this pass

        n_samples = max(4, int(dist))
        line = np.linspace(positions[a], positions[b], n_samples)

        # Sample only the INTERIOR of the gap. The endpoints themselves
        # sit on already-observed, high-confidence road pixels, so
        # including them would bias the evidence estimate upward and
        # defeat the point of the gate -- we specifically need to know
        # whether there's support for a road in the UNOBSERVED middle
        # section, not whether the two known endpoints are roads
        # (they are, by construction).
        margin = max(1, n_samples // 5)
        interior = line[margin:-margin] if n_samples > 2 * margin else line
        evidence = _sample_along_path(prob_map, interior)

        if evidence >= min_gap_confidence:
            G.add_edge(
                a, b,
                length_px=float(dist),
                confidence=float(evidence),
                healed=True,
                pts=line,
            )

    return G


def resolve_grade_separation(G: nx.Graph, dem: Optional[np.ndarray] = None) -> nx.Graph:
    """
    TODO: raster-only reconstruction cannot distinguish an at-grade
    intersection from a flyover crossing a surface road. Per Phase 0,
    this MUST be resolved with a DEM (elevation discontinuity check at
    each junction) or an OSM bridge/tunnel tag cross-reference before
    the graph is trusted for criticality analysis -- do not skip this
    for a "real" AOI. Left unimplemented here because it needs
    georeferenced DEM data this module doesn't have; wire it in once
    tile geotransforms + a DEM raster are available (see Phase 0, §C).
    """
    if dem is None:
        import warnings
        warnings.warn(
            "resolve_grade_separation called without a DEM -- every "
            "crossing in this graph is being treated as an at-grade "
            "intersection. Do not trust criticality results computed "
            "from this graph over a real AOI until this is wired in.",
            stacklevel=2,
        )
    return G


def connectivity_ratio(skeleton: np.ndarray) -> float:
    """
    Fraction of skeleton pixels belonging to the single largest
    connected component. Low values indicate the extraction is
    fragmenting the road network -- exactly the failure mode occlusion
    causes (Phase 0, Part 9). This is a cheap, fast diagnostic to run
    per-tile before ever building a full graph.
    """
    from skimage.measure import label

    if skeleton.sum() == 0:
        return 0.0
    labeled = label(skeleton, connectivity=2)
    sizes = np.bincount(labeled.ravel())[1:]
    if len(sizes) == 0:
        return 0.0
    return float(sizes.max() / sizes.sum())
