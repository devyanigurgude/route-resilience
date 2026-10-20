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

import heapq
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import sknw
from scipy.ndimage import distance_transform_edt
from skimage.morphology import remove_small_objects, skeletonize


def clean_road_mask(binary_mask: np.ndarray, min_component_pixels: int = 3) -> np.ndarray:
    """Drop tiny disconnected mask specks without altering connected road geometry."""
    binary_mask = np.asarray(binary_mask, dtype=bool)
    if binary_mask.ndim != 2:
        raise ValueError(f"Expected a 2D road mask, got shape {binary_mask.shape}")
    if min_component_pixels < 1:
        raise ValueError("min_component_pixels must be positive")
    return remove_small_objects(
        binary_mask,
        min_size=min_component_pixels,
        connectivity=2,
    )


def mask_to_skeleton(binary_mask: np.ndarray) -> np.ndarray:
    """binary_mask: 2D bool/uint8 array, True/1 = predicted road pixel."""
    return skeletonize(binary_mask.astype(bool))


def _sample_along_path(image: np.ndarray, pts: np.ndarray) -> float:
    """Mean value of `image` at the integer pixel coordinates in `pts` (row, col)."""
    rows = np.clip(np.rint(pts[:, 0]).astype(int), 0, image.shape[0] - 1)
    cols = np.clip(np.rint(pts[:, 1]).astype(int), 0, image.shape[1] - 1)
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
    if prob_map is not None and prob_map.shape != skeleton.shape:
        raise ValueError(
            f"Probability map shape {prob_map.shape} does not match skeleton shape {skeleton.shape}"
        )

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
            occlusion_recovered=False,
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


def detect_candidate_occlusions(
    G: nx.Graph,
    max_gap_px: float = 30.0,
    heading_threshold: float = 0.70,
    max_lateral_offset_px: float = 4.5,
    min_trunk_len: float = 6.0,
) -> list[dict]:
    """Detect candidate road occlusion gaps between disconnected degree-1 endpoints.

    Evaluates mutual geometric heading, lateral alignment, and road branch substance.
    """
    endpoints = [n for n in G.nodes() if G.degree(n) == 1]
    positions = {n: np.array(G.nodes[n]["pos"], dtype=float) for n in endpoints}
    tangents = {n: _outward_endpoint_tangent(G, n) for n in endpoints}

    comp_len_map = {}
    for cc in nx.connected_components(G):
        c_len = sum(G.edges[u, v].get("length_px", 0.0) for u, v in G.subgraph(cc).edges())
        for n in cc:
            comp_len_map[n] = c_len

    candidate_records = []
    for i, a in enumerate(endpoints):
        for b in endpoints[i + 1:]:
            if G.has_edge(a, b) or nx.has_path(G, a, b):
                continue
            delta = positions[b] - positions[a]
            dist = float(np.linalg.norm(delta))
            if dist > max_gap_px or dist < 2.0:
                continue

            # Trunk substance check: prune floating noise specks
            if comp_len_map.get(a, 0.0) < min_trunk_len or comp_len_map.get(b, 0.0) < min_trunk_len:
                continue

            direction = delta / dist
            dot_a = float(np.dot(tangents[a], direction))
            dot_b = float(np.dot(tangents[b], -direction))
            mutual_heading = float(np.dot(tangents[a], -tangents[b]))
            longitudinal = float(np.dot(delta, tangents[a]))
            lateral_offset = float(np.linalg.norm(delta - longitudinal * tangents[a]))

            if (
                dot_a >= heading_threshold
                and dot_b >= heading_threshold
                and mutual_heading >= 0.50
                and lateral_offset <= max_lateral_offset_px
            ):
                candidate_records.append({
                    "source": int(a) if isinstance(a, (np.integer, int)) else str(a),
                    "target": int(b) if isinstance(b, (np.integer, int)) else str(b),
                    "pos_source": [float(positions[a][0]), float(positions[a][1])],
                    "pos_target": [float(positions[b][0]), float(positions[b][1])],
                    "gap_px": round(dist, 2),
                    "heading_alignment": round((dot_a + dot_b) / 2.0, 3),
                    "mutual_heading": round(mutual_heading, 3),
                    "lateral_offset_px": round(lateral_offset, 2),
                    "status": "candidate",
                    "recovered": False,
                })
    return candidate_records


def _find_candidate_key(cand_dict: Dict[Any, Any], u: Any, v: Any) -> Optional[Any]:
    if (u, v) in cand_dict:
        return (u, v)
    if (v, u) in cand_dict:
        return (v, u)
    return None


def _record_candidate_recovery(cand_dict: Dict[Any, Any], u: Any, v: Any, evidence_data: Dict[str, Any]) -> None:
    key = _find_candidate_key(cand_dict, u, v)
    if key is not None:
        cand_dict[key]["status"] = "recovered"
        cand_dict[key]["recovered"] = True
        cand_dict[key]["evidence"] = evidence_data
        cand_dict[key]["checks"] = evidence_data.get("checks", [])
        cand_dict[key]["acceptance_reason"] = evidence_data.get("acceptance_reason")


def _record_candidate_rejection(
    cand_dict: Dict[Any, Any], u: Any, v: Any, reason: str, checks: Optional[List[Dict[str, Any]]] = None
) -> None:
    key = _find_candidate_key(cand_dict, u, v)
    if key is not None and not cand_dict[key].get("recovered", False):
        cand_dict[key]["status"] = "rejected"
        cand_dict[key]["rejection_reason"] = reason
        if checks is not None:
            cand_dict[key]["checks"] = checks


def confidence_gated_gap_healing(
    G: nx.Graph,
    prob_map: np.ndarray,
    max_gap_px: float = 15.0,
    min_gap_confidence: float = 0.15,
    tangent_dot_threshold: float = 0.7,
    continuity_penalty_weight: float = 0.0,
    min_component_len: float = 0.0,
    min_total_component_len: float = 0.0,
    max_sinuosity: float = 1.5,
    min_flank_contrast: Optional[float] = None,
    filter_tile_borders: bool = False,
    enable_occlusion_recovery: bool = False,
    max_occlusion_gap_px: float = 30.0,
    min_occlusion_confidence: float = 0.05,
    occlusion_heading_threshold: float = 0.75,
    max_lateral_offset_px: float = 4.0,
    occlusion_mask: Optional[np.ndarray] = None,
    cloud_mask: Optional[np.ndarray] = None,
    enable_cloud_occlusion_recovery: bool = False,
    max_cloud_occlusion_gap_px: float = 240.0,
    cloud_heading_threshold: float = 0.78,
    cloud_min_component_len: float = 20.0,
    cloud_min_coverage: float = 0.55,
    cloud_max_sinuosity: float = 1.35,
) -> nx.Graph:
    """Joins nearby degree-1 endpoints when road directions align and model supports continuity.

    Supports standard evidence-gated healing and an opt-in continuity recovery
    mode for low-confidence gaps. Raster evidence does not itself identify
    whether the cause is cloud, canopy, shadow, or another occlusion.
    """
    if prob_map.ndim != 2:
        raise ValueError(f"Expected a 2D probability map, got shape {prob_map.shape}")
    if not 0.0 <= min_gap_confidence <= 1.0:
        raise ValueError("min_gap_confidence must be in [0, 1]")
    if max_gap_px < 0:
        raise ValueError("max_gap_px must be non-negative")
    if not 0.0 <= tangent_dot_threshold <= 1.0:
        raise ValueError("tangent_dot_threshold must be in [0, 1]")
    if continuity_penalty_weight < 0.0:
        raise ValueError("continuity_penalty_weight must be non-negative")
    if min_component_len < 0.0 or min_total_component_len < 0.0:
        raise ValueError("component length thresholds must be non-negative")
    if max_sinuosity <= 0.0:
        raise ValueError("max_sinuosity must be positive")
    if max_cloud_occlusion_gap_px <= 0.0:
        raise ValueError("max_cloud_occlusion_gap_px must be positive")
    if not 0.0 <= cloud_heading_threshold <= 1.0:
        raise ValueError("cloud_heading_threshold must be in [0, 1]")
    if cloud_min_component_len < 0.0:
        raise ValueError("cloud_min_component_len must be non-negative")
    if not 0.0 <= cloud_min_coverage <= 1.0:
        raise ValueError("cloud_min_coverage must be in [0, 1]")
    if cloud_max_sinuosity <= 0.0:
        raise ValueError("cloud_max_sinuosity must be positive")

    endpoints = [n for n in G.nodes() if G.degree(n) == 1]
    positions = {n: np.array(G.nodes[n]["pos"], dtype=float) for n in endpoints}
    tangents = {n: _outward_endpoint_tangent(G, n) for n in endpoints}

    comp_len_map = {}
    for cc in nx.connected_components(G):
        c_len = sum(G.edges[u, v].get("length_px", 0.0) for u, v in G.subgraph(cc).edges())
        for n in cc:
            comp_len_map[n] = c_len

    H, W = prob_map.shape
    if occlusion_mask is not None:
        occlusion_mask = np.asarray(occlusion_mask, dtype=bool)
        if occlusion_mask.shape != prob_map.shape:
            raise ValueError("occlusion_mask must have the same shape as prob_map")
    if cloud_mask is not None:
        cloud_mask = np.asarray(cloud_mask, dtype=bool)
        if cloud_mask.shape != prob_map.shape:
            raise ValueError("cloud_mask must have the same shape as prob_map")
    effective_cloud_mask = cloud_mask if cloud_mask is not None else occlusion_mask

    search_dist_limit = max_occlusion_gap_px if enable_occlusion_recovery else max_gap_px
    if enable_cloud_occlusion_recovery and effective_cloud_mask is not None:
        search_dist_limit = max(search_dist_limit, max_cloud_occlusion_gap_px)

    candidates = []
    candidate_occlusions_dict = {}

    for i, a in enumerate(endpoints):
        for b in endpoints[i + 1:]:
            if G.has_edge(a, b):
                continue
            delta = positions[b] - positions[a]
            dist = float(np.linalg.norm(delta))
            if dist > search_dist_limit or dist == 0:
                continue

            # Topological scale checks: prune unsubstantiated floating noise spurs
            if min_component_len > 0.0 and dist > 5.0:
                if min(comp_len_map.get(a, 0.0), comp_len_map.get(b, 0.0)) < min_component_len:
                    continue
            if min_total_component_len > 0.0:
                if comp_len_map.get(a, 0.0) + comp_len_map.get(b, 0.0) < min_total_component_len:
                    continue

            # Tile boundary guard: prune artificial border padding stumps
            if filter_tile_borders:
                pa, pb = positions[a], positions[b]
                border_a = (pa[0] <= 4 or pa[0] >= H - 5 or pa[1] <= 4 or pa[1] >= W - 5)
                border_b = (pb[0] <= 4 or pb[0] >= H - 5 or pb[1] <= 4 or pb[1] >= W - 5)
                if border_a and border_b:
                    continue

            direction = delta / dist
            dot_a = float(np.dot(tangents[a], direction))
            dot_b = float(np.dot(tangents[b], -direction))
            mutual_heading = float(np.dot(tangents[a], -tangents[b]))
            longitudinal = float(np.dot(delta, tangents[a]))
            lateral_offset = float(np.linalg.norm(delta - longitudinal * tangents[a]))

            is_occlusion_candidate = (
                enable_occlusion_recovery
                and dist <= max_occlusion_gap_px
                and dot_a >= 0.70
                and dot_b >= 0.70
                and mutual_heading >= 0.50
                and lateral_offset <= max_lateral_offset_px
                and comp_len_map.get(a, 0.0) >= 6.0
                and comp_len_map.get(b, 0.0) >= 6.0
                and not nx.has_path(G, a, b)
            )

            if is_occlusion_candidate:
                candidate_record = {
                    "source": int(a) if isinstance(a, (np.integer, int)) else str(a),
                    "target": int(b) if isinstance(b, (np.integer, int)) else str(b),
                    "pos_source": [float(positions[a][0]), float(positions[a][1])],
                    "pos_target": [float(positions[b][0]), float(positions[b][1])],
                    "gap_px": round(dist, 2),
                    "heading_alignment": round((dot_a + dot_b) / 2.0, 3),
                    "mutual_heading": round(mutual_heading, 3),
                    "lateral_offset_px": round(lateral_offset, 2),
                    "status": "candidate",
                    "recovered": False,
                    "suspected_occlusion": True,
                }
                candidate_occlusions_dict[(a, b)] = candidate_record

            # Large clouds invalidate the probability-evidence assumption used
            # above: the road may be entirely absent from the segmentation map.
            # Only bridge such spans when the *image-derived* cloud mask covers
            # a smooth continuation between two long, strongly facing branches.
            cloud_path = None
            cloud_coverage = 0.0
            is_cloud_candidate = False
            if (
                enable_cloud_occlusion_recovery
                and effective_cloud_mask is not None
                and max_gap_px < dist <= max_cloud_occlusion_gap_px
                and dot_a >= cloud_heading_threshold
                and dot_b >= cloud_heading_threshold
                and comp_len_map.get(a, 0.0) >= cloud_min_component_len
                and comp_len_map.get(b, 0.0) >= cloud_min_component_len
                and not nx.has_path(G, a, b)
            ):
                # Distance-adaptive mutual heading: for short cloud gaps
                # (~30 px) the road is unlikely to turn much → strict.
                # For long cloud gaps (~200 px), a road can legitimately
                # curve up to ~55° beneath the cloud → relaxed.
                gap_fraction = min(1.0, (dist - max_gap_px) / max(1.0, max_cloud_occlusion_gap_px - max_gap_px))
                adaptive_mutual_heading = 0.85 - 0.25 * gap_fraction  # 0.85 → 0.60
                # Lateral offset tolerance also scales with gap distance:
                # roads at 30 px offset are fine over a 200 px cloud gap.
                adaptive_lateral_limit = max_lateral_offset_px + dist * 0.20
                if (
                    mutual_heading >= adaptive_mutual_heading
                    and lateral_offset <= adaptive_lateral_limit
                ):
                    cloud_path = _tangent_continuity_curve(
                        positions[a], positions[b], tangents[a], tangents[b]
                    )
                    cloud_length = _polyline_length(cloud_path)
                    cloud_sinuosity = cloud_length / max(1e-3, dist)
                    cloud_coverage = _mask_coverage(effective_cloud_mask, cloud_path)
                    is_cloud_candidate = (
                        cloud_sinuosity <= cloud_max_sinuosity
                        and cloud_coverage >= cloud_min_coverage
                    )
                    if is_cloud_candidate:
                        candidate_occlusions_dict[(a, b)] = {
                            "source": int(a) if isinstance(a, (np.integer, int)) else str(a),
                            "target": int(b) if isinstance(b, (np.integer, int)) else str(b),
                            "pos_source": [float(positions[a][0]), float(positions[a][1])],
                            "pos_target": [float(positions[b][0]), float(positions[b][1])],
                            "gap_px": round(dist, 2),
                            "heading_alignment": round((dot_a + dot_b) / 2.0, 3),
                            "mutual_heading": round(mutual_heading, 3),
                            "lateral_offset_px": round(lateral_offset, 2),
                            "cloud_coverage": round(cloud_coverage, 3),
                            "occlusion_kind": "cloud",
                            "status": "candidate",
                            "recovered": False,
                        }

            # Qualifying check for healing candidate evaluation
            qualifies_standard = (
                dist <= max_gap_px
                and dot_a >= tangent_dot_threshold
                and dot_b >= tangent_dot_threshold
            )
            qualifies_occlusion = (
                is_occlusion_candidate
                and dot_a >= occlusion_heading_threshold
                and dot_b >= occlusion_heading_threshold
                and mutual_heading >= 0.60
            )

            if qualifies_standard or qualifies_occlusion:
                alignment_deficit = max(0.0, 2.0 - (dot_a + dot_b))
                weight_eff = continuity_penalty_weight if continuity_penalty_weight > 0.0 else (0.5 if qualifies_occlusion else 0.0)
                # A zero continuity weight is the documented baseline: rank by
                # Euclidean distance only.  Lateral offset is a continuity
                # criterion and must not silently alter that baseline.
                sort_score = dist * (1.0 + weight_eff * alignment_deficit)
                candidates.append((sort_score, dist, a, b, is_occlusion_candidate, lateral_offset, mutual_heading, dot_a, dot_b))
            elif is_occlusion_candidate:
                reasons = []
                if dot_a < occlusion_heading_threshold or dot_b < occlusion_heading_threshold:
                    reasons.append(f"Heading alignment below threshold ({dot_a:.2f}, {dot_b:.2f} < {occlusion_heading_threshold:.2f})")
                if mutual_heading < 0.60:
                    reasons.append(f"Mutual heading below threshold ({mutual_heading:.2f} < 0.60)")
                _record_candidate_rejection(
                    candidate_occlusions_dict, a, b,
                    "; ".join(reasons) if reasons else "Continuity qualification thresholds not met",
                    [
                        {"name": "Gap Distance", "passed": dist <= max_occlusion_gap_px, "detail": f"{dist:.1f} px <= {max_occlusion_gap_px:.1f} px"},
                        {"name": "Endpoint Heading Alignment", "passed": dot_a >= occlusion_heading_threshold and dot_b >= occlusion_heading_threshold, "detail": f"dot_a={dot_a:.2f}, dot_b={dot_b:.2f} (threshold: {occlusion_heading_threshold:.2f})"},
                        {"name": "Mutual Tangent Alignment", "passed": mutual_heading >= 0.60, "detail": f"{mutual_heading:.2f} (threshold: 0.60)"},
                        {"name": "Lateral Offset", "passed": lateral_offset <= max_lateral_offset_px, "detail": f"{lateral_offset:.1f} px <= {max_lateral_offset_px:.1f} px"},
                    ],
                )
            if is_cloud_candidate:
                # Cloud candidates are processed before local fragment repairs so
                # a short spur cannot consume a genuine cloud-facing endpoint.
                cloud_score = -((dot_a + dot_b) / 2.0 + mutual_heading + cloud_coverage)
                candidates.append((cloud_score, dist, a, b, "cloud", lateral_offset, mutual_heading, cloud_path, cloud_coverage, dot_a, dot_b))

    candidates.sort(key=lambda c: c[0])  # cloud confidence first, otherwise best-aligned / shortest gaps

    for candidate in candidates:
        if candidate[4] == "cloud":
            sort_score, dist, a, b, is_occ, lat_offset, mut_heading, cloud_path, cloud_coverage, dot_a, dot_b = candidate
        else:
            sort_score, dist, a, b, is_occ, lat_offset, mut_heading, dot_a, dot_b = candidate[:9]
        if a not in G or b not in G:
            _record_candidate_rejection(
                candidate_occlusions_dict, a, b, "Endpoint node no longer in graph"
            )
            continue
        if G.degree(a) != 1 or G.degree(b) != 1:
            _record_candidate_rejection(
                candidate_occlusions_dict, a, b,
                f"Endpoint degree is not 1 (degree(a)={G.degree(a)}, degree(b)={G.degree(b)}); endpoints already connected or junction"
            )
            continue  # may have been healed already this pass
        if nx.has_path(G, a, b):
            _record_candidate_rejection(
                candidate_occlusions_dict, a, b,
                "Cycle prevention: graph already contains an alternate path between endpoints"
            )
            continue

        if is_occ == "cloud":
            cloud_path, cloud_coverage = candidate[7], candidate[8]
            cloud_len = float(_polyline_length(cloud_path))
            cloud_sinuosity = float(cloud_len / max(1e-3, dist))
            gap_fraction = min(1.0, (dist - max_gap_px) / max(1.0, max_cloud_occlusion_gap_px - max_gap_px))
            adaptive_mutual_heading = 0.85 - 0.25 * gap_fraction
            adaptive_lateral_limit = max_lateral_offset_px + dist * 0.20

            evidence_data = {
                "gap_px": round(float(dist), 2),
                "heading_alignment": round(float((dot_a + dot_b) / 2.0), 3),
                "dot_a": round(float(dot_a), 3),
                "dot_b": round(float(dot_b), 3),
                "mutual_heading": round(float(mut_heading), 3),
                "lateral_offset_px": round(float(lat_offset), 2),
                "cloud_coverage": round(float(cloud_coverage), 3),
                "sinuosity": round(float(cloud_sinuosity), 3),
                "acceptance_reason": "Accepted via Cloud Occlusion Continuity: smooth tangent continuation constrained by endpoint angles bridges the cloud-obscured span.",
                "checks": [
                    {"name": "Gap Distance", "passed": True, "detail": f"{dist:.1f} px <= {max_cloud_occlusion_gap_px:.1f} px limit"},
                    {"name": "Endpoint Heading Alignment", "passed": True, "detail": f"dot_a={dot_a:.2f}, dot_b={dot_b:.2f} >= {cloud_heading_threshold:.2f}"},
                    {"name": "Mutual Tangent Alignment", "passed": True, "detail": f"{mut_heading:.2f} >= {adaptive_mutual_heading:.2f}"},
                    {"name": "Lateral Offset", "passed": True, "detail": f"{lat_offset:.1f} px <= {adaptive_lateral_limit:.1f} px"},
                    {"name": "Cloud Mask Coverage", "passed": True, "detail": f"{float(cloud_coverage):.1%} >= {cloud_min_coverage:.1%}"},
                    {"name": "Path Sinuosity Guard", "passed": True, "detail": f"{cloud_sinuosity:.2f} <= {cloud_max_sinuosity:.2f}"},
                    {"name": "Acyclicity & Degree", "passed": True, "detail": "Endpoints degree 1, no prior alternate path"},
                ],
            }

            G.add_edge(
                a, b,
                length_px=cloud_len,
                confidence=float(cloud_coverage),
                healed=True,
                inferred=True,
                occlusion_suspected=True,
                occlusion_recovered=True,
                occlusion_kind="cloud",
                cloud_coverage=float(cloud_coverage),
                recovery_type="cloud_continuity",
                pts=cloud_path,
                evidence=evidence_data,
                gap_px=evidence_data["gap_px"],
                heading_alignment=evidence_data["heading_alignment"],
                dot_a=evidence_data["dot_a"],
                dot_b=evidence_data["dot_b"],
                mutual_heading=evidence_data["mutual_heading"],
                lateral_offset_px=evidence_data["lateral_offset_px"],
                sinuosity=evidence_data["sinuosity"],
                checks=evidence_data["checks"],
                acceptance_reason=evidence_data["acceptance_reason"],
            )
            _record_candidate_recovery(candidate_occlusions_dict, a, b, evidence_data)
            continue

        n_samples = max(4, int(dist))
        line = np.linspace(positions[a], positions[b], n_samples)

        margin = max(1, n_samples // 5)
        interior = line[margin:-margin] if n_samples > 2 * margin else line
        rows = np.clip(np.rint(interior[:, 0]).astype(int), 0, H - 1)
        cols = np.clip(np.rint(interior[:, 1]).astype(int), 0, W - 1)
        support = prob_map[rows, cols]
        evidence = float(support.mean())
        supported_fraction = float(np.mean(support >= min_gap_confidence))

        # Check flank contrast for straight line
        delta = positions[b] - positions[a]
        direction = delta / dist
        normal = np.array([-direction[1], direction[0]])
        contrast = 0.0
        if min_flank_contrast is not None:
            l_pts = interior + 3.0 * normal
            r_pts = interior - 3.0 * normal
            l_rows = np.clip(np.rint(l_pts[:, 0]).astype(int), 0, H - 1)
            l_cols = np.clip(np.rint(l_pts[:, 1]).astype(int), 0, W - 1)
            r_rows = np.clip(np.rint(r_pts[:, 0]).astype(int), 0, H - 1)
            r_cols = np.clip(np.rint(r_pts[:, 1]).astype(int), 0, W - 1)
            flank_evidence = 0.5 * float(prob_map[l_rows, l_cols].mean() + prob_map[r_rows, r_cols].mean())
            contrast = evidence - flank_evidence
            if contrast < min_flank_contrast and not is_occ:
                _record_candidate_rejection(
                    candidate_occlusions_dict, a, b,
                    f"Flank contrast ({contrast:.3f}) below threshold ({min_flank_contrast:.3f})",
                    [
                        {"name": "Gap Distance", "passed": True, "detail": f"{dist:.1f} px <= {max_gap_px:.1f} px"},
                        {"name": "Flank Contrast", "passed": False, "detail": f"{contrast:.3f} < {min_flank_contrast:.3f}"},
                    ],
                )
                continue

        # Path 1: Standard high-confidence straight gap
        if dist <= max_gap_px and evidence >= min_gap_confidence and supported_fraction >= 0.7:
            checks = [
                {"name": "Gap Distance", "passed": True, "detail": f"{dist:.1f} px <= {max_gap_px:.1f} px"},
                {"name": "Probability Evidence", "passed": True, "detail": f"{evidence:.3f} >= {min_gap_confidence:.3f}"},
                {"name": "Supported Pixel Fraction", "passed": True, "detail": f"{supported_fraction:.1%} >= 70.0%"},
            ]
            if min_flank_contrast is not None:
                checks.append({"name": "Flank Contrast", "passed": True, "detail": f"{contrast:.3f} >= {min_flank_contrast:.3f}"})
            checks.append({"name": "Acyclicity & Degree", "passed": True, "detail": "Endpoints degree 1, no prior alternate path"})

            evidence_data = {
                "gap_px": round(float(dist), 2),
                "heading_alignment": round(float((dot_a + dot_b) / 2.0), 3),
                "dot_a": round(float(dot_a), 3),
                "dot_b": round(float(dot_b), 3),
                "mutual_heading": round(float(mut_heading), 3),
                "lateral_offset_px": round(float(lat_offset), 2),
                "probability_evidence": round(float(evidence), 3),
                "supported_fraction": round(float(supported_fraction), 3),
                "flank_contrast": round(float(contrast), 3) if min_flank_contrast is not None else None,
                "sinuosity": 1.0,
                "acceptance_reason": "Accepted via Direct Evidence-Gated Gap: straight segment with strong road probability support.",
                "checks": checks,
            }

            G.add_edge(
                a, b,
                length_px=float(dist),
                confidence=float(evidence),
                healed=True,
                inferred=True,
                occlusion_recovered=False,
                recovery_type="evidence_gated_direct",
                pts=line,
                evidence=evidence_data,
                gap_px=evidence_data["gap_px"],
                heading_alignment=evidence_data["heading_alignment"],
                dot_a=evidence_data["dot_a"],
                dot_b=evidence_data["dot_b"],
                mutual_heading=evidence_data["mutual_heading"],
                lateral_offset_px=evidence_data["lateral_offset_px"],
                probability_evidence=evidence_data["probability_evidence"],
                supported_fraction=evidence_data["supported_fraction"],
                flank_contrast=evidence_data["flank_contrast"],
                sinuosity=evidence_data["sinuosity"],
                checks=evidence_data["checks"],
                acceptance_reason=evidence_data["acceptance_reason"],
            )
            _record_candidate_recovery(candidate_occlusions_dict, a, b, evidence_data)
        else:
            # Path 2: Probability-guided search across curved or occluded paths
            search_max_len = float(max(dist * 1.4, max_gap_px * 1.5))
            recovered_path = _find_probability_guided_path(
                prob_map=prob_map,
                start_pos=positions[a],
                end_pos=positions[b],
                max_path_length=search_max_len,
            )
            if (recovered_path is None or len(recovered_path) < 2) and is_occ and occlusion_mask is not None:
                recovered_path = line

            if recovered_path is not None and len(recovered_path) >= 2:
                diffs = np.diff(recovered_path, axis=0)
                path_length = float(np.sum(np.sqrt(np.sum(diffs ** 2, axis=1))))
                sinuosity = path_length / max(1e-3, dist)
                if sinuosity > max_sinuosity or path_length > search_max_len:
                    _record_candidate_rejection(
                        candidate_occlusions_dict, a, b,
                        f"Path sinuosity ({sinuosity:.2f}) or length ({path_length:.1f} px) exceeded limit",
                        [
                            {"name": "Gap Distance", "passed": True, "detail": f"{dist:.1f} px"},
                            {"name": "Path Sinuosity Guard", "passed": False, "detail": f"{sinuosity:.2f} > {max_sinuosity:.2f}"},
                        ],
                    )
                    continue

                n_pts = len(recovered_path)
                p_margin = max(1, n_pts // 5)
                p_interior = (
                    recovered_path[p_margin:-p_margin]
                    if n_pts > 2 * p_margin
                    else recovered_path
                )
                p_rows = np.clip(np.rint(p_interior[:, 0]).astype(int), 0, H - 1)
                p_cols = np.clip(np.rint(p_interior[:, 1]).astype(int), 0, W - 1)
                p_support = prob_map[p_rows, p_cols]
                p_evidence = float(p_support.mean())
                p_supported_fraction = float(np.mean(p_support >= min_gap_confidence))

                p_contrast = 0.0
                if min_flank_contrast is not None:
                    l_pts = p_interior + 3.0 * normal
                    r_pts = p_interior - 3.0 * normal
                    l_rows = np.clip(np.rint(l_pts[:, 0]).astype(int), 0, H - 1)
                    l_cols = np.clip(np.rint(l_pts[:, 1]).astype(int), 0, W - 1)
                    r_rows = np.clip(np.rint(r_pts[:, 0]).astype(int), 0, H - 1)
                    r_cols = np.clip(np.rint(r_pts[:, 1]).astype(int), 0, W - 1)
                    p_flank = 0.5 * float(prob_map[l_rows, l_cols].mean() + prob_map[r_rows, r_cols].mean())
                    p_contrast = p_evidence - p_flank

                occ_coverage = 0.0
                if occlusion_mask is not None:
                    occ_coverage = float(_mask_coverage(occlusion_mask, recovered_path))

                # Standard A* recovery: high evidence throughout
                standard_a_star_ok = (
                    dist <= max_gap_px
                    and p_evidence >= min_gap_confidence
                    and p_supported_fraction >= 0.7
                    and (min_flank_contrast is None or p_contrast >= min_flank_contrast)
                )

                # Occlusion-aware continuity recovery: handles low-confidence spans (trees, shadows, clouds)
                # where strong collinear continuation justifies bridging across suppressed probabilities
                occlusion_recovery_ok = (
                    enable_occlusion_recovery
                    and is_occ
                    and dist <= max_occlusion_gap_px
                    and sinuosity <= 1.25
                    and p_evidence >= min_occlusion_confidence
                    and (min_flank_contrast is None or p_contrast >= -0.05)
                )

                # Narrow exception for fully hidden spans under verified photometric occlusion:
                # When dense canopy or deep shadow zeroes out the U-Net probability, allow bridging
                # ONLY IF:
                # 1. Endpoints strictly qualify as an occlusion candidate (is_occ=True, degree=1, collinear)
                # 2. Both endpoints have strict heading alignment (dot_a, dot_b >= occlusion_heading_threshold)
                # 3. Path is nearly straight (sinuosity <= 1.20)
                # 4. Verified physical occlusion coverage along path >= 50%
                occlusion_mask_recovery_ok = (
                    enable_occlusion_recovery
                    and is_occ
                    and dist <= max_occlusion_gap_px
                    and dot_a >= occlusion_heading_threshold
                    and dot_b >= occlusion_heading_threshold
                    and mutual_heading >= 0.60
                    and sinuosity <= 1.20
                    and occ_coverage >= 0.50
                )

                if standard_a_star_ok or occlusion_recovery_ok or occlusion_mask_recovery_ok:
                    is_occ_recovered = True
                    recovery_type = (
                        "occlusion_continuity"
                        if (occlusion_recovery_ok or occlusion_mask_recovery_ok)
                        else "evidence_guided"
                    )
                    acceptance_reason = (
                        "Accepted via Verified Occlusion Continuity: collinear endpoints bridged across verified photometric occlusion (coverage >= 50%)."
                        if (occlusion_mask_recovery_ok and not occlusion_recovery_ok)
                        else (
                            "Accepted via Occlusion-Aware Collinear Continuity: strong collinear endpoint alignment justifies bridging across low-confidence occlusion span."
                            if occlusion_recovery_ok
                            else "Accepted via Evidence-Guided Path Search: continuous high-probability corridor discovered."
                        )
                    )
                    checks = [
                        {
                            "name": "Gap Distance",
                            "passed": True,
                            "detail": f"{dist:.1f} px <= {max_occlusion_gap_px if (occlusion_recovery_ok or occlusion_mask_recovery_ok) else max_gap_px:.1f} px",
                        },
                        {
                            "name": "Endpoint Heading Alignment",
                            "passed": True,
                            "detail": f"avg dot {(dot_a + dot_b) / 2.0:.2f}, mutual {mut_heading:.2f}",
                        },
                        {
                            "name": "Lateral Offset",
                            "passed": True,
                            "detail": f"{lat_offset:.1f} px <= {max_lateral_offset_px:.1f} px",
                        },
                    ]
                    if occlusion_mask_recovery_ok and not occlusion_recovery_ok:
                        checks.append({
                            "name": "Photometric Occlusion Coverage",
                            "passed": True,
                            "detail": f"{occ_coverage:.1%} >= 50.0% (bridges zero-evidence span under verified shadow/canopy/cloud)",
                        })
                    else:
                        checks.append({
                            "name": "Probability Evidence",
                            "passed": True,
                            "detail": f"{p_evidence:.3f} >= {min_occlusion_confidence if occlusion_recovery_ok else min_gap_confidence:.3f}",
                        })
                    checks.extend([
                        {
                            "name": "Path Sinuosity Guard",
                            "passed": True,
                            "detail": f"{sinuosity:.2f} <= {1.25 if (occlusion_recovery_ok or occlusion_mask_recovery_ok) else max_sinuosity:.2f}",
                        },
                    ])
                    if min_flank_contrast is not None:
                        checks.append({
                            "name": "Flank Contrast",
                            "passed": True,
                            "detail": f"{p_contrast:.3f} >= {'-0.05' if (occlusion_recovery_ok or occlusion_mask_recovery_ok) else str(min_flank_contrast)}",
                        })
                    checks.append({
                        "name": "Acyclicity & Degree",
                        "passed": True,
                        "detail": "Endpoints degree 1, no prior alternate path",
                    })

                    evidence_data = {
                        "gap_px": round(float(dist), 2),
                        "heading_alignment": round(float((dot_a + dot_b) / 2.0), 3),
                        "dot_a": round(float(dot_a), 3),
                        "dot_b": round(float(dot_b), 3),
                        "mutual_heading": round(float(mut_heading), 3),
                        "lateral_offset_px": round(float(lat_offset), 2),
                        "probability_evidence": round(float(p_evidence), 3),
                        "supported_fraction": round(float(p_supported_fraction), 3),
                        "flank_contrast": round(float(p_contrast), 3) if min_flank_contrast is not None else None,
                        "sinuosity": round(float(sinuosity), 3),
                        "occlusion_coverage": round(float(occ_coverage), 3) if occlusion_mask is not None else None,
                        "acceptance_reason": acceptance_reason,
                        "checks": checks,
                    }

                    G.add_edge(
                        a, b,
                        length_px=path_length,
                        confidence=p_evidence,
                        healed=True,
                        occlusion_recovered=is_occ_recovered,
                        inferred=True,
                        occlusion_suspected=bool(occlusion_recovery_ok or occlusion_mask_recovery_ok),
                        occlusion_kind="photometric_occlusion" if (occlusion_mask_recovery_ok and not occlusion_recovery_ok) else ("occlusion" if occlusion_recovery_ok else None),
                        occlusion_coverage=float(occ_coverage) if occlusion_mask is not None else 0.0,
                        recovery_type=recovery_type,
                        pts=recovered_path,
                        evidence=evidence_data,
                        gap_px=evidence_data["gap_px"],
                        heading_alignment=evidence_data["heading_alignment"],
                        dot_a=evidence_data["dot_a"],
                        dot_b=evidence_data["dot_b"],
                        mutual_heading=evidence_data["mutual_heading"],
                        lateral_offset_px=evidence_data["lateral_offset_px"],
                        probability_evidence=evidence_data["probability_evidence"],
                        supported_fraction=evidence_data["supported_fraction"],
                        flank_contrast=evidence_data["flank_contrast"],
                        sinuosity=evidence_data["sinuosity"],
                        checks=evidence_data["checks"],
                        acceptance_reason=evidence_data["acceptance_reason"],
                    )
                    _record_candidate_recovery(candidate_occlusions_dict, a, b, evidence_data)
                else:
                    reasons = []
                    checks = []
                    dist_ok = (dist <= max_occlusion_gap_px if is_occ else dist <= max_gap_px)
                    checks.append({"name": "Gap Distance", "passed": dist_ok, "detail": f"{dist:.1f} px (limit: {max_occlusion_gap_px if is_occ else max_gap_px:.1f} px)"})
                    if not dist_ok:
                        reasons.append(f"Gap distance ({dist:.1f} px) exceeds limit")

                    min_ev = min_occlusion_confidence if is_occ else min_gap_confidence
                    ev_ok = (p_evidence >= min_ev) or (is_occ and occ_coverage >= 0.50)
                    checks.append({"name": "Probability Evidence", "passed": ev_ok, "detail": f"{p_evidence:.3f} (threshold: {min_ev:.3f}, occlusion_coverage={occ_coverage:.1%})"})
                    if not ev_ok:
                        reasons.append(f"Probability evidence ({p_evidence:.3f}) below threshold ({min_ev:.3f}) without occlusion mask support")

                    if not is_occ:
                        frac_ok = (p_supported_fraction >= 0.7)
                        checks.append({"name": "Supported Fraction", "passed": frac_ok, "detail": f"{p_supported_fraction:.1%} (threshold: 70.0%)"})
                        if not frac_ok:
                            reasons.append(f"Supported pixel fraction below 70% ({p_supported_fraction:.1%})")

                    sin_limit = 1.25 if is_occ else max_sinuosity
                    sin_ok = (sinuosity <= sin_limit)
                    checks.append({"name": "Path Sinuosity Guard", "passed": sin_ok, "detail": f"{sinuosity:.2f} (limit: {sin_limit:.2f})"})
                    if not sin_ok:
                        reasons.append(f"Path sinuosity ({sinuosity:.2f}) exceeds limit ({sin_limit:.2f})")

                    if min_flank_contrast is not None:
                        fc_limit = -0.05 if is_occ else min_flank_contrast
                        fc_ok = (p_contrast >= fc_limit)
                        checks.append({"name": "Flank Contrast", "passed": fc_ok, "detail": f"{p_contrast:.3f} (limit: {fc_limit:.3f})"})
                        if not fc_ok:
                            reasons.append(f"Flank contrast ({p_contrast:.3f}) below limit ({fc_limit:.3f})")

                    _record_candidate_rejection(
                        candidate_occlusions_dict, a, b,
                        "; ".join(reasons) if reasons else "Continuity criteria not satisfied",
                        checks,
                    )
            else:
                _record_candidate_rejection(
                    candidate_occlusions_dict, a, b,
                    "No continuous path found through probability map",
                    [
                        {"name": "Gap Distance", "passed": True, "detail": f"{dist:.1f} px"},
                        {"name": "Path Search", "passed": False, "detail": "A* search reached max depth without continuous path"},
                    ],
                )

    G.graph["candidate_occlusions"] = list(candidate_occlusions_dict.values())
    return G


def _tangent_continuity_curve(
    start: np.ndarray,
    end: np.ndarray,
    start_tangent: np.ndarray,
    end_tangent: np.ndarray,
) -> np.ndarray:
    """Return a smooth cubic Bézier continuation constrained by endpoint tangents.

    ``end_tangent`` points outward from the second road branch, so the curve
    arrives with direction ``-end_tangent``.  This captures a road bend hidden
    by a cloud without allowing arbitrary detours through the image.
    """
    chord = float(np.linalg.norm(end - start))
    control = min(max(12.0, chord * 0.45), chord * 0.60)
    p0 = np.asarray(start, dtype=float)
    p1 = p0 + control * np.asarray(start_tangent, dtype=float)
    p3 = np.asarray(end, dtype=float)
    p2 = p3 + control * np.asarray(end_tangent, dtype=float)
    sample_count = max(16, int(np.ceil(chord / 2.0)))
    t = np.linspace(0.0, 1.0, sample_count)[:, None]
    return (
        (1.0 - t) ** 3 * p0
        + 3.0 * (1.0 - t) ** 2 * t * p1
        + 3.0 * (1.0 - t) * t ** 2 * p2
        + t ** 3 * p3
    )


def _polyline_length(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _mask_coverage(mask: np.ndarray, points: np.ndarray) -> float:
    """Measure mask support on the middle of a proposed continuation."""
    points = np.asarray(points, dtype=float)
    margin = max(1, len(points) // 8)
    interior = points[margin:-margin] if len(points) > 2 * margin else points
    rows = np.clip(np.rint(interior[:, 0]).astype(int), 0, mask.shape[0] - 1)
    cols = np.clip(np.rint(interior[:, 1]).astype(int), 0, mask.shape[1] - 1)
    return float(mask[rows, cols].mean()) if len(rows) else 0.0


def _find_probability_guided_path(
    prob_map: np.ndarray,
    start_pos: np.ndarray,
    end_pos: np.ndarray,
    max_path_length: float,
) -> Optional[np.ndarray]:
    """
    Finds a plausible 8-connected path between two endpoints guided by the
    probability map using deterministic A* / Dijkstra search with heapq.

    Step cost prioritizes high-confidence road probabilities:
        step_cost = step_dist * (1.0 - clip(p, 0.0, 1.0) + 0.05)
    """
    H, W = prob_map.shape
    start_rc = (
        int(np.clip(np.round(start_pos[0]), 0, H - 1)),
        int(np.clip(np.round(start_pos[1]), 0, W - 1)),
    )
    goal_rc = (
        int(np.clip(np.round(end_pos[0]), 0, H - 1)),
        int(np.clip(np.round(end_pos[1]), 0, W - 1)),
    )

    if start_rc == goal_rc:
        return np.array([start_pos, end_pos], dtype=float)

    pad = int(np.ceil(max(5.0, max_path_length * 0.5)))
    r_min = max(0, min(start_rc[0], goal_rc[0]) - pad)
    r_max = min(H - 1, max(start_rc[0], goal_rc[0]) + pad)
    c_min = max(0, min(start_rc[1], goal_rc[1]) - pad)
    c_max = min(W - 1, max(start_rc[1], goal_rc[1]) + pad)

    DIRECTIONS = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, 1.4142135623730951),
        (-1, 1, 1.4142135623730951),
        (1, -1, 1.4142135623730951),
        (1, 1, 1.4142135623730951),
    ]

    def heuristic(r: int, c: int) -> float:
        dr = r - goal_rc[0]
        dc = c - goal_rc[1]
        return np.hypot(dr, dc) * 0.05

    counter = 0
    pq = [(heuristic(*start_rc), 0.0, counter, start_rc)]
    g_cost = {start_rc: 0.0}
    geo_dist = {start_rc: 0.0}
    parent = {}

    found = False
    while pq:
        f, g, _, u = heapq.heappop(pq)

        if g > g_cost.get(u, float("inf")):
            continue

        if u == goal_rc:
            found = True
            break

        ur, uc = u
        u_geo = geo_dist[u]

        for dr, dc, step_dist in DIRECTIONS:
            nr = ur + dr
            nc = uc + dc

            if not (r_min <= nr <= r_max and c_min <= nc <= c_max):
                continue

            n_geo = u_geo + step_dist
            if n_geo > max_path_length:
                continue

            p = float(prob_map[nr, nc])
            step_weight = step_dist * (1.0 - float(np.clip(p, 0.0, 1.0)) + 0.05)
            new_g = g + step_weight

            v = (nr, nc)
            if new_g < g_cost.get(v, float("inf")):
                g_cost[v] = new_g
                geo_dist[v] = n_geo
                parent[v] = u
                counter += 1
                f_score = new_g + heuristic(nr, nc)
                heapq.heappush(pq, (f_score, new_g, counter, v))

    if not found:
        return None

    path = []
    curr = goal_rc
    while curr in parent:
        path.append(curr)
        curr = parent[curr]
    path.append(start_rc)
    path.reverse()

    if len(path) <= 2:
        coords = np.array([start_pos, end_pos], dtype=float)
    else:
        mid_points = [np.array([float(r), float(c)], dtype=float) for r, c in path[1:-1]]
        coords = np.array([start_pos] + mid_points + [end_pos], dtype=float)

    return coords


def _outward_endpoint_tangent(G: nx.Graph, node, lookback: int = 12) -> np.ndarray:
    """Estimate the local outward direction from an endpoint using its edge geometry.

    ``lookback`` controls how many inner skeleton points (up to *lookback*)
    are averaged to compute the tangent.  A deeper lookback (default 12)
    yields a more stable direction estimate for roads approaching large
    cloud boundaries, where the segment closest to the endpoint may be
    short and noisy.  Short edges automatically use fewer points.
    """
    position = np.asarray(G.nodes[node]["pos"], dtype=float)
    (neighbor,) = G.neighbors(node)
    points = np.asarray(G.edges[node, neighbor].get("pts", []), dtype=float)

    if points.ndim == 2 and len(points) >= 2:
        start_distance = np.linalg.norm(points[0] - position)
        end_distance = np.linalg.norm(points[-1] - position)
        oriented = points if start_distance <= end_distance else points[::-1]
        inner_points = oriented[1:min(lookback + 1, len(oriented))]
        tangent = position - inner_points.mean(axis=0)
    else:
        tangent = position - np.asarray(G.nodes[neighbor]["pos"], dtype=float)

    magnitude = np.linalg.norm(tangent)
    if magnitude == 0:
        return np.zeros(2, dtype=float)
    return tangent / magnitude


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
