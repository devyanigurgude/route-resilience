from __future__ import annotations

import io
import json
import os
import shutil
import sys
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

_backend_dir = str(Path(__file__).resolve().parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)
from threading import Lock
from typing import Any, Dict

ACTIVE_ANALYSES: set[str] = set()

from pydantic import BaseModel, Field
import numpy as np
import networkx as nx
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw

from route_resilience.config import Config
from route_resilience.criticality import criticality_report, network_efficiency
from route_resilience.disruption import create_failure_scenario, simulate_removal
from route_resilience.evaluate_topology import graph_summary
from route_resilience.phase3_pipeline import graph_to_jsonable
from route_resilience.real_graph_pipeline import graph_from_probability_map, load_model_for_inference, predict_probability_map, save_artifacts
from route_resilience.recovery import reinstate_baseline_assets
from route_resilience.resilience import evaluate_resilience, evaluate_state_service
from route_resilience.routing import (
    route_result,
    select_automatic_route_pair,
    select_baseline_od_pairs,
    select_reachable_node,
)
from route_resilience.train import _torch_load_checkpoint, get_device


BACKEND_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_ROOT.parent
PROJECT_ROOT = BACKEND_ROOT

def _resolve_checkpoint_file(filename: str) -> Path:
    for candidate in [BACKEND_ROOT / "checkpoints" / filename, REPO_ROOT / "checkpoints" / filename]:
        if candidate.exists():
            return candidate
    return BACKEND_ROOT / "checkpoints" / filename

CHECKPOINT_PATH = _resolve_checkpoint_file("best_unet_smp_resnet34_focaldice.pth")
THRESHOLD_ARTIFACT_PATH = _resolve_checkpoint_file("threshold_unet_smp_resnet34_focaldice.json")
DEFAULT_ANALYSIS_ROOT = BACKEND_ROOT / "outputs" / "analysis"
ALLOWED_TYPES = {"image/png", "image/jpeg", "image/jpg"}
MODEL_CACHE: Dict[str, Any] | None = None
MODEL_CACHE_LOCK = Lock()


def _analysis_root() -> Path:
    env_dir = os.environ.get("ROUTE_RESILIENCE_ANALYSIS_DIR")
    if env_dir:
        root = Path(env_dir)
    elif (BACKEND_ROOT / "outputs" / "analysis").exists():
        root = BACKEND_ROOT / "outputs" / "analysis"
    elif (REPO_ROOT / "outputs" / "analysis").exists():
        root = REPO_ROOT / "outputs" / "analysis"
    else:
        root = DEFAULT_ANALYSIS_ROOT
    root.mkdir(parents=True, exist_ok=True)
    return root


def _resolve_threshold(value: str | float, checkpoint: Dict[str, Any]) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        with open(value, encoding="utf-8") as handle:
            artifact = json.load(handle)
        if artifact.get("selection_split") != "validation":
            raise ValueError("Threshold artifact must state selection_split='validation'.")
        threshold = artifact.get("threshold")
        if artifact.get("config") != checkpoint.get("config"):
            raise ValueError("Threshold artifact configuration does not match the checkpoint.")

    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    return threshold


def _validate_graph_config(checkpoint: Dict[str, Any], graph_cfg) -> None:
    """Require checkpointed graph settings, with safe defaults for later fields.

    A legacy checkpoint can omit options added after it was produced only when
    the requested value is the current compatibility default.  This preserves
    the checkpoint's original extraction behaviour while rejecting both
    changed checkpointed settings and non-default new recovery settings.
    """
    saved = checkpoint.get("config", {}).get("graph")
    if saved is not None:
        current = asdict(graph_cfg)
        compatibility_defaults = asdict(type(graph_cfg)())
        missing_legacy_fields = set(compatibility_defaults) - {
            "max_gap_px", "min_gap_confidence", "min_edge_length_px"
        }
        mismatch = (
            any(key not in current or current[key] != value for key, value in saved.items())
            or any(
                key not in saved
                and (key not in missing_legacy_fields or value != compatibility_defaults[key])
                for key, value in current.items()
            )
        )
        if mismatch:
            raise ValueError(
                "Graph configuration differs from the checkpointed run; refusing an "
                "unverifiable real-image graph extraction."
            )


def _get_model_context() -> Dict[str, Any]:
    global MODEL_CACHE
    if MODEL_CACHE is not None:
        return MODEL_CACHE

    with MODEL_CACHE_LOCK:
        if MODEL_CACHE is not None:
            return MODEL_CACHE

        ckpt_file = Path(os.environ.get("ROUTE_RESILIENCE_CHECKPOINT", str(CHECKPOINT_PATH)))
        thresh_file = Path(os.environ.get("ROUTE_RESILIENCE_THRESHOLD", str(THRESHOLD_ARTIFACT_PATH)))
        device = get_device("cpu")

        # Peek at checkpoint config to support both ResNet-34 U-Net and SegFormer-B0
        raw_checkpoint = _torch_load_checkpoint(str(ckpt_file), device)
        saved_model_dict = (
            raw_checkpoint.get("config", {}).get("model", {})
            if isinstance(raw_checkpoint, dict)
            else {}
        )
        saved_train_dict = (
            raw_checkpoint.get("config", {}).get("train", {})
            if isinstance(raw_checkpoint, dict)
            else {}
        )

        cfg = Config()
        cfg.model.arch = saved_model_dict.get("arch", "unet_smp")
        cfg.model.encoder_name = saved_model_dict.get("encoder_name", "resnet34")
        cfg.model.encoder_weights = "imagenet"
        cfg.train.loss = saved_train_dict.get("loss", "focal_dice")

        model, checkpoint = load_model_for_inference(
            str(ckpt_file),
            cfg.model,
            cfg.train.loss,
            device,
        )
        _validate_graph_config(checkpoint, cfg.graph)
        threshold = _resolve_threshold(str(thresh_file), checkpoint)
        MODEL_CACHE = {
            "model": model,
            "checkpoint": checkpoint,
            "config": cfg,
            "device": device,
            "threshold": float(threshold),
        }
        return MODEL_CACHE


def _as_url(base_url: str, relative_path: str) -> str:
    return f"{base_url.rstrip('/')}/{relative_path.strip('/')}"


def _normalize_uploaded_image(file_bytes: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(file_bytes)) as image:
            image.load()
            rgb_image = image.convert("RGB")
    except Exception as exc:  # pragma: no cover - exercised via HTTP validation
        raise ValueError(f"Invalid or corrupt image: {exc}") from exc

    width, height = rgb_image.size
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive integers.")
    return np.asarray(rgb_image, dtype=np.uint8)


def _analysis_metadata_file(analysis_dir: Path) -> Path:
    return analysis_dir / "analysis_manifest.json"


def _build_artifact_map(base_url: str, analysis_dir: Path) -> Dict[str, str]:
    files = {
        "original": analysis_dir / "original.png",
        "probability_npy": analysis_dir / "probability.npy",
        "binary_mask": analysis_dir / "binary_mask.png",
        "skeleton": analysis_dir / "skeleton.png",
        "graph_json": analysis_dir / "graph.json",
        "healed_graph_json": analysis_dir / "healed_graph.json",
        "repaired_network": analysis_dir / "repaired_network.png",
        "visualization": analysis_dir / "visualization.png",
        "summary_json": analysis_dir / "summary.json",
        "criticality": analysis_dir / "criticality.json",
        "critical_roads": analysis_dir / "critical_roads.png",
    }
    artifact_map: Dict[str, str] = {}
    for key, path in files.items():
        if path.exists():
            artifact_map[key] = _as_url(base_url, f"analysis-files/{analysis_dir.name}/{path.name}")
    return artifact_map


def _load_healed_graph(graph_path: Path) -> nx.Graph:
    payload = json.loads(graph_path.read_text(encoding="utf-8"))
    graph = nx.Graph()
    for node in payload.get("nodes", []):
        graph.add_node(node["id"], **node.get("attributes", {}))
    for edge in payload.get("edges", []):
        graph.add_edge(edge["source"], edge["target"], **edge.get("attributes", {}))
    if "candidate_occlusions" in payload:
        graph.graph["candidate_occlusions"] = payload["candidate_occlusions"]
    return graph


def _empty_criticality_report() -> Dict[str, Any]:
    return {
        "graph_state": "G_healed",
        "weight": "length_px",
        "node_criticality": [],
        "edge_criticality": [],
        "articulation_points": [],
        "bridge_edges": [],
        "top_vulnerable_nodes_by_betweenness": [],
        "top_vulnerable_edges_by_betweenness": [],
        "top_vulnerable_edges_by_disruption_impact": [],
    }


def _save_critical_roads_artifact(
    analysis_dir: Path,
    graph: nx.Graph,
    report: Dict[str, Any],
) -> Path:
    original_path = analysis_dir / "original.png"
    if not original_path.exists():
        raise FileNotFoundError("Original image is missing for critical roads visualization.")

    canvas = Image.open(original_path).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    ranked_edges = report.get("edge_criticality", [])[:10]
    ranked_pairs = {
        frozenset(record.get("edge_id", []))
        for record in ranked_edges
        if len(record.get("edge_id", [])) == 2
    }

    for source, target, data in graph.edges(data=True):
        points = data.get("pts")
        if points is None:
            continue
        line = [(int(point[1]), int(point[0])) for point in np.asarray(points)]
        if len(line) < 2:
            continue
        is_critical = frozenset((source, target)) in ranked_pairs
        draw.line(
            line,
            fill=(255, 64, 64) if is_critical else (150, 150, 150),
            width=4 if is_critical else 1,
        )

    output_path = analysis_dir / "critical_roads.png"
    canvas.save(output_path)
    return output_path


def _persist_criticality_artifacts(analysis_dir: Path, graph: nx.Graph) -> Dict[str, Any]:
    report_path = analysis_dir / "criticality.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = _empty_criticality_report() if graph.number_of_edges() == 0 else criticality_report(
            graph,
            weight="length_px",
            top_n=10,
        )
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    _save_critical_roads_artifact(analysis_dir, graph, report)
    return report


def _disruption_state_file(analysis_dir: Path) -> Path:
    return analysis_dir / "disruption.json"


def _load_disruption_state(analysis_dir: Path) -> Dict[str, Any] | None:
    state_path = _disruption_state_file(analysis_dir)
    if not state_path.exists():
        return None
    return json.loads(state_path.read_text(encoding="utf-8"))


def _save_disruption_state(analysis_dir: Path, state: Dict[str, Any]) -> None:
    state_path = _disruption_state_file(analysis_dir)
    temporary_path = state_path.with_name(f".{state_path.name}.{uuid.uuid4().hex}.tmp")
    temporary_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary_path.replace(state_path)


class DisruptionRequest(BaseModel):
    strategy: str = Field(
        default="explicit",
        description="Strategy: 'explicit', 'targeted_betweenness', or 'random'",
    )
    target_type: str = Field(
        default="edge",
        description="Target asset type: 'edge' or 'node'",
    )
    failed_edges: list[list[Any]] | None = Field(
        default=None,
        description="List of explicit edges as [u, v] pairs",
    )
    failed_nodes: list[Any] | None = Field(
        default=None,
        description="List of explicit node identifiers",
    )
    count: int = Field(
        default=1,
        ge=1,
        description="Number of assets to fail for random or targeted strategies",
    )
    seed: int | None = Field(
        default=42,
        description="Seed for reproducible random failure",
    )
    weight: str = Field(
        default="length_px",
        description="Edge attribute to use as weight",
    )


class RouteRequest(BaseModel):
    origin: Any | None = Field(
        default=None,
        description="Optional origin node identifier; omitted endpoints are selected from the extracted graph.",
    )
    destination: Any | None = Field(
        default=None,
        description="Optional destination node identifier; omitted endpoints are selected from the extracted graph.",
    )
    weight: str = Field(
        default="length_px",
        description="Edge attribute to use as routing cost",
    )
    current_state: str = Field(
        default="baseline",
        description="Current graph state: 'baseline' or 'disrupted'",
    )
    failed_edges: list[list[Any]] | None = Field(
        default=None,
        description="Optional edges to remove before routing when current_state='disrupted'",
    )
    failed_nodes: list[Any] | None = Field(
        default=None,
        description="Optional nodes to remove before routing when current_state='disrupted'",
    )


class RecoveryRequest(BaseModel):
    disruption: dict[str, Any] | None = Field(
        default=None,
        description="Optional nested disruption payload with failed_edges/failed_nodes.",
    )
    failed_edges: list[list[Any]] | None = Field(
        default=None,
        description="Edges that were failed/removed in the disruption",
    )
    failed_nodes: list[Any] | None = Field(
        default=None,
        description="Nodes that were failed/removed in the disruption",
    )
    budget: int | None = Field(
        default=None,
        ge=1,
        description="Max number of assets to restore (None = restore all)",
    )
    ordering: str = Field(
        default="explicit_failure_order",
        description="Recovery ordering policy: 'explicit_failure_order' or 'criticality_greedy'",
    )
    weight: str = Field(
        default="length_px",
        description="Edge attribute to use as weight during resilience evaluation",
    )
    od_seed: int = Field(
        default=42,
        description="Seed for baseline OD pair sampling",
    )
    od_max_pairs: int = Field(
        default=100,
        ge=1,
        description="Max number of OD pairs to evaluate",
    )


def create_app() -> FastAPI:
    app = FastAPI(title="Route Resilience API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:8080",
            "http://127.0.0.1:8080",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    analysis_root = _analysis_root()
    app.mount("/analysis-files", StaticFiles(directory=str(analysis_root)), name="analysis-files")

    @app.post("/analyze")
    @app.post("/api/road-analysis")
    async def road_analysis(request: Request, file: UploadFile = File(...)) -> Dict[str, Any]:
        if file.filename is None:
            raise HTTPException(status_code=400, detail="Missing file name.")

        file_ext = Path(file.filename).suffix.lower()
        if file.content_type not in ALLOWED_TYPES and file_ext not in {".png", ".jpg", ".jpeg"}:
            raise HTTPException(status_code=415, detail="Unsupported file type. Only PNG, JPG, and JPEG are allowed.")

        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        try:
            normalized_rgb = _normalize_uploaded_image(file_bytes)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        analysis_id = uuid.uuid4().hex
        ACTIVE_ANALYSES.add(analysis_id)
        try:
            analysis_dir = _analysis_root() / analysis_id
            analysis_dir.mkdir(parents=True, exist_ok=True)

            original_path = analysis_dir / "original.png"
            original_img = Image.fromarray(np.asarray(normalized_rgb, dtype=np.uint8)).convert("RGB")
            original_img.save(original_path)

            context = _get_model_context()
            model = context["model"]
            device = context["device"]
            threshold = context["threshold"]

            try:
                probability_map = predict_probability_map(model, normalized_rgb, device)
                result = graph_from_probability_map(
                    probability_map, threshold, context["config"].graph, normalized_rgb
                )
                save_artifacts(result, normalized_rgb, analysis_dir)
                _persist_criticality_artifacts(analysis_dir, result["graph_after"])
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

            artifact_map = _build_artifact_map(str(request.base_url), analysis_dir)
            summary = json.loads((analysis_dir / "summary.json").read_text(encoding="utf-8"))
        finally:
            ACTIVE_ANALYSES.discard(analysis_id)

        manifest = {
            "analysis_id": analysis_id,
            "status": "completed",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input_image_name": file.filename,
            "input_image_ref": _as_url(str(request.base_url), f"analysis-files/{analysis_id}/original.png"),
            "artifacts": artifact_map,
            "summary": summary,
            "threshold_used": float(threshold),
        }
        _analysis_metadata_file(analysis_dir).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        return {
            "analysis_id": analysis_id,
            "status": "completed",
            "input_image_ref": manifest["input_image_ref"],
            "artifacts": artifact_map,
            "summary": summary,
            "threshold_used": float(threshold),
        }

    @app.get("/api/analyses")
    async def list_analyses(request: Request) -> Dict[str, Any]:
        root = _analysis_root()
        analyses = []
        if root.exists():
            for item in root.iterdir():
                if not item.is_dir():
                    continue
                manifest_file = item / "analysis_manifest.json"
                if manifest_file.exists():
                    try:
                        data = json.loads(manifest_file.read_text(encoding="utf-8"))
                        data["artifacts"] = _build_artifact_map(str(request.base_url), item)
                        data["input_image_ref"] = _as_url(str(request.base_url), f"analysis-files/{item.name}/original.png")
                        analyses.append(data)
                        continue
                    except Exception:
                        pass
                summary_file = item / "summary.json"
                if summary_file.exists():
                    try:
                        summary = json.loads(summary_file.read_text(encoding="utf-8"))
                        mtime = item.stat().st_mtime
                        created_at = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
                        analyses.append({
                            "analysis_id": item.name,
                            "status": "completed",
                            "created_at": created_at,
                            "input_image_name": f"Image {item.name[:8]}",
                            "input_image_ref": _as_url(str(request.base_url), f"analysis-files/{item.name}/original.png"),
                            "artifacts": _build_artifact_map(str(request.base_url), item),
                            "summary": summary,
                        })
                    except Exception:
                        pass
        analyses.sort(key=lambda a: a.get("created_at", ""), reverse=True)
        return {"analyses": analyses}

    @app.get("/api/analysis/{analysis_id}")
    async def get_analysis(request: Request, analysis_id: str) -> Dict[str, Any]:
        analysis_dir = _analysis_root() / analysis_id
        if not analysis_dir.exists():
            raise HTTPException(status_code=404, detail=f"Analysis '{analysis_id}' was not found.")

        manifest_path = _analysis_metadata_file(analysis_dir)
        if manifest_path.exists():
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["artifacts"] = _build_artifact_map(str(request.base_url), analysis_dir)
            return payload

        summary_path = analysis_dir / "summary.json"
        if not summary_path.exists():
            raise HTTPException(status_code=404, detail=f"No persisted artifacts found for analysis '{analysis_id}'.")

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return {
            "analysis_id": analysis_id,
            "status": "completed",
            "input_image_ref": str(analysis_dir / "original.png"),
            "artifacts": _build_artifact_map(str("http://localhost:8000"), analysis_dir),
            "summary": summary,
            "threshold_used": summary.get("threshold"),
        }

    @app.delete("/api/analysis/{analysis_id}")
    async def delete_analysis(analysis_id: str) -> Dict[str, Any]:
        if not analysis_id or "/" in analysis_id or "\\" in analysis_id or ".." in analysis_id:
            raise HTTPException(status_code=400, detail="Invalid analysis ID.")

        root = _analysis_root().resolve()
        analysis_dir = (_analysis_root() / analysis_id).resolve()
        if not analysis_dir.is_relative_to(root):
            raise HTTPException(status_code=400, detail="Invalid analysis ID.")

        if analysis_id in ACTIVE_ANALYSES:
            raise HTTPException(
                status_code=409,
                detail=f"Cannot delete analysis '{analysis_id}' while it is actively processing.",
            )

        if not analysis_dir.exists() or not analysis_dir.is_dir():
            raise HTTPException(status_code=404, detail=f"Analysis '{analysis_id}' was not found.")

        manifest_path = _analysis_metadata_file(analysis_dir)
        if manifest_path.exists():
            try:
                manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_data.get("status") in {"processing", "running", "in_progress"}:
                    raise HTTPException(
                        status_code=409,
                        detail=f"Cannot delete analysis '{analysis_id}' while it is still running.",
                    )
            except (json.JSONDecodeError, OSError):
                pass

        try:
            shutil.rmtree(analysis_dir)
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"Failed to delete analysis data: {exc}") from exc

        return {
            "status": "deleted",
            "analysis_id": analysis_id,
            "message": f"Analysis '{analysis_id}' and all associated files deleted successfully.",
        }

    @app.get("/api/analysis/{analysis_id}/criticality")
    async def get_criticality(request: Request, analysis_id: str) -> Dict[str, Any]:
        analysis_dir = _analysis_root() / analysis_id
        if not analysis_dir.exists():
            raise HTTPException(status_code=404, detail=f"Analysis '{analysis_id}' was not found.")

        graph_path = analysis_dir / "healed_graph.json"
        if not graph_path.exists():
            raise HTTPException(status_code=404, detail=f"No healed graph found for analysis '{analysis_id}'.")

        try:
            report = _persist_criticality_artifacts(analysis_dir, _load_healed_graph(graph_path))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Criticality analysis failed: {exc}") from exc

        return {
            "analysis_id": analysis_id,
            "status": "completed",
            "criticality": report,
            "artifact": _as_url(str(request.base_url), f"analysis-files/{analysis_id}/criticality.json"),
            "artifacts": _build_artifact_map(str(request.base_url), analysis_dir),
        }

    @app.get("/api/analysis/{analysis_id}/recovery-evidence")
    async def get_recovery_evidence(request: Request, analysis_id: str) -> Dict[str, Any]:
        analysis_dir = _analysis_root() / analysis_id
        if not analysis_dir.exists():
            raise HTTPException(status_code=404, detail=f"Analysis '{analysis_id}' was not found.")

        graph_path = analysis_dir / "healed_graph.json"
        if not graph_path.exists():
            raise HTTPException(status_code=404, detail=f"No healed graph found for analysis '{analysis_id}'.")

        data = json.loads(graph_path.read_text(encoding="utf-8"))
        inferred_connections = []
        for edge in data.get("edges", []):
            attr = edge.get("attributes", {})
            if attr.get("inferred") or attr.get("occlusion_recovered") or attr.get("healed"):
                inferred_connections.append({
                    "source": edge["source"],
                    "target": edge["target"],
                    "recovery_type": attr.get("recovery_type"),
                    "length_px": attr.get("length_px"),
                    "confidence": attr.get("confidence"),
                    "healed": attr.get("healed", False),
                    "inferred": attr.get("inferred", False),
                    "occlusion_recovered": attr.get("occlusion_recovered", False),
                    "occlusion_suspected": attr.get("occlusion_suspected", False),
                    "evidence": attr.get("evidence"),
                    "checks": attr.get("checks", []),
                    "acceptance_reason": attr.get("acceptance_reason"),
                })

        candidates = data.get("candidate_occlusions", [])
        return {
            "analysis_id": analysis_id,
            "status": "completed",
            "inferred_count": len(inferred_connections),
            "candidates_count": len(candidates),
            "inferred_connections": inferred_connections,
            "candidate_occlusions": candidates,
        }

    @app.post("/api/analysis/{analysis_id}/disrupt")
    async def simulate_disruption(
        request: Request,
        analysis_id: str,
        payload: DisruptionRequest,
    ) -> Dict[str, Any]:
        analysis_dir = _analysis_root() / analysis_id
        if not analysis_dir.exists():
            raise HTTPException(status_code=404, detail=f"Analysis '{analysis_id}' was not found.")

        graph_path = analysis_dir / "healed_graph.json"
        if not graph_path.exists():
            raise HTTPException(status_code=404, detail=f"No healed graph found for analysis '{analysis_id}'.")

        try:
            G_baseline = _load_healed_graph(graph_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to load baseline graph: {exc}") from exc

        if G_baseline.number_of_nodes() == 0:
            raise HTTPException(status_code=400, detail="Cannot run disruption on an empty road network.")

        strategy = payload.strategy.lower().strip()
        target_type = payload.target_type.lower().strip()
        if target_type not in {"edge", "node"}:
            raise HTTPException(status_code=400, detail=f"target_type must be 'edge' or 'node', got '{payload.target_type}'.")

        edges_to_remove: list[tuple[Any, Any]] = []
        nodes_to_remove: list[Any] = []

        if strategy == "explicit":
            if payload.failed_nodes:
                for n in payload.failed_nodes:
                    if not G_baseline.has_node(n):
                        try:
                            int_n = int(n)
                            if G_baseline.has_node(int_n):
                                n = int_n
                            else:
                                raise HTTPException(status_code=400, detail=f"Node '{n}' does not exist in graph.")
                        except (ValueError, TypeError):
                            raise HTTPException(status_code=400, detail=f"Node '{n}' does not exist in graph.")
                    nodes_to_remove.append(n)

            if payload.failed_edges:
                for edge in payload.failed_edges:
                    if len(edge) != 2:
                        raise HTTPException(status_code=400, detail=f"Invalid edge format: {edge}. Expected [u, v].")
                    u, v = edge[0], edge[1]
                    if not G_baseline.has_edge(u, v):
                        try:
                            iu, iv = int(u), int(v)
                            if G_baseline.has_edge(iu, iv):
                                u, v = iu, iv
                            else:
                                raise HTTPException(status_code=400, detail=f"Edge ({u}, {v}) does not exist in graph.")
                        except (ValueError, TypeError):
                            raise HTTPException(status_code=400, detail=f"Edge ({u}, {v}) does not exist in graph.")
                    edges_to_remove.append((u, v))

            if not nodes_to_remove and not edges_to_remove:
                raise HTTPException(status_code=400, detail="Explicit disruption strategy requires at least one item in 'failed_nodes' or 'failed_edges'.")

        elif strategy in {"targeted_betweenness", "targeted", "betweenness"}:
            selection_method = "top_node_betweenness" if target_type == "node" else "top_edge_betweenness"
            try:
                scenario = create_failure_scenario(
                    G_baseline,
                    scenario_id=f"targeted_{target_type}",
                    failure_type=target_type,
                    selection_method=selection_method,
                    count=payload.count,
                    seed=payload.seed,
                    weight=payload.weight,
                )
                nodes_to_remove = scenario["failed_nodes"]
                edges_to_remove = scenario["failed_edges"]
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        elif strategy in {"random", "random_uniform"}:
            try:
                scenario = create_failure_scenario(
                    G_baseline,
                    scenario_id=f"random_{target_type}",
                    failure_type=target_type,
                    selection_method="random_uniform",
                    count=payload.count,
                    seed=payload.seed,
                    weight=payload.weight,
                )
                nodes_to_remove = scenario["failed_nodes"]
                edges_to_remove = scenario["failed_edges"]
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported strategy '{payload.strategy}'. Supported strategies: 'explicit', 'targeted_betweenness', 'random'.")

        disruption_result = simulate_removal(
            G_baseline,
            edges_to_remove=edges_to_remove,
            nodes_to_remove=nodes_to_remove,
            weight=payload.weight,
        )
        G_disrupted: nx.Graph = disruption_result["surviving_graph"]

        od_pairs = select_baseline_od_pairs(G_baseline, max_pairs=100, seed=payload.seed or 42, weight=payload.weight)
        od_evaluation = evaluate_state_service(G_disrupted, od_pairs, weight=payload.weight) if od_pairs else None

        baseline_summary = {
            "n_nodes": G_baseline.number_of_nodes(),
            "n_edges": G_baseline.number_of_edges(),
            "n_components": disruption_result["before_components"],
            "connectivity_ratio": float(disruption_result.get("before_connectivity_ratio", 1.0)),
            "network_efficiency": float(disruption_result["before_efficiency"]),
        }
        disrupted_summary = {
            "n_nodes": G_disrupted.number_of_nodes(),
            "n_edges": G_disrupted.number_of_edges(),
            "n_components": disruption_result["after_components"],
            "connectivity_ratio": float(disruption_result.get("after_connectivity_ratio", 1.0)),
            "network_efficiency": float(disruption_result["after_efficiency"]),
        }

        affected_od_summary = None
        if od_evaluation:
            pair_records = od_evaluation.get("pair_records", [])
            severed_pairs = []
            detoured_pairs = []
            intact_pairs = []

            for p in pair_records:
                if not p["state"]["reachable"]:
                    severed_pairs.append({
                        "origin": p["origin"],
                        "destination": p["destination"],
                        "status": "disconnected",
                        "baseline_cost_px": p["baseline"]["cost_px"],
                        "disrupted_cost_px": None,
                        "cost_increase_px": None,
                        "detour_ratio": None,
                        "alternative_path": None,
                    })
                elif p["service_score"] < 1.0:
                    cost_inc = float(p["state"]["cost_px"] - p["baseline"]["cost_px"])
                    detour_r = float(p["state"]["cost_px"] / p["baseline"]["cost_px"]) if p["baseline"]["cost_px"] else 1.0
                    detoured_pairs.append({
                        "origin": p["origin"],
                        "destination": p["destination"],
                        "status": "detoured",
                        "baseline_cost_px": p["baseline"]["cost_px"],
                        "disrupted_cost_px": p["state"]["cost_px"],
                        "cost_increase_px": cost_inc,
                        "detour_ratio": detour_r,
                        "alternative_path": p["state"]["path"],
                    })
                else:
                    intact_pairs.append(p)

            cost_increases = [p["cost_increase_px"] for p in detoured_pairs if p["cost_increase_px"] is not None]
            detour_ratios = [p["detour_ratio"] for p in detoured_pairs if p["detour_ratio"] is not None]

            affected_od_summary = {
                "total_od_pairs_evaluated": len(pair_records),
                "baseline_service_performance": 1.0,
                "disrupted_service_performance": float(od_evaluation["performance"]),
                "performance_drop": float(1.0 - od_evaluation["performance"]),
                "severed_pairs_count": len(severed_pairs),
                "detoured_pairs_count": len(detoured_pairs),
                "intact_pairs_count": len(intact_pairs),
                "mean_cost_increase_px": float(np.mean(cost_increases)) if cost_increases else 0.0,
                "max_cost_increase_px": float(np.max(cost_increases)) if cost_increases else 0.0,
                "mean_detour_ratio": float(np.mean(detour_ratios)) if detour_ratios else 1.0,
                "max_detour_ratio": float(np.max(detour_ratios)) if detour_ratios else 1.0,
                "affected_pairs": detoured_pairs + severed_pairs,
            }

        response = {
            "analysis_id": analysis_id,
            "status": "completed",
            "scenario": {
                "strategy": strategy,
                "target_type": target_type,
                "failed_nodes": [int(n) if isinstance(n, (np.integer, int)) else str(n) for n in nodes_to_remove],
                "failed_edges": [
                    [int(u) if isinstance(u, (np.integer, int)) else str(u),
                     int(v) if isinstance(v, (np.integer, int)) else str(v)]
                    for u, v in edges_to_remove
                ],
            },
            "impact_metrics": {
                "efficiency_drop_frac": float(disruption_result["efficiency_drop_frac"]),
                "newly_disconnected_components": int(disruption_result["newly_disconnected"]),
                "before_efficiency": float(disruption_result["before_efficiency"]),
                "after_efficiency": float(disruption_result["after_efficiency"]),
                "before_connectivity_ratio": float(disruption_result.get("before_connectivity_ratio", 1.0)),
                "after_connectivity_ratio": float(disruption_result.get("after_connectivity_ratio", 1.0)),
                "connectivity_ratio_drop": float(disruption_result.get("connectivity_ratio_drop", 0.0)),
            },
            "network_state": {
                "baseline": baseline_summary,
                "disrupted": disrupted_summary,
            },
            "affected_od_service": affected_od_summary,
            "disrupted_graph": graph_to_jsonable(G_disrupted),
        }
        _save_disruption_state(analysis_dir, response)
        return response

    @app.get("/api/analysis/{analysis_id}/scenario")
    async def get_disruption_scenario(analysis_id: str) -> Dict[str, Any]:
        analysis_dir = _analysis_root() / analysis_id
        if not analysis_dir.exists():
            raise HTTPException(status_code=404, detail=f"Analysis '{analysis_id}' was not found.")
        try:
            disruption = _load_disruption_state(analysis_dir)
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=500, detail=f"Failed to load disruption scenario: {exc}") from exc
        return {"analysis_id": analysis_id, "disruption": disruption}

    @app.post("/api/analysis/{analysis_id}/route")
    async def route_query(
        request: Request,
        analysis_id: str,
        payload: RouteRequest,
    ) -> Dict[str, Any]:
        analysis_dir = _analysis_root() / analysis_id
        if not analysis_dir.exists():
            raise HTTPException(status_code=404, detail=f"Analysis '{analysis_id}' was not found.")

        graph_path = analysis_dir / "healed_graph.json"
        if not graph_path.exists():
            raise HTTPException(status_code=404, detail=f"No healed graph found for analysis '{analysis_id}'.")

        try:
            G_baseline = _load_healed_graph(graph_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to load graph: {exc}") from exc

        if G_baseline.number_of_nodes() == 0:
            raise HTTPException(status_code=400, detail="Cannot route on an empty road network.")

        def _coerce_node(n):
            if G_baseline.has_node(n):
                return n
            try:
                n_int = int(n)
                if G_baseline.has_node(n_int):
                    return n_int
            except (TypeError, ValueError):
                pass
            raise HTTPException(status_code=400, detail=f"Node '{n}' does not exist in the graph.")

        current_state = str(payload.current_state).lower().strip()
        if current_state not in {"baseline", "disrupted"}:
            raise HTTPException(status_code=400, detail="current_state must be 'baseline' or 'disrupted'.")

        if payload.origin is None and payload.destination is None:
            selected_pair = select_automatic_route_pair(G_baseline, weight=payload.weight)
            if selected_pair is None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "The extracted graph has no connected component with a traversable edge. "
                        "Inspect the road mask and graph diagnostics before routing."
                    ),
                )
            origin, destination = selected_pair
        elif payload.origin is None:
            destination = _coerce_node(payload.destination)
            origin = select_reachable_node(G_baseline, destination, weight=payload.weight)
            if origin is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Node '{destination}' has no other reachable node in the extracted graph.",
                )
        elif payload.destination is None:
            origin = _coerce_node(payload.origin)
            destination = select_reachable_node(G_baseline, origin, weight=payload.weight)
            if destination is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Node '{origin}' has no other reachable node in the extracted graph.",
                )
        else:
            origin = _coerce_node(payload.origin)
            destination = _coerce_node(payload.destination)

        baseline_route = route_result(G_baseline, origin, destination, weight=payload.weight)
        G_current = G_baseline.copy()
        failed_edges: list[tuple[Any, Any]] = []
        failed_nodes: list[Any] = []
        if current_state == "disrupted":
            edges_to_remove: list[tuple[Any, Any]] = []
            nodes_to_remove: list[Any] = []
            saved_disruption = None
            if not {"failed_edges", "failed_nodes"}.intersection(payload.model_fields_set):
                saved_disruption = _load_disruption_state(analysis_dir)
            saved_scenario = (saved_disruption or {}).get("scenario", {})
            failed_edges = payload.failed_edges if payload.failed_edges is not None else saved_scenario.get("failed_edges", [])
            failed_nodes = payload.failed_nodes if payload.failed_nodes is not None else saved_scenario.get("failed_nodes", [])
            for edge in failed_edges:
                if len(edge) != 2:
                    raise HTTPException(status_code=400, detail=f"Invalid edge format: {edge}")
                u, v = edge[0], edge[1]
                if not G_baseline.has_edge(u, v):
                    try:
                        iu, iv = int(u), int(v)
                        if G_baseline.has_edge(iu, iv):
                            u, v = iu, iv
                        else:
                            raise HTTPException(status_code=400, detail=f"Edge ({u}, {v}) does not exist in graph.")
                    except (TypeError, ValueError):
                        raise HTTPException(status_code=400, detail=f"Edge ({u}, {v}) does not exist in graph.")
                edges_to_remove.append((u, v))
            failed_edges = edges_to_remove
            for n in failed_nodes:
                nodes_to_remove.append(_coerce_node(n))
            G_current.remove_edges_from(edges_to_remove)
            G_current.remove_nodes_from(nodes_to_remove)

        current_route = route_result(G_current, origin, destination, weight=payload.weight)
        detour_ratio = None
        if (
            baseline_route["reachable"]
            and current_route["reachable"]
            and baseline_route["cost_px"] is not None
            and current_route["cost_px"] is not None
            and baseline_route["cost_px"] > 0
        ):
            detour_ratio = float(current_route["cost_px"] / baseline_route["cost_px"])

        route_status = (
            "no_original_route"
            if not baseline_route["reachable"]
            else "no_alternative_route"
            if current_state == "disrupted" and not current_route["reachable"]
            else "alternative_available"
            if current_state == "disrupted"
            else "original_route_available"
        )
        extraction_warnings = []
        summary_path = analysis_dir / "summary.json"
        if summary_path.exists():
            try:
                extraction_warnings = json.loads(
                    summary_path.read_text(encoding="utf-8")
                ).get("warnings", [])
            except (OSError, json.JSONDecodeError) as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to load extraction diagnostics: {exc}",
                ) from exc
        route_reason = (
            "selected_nodes_are_in_separate_extracted_components"
            if not baseline_route["reachable"]
            else "failed_assets_disconnect_the_selected_pair"
            if current_state == "disrupted" and not current_route["reachable"]
            else None
        )
        if route_reason == "selected_nodes_are_in_separate_extracted_components":
            extraction_warnings = [
                *extraction_warnings,
                "The selected nodes lie in separate extracted graph components. "
                "This may be a genuine network disconnection or a segmentation omission; "
                "verify the image and extracted overlay.",
            ]
        elif route_reason == "failed_assets_disconnect_the_selected_pair":
            extraction_warnings = [
                *extraction_warnings,
                "The selected failed assets disconnect this origin-destination pair; "
                "no replacement road was inferred.",
            ]
        extra_distance = None
        if (
            current_state == "disrupted"
            and baseline_route["cost_px"] is not None
            and current_route["cost_px"] is not None
        ):
            extra_distance = float(current_route["cost_px"] - baseline_route["cost_px"])

        return {
            "analysis_id": analysis_id,
            "status": "completed",
            "origin": origin,
            "destination": destination,
            "current_state": current_state,
            "route_status": route_status,
            "route_reason": route_reason,
            "warnings": extraction_warnings,
            "route_exists": current_route["reachable"],
            "route": current_route["path"] if current_route["reachable"] else [],
            "original_route": baseline_route["path"] if baseline_route["reachable"] else [],
            "alternative_route": (
                current_route["path"]
                if current_state == "disrupted" and current_route["reachable"]
                else []
            ),
            "blocked_edges": failed_edges,
            "blocked_nodes": failed_nodes,
            "route_length": current_route["cost_px"],
            "original_route_length": baseline_route["cost_px"],
            "alternative_route_length": (
                current_route["cost_px"]
                if current_state == "disrupted" and current_route["reachable"]
                else None
            ),
            "extra_distance": extra_distance,
            "baseline_route_length": baseline_route["cost_px"],
            "disrupted_route_length": current_route["cost_px"] if current_state == "disrupted" else baseline_route["cost_px"],
            "detour_ratio": detour_ratio,
            "baseline_vs_disrupted": {
                "baseline": baseline_route["cost_px"],
                "disrupted": current_route["cost_px"],
            },
        }

    @app.post("/api/analysis/{analysis_id}/recovery")
    async def run_recovery(
        request: Request,
        analysis_id: str,
        payload: RecoveryRequest,
    ) -> Dict[str, Any]:
        analysis_dir = _analysis_root() / analysis_id
        if not analysis_dir.exists():
            raise HTTPException(status_code=404, detail=f"Analysis '{analysis_id}' was not found.")

        graph_path = analysis_dir / "healed_graph.json"
        if not graph_path.exists():
            raise HTTPException(status_code=404, detail=f"No healed graph found for analysis '{analysis_id}'.")

        try:
            G_baseline = _load_healed_graph(graph_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to load graph: {exc}") from exc

        if G_baseline.number_of_nodes() == 0:
            raise HTTPException(status_code=400, detail="Cannot run recovery on an empty road network.")

        if payload.ordering not in {"explicit_failure_order", "criticality_greedy"}:
            raise HTTPException(status_code=400, detail=f"ordering must be 'explicit_failure_order' or 'criticality_greedy', got '{payload.ordering}'.")

        disruption_payload = payload.disruption or {}
        failed_edges = [list(edge) for edge in (payload.failed_edges or disruption_payload.get("failed_edges") or [])]
        failed_nodes = list(payload.failed_nodes or disruption_payload.get("failed_nodes") or [])
        if not failed_edges and not failed_nodes:
            if not {"failed_edges", "failed_nodes", "disruption"}.intersection(payload.model_fields_set):
                saved_disruption = _load_disruption_state(analysis_dir)
                saved_scenario = (saved_disruption or {}).get("scenario", {})
                failed_edges = [list(edge) for edge in saved_scenario.get("failed_edges", [])]
                failed_nodes = list(saved_scenario.get("failed_nodes", []))
        if not failed_edges and not failed_nodes:
            raise HTTPException(status_code=400, detail="Recovery requires at least one failed_edge or failed_node.")

        failed_edges_parsed: list[tuple] = []
        for edge in failed_edges:
            if len(edge) != 2:
                raise HTTPException(status_code=400, detail=f"Invalid edge format: {edge}")
            u, v = edge[0], edge[1]
            if not G_baseline.has_edge(u, v):
                try:
                    iu, iv = int(u), int(v)
                    if G_baseline.has_edge(iu, iv):
                        u, v = iu, iv
                    else:
                        raise HTTPException(status_code=400, detail=f"Edge ({u}, {v}) does not exist in baseline graph.")
                except (ValueError, TypeError):
                    raise HTTPException(status_code=400, detail=f"Edge ({u}, {v}) does not exist in baseline graph.")
            failed_edges_parsed.append((u, v))

        failed_nodes_parsed: list[Any] = []
        for n in failed_nodes:
            if not G_baseline.has_node(n):
                try:
                    n_int = int(n)
                    if G_baseline.has_node(n_int):
                        n = n_int
                    else:
                        raise HTTPException(status_code=400, detail=f"Node '{n}' does not exist in baseline graph.")
                except (ValueError, TypeError):
                    raise HTTPException(status_code=400, detail=f"Node '{n}' does not exist in baseline graph.")
            failed_nodes_parsed.append(n)

        disruption_result = simulate_removal(
            G_baseline,
            edges_to_remove=failed_edges_parsed,
            nodes_to_remove=failed_nodes_parsed,
            weight=payload.weight,
        )
        G_disrupted: nx.Graph = disruption_result["surviving_graph"]

        od_pairs = select_baseline_od_pairs(
            G_baseline,
            max_pairs=payload.od_max_pairs,
            seed=payload.od_seed,
            weight=payload.weight,
        )

        try:
            recovery_result = reinstate_baseline_assets(
                G_baseline,
                G_disrupted,
                failed_nodes=failed_nodes_parsed,
                failed_edges=failed_edges_parsed,
                budget=payload.budget,
                ordering=payload.ordering,
                weight=payload.weight,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        G_recovered: nx.Graph = recovery_result["recovered_graph"]

        if payload.budget is None:
            budget_used = len(failed_edges_parsed) + len(failed_nodes_parsed)
        else:
            budget_used = min(payload.budget, len(failed_edges_parsed) + len(failed_nodes_parsed))

        restoration_steps = []
        H_step = G_disrupted.copy()
        assets_ordered = [("node", n) for n in failed_nodes_parsed] + [("edge", e) for e in failed_edges_parsed]
        for step_index, (asset_type, asset) in enumerate(assets_ordered[:budget_used], start=1):
            if asset_type == "node":
                if G_baseline.has_node(asset):
                    H_step.add_node(asset, **G_baseline.nodes[asset])
                    for neighbor, data in G_baseline[asset].items():
                        if H_step.has_node(neighbor):
                            H_step.add_edge(asset, neighbor, **data)
            else:
                u, v = asset
                if G_baseline.has_edge(u, v) and H_step.has_node(u) and H_step.has_node(v):
                    H_step.add_edge(u, v, **G_baseline.edges[u, v])
            step_summary = graph_summary(H_step)
            step_summary["efficiency"] = float(network_efficiency(H_step, weight=payload.weight))
            restoration_steps.append({
                "step": step_index,
                "restored_asset_type": asset_type,
                "restored_asset": (
                    [int(asset[0]) if isinstance(asset[0], (np.integer, int)) else str(asset[0]),
                     int(asset[1]) if isinstance(asset[1], (np.integer, int)) else str(asset[1])]
                    if asset_type == "edge" else
                    (int(asset) if isinstance(asset, (np.integer, int)) else str(asset))
                ),
                "connectivity": {
                    "n_nodes": int(step_summary.get("n_nodes", H_step.number_of_nodes())),
                    "n_edges": int(step_summary.get("n_edges", H_step.number_of_edges())),
                    "n_components": int(step_summary.get("n_components", nx.number_connected_components(H_step))),
                    "is_connected": bool(step_summary.get("is_connected", nx.is_connected(H_step) if H_step.number_of_nodes() > 1 else True)),
                },
                **step_summary,
            })

        resilience = evaluate_resilience(G_baseline, G_disrupted, G_recovered, od_pairs, weight=payload.weight)
        resilience.pop("od_routes", None)

        def _safe_int_or_str(v):
            return int(v) if isinstance(v, (np.integer, int)) else str(v)

        return {
            "analysis_id": analysis_id,
            "status": "completed",
            "scenario": {
                "failed_edges": [[_safe_int_or_str(u), _safe_int_or_str(v)] for u, v in failed_edges_parsed],
                "failed_nodes": [_safe_int_or_str(n) for n in failed_nodes_parsed],
                "ordering": payload.ordering,
                "budget": recovery_result["budget"],
            },
            "disrupted_state": {
                **graph_summary(G_disrupted),
                "efficiency": float(disruption_result["after_efficiency"]),
                "efficiency_drop_frac": float(disruption_result["efficiency_drop_frac"]),
            },
            "restoration_sequence": restoration_steps,
            "connectivity_at_each_step": [{
                "step": step["step"],
                "is_connected": step["connectivity"]["is_connected"],
                "n_components": step["connectivity"]["n_components"],
                "n_nodes": step["connectivity"]["n_nodes"],
                "n_edges": step["connectivity"]["n_edges"],
            } for step in restoration_steps],
            "recovery_summary": {
                "recovered_nodes": [_safe_int_or_str(n) for n in recovery_result["recovered_nodes"]],
                "recovered_edges": [[_safe_int_or_str(u), _safe_int_or_str(v)] for u, v in recovery_result["recovered_edges"]],
                "assets_restored": len(recovery_result["recovered_nodes"]) + len(recovery_result["recovered_edges"]),
            },
            "final_recovered_state": {
                **graph_summary(G_recovered),
                "efficiency": float(network_efficiency(G_recovered, weight=payload.weight)),
            },
            "recovered_state": {
                **graph_summary(G_recovered),
                "efficiency": float(network_efficiency(G_recovered, weight=payload.weight)),
            },
            "resilience_metrics": resilience,
        }

    @app.get("/")
    @app.get("/health")
    async def health() -> Dict[str, str]:
        return {"status": "ok", "service": "Route Resilience API", "version": "1.0.0"}

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
