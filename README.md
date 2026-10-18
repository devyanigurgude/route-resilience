# Route Resilience: Satellite Road Extraction & Topology Resilience Platform

Route Resilience is an autonomous deep learning and network science framework for extracting road topologies from satellite imagery, reconstructing severed corridors through heavy tree canopy and cloud occlusion, simulating infrastructure disruptions, calculating alternative detour routes, and providing explainable continuity evidence.

---

## Table of Contents
1. [Overview & Architecture](#overview--architecture)
2. [Repository Structure](#repository-structure)
3. [Dataset Provenance & Spatial Audit](#dataset-provenance--spatial-audit)
4. [Models & Training Strategy](#models--training-strategy)
5. [Graph Extraction & Occlusion Recovery Pipeline](#graph-extraction--occlusion-recovery-pipeline)
6. [Resilience Analytics, Disruption & Rerouting](#resilience-analytics-disruption--rerouting)
7. [Explainable Recovery Evidence](#explainable-recovery-evidence)
8. [Setup & Installation](#setup--installation)
9. [API Reference](#api-reference)
10. [Testing & Verification](#testing--verification)
11. [Empirical Results & Benchmark Metrics](#empirical-results--benchmark-metrics)
12. [Known Limitations & Operating Bounds](#known-limitations--operating-bounds)

---

## Overview & Architecture

Satellite-derived road networks frequently suffer from fragmentation caused by shadows, tree canopy, cloud cover, and sensor occlusions. Traditional nearest-neighbor heuristics often bridge unrelated roads, while standard segmentation models drop continuity under low contrast.

Route Resilience addresses this with a multi-stage, evidence-gated system:
1. **Perception**: U-Net (ResNet-34 encoder) and SegFormer-B0 models segment road corridors with synthetic occlusion augmentation.
2. **Topological Extraction**: Morphological thinning and coordinate-preserving skeletonization convert segmentation probability maps into NetworkX multigraphs.
3. **Evidence-Gated Recovery**: Conservative geometric and photometric checks reconnect severed corridors across small canopy gaps and large cloud-covered spans without creating spurious bridges.
4. **Resilience & Disruption Simulation**: Simulates edge/node removals (random, explicit, or targeted by betweenness centrality) and evaluates global network efficiency and origin-destination service degradation.
5. **Interactive Scientific Frontend**: A React 19 application providing interactive network inspection, disruption simulations, dynamic detour routing, and explainable evidence panels.

```
+---------------------------------------------------------------------------------------+
|                                    INPUT SATELLITE IMAGE                              |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|  PERCEPTION: Tiled Inference (256x256 stride) -> Probability Map P(x, y)              |
|  - U-Net (ResNet-34 SMP) / SegFormer-B0                                               |
|  - Validation-Selected Threshold Gating                                               |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|  TOPOLOGY: Morphological Skeletonization (Zhang-Suen) -> Initial Road Graph G_0       |
|  - Distance transform width estimation                                                |
|  - Stable pixel-space node coordinate mapping                                         |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|  CONTINUITY RECOVERY: Multi-Stage Evidence Gating                                     |
|  - Stage 1: Standard Evidence Gating (gap <= 15px, mean P >= 0.15)                    |
|  - Stage 2: Canopy & Shadow Recovery (gap <= 30px, mutual heading >= 0.60, offset)    |
|  - Stage 3: Cloud Continuation (gap <= 240px, Bézier curve, cloud mask coverage)      |
|  - Candidate Rejection Logging: records exact criteria for rejected bridges           |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|  NETWORK SCIENCE & RESILIENCE ANALYTICS                                               |
|  - Global Efficiency E(G) = [1 / N(N-1)] * Sum[1 / d(u, v)]                           |
|  - Failure Simulations: Explicit, Targeted Betweenness, Random                        |
|  - Origin-Destination Service Availability & Detour Length Deltas                     |
|  - Resilience Index RI = (P_Disrupted + P_Recovered) / 2                              |
+---------------------------------------------------------------------------------------+
```

---

## Repository Structure

The repository is organized into a decoupled `backend/` and `frontend/` architecture:

```text
route_resilience/
├── backend/                              # Core Python backend & ML services
│   ├── route_resilience/                 # Python package
│   │   ├── config.py                     # Central configuration dataclasses
│   │   ├── models.py                     # U-Net, SegFormer-B0, and MC-Dropout models
│   │   ├── losses.py                     # Focal-Dice and BCE-Dice loss implementations
│   │   ├── data.py                       # Dataset loading, caching, and augmentations
│   │   ├── graph_extraction.py           # Skeleton extraction, gap healing & cloud recovery
│   │   ├── real_graph_pipeline.py        # Tiled inference, stitching, and artifact generation
│   │   ├── criticality.py                # Betweenness centrality, articulation points, efficiency
│   │   ├── disruption.py                 # Network degradation & failure simulation engine
│   │   ├── routing.py                    # Dijkstra/A* routing, detour detection, OD sampling
│   │   ├── recovery.py                   # Asset restoration and resilience calculations
│   │   ├── resilience.py                 # Multi-state service evaluation and resilience scoring
│   │   └── train.py                      # Training loop with early stopping & checkpointing
│   ├── checkpoints/                      # Trained model weights and threshold artifacts
│   │   ├── best_unet_smp_resnet34_focaldice.pth       # Primary ResNet-34 U-Net weights (293 MB)
│   │   ├── threshold_unet_smp_resnet34_focaldice.json # Validation threshold artifact (~0.50)
│   │   ├── best_segformer_mit_b0_focaldice.pth        # SegFormer-B0 weights (44 MB)
│   │   ├── threshold_segformer_mit_b0_focaldice.json  # SegFormer threshold artifact
│   │   └── results_unet_smp_resnet34.json             # Benchmark evaluation metrics
│   ├── api_server.py                     # FastAPI REST API application
│   ├── scripts/                          # Training, ablation, and evaluation CLI tools
│   ├── tests/                            # 50+ deterministic unit and integration tests
│   ├── outputs/                          # Generated analysis outputs and logs (git-ignored)
│   ├── requirements.txt                  # Backend Python dependencies
│   └── pytest.ini                        # Backend-local Pytest configuration
│
├── frontend/                             # React 19 + TanStack Start web application
│   ├── src/
│   │   ├── routes/                       # Application pages:
│   │   │   ├── index.tsx                 # Landing presentation and system overview
│   │   │   ├── explore.tsx               # Satellite image upload, inference & history
│   │   │   ├── network.tsx               # Interactive topology, node inspection & evidence
│   │   │   ├── disruption.tsx            # Targeted failure simulations & efficiency drops
│   │   │   ├── routing.tsx               # Origin-Destination pathfinding & detour analysis
│   │   │   ├── recovery.tsx              # Infrastructure restoration & resilience scoring
│   │   │   └── __root.tsx                # App shell, theme provider, and metadata
│   │   ├── components/                   # Scientific UI components, map overlays, navigation
│   │   └── lib/                          # API client (`backend.ts`), theme context, types
│   ├── public/
│   │   ├── favicon.svg                   # Custom Route Resilience vector icon
│   │   ├── favicon.ico                   # Multi-resolution favicon icon (16, 32, 48, 64px)
│   │   └── ...
│   ├── package.json
│   ├── vite.config.ts
│   └── tailwind.config.ts
│
├── RouteResilience/                      # Benchmark satellite dataset (images_png, masks_png)
├── requirements.txt                      # Root requirements pointer (-r backend/requirements.txt)
├── pytest.ini                            # Root Pytest configuration
├── .gitignore                            # Excludes large weights, images, and runtime caches
├── AGENTS.md                             # Agent development and Git safety guidelines
└── README.md                             # Comprehensive project documentation
```

---

## Dataset Provenance & Spatial Audit

The benchmark dataset is located in `RouteResilience/`:
- **Imagery**: 5,637 satellite RGB tiles (`RouteResilience/images_png/`), 256x256 pixels each.
- **Labels**: 5,637 binary road masks (`RouteResilience/masks_png/`), where 1 represents road centerline/corridor and 0 represents background.
- **Data Splits**:
  - `data_split.csv`: Baseline random split dividing tiles into train, validation, and test sets.
  - `data_split_grouped.csv` & `create_group_split.py`: A spatially aware grouped split developed to prevent data leakage from spatial autocorrelation (neighboring geographic tiles falling into both train and test splits).
  - `phase7_selected_tiles.csv`: 60 held-out real satellite test tiles representing challenging environments (heavy tree canopy, bridges, urban corridors, cloud/shadow gaps, and rural roads) used for continuity recovery benchmarking.

---

## Models & Training Strategy

### 1. Model Architectures
- **Primary Model (`unet_smp`)**: U-Net architecture with a ResNet-34 encoder pretrained on ImageNet, implemented via `segmentation_models_pytorch`. Produces single-channel road logits.
- **Transformer Alternative (`segformer`)**: SegFormer architecture with an `mit_b0` hierarchical transformer encoder for lightweight cross-scale context.
- **Uncertainty Model (`unet` with MC-Dropout)**: Custom U-Net with spatial dropout ($p=0.20$) enabled during inference to estimate epistemic uncertainty across Monte Carlo passes.

### 2. Loss Functions & Optimization
- **Focal-Dice Loss**: Combines Focal Loss (handling extreme road vs. non-road pixel imbalance, $\alpha=0.75, \gamma=2.0$) with soft Dice Loss to maximize intersection over union:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Focal}} + \mathcal{L}_{\text{Dice}}$$
- **Synthetic Occlusion Augmentation**: During training, coarse dropout randomly injects up to 6 rectangular occlusion holes (up to 15% of tile dimension) into training images to force the model to rely on road context and partial geometry.
- **Validation-Selected Threshold**: The decision threshold is tuned explicitly on the validation split (`threshold ≈ 0.50`), preventing arbitrary 0.5 cutoffs from skewing topological extraction.

---

## Graph Extraction & Occlusion Recovery Pipeline

Extracting a valid graph from a raster prediction requires strict guards against topological hallucination:

1. **Noise Cleaning**: `clean_road_mask` eliminates isolated speckles smaller than `min_edge_length_px` while preserving connected road components.
2. **Skeleton Thinning**: Zhang-Suen morphological skeletonization reduces binary road masks to 1-pixel-wide topological medial axes.
3. **Graph Construction**: Sknw builds a planar graph where nodes represent junctions (degree $\ge 3$) or endpoints (degree 1), and edges carry pixel polylines (`pts`) and Euclidean pixel lengths (`length_px`).
4. **Evidence-Gated Gap Healing (`confidence_gated_gap_healing`)**:
   - **Standard Gaps ($\le 15\text{ px}$)**: Facing degree-1 endpoints are joined only if the linear path across the gap maintains sustained road probability ($P \ge 0.15$).
   - **Canopy & Shadow Recovery ($\le 30\text{ px}$)**: Facing endpoints must satisfy:
     - Heading alignment: $\text{dot}_a \ge 0.70$ and $\text{dot}_b \ge 0.70$.
     - Mutual tangent alignment: $\text{tangent}_a \cdot (-\text{tangent}_b) \ge 0.60$.
     - Lateral offset: $\le 4.0\text{ px}$.
     - Minimum trunk length: $\ge 6.0\text{ px}$.
     - Acyclicity: endpoints must belong to disconnected components (rejecting shortcut cycles).
     - Photometric contrast: path flank contrast distinguishes roads from uniform dark patches.
   - **Cloud Occlusion Recovery ($\le 240\text{ px}$)**: Bridges long obscured spans:
     - **Cloud Mask Gate**: Requires an image-derived cloud mask (`cloud_occlusion_mask` with luminance $\ge 0.68$, saturation $\le 0.30$, and edge dilation) covering $\ge 55\%$ of the proposed continuation.
     - **Tangent Continuation Curve**: Fits a cubic Bézier curve constrained by endpoint outward tangents.
     - **Distance-Adaptive Checks**: Mutual heading threshold adapts from $0.85$ (short gaps) down to $0.60$ (long gaps permitting gentle bends); lateral offset limit scales adaptively ($4.0 + \text{dist} \times 0.20$).
     - **Sinuosity Guard**: Limits curve path length over straight chord distance ($\le 1.35$).
5. **Rejection Audit**: Candidates that fail any continuity check (degree constraints, cycle prevention, flank contrast, or sinuosity) are recorded in `graph.graph["candidate_occlusions"]` with exact rejection reasons.

---

## Resilience Analytics, Disruption & Rerouting

Route Resilience models road networks as weighted graphs $G = (V, E)$ where edge weights represent Euclidean pixel distances $d(u, v)$:

### 1. Global Network Efficiency
Network efficiency measures how effectively information or traffic flows across all node pairs:
$$E(G) = \frac{1}{N(N-1)} \sum_{i \ne j \in V} \frac{1}{d(i, j)}$$
When a node pair $(i, j)$ is disconnected, $d(i, j) = \infty$ and $1/d(i, j) = 0$.

### 2. Disruption Simulations (`/api/analysis/{id}/disrupt`)
Supports three disruption strategies:
- **`explicit`**: Specific user-selected edges or nodes are removed from the network.
- **`betweenness_edges` / `betweenness_nodes`**: Targeted removal of the highest edge/node betweenness centrality lifelines.
- **`random`**: Stochastic failure of edges/nodes parameterized by seed and removal count.

Evaluates efficiency drop fraction $\Delta E = (E_{\text{base}} - E_{\text{disrupt}}) / E_{\text{base}}$, component fragmentation, and severed origin-destination (OD) pairs.

### 3. Alternative Rerouting (`/api/analysis/{id}/route`)
Computes baseline shortest routes and evaluates detour paths when segments are blocked:
- Identifies whether the destination remains reachable.
- Calculates detour penalty: $\Delta d = d_{\text{alternative}} - d_{\text{original}}$.
- Reports affected intermediate nodes and diagnostics for unreachable components.

### 4. Recovery & Resilience Scoring (`/api/analysis/{id}/recovery`)
Simulates phased infrastructure reconstruction:
- Evaluates restored network efficiency after reinstating critical edges.
- Calculates the integrated Resilience Index:
  $$RI = \frac{P_{\text{Disrupted}} + P_{\text{Recovered}}}{2}$$
  where $P$ is normalized origin-destination service performance.

---

## Explainable Recovery Evidence

When inspecting the network in the frontend or querying `GET /api/analysis/{id}/recovery-evidence`, every inferred road connection provides verifiable algorithmic evidence:

- **Stable Endpoint IDs**: Permanent graph node IDs for endpoints.
- **Geometric Metrics**: Gap length in pixels (`gap_px`), tangent alignment angles (`dot_a`, `dot_b`, `heading_alignment`), mutual heading, lateral offset, and path sinuosity.
- **Photometric & Spatial Evidence**: Mean segmentation probability along corridor, cloud mask coverage fraction, and flank contrast.
- **Algorithm Checklist**: Real-time breakdown of every gate evaluated by the recovery engine:
  - `Gap Distance` (pass/fail against gap limit)
  - `Endpoint Heading Alignment` (pass/fail against tangent threshold)
  - `Mutual Tangent Alignment` (pass/fail against adaptive limit)
  - `Lateral Offset` (pass/fail against adaptive limit)
  - `Cloud Mask Coverage` (pass/fail against cloud threshold)
  - `Path Sinuosity Guard` (pass/fail against sinuosity limit)
  - `Acyclicity & Degree` (verifies degree-1 tips without prior alternate path)
- **Rejection Diagnostics**: For rejected candidates, displays the exact failure condition (e.g., `Flank contrast below limit`, `A* search reached max depth without continuous path`, or `Cycle prevention: alternate path already exists`).

---

## Setup & Installation

### Prerequisites
- **Python**: 3.10 or higher
- **Node.js**: 18+ and **npm**

### 1. Backend Installation & Startup
```bash
# Clone the repository
git clone <repository-url>
cd route_resilience

# Install Python dependencies (from root or backend/)
pip install -r requirements.txt

# Start the FastAPI backend server (port 8000)
# Option A: From repository root
python -m uvicorn backend.api_server:app --reload --host 127.0.0.1 --port 8000

# Option B: From backend directory
cd backend
python api_server.py
```
Verify the server is running:
```bash
curl http://127.0.0.1:8000/health
# Response: {"status": "ok", "service": "Route Resilience API", "version": "1.0.0"}
```

### 2. Frontend Installation & Startup
```bash
# Navigate to frontend directory
cd frontend

# Install npm dependencies
npm install

# Start Vite development server (port 5173)
npm run dev
```
Open your browser at `http://localhost:5173`.

### 3. Production Frontend Build
```bash
cd frontend
npm run build
```
Emits optimized client assets (`dist/client/`) and SSR bundles (`dist/server/`).

---

## API Reference

The FastAPI backend exposes the following endpoints:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` & `/health` | API service health check |
| `POST` | `/analyze` / `/api/road-analysis` | Upload satellite image (PNG/JPEG), runs U-Net inference, extracts graph, performs evidence healing, and persists artifacts |
| `GET` | `/api/analyses` | List previous analysis sessions with metadata and thumbnails |
| `GET` | `/api/analysis/{id}` | Retrieve analysis summary, graph metrics, and file references |
| `GET` | `/api/analysis/{id}/healed-graph` | Fetch complete graph JSON (`nodes`, `edges`, `candidate_occlusions`) |
| `GET` | `/api/analysis/{id}/criticality` | Compute and return betweenness centrality, articulation points, and critical lifelines |
| `POST` | `/api/analysis/{id}/disrupt` | Simulate edge/node failure scenario (`explicit`, `betweenness`, `random`) and return efficiency impact |
| `POST` | `/api/analysis/{id}/route` | Calculate original and detour routes between OD pairs around blocked segments |
| `POST` | `/api/analysis/{id}/recovery` | Restore baseline edges and compute resilience recovery metrics |
| `GET` | `/api/analysis/{id}/recovery-evidence` | Return structured continuity evidence and check evaluations for inferred roads and rejected candidates |
| `GET` | `/analysis-files/{id}/{filename}` | Static file endpoint serving original images, masks, overlays, and graph artifacts |

---

## Testing & Verification

The repository includes over 50 automated tests covering unit logic, graph algorithms, and end-to-end API workflows.

```bash
# Run the complete test suite from repository root:
pytest

# Run specific test suites:
pytest backend/tests/test_recovery_evidence.py -v
pytest backend/tests/test_api_server.py -v
pytest backend/tests/test_occlusion_aware_recovery.py -v
pytest backend/tests/test_reliable_recovery.py -v

# Or run from inside backend/:
cd backend
pytest -v
```

### Standalone End-to-End Verification Script
```bash
python backend/verify_core_pipeline.py
```
Loads a real satellite tile (`RouteResilience/images_png/1.png`), executes model inference, extracts the skeleton, runs evidence gap healing, and verifies graph connectivity.

---

## Empirical Results & Benchmark Metrics

Documented evaluation metrics from model training checkpoints (`backend/checkpoints/results_unet_smp_resnet34.json`):

| Metric | ResNet-34 U-Net Value | Description |
| :--- | :--- | :--- |
| **Validation Loss** | `0.490` | Focal-Dice loss on validation split |
| **Test IoU** | `0.289` | Intersection over Union on held-out test split |
| **Test Dice Score** | `0.449` | F1 / Dice score on held-out test split |
| **Model Size** | `293.6 MB` | Checkpoint weights (`.pth`) |
| **Inference Time (CPU)** | `~0.85s` | 256x256 tile on standard CPU |
| **Transformer Model Size** | `44.8 MB` | SegFormer-B0 alternative (`.pth`) |

---

## Known Limitations & Operating Bounds

1. **Pixel-Space Proxy Coordinates**: Graph edge lengths and node positions are represented in image pixel coordinates, not real-world geographic distances (meters/kilometers) or GPS coordinates. Detour deltas reflect pixel distance changes.
2. **Grade Separation**: Monocular raster imagery cannot distinguish grade-separated flyovers or underpasses from at-grade intersections without 3D digital elevation models (DEM) or vector overpass tags. Sknw treats planar crossings as junctions.
3. **Sensor & Domain Generalization**: The models were trained on 256x256 satellite tiles from the `RouteResilience` distribution. Generalization to aerial photography with different ground sample distances (GSD), extreme off-nadir angles, or snow/desert terrain may exhibit false negatives.
4. **Cloud Occlusion Thresholds**: Cloud gap recovery requires that road branches on either side exhibit coherent alignment ($\ge 0.60$ mutual heading) and cloud mask support. Roads that make sharp $90^\circ$ turns entirely underneath cloud cover cannot be reconstructed without external map priors.
