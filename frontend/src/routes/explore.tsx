import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import {
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  FileImage,
  Image as ImageIcon,
  LoaderCircle,
  UploadCloud,
} from "lucide-react";
import { ChangeEvent, DragEvent, useEffect, useRef, useState } from "react";

import { Navbar } from "@/components/Navbar";

type LayerKey = "satellite" | "roads" | "repaired" | "skeleton";
type AnalysisPhase = "idle" | "processing" | "success" | "error";

type BackendAnalysisResponse = {
  analysis_id: string;
  status: "completed";
  input_image_name?: string;
  input_image_ref: string;
  artifacts: {
    original?: string;
    binary_mask?: string;
    skeleton?: string;
    visualization?: string;
    [key: string]: string | undefined;
  };
  summary?: {
    threshold?: number;
    healed_edge_count?: number;
    before_healing?: {
      n_nodes?: number;
      n_edges?: number;
      n_components?: number;
    };
    after_healing?: {
      n_nodes?: number;
      n_edges?: number;
      n_components?: number;
    };
  };
};

type AnalysisState = {
  phase: AnalysisPhase;
  message?: string;
  result?: BackendAnalysisResponse | null;
};

type ImageMeta = {
  width: number | undefined;
  height: number | undefined;
  size: string;
};

const MODEL_PERFORMANCE = {
  iou: 54.89,
  dice: 70.88,
  threshold: 0.4,
  testSetImages: 846,
};

const PROCESSING_STEPS = ["VALIDATING IMAGE", "ROAD EXTRACTION", "NETWORK PREPARATION"];
const SAVED_ANALYSIS_KEY = "route-resilience.analysis-id";

const LAYER_OPTIONS: { key: LayerKey; label: string }[] = [
  { key: "satellite", label: "SATELLITE" },
  { key: "roads", label: "AI ROADS" },
  { key: "repaired", label: "REPAIRED" },
  { key: "skeleton", label: "ROAD SKELETON" },
];

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

async function analyzeRoadImage(file: File): Promise<BackendAnalysisResponse | { status: "error"; message: string }> {
  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("http://127.0.0.1:8000/api/road-analysis", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      let message = `The analysis backend returned an error (${response.status}).`;
      try {
        const errorPayload = (await response.json()) as { detail?: string | Record<string, unknown> };
        if (typeof errorPayload.detail === "string") {
          message = errorPayload.detail;
        } else if (errorPayload.detail) {
          message = JSON.stringify(errorPayload.detail);
        }
      } catch {
        const errorText = await response.text();
        if (errorText) message = errorText;
      }
      return {
        status: "error",
        message,
      };
    }

    const payload = (await response.json()) as BackendAnalysisResponse;
    if (payload.status !== "completed" || !payload.analysis_id || !payload.artifacts) {
      return { status: "error", message: "The analysis backend returned an incomplete result." };
    }
    return payload;
  } catch (error) {
    return {
      status: "error",
      message: error instanceof TypeError
        ? "The analysis backend is unavailable. Start the FastAPI server on 127.0.0.1:8000 and try again."
        : "The analysis request failed. Please try again.",
    };
  }
}

export const Route = createFileRoute("/explore")({
  head: () => ({
    meta: [
      { title: "Explore — ROUTE RESILIENCE" },
      { name: "description", content: "Upload a satellite image and run AI road extraction." },
    ],
  }),
  component: ExplorePage,
});

function ExplorePage() {
  const navigate = useNavigate();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [imageMeta, setImageMeta] = useState<ImageMeta | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [analysis, setAnalysis] = useState<AnalysisState>({ phase: "idle" });
  const [activeLayer, setActiveLayer] = useState<LayerKey>("satellite");
  const [pointer, setPointer] = useState({ x: 0, y: 0 });
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    const onPointerMove = (event: PointerEvent) => {
      setPointer({
        x: (event.clientX / window.innerWidth - 0.5) * 2,
        y: (event.clientY / window.innerHeight - 0.5) * 2,
      });
    };

    window.addEventListener("pointermove", onPointerMove, { passive: true });
    return () => window.removeEventListener("pointermove", onPointerMove);
  }, []);

  useEffect(() => {
    const savedAnalysisId = window.localStorage.getItem(SAVED_ANALYSIS_KEY);
    if (!savedAnalysisId) return;

    const controller = new AbortController();
    setAnalysis({ phase: "processing", message: "Restoring current analysis..." });

    fetch(`http://127.0.0.1:8000/api/analysis/${encodeURIComponent(savedAnalysisId)}`, {
      signal: controller.signal,
    })
      .then(async (response) => {
        const payload = (await response.json()) as BackendAnalysisResponse | { detail?: string };
        if (!response.ok) {
          const detail = "detail" in payload ? payload.detail : undefined;
          throw new Error(typeof detail === "string" ? detail : `Analysis restore failed (${response.status}).`);
        }
        return payload as BackendAnalysisResponse;
      })
      .then((payload) => {
        if (payload.status !== "completed" || payload.analysis_id !== savedAnalysisId || !payload.artifacts) {
          throw new Error("The saved analysis is incomplete.");
        }
        setAnalysis({ phase: "success", result: payload });
      })
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === "AbortError") return;
        window.localStorage.removeItem(SAVED_ANALYSIS_KEY);
        setAnalysis({ phase: "idle" });
      });

    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl(null);
      setImageMeta(null);
      return;
    }

    const nextUrl = URL.createObjectURL(selectedFile);
    setPreviewUrl(nextUrl);
    setImageMeta({
      width: undefined,
      height: undefined,
      size: formatFileSize(selectedFile.size),
    });

    const img = new Image();
    img.onload = () => {
      setImageMeta((current) => ({
        ...current,
        size: current?.size ?? formatFileSize(selectedFile.size),
        width: img.width,
        height: img.height,
      }));
    };
    img.src = nextUrl;

    return () => URL.revokeObjectURL(nextUrl);
  }, [selectedFile]);

  const handleFileSelection = (file: File | null | undefined) => {
    if (!file) return;
    const isValidType = ["image/png", "image/jpeg", "image/jpg"].includes(file.type) || /\.(png|jpe?g)$/i.test(file.name);

    if (!isValidType) {
      setAnalysis({ phase: "error", message: "Please upload a PNG or JPG/JPEG image." });
      return;
    }

    window.localStorage.removeItem(SAVED_ANALYSIS_KEY);
    setSelectedFile(file);
    setAnalysis({ phase: "idle" });
  };

  const handleInputChange = (event: ChangeEvent<HTMLInputElement>) => {
    handleFileSelection(event.target.files?.[0]);
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);
    handleFileSelection(event.dataTransfer.files?.[0]);
  };

  const runRoadAnalysis = async () => {
    if (!selectedFile) {
      setAnalysis({ phase: "error", message: "Select a satellite image before running the analysis." });
      return;
    }

    setAnalysis({ phase: "processing", message: "Running AI road extraction..." });

    const response = await analyzeRoadImage(selectedFile);

    if (response.status === "completed") {
      window.localStorage.setItem(SAVED_ANALYSIS_KEY, response.analysis_id);
      setAnalysis({ phase: "success", result: response });
      return;
    }

    setAnalysis({ phase: "error", message: response.message });
  };

  const getImageSource = () => {
    if (analysis.phase === "success" && analysis.result) {
      const artifacts = analysis.result.artifacts;
      if (activeLayer === "satellite") return artifacts.original ?? analysis.result.input_image_ref;
      if (activeLayer === "roads") return artifacts.binary_mask ?? null;
      if (activeLayer === "repaired") return artifacts.visualization ?? null;
      if (activeLayer === "skeleton") return artifacts.skeleton ?? null;
    }

    return selectedFile && previewUrl ? previewUrl : null;
  };

  const resetAnalysis = () => {
    window.localStorage.removeItem(SAVED_ANALYSIS_KEY);
    setSelectedFile(null);
    setAnalysis({ phase: "idle" });
    setActiveLayer("satellite");
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const viewCurrentNetwork = (analysisId: string) => {
    void navigate({
      to: "/network",
      search: { analysis: analysisId },
    });
  };

  const currentImage = getImageSource();
  const isResultVisible = analysis.phase === "success" && analysis.result;
  const imageSummary = analysis.result?.summary;

  return (
    <main className="relative isolate min-h-screen bg-background text-foreground">
      <div aria-hidden className="pointer-events-none absolute inset-0 z-0">
        <div className="explore-bg-grid absolute inset-0 opacity-[0.04]" />
        <div className="explore-bg-radial absolute inset-0" />
      </div>

      <svg
        aria-hidden
        className="pointer-events-none absolute inset-0 z-[1] h-full w-full opacity-25 mix-blend-screen"
        viewBox="0 0 1440 900"
        preserveAspectRatio="xMidYMid slice"
        style={{ transform: `translate3d(${pointer.x * 4}px, ${pointer.y * 4}px, 0)` }}
      >
        <path className="route-line" d="M-90 720 C 140 650, 180 770, 360 674 S 610 490, 760 566 S 970 750, 1170 588 S 1340 390, 1530 330" fill="none" stroke="var(--primary)" strokeWidth="1" strokeDasharray="7 15" />
        <path className="route-line" d="M210 -60 C 290 90, 210 226, 376 320 S 540 454, 496 624 S 590 810, 746 970" fill="none" stroke="var(--accent)" strokeWidth="0.8" strokeDasharray="4 18" />
        <circle cx="760" cy="566" r="3" fill="var(--primary)" opacity="0.7" />
        <circle cx="760" cy="566" r="10" fill="none" stroke="var(--primary)" strokeWidth="0.6" opacity="0.4" />
      </svg>

      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 z-[2] opacity-20 [background-image:url('data:image/svg+xml,%3Csvg viewBox=%220 0 180 180%22 xmlns=%22http://www.w3.org/2000/svg%22%3E%3Cfilter id=%22n%22%3E%3CfeTurbulence type=%22fractalNoise%22 baseFrequency=%22.85%22 numOctaves=%224%22 stitchTiles=%22stitch%22/%3E%3C/filter%3E%3Crect width=%22100%25%22 height=%22100%25%22 filter=%22url(%23n)%22 opacity=%22.34%22/%3E%3C/svg%3E')]"
      />

      <Navbar />

      <header className="relative z-10 border-b border-border/30 px-6 py-8 sm:px-10 lg:px-14">
        <p className="mb-2 font-display text-[0.6rem] tracking-[0.28em] text-primary/70">
          ISRO <span className="px-1 text-foreground/40">•</span> BAH 2026 <span className="px-1 text-foreground/40">•</span> AI EXTRACTION
        </p>
        <h1 className="font-display text-3xl font-medium tracking-[-0.03em] text-foreground sm:text-4xl">
          What does our AI <span className="text-primary">see</span> in this image?
        </h1>
      </header>

      <div className="relative z-10 px-6 py-8 sm:px-10 lg:px-14">
        <div className="grid gap-6 xl:grid-cols-[360px_minmax(0,1fr)]">
          <aside className="space-y-4">
            <div
              onDragOver={(event) => {
                event.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
              className={`relative overflow-hidden border border-dashed p-5 transition-all duration-300 ${
                isDragging ? "border-primary bg-primary/5" : "border-border/70 bg-card/40"
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/png,image/jpeg,image/jpg"
                className="hidden"
                onChange={handleInputChange}
              />

              <div className="flex items-center justify-between">
                <p className="font-display text-[0.62rem] tracking-[0.18em] text-primary uppercase">INPUT</p>
              </div>

              <div className="mt-5 flex flex-col items-center justify-center rounded-md border border-border/50 bg-background/50 px-4 py-8 text-center">
                <UploadCloud className="h-10 w-10 text-primary" strokeWidth={1.3} />
                <p className="mt-4 font-display text-[0.68rem] tracking-[0.15em] text-foreground uppercase">
                  Drag & drop image
                </p>
                <p className="mt-2 text-xs text-muted-foreground">or browse to select a PNG / JPG / JPEG</p>
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="mt-5 inline-flex items-center gap-2 border border-primary/70 bg-primary/10 px-4 py-2 font-display text-[0.58rem] tracking-[0.16em] text-primary transition-colors hover:bg-primary/20"
                >
                  <FileImage size={14} />
                  BROWSE IMAGE
                </button>
              </div>
            </div>

            {selectedFile && (
              <div className="border border-border/60 bg-card/40 p-4 backdrop-blur-sm">
                <p className="font-display text-[0.56rem] tracking-[0.16em] text-muted-foreground uppercase">FILE</p>
                <div className="mt-3 space-y-2 text-sm">
                  <div className="flex items-center justify-between gap-3">
                    <span className="text-muted-foreground">Name</span>
                    <span className="max-w-[180px] truncate text-right text-foreground">{selectedFile.name}</span>
                  </div>
                  <div className="flex items-center justify-between gap-3">
                    <span className="text-muted-foreground">Size</span>
                    <span className="text-foreground">{formatFileSize(selectedFile.size)}</span>
                  </div>
                  <div className="flex items-center justify-between gap-3">
                    <span className="text-muted-foreground">Dimensions</span>
                    <span className="text-foreground">
                      {imageMeta?.width && imageMeta?.height ? `${imageMeta.width} × ${imageMeta.height}px` : "Loading…"}
                    </span>
                  </div>
                </div>
              </div>
            )}

            {analysis.phase === "success" && analysis.result && (
              <div className="border border-primary/40 bg-primary/5 p-4 backdrop-blur-sm">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <p className="font-display text-[0.56rem] tracking-[0.16em] text-primary uppercase">CURRENT ANALYSIS</p>
                    <p className="mt-2 font-display text-[0.68rem] tracking-[0.12em] text-foreground uppercase">ANALYSIS READY</p>
                  </div>
                  <CheckCircle2 className="h-5 w-5 text-primary" />
                </div>
                <p className="mt-3 truncate text-sm text-foreground/80">
                  {analysis.result.input_image_name ?? "Restored uploaded image"}
                </p>
                <p className="mt-1 break-all font-mono text-[0.68rem] text-muted-foreground">{analysis.result.analysis_id}</p>
                <div className="mt-4 grid gap-2 sm:grid-cols-2">
                  <Link
                    to="/network"
                    search={{ analysis: analysis.result.analysis_id }}
                    onClick={(event) => {
                      event.preventDefault();
                      viewCurrentNetwork(analysis.result!.analysis_id);
                    }}
                    className="inline-flex items-center justify-center gap-2 border border-primary/70 bg-primary/10 px-3 py-2 font-display text-[0.56rem] tracking-[0.16em] text-primary transition-colors hover:bg-primary/20"
                  >
                    VIEW NETWORK <ArrowRight size={13} />
                  </Link>
                  <button
                    type="button"
                    onClick={resetAnalysis}
                    className="border border-border/70 px-3 py-2 font-display text-[0.56rem] tracking-[0.16em] text-foreground/70 transition-colors hover:border-primary/50 hover:text-primary"
                  >
                    NEW ANALYSIS
                  </button>
                </div>
              </div>
            )}

            <button
              type="button"
              onClick={runRoadAnalysis}
              disabled={!selectedFile || analysis.phase === "processing"}
              className="flex w-full items-center justify-between border border-primary/70 bg-primary px-4 py-3 font-display text-[0.62rem] tracking-[0.18em] text-primary-foreground transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
            >
              <span>RUN ROAD ANALYSIS</span>
              <ArrowRight size={15} />
            </button>
          </aside>

          <section className="space-y-5">
            {analysis.phase === "idle" && !selectedFile && (
              <div className="flex min-h-[420px] items-center justify-center border border-border/50 bg-card/30 p-10 text-center">
                <div>
                  <ImageIcon className="mx-auto h-10 w-10 text-muted-foreground" strokeWidth={1.2} />
                  <p className="mt-4 font-display text-[0.7rem] tracking-[0.18em] text-muted-foreground uppercase">
                    Upload a satellite image to begin
                  </p>
                </div>
              </div>
            )}

            {analysis.phase === "idle" && selectedFile && !analysis.result && (
              <div className="border border-border/50 bg-card/30 p-4">
                <div className="mb-3 flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 text-primary" />
                  <p className="font-display text-[0.56rem] tracking-[0.16em] text-primary uppercase">READY FOR ANALYSIS</p>
                </div>
                {previewUrl && (
                  <img
                    src={previewUrl}
                    alt="Selected satellite input preview"
                    className="h-[320px] w-full object-contain border border-border/50 bg-background/50"
                  />
                )}
              </div>
            )}

            {(analysis.phase === "processing" || analysis.phase === "success" || analysis.phase === "error") && (
              <div className="space-y-4">
                {analysis.phase === "processing" && (
                  <div className="border border-border/60 bg-card/30 p-5">
                    <div className="flex items-center gap-3 text-primary">
                      <LoaderCircle className="h-4 w-4 animate-spin" />
                      <p className="font-display text-[0.62rem] tracking-[0.18em] uppercase">PROCESSING</p>
                    </div>
                    <div className="mt-4 flex flex-wrap gap-2">
                      {PROCESSING_STEPS.map((step) => (
                        <span key={step} className="border border-border/60 bg-background/40 px-3 py-2 font-display text-[0.52rem] tracking-[0.16em] text-foreground/80">
                          {step}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {analysis.phase === "error" && (
                  <div className="border border-amber-500/40 bg-amber-500/5 p-5 text-amber-200">
                    <div className="flex items-center gap-2">
                      <AlertTriangle className="h-4 w-4" />
                      <p className="font-display text-[0.58rem] tracking-[0.18em] uppercase">INPUT ERROR</p>
                    </div>
                    <p className="mt-3 text-sm text-amber-100">{analysis.message}</p>
                  </div>
                )}

                {analysis.phase === "success" && analysis.result && (
                  <>
                    <div className="border border-border/60 bg-card/30 p-3">
                      <div className="mb-3 flex items-center gap-1 self-start border border-border/60 p-1 backdrop-blur-sm bg-card/60">
                        {LAYER_OPTIONS.map(({ key, label }) => (
                          <button
                            key={key}
                            type="button"
                            onClick={() => setActiveLayer(key)}
                            className={`px-3 py-2 font-display text-[0.56rem] tracking-[0.16em] transition-all duration-200 ${
                              activeLayer === key ? "bg-primary text-primary-foreground" : "text-foreground/60 hover:text-foreground hover:bg-secondary/60"
                            }`}
                          >
                            {label}
                          </button>
                        ))}
                      </div>

                      {currentImage ? (
                        <img
                          src={currentImage}
                          alt="Road extraction result"
                          className="h-[380px] w-full border border-border/60 bg-background/50 object-contain"
                        />
                      ) : (
                        <div className="flex h-[380px] items-center justify-center border border-border/60 bg-background/50 text-sm text-muted-foreground">
                          No result available for this layer.
                        </div>
                      )}
                    </div>

                    <div className="grid gap-4 lg:grid-cols-2">
                      <div className="border border-border/60 bg-card/30 p-4">
                        <p className="font-display text-[0.56rem] tracking-[0.18em] text-muted-foreground uppercase">EXTRACTION SUMMARY</p>
                        <div className="mt-4 grid gap-3 text-sm sm:grid-cols-2">
                          {imageSummary?.after_healing?.n_nodes !== undefined && (
                            <div className="border border-border/50 bg-background/40 p-3">
                              <p className="text-[0.55rem] tracking-[0.12em] text-muted-foreground uppercase">Healed graph nodes</p>
                              <p className="mt-2 text-lg font-medium text-primary">{imageSummary.after_healing.n_nodes}</p>
                            </div>
                          )}
                          {imageSummary?.after_healing?.n_edges !== undefined && (
                            <div className="border border-border/50 bg-background/40 p-3">
                              <p className="text-[0.55rem] tracking-[0.12em] text-muted-foreground uppercase">Healed graph edges</p>
                              <p className="mt-2 text-lg font-medium text-primary">{imageSummary.after_healing.n_edges}</p>
                            </div>
                          )}
                          {imageSummary?.healed_edge_count !== undefined && (
                            <div className="border border-border/50 bg-background/40 p-3">
                              <p className="text-[0.55rem] tracking-[0.12em] text-muted-foreground uppercase">Healed edges</p>
                              <p className="mt-2 text-lg font-medium text-primary">{imageSummary.healed_edge_count}</p>
                            </div>
                          )}
                          {imageSummary?.threshold !== undefined && (
                            <div className="border border-border/50 bg-background/40 p-3">
                              <p className="text-[0.55rem] tracking-[0.12em] text-muted-foreground uppercase">Threshold</p>
                              <p className="mt-2 text-lg font-medium text-primary">{imageSummary.threshold}</p>
                            </div>
                          )}
                          {analysis.result.analysis_id && (
                            <div className="border border-border/50 bg-background/40 p-3 sm:col-span-2">
                              <p className="text-[0.55rem] tracking-[0.12em] text-muted-foreground uppercase">Analysis ID</p>
                              <p className="mt-2 break-all text-sm font-medium text-primary">{analysis.result.analysis_id}</p>
                            </div>
                          )}
                        </div>
                      </div>

                      <div className="border border-border/60 bg-card/30 p-4">
                        <p className="font-display text-[0.56rem] tracking-[0.18em] text-muted-foreground uppercase">MODEL PERFORMANCE · TEST SET · 846 IMAGES</p>
                        <div className="mt-4 grid gap-3 text-sm sm:grid-cols-2">
                          <div className="border border-border/50 bg-background/40 p-3">
                            <p className="text-[0.55rem] tracking-[0.12em] text-muted-foreground uppercase">IoU</p>
                            <p className="mt-2 text-lg font-medium text-primary">{MODEL_PERFORMANCE.iou.toFixed(2)}%</p>
                          </div>
                          <div className="border border-border/50 bg-background/40 p-3">
                            <p className="text-[0.55rem] tracking-[0.12em] text-muted-foreground uppercase">Dice</p>
                            <p className="mt-2 text-lg font-medium text-primary">{MODEL_PERFORMANCE.dice.toFixed(2)}%</p>
                          </div>
                          <div className="border border-border/50 bg-background/40 p-3">
                            <p className="text-[0.55rem] tracking-[0.12em] text-muted-foreground uppercase">Threshold</p>
                            <p className="mt-2 text-lg font-medium text-primary">{MODEL_PERFORMANCE.threshold}</p>
                          </div>
                          <div className="border border-border/50 bg-background/40 p-3">
                            <p className="text-[0.55rem] tracking-[0.12em] text-muted-foreground uppercase">Test set</p>
                            <p className="mt-2 text-lg font-medium text-primary">{MODEL_PERFORMANCE.testSetImages}</p>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="flex justify-end pt-2">
                      <Link
                        to="/network"
                        search={{ analysis: analysis.result.analysis_id }}
                        onClick={(event) => {
                          event.preventDefault();
                          viewCurrentNetwork(analysis.result!.analysis_id);
                        }}
                        className="group inline-flex items-center gap-3 border border-primary/70 bg-primary/10 px-4 py-3 font-display text-[0.62rem] tracking-[0.18em] text-primary transition-colors hover:bg-primary/20"
                      >
                        ANALYZE NETWORK <ArrowRight size={15} className="transition-transform duration-300 group-hover:translate-x-1" />
                      </Link>
                    </div>
                  </>
                )}
              </div>
            )}
          </section>
        </div>
      </div>
    </main>
  );
}
