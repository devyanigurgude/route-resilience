import { createFileRoute, useNavigate } from "@tanstack/react-router";
import {
  AlertCircle,
  ArrowRight,
  CheckCircle2,
  Clock,
  ExternalLink,
  FileImage,
  FolderOpen,
  Layers,
  LoaderCircle,
  Network,
  Plus,
  RefreshCw,
  Sparkles,
  Trash2,
  UploadCloud,
} from "lucide-react";
import { ChangeEvent, DragEvent, useEffect, useRef, useState } from "react";

import {
  API_BASE,
  deleteAnalysis,
  getAnalyses,
  saveAnalysisToHistory,
  SAVED_ANALYSIS_KEY,
  type AnalysisItem,
} from "@/lib/backend";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { cn } from "@/lib/utils";

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
    image_dimensions?: { height: number; width: number };
    mask_quality?: { road_fraction?: number; road_pixel_count?: number };
    before_healing?: { n_nodes: number; n_edges: number; n_components: number };
    after_healing?: { n_nodes: number; n_edges: number; n_components: number };
    graph_quality?: {
      component_sizes?: number[];
      isolated_node_count?: number;
      junction_count?: number;
      routable_component_count?: number;
    };
    graph_status?: string;
    warnings?: string[];
  };
};

type AnalysisState = {
  phase: AnalysisPhase;
  message?: string;
  result?: BackendAnalysisResponse | null;
};

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDate(dateStr?: string): string {
  if (!dateStr) return "Recent";
  try {
    const d = new Date(dateStr);
    if (isNaN(d.getTime())) return "Recent";
    return new Intl.DateTimeFormat(undefined, {
      month: "short",
      day: "numeric",
      hour: "numeric",
      minute: "2-digit",
    }).format(d);
  } catch {
    return "Recent";
  }
}

async function analyzeRoadImage(
  file: File,
): Promise<BackendAnalysisResponse | { status: "error"; message: string }> {
  const formData = new FormData();
  formData.append("file", file);
  try {
    const response = await fetch(`${API_BASE}/api/road-analysis`, {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      let message = `Analysis failed (${response.status}).`;
      try {
        const errorPayload = (await response.json()) as {
          detail?: string | Record<string, unknown>;
        };
        if (typeof errorPayload.detail === "string") message = errorPayload.detail;
        else if (errorPayload.detail) message = JSON.stringify(errorPayload.detail);
      } catch {
        const errorText = await response.text();
        if (errorText) message = errorText;
      }
      return { status: "error", message };
    }
    const payload = (await response.json()) as BackendAnalysisResponse;
    if (payload.status !== "completed" || !payload.analysis_id || !payload.artifacts) {
      return { status: "error", message: "Incomplete result returned by the backend." };
    }
    return payload;
  } catch (error) {
    return {
      status: "error",
      message:
        error instanceof TypeError
          ? "The analysis server is offline. Please ensure the backend is running."
          : "The analysis request failed. Please try again.",
    };
  }
}

export const Route = createFileRoute("/explore")({
  head: () => ({
    meta: [
      { title: "Route Resilience" },
      {
        name: "description",
        content: "Upload satellite imagery to extract road networks and evaluate network resilience.",
      },
    ],
  }),
  component: ExplorePage,
});

function ExplorePage() {
  const navigate = useNavigate();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [analysis, setAnalysis] = useState<AnalysisState>({ phase: "idle" });
  const [history, setHistory] = useState<AnalysisItem[]>([]);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<"new" | "history">("new");
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const [itemToDelete, setItemToDelete] = useState<AnalysisItem | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [actionFeedback, setActionFeedback] = useState<{
    type: "success" | "error";
    message: string;
  } | null>(null);

  const promptDelete = (e: React.MouseEvent, item: AnalysisItem) => {
    e.stopPropagation();
    if (item.status === "processing" || item.status === "running") {
      setActionFeedback({
        type: "error",
        message: "Cannot delete an analysis while it is still running.",
      });
      return;
    }
    setItemToDelete(item);
  };

  const confirmDelete = async () => {
    if (!itemToDelete) return;
    setIsDeleting(true);
    setActionFeedback(null);
    try {
      await deleteAnalysis(itemToDelete.analysis_id);
      setHistory((prev) => prev.filter((it) => it.analysis_id !== itemToDelete.analysis_id));
      if (
        typeof window !== "undefined" &&
        window.localStorage.getItem(SAVED_ANALYSIS_KEY) === itemToDelete.analysis_id
      ) {
        window.localStorage.removeItem(SAVED_ANALYSIS_KEY);
        if (analysis.result?.analysis_id === itemToDelete.analysis_id) {
          setAnalysis({ phase: "idle" });
        }
      }
      const label = itemToDelete.input_image_name ?? `#${itemToDelete.analysis_id.slice(0, 8)}`;
      setActionFeedback({
        type: "success",
        message: `Analysis ${label} was deleted successfully.`,
      });
      setItemToDelete(null);
    } catch (err) {
      setActionFeedback({
        type: "error",
        message: err instanceof Error ? err.message : "Failed to delete analysis.",
      });
    } finally {
      setIsDeleting(false);
    }
  };

  // Load history from backend & local storage
  const loadHistory = async () => {
    setHistoryLoading(true);
    try {
      const items = await getAnalyses();
      setHistory(items);
    } catch {
      setHistory([]);
    } finally {
      setHistoryLoading(false);
    }
  };

  useEffect(() => {
    void loadHistory();
  }, []);

  // Check saved analysis on first mount
  useEffect(() => {
    const savedId = window.localStorage.getItem(SAVED_ANALYSIS_KEY);
    if (!savedId) return;

    const controller = new AbortController();
    fetch(`${API_BASE}/api/analysis/${encodeURIComponent(savedId)}`, {
      signal: controller.signal,
    })
      .then(async (res) => {
        if (!res.ok) throw new Error("Could not restore saved analysis");
        return (await res.json()) as BackendAnalysisResponse;
      })
      .then((payload) => {
        if (payload.status === "completed" && payload.analysis_id === savedId) {
          setAnalysis({ phase: "success", result: payload });
        }
      })
      .catch(() => {
        // Safe to ignore restore failures
      });

    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl(null);
      return;
    }
    const nextUrl = URL.createObjectURL(selectedFile);
    setPreviewUrl(nextUrl);
    return () => URL.revokeObjectURL(nextUrl);
  }, [selectedFile]);

  const handleFileSelection = (file: File | null | undefined) => {
    if (!file) return;
    const isValidType =
      ["image/png", "image/jpeg", "image/jpg"].includes(file.type) ||
      /\.(png|jpe?g)$/i.test(file.name);
    if (!isValidType) {
      setAnalysis({ phase: "error", message: "Please upload a PNG or JPG/JPEG image." });
      return;
    }
    setSelectedFile(file);
    setAnalysis({ phase: "idle" });
    setActiveTab("new");
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);
    handleFileSelection(event.dataTransfer.files?.[0]);
  };

  const runRoadAnalysis = async () => {
    if (!selectedFile) {
      setAnalysis({
        phase: "error",
        message: "Please choose a satellite image before running analysis.",
      });
      return;
    }
    setAnalysis({ phase: "processing", message: "Extracting road topology and healing gaps..." });
    const response = await analyzeRoadImage(selectedFile);
    if (response.status === "completed") {
      window.localStorage.setItem(SAVED_ANALYSIS_KEY, response.analysis_id);
      saveAnalysisToHistory({
        analysis_id: response.analysis_id,
        status: "completed",
        created_at: new Date().toISOString(),
        input_image_name: selectedFile.name,
        input_image_ref: response.input_image_ref,
        artifacts: response.artifacts,
        summary: response.summary,
      });
      setAnalysis({ phase: "success", result: response });
      void loadHistory();
      return;
    }
    setAnalysis({ phase: "error", message: response.message });
  };

  const startNewAnalysis = () => {
    setSelectedFile(null);
    setAnalysis({ phase: "idle" });
    if (fileInputRef.current) fileInputRef.current.value = "";
    setActiveTab("new");
  };

  const reopenAnalysis = (item: AnalysisItem) => {
    window.localStorage.setItem(SAVED_ANALYSIS_KEY, item.analysis_id);
    void navigate({ to: "/network", search: { analysis: item.analysis_id } });
  };

  const isProcessing = analysis.phase === "processing";
  const completedResult = analysis.phase === "success" ? analysis.result : null;

  return (
    <main className="workflow-page min-h-[calc(100vh-104px)] px-5 py-8 text-foreground sm:px-8 max-w-6xl mx-auto">
      {/* Page Header */}
      <div className="mb-8 border-b border-border pb-5 flex flex-col sm:flex-row sm:items-end sm:justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center rounded-md bg-teal-500/10 px-2 py-0.5 text-xs font-semibold text-teal-700 dark:text-teal-300 border border-teal-500/20">
              01 · Analyze
            </span>
            <span className="text-xs text-muted-foreground">Step 1 of 5</span>
          </div>
          <h1 className="mt-2 text-2xl sm:text-3xl font-bold tracking-tight text-foreground">
            Satellite Road Network Extraction
          </h1>
          <p className="mt-1.5 max-w-2xl text-sm leading-relaxed text-muted-foreground">
            Upload high-resolution satellite imagery to extract topological roads, bridge
            occlusions, and generate a traversable network graph.
          </p>
        </div>

        {/* Tab switch between new analysis and previous analyses */}
        <div className="flex items-center gap-1 p-1 bg-muted/40 rounded-xl border border-border text-xs font-medium self-start sm:self-end text-muted-foreground">
          <button
            type="button"
            onClick={() => setActiveTab("new")}
            className={`px-3 py-1.5 rounded-lg transition-colors flex items-center gap-1.5 ${
              activeTab === "new"
                ? "bg-card text-foreground shadow-2xs border border-border/80 font-semibold"
                : "hover:text-foreground"
            }`}
          >
            <Plus size={14} />
            <span>New Upload</span>
          </button>
          <button
            type="button"
            onClick={() => setActiveTab("history")}
            className={`px-3 py-1.5 rounded-lg transition-colors flex items-center gap-1.5 ${
              activeTab === "history"
                ? "bg-card text-foreground shadow-2xs border border-border/80 font-semibold"
                : "hover:text-foreground"
            }`}
          >
            <Clock size={14} />
            <span>Previous Analyses ({history.length})</span>
          </button>
        </div>
      </div>

      {/* Main Content Layout */}
      {activeTab === "new" && (
        <div className="grid gap-8 lg:grid-cols-[1fr_340px]">
          {/* Uploader Card */}
          <section className="rounded-xl border border-border bg-card p-6 sm:p-8 shadow-xs">
            <input
              ref={fileInputRef}
              type="file"
              accept="image/png,image/jpeg,image/jpg"
              className="hidden"
              onChange={(event: ChangeEvent<HTMLInputElement>) =>
                handleFileSelection(event.target.files?.[0])
              }
            />

            {/* Dropzone */}
            <div
              onDragOver={(event) => {
                event.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
              className={`rounded-xl border-2 border-dashed p-8 sm:p-12 text-center transition-all ${
                isDragging
                  ? "border-primary bg-primary/10"
                  : "border-border bg-muted/20 hover:border-teal-500/80 hover:bg-muted/40"
              }`}
            >
              <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-teal-500/10 text-teal-600 dark:text-teal-400 border border-teal-500/20">
                <UploadCloud size={28} strokeWidth={2} />
              </div>
              <h2 className="mt-4 text-base sm:text-lg font-bold text-foreground">
                Drag and drop your satellite image
              </h2>
              <p className="mt-1.5 text-xs sm:text-sm text-muted-foreground max-w-sm mx-auto">
                Supports standard high-resolution aerial and satellite tiles (PNG, JPG, JPEG).
              </p>
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                disabled={isProcessing}
                className="mt-5 inline-flex items-center gap-2 rounded-lg border border-border bg-card px-4 py-2 text-xs font-semibold text-foreground shadow-2xs transition hover:border-primary hover:text-primary disabled:opacity-50"
              >
                <FileImage size={15} />
                <span>Browse files</span>
              </button>
            </div>

            {/* Selected File Details */}
            {selectedFile && (
              <div className="mt-5 flex flex-col sm:flex-row sm:items-center justify-between gap-4 rounded-xl border border-border bg-muted/30 p-4">
                <div className="flex items-center gap-3 min-w-0">
                  {previewUrl ? (
                    <img
                      src={previewUrl}
                      alt="Thumbnail preview"
                      className="h-12 w-12 rounded-lg object-cover border border-border shrink-0 satellite-imagery"
                      style={{ filter: "none" }}
                    />
                  ) : (
                    <div className="h-12 w-12 rounded-lg bg-teal-500/10 text-teal-600 dark:text-teal-400 flex items-center justify-center shrink-0 border border-teal-500/20">
                      <FileImage size={20} />
                    </div>
                  )}
                  <div className="min-w-0">
                    <p className="truncate text-sm font-semibold text-foreground">
                      {selectedFile.name}
                    </p>
                    <p className="text-xs text-muted-foreground">{formatFileSize(selectedFile.size)}</p>
                  </div>
                </div>

                <div className="flex items-center gap-2 self-end sm:self-center">
                  <button
                    type="button"
                    onClick={startNewAnalysis}
                    disabled={isProcessing}
                    className="text-xs font-medium text-muted-foreground hover:text-destructive px-2 py-1 transition"
                  >
                    Remove
                  </button>
                </div>
              </div>
            )}

            {/* Error Message */}
            {analysis.phase === "error" && (
              <div
                role="alert"
                className="mt-5 flex items-start gap-3 rounded-xl border border-destructive/40 bg-destructive/10 p-4 text-xs sm:text-sm text-destructive"
              >
                <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-destructive" />
                <div className="flex-1">{analysis.message}</div>
              </div>
            )}

            {/* Processing State */}
            {isProcessing && (
              <div className="mt-5 rounded-xl border border-teal-500/30 bg-teal-500/10 p-5">
                <div className="flex items-center gap-3 text-sm font-semibold text-teal-700 dark:text-teal-300">
                  <LoaderCircle className="h-5 w-5 animate-spin" />
                  <span>{analysis.message ?? "Analyzing satellite imagery..."}</span>
                </div>
                <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-teal-500/20">
                  <div className="h-full w-2/3 animate-pulse rounded-full bg-teal-600 dark:bg-teal-400" />
                </div>
                <p className="mt-2 text-xs text-muted-foreground">
                  Segmenting road footprints, skeletonizing centerlines, and validating topological
                  corridors.
                </p>
              </div>
            )}

            {/* Completed Result Card */}
            {completedResult && (
              <div className="mt-6 rounded-xl border border-teal-500/40 bg-teal-500/10 p-5 sm:p-6">
                <div className="flex items-start gap-3.5">
                  <div className="rounded-xl bg-teal-500/20 p-2 text-teal-700 dark:text-teal-300 shrink-0 border border-teal-500/30">
                    <CheckCircle2 className="h-5 w-5" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <h3 className="text-sm font-bold text-foreground">
                      Road network extracted successfully
                    </h3>
                    <p className="mt-1 text-xs text-muted-foreground">
                      Topology reconstruction is complete. You can now inspect critical lifelines or
                      simulate disruptions.
                    </p>

                    {completedResult.summary && (
                      <div className="mt-4 grid grid-cols-2 sm:grid-cols-4 gap-2.5">
                        <div className="bg-card rounded-lg p-2.5 border border-border text-center">
                          <span className="block text-[0.62rem] font-bold text-teal-600 dark:text-teal-400 uppercase tracking-wider">
                            Nodes
                          </span>
                          <span className="text-sm font-bold text-foreground font-mono">
                            {completedResult.summary.after_healing?.n_nodes ?? "—"}
                          </span>
                        </div>
                        <div className="bg-card rounded-lg p-2.5 border border-border text-center">
                          <span className="block text-[0.62rem] font-bold text-teal-600 dark:text-teal-400 uppercase tracking-wider">
                            Edges
                          </span>
                          <span className="text-sm font-bold text-foreground font-mono">
                            {completedResult.summary.after_healing?.n_edges ?? "—"}
                          </span>
                        </div>
                        <div className="bg-card rounded-lg p-2.5 border border-border text-center">
                          <span className="block text-[0.62rem] font-bold text-teal-600 dark:text-teal-400 uppercase tracking-wider">
                            Components
                          </span>
                          <span className="text-sm font-bold text-foreground font-mono">
                            {completedResult.summary.after_healing?.n_components ?? "—"}
                          </span>
                        </div>
                        <div className="bg-card rounded-lg p-2.5 border border-border text-center">
                          <span className="block text-[0.62rem] font-bold text-teal-600 dark:text-teal-400 uppercase tracking-wider">
                            Routable
                          </span>
                          <span className="text-sm font-bold text-foreground font-mono">
                            {completedResult.summary.graph_quality?.routable_component_count ?? 1}
                          </span>
                        </div>
                      </div>
                    )}

                    <div className="mt-5 flex flex-wrap items-center gap-3">
                      <button
                        type="button"
                        onClick={() =>
                          void navigate({
                            to: "/network",
                            search: { analysis: completedResult.analysis_id },
                          })
                        }
                        className="inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-2.5 text-xs font-semibold text-primary-foreground shadow-xs hover:bg-primary/90 transition"
                      >
                        <span>Open in Network</span>
                        <ArrowRight size={14} />
                      </button>
                      <button
                        type="button"
                        onClick={startNewAnalysis}
                        className="rounded-lg border border-border bg-card px-4 py-2.5 text-xs font-semibold text-foreground hover:bg-muted/60 transition shadow-2xs"
                      >
                        Start another analysis
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Run Analysis Trigger Button */}
            {!completedResult && (
              <button
                type="button"
                onClick={runRoadAnalysis}
                disabled={!selectedFile || isProcessing}
                className="mt-6 inline-flex w-full items-center justify-center gap-2 rounded-lg bg-primary px-5 py-3 text-sm font-semibold text-primary-foreground shadow-xs transition hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {isProcessing ? (
                  <>
                    <LoaderCircle className="h-4 w-4 animate-spin" />
                    <span>Analyzing Image...</span>
                  </>
                ) : (
                  <>
                    <Sparkles size={16} />
                    <span>Extract Road Network</span>
                  </>
                )}
              </button>
            )}
          </section>

          {/* Quick Info & Recent Analyses Sidebar */}
          <aside className="flex flex-col gap-5">
            {/* Quick Helper Box */}
            <div className="rounded-xl border border-border bg-card p-5 shadow-xs">
              <h3 className="text-xs font-bold text-foreground uppercase tracking-wider flex items-center gap-2">
                <Network size={15} className="text-teal-600 dark:text-teal-400" />
                <span>Pipeline Stages</span>
              </h3>
              <ol className="mt-3.5 space-y-2.5 text-xs text-muted-foreground">
                <li className="flex gap-2">
                  <span className="font-bold text-teal-600 dark:text-teal-400">1.</span>
                  <span>
                    <strong className="text-foreground">Segmentation:</strong> Detects road footprints from RGB satellite
                    imagery.
                  </span>
                </li>
                <li className="flex gap-2">
                  <span className="font-bold text-teal-600 dark:text-teal-400">2.</span>
                  <span>
                    <strong className="text-foreground">Skeletonization:</strong> Thins footprints to a 1-pixel centerline
                    topological graph.
                  </span>
                </li>
                <li className="flex gap-2">
                  <span className="font-bold text-teal-600 dark:text-teal-400">3.</span>
                  <span>
                    <strong className="text-foreground">Occlusion Healing:</strong> Reconstructs severed roads through clouds
                    and canopy.
                  </span>
                </li>
                <li className="flex gap-2">
                  <span className="font-bold text-teal-600 dark:text-teal-400">4.</span>
                  <span>
                    <strong className="text-foreground">Resilience Audit:</strong> Evaluates connectivity and isolates critical
                    lifelines.
                  </span>
                </li>
              </ol>
            </div>

            {/* Recent History Preview */}
            <div className="rounded-xl border border-border bg-card p-5 shadow-xs">
              <div className="flex items-center justify-between">
                <h3 className="text-xs font-bold text-foreground uppercase tracking-wider flex items-center gap-1.5">
                  <Clock size={14} className="text-muted-foreground" />
                  <span>Recent Analyses</span>
                </h3>
                {history.length > 0 && (
                  <button
                    type="button"
                    onClick={() => setActiveTab("history")}
                    className="text-xs text-teal-600 dark:text-teal-400 font-medium hover:underline"
                  >
                    View all
                  </button>
                )}
              </div>

              {historyLoading ? (
                <div className="py-6 text-center text-xs text-muted-foreground">
                  <LoaderCircle className="h-4 w-4 animate-spin mx-auto mb-1.5 text-muted-foreground" />
                  Loading history...
                </div>
              ) : history.length === 0 ? (
                <p className="mt-3 text-xs text-muted-foreground">
                  No previous analyses found yet. Run an analysis above to save it here.
                </p>
              ) : (
                <div className="mt-3.5 divide-y divide-border">
                  {history.slice(0, 4).map((item) => (
                    <div
                      key={item.analysis_id}
                      onClick={() => reopenAnalysis(item)}
                      className="py-2.5 first:pt-0 last:pb-0 flex items-center justify-between gap-3 cursor-pointer group"
                    >
                      <div className="min-w-0">
                        <p className="text-xs font-semibold text-foreground truncate group-hover:text-teal-600 dark:group-hover:text-teal-400 transition">
                          {item.input_image_name ?? `Analysis #${item.analysis_id.slice(0, 8)}`}
                        </p>
                        <p className="text-[0.68rem] text-muted-foreground">
                          {formatDate(item.created_at)}
                        </p>
                      </div>
                      <span className="shrink-0 text-muted-foreground group-hover:text-teal-600 dark:group-hover:text-teal-400 transition">
                        <ArrowRight size={14} />
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </aside>
        </div>
      )}

      {/* History Cards Tab */}
      {activeTab === "history" && (
        <section aria-label="Analysis History">
          <div className="flex items-center justify-between mb-5">
            <div>
              <h2 className="text-lg font-bold text-foreground">Analysis History</h2>
              <p className="text-xs text-muted-foreground">
                Click any previously evaluated network to reopen it directly in the interactive
                workspace.
              </p>
            </div>
            <button
              type="button"
              onClick={() => void loadHistory()}
              className="inline-flex items-center gap-1.5 text-xs font-medium text-foreground hover:text-teal-600 dark:hover:text-teal-400 bg-card border border-border rounded-lg px-3 py-1.5 shadow-2xs transition"
            >
              <RefreshCw size={13} />
              <span>Refresh</span>
            </button>
          </div>

          {actionFeedback && (
            <div
              className={cn(
                "mb-4 flex items-center justify-between rounded-lg p-3 text-xs border shadow-2xs transition",
                actionFeedback.type === "success"
                  ? "bg-teal-500/10 text-teal-300 border-teal-500/30"
                  : "bg-destructive/10 text-destructive border-destructive/30",
              )}
            >
              <div className="flex items-center gap-2">
                {actionFeedback.type === "success" ? (
                  <CheckCircle2 size={14} className="text-teal-400 shrink-0" />
                ) : (
                  <AlertCircle size={14} className="text-destructive shrink-0" />
                )}
                <span>{actionFeedback.message}</span>
              </div>
              <button
                type="button"
                onClick={() => setActionFeedback(null)}
                className="text-xs opacity-70 hover:opacity-100 px-1 py-0.5 rounded cursor-pointer"
                aria-label="Dismiss message"
              >
                ✕
              </button>
            </div>
          )}

          {historyLoading ? (
            <div className="py-16 text-center">
              <LoaderCircle className="h-6 w-6 animate-spin mx-auto text-primary mb-2" />
              <p className="text-xs text-muted-foreground">Retrieving past analyses...</p>
            </div>
          ) : history.length === 0 ? (
            <div className="rounded-xl border border-dashed border-border bg-card p-12 text-center max-w-md mx-auto">
              <div className="w-12 h-12 rounded-xl bg-muted text-muted-foreground flex items-center justify-center mx-auto mb-3">
                <FolderOpen size={24} />
              </div>
              <h3 className="text-sm font-bold text-foreground">No Previous Analyses</h3>
              <p className="mt-1 text-xs text-muted-foreground">
                Upload a satellite image using the "New Upload" tab to analyze and save networks.
              </p>
              <button
                type="button"
                onClick={() => setActiveTab("new")}
                className="mt-4 inline-flex items-center gap-1.5 rounded-lg bg-primary px-4 py-2 text-xs font-semibold text-primary-foreground shadow-xs hover:bg-primary/90"
              >
                <Plus size={14} />
                <span>Upload an image</span>
              </button>
            </div>
          ) : (
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {history.map((item) => {
                const thumb = item.artifacts?.original ?? item.input_image_ref;
                const nodeCount = item.summary?.after_healing?.n_nodes ?? "—";
                const edgeCount = item.summary?.after_healing?.n_edges ?? "—";
                const compCount = item.summary?.after_healing?.n_components ?? "—";

                return (
                  <div
                    key={item.analysis_id}
                    onClick={() => reopenAnalysis(item)}
                    className="group rounded-xl border border-border bg-card p-4 shadow-xs hover:border-teal-500/60 hover:shadow-sm transition cursor-pointer flex flex-col justify-between"
                  >
                    <div>
                      {/* Image Thumbnail */}
                      <div className="relative aspect-video w-full rounded-lg bg-[#0a0d12] overflow-hidden mb-3 border border-border">
                        {thumb ? (
                          <img
                            src={thumb}
                            alt={item.input_image_name ?? "Satellite preview"}
                            className="h-full w-full object-cover group-hover:scale-105 transition duration-300 satellite-imagery"
                            style={{ filter: "none" }}
                            loading="lazy"
                          />
                        ) : (
                          <div className="h-full w-full flex items-center justify-center text-muted-foreground font-mono text-xs">
                            <Layers size={24} />
                          </div>
                        )}
                        <span className="absolute top-2 left-2 rounded-md bg-slate-900/85 backdrop-blur-xs px-2 py-0.5 text-[0.62rem] font-mono font-medium text-white border border-slate-700/50">
                          #{item.analysis_id.slice(0, 8)}
                        </span>
                        <div className="absolute top-2 right-2 flex items-center gap-1.5 z-10">
                          <span className="rounded-md bg-teal-600/90 px-2 py-0.5 text-[0.62rem] font-semibold text-white">
                            Completed
                          </span>
                          <button
                            type="button"
                            aria-label={`Delete analysis ${item.input_image_name ?? item.analysis_id.slice(0, 8)}`}
                            title="Delete Analysis"
                            onClick={(e) => promptDelete(e, item)}
                            className="rounded-md bg-slate-900/85 hover:bg-rose-600 text-slate-300 hover:text-white p-1 backdrop-blur-xs transition border border-slate-700/50 shadow-xs cursor-pointer"
                          >
                            <Trash2 size={12} />
                          </button>
                        </div>
                      </div>

                      {/* Title & Date */}
                      <h3 className="font-semibold text-sm text-foreground truncate group-hover:text-teal-600 dark:group-hover:text-teal-400 transition">
                        {item.input_image_name ?? `Analysis #${item.analysis_id.slice(0, 8)}`}
                      </h3>
                      <p className="mt-1 text-xs text-muted-foreground flex items-center gap-1.5">
                        <Clock size={12} />
                        <span>{formatDate(item.created_at)}</span>
                      </p>

                      {/* Metrics summary */}
                      <div className="mt-3 grid grid-cols-3 gap-1.5 text-center text-xs">
                        <div className="rounded-lg bg-muted/40 p-1.5 border border-border">
                          <span className="block text-[0.58rem] font-semibold text-muted-foreground uppercase">
                            Nodes
                          </span>
                          <span className="font-bold text-foreground font-mono">{nodeCount}</span>
                        </div>
                        <div className="rounded-lg bg-muted/40 p-1.5 border border-border">
                          <span className="block text-[0.58rem] font-semibold text-muted-foreground uppercase">
                            Roads
                          </span>
                          <span className="font-bold text-foreground font-mono">{edgeCount}</span>
                        </div>
                        <div className="rounded-lg bg-muted/40 p-1.5 border border-border">
                          <span className="block text-[0.58rem] font-semibold text-muted-foreground uppercase">
                            Zones
                          </span>
                          <span className="font-bold text-foreground font-mono">{compCount}</span>
                        </div>
                      </div>
                    </div>

                    {/* Card Footer CTA */}
                    <div className="mt-4 pt-3 border-t border-border flex items-center justify-between text-xs font-semibold text-teal-600 dark:text-teal-400">
                      <span>Open Network Workspace</span>
                      <ArrowRight
                        size={14}
                        className="group-hover:translate-x-1 transition-transform"
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </section>
      )}

      {/* Delete Confirmation Alert Dialog */}
      <AlertDialog
        open={itemToDelete !== null}
        onOpenChange={(open) => !open && !isDeleting && setItemToDelete(null)}
      >
        <AlertDialogContent className="bg-card border-border text-foreground">
          <AlertDialogHeader>
            <AlertDialogTitle className="text-foreground">Delete Analysis?</AlertDialogTitle>
            <AlertDialogDescription className="text-muted-foreground">
              Are you sure you want to permanently delete analysis{" "}
              <strong className="text-foreground">
                {itemToDelete?.input_image_name ?? `#${itemToDelete?.analysis_id.slice(0, 8)}`}
              </strong>{" "}
              <span className="font-mono text-xs opacity-80">({itemToDelete?.analysis_id})</span>?
              This will remove its extracted network, visual artifacts, and disruption results from storage.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel
              disabled={isDeleting}
              className="bg-muted text-foreground border-border hover:bg-muted/80"
            >
              Cancel
            </AlertDialogCancel>
            <AlertDialogAction
              disabled={isDeleting}
              onClick={(e) => {
                e.preventDefault();
                void confirmDelete();
              }}
              className="bg-rose-600 text-white hover:bg-rose-700 font-semibold"
            >
              {isDeleting ? "Deleting..." : "Delete Analysis"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </main>
  );
}
