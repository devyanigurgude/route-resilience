import { createFileRoute, Link } from "@tanstack/react-router";
import {
  Activity,
  ArrowRight,
  Clock3,
  FilePlus2,
  LoaderCircle,
  Server,
  TriangleAlert,
} from "lucide-react";
import { useEffect, useState } from "react";

import { Navbar } from "@/components/Navbar";
import { API_BASE, SAVED_ANALYSIS_KEY } from "@/lib/backend";

type AnalysisRecord = {
  analysis_id: string;
  status: string;
  created_at?: string;
  input_image_name?: string;
  input_image_ref?: string;
  artifacts?: { original?: string };
};

type LoadState = "loading" | "ready" | "empty" | "error";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Route Resilience — Operations" },
      {
        name: "description",
        content: "Operational workspace for road network analysis and resilience.",
      },
    ],
  }),
  component: HomePage,
});

function HomePage() {
  const [analysis, setAnalysis] = useState<AnalysisRecord | null>(null);
  const [loadState, setLoadState] = useState<LoadState>("loading");
  const [error, setError] = useState<string | null>(null);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);

  useEffect(() => {
    const savedAnalysisId = window.localStorage.getItem(SAVED_ANALYSIS_KEY);
    if (!savedAnalysisId) {
      setLoadState("empty");
      setBackendOnline(false);
      return;
    }

    const controller = new AbortController();
    Promise.all([
      fetch(`${API_BASE}/health`, { signal: controller.signal }).then((response) => response.ok),
      fetch(`${API_BASE}/api/analysis/${encodeURIComponent(savedAnalysisId)}`, {
        signal: controller.signal,
      }).then(async (response) => {
        const payload = (await response.json()) as AnalysisRecord | { detail?: string };
        if (!response.ok)
          throw new Error(
            typeof payload.detail === "string"
              ? payload.detail
              : `Analysis request failed (${response.status}).`,
          );
        return payload as AnalysisRecord;
      }),
    ])
      .then(([online, record]) => {
        setBackendOnline(online);
        setAnalysis(record);
        setLoadState("ready");
      })
      .catch((requestError: unknown) => {
        if (requestError instanceof DOMException && requestError.name === "AbortError") return;
        setBackendOnline(false);
        setError(
          requestError instanceof Error
            ? requestError.message
            : "Unable to load the current analysis.",
        );
        setLoadState("error");
      });

    return () => controller.abort();
  }, []);

  const createdAt = analysis?.created_at
    ? new Date(analysis.created_at).toLocaleString()
    : "Timestamp unavailable";
  const thumbnail = analysis?.artifacts?.original ?? analysis?.input_image_ref;

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Navbar />
      <main className="min-h-screen px-5 py-6 sm:px-8 lg:px-10 lg:py-8">
        <header className="mb-8 flex flex-col gap-5 border-b border-border/60 pb-6 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <p className="mb-2 font-display text-[0.58rem] tracking-[0.22em] text-primary uppercase">
              OPERATIONS / HOME
            </p>
            <h1 className="font-display text-3xl font-semibold tracking-tight text-foreground sm:text-4xl">
              Route Resilience
            </h1>
            <p className="mt-2 max-w-xl text-sm text-muted-foreground">
              Monitor road network analyses, criticality, disruption, routing, and recovery from one
              workspace.
            </p>
          </div>
          <div className="flex items-center gap-2 border border-border/60 bg-card/70 px-3 py-2 font-display text-[0.6rem] tracking-[0.14em] uppercase">
            <span
              className={`h-2 w-2 ${backendOnline ? "bg-emerald-400" : "bg-muted-foreground/50"}`}
            />
            <span className="text-muted-foreground">Backend</span>
            <span className={backendOnline ? "text-emerald-400" : "text-muted-foreground"}>
              {backendOnline ? "Online" : "Standby"}
            </span>
          </div>
        </header>

        <section
          className="mb-8 flex flex-col gap-4 border border-primary/30 bg-primary/5 p-5 sm:flex-row sm:items-center sm:justify-between"
          aria-label="New analysis"
        >
          <div>
            <p className="font-display text-[0.62rem] tracking-[0.18em] text-primary uppercase">
              START A WORKFLOW
            </p>
            <p className="mt-2 text-sm text-muted-foreground">
              Upload an image and generate a live road network analysis.
            </p>
          </div>
          <Link
            to="/explore"
            className="inline-flex items-center justify-center gap-2 bg-primary px-4 py-3 font-display text-[0.62rem] font-semibold tracking-[0.16em] text-primary-foreground uppercase transition-colors hover:bg-primary/90"
          >
            <FilePlus2 size={15} /> New Analysis
          </Link>
        </section>

        <section aria-labelledby="recent-analyses-heading">
          <div className="mb-3 flex items-center justify-between">
            <div>
              <p className="font-display text-[0.58rem] tracking-[0.2em] text-muted-foreground uppercase">
                WORKSPACE HISTORY
              </p>
              <h2
                id="recent-analyses-heading"
                className="mt-1 font-display text-lg font-medium text-foreground"
              >
                Recent Analyses
              </h2>
            </div>
            <span className="font-display text-[0.58rem] tracking-[0.14em] text-muted-foreground uppercase">
              Persisted locally
            </span>
          </div>
          {loadState === "loading" && (
            <div className="flex items-center gap-3 border border-border/60 bg-card/50 p-8 text-sm text-muted-foreground">
              <LoaderCircle size={18} className="animate-spin text-primary" /> Loading current
              analysis
            </div>
          )}
          {loadState === "error" && (
            <div className="flex items-start gap-3 border border-amber-500/40 bg-amber-500/5 p-5 text-amber-100">
              <TriangleAlert size={18} className="mt-0.5 text-amber-400" />
              <div>
                <p className="font-display text-[0.62rem] tracking-[0.16em] uppercase">
                  Unable to load analysis
                </p>
                <p className="mt-2 text-sm">{error}</p>
              </div>
            </div>
          )}
          {loadState === "empty" && (
            <div className="border border-dashed border-border/60 bg-card/30 p-10 text-center">
              <Activity size={24} className="mx-auto text-muted-foreground/60" />
              <p className="mt-3 font-display text-sm text-foreground">No analyses yet</p>
              <p className="mt-1 text-sm text-muted-foreground">
                Start with a real image analysis to populate this workspace.
              </p>
              <Link
                to="/explore"
                className="mt-5 inline-flex items-center gap-2 border border-primary/70 bg-primary/10 px-4 py-2.5 font-display text-[0.6rem] tracking-[0.14em] text-primary uppercase"
              >
                <FilePlus2 size={14} /> New Analysis
              </Link>
            </div>
          )}
          {loadState === "ready" && analysis && (
            <article className="grid gap-4 border border-border/60 bg-card/60 p-4 sm:grid-cols-[180px_minmax(0,1fr)_auto] sm:items-center">
              <div className="aspect-[4/3] overflow-hidden border border-border/50 bg-background/60">
                {thumbnail ? (
                  <img
                    src={thumbnail}
                    alt={analysis.input_image_name ?? "Analysis source image"}
                    className="h-full w-full object-cover"
                  />
                ) : (
                  <div className="flex h-full items-center justify-center text-muted-foreground">
                    <Server size={22} />
                  </div>
                )}
              </div>
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="border border-emerald-500/30 bg-emerald-500/10 px-2 py-1 font-display text-[0.55rem] tracking-[0.12em] text-emerald-400 uppercase">
                    {analysis.status}
                  </span>
                  <span className="font-display text-[0.58rem] tracking-[0.12em] text-muted-foreground uppercase">
                    {analysis.input_image_name ?? "Uploaded image"}
                  </span>
                </div>
                <p className="mt-3 break-all font-mono text-sm text-foreground">
                  {analysis.analysis_id}
                </p>
                <p className="mt-2 flex items-center gap-2 text-xs text-muted-foreground">
                  <Clock3 size={13} /> {createdAt}
                </p>
              </div>
              <Link
                to="/network"
                search={{ analysis: analysis.analysis_id }}
                className="inline-flex items-center justify-center gap-2 border border-primary/70 bg-primary/10 px-4 py-3 font-display text-[0.6rem] tracking-[0.14em] text-primary uppercase transition-colors hover:bg-primary/20"
              >
                Open Analysis <ArrowRight size={14} />
              </Link>
            </article>
          )}
        </section>
      </main>
    </div>
  );
}
