import { createFileRoute, Link, useLocation } from "@tanstack/react-router";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  ChevronDown,
  ChevronUp,
  GitBranch,
  Info,
  LoaderCircle,
  Network,
  ShieldAlert,
  Zap,
} from "lucide-react";
import { useEffect, useState } from "react";

import { Navbar } from "@/components/Navbar";

type ModeKey = "detected" | "extracted" | "repaired" | "critical";
type LoadState = "loading" | "success" | "error" | "missing";

type GraphSummary = {
  n_nodes: number;
  n_edges: number;
  n_components: number;
  total_length_px: number;
  skeleton_connectivity_ratio?: number;
};

type AnalysisResponse = {
  analysis_id: string;
  status: string;
  artifacts: {
    binary_mask?: string;
    skeleton?: string;
    repaired_network?: string;
    critical_roads?: string;
    visualization?: string;
    graph_json?: string;
    healed_graph_json?: string;
  };
  summary: {
    threshold: number;
    before_healing: GraphSummary;
    after_healing: GraphSummary;
    delta_after_minus_before: Record<string, number>;
    healed_edge_count: number;
    coordinate_system: string;
  };
};

type CriticalityEdge = {
  edge_id: [string | number, string | number] | string[];
  length_px?: number;
  confidence?: number | null;
  healed?: boolean;
  width_px?: number | null;
  edge_betweenness?: number;
  detour_cost_px?: number | null;
  detour_disconnected?: boolean;
  is_bridge?: boolean;
  standalone_efficiency_drop?: number;
  standalone_component_delta?: number;
};

type CriticalityReport = {
  edge_criticality?: CriticalityEdge[];
  top_vulnerable_edges_by_betweenness?: Array<[string | number, string | number] | string[]>;
};

type CriticalityResponse = {
  analysis_id: string;
  status: string;
  criticality: CriticalityReport;
  artifacts?: {
    critical_roads?: string;
  };
};

const MODES: { key: ModeKey; label: string; desc: string }[] = [
  { key: "detected", label: "Detected Roads", desc: "Binary segmentation mask from deep model" },
  { key: "extracted", label: "Extracted Network", desc: "Initial structural graph before topology healing" },
  { key: "repaired", label: "Repaired Network", desc: "Healed graph topology with bridged road gaps" },
  { key: "critical", label: "Critical Roads", desc: "Ranked critical edges from the API" },
];

export const Route = createFileRoute("/network")({
  head: () => ({
    meta: [
      { title: "Network Analysis — ROUTE RESILIENCE" },
      { name: "description", content: "Graph extraction, topology repair, and critical road analysis." },
    ],
  }),
  component: NetworkPage,
});

function MetricCard({
  icon,
  label,
  value,
  sub,
  highlight,
}: {
  icon: React.ReactNode;
  label: string;
  value: string | number;
  sub?: string;
  highlight?: boolean;
}) {
  return (
    <div className={`group relative overflow-hidden border p-4 backdrop-blur-sm transition-all duration-300 ${highlight ? "border-primary/40 bg-card/90" : "border-border/60 bg-card/70 hover:border-primary/30 hover:bg-card"}`}>
      <div className="relative z-10 flex h-full flex-col justify-between">
        <div className="mb-2 flex items-center justify-between text-muted-foreground">
          <span className="font-display text-[0.58rem] tracking-[0.18em] uppercase">{label}</span>
          <div className="text-primary/70">{icon}</div>
        </div>
        <div>
          <p className="font-display text-2xl font-bold tracking-tight text-primary">{value}</p>
          {sub && <p className="mt-1 font-display text-[0.58rem] tracking-[0.14em] text-muted-foreground uppercase">{sub}</p>}
        </div>
      </div>
    </div>
  );
}

function NetworkPage() {
  const { searchStr } = useLocation();
  const analysisId = new URLSearchParams(searchStr).get("analysis") ?? undefined;
  const [loadState, setLoadState] = useState<LoadState>(analysisId ? "loading" : "missing");
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [criticalityState, setCriticalityState] = useState<LoadState>(analysisId ? "loading" : "missing");
  const [criticality, setCriticality] = useState<CriticalityResponse | null>(null);
  const [criticalityError, setCriticalityError] = useState<string | null>(null);
  const [mode, setMode] = useState<ModeKey>("repaired");
  const [detailsOpen, setDetailsOpen] = useState(false);

  useEffect(() => {
    if (!analysisId) {
      setAnalysis(null);
      setErrorMessage(null);
      setLoadState("missing");
      return;
    }

    const controller = new AbortController();
    setLoadState("loading");
    setAnalysis(null);
    setErrorMessage(null);

    fetch(`http://127.0.0.1:8000/api/analysis/${encodeURIComponent(analysisId)}`, { signal: controller.signal })
      .then(async (response) => {
        const payload = (await response.json()) as AnalysisResponse | { detail?: string };
        if (!response.ok) {
          const detail = "detail" in payload ? payload.detail : undefined;
          throw new Error(typeof detail === "string" ? detail : `Analysis request failed (${response.status}).`);
        }
        return payload as AnalysisResponse;
      })
      .then((payload) => {
        setAnalysis(payload);
        setLoadState("success");
      })
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === "AbortError") return;
        setErrorMessage(error instanceof Error ? error.message : "Unable to load this analysis.");
        setLoadState("error");
      });

    return () => controller.abort();
  }, [analysisId]);

  useEffect(() => {
    if (!analysisId) {
      setCriticality(null);
      setCriticalityError(null);
      setCriticalityState("missing");
      return;
    }

    const controller = new AbortController();
    setCriticalityState("loading");
    setCriticality(null);
    setCriticalityError(null);

    fetch(`http://127.0.0.1:8000/api/analysis/${encodeURIComponent(analysisId)}/criticality`, { signal: controller.signal })
      .then(async (response) => {
        const payload = (await response.json()) as CriticalityResponse | { detail?: string };
        if (!response.ok) {
          const detail = "detail" in payload ? payload.detail : undefined;
          throw new Error(typeof detail === "string" ? detail : `Criticality request failed (${response.status}).`);
        }
        return payload as CriticalityResponse;
      })
      .then((payload) => {
        if (payload.analysis_id !== analysisId || !payload.criticality) {
          throw new Error("The criticality response is incomplete.");
        }
        setCriticality(payload);
        setCriticalityState("success");
      })
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === "AbortError") return;
        setCriticalityError(error instanceof Error ? error.message : "Unable to load criticality analysis.");
        setCriticalityState("error");
      });

    return () => controller.abort();
  }, [analysisId]);

  const summary = analysis?.summary;
  const before = summary?.before_healing;
  const after = summary?.after_healing;
  const artifacts = analysis?.artifacts;
  const connectivity = after?.skeleton_connectivity_ratio;
  const delta = summary?.delta_after_minus_before;
  const imageSource = mode === "detected" ? artifacts?.binary_mask : mode === "extracted" ? artifacts?.skeleton : mode === "repaired" ? artifacts?.repaired_network : mode === "critical" ? artifacts?.critical_roads ?? criticality?.artifacts?.critical_roads : undefined;
  const activeMode = MODES.find((item) => item.key === mode);
  const criticalEdges = criticality?.criticality.edge_criticality?.slice(0, 10) ?? [];

  return (
    <div className="relative min-h-screen bg-background text-foreground selection:bg-primary/20">
      <div aria-hidden className="explore-bg-grid pointer-events-none absolute inset-0 opacity-[0.04]" />
      <div aria-hidden className="explore-bg-radial pointer-events-none absolute inset-0" />
      <Navbar />

      <main className="relative z-10 px-6 py-6 sm:px-10 lg:px-14">
        <div className="mb-6 flex flex-col gap-4 border-b border-border/40 pb-6 md:flex-row md:items-end md:justify-between">
          <div>
            <p className="mb-1.5 font-display text-[0.58rem] tracking-[0.28em] text-primary/70 uppercase">ISRO <span className="px-1 text-foreground/40">•</span> BAH 2026 <span className="px-1 text-foreground/40">•</span> GRAPH TOPOLOGY</p>
            <h1 className="font-display text-3xl font-light tracking-tight text-foreground sm:text-4xl">Road Network <span className="font-medium text-primary">Analysis</span></h1>
          </div>
          <div className="flex items-center gap-2 rounded-lg border border-border/50 bg-card/60 p-2 text-[0.6rem] font-display tracking-[0.14em] uppercase">
            <span className="flex items-center gap-1.5 px-2.5 py-1 text-muted-foreground"><span className="h-1.5 w-1.5 rounded-full bg-muted-foreground/60" /> 1. Detection</span>
            <span className="text-border/60">→</span>
            <span className="flex items-center gap-1.5 rounded border border-primary/30 bg-primary/10 px-2.5 py-1 font-medium text-primary"><Network size={12} /> 2. Network Graph</span>
            <span className="text-border/60">→</span>
            <span className="flex items-center gap-1.5 px-2.5 py-1 text-muted-foreground"><Zap size={12} /> 3. Topology Repair</span>
          </div>
        </div>

        <section className="mb-6 flex flex-wrap items-center justify-between gap-3 border border-border/60 bg-card/50 p-4" aria-label="Analysis status">
          <div>
            <p className="font-display text-[0.58rem] tracking-[0.22em] text-muted-foreground uppercase">CURRENT ANALYSIS</p>
            <p className="mt-2 break-all font-display text-sm text-primary">{analysisId ?? "No analysis selected"}</p>
          </div>
          <span className={`border px-3 py-2 font-display text-[0.56rem] tracking-[0.16em] uppercase ${loadState === "success" ? "border-primary/40 bg-primary/10 text-primary" : "border-border/60 text-muted-foreground"}`}>
            {loadState === "success" ? "ANALYSIS LOADED" : loadState === "loading" ? "LOADING" : loadState === "missing" ? "NO ANALYSIS" : "LOAD ERROR"}
          </span>
        </section>

        {loadState === "loading" && <div className="flex min-h-[420px] items-center justify-center border border-border/50 bg-card/30 p-10 text-center"><div><LoaderCircle className="mx-auto h-8 w-8 animate-spin text-primary" /><p className="mt-4 font-display text-[0.7rem] tracking-[0.18em] text-muted-foreground uppercase">Loading analysis results</p></div></div>}
        {loadState === "missing" && <div className="flex min-h-[420px] items-center justify-center border border-border/50 bg-card/30 p-10 text-center"><div><Info className="mx-auto h-10 w-10 text-muted-foreground" /><p className="mt-4 font-display text-[0.7rem] tracking-[0.18em] text-muted-foreground uppercase">Run an analysis in Explore first</p><Link to="/explore" className="mt-5 inline-flex border border-primary/70 bg-primary/10 px-4 py-3 font-display text-[0.6rem] tracking-[0.16em] text-primary uppercase">GO TO EXPLORE</Link></div></div>}
        {loadState === "error" && <div className="flex min-h-[420px] items-center justify-center border border-amber-500/40 bg-amber-500/5 p-10 text-center text-amber-100"><div><AlertTriangle className="mx-auto h-8 w-8 text-amber-400" /><p className="mt-4 font-display text-[0.7rem] tracking-[0.18em] text-amber-200 uppercase">Unable to load analysis</p><p className="mt-3 max-w-xl text-sm">{errorMessage}</p></div></div>}

        {loadState === "success" && analysis && summary && before && after && (
          <>
            <section className="mb-6 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5" aria-label="Key network metrics">
              <MetricCard icon={<GitBranch size={16} />} label="Nodes" value={after.n_nodes} sub="Healed graph" />
              <MetricCard icon={<Network size={16} />} label="Edges" value={after.n_edges} sub={`Before: ${before.n_edges}`} />
              <MetricCard icon={<Activity size={16} />} label="Connectivity" value={connectivity !== undefined ? `${(connectivity * 100).toFixed(1)}%` : "N/A"} sub="Skeleton ratio" highlight />
              <MetricCard icon={<Zap size={16} />} label="Healed Edges" value={`+${summary.healed_edge_count}`} sub={summary.healed_edge_count > 0 ? "Gaps bridged" : "No gaps detected"} highlight={summary.healed_edge_count > 0} />
              <MetricCard icon={<GitBranch size={16} />} label="Network Length" value={`${after.total_length_px.toFixed(1)} px`} sub="Healed graph total" />
            </section>

            <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
              <section className="flex flex-col lg:col-span-8" aria-label="Network visualization">
                <div className="flex flex-wrap items-center gap-1 border-b border-border/50 bg-card/40 p-1.5 backdrop-blur-sm">{MODES.map((tab) => <button key={tab.key} type="button" onClick={() => setMode(tab.key)} className={`min-w-[120px] flex-1 px-3 py-2 text-center font-display text-[0.6rem] font-medium tracking-[0.16em] uppercase transition-all ${mode === tab.key ? "bg-primary text-primary-foreground shadow-sm" : "text-foreground/60 hover:bg-secondary hover:text-foreground"}`}>{tab.label}</button>)}</div>
                <div className="relative aspect-[4/3] w-full overflow-hidden border border-t-0 border-border/60 bg-black/40 backdrop-blur-md sm:aspect-[16/10]">
                  {imageSource ? <img src={imageSource} alt={`${activeMode?.label} visualization`} className="h-full w-full object-cover transition-all duration-500" /> : <div className="flex h-full items-center justify-center p-8 text-center"><p className="font-display text-[0.65rem] tracking-[0.14em] text-muted-foreground uppercase">Critical road image not available</p></div>}
                  {mode === "repaired" && summary.healed_edge_count > 0 && <div className="absolute right-4 top-4 flex items-center gap-2 rounded border border-emerald-500/40 bg-emerald-950/80 px-3 py-1.5 text-[0.6rem] font-display tracking-widest text-emerald-400"><span className="h-2 w-2 animate-pulse rounded-full bg-emerald-400" /> HEALED TOPOLOGY ({summary.healed_edge_count} BRIDGED)</div>}
                  <div className="absolute inset-x-0 bottom-0 flex items-end justify-between bg-gradient-to-t from-black/90 via-black/50 to-transparent p-4"><div><p className="font-display text-[0.62rem] font-semibold tracking-wider text-primary uppercase">{activeMode?.label}</p><p className="font-display text-[0.55rem] tracking-wide text-muted-foreground">{activeMode?.desc}</p></div><div className="font-display text-[0.55rem] tracking-wider text-muted-foreground/80 uppercase">ANALYSIS {analysis.analysis_id}</div></div>
                </div>

                <div className="mt-4 border border-border/60 bg-card/60 p-4 backdrop-blur-sm">
                  <div className="mb-3 flex items-center justify-between"><span className="flex items-center gap-2 font-display text-[0.6rem] font-semibold tracking-widest text-primary uppercase"><GitBranch size={13} /> BEFORE / AFTER TOPOLOGY COMPARISON</span><span className="font-display text-[0.55rem] text-muted-foreground uppercase">HEALING IMPACT</span></div>
                  <div className="grid grid-cols-2 gap-4"><div className="border border-border/40 bg-background/50 p-3"><p className="mb-1 font-display text-[0.58rem] tracking-wider text-muted-foreground uppercase">Extracted Graph (Raw)</p><span className="font-display text-lg font-bold text-foreground">{before.n_components} Components</span><span className="ml-2 font-display text-[0.58rem] text-muted-foreground">({before.n_edges} edges)</span></div><div className="border border-primary/30 bg-primary/5 p-3"><p className="mb-1 font-display text-[0.58rem] font-medium tracking-wider text-primary uppercase">Repaired Graph (Healed)</p><span className="font-display text-lg font-bold text-primary">{after.n_components} Components</span><span className="ml-2 font-display text-[0.58rem] text-primary/80">({after.n_edges} edges, +{summary.healed_edge_count} healed)</span></div></div>
                  <p className="mt-3 font-display text-[0.55rem] tracking-wide text-muted-foreground uppercase">Delta: {delta?.["n_components"] ?? 0} components · {delta?.["n_edges"] ?? 0} edges · {(delta?.["total_length_px"] ?? 0).toFixed(1)} px length</p>
                </div>
              </section>

              <aside className="flex flex-col gap-4 lg:col-span-4" aria-label="Critical roads and technical details">
                <div className="border border-border/60 bg-card/70 p-4 backdrop-blur-sm">
                  <div className="mb-3 flex items-center justify-between border-b border-border/40 pb-2">
                    <span className="flex items-center gap-1.5 font-display text-[0.62rem] font-semibold tracking-wider text-primary uppercase"><ShieldAlert size={14} className="text-amber-400" /> TOP CRITICAL ROAD SEGMENTS</span>
                    <span className="font-display text-[0.55rem] text-muted-foreground uppercase">RANKED BY API</span>
                  </div>
                  {criticalityState === "loading" && <div className="flex items-center gap-2 border border-dashed border-border/60 p-6 text-center text-sm text-muted-foreground"><LoaderCircle className="h-4 w-4 animate-spin text-primary" /> Loading criticality analysis</div>}
                  {criticalityState === "error" && <div className="border border-amber-500/40 bg-amber-500/5 p-5 text-amber-100"><div className="flex items-center gap-2"><AlertTriangle className="h-4 w-4 text-amber-400" /><p className="font-display text-[0.62rem] tracking-[0.16em] uppercase">CRITICALITY ERROR</p></div><p className="mt-3 text-sm">{criticalityError}</p></div>}
                  {criticalityState === "success" && criticalEdges.length === 0 && <div className="border border-dashed border-border/60 p-6 text-center"><Info size={18} className="mx-auto mb-2 text-muted-foreground/60" /><p className="font-display text-[0.62rem] font-medium text-muted-foreground">Insufficient network data</p><p className="mt-1 font-display text-[0.55rem] text-muted-foreground/70">No critical edges were returned for this graph.</p></div>}
                  {criticalityState === "success" && criticalEdges.length > 0 && <div className="space-y-3">{criticalEdges.map((edge, index) => { const source = edge.edge_id?.[0]; const target = edge.edge_id?.[1]; return <div key={`${String(source)}-${String(target)}-${index}`} className="border border-border/50 bg-background/40 p-3 transition-colors hover:border-amber-500/40 hover:bg-background/80"><div className="flex items-center justify-between gap-2"><div className="flex items-center gap-2"><span className="flex h-5 w-5 items-center justify-center rounded bg-amber-500/20 font-display text-[0.6rem] font-bold text-amber-400">#{index + 1}</span><span className="font-display text-[0.65rem] font-semibold text-foreground">{String(source)} ↔ {String(target)}</span></div>{edge.is_bridge !== undefined && <span className="rounded border border-amber-500/30 bg-amber-500/10 px-2 py-0.5 font-display text-[0.52rem] font-medium text-amber-400 uppercase">{edge.is_bridge ? "BRIDGE" : "NON-BRIDGE"}</span>}</div><div className="mt-2 grid gap-1 text-[0.57rem] font-display tracking-wide text-muted-foreground sm:grid-cols-2">{edge.edge_betweenness !== undefined && <span>Betweenness: <strong className="text-primary">{edge.edge_betweenness.toFixed(4)}</strong></span>}{edge.detour_cost_px !== undefined && <span>Detour: <strong className="text-primary">{edge.detour_cost_px === null ? "Disconnected" : `${edge.detour_cost_px.toFixed(2)} px`}</strong></span>}{edge.standalone_efficiency_drop !== undefined && <span>Efficiency impact: <strong className="text-primary">{edge.standalone_efficiency_drop.toFixed(4)}</strong></span>}{edge.length_px !== undefined && <span>Length: <strong className="text-primary">{edge.length_px.toFixed(2)} px</strong></span>}</div></div>; })}</div>}
                </div>

                <div className="border border-border/60 bg-card/70 p-4 backdrop-blur-sm"><button type="button" onClick={() => setDetailsOpen(!detailsOpen)} className="flex w-full items-center justify-between text-left"><span className="flex items-center gap-1.5 font-display text-[0.6rem] font-semibold tracking-wider text-primary uppercase"><Info size={13} /> TECHNICAL GRAPH METRICS</span>{detailsOpen ? <ChevronUp size={14} /> : <ChevronDown size={14} />}</button>{detailsOpen && <div className="mt-3 space-y-2 border-t border-border/40 pt-3 font-display text-[0.58rem] tracking-wide text-muted-foreground"><div className="flex justify-between gap-4 border-b border-border/20 pb-1"><span>Coordinate System:</span><span className="text-right font-medium text-foreground">{summary.coordinate_system}</span></div><div className="flex justify-between border-b border-border/20 pb-1"><span>Binarization Threshold:</span><span className="font-medium text-foreground">{summary.threshold}</span></div><div className="flex justify-between border-b border-border/20 pb-1"><span>Total Segment Length:</span><span className="font-medium text-foreground">{after.total_length_px.toFixed(1)} px</span></div><div className="flex justify-between border-b border-border/20 pb-1"><span>Graph JSON:</span><a href={artifacts?.graph_json} target="_blank" rel="noreferrer" className="font-medium text-primary hover:underline">OPEN</a></div><div className="flex justify-between"><span>Healed Graph JSON:</span><a href={artifacts?.healed_graph_json} target="_blank" rel="noreferrer" className="font-medium text-primary hover:underline">OPEN</a></div></div>}</div>
              </aside>
            </div>
          </>
        )}

        <footer className="mt-10 flex flex-wrap items-center justify-between gap-4 border-t border-border/40 pt-6"><Link to="/explore" className="group flex items-center gap-2 font-display text-[0.62rem] tracking-[0.18em] text-foreground/70 transition-colors hover:text-primary uppercase">← BACK TO EXPLORE</Link><Link to="/disruption" search={analysisId ? { analysis: analysisId } : {}} className="group flex items-center gap-2 font-display text-[0.62rem] tracking-[0.18em] text-foreground/70 transition-colors hover:text-primary uppercase"><span>NEXT: DISRUPTION ANALYSIS</span><span className="flex h-7 w-7 items-center justify-center border border-border/70 text-primary transition-colors group-hover:border-primary"><ArrowRight size={13} /></span></Link></footer>
      </main>
    </div>
  );
}
