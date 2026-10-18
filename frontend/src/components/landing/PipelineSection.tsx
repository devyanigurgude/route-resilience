import {
  ArrowRight,
  CheckCircle2,
  Cpu,
  GitBranch,
  Layers,
  Satellite,
  ShieldAlert,
  Sparkles,
} from "lucide-react";
import { useState } from "react";
import terrainAsset from "@/assets/route-resilience-terrain.jpg";

const STEPS = [
  {
    step: "01",
    title: "Satellite Ingestion",
    tagline: "Earth Observation Ingestion",
    description:
      "Captures high-resolution optical and radar satellite imagery across rugged rural, mountainous, and coastal terrains.",
    icon: Satellite,
    badge: "Input Stream",
    stats: "0.5m Spatial Resolution",
    visual: {
      type: "image",
      src: terrainAsset,
      alt: "Raw satellite remote sensing imagery",
      caption: "Raw Optical Terrain Tile (ISRO Benchmark)",
    },
  },
  {
    step: "02",
    title: "AI Road Segmentation",
    tagline: "Pixel-Level Neural Extraction",
    description:
      "Deep convolutional neural networks detect obscured, unpaved, and canopy-covered roads into continuous probability masks.",
    icon: Cpu,
    badge: "Deep Learning",
    stats: "93.8% Mean IoU",
    visual: {
      type: "image",
      src: "/scene_10013_mask.png",
      alt: "Binary road segmentation mask",
      caption: "Binarized Neural Mask (DL Probability Field)",
    },
  },
  {
    step: "03",
    title: "Graph Extraction & Healing",
    tagline: "Topological Vectorization",
    description:
      "Morphological thinning converts road pixels to graph centerlines, automatically bridging occlusions and detecting road junctions.",
    icon: GitBranch,
    badge: "Topology Engine",
    stats: "248 Nodes • 312 Edges",
    visual: {
      type: "image",
      src: "/scene_10013_skeleton.png",
      alt: "Graph skeleton representation",
      caption: "Pruned Centerline Skeleton & Nodes",
    },
  },
  {
    step: "04",
    title: "Criticality & Failure Modeling",
    tagline: "Percolation Stress-Testing",
    description:
      "Applies Betweenness Centrality and edge percolation to identify single points of failure before landslides or floods strike.",
    icon: ShieldAlert,
    badge: "Disruption Engine",
    stats: "Top 1% Vulnerable Corridors",
    visual: {
      type: "image",
      src: "/scene_10013_visualization.png",
      alt: "Network criticality visualization",
      caption: "Disruption Heatmap & Severance Analysis",
    },
  },
  {
    step: "05",
    title: "Recovery & Alternate Routing",
    tagline: "Automated Pathfinding",
    description:
      "Computes dynamic detour routes and schedules optimal multi-step restoration to restore 90%+ connectivity with minimal budget.",
    icon: CheckCircle2,
    badge: "Resilience Planner",
    stats: "88.6 Resilience Index",
    visual: {
      type: "custom-recovery",
      caption: "Optimal Detour Path & Resilience Curve",
    },
  },
];

export function PipelineSection() {
  const [activeStepIndex, setActiveStepIndex] = useState(0);
  const current = STEPS[activeStepIndex];

  return (
    <section id="technology" className="py-24 sm:py-32 bg-white relative overflow-hidden border-b border-slate-100">
      <div className="max-w-7xl mx-auto px-6 sm:px-8 lg:px-12">
        {/* Section Header */}
        <div className="max-w-3xl mb-16 sm:mb-20">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-50 border border-blue-200/50 text-[#0066FF] font-display text-[0.66rem] font-bold tracking-widest uppercase mb-4">
            <span>5-STEP ARCHITECTURE</span>
          </div>
          <h2 className="font-display text-3xl sm:text-5xl font-bold tracking-tight text-[#0A192F] leading-tight mb-5">
            From satellite pixels to resilient networks.
          </h2>
          <p className="text-base sm:text-lg text-slate-600 font-normal leading-relaxed">
            Our pipeline bridges cutting-edge computer vision with structural graph theory to turn raw
            aerial imagery into actionable, stress-tested transportation graphs.
          </p>
        </div>

        {/* 5-Step Progress Tabs */}
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 mb-10">
          {STEPS.map((item, idx) => {
            const isActive = idx === activeStepIndex;
            const Icon = item.icon;
            return (
              <button
                key={item.step}
                type="button"
                onClick={() => setActiveStepIndex(idx)}
                className={`text-left p-4 rounded-2xl border transition-all duration-200 cursor-pointer ${
                  isActive
                    ? "bg-[#FAFBFD] border-[#0066FF] shadow-sm ring-1 ring-[#0066FF]/20"
                    : "bg-white border-slate-200/80 hover:border-slate-300 hover:bg-slate-50/50"
                }`}
              >
                <div className="flex items-center justify-between mb-3">
                  <span
                    className={`font-display text-xs font-bold ${
                      isActive ? "text-[#0066FF]" : "text-slate-400"
                    }`}
                  >
                    {item.step}
                  </span>
                  <div
                    className={`w-7 h-7 rounded-lg flex items-center justify-center transition-colors ${
                      isActive
                        ? "bg-[#0066FF] text-white"
                        : "bg-slate-100 text-slate-500"
                    }`}
                  >
                    <Icon size={14} />
                  </div>
                </div>
                <h3
                  className={`font-display text-xs sm:text-sm font-semibold tracking-tight truncate ${
                    isActive ? "text-[#0A192F]" : "text-slate-700"
                  }`}
                >
                  {item.title}
                </h3>
              </button>
            );
          })}
        </div>

        {/* Active Step Showcase Card */}
        <div className="grid lg:grid-cols-12 gap-8 items-center bg-[#FAFBFD] rounded-3xl border border-slate-200/90 p-6 sm:p-10 shadow-sm">
          {/* Left Description Column */}
          <div className="lg:col-span-5 space-y-6">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-100/60 text-[#0066FF] font-semibold text-xs">
              <Sparkles size={13} />
              <span>{current.badge}</span>
            </div>

            <div>
              <p className="font-display text-xs font-bold text-slate-400 uppercase tracking-widest mb-1">
                STEP {current.step} — {current.tagline}
              </p>
              <h3 className="font-display text-2xl sm:text-3xl font-bold text-[#0A192F] tracking-tight">
                {current.title}
              </h3>
            </div>

            <p className="text-slate-600 text-sm sm:text-base leading-relaxed">
              {current.description}
            </p>

            {/* Metrics highlight */}
            <div className="p-4 rounded-xl bg-white border border-slate-200/70 shadow-2xs flex items-center justify-between">
              <span className="text-xs font-medium text-slate-500">Pipeline Performance:</span>
              <span className="font-display text-sm font-bold text-[#0066FF]">
                {current.stats}
              </span>
            </div>

            {/* Next / Previous Controls */}
            <div className="flex items-center gap-3 pt-2">
              <button
                type="button"
                onClick={() =>
                  setActiveStepIndex((prev) => (prev > 0 ? prev - 1 : STEPS.length - 1))
                }
                className="px-4 py-2 rounded-full border border-slate-200 text-xs font-semibold text-slate-700 hover:bg-white cursor-pointer"
              >
                ← Previous
              </button>
              <button
                type="button"
                onClick={() =>
                  setActiveStepIndex((prev) => (prev < STEPS.length - 1 ? prev + 1 : 0))
                }
                className="inline-flex items-center gap-1.5 px-4 py-2 rounded-full bg-[#0066FF] text-white text-xs font-semibold hover:bg-blue-600 cursor-pointer shadow-xs"
              >
                <span>Next Step</span>
                <ArrowRight size={13} />
              </button>
            </div>
          </div>

          {/* Right Visual Column */}
          <div className="lg:col-span-7">
            <div className="relative rounded-2xl overflow-hidden border border-slate-200/80 bg-slate-900 shadow-md aspect-[16/10] flex items-center justify-center">
              {current.visual.type === "image" && (
                <img
                  key={current.step}
                  src={current.visual.src}
                  alt={current.visual.alt}
                  className="w-full h-full object-cover animate-in fade-in zoom-in-95 duration-300"
                />
              )}

              {current.visual.type === "custom-recovery" && (
                <div className="w-full h-full p-6 bg-slate-950 flex flex-col justify-between text-white relative">
                  {/* Subtle graph background */}
                  <svg className="absolute inset-0 w-full h-full opacity-40 pointer-events-none" viewBox="0 0 500 300">
                    <path d="M 50 150 Q 150 80 250 150 T 450 150" stroke="#334155" strokeWidth="2" fill="none" />
                    <path d="M 50 150 C 120 220 220 240 320 210 S 420 180 450 150" stroke="#10B981" strokeWidth="4" strokeDasharray="6 4" fill="none" className="animate-road-draw" />
                    <circle cx="50" cy="150" r="6" fill="#0066FF" />
                    <circle cx="450" cy="150" r="6" fill="#10B981" />
                    <circle cx="250" cy="150" r="6" fill="#FF5A36" />
                    <line x1="240" y1="140" x2="260" y2="160" stroke="#FF5A36" strokeWidth="3" />
                    <line x1="260" y1="140" x2="240" y2="160" stroke="#FF5A36" strokeWidth="3" />
                  </svg>

                  <div className="relative z-10 flex items-center justify-between">
                    <span className="text-xs font-semibold text-emerald-400 bg-emerald-500/10 px-2.5 py-1 rounded-md border border-emerald-500/30">
                      Reroute Computed: +24% Detour
                    </span>
                    <span className="text-xs font-bold text-white bg-slate-800 px-3 py-1 rounded-md">
                      Dijkstra Multi-Hop
                    </span>
                  </div>

                  <div className="relative z-10 grid grid-cols-3 gap-3 bg-slate-900/90 backdrop-blur-md p-3.5 rounded-xl border border-slate-800">
                    <div>
                      <p className="text-[0.62rem] text-slate-400 uppercase">Baseline Path</p>
                      <p className="text-sm font-bold text-slate-200">12.4 km</p>
                    </div>
                    <div>
                      <p className="text-[0.62rem] text-amber-400 uppercase">Primary Link</p>
                      <p className="text-sm font-bold text-[#FF5A36]">Severed</p>
                    </div>
                    <div>
                      <p className="text-[0.62rem] text-emerald-400 uppercase">Alternate Detour</p>
                      <p className="text-sm font-bold text-emerald-400">15.4 km</p>
                    </div>
                  </div>
                </div>
              )}

              {/* Caption pill */}
              <div className="absolute bottom-3 left-3 bg-white/90 backdrop-blur-md px-3 py-1 rounded-lg text-slate-800 text-[0.7rem] font-semibold border border-slate-200/80 shadow-xs">
                {current.visual.caption}
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
