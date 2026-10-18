import { Award, BookOpen, BrainCircuit, Cpu, Database, Network, ShieldCheck } from "lucide-react";

export function AboutSection() {
  return (
    <section id="about" className="py-24 sm:py-32 bg-white relative overflow-hidden border-b border-slate-100">
      <div className="max-w-7xl mx-auto px-6 sm:px-8 lg:px-12">
        {/* Section Header */}
        <div className="max-w-3xl mb-16 sm:mb-20">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-50 border border-blue-200/50 text-[#0066FF] font-display text-[0.66rem] font-bold tracking-widest uppercase mb-4">
            <span>METHODOLOGY & ARCHITECTURE</span>
          </div>
          <h2 className="font-display text-3xl sm:text-5xl font-bold tracking-tight text-[#0A192F] leading-tight mb-5">
            Engineered for high-stakes geospatial resilience.
          </h2>
          <p className="text-base sm:text-lg text-slate-600 font-normal leading-relaxed">
            Built as part of advanced deep learning remote sensing research, Route Resilience integrates
            computer vision pixel segmentation with deterministic topological graphs to protect lives
            and lifelines during disasters.
          </p>
        </div>

        {/* 3 Pillars */}
        <div className="grid md:grid-cols-3 gap-8">
          <div className="p-8 rounded-3xl bg-[#FAFBFD] border border-slate-200/90 shadow-2xs hover:shadow-md transition-shadow duration-200 space-y-4">
            <div className="w-12 h-12 rounded-2xl bg-blue-50 text-[#0066FF] flex items-center justify-center">
              <BrainCircuit size={24} />
            </div>
            <h3 className="font-display text-lg font-bold text-[#0A192F]">
              Neural Segmentation
            </h3>
            <p className="text-slate-600 text-sm leading-relaxed">
              Trained on satellite remote sensing benchmarks to extract complex rural, canopy-obscured,
              and degraded road paths where standard OpenStreetMap data is outdated or missing.
            </p>
            <div className="pt-2 text-xs font-semibold text-slate-500">
              PyTorch • DeepLabV3+ • SegFormer
            </div>
          </div>

          <div className="p-8 rounded-3xl bg-[#FAFBFD] border border-slate-200/90 shadow-2xs hover:shadow-md transition-shadow duration-200 space-y-4">
            <div className="w-12 h-12 rounded-2xl bg-indigo-50 text-indigo-600 flex items-center justify-center">
              <Network size={24} />
            </div>
            <h3 className="font-display text-lg font-bold text-[#0A192F]">
              Topological Graph Healing
            </h3>
            <p className="text-slate-600 text-sm leading-relaxed">
              Converts raw pixel masks into clean, navigable mathematical graphs. Connects disjointed
              components and heals small occlusions using morphological skeletonization and Euclidean bridging.
            </p>
            <div className="pt-2 text-xs font-semibold text-slate-500">
              NetworkX • Planar Graphs • Medial Axis
            </div>
          </div>

          <div className="p-8 rounded-3xl bg-[#FAFBFD] border border-slate-200/90 shadow-2xs hover:shadow-md transition-shadow duration-200 space-y-4">
            <div className="w-12 h-12 rounded-2xl bg-orange-50 text-[#FF5A36] flex items-center justify-center">
              <ShieldCheck size={24} />
            </div>
            <h3 className="font-display text-lg font-bold text-[#0A192F]">
              Percolation & Recovery
            </h3>
            <p className="text-slate-600 text-sm leading-relaxed">
              Evaluates betweenness centrality to quantify vulnerability before failures occur. Calculates
              optimal recovery sequences to restore maximum network efficiency under tight budget constraints.
            </p>
            <div className="pt-2 text-xs font-semibold text-slate-500">
              Graph Percolation • Greedy Optimization
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
